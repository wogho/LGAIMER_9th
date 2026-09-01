#!/usr/bin/env python3
"""Independent integrity and metric audit for REF4-110-ORIGINAL-R2."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model/REF4-110-ORIGINAL-R2"
YEARS = (2022, 2023, 2024)
WEIGHTS = {2022: 0.2, 2023: 0.3, 2024: 0.5}
ATOL = 1e-12


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def metric(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    brier = float(np.mean((y - p) ** 2))
    prior = float(y.mean())
    bss = float(1.0 - brier / (prior * (1.0 - prior)))
    return {"rows": int(len(y)), "brier": brier, "bss": bss, "local_cv_proxy_score": 1e5 * bss}


def cluster_bootstrap(frame: pd.DataFrame, candidate: str, reps: int = 2000) -> dict[str, float | int]:
    part = frame.loc[frame["season"].eq(2024), ["pitcher_id", "target", "p109c", candidate]].copy()
    part["delta"] = (part[candidate] - part["target"]) ** 2 - (part["p109c"] - part["target"]) ** 2
    grouped = part.groupby("pitcher_id", sort=False)["delta"].agg(["sum", "count"])
    sums, counts = grouped["sum"].to_numpy(float), grouped["count"].to_numpy(float)
    rng = np.random.default_rng(110)
    draws = np.empty(reps, dtype=float)
    for pos in range(reps):
        selected = rng.integers(0, len(grouped), len(grouped))
        draws[pos] = sums[selected].sum() / counts[selected].sum()
    return {
        "reps": reps,
        "pitcher_clusters": int(len(grouped)),
        "mean_delta": float(part["delta"].mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def main() -> None:
    checks: list[dict] = []

    def check(name: str, passed: bool, actual=None) -> None:
        checks.append({"name": name, "checked": True, "passed": bool(passed), "actual": actual})

    contract = json.loads((OUT / "audit_contract.json").read_text())
    preflight = json.loads((OUT / "preflight_report.json").read_text())
    generated = json.loads((OUT / "result.json").read_text())
    predictions = pd.read_csv(OUT / "expert_oof.csv", dtype={"row_id": str, "pitcher_id": str})
    raw = pd.read_csv(ROOT / "data/train.csv", low_memory=False)
    expected = raw.loc[raw["season"].isin(YEARS), ["row_id", "season", "control_success"]].copy()
    expected["row_id"] = expected["row_id"].astype(str)

    check("preflight_verified", preflight.get("status") == "AUDIT_VERIFIED" and preflight.get("mismatch_count") == 0, preflight.get("status"))
    check("candidate_count", generated.get("candidate_count") == 3 and len(generated.get("candidates", [])) == 3, generated.get("candidate_count"))
    check("candidate_names", [item["candidate_name"] for item in generated["candidates"]] == ["110A", "110B", "110C"], [item["candidate_name"] for item in generated["candidates"]])
    check("oof_rows", len(predictions) == len(expected) == 746_504, len(predictions))
    check("oof_row_ids_unique", predictions["row_id"].nunique() == len(predictions), predictions["row_id"].nunique())
    joined = expected.merge(predictions, on="row_id", how="outer", indicator=True, suffixes=("_raw", "_oof"))
    check("row_pairing", joined["_merge"].eq("both").all(), joined["_merge"].value_counts().to_dict())
    check("season_pairing", joined["season_raw"].eq(joined["season_oof"]).all(), None)
    check("target_pairing", np.allclose(joined["control_success"], joined["target"], atol=0.0, rtol=0.0), float(np.max(np.abs(joined["control_success"] - joined["target"]))))
    prediction_columns = ["p103a", "p107", "p108c", "p109c", "110A", "110B", "110C"]
    values = predictions[prediction_columns].to_numpy(float)
    check("prediction_finite", np.isfinite(values).all(), None)
    check("prediction_range", bool(((values >= 0.0) & (values <= 1.0)).all()), [float(values.min()), float(values.max())])
    check("season_counts", predictions["season"].value_counts().sort_index().to_dict() == {2022: 247472, 2023: 245525, 2024: 253507}, predictions["season"].value_counts().sort_index().to_dict())

    p103 = pd.read_csv(OUT / "p103a_oof.csv", dtype={"row_id": str})
    bound = p103[["row_id", "p103a"]].merge(predictions[["row_id", "p103a"]], on="row_id", suffixes=("_build", "_candidate"), validate="one_to_one")
    p103_diff = float(np.max(np.abs(bound["p103a_build"] - bound["p103a_candidate"])))
    check("p103_oof_binding", len(bound) == 746_504 and p103_diff <= ATOL, p103_diff)
    for year in YEARS:
        provenance = json.loads((OUT / f"backbone_fold_{year}/provenance.json").read_text())
        train_seasons = [int(value) for value in provenance["train_seasons"]]
        check(f"fold_{year}_strict_years", bool(train_seasons) and max(train_seasons) < year, train_seasons)
        check(f"fold_{year}_labels_excluded", provenance.get("validation_labels_used_in_fit") is False, provenance.get("validation_labels_used_in_fit"))
        check(f"fold_{year}_valid_rows", provenance.get("valid_rows") == int((predictions["season"] == year).sum()), provenance.get("valid_rows"))

    actual_model_files = []
    for pattern in ("expert_fold_*/*", "nested_110b_*/*", "router_*/*"):
        actual_model_files.extend(path for path in OUT.glob(pattern) if path.is_file() and path.suffix in {".cbm", ".txt", ".json"})
    check("candidate_model_count", len(actual_model_files) == generated.get("model_count"), {"actual": len(actual_model_files), "reported": generated.get("model_count")})
    check("test_not_read", generated.get("test_read") is False, generated.get("test_read"))
    check("zip_not_created_before_audit", generated.get("zip_created") is False, generated.get("zip_created"))
    rollback = ROOT / contract["official_champion"]["zip"]
    check("rollback_hash", sha256(rollback) == contract["official_champion"]["sha256"], sha256(rollback))

    baseline: dict[str, dict] = {}
    for year in YEARS:
        mask = predictions["season"].eq(year)
        baseline[str(year)] = metric(predictions.loc[mask, "target"].to_numpy(float), predictions.loc[mask, "p109c"].to_numpy(float))
    audited_candidates = []
    generated_by_name = {item["candidate_name"]: item for item in generated["candidates"]}
    for name in ("110A", "110B", "110C"):
        metrics, deltas = {}, {}
        for year in YEARS:
            mask = predictions["season"].eq(year)
            metrics[str(year)] = metric(predictions.loc[mask, "target"].to_numpy(float), predictions.loc[mask, name].to_numpy(float))
            deltas[str(year)] = metrics[str(year)]["brier"] - baseline[str(year)]["brier"]
        weighted = sum(WEIGHTS[year] * deltas[str(year)] for year in YEARS)
        worst = max(deltas.values())
        bootstrap = cluster_bootstrap(predictions, name)
        gates = {
            "2024": deltas["2024"] <= -0.0001,
            "2022": deltas["2022"] <= 0.00005,
            "weighted": weighted < 0.0,
            "worst": worst <= 0.00005,
            "pitcher_bootstrap_ci_high_below_zero": bootstrap["ci_high"] < 0.0,
        }
        source = generated_by_name[name]
        numerical_match = (
            all(abs(metrics[str(year)]["brier"] - source["metrics"][str(year)]["brier"]) <= ATOL for year in YEARS)
            and all(abs(deltas[str(year)] - source["delta_brier_vs_p109c"][str(year)]) <= ATOL for year in YEARS)
            and abs(weighted - source["time_weighted_delta"]) <= ATOL
            and abs(bootstrap["ci_high"] - source["pitcher_cluster_bootstrap_2024"]["ci_high"]) <= ATOL
            and gates == source["gate_results"]
        )
        check(f"{name}_metric_recompute", numerical_match, {"weighted_delta": weighted, "bootstrap_ci_high": bootstrap["ci_high"]})
        passed = all(gates.values())
        audited_candidates.append({
            "candidate_name": name,
            "candidate_status": "PERFORMANCE_GATE_PASS" if passed else "REJECTED_PERFORMANCE_GATE",
            "metrics": metrics,
            "delta_brier_vs_p109c": deltas,
            "time_weighted_delta": weighted,
            "worst_season_delta": worst,
            "pitcher_cluster_bootstrap_2024": bootstrap,
            "gate_results": gates,
            "performance_gate_pass": passed,
        })

    failed = [item for item in checks if not item["passed"]]
    integrity_verified = not failed
    passing = [item for item in audited_candidates if item["performance_gate_pass"]]
    winner = min(passing, key=lambda item: item["time_weighted_delta"])["candidate_name"] if passing else None
    status = "AUDIT_VERIFIED" if integrity_verified else "MISMATCH"
    report = {
        "experiment_id": contract["experiment_id"],
        "status": status,
        "diagnostic_status": "COMPLETE" if integrity_verified else "INCOMPLETE",
        "checked_count": len(checks),
        "passed_count": len(checks) - len(failed),
        "mismatch_count": len(failed),
        "failures": failed,
        "candidate_count": len(audited_candidates),
        "actual_leaf_count": len(audited_candidates),
        "gate_checks_count": sum(len(item["gate_results"]) for item in audited_candidates),
        "baseline_p109c": baseline,
        "candidates": audited_candidates,
        "performance_passing_candidates": [item["candidate_name"] for item in passing],
        "provisional_winner": winner,
        "production_status": "ELIGIBLE_FOR_FULL_FIT_TECHNICAL_AUDIT" if winner else "NO_ZIP_RETAIN_109C",
        "technical_gates_pending_after_full_fit": ["row_independence", "runtime_seconds_max_600", "package_integrity"],
        "test_read": False,
        "zip_created": False,
        "checks": checks,
    }
    report_path = OUT / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    markdown = [
        f"# {contract['experiment_id']} audit result",
        "",
        f"- status: `{status}`",
        f"- candidate_count / actual_leaf_count: `3 / {len(audited_candidates)}`",
        f"- gate_checks_count: `{report['gate_checks_count']}`",
        f"- integrity checks: `{report['passed_count']}/{report['checked_count']}`",
        f"- provisional winner: `{winner or 'NONE'}`",
        f"- production status: `{report['production_status']}`",
        "",
        "| candidate | 2022 delta | 2023 delta | 2024 delta | weighted delta | bootstrap CI high | result |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for item in audited_candidates:
        delta = item["delta_brier_vs_p109c"]
        markdown.append(
            f"| {item['candidate_name']} | {delta['2022']:+.9f} | {delta['2023']:+.9f} | "
            f"{delta['2024']:+.9f} | {item['time_weighted_delta']:+.9f} | "
            f"{item['pitcher_cluster_bootstrap_2024']['ci_high']:+.9f} | {item['candidate_status']} |"
        )
    markdown.extend([
        "",
        "110B's 2022 rows are a frozen 109C warm-up control; its first strictly trainable residual fold is 2023, trained only from 2022 OOF anchors.",
        "No test data, full production fit, or candidate ZIP was used in this comparison.",
        "",
    ])
    result_md = OUT / "result.md"
    result_md.write_text("\n".join(markdown), encoding="utf-8")
    manifest = {
        "experiment_id": contract["experiment_id"],
        "status": status,
        "candidate_count": 3,
        "actual_leaf_count": len(audited_candidates),
        "gate_checks_count": report["gate_checks_count"],
        "model_count": len(actual_model_files),
        "oof_rows": len(predictions),
        "artifacts": {
            "audit_contract.json": sha256(OUT / "audit_contract.json"),
            "preflight_report.json": sha256(OUT / "preflight_report.json"),
            "p103a_oof.csv": sha256(OUT / "p103a_oof.csv"),
            "expert_oof.csv": sha256(OUT / "expert_oof.csv"),
            "result.json": sha256(OUT / "result.json"),
            "validation_report.json": sha256(report_path),
            "result.md": sha256(result_md),
            "build_ref4_103a_strict_oof_r2.py": sha256(ROOT / "scripts/build_ref4_103a_strict_oof_r2.py"),
            "run_ref4_110_original_r2.py": sha256(ROOT / "scripts/run_ref4_110_original_r2.py"),
            "verify_ref4_110_original_r2.py": sha256(Path(__file__)),
        },
    }
    manifest_path = OUT / "audit_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    attestation = {
        "experiment_id": contract["experiment_id"],
        "status": status,
        "manifest_sha256": sha256(manifest_path),
        "validation_report_sha256": sha256(report_path),
        "validator_sha256": sha256(Path(__file__)),
        "checked_count": report["checked_count"],
        "passed_count": report["passed_count"],
        "mismatch_count": report["mismatch_count"],
        "actual_leaf_count": report["actual_leaf_count"],
        "gate_checks_count": report["gate_checks_count"],
        "model_count": len(actual_model_files),
        "oof_rows": len(predictions),
    }
    (OUT / "audit_attestation.json").write_text(json.dumps(attestation, indent=2) + "\n")
    print(json.dumps(attestation, indent=2))
    if not integrity_verified:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
