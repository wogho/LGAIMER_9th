#!/usr/bin/env python3
"""Independent source-level validator for REF4-110-CODEX-R1."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model/REF4-110-CODEX-R1"
TRAIN = ROOT / "data/train.csv"
SOURCE = ROOT / "model/REF4-OOF-DIAG-034A/diagnostic_rows.csv"
YEARS = [2022, 2023, 2024]
NAMES = ["110A", "110B", "110C"]
RECENT_WEIGHTS = {2022: 0.20, 2023: 0.30, 2024: 0.50}
REPS = 2000
SEED = 110_2026


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    rate = float(y.mean())
    brier = float(np.mean((y - prediction) ** 2))
    bss = float(1.0 - brier / (rate * (1.0 - rate)))
    return {"rows": int(len(y)), "target_rate": rate, "prediction_mean": float(prediction.mean()), "brier": brier, "bss": bss, "local_score": 100000.0 * bss}


def cluster_ci(frame: pd.DataFrame, base: np.ndarray, candidate: np.ndarray, seed: int) -> dict[str, float | int]:
    y = frame["target"].to_numpy(float)
    delta = (y - candidate) ** 2 - (y - base) ** 2
    grouped = pd.DataFrame({"pitcher": frame["pitcher_id"].astype(str), "delta": delta}).groupby("pitcher", sort=True)["delta"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    samples = np.empty(REPS, dtype=float)
    for start in range(0, REPS, 100):
        stop = min(start + 100, REPS)
        draws = rng.integers(0, len(grouped), size=(stop - start, len(grouped)))
        samples[start:stop] = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {"clusters": int(len(grouped)), "repetitions": REPS, "seed": seed, "delta_brier": float(delta.mean()), "ci_low": float(np.quantile(samples, .025)), "ci_high": float(np.quantile(samples, .975))}


def close(a: object, b: object, tolerance: float = 1e-13) -> bool:
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(close(a[key], b[key], tolerance) for key in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y, tolerance) for x, y in zip(a, b))
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return math.isclose(float(a), float(b), rel_tol=0.0, abs_tol=tolerance)
    return a == b


def parse_markdown(path: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| 110"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 10:
            continue
        gate_values = {}
        for token in cells[9].split(";"):
            key, value = token.split("=", 1)
            gate_values[key] = value == "true"
        rows[cells[0]] = {
            "candidate_status": cells[1],
            "2022_delta": float(cells[2]), "2023_delta": float(cells[3]), "2024_delta": float(cells[4]),
            "2024_bss": float(cells[5]), "2024_local": float(cells[6]),
            "weighted": float(cells[7]), "worst": float(cells[8]), "gates": gate_values,
        }
    return rows


def main() -> None:
    checks: list[dict[str, object]] = []
    failures: list[str] = []

    def check(name: str, passed: bool, actual: object) -> None:
        checks.append({"name": name, "checked": True, "passed": bool(passed), "actual": actual})
        if not passed:
            failures.append(name)

    required = [OUT / name for name in ["audit_contract.json", "preflight_report.json", "oof_predictions.csv", "result.json", "result.md", "audit_manifest.json"]]
    for path in required:
        check(f"exists:{path.name}", path.is_file(), path.is_file())
    if failures:
        raise RuntimeError(f"missing required artifacts: {failures}")

    contract = json.loads((OUT / "audit_contract.json").read_text(encoding="utf-8"))
    preflight = json.loads((OUT / "preflight_report.json").read_text(encoding="utf-8"))
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    manifest = json.loads((OUT / "audit_manifest.json").read_text(encoding="utf-8"))
    check("preflight_verified", preflight.get("status") == "AUDIT_VERIFIED" and preflight.get("mismatch_count") == 0, {"status": preflight.get("status"), "mismatch_count": preflight.get("mismatch_count")})
    check("contract_candidate_count", contract.get("candidate_count") == 3 and len(contract.get("candidates", {})) == 3, {"count": contract.get("candidate_count"), "ids": sorted(contract.get("candidates", {}))})

    hash_mismatches = []
    size_mismatches = []
    for relative, record in manifest["artifacts"].items():
        path = ROOT / relative
        if not path.is_file() or sha256_path(path) != record["sha256"]:
            hash_mismatches.append(relative)
        if not path.is_file() or path.stat().st_size != record["size"]:
            size_mismatches.append(relative)
    check("manifest_hashes", not hash_mismatches, hash_mismatches)
    check("manifest_sizes", not size_mismatches, size_mismatches)
    check("manifest_artifact_count", manifest["artifact_count"] == len(manifest["artifacts"]), {"recorded": manifest["artifact_count"], "actual": len(manifest["artifacts"])})

    oof = pd.read_csv(OUT / "oof_predictions.csv", dtype={"row_id": str, "game_type": str, "pitcher_id": str})
    source = pd.read_csv(SOURCE, dtype={"row_id": str, "game_type": str, "pitcher_id": str})
    train = pd.read_csv(TRAIN, usecols=["row_id", "season", "control_success"], dtype={"row_id": str})
    train = train.loc[train["season"].isin(YEARS)].reset_index(drop=True)
    check("oof_rows", len(oof) == 746_504, len(oof))
    check("oof_row_id_unique", oof["row_id"].is_unique, int(oof["row_id"].nunique()))
    check("oof_columns", list(oof.columns) == ["row_id", "season", "game_type", "pitcher_id", "target", "base_prediction", "110A", "110B", "110C"], list(oof.columns))
    check("oof_source_row_order", np.array_equal(oof["row_id"].to_numpy(), source["row_id"].to_numpy()), None)
    check("oof_source_base", np.allclose(oof["base_prediction"], source["prediction_current"], rtol=0, atol=1e-15), float(np.max(np.abs(oof["base_prediction"] - source["prediction_current"]))))
    check("oof_source_target", np.array_equal(oof["target"].to_numpy(float), source["target"].to_numpy(float)), None)
    paired = oof[["row_id", "season", "target"]].merge(train, on="row_id", how="outer", validate="one_to_one", indicator=True, suffixes=("_oof", "_train"))
    pair_ok = len(paired) == len(oof) and paired["_merge"].eq("both").all() and paired["season_oof"].eq(paired["season_train"]).all() and np.array_equal(paired["target"].to_numpy(float), paired["control_success"].to_numpy(float))
    check("oof_train_pairing", pair_ok, {"rows": len(paired), "merge": paired["_merge"].value_counts().to_dict()})
    prediction_values = oof[["base_prediction", *NAMES]].to_numpy(float)
    check("prediction_finite", np.isfinite(prediction_values).all(), int(np.isfinite(prediction_values).sum()))
    check("prediction_range", bool(((prediction_values >= 0) & (prediction_values <= 1)).all()), {"min": float(prediction_values.min()), "max": float(prediction_values.max())})
    mask_2022 = oof["season"].eq(2022).to_numpy()
    check("warmup_2022_identity", all(np.array_equal(oof.loc[mask_2022, name].to_numpy(), oof.loc[mask_2022, "base_prediction"].to_numpy()) for name in NAMES), None)

    model_paths = sorted([*OUT.glob("fold_*/*/*.cbm"), *OUT.glob("fold_*/*/*.txt"), *OUT.glob("fold_*/*/*.json")])
    expected_counts = {
        "fold_2023/110B": 19, "fold_2023/110C": 19,
        "fold_2024/110B": 19, "fold_2024/110C": 19, "fold_2024/110A": 3,
    }
    actual_counts = {}
    for key in expected_counts:
        actual_counts[key] = len([p for p in model_paths if str(p.relative_to(OUT)).startswith(key + "/")])
    check("model_counts_by_fold_candidate", actual_counts == expected_counts, {"actual": actual_counts, "expected": expected_counts})
    check("model_count_total", len(model_paths) == result["model_count"] == manifest["model_count"] == 79, {"actual": len(model_paths), "result": result["model_count"], "manifest": manifest["model_count"]})

    expected_fold_provenance = {
        "2023": {"train_seasons": [2022], "train_rows": 247472, "valid_season": 2023, "valid_rows": 245525, "validation_labels_used_in_fit": False},
        "2024": {"train_seasons": [2022, 2023], "train_rows": 492997, "valid_season": 2024, "valid_rows": 253507, "validation_labels_used_in_fit": False},
    }
    check("temporal_fold_provenance", result["fold_provenance"] == expected_fold_provenance, result["fold_provenance"])
    eb_order_ok = all(int(value["history_max_season"]) < int(year) for year, value in result["eb_provenance"].items())
    check("eb_temporal_order", eb_order_ok and sorted(map(int, result["eb_provenance"])) == YEARS, result["eb_provenance"])

    base_metrics = {}
    recomputed = []
    for year in YEARS:
        mask = oof["season"].eq(year).to_numpy()
        base_metrics[str(year)] = metric(oof.loc[mask, "target"].to_numpy(float), oof.loc[mask, "base_prediction"].to_numpy(float))
    check("base_metrics", close(base_metrics, result["base_metrics"]), base_metrics)
    for index, name in enumerate(NAMES):
        stored = next(item for item in result["candidates"] if item["candidate_name"] == name)
        metrics = {}
        deltas = {}
        cis = {}
        for year in YEARS:
            mask = oof["season"].eq(year).to_numpy()
            y = oof.loc[mask, "target"].to_numpy(float)
            pred = oof.loc[mask, name].to_numpy(float)
            base = oof.loc[mask, "base_prediction"].to_numpy(float)
            metrics[str(year)] = metric(y, pred)
            deltas[str(year)] = float(metrics[str(year)]["brier"] - base_metrics[str(year)]["brier"])
            cis[str(year)] = cluster_ci(oof.loc[mask].reset_index(drop=True), base, pred, SEED + year + index * 10000)
        weighted = float(sum(RECENT_WEIGHTS[year] * deltas[str(year)] for year in YEARS))
        worst = float(max(deltas.values()))
        gates = {
            "delta_2024_at_most_minus_0_0001": deltas["2024"] <= -0.0001,
            "worst_season_delta_at_most_plus_0_00005": worst <= 0.00005,
            "time_weighted_delta_negative": weighted < 0,
            "bootstrap_2024_ci_high_below_zero": float(cis["2024"]["ci_high"]) < 0,
        }
        current = {"candidate_name": name, "metrics": metrics, "deltas": deltas, "weighted": weighted, "worst": worst, "cis": cis, "gates": gates}
        recomputed.append(current)
        check(f"candidate_metrics:{name}", close(metrics, stored["metrics"]), metrics)
        check(f"candidate_deltas:{name}", close(deltas, stored["delta_brier_vs_strict_base"]), deltas)
        check(f"candidate_weighted_worst:{name}", close(weighted, stored["time_weighted_delta"]) and close(worst, stored["worst_season_delta"]), {"weighted": weighted, "worst": worst})
        check(f"candidate_bootstrap:{name}", close(cis, stored["cluster_bootstrap"]), cis)
        check(f"candidate_gates:{name}", gates == stored["gate_results"] and stored["performance_gate_pass"] == all(gates.values()), gates)

    expected_winner = min(recomputed, key=lambda item: item["weighted"])["candidate_name"]
    check("winner_recomputed", result["provisional_winner"] == expected_winner, {"stored": result["provisional_winner"], "expected": expected_winner})
    check("candidate_leaf_counts", result["candidate_count"] == result["actual_leaf_count"] == manifest["candidate_count"] == manifest["leaf_count"] == len(NAMES), {"result": result["candidate_count"], "leaf": result["actual_leaf_count"], "manifest": manifest["candidate_count"]})
    check("gate_count", result["gate_checks_count"] == manifest["gate_checks_count"] == len(NAMES) * 4, {"result": result["gate_checks_count"], "manifest": manifest["gate_checks_count"]})
    check("no_test_full_zip", result["test_read"] is False and result["test_inference_performed"] is False and result["full_train_performed"] is False and result["zip_created"] is False, {key: result[key] for key in ["test_read", "test_inference_performed", "full_train_performed", "zip_created"]})
    check("anchor_transfer_not_claimed", result["anchor_transfer_to_109c_verified"] is False and result["official_score_estimated"] is False, {"anchor_transfer": result["anchor_transfer_to_109c_verified"], "score_estimated": result["official_score_estimated"]})

    md = parse_markdown(OUT / "result.md")
    md_failures = []
    for item in recomputed:
        name = item["candidate_name"]
        row = md.get(name)
        stored = next(value for value in result["candidates"] if value["candidate_name"] == name)
        if row is None:
            md_failures.append(f"missing:{name}")
            continue
        expected = {
            "candidate_status": stored["candidate_status"],
            "2022_delta": item["deltas"]["2022"], "2023_delta": item["deltas"]["2023"], "2024_delta": item["deltas"]["2024"],
            "2024_bss": item["metrics"]["2024"]["bss"], "2024_local": item["metrics"]["2024"]["local_score"],
            "weighted": item["weighted"], "worst": item["worst"], "gates": item["gates"],
        }
        if not close(row, expected, 5e-13):
            md_failures.append(f"mismatch:{name}")
    check("json_markdown_alignment", not md_failures and len(md) == len(NAMES), {"checked": len(md), "expected": len(NAMES), "failures": md_failures})

    status = "AUDIT_VERIFIED" if not failures else "AUDIT_FAIL"
    performance_pass = [item["candidate_name"] for item in result["candidates"] if item["performance_gate_pass"]]
    report = {
        "experiment_id": "REF4-110-CODEX-R1", "status": status,
        "performance_status": "PASS" if performance_pass else "FAIL",
        "submission_approval": "HOLD",
        "submission_hold_reasons": ["no_candidate_passed_all_performance_gates"] if not performance_pass else ["109C_anchor_transfer_not_verified"],
        "checked_count": len(checks), "passed_count": sum(bool(item["passed"]) for item in checks),
        "mismatch_count": len(failures), "failures": failures, "checks": checks,
        "actual_leaf_count": len(NAMES), "gate_checks_count": len(NAMES) * 4,
        "model_count": len(model_paths), "oof_rows": len(oof),
        "recomputed_candidates": recomputed, "winner": expected_winner,
        "performance_pass_candidates": performance_pass,
    }
    (OUT / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    attestation = {
        "experiment_id": "REF4-110-CODEX-R1", "status": status,
        "performance_status": report["performance_status"], "submission_approval": "HOLD",
        "audit_manifest_sha256": sha256_path(OUT / "audit_manifest.json"),
        "validation_report_sha256": sha256_path(OUT / "validation_report.json"),
        "validator_sha256": sha256_path(Path(__file__).resolve()),
        "leaf_count": len(NAMES), "gate_count": len(NAMES) * 4,
        "checked_count": len(checks), "passed_count": report["passed_count"],
        "fail_count": len(failures), "mismatch_count": len(failures),
        "model_count": len(model_paths), "oof_rows": len(oof), "winner": expected_winner,
        "performance_pass_candidates": performance_pass,
    }
    (OUT / "audit_attestation.json").write_text(json.dumps(attestation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(attestation, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
