#!/usr/bin/env python3
"""Independent validator for the fixed F-Regime 0.75 OOF ablation."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "REF4-F-REGIME-032A"
OUT = ROOT / "model" / EXPERIMENT_ID
BASE = ROOT / "model" / "REF4-EXACT-OOF-031A"
SOURCE = ROOT / "model" / "REF4-CHAMPION-STACK-030"
REPS = 2000
SEED = 320200


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""): digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def recombine(oof: pd.DataFrame, regime: dict[str, float], manifest: dict[str, object]) -> np.ndarray:
    is_f = oof["game_type"].eq("F").to_numpy()
    globals_ = [oof["p_v2_global"].to_numpy(float), oof["p_v3_55_global"].to_numpy(float), oof["p_v3_30_global"].to_numpy(float)]
    f0 = oof["p_v2_f"].to_numpy(float); f1 = oof["p_v3_55_f"].to_numpy(float)
    inner = globals_[2] + regime["v330_recent_inner_scale"] * (oof["p_v3_30_f_recent"].to_numpy(float) - globals_[2])
    f2 = regime["v330_all_weight"] * oof["p_v3_30_f_all"].to_numpy(float) + (1 - regime["v330_all_weight"]) * inner
    channels = [
        np.where(is_f, globals_[0] + regime["v2_scale"] * (f0 - globals_[0]), globals_[0]),
        np.where(is_f, globals_[1] + regime["v355_scale"] * (f1 - globals_[1]), globals_[1]),
        np.where(is_f, globals_[2] + regime["v330_scale"] * (f2 - globals_[2]), globals_[2]),
    ]
    risks = []
    for name in ("middle", "wild", "reverse"):
        global_risk = oof[f"risk_{name}_global"].to_numpy(float); f_risk = oof[f"risk_{name}_f"].to_numpy(float)
        risks.append(np.where(is_f, global_risk + regime["subtype_scale"] * (f_risk - global_risk), global_risk))
    main = np.average(np.vstack(channels), axis=0, weights=np.asarray(manifest["main_weights"], float))
    stacked = float(manifest["stack_intercept"]) + np.column_stack([main, *risks]) @ np.asarray(manifest["stack_coefficients"], float)
    return np.clip(stacked + float(manifest["global_shift"]), 1e-5, 1 - 1e-5)


def metric(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    rate = float(y.mean()); brier = float(np.mean((y - p) ** 2)); bss = float(1 - brier / (rate * (1 - rate)))
    return {"rows": int(len(y)), "target_rate": rate, "brier": brier, "bss": bss, "local_score": 100000 * bss}


def cluster_ci(y: np.ndarray, base: np.ndarray, candidate: np.ndarray, pitcher: np.ndarray, seed: int) -> dict[str, float | int]:
    gain = (y - base) ** 2 - (y - candidate) ** 2
    groups = pd.DataFrame({"pitcher": pitcher.astype(str), "gain": gain}).groupby("pitcher", sort=True)["gain"].agg(["sum", "count"])
    sums = groups["sum"].to_numpy(float); counts = groups["count"].to_numpy(float)
    draws = np.random.default_rng(seed).integers(0, len(groups), size=(REPS, len(groups)))
    values = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {"clusters": int(len(groups)), "repetitions": REPS, "seed": seed, "brier_gain": float(gain.mean()), "ci_low": float(np.quantile(values, .025)), "ci_high": float(np.quantile(values, .975))}


def compare(left: object, right: object, path: str = "") -> tuple[float, list[str]]:
    maximum = 0.0; failures: list[str] = []
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right): return 0.0, [f"{path}:keys"]
        for key in left:
            diff, child = compare(left[key], right[key], f"{path}.{key}" if path else str(key)); maximum = max(maximum, diff); failures.extend(child)
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right): failures.append(f"{path}:length")
        else:
            for index, pair in enumerate(zip(left, right)):
                diff, child = compare(pair[0], pair[1], f"{path}[{index}]"); maximum = max(maximum, diff); failures.extend(child)
    elif isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)) and not isinstance(right, bool):
        diff = abs(float(left) - float(right)); maximum = max(maximum, diff)
        if diff > 1e-10: failures.append(f"{path}:{diff}")
    elif left != right: failures.append(f"{path}:{left!r}!={right!r}")
    return maximum, failures


def main() -> None:
    checks: list[dict[str, object]] = []; mismatches: list[str] = []
    def check(name: str, passed: bool, actual: object) -> None:
        checks.append({"name": name, "checked": True, "passed": bool(passed), "actual": actual})
        if not passed: mismatches.append(name)

    required = [OUT / "audit_manifest.json", OUT / "result.json", OUT / "result.md", OUT / "oof_predictions.csv"]
    for path in required: check(f"exists:{path.name}", path.is_file(), path.is_file())
    audit = json.loads((OUT / "audit_manifest.json").read_text(encoding="utf-8")); result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    failures = []
    for relative, record in audit["artifacts"].items():
        path = ROOT / relative
        if not path.is_file(): failures.append(f"missing:{relative}"); continue
        if sha256_path(path) != record["sha256"]: failures.append(f"sha256:{relative}")
        if path.stat().st_size != record["size"]: failures.append(f"size:{relative}")
    check("manifest_artifact_hashes", not failures, failures)
    check("manifest_artifact_count", len(audit["artifacts"]) == audit["artifact_count"], {"actual": len(audit["artifacts"]), "recorded": audit["artifact_count"]})
    check("no_models", audit["model_count"] == result["model_count"] == 0 and not list(OUT.glob("**/*.cbm")), {"audit": audit["model_count"], "result": result["model_count"], "files": len(list(OUT.glob("**/*.cbm")))})

    base_audit = json.loads((BASE / "audit_manifest.json").read_text(encoding="utf-8")); base_oof_path = BASE / "oof_predictions.csv"; base_key = str(base_oof_path.relative_to(ROOT))
    check("frozen_base_hash", sha256_path(base_oof_path) == base_audit["artifacts"][base_key]["sha256"], {"actual": sha256_path(base_oof_path), "recorded": base_audit["artifacts"][base_key]["sha256"]})
    oof = pd.read_csv(base_oof_path, dtype={"row_id": str, "game_type": str, "pitcher_id": str})
    output = pd.read_csv(OUT / "oof_predictions.csv", dtype={"row_id": str, "game_type": str, "pitcher_id": str})
    raw = pd.read_csv(ROOT / "data" / "train.csv", usecols=["row_id", "season", "game_type", "pitcher_id", "control_success"], low_memory=False)
    expected = raw.loc[raw["season"].isin([2023, 2024])].reset_index(drop=True)
    check("row_count", len(output) == len(oof) == len(expected) == audit["oof_rows"] == result["oof_rows"], {"output": len(output), "base": len(oof), "expected": len(expected)})
    for name, actual, wanted in (
        ("row_order", output["row_id"].astype(str).to_numpy(), expected["row_id"].astype(str).to_numpy()),
        ("season", output["season"].to_numpy(int), expected["season"].to_numpy(int)),
        ("game_type", output["game_type"].astype(str).to_numpy(), expected["game_type"].astype(str).to_numpy()),
        ("pitcher", output["pitcher_id"].astype(str).to_numpy(), expected["pitcher_id"].astype(str).to_numpy()),
        ("target", output["target"].to_numpy(float), expected["control_success"].to_numpy(float)),
    ): check(name, np.array_equal(actual, wanted), int(np.sum(actual != wanted)))

    current = {key: float(value) for key, value in result["base_regime"].items()}; candidate_regime = {key: float(value) for key, value in result["candidate"]["regime"].items()}
    expected_current = {"v2_scale": 2.0, "v355_scale": 0.5, "v330_scale": 0.5, "v330_all_weight": 0.25, "v330_recent_inner_scale": 0.25, "subtype_scale": 0.75, "transition_scale": 0.0}
    expected_candidate = {"v2_scale": 1.5, "v355_scale": 0.375, "v330_scale": 0.375, "v330_all_weight": 0.25, "v330_recent_inner_scale": 0.25, "subtype_scale": 0.5625, "transition_scale": 0.0}
    check("regime_contract", current == expected_current and candidate_regime == expected_candidate, {"current": current, "candidate": candidate_regime})
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    base_prediction = recombine(oof, current, manifest); candidate_prediction = recombine(oof, candidate_regime, manifest)
    formula_diffs = {"base_to_frozen": float(np.max(np.abs(base_prediction - oof["prediction"].to_numpy(float)))), "base_to_output": float(np.max(np.abs(base_prediction - output["base_prediction"].to_numpy(float)))), "candidate_to_output": float(np.max(np.abs(candidate_prediction - output["f_regime_075_prediction"].to_numpy(float))))}
    check("formula_recomputed", max(formula_diffs.values()) <= 1e-12, formula_diffs)
    check("finite_range", np.isfinite(candidate_prediction).all() and ((candidate_prediction >= 0) & (candidate_prediction <= 1)).all(), {"min": float(candidate_prediction.min()), "max": float(candidate_prediction.max())})

    base_metrics: dict[str, object] = {}; candidate_metrics: dict[str, object] = {}; cis: dict[str, object] = {}
    for year in (2023, 2024):
        mask = oof["season"].eq(year).to_numpy(); y = oof.loc[mask, "target"].to_numpy(float)
        base_metrics[str(year)] = metric(y, base_prediction[mask]); candidate_metrics[str(year)] = metric(y, candidate_prediction[mask])
        cis[str(year)] = cluster_ci(y, base_prediction[mask], candidate_prediction[mask], oof.loc[mask, "pitcher_id"].astype(str).to_numpy(), SEED + year)
    y_all = oof["target"].to_numpy(float); base_metrics["pooled"] = metric(y_all, base_prediction); candidate_metrics["pooled"] = metric(y_all, candidate_prediction)
    base_metrics["worst_season_bss"] = min(float(base_metrics[str(y)]["bss"]) for y in (2023, 2024))  # type: ignore[index]
    candidate_metrics["worst_season_bss"] = min(float(candidate_metrics[str(y)]["bss"]) for y in (2023, 2024))  # type: ignore[index]
    season_gain = {str(year): float(base_metrics[str(year)]["brier"] - candidate_metrics[str(year)]["brier"]) for year in (2023, 2024)}  # type: ignore[index]
    pooled_gain = float(base_metrics["pooled"]["brier"] - candidate_metrics["pooled"]["brier"])  # type: ignore[index]
    worst_gain = float(candidate_metrics["worst_season_bss"] - base_metrics["worst_season_bss"])
    gates = {"2023_brier_gain_positive": season_gain["2023"] > 0, "2024_brier_gain_positive": season_gain["2024"] > 0, "pooled_brier_gain_positive": pooled_gain > 0, "worst_season_bss_gain_positive": worst_gain > 0, "2023_cluster_ci_low_positive": float(cis["2023"]["ci_low"]) > 0, "2024_cluster_ci_low_positive": float(cis["2024"]["ci_low"]) > 0}  # type: ignore[index]
    expected_candidate_result = {"name": "f_regime_075", "regime": candidate_regime, "metrics": candidate_metrics, "season_brier_gain": season_gain, "pooled_brier_gain": pooled_gain, "worst_season_bss_gain": worst_gain, "cluster_ci": cis, "mean_abs_change": float(np.mean(np.abs(candidate_prediction - base_prediction))), "mean_change": float(np.mean(candidate_prediction - base_prediction)), "gate_checks": gates, "promotion_pass": all(gates.values())}
    diff, metric_failures = compare({"base_metrics": result["base_metrics"], "candidate": result["candidate"]}, {"base_metrics": base_metrics, "candidate": expected_candidate_result})
    check("metrics_ci_gate_recomputed", not metric_failures, {"max_abs_diff": diff, "mismatches": metric_failures})
    check("leaf_and_gate_counts", result["candidate_count"] == result["actual_leaf_count"] == audit["candidate_count"] == audit["leaf_count"] == 1 and result["gate_checks_count"] == audit["gate_checks_count"] == 6, {"candidate": result["candidate_count"], "leaf": result["actual_leaf_count"], "gates": result["gate_checks_count"]})
    check("no_test_fulltrain_zip", result["test_inference_performed"] is False and result["full_train_performed"] is False and result["zip_created"] is False, {"test": result["test_inference_performed"], "fulltrain": result["full_train_performed"], "zip": result["zip_created"]})
    text = (OUT / "result.md").read_text(encoding="utf-8"); match = re.search(r"<!-- RESULT_JSON_BEGIN\n(.*?)\nRESULT_JSON_END -->", text, re.DOTALL); embedded = json.loads(match.group(1)) if match else None
    expected_embedded = {"candidate_status": result["candidate_status"], "base_metrics": result["base_metrics"], "candidate": result["candidate"], "candidate_count": 1, "model_count": 0, "oof_rows": result["oof_rows"]}
    check("json_markdown_embedded", embedded == expected_embedded, embedded == expected_embedded)

    status = "AUDIT_VERIFIED" if not mismatches else "FAIL"; promotion = "PASS" if status == "AUDIT_VERIFIED" and all(gates.values()) else "FAIL/HOLD"
    report = {"experiment_id": EXPERIMENT_ID, "status": status, "promotion_status": promotion, "checked_count": len(checks), "passed_count": sum(bool(x["passed"]) for x in checks), "mismatch_count": len(mismatches), "mismatches": mismatches, "actual_leaf_count": 1, "gate_checks_count": 6, "model_count": 0, "oof_rows": len(output), "base_metrics": base_metrics, "candidate": expected_candidate_result, "formula_max_abs_diff": max(formula_diffs.values()), "checks": checks}
    report_path = OUT / "validation_report.json"; write_json(report_path, report)
    attestation = {"experiment_id": EXPERIMENT_ID, "status": status, "promotion_status": promotion, "manifest_sha256": sha256_path(OUT / "audit_manifest.json"), "validation_report_sha256": sha256_path(report_path), "validator_sha256": sha256_path(Path(__file__).resolve()), "checked_count": report["checked_count"], "passed_count": report["passed_count"], "mismatch_count": report["mismatch_count"], "actual_leaf_count": 1, "gate_checks_count": 6, "model_count": 0, "oof_rows": len(output)}
    write_json(OUT / "audit_attestation.json", attestation); print(json.dumps(attestation, indent=2))
    if mismatches: raise SystemExit(1)


if __name__ == "__main__": main()
