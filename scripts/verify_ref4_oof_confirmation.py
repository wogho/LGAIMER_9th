#!/usr/bin/env python3
"""Independent source-level validator for fixed REF4 OOF confirmations."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "model" / "REF4-EXACT-OOF-031A"
OOF22 = ROOT / "model" / "REF4-ADAPTIVE-GATE-031B"
SOURCE = ROOT / "model" / "REF4-CHAMPION-STACK-030"
REPS = 2000
SEED = 340200
CONFIG = {
    "shift": {"id": "REF4-SHIFT-034B", "name": "shift_000", "gates": 8},
    "f-regime": {"id": "REF4-F-REGIME-032B", "name": "f_regime_075", "gates": 14},
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def metric(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    rate = float(y.mean())
    brier = float(np.mean((y - prediction) ** 2))
    bss = float(1.0 - brier / (rate * (1.0 - rate)))
    return {"rows": int(len(y)), "target_rate": rate, "prediction_mean": float(prediction.mean()), "brier": brier, "bss": bss, "local_score": 100000.0 * bss}


def cluster_ci(y: np.ndarray, base: np.ndarray, candidate: np.ndarray, pitcher: np.ndarray, seed: int) -> dict[str, float | int]:
    gain = (y - base) ** 2 - (y - candidate) ** 2
    groups = pd.DataFrame({"pitcher": pitcher.astype(str), "gain": gain}).groupby("pitcher", sort=True)["gain"].agg(["sum", "count"])
    sums = groups["sum"].to_numpy(float)
    counts = groups["count"].to_numpy(float)
    draws = np.random.default_rng(seed).integers(0, len(groups), size=(REPS, len(groups)))
    samples = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {"clusters": int(len(groups)), "repetitions": REPS, "seed": seed, "brier_gain": float(gain.mean()), "ci_low": float(np.quantile(samples, .025)), "ci_high": float(np.quantile(samples, .975))}


def recombine(oof: pd.DataFrame, regime: dict[str, float], manifest: dict[str, object]) -> tuple[np.ndarray, np.ndarray]:
    is_f = oof["game_type"].eq("F").to_numpy()
    global_channels = [oof["p_v2_global"].to_numpy(float), oof["p_v3_55_global"].to_numpy(float), oof["p_v3_30_global"].to_numpy(float)]
    inner = global_channels[2] + regime["v330_recent_inner_scale"] * (oof["p_v3_30_f_recent"].to_numpy(float) - global_channels[2])
    f30 = regime["v330_all_weight"] * oof["p_v3_30_f_all"].to_numpy(float) + (1.0 - regime["v330_all_weight"]) * inner
    channels = [
        np.where(is_f, global_channels[0] + regime["v2_scale"] * (oof["p_v2_f"].to_numpy(float) - global_channels[0]), global_channels[0]),
        np.where(is_f, global_channels[1] + regime["v355_scale"] * (oof["p_v3_55_f"].to_numpy(float) - global_channels[1]), global_channels[1]),
        np.where(is_f, global_channels[2] + regime["v330_scale"] * (f30 - global_channels[2]), global_channels[2]),
    ]
    risks = []
    for name in ("middle", "wild", "reverse"):
        global_risk = oof[f"risk_{name}_global"].to_numpy(float)
        f_risk = oof[f"risk_{name}_f"].to_numpy(float)
        risks.append(np.where(is_f, global_risk + regime["subtype_scale"] * (f_risk - global_risk), global_risk))
    main = np.average(np.vstack(channels), axis=0, weights=np.asarray(manifest["main_weights"], float))
    no_shift = float(manifest["stack_intercept"]) + np.column_stack([main, *risks]) @ np.asarray(manifest["stack_coefficients"], float)
    current = np.clip(no_shift + float(manifest["global_shift"]), 1e-5, 1 - 1e-5)
    return current, np.clip(no_shift, 1e-5, 1 - 1e-5)


def compare(left: object, right: object, path: str = "") -> tuple[float, list[str]]:
    maximum = 0.0
    failures: list[str] = []
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return 0.0, [f"{path}:keys"]
        for key in left:
            diff, child = compare(left[key], right[key], f"{path}.{key}" if path else str(key))
            maximum = max(maximum, diff)
            failures.extend(child)
    elif isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            failures.append(f"{path}:length")
        else:
            for index, (a, b) in enumerate(zip(left, right)):
                diff, child = compare(a, b, f"{path}[{index}]")
                maximum = max(maximum, diff)
                failures.extend(child)
    elif isinstance(left, (int, float)) and not isinstance(left, bool) and isinstance(right, (int, float)) and not isinstance(right, bool):
        diff = abs(float(left) - float(right))
        maximum = max(maximum, diff)
        if diff > 1e-10:
            failures.append(f"{path}:{diff}")
    elif left != right:
        failures.append(f"{path}:{left!r}!={right!r}")
    return maximum, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=CONFIG)
    args = parser.parse_args()
    config = CONFIG[args.experiment]
    experiment_id = str(config["id"])
    out = ROOT / "model" / experiment_id
    checks: list[dict[str, object]] = []
    mismatches: list[str] = []

    def check(name: str, passed: bool, actual: object) -> None:
        checks.append({"name": name, "checked": True, "passed": bool(passed), "actual": actual})
        if not passed:
            mismatches.append(name)

    required = [out / "audit_manifest.json", out / "result.json", out / "result.md", out / "oof_predictions.csv"]
    for path in required:
        check(f"exists:{path.name}", path.is_file(), path.is_file())
    audit = json.loads((out / "audit_manifest.json").read_text(encoding="utf-8"))
    result = json.loads((out / "result.json").read_text(encoding="utf-8"))
    artifact_failures: list[str] = []
    for relative, record in audit["artifacts"].items():
        path = ROOT / relative
        if not path.is_file():
            artifact_failures.append(f"missing:{relative}")
            continue
        if sha256_path(path) != record["sha256"]:
            artifact_failures.append(f"sha256:{relative}")
        if path.stat().st_size != record["size"]:
            artifact_failures.append(f"size:{relative}")
    check("manifest_artifact_hashes", not artifact_failures, artifact_failures)
    check("manifest_artifact_count", len(audit["artifacts"]) == audit["artifact_count"], {"actual": len(audit["artifacts"]), "recorded": audit["artifact_count"]})
    check("no_models", audit["model_count"] == result["model_count"] == 0 and not list(out.glob("**/*.cbm")), {"audit": audit["model_count"], "result": result["model_count"]})

    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    current_regime = {key: float(value) for key, value in json.loads((SOURCE / "f_regime_meta.json").read_text(encoding="utf-8")).items()}
    candidate_regime = dict(current_regime)
    if args.experiment == "f-regime":
        for key in ("v2_scale", "v355_scale", "v330_scale", "subtype_scale"):
            candidate_regime[key] *= .75
    source_paths = {2022: OOF22 / "oof_2022.csv", 2023: BASE / "oof_2023.csv", 2024: BASE / "oof_2024.csv"}
    parts: list[pd.DataFrame] = []
    formula_diffs: dict[str, float] = {}
    for year in (2022, 2023, 2024):
        oof = pd.read_csv(source_paths[year], dtype={"row_id": str, "game_type": str, "pitcher_id": str})
        base, no_shift = recombine(oof, current_regime, manifest)
        if args.experiment == "shift":
            candidate = no_shift
        else:
            candidate, _ = recombine(oof, candidate_regime, manifest)
        formula_diffs[str(year)] = float(np.max(np.abs(base - oof["prediction"].to_numpy(float))))
        parts.append(pd.DataFrame({"row_id": oof["row_id"].astype(str), "season": year, "game_type": oof["game_type"].astype(str), "pitcher_id": oof["pitcher_id"].astype(str), "target": oof["target"].to_numpy(float), "base_prediction": base, "candidate_prediction": candidate}))
    expected = pd.concat(parts, ignore_index=True)
    stored = pd.read_csv(out / "oof_predictions.csv", dtype={"row_id": str, "game_type": str, "pitcher_id": str})
    check("source_formula_current", max(formula_diffs.values()) <= 1e-12, formula_diffs)
    check("row_count", len(stored) == len(expected) == audit["oof_rows"] == result["oof_rows"], {"stored": len(stored), "expected": len(expected)})
    frame_failures: list[str] = []
    for column in expected.columns:
        if pd.api.types.is_numeric_dtype(expected[column]):
            diff = float(np.max(np.abs(stored[column].to_numpy(float) - expected[column].to_numpy(float))))
            if diff > 1e-12:
                frame_failures.append(f"{column}:{diff}")
        elif not np.array_equal(stored[column].astype(str).to_numpy(), expected[column].astype(str).to_numpy()):
            frame_failures.append(f"{column}:values")
    check("oof_rows_recomputed", not frame_failures, frame_failures)
    check("finite_range", np.isfinite(expected["candidate_prediction"]).all() and expected["candidate_prediction"].between(0, 1).all(), {"min": float(expected["candidate_prediction"].min()), "max": float(expected["candidate_prediction"].max())})

    years = (2022, 2023, 2024)
    base_metrics: dict[str, object] = {}
    candidate_metrics: dict[str, object] = {}
    cis: dict[str, object] = {}
    for year in years:
        part = expected.loc[expected["season"].eq(year)]
        y = part["target"].to_numpy(float)
        base = part["base_prediction"].to_numpy(float)
        candidate = part["candidate_prediction"].to_numpy(float)
        base_metrics[str(year)] = metric(y, base)
        candidate_metrics[str(year)] = metric(y, candidate)
        cis[str(year)] = cluster_ci(y, base, candidate, part["pitcher_id"].astype(str).to_numpy(), SEED + year)
    y_all = expected["target"].to_numpy(float)
    base_all = expected["base_prediction"].to_numpy(float)
    candidate_all = expected["candidate_prediction"].to_numpy(float)
    base_metrics["pooled"] = metric(y_all, base_all)
    candidate_metrics["pooled"] = metric(y_all, candidate_all)
    base_metrics["worst_season_bss"] = min(float(base_metrics[str(year)]["bss"]) for year in years)  # type: ignore[index]
    candidate_metrics["worst_season_bss"] = min(float(candidate_metrics[str(year)]["bss"]) for year in years)  # type: ignore[index]
    season_gain = {str(year): float(base_metrics[str(year)]["brier"] - candidate_metrics[str(year)]["brier"]) for year in years}  # type: ignore[index]
    pooled_gain = float(base_metrics["pooled"]["brier"] - candidate_metrics["pooled"]["brier"])  # type: ignore[index]
    worst_gain = float(candidate_metrics["worst_season_bss"] - base_metrics["worst_season_bss"])
    gates = {f"{year}_brier_gain_positive": season_gain[str(year)] > 0 for year in years}
    gates["pooled_brier_gain_positive"] = pooled_gain > 0
    gates["worst_season_bss_gain_positive"] = worst_gain > 0
    gates.update({f"{year}_cluster_ci_low_positive": float(cis[str(year)]["ci_low"]) > 0 for year in years})  # type: ignore[index]
    f_slice: dict[str, object] | None = None
    if args.experiment == "f-regime":
        f_slice = {}
        for year in years:
            part = expected.loc[expected["season"].eq(year) & expected["game_type"].eq("F")]
            y = part["target"].to_numpy(float)
            base = part["base_prediction"].to_numpy(float)
            candidate = part["candidate_prediction"].to_numpy(float)
            base_metric = metric(y, base)
            candidate_metric = metric(y, candidate)
            ci = cluster_ci(y, base, candidate, part["pitcher_id"].astype(str).to_numpy(), SEED + 10000 + year)
            gain = float(base_metric["brier"] - candidate_metric["brier"])
            f_slice[str(year)] = {"base": base_metric, "candidate": candidate_metric, "brier_gain": gain, "cluster_ci": ci}
            gates[f"f_slice_{year}_brier_gain_positive"] = gain > 0
            gates[f"f_slice_{year}_cluster_ci_low_positive"] = float(ci["ci_low"]) > 0
    expected_candidate = {"name": config["name"], "metrics": candidate_metrics, "season_brier_gain": season_gain, "pooled_brier_gain": pooled_gain, "worst_season_bss_gain": worst_gain, "cluster_ci": cis, "f_slice": f_slice, "mean_abs_change": float(np.mean(np.abs(candidate_all - base_all))), "mean_change": float(np.mean(candidate_all - base_all)), "gate_checks": gates, "promotion_pass": all(gates.values())}
    diff, result_failures = compare({"base_metrics": result["base_metrics"], "candidate": result["candidate"]}, {"base_metrics": base_metrics, "candidate": expected_candidate})
    check("metrics_ci_gate_recomputed", not result_failures, {"max_abs_diff": diff, "failures": result_failures})
    check("leaf_gate_counts", result["candidate_count"] == result["actual_leaf_count"] == audit["candidate_count"] == audit["leaf_count"] == 1 and result["gate_checks_count"] == audit["gate_checks_count"] == len(gates) == int(config["gates"]), {"leaf": result["actual_leaf_count"], "gates": result["gate_checks_count"]})
    check("no_test_fulltrain_zip", result["test_inference_performed"] is False and result["full_train_performed"] is False and result["zip_created"] is False, {"test": result["test_inference_performed"], "fulltrain": result["full_train_performed"], "zip": result["zip_created"]})
    markdown = (out / "result.md").read_text(encoding="utf-8")
    match = re.search(r"<!-- RESULT_JSON_BEGIN\n(.*?)\nRESULT_JSON_END -->", markdown, re.DOTALL)
    embedded = json.loads(match.group(1)) if match else None
    expected_embedded = {"candidate_status": result["candidate_status"], "base_metrics": result["base_metrics"], "candidate": result["candidate"], "candidate_count": 1, "model_count": 0, "oof_rows": result["oof_rows"]}
    check("json_markdown_embedded", embedded == expected_embedded, embedded == expected_embedded)

    status = "AUDIT_VERIFIED" if not mismatches else "FAIL"
    promotion = "PASS" if status == "AUDIT_VERIFIED" and all(gates.values()) else "FAIL/HOLD"
    report = {"experiment_id": experiment_id, "status": status, "promotion_status": promotion, "checked_count": len(checks), "passed_count": sum(bool(x["passed"]) for x in checks), "mismatch_count": len(mismatches), "mismatches": mismatches, "actual_leaf_count": 1, "gate_checks_count": len(gates), "passed_gate_count": sum(gates.values()), "model_count": 0, "oof_rows": len(expected), "base_metrics": base_metrics, "candidate": expected_candidate, "formula_max_abs_diff": max(formula_diffs.values()), "checks": checks}
    report_path = out / "validation_report.json"
    write_json(report_path, report)
    attestation = {"experiment_id": experiment_id, "status": status, "promotion_status": promotion, "manifest_sha256": sha256_path(out / "audit_manifest.json"), "validation_report_sha256": sha256_path(report_path), "validator_sha256": sha256_path(Path(__file__).resolve()), "checked_count": report["checked_count"], "passed_count": report["passed_count"], "mismatch_count": report["mismatch_count"], "actual_leaf_count": 1, "gate_checks_count": len(gates), "passed_gate_count": sum(gates.values()), "model_count": 0, "oof_rows": len(expected)}
    write_json(out / "audit_attestation.json", attestation)
    print(json.dumps(attestation, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
