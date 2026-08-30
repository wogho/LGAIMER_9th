#!/usr/bin/env python3
"""Independent refit validator for REF4-F-ERA-CAL-036A."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "REF4-F-ERA-CAL-036A"
OUT = ROOT / "model" / EXPERIMENT_ID
BASE = ROOT / "model" / "REF4-EXACT-OOF-031A"
OOF22 = ROOT / "model" / "REF4-ADAPTIVE-GATE-031B"
TRAIN = ROOT / "data" / "train.csv"
YEARS = (2022, 2023, 2024)


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
    return {"rows": int(len(y)), "target_rate": rate, "prediction_mean": float(prediction.mean()), "mean_bias": float(prediction.mean() - rate), "absolute_mean_bias": abs(float(prediction.mean() - rate)), "brier": brier, "bss": bss, "local_score": 100000.0 * bss}


def ece(y: np.ndarray, prediction: np.ndarray) -> float:
    bin_ids = np.minimum((prediction * 10).astype(int), 9)
    accumulator = 0.0
    for bin_id in range(10):
        selected = bin_ids == bin_id
        if selected.any():
            accumulator += float(selected.mean()) * abs(float(prediction[selected].mean() - y[selected].mean()))
    return float(accumulator)


def cluster_ci(y: np.ndarray, base: np.ndarray, candidate: np.ndarray, pitcher: np.ndarray, seed: int) -> dict[str, float | int]:
    row_gain = (y - base) ** 2 - (y - candidate) ** 2
    grouped = pd.DataFrame({"pitcher": pitcher.astype(str), "gain": row_gain}).groupby("pitcher", sort=True)["gain"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    selections = np.random.default_rng(seed).integers(0, len(grouped), size=(2000, len(grouped)))
    bootstrap = sums[selections].sum(axis=1) / counts[selections].sum(axis=1)
    return {"clusters": int(len(grouped)), "repetitions": 2000, "seed": seed, "brier_gain": float(row_gain.mean()), "ci_low": float(np.quantile(bootstrap, .025)), "ci_high": float(np.quantile(bootstrap, .975))}


def compare_objects(actual: object, expected: object, path: str = "") -> tuple[float, list[str]]:
    maximum = 0.0
    failures: list[str] = []
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            return 0.0, [f"{path}:keys"]
        for key in actual:
            diff, child = compare_objects(actual[key], expected[key], f"{path}.{key}" if path else str(key))
            maximum = max(maximum, diff)
            failures.extend(child)
    elif isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            failures.append(f"{path}:length")
        else:
            for index, (left, right) in enumerate(zip(actual, expected)):
                diff, child = compare_objects(left, right, f"{path}[{index}]")
                maximum = max(maximum, diff)
                failures.extend(child)
    elif isinstance(actual, (int, float)) and not isinstance(actual, bool) and isinstance(expected, (int, float)) and not isinstance(expected, bool):
        diff = abs(float(actual) - float(expected))
        maximum = max(maximum, diff)
        if diff > 1e-10:
            failures.append(f"{path}:{diff}")
    elif actual != expected:
        failures.append(f"{path}:{actual!r}!={expected!r}")
    return maximum, failures


def main() -> None:
    checks: list[dict[str, object]] = []
    mismatches: list[str] = []

    def check(name: str, passed: bool, actual: object) -> None:
        checks.append({"name": name, "checked": True, "passed": bool(passed), "actual": actual})
        if not passed:
            mismatches.append(name)

    required = [OUT / name for name in ("audit_manifest.json", "gate_contract.json", "calibrator.json", "oof_predictions.csv", "result.json", "result.md")]
    for path in required:
        check(f"exists:{path.name}", path.is_file(), path.is_file())
    audit = json.loads((OUT / "audit_manifest.json").read_text(encoding="utf-8"))
    contract = json.loads((OUT / "gate_contract.json").read_text(encoding="utf-8"))
    calibrator = json.loads((OUT / "calibrator.json").read_text(encoding="utf-8"))
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
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
    check("contract_time_isolation", contract["fit_year"] == 2023 and contract["validation_year"] == 2024 and contract["fit_year"] < contract["validation_year"] and contract["historical_counterfactual_year"] == 2022, contract)
    check("contract_method", contract["method"] == "affine_least_squares_brier" and contract["solver"] == "numpy.linalg.lstsq" and contract["rcond"] is None and contract["clip"] == [1e-5, .99999] and contract["ece_edges"] == [index / 10 for index in range(11)] and contract["bootstrap_repetitions"] == 2000 and contract["bootstrap_seed"] == 360200, contract)
    check("contract_counts", len(contract["candidate_names"]) == contract["candidate_count"] == contract["actual_leaf_count"] == 1 and contract["gate_checks_count"] == 1 and contract["promotion_subcheck_count"] == 6 and contract["model_count"] == 1, {key: contract[key] for key in ("candidate_count", "actual_leaf_count", "gate_checks_count", "promotion_subcheck_count", "model_count")})
    check("no_unexpected_models", audit["model_count"] == result["model_count"] == 1 and not list(OUT.glob("**/*.cbm")) and not list(OUT.glob("**/*.pkl")), {"audit": audit["model_count"], "result": result["model_count"]})

    paths = {2022: OOF22 / "oof_2022.csv", 2023: BASE / "oof_2023.csv", 2024: BASE / "oof_2024.csv"}
    source_frames = {year: pd.read_csv(path, dtype={"row_id": str, "game_type": str, "pitcher_id": str}) for year, path in paths.items()}
    raw = pd.read_csv(TRAIN, dtype={"row_id": str, "pitcher_id": str}, usecols=["row_id", "season", "pitcher_id", "game_type", "control_success"], low_memory=False)
    check("raw_row_id_unique", raw["row_id"].is_unique, int(raw["row_id"].nunique()))
    alignment_failures: list[str] = []
    for year in YEARS:
        expected_raw = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        source = source_frames[year]
        if not np.array_equal(source["row_id"].astype(str).to_numpy(), expected_raw["row_id"].astype(str).to_numpy()):
            alignment_failures.append(f"row:{year}")
        if not np.array_equal(source["pitcher_id"].astype(str).to_numpy(), expected_raw["pitcher_id"].astype(str).to_numpy()):
            alignment_failures.append(f"pitcher:{year}")
        if not np.array_equal(source["game_type"].astype(str).to_numpy(), expected_raw["game_type"].astype(str).to_numpy()):
            alignment_failures.append(f"game_type:{year}")
        if not np.array_equal(source["target"].to_numpy(float), expected_raw["control_success"].to_numpy(float)):
            alignment_failures.append(f"target:{year}")
    check("source_raw_alignment", not alignment_failures, alignment_failures)
    all_targets = np.concatenate([source_frames[year]["target"].to_numpy(float) for year in YEARS])
    check("target_finite_binary", np.isfinite(all_targets).all() and set(np.unique(all_targets)).issubset({0.0, 1.0}), {"finite": bool(np.isfinite(all_targets).all()), "unique": np.unique(all_targets).tolist()})

    fit = source_frames[2023].loc[source_frames[2023]["game_type"].eq("F")]
    matrix = np.column_stack([np.ones(len(fit)), fit["prediction"].to_numpy(float)])
    coefficients, residuals, rank, singular_values = np.linalg.lstsq(matrix, fit["target"].to_numpy(float), rcond=None)
    expected_calibrator = {"experiment_id": EXPERIMENT_ID, "candidate_name": contract["candidate_names"][0], "fit_year": 2023, "fit_game_type": "F", "fit_rows": int(len(fit)), "intercept": float(coefficients[0]), "slope": float(coefficients[1]), "rank": int(rank), "singular_values": [float(value) for value in singular_values], "residual_sum_squares": [float(value) for value in residuals], "clip_low": float(contract["clip"][0]), "clip_high": float(contract["clip"][1]), "source_oof_sha256": sha256_path(paths[2023])}
    cal_diff, cal_failures = compare_objects(calibrator, expected_calibrator)
    check("calibrator_independent_refit", not cal_failures, {"max_abs_diff": cal_diff, "failures": cal_failures})

    parts: list[pd.DataFrame] = []
    for year in YEARS:
        source = source_frames[year]
        base = source["prediction"].to_numpy(float)
        candidate = base.copy()
        f_mask = source["game_type"].eq("F").to_numpy()
        candidate[f_mask] = np.clip(float(coefficients[0]) + float(coefficients[1]) * base[f_mask], float(contract["clip"][0]), float(contract["clip"][1]))
        role = {2022: "HISTORICAL_COUNTERFACTUAL_NOT_GATE", 2023: "IN_SAMPLE_FIT_DIAGNOSTIC_NOT_GATE", 2024: "NESTED_VALIDATION_GATE"}[year]
        parts.append(pd.DataFrame({"row_id": source["row_id"].astype(str), "season": year, "game_type": source["game_type"].astype(str), "pitcher_id": source["pitcher_id"].astype(str), "target": source["target"].to_numpy(float), "base_prediction": base, "candidate_prediction": candidate, "evaluation_role": role}))
    expected_output = pd.concat(parts, ignore_index=True)
    stored_output = pd.read_csv(OUT / "oof_predictions.csv", dtype={"row_id": str, "game_type": str, "pitcher_id": str, "evaluation_role": str})
    output_failures: list[str] = []
    if list(stored_output.columns) != list(expected_output.columns):
        output_failures.append("columns")
    elif len(stored_output) != len(expected_output):
        output_failures.append("rows")
    else:
        for column in expected_output:
            if pd.api.types.is_numeric_dtype(expected_output[column]):
                difference = float(np.max(np.abs(stored_output[column].to_numpy(float) - expected_output[column].to_numpy(float))))
                if difference > 1e-12:
                    output_failures.append(f"{column}:{difference}")
            elif not np.array_equal(stored_output[column].astype(str).to_numpy(), expected_output[column].astype(str).to_numpy()):
                output_failures.append(f"{column}:values")
    check("oof_predictions_recomputed", not output_failures, output_failures)
    check("oof_row_id_unique", stored_output["row_id"].is_unique, int(stored_output["row_id"].nunique()))
    prediction_values = stored_output[["base_prediction", "candidate_prediction"]].to_numpy(float)
    check("predictions_finite_range", np.isfinite(prediction_values).all() and ((prediction_values >= 0) & (prediction_values <= 1)).all(), {"min": float(prediction_values.min()), "max": float(prediction_values.max())})
    r_mask = stored_output["game_type"].eq("R").to_numpy()
    check("r_predictions_unchanged", np.array_equal(stored_output.loc[r_mask, "base_prediction"].to_numpy(float), stored_output.loc[r_mask, "candidate_prediction"].to_numpy(float)), int(np.sum(stored_output.loc[r_mask, "base_prediction"].to_numpy(float) != stored_output.loc[r_mask, "candidate_prediction"].to_numpy(float))))

    overall: dict[str, object] = {}
    f_slice: dict[str, object] = {}
    for year in YEARS:
        year_part = expected_output.loc[expected_output["season"].eq(year)]
        for destination, part in ((overall, year_part), (f_slice, year_part.loc[year_part["game_type"].eq("F")])):
            y = part["target"].to_numpy(float)
            base_prediction = part["base_prediction"].to_numpy(float)
            candidate_prediction = part["candidate_prediction"].to_numpy(float)
            base_ece = ece(y, base_prediction)
            candidate_ece = ece(y, candidate_prediction)
            destination[str(year)] = {"base": metric(y, base_prediction), "candidate": metric(y, candidate_prediction), "brier_gain": float(np.mean((y - base_prediction) ** 2) - np.mean((y - candidate_prediction) ** 2)), "base_ece": base_ece, "candidate_ece": candidate_ece, "ece_gain": float(base_ece - candidate_ece)}
    valid = expected_output.loc[expected_output["season"].eq(2024)]
    valid_f = valid.loc[valid["game_type"].eq("F")]
    ci_all = cluster_ci(valid["target"].to_numpy(float), valid["base_prediction"].to_numpy(float), valid["candidate_prediction"].to_numpy(float), valid["pitcher_id"].astype(str).to_numpy(), 360200)
    ci_f = cluster_ci(valid_f["target"].to_numpy(float), valid_f["base_prediction"].to_numpy(float), valid_f["candidate_prediction"].to_numpy(float), valid_f["pitcher_id"].astype(str).to_numpy(), 360201)
    subchecks = {"2024_all_brier_gain_positive": float(overall["2024"]["brier_gain"]) > 0, "2024_f_brier_gain_positive": float(f_slice["2024"]["brier_gain"]) > 0, "2024_all_cluster_ci_low_positive": float(ci_all["ci_low"]) > 0, "2024_f_cluster_ci_low_positive": float(ci_f["ci_low"]) > 0, "2024_f_absolute_mean_bias_reduced": float(f_slice["2024"]["candidate"]["absolute_mean_bias"]) < float(f_slice["2024"]["base"]["absolute_mean_bias"]), "2024_f_ece_reduced": float(f_slice["2024"]["candidate_ece"]) < float(f_slice["2024"]["base_ece"])}
    composite_gate = {"all_six_subchecks": all(subchecks.values())}
    expected_candidate = {"candidate_name": contract["candidate_names"][0], "calibrator": expected_calibrator, "overall": overall, "f_slice": f_slice, "cluster_ci": {"2024_all": ci_all, "2024_f": ci_f}, "promotion_subchecks": subchecks, "promotion_subcheck_count": len(subchecks), "passed_subcheck_count": sum(subchecks.values()), "composite_gate": composite_gate, "promotion_pass": all(composite_gate.values())}
    candidate_diff, candidate_failures = compare_objects(result["candidate"], expected_candidate)
    check("metrics_ci_ece_gates_recomputed", not candidate_failures, {"max_abs_diff": candidate_diff, "failures": candidate_failures})
    expected_scope = "EVAL_PASS/HOLD_FOR_FULLTRAIN_APPROVAL" if all(composite_gate.values()) else "FAIL/HOLD"
    check("promotion_scope", result["promotion_scope"] == expected_scope, {"actual": result["promotion_scope"], "expected": expected_scope})
    check("leaf_gate_subcheck_counts", result["candidate_count"] == result["actual_leaf_count"] == audit["candidate_count"] == audit["leaf_count"] == 1 and result["gate_checks_count"] == audit["gate_checks_count"] == 1 and result["promotion_subcheck_count"] == audit["promotion_subcheck_count"] == len(subchecks) == 6, {"candidate": result["candidate_count"], "leaf": result["actual_leaf_count"], "gate": result["gate_checks_count"], "subchecks": result["promotion_subcheck_count"]})
    check("row_model_counts", result["oof_rows"] == audit["oof_rows"] == len(expected_output) and result["model_count"] == audit["model_count"] == 1, {"rows": result["oof_rows"], "models": result["model_count"]})
    check("no_test_fulltrain_zip", result["test_inference_performed"] is False and result["full_train_performed"] is False and result["zip_created"] is False, {"test": result["test_inference_performed"], "fulltrain": result["full_train_performed"], "zip": result["zip_created"]})
    markdown = (OUT / "result.md").read_text(encoding="utf-8")
    match = re.search(r"<!-- RESULT_JSON_BEGIN\n(.*?)\nRESULT_JSON_END -->", markdown, re.DOTALL)
    embedded = json.loads(match.group(1)) if match else None
    expected_embedded = {key: value for key, value in result.items() if key != "elapsed_seconds"}
    check("json_markdown_embedded", embedded == expected_embedded, embedded == expected_embedded)

    status = "AUDIT_VERIFIED" if not mismatches else "FAIL"
    promotion_status = expected_scope if status == "AUDIT_VERIFIED" else "FAIL/HOLD"
    report = {"experiment_id": EXPERIMENT_ID, "status": status, "promotion_status": promotion_status, "checked_count": len(checks), "passed_count": sum(bool(row["passed"]) for row in checks), "mismatch_count": len(mismatches), "mismatches": mismatches, "candidate_count": 1, "actual_leaf_count": 1, "gate_checks_count": 1, "promotion_subcheck_count": 6, "passed_subcheck_count": sum(subchecks.values()), "model_count": 1, "oof_rows": len(expected_output), "candidate": expected_candidate, "checks": checks}
    report_path = OUT / "validation_report.json"
    write_json(report_path, report)
    attestation = {"experiment_id": EXPERIMENT_ID, "status": status, "promotion_status": promotion_status, "manifest_sha256": sha256_path(OUT / "audit_manifest.json"), "validation_report_sha256": sha256_path(report_path), "validator_sha256": sha256_path(Path(__file__).resolve()), "checked_count": report["checked_count"], "passed_count": report["passed_count"], "mismatch_count": report["mismatch_count"], "candidate_count": 1, "actual_leaf_count": 1, "gate_checks_count": 1, "promotion_subcheck_count": 6, "passed_subcheck_count": report["passed_subcheck_count"], "model_count": 1, "oof_rows": len(expected_output)}
    write_json(OUT / "audit_attestation.json", attestation)
    print(json.dumps(attestation, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
