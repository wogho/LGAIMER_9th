#!/usr/bin/env python3
"""Independent source-level validator for REF4-F-DRIFT-DIAG-035A."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "REF4-F-DRIFT-DIAG-035A"
OUT = ROOT / "model" / EXPERIMENT_ID
BASE = ROOT / "model" / "REF4-EXACT-OOF-031A"
OOF22 = ROOT / "model" / "REF4-ADAPTIVE-GATE-031B"
SOURCE = ROOT / "model" / "REF4-CHAMPION-STACK-030"
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
    return {"rows": int(len(y)), "target_rate": rate, "prediction_mean": float(prediction.mean()), "mean_bias": float(prediction.mean() - rate), "brier": brier, "bss": bss, "local_score": 100000.0 * bss}


def independent_prior(raw: pd.DataFrame, year: int) -> pd.Series:
    history = raw.loc[raw["season"].lt(year)]
    counts = history.groupby(["pitcher_id", "season", "game_type"], observed=True).size().rename("n").reset_index()
    dominant = counts.sort_values("n").groupby(["pitcher_id", "season"], observed=True).tail(1)
    latest = dominant.sort_values("season").groupby("pitcher_id", observed=True).tail(1)
    return latest.set_index(latest["pitcher_id"].astype(str))["game_type"].astype(str)


def independent_recombine(oof: pd.DataFrame, regime: dict[str, float], manifest: dict[str, object]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    f_mask = oof["game_type"].astype(str).eq("F").to_numpy()
    g0 = oof["p_v2_global"].to_numpy(float)
    g1 = oof["p_v3_55_global"].to_numpy(float)
    g2 = oof["p_v3_30_global"].to_numpy(float)
    recent = g2 + float(regime["v330_recent_inner_scale"]) * (oof["p_v3_30_f_recent"].to_numpy(float) - g2)
    f2 = float(regime["v330_all_weight"]) * oof["p_v3_30_f_all"].to_numpy(float) + (1.0 - float(regime["v330_all_weight"])) * recent
    adjusted = [
        np.where(f_mask, g0 + float(regime["v2_scale"]) * (oof["p_v2_f"].to_numpy(float) - g0), g0),
        np.where(f_mask, g1 + float(regime["v355_scale"]) * (oof["p_v3_55_f"].to_numpy(float) - g1), g1),
        np.where(f_mask, g2 + float(regime["v330_scale"]) * (f2 - g2), g2),
    ]
    risk_global: list[np.ndarray] = []
    risk_adjusted: list[np.ndarray] = []
    for label in ("middle", "wild", "reverse"):
        global_value = oof[f"risk_{label}_global"].to_numpy(float)
        f_value = oof[f"risk_{label}_f"].to_numpy(float)
        risk_global.append(global_value)
        risk_adjusted.append(np.where(f_mask, global_value + float(regime["subtype_scale"]) * (f_value - global_value), global_value))
    weights = np.asarray(manifest["main_weights"], dtype=float)
    ensemble = np.sum(np.vstack(adjusted) * weights[:, None], axis=0) / weights.sum()
    raw_prediction = float(manifest["stack_intercept"]) + np.column_stack([ensemble, *risk_adjusted]) @ np.asarray(manifest["stack_coefficients"], dtype=float)
    final = np.clip(raw_prediction + float(manifest["global_shift"]), 1e-5, 1 - 1e-5)
    components: dict[str, np.ndarray] = {"prediction_no_shift": np.clip(raw_prediction, 1e-5, 1 - 1e-5)}
    for label, values in zip(("v2", "v355", "v330"), (g0, g1, g2)):
        components[f"{label}_global"] = values
    for label, values in zip(("v2", "v355", "v330"), adjusted):
        components[f"{label}_adjusted"] = values
    for label, values in zip(("middle", "wild", "reverse"), risk_global):
        components[f"risk_{label}_global"] = values
    for label, values in zip(("middle", "wild", "reverse"), risk_adjusted):
        components[f"risk_{label}_adjusted"] = values
    return final, components


def safe_corr(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def slice_metrics(frame: pd.DataFrame, dimensions: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dimension in dimensions:
        for (year, value), part in frame.groupby(["season", dimension], observed=True, sort=True, dropna=False):
            y = part["target"].to_numpy(float)
            metrics = {name: metric(y, part[f"prediction_{name}"].to_numpy(float)) for name in ("current", "no_shift", "global_only", "f075")}
            season_rate = float(frame.loc[frame["season"].eq(year), "target"].mean())
            current_loss = (y - part["prediction_current"].to_numpy(float)) ** 2
            rows.append({
                "dimension": dimension, "value": str(value), "season": int(year), "rows": int(len(part)), "pitcher_count": int(part["pitcher_id"].nunique()),
                "target_rate": metrics["current"]["target_rate"], "current_prediction_mean": metrics["current"]["prediction_mean"],
                "current_mean_bias": metrics["current"]["mean_bias"], "current_brier": metrics["current"]["brier"], "current_bss": metrics["current"]["bss"],
                "no_shift_brier": metrics["no_shift"]["brier"], "shift_removal_brier_gain": float(metrics["current"]["brier"] - metrics["no_shift"]["brier"]),
                "global_only_brier": metrics["global_only"]["brier"], "global_only_brier_gain": float(metrics["current"]["brier"] - metrics["global_only"]["brier"]),
                "f075_brier": metrics["f075"]["brier"], "f075_brier_gain": float(metrics["current"]["brier"] - metrics["f075"]["brier"]),
                "current_excess_loss_sum": float(np.sum(current_loss - (y - season_rate) ** 2)),
            })
    return pd.DataFrame(rows).sort_values(["dimension", "season", "value"]).reset_index(drop=True)


def compare_frames(actual: pd.DataFrame, expected: pd.DataFrame, tolerance: float = 1e-10) -> dict[str, object]:
    failures: list[str] = []
    maximum = 0.0
    if list(actual.columns) != list(expected.columns):
        return {"failures": ["columns"], "max_abs_diff": maximum}
    if len(actual) != len(expected):
        return {"failures": ["rows"], "max_abs_diff": maximum}
    for column in expected.columns:
        if pd.api.types.is_numeric_dtype(expected[column]):
            left = pd.to_numeric(actual[column], errors="coerce").to_numpy(float)
            right = pd.to_numeric(expected[column], errors="coerce").to_numpy(float)
            if not np.array_equal(np.isfinite(left), np.isfinite(right)):
                failures.append(f"{column}:finite")
                continue
            finite = np.isfinite(right)
            diff = float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else 0.0
            maximum = max(maximum, diff)
            if diff > tolerance:
                failures.append(f"{column}:{diff}")
        elif not np.array_equal(actual[column].astype(str).to_numpy(), expected[column].astype(str).to_numpy()):
            failures.append(f"{column}:values")
    return {"failures": failures, "max_abs_diff": maximum}


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

    required = [OUT / name for name in ("audit_manifest.json", "diagnostic_contract.json", "result.json", "result.md", "f_diagnostic_rows.csv", "historical_rates.csv", "slice_metrics.csv", "cohort_metrics.csv", "feature_shift.csv", "channel_summary.csv")]
    for path in required:
        check(f"exists:{path.name}", path.is_file(), path.is_file())
    audit = json.loads((OUT / "audit_manifest.json").read_text(encoding="utf-8"))
    contract = json.loads((OUT / "diagnostic_contract.json").read_text(encoding="utf-8"))
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
    check("diagnostic_only_counts", contract["candidate_count"] == result["candidate_count"] == result["actual_leaf_count"] == audit["candidate_count"] == audit["leaf_count"] == 0 and contract["gate_checks_count"] == result["gate_checks_count"] == audit["gate_checks_count"] == 0, {"candidate": result["candidate_count"], "leaf": result["actual_leaf_count"], "gates": result["gate_checks_count"]})
    check("no_models", result["model_count"] == audit["model_count"] == 0 and not list(OUT.glob("**/*.cbm")), {"result": result["model_count"], "audit": audit["model_count"]})
    check("no_test_fulltrain_zip", result["test_inference_performed"] is False and result["full_train_performed"] is False and result["zip_created"] is False, {"test": result["test_inference_performed"], "fulltrain": result["full_train_performed"], "zip": result["zip_created"]})

    raw = pd.read_csv(TRAIN, dtype={"row_id": str, "pitcher_id": str}, low_memory=False)
    check("raw_row_id_unique", raw["row_id"].is_unique, int(raw["row_id"].nunique()))
    target = pd.to_numeric(raw["control_success"], errors="coerce").to_numpy(float)
    check("raw_target_finite_binary", np.isfinite(target).all() and set(np.unique(target)).issubset({0.0, 1.0}), {"finite": bool(np.isfinite(target).all()), "unique": np.unique(target).tolist()})
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    current_regime = {key: float(value) for key, value in json.loads((SOURCE / "f_regime_meta.json").read_text(encoding="utf-8")).items()}
    global_regime = dict(current_regime)
    f075_regime = dict(current_regime)
    for key in ("v2_scale", "v355_scale", "v330_scale", "subtype_scale"):
        global_regime[key] = 0.0
        f075_regime[key] *= .75
    input_paths = {2022: OOF22 / "oof_2022.csv", 2023: BASE / "oof_2023.csv", 2024: BASE / "oof_2024.csv"}
    mapping_paths = {2022: OOF22 / "fold_2022" / "pitcher_trackman_mapping.csv", 2023: BASE / "fold_2023" / "pitcher_trackman_mapping.csv", 2024: BASE / "fold_2024" / "pitcher_trackman_mapping.csv"}
    parts: list[pd.DataFrame] = []
    formula_differences: dict[str, float] = {}
    alignment_failures: list[str] = []
    for year in YEARS:
        oof = pd.read_csv(input_paths[year], dtype={"row_id": str, "game_type": str, "pitcher_id": str})
        raw_year = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        if not np.array_equal(oof["row_id"].astype(str).to_numpy(), raw_year["row_id"].astype(str).to_numpy()):
            alignment_failures.append(f"row:{year}")
        if not np.array_equal(oof["target"].to_numpy(float), raw_year["control_success"].to_numpy(float)):
            alignment_failures.append(f"target:{year}")
        current, components = independent_recombine(oof, current_regime, manifest)
        global_only, _ = independent_recombine(oof, global_regime, manifest)
        f075, _ = independent_recombine(oof, f075_regime, manifest)
        formula_differences[str(year)] = float(np.max(np.abs(current - oof["prediction"].to_numpy(float))))
        mask = raw_year["game_type"].astype(str).eq("F").to_numpy()
        indices = np.flatnonzero(mask)
        prior = independent_prior(raw, year)
        pitcher = raw_year.loc[mask, "pitcher_id"].astype(str)
        prior_type = pitcher.map(prior).fillna("NEW").astype(str)
        mapped = set(pd.read_csv(mapping_paths[year], dtype={"pitcher_id": str})["pitcher_id"].astype(str))
        pitcher_n = pd.to_numeric(raw_year.loc[mask, "asof_pitcher_n"], errors="coerce").fillna(0)
        part = pd.DataFrame({
            "row_id": oof.loc[mask, "row_id"].astype(str).to_numpy(), "season": year,
            "game_month": pd.to_numeric(raw_year.loc[mask, "game_month"], errors="coerce").fillna(-1).astype(int).to_numpy(),
            "pitcher_id": pitcher.to_numpy(), "pitcher_hand": raw_year.loc[mask, "pitcher_hand"].astype(str).to_numpy(),
            "prior_type": prior_type.to_numpy(), "known_status": np.where(prior_type.eq("NEW"), "NEW", "KNOWN"),
            "pitcher_n_bin": pd.cut(pitcher_n, [-1, 0, 9, 49, 199, np.inf], labels=["0", "1-9", "10-49", "50-199", "200+"]).astype(str).to_numpy(),
            "trackman_status": np.where(pitcher.isin(mapped), "MAPPED", "UNMAPPED"),
            "target": raw_year.loc[mask, "control_success"].to_numpy(float),
            "prediction_current": current[indices], "prediction_no_shift": components["prediction_no_shift"][indices],
            "prediction_global_only": global_only[indices], "prediction_f075": f075[indices],
        })
        for name, values in components.items():
            if name != "prediction_no_shift":
                part[name] = values[indices]
        for column in raw.columns:
            if column.startswith("asof_") or column in ("inning", "balls_before", "strikes_before", "outs_before", "run_total_before", "score_diff_home", "score_diff_pitcher_team", "num_runners_on", "home_win_expectancy", "away_win_expectancy", "li"):
                part[f"raw__{column}"] = pd.to_numeric(raw_year.loc[mask, column], errors="coerce").to_numpy(float)
        parts.append(part)
    check("oof_raw_alignment", not alignment_failures, alignment_failures)
    check("source_formula_current", max(formula_differences.values()) <= 1e-12, formula_differences)
    expected_frame = pd.concat(parts, ignore_index=True)
    patterns = expected_frame.groupby("pitcher_id", sort=True)["season"].agg(lambda values: "|".join(str(int(value)) for value in sorted(set(values)))).rename("f_years_pattern")
    expected_frame["f_years_pattern"] = expected_frame["pitcher_id"].map(patterns).astype(str)
    stored_frame = pd.read_csv(OUT / "f_diagnostic_rows.csv", dtype={"row_id": str, "pitcher_id": str, "prior_type": str, "known_status": str, "pitcher_n_bin": str, "trackman_status": str, "f_years_pattern": str})
    frame_comparison = compare_frames(stored_frame, expected_frame)
    check("f_diagnostic_rows_recomputed", not frame_comparison["failures"], frame_comparison)
    check("f_row_id_unique", stored_frame["row_id"].is_unique, int(stored_frame["row_id"].nunique()))
    prediction_columns = [column for column in stored_frame if column.startswith("prediction_")]
    prediction_valid = all(np.isfinite(stored_frame[column].to_numpy(float)).all() and stored_frame[column].between(0, 1).all() for column in prediction_columns)
    check("predictions_finite_range", prediction_valid, prediction_columns)

    expected_historical = raw.groupby(["season", "game_type"], observed=True, sort=True)["control_success"].agg(rows="size", target_rate="mean").reset_index()
    stored_historical = pd.read_csv(OUT / "historical_rates.csv")
    historical_comparison = compare_frames(stored_historical, expected_historical)
    check("historical_rates_recomputed", not historical_comparison["failures"], historical_comparison)
    expected_slices = slice_metrics(expected_frame, [str(value) for value in contract["slice_dimensions"]])
    stored_slices = pd.read_csv(OUT / "slice_metrics.csv", dtype={"dimension": str, "value": str})
    slice_comparison = compare_frames(stored_slices, expected_slices)
    check("slice_metrics_recomputed", not slice_comparison["failures"], slice_comparison)
    expected_cohort = expected_slices.loc[expected_slices["dimension"].eq("f_years_pattern")].reset_index(drop=True)
    stored_cohort = pd.read_csv(OUT / "cohort_metrics.csv", dtype={"dimension": str, "value": str})
    cohort_comparison = compare_frames(stored_cohort, expected_cohort)
    check("cohort_metrics_recomputed", not cohort_comparison["failures"], cohort_comparison)

    feature_rows: list[dict[str, object]] = []
    feature_columns = sorted(column for column in expected_frame if column.startswith("raw__"))
    for feature in feature_columns:
        target_values = expected_frame.loc[expected_frame["season"].eq(2023), feature].to_numpy(float)
        target_finite = target_values[np.isfinite(target_values)]
        for reference_year in (2022, 2024):
            reference_values = expected_frame.loc[expected_frame["season"].eq(reference_year), feature].to_numpy(float)
            reference_finite = reference_values[np.isfinite(reference_values)]
            pooled_std = float(np.sqrt((np.var(target_finite) + np.var(reference_finite)) / 2.0)) if len(target_finite) and len(reference_finite) else 0.0
            feature_rows.append({
                "feature": feature.removeprefix("raw__"), "target_year": 2023, "reference_year": reference_year,
                "target_rows": int(len(target_values)), "reference_rows": int(len(reference_values)),
                "target_missing_rate": float(1.0 - len(target_finite) / len(target_values)), "reference_missing_rate": float(1.0 - len(reference_finite) / len(reference_values)),
                "target_mean": float(np.mean(target_finite)), "reference_mean": float(np.mean(reference_finite)),
                "target_median": float(np.median(target_finite)), "reference_median": float(np.median(reference_finite)),
                "target_q10": float(np.quantile(target_finite, .1)), "reference_q10": float(np.quantile(reference_finite, .1)),
                "target_q90": float(np.quantile(target_finite, .9)), "reference_q90": float(np.quantile(reference_finite, .9)),
                "standardized_mean_difference": float((np.mean(target_finite) - np.mean(reference_finite)) / pooled_std) if pooled_std > 0 else 0.0,
            })
    expected_feature = pd.DataFrame(feature_rows).sort_values(["reference_year", "feature"]).reset_index(drop=True)
    stored_feature = pd.read_csv(OUT / "feature_shift.csv")
    feature_comparison = compare_frames(stored_feature, expected_feature)
    check("feature_shift_recomputed", not feature_comparison["failures"], feature_comparison)

    weights = np.asarray(manifest["main_weights"], float)
    coefficients = np.asarray(manifest["stack_coefficients"], float)
    specs = [("v2", "v2_global", "v2_adjusted", coefficients[0] * weights[0] / weights.sum()), ("v355", "v355_global", "v355_adjusted", coefficients[0] * weights[1] / weights.sum()), ("v330", "v330_global", "v330_adjusted", coefficients[0] * weights[2] / weights.sum()), ("risk_middle", "risk_middle_global", "risk_middle_adjusted", coefficients[1]), ("risk_wild", "risk_wild_global", "risk_wild_adjusted", coefficients[2]), ("risk_reverse", "risk_reverse_global", "risk_reverse_adjusted", coefficients[3])]
    channel_rows: list[dict[str, object]] = []
    for year in YEARS:
        part = expected_frame.loc[expected_frame["season"].eq(year)]
        y = part["target"].to_numpy(float)
        current_loss = (y - part["prediction_current"].to_numpy(float)) ** 2
        global_loss = (y - part["prediction_global_only"].to_numpy(float)) ** 2
        for name, global_column, adjusted_column, coefficient in specs:
            delta = part[adjusted_column].to_numpy(float) - part[global_column].to_numpy(float)
            contribution = coefficient * delta
            channel_rows.append({"season": year, "channel": name, "rows": int(len(part)), "stack_multiplier": float(coefficient), "global_mean": float(part[global_column].mean()), "adjusted_mean": float(part[adjusted_column].mean()), "raw_delta_mean": float(delta.mean()), "raw_delta_abs_mean": float(np.abs(delta).mean()), "prediction_contribution_mean": float(contribution.mean()), "prediction_contribution_abs_mean": float(np.abs(contribution).mean()), "correlation_contribution_target": safe_corr(contribution, y), "correlation_contribution_current_loss": safe_corr(contribution, current_loss), "f_expert_brier_gain_vs_global_only": float(global_loss.mean() - current_loss.mean())})
    expected_channel = pd.DataFrame(channel_rows).sort_values(["season", "channel"]).reset_index(drop=True)
    stored_channel = pd.read_csv(OUT / "channel_summary.csv")
    channel_comparison = compare_frames(stored_channel, expected_channel)
    check("channel_summary_recomputed", not channel_comparison["failures"], channel_comparison)

    overall: dict[str, object] = {}
    for year in YEARS:
        part = expected_frame.loc[expected_frame["season"].eq(year)]
        y = part["target"].to_numpy(float)
        variants = {name: metric(y, part[f"prediction_{name}"].to_numpy(float)) for name in ("current", "no_shift", "global_only", "f075")}
        overall[str(year)] = {"metrics": variants, "shift_removal_brier_gain": float(variants["current"]["brier"] - variants["no_shift"]["brier"]), "global_only_brier_gain": float(variants["current"]["brier"] - variants["global_only"]["brier"]), "f075_brier_gain": float(variants["current"]["brier"] - variants["f075"]["brier"])}
    f_history = expected_historical.loc[expected_historical["game_type"].astype(str).eq("F")]
    top_features = expected_feature.assign(abs_smd=expected_feature["standardized_mean_difference"].abs()).sort_values("abs_smd", ascending=False).head(10)
    top_slices = expected_slices.loc[expected_slices["season"].eq(2023) & expected_slices["rows"].ge(500)].sort_values("current_excess_loss_sum", ascending=False).head(10)
    findings = {"historical_f_target_rate": {str(int(row.season)): {"rows": int(row.rows), "target_rate": float(row.target_rate)} for row in f_history.itertuples(index=False)}, "f_2023_target_drop_vs_2022": float(overall["2023"]["metrics"]["current"]["target_rate"] - overall["2022"]["metrics"]["current"]["target_rate"]), "f_2023_target_difference_vs_2024": float(overall["2023"]["metrics"]["current"]["target_rate"] - overall["2024"]["metrics"]["current"]["target_rate"]), "f_2023_current_bias": float(overall["2023"]["metrics"]["current"]["mean_bias"]), "f_2023_shift_removal_gain": float(overall["2023"]["shift_removal_brier_gain"]), "f_2023_global_only_gain": float(overall["2023"]["global_only_brier_gain"]), "f_2023_f075_gain": float(overall["2023"]["f075_brier_gain"]), "top_feature_shifts": top_features[["feature", "reference_year", "standardized_mean_difference"]].to_dict(orient="records"), "top_2023_loss_slices": top_slices[["dimension", "value", "rows", "current_excess_loss_sum", "current_mean_bias"]].to_dict(orient="records"), "diagnostic_status": "DESCRIPTIVE_NOT_CAUSAL", "automatic_candidate_status": "NO_AUTOMATIC_CANDIDATE"}
    output_counts = {"f_diagnostic_rows": int(len(expected_frame)), "historical_rate_rows": int(len(expected_historical)), "slice_rows": int(len(expected_slices)), "cohort_rows": int(len(expected_cohort)), "feature_shift_rows": int(len(expected_feature)), "channel_summary_rows": int(len(expected_channel))}
    expected_result = {"experiment_id": EXPERIMENT_ID, "status": "PENDING_AUDIT", "diagnostic_status": "COMPLETE", "source_official_score": 1068.25021, "overall": overall, "findings": findings, "output_counts": output_counts, "candidate_count": 0, "actual_leaf_count": 0, "gate_checks_count": 0, "model_count": 0, "test_inference_performed": False, "full_train_performed": False, "zip_created": False}
    stored_result = {key: value for key, value in result.items() if key != "elapsed_seconds"}
    result_diff, result_failures = compare_objects(stored_result, expected_result)
    check("result_recomputed", not result_failures, {"max_abs_diff": result_diff, "failures": result_failures})
    check("output_counts", audit["output_counts"] == output_counts and audit["f_diagnostic_rows"] == len(expected_frame), {"audit": audit["output_counts"], "expected": output_counts})
    markdown = (OUT / "result.md").read_text(encoding="utf-8")
    match = re.search(r"<!-- RESULT_JSON_BEGIN\n(.*?)\nRESULT_JSON_END -->", markdown, re.DOTALL)
    embedded = json.loads(match.group(1)) if match else None
    check("json_markdown_embedded", embedded == expected_result, embedded == expected_result)

    status = "AUDIT_VERIFIED" if not mismatches else "FAIL"
    report = {"experiment_id": EXPERIMENT_ID, "status": status, "diagnostic_status": "COMPLETE" if status == "AUDIT_VERIFIED" else "INCOMPLETE", "checked_count": len(checks), "passed_count": sum(bool(row["passed"]) for row in checks), "mismatch_count": len(mismatches), "mismatches": mismatches, "candidate_count": 0, "actual_leaf_count": 0, "gate_checks_count": 0, "model_count": 0, "oof_rows": audit["oof_rows"], "output_counts": output_counts, "overall": overall, "findings": findings, "formula_max_abs_diff": max(formula_differences.values()), "checks": checks}
    report_path = OUT / "validation_report.json"
    write_json(report_path, report)
    attestation = {"experiment_id": EXPERIMENT_ID, "status": status, "diagnostic_status": report["diagnostic_status"], "manifest_sha256": sha256_path(OUT / "audit_manifest.json"), "validation_report_sha256": sha256_path(report_path), "validator_sha256": sha256_path(Path(__file__).resolve()), "checked_count": report["checked_count"], "passed_count": report["passed_count"], "mismatch_count": report["mismatch_count"], "candidate_count": 0, "actual_leaf_count": 0, "gate_checks_count": 0, "model_count": 0, "oof_rows": audit["oof_rows"], "output_counts": output_counts}
    write_json(OUT / "audit_attestation.json", attestation)
    print(json.dumps(attestation, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
