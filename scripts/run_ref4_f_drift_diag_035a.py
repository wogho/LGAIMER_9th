#!/usr/bin/env python3
"""Read-only diagnosis of the 2023 F distribution and expert-adjustment drift."""
from __future__ import annotations

import hashlib
import json
import time
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
    denominator = rate * (1.0 - rate)
    bss = float(1.0 - brier / denominator)
    return {"rows": int(len(y)), "target_rate": rate, "prediction_mean": float(prediction.mean()), "mean_bias": float(prediction.mean() - rate), "brier": brier, "bss": bss, "local_score": 100000.0 * bss}


def prior_type_table(raw: pd.DataFrame, year: int) -> pd.Series:
    counts = raw.loc[raw["season"].lt(year)].groupby(["pitcher_id", "season", "game_type"], observed=True).size().rename("n").reset_index()
    dominant = counts.sort_values("n").groupby(["pitcher_id", "season"], observed=True).tail(1)
    latest = dominant.sort_values("season").groupby("pitcher_id", observed=True).tail(1)
    return latest.set_index(latest["pitcher_id"].astype(str))["game_type"].astype(str)


def recombine(oof: pd.DataFrame, regime: dict[str, float], manifest: dict[str, object]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    is_f = oof["game_type"].eq("F").to_numpy()
    globals_ = [oof["p_v2_global"].to_numpy(float), oof["p_v3_55_global"].to_numpy(float), oof["p_v3_30_global"].to_numpy(float)]
    inner = globals_[2] + regime["v330_recent_inner_scale"] * (oof["p_v3_30_f_recent"].to_numpy(float) - globals_[2])
    f30 = regime["v330_all_weight"] * oof["p_v3_30_f_all"].to_numpy(float) + (1.0 - regime["v330_all_weight"]) * inner
    adjusted = [
        np.where(is_f, globals_[0] + regime["v2_scale"] * (oof["p_v2_f"].to_numpy(float) - globals_[0]), globals_[0]),
        np.where(is_f, globals_[1] + regime["v355_scale"] * (oof["p_v3_55_f"].to_numpy(float) - globals_[1]), globals_[1]),
        np.where(is_f, globals_[2] + regime["v330_scale"] * (f30 - globals_[2]), globals_[2]),
    ]
    risk_globals: list[np.ndarray] = []
    risk_adjusted: list[np.ndarray] = []
    for name in ("middle", "wild", "reverse"):
        global_risk = oof[f"risk_{name}_global"].to_numpy(float)
        f_risk = oof[f"risk_{name}_f"].to_numpy(float)
        risk_globals.append(global_risk)
        risk_adjusted.append(np.where(is_f, global_risk + regime["subtype_scale"] * (f_risk - global_risk), global_risk))
    weights = np.asarray(manifest["main_weights"], float)
    main = np.average(np.vstack(adjusted), axis=0, weights=weights)
    no_shift_raw = float(manifest["stack_intercept"]) + np.column_stack([main, *risk_adjusted]) @ np.asarray(manifest["stack_coefficients"], float)
    current = np.clip(no_shift_raw + float(manifest["global_shift"]), 1e-5, 1 - 1e-5)
    components: dict[str, np.ndarray] = {"prediction_no_shift": np.clip(no_shift_raw, 1e-5, 1 - 1e-5)}
    for name, values in zip(("v2", "v355", "v330"), globals_):
        components[f"{name}_global"] = values
    for name, values in zip(("v2", "v355", "v330"), adjusted):
        components[f"{name}_adjusted"] = values
    for name, values in zip(("middle", "wild", "reverse"), risk_globals):
        components[f"risk_{name}_global"] = values
    for name, values in zip(("middle", "wild", "reverse"), risk_adjusted):
        components[f"risk_{name}_adjusted"] = values
    return current, components


def safe_corr(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or float(np.std(left)) == 0.0 or float(np.std(right)) == 0.0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def add_slice(rows: list[dict[str, object]], frame: pd.DataFrame, dimension: str) -> None:
    for (year, value), part in frame.groupby(["season", dimension], observed=True, sort=True, dropna=False):
        y = part["target"].to_numpy(float)
        metrics = {name: metric(y, part[f"prediction_{name}"].to_numpy(float)) for name in ("current", "no_shift", "global_only", "f075")}
        reference_rate = float(frame.loc[frame["season"].eq(year), "target"].mean())
        reference_loss = (y - reference_rate) ** 2
        current_loss = (y - part["prediction_current"].to_numpy(float)) ** 2
        rows.append({
            "dimension": dimension, "value": str(value), "season": int(year), "rows": int(len(part)), "pitcher_count": int(part["pitcher_id"].nunique()),
            "target_rate": metrics["current"]["target_rate"], "current_prediction_mean": metrics["current"]["prediction_mean"],
            "current_mean_bias": metrics["current"]["mean_bias"], "current_brier": metrics["current"]["brier"], "current_bss": metrics["current"]["bss"],
            "no_shift_brier": metrics["no_shift"]["brier"], "shift_removal_brier_gain": float(metrics["current"]["brier"] - metrics["no_shift"]["brier"]),
            "global_only_brier": metrics["global_only"]["brier"], "global_only_brier_gain": float(metrics["current"]["brier"] - metrics["global_only"]["brier"]),
            "f075_brier": metrics["f075"]["brier"], "f075_brier_gain": float(metrics["current"]["brier"] - metrics["f075"]["brier"]),
            "current_excess_loss_sum": float(np.sum(current_loss - reference_loss)),
        })


def main() -> None:
    started = time.time()
    preflight = json.loads((OUT / "preflight_report.json").read_text(encoding="utf-8"))
    contract = json.loads((OUT / "diagnostic_contract.json").read_text(encoding="utf-8"))
    if preflight["status"] != "AUDIT_VERIFIED" or preflight["mismatch_count"] != 0:
        raise RuntimeError("035A preflight must be AUDIT_VERIFIED")
    if contract["candidate_count"] != 0 or contract["gate_checks_count"] != 0:
        raise RuntimeError("035A contract must remain diagnostic-only")

    raw = pd.read_csv(TRAIN, dtype={"row_id": str, "pitcher_id": str}, low_memory=False)
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
    for year in YEARS:
        oof = pd.read_csv(input_paths[year], dtype={"row_id": str, "game_type": str, "pitcher_id": str})
        raw_year = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        if not np.array_equal(oof["row_id"].astype(str).to_numpy(), raw_year["row_id"].astype(str).to_numpy()):
            raise RuntimeError(f"row order mismatch: {year}")
        current, components = recombine(oof, current_regime, manifest)
        global_only, _ = recombine(oof, global_regime, manifest)
        f075, _ = recombine(oof, f075_regime, manifest)
        if float(np.max(np.abs(current - oof["prediction"].to_numpy(float)))) > 1e-12:
            raise RuntimeError(f"current formula mismatch: {year}")
        mask = raw_year["game_type"].astype(str).eq("F").to_numpy()
        prior = prior_type_table(raw, year)
        pitcher = raw_year.loc[mask, "pitcher_id"].astype(str)
        prior_type = pitcher.map(prior).fillna("NEW").astype(str)
        mapped = set(pd.read_csv(mapping_paths[year], dtype={"pitcher_id": str})["pitcher_id"].astype(str))
        pitcher_n = pd.to_numeric(raw_year.loc[mask, "asof_pitcher_n"], errors="coerce").fillna(0)
        pitcher_n_bin = pd.cut(pitcher_n, [-1, 0, 9, 49, 199, np.inf], labels=["0", "1-9", "10-49", "50-199", "200+"]).astype(str)
        indices = np.flatnonzero(mask)
        part = pd.DataFrame({
            "row_id": oof.loc[mask, "row_id"].astype(str).to_numpy(), "season": year,
            "game_month": pd.to_numeric(raw_year.loc[mask, "game_month"], errors="coerce").fillna(-1).astype(int).to_numpy(),
            "pitcher_id": pitcher.to_numpy(), "pitcher_hand": raw_year.loc[mask, "pitcher_hand"].astype(str).to_numpy(),
            "prior_type": prior_type.to_numpy(), "known_status": np.where(prior_type.eq("NEW"), "NEW", "KNOWN"),
            "pitcher_n_bin": pitcher_n_bin.to_numpy(), "trackman_status": np.where(pitcher.isin(mapped), "MAPPED", "UNMAPPED"),
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
    frame = pd.concat(parts, ignore_index=True)
    pitcher_patterns = frame.groupby("pitcher_id", sort=True)["season"].agg(lambda values: "|".join(str(int(value)) for value in sorted(set(values)))).rename("f_years_pattern")
    frame["f_years_pattern"] = frame["pitcher_id"].map(pitcher_patterns).astype(str)
    frame.to_csv(OUT / "f_diagnostic_rows.csv", index=False)

    historical = raw.groupby(["season", "game_type"], observed=True, sort=True)["control_success"].agg(rows="size", target_rate="mean").reset_index()
    historical.to_csv(OUT / "historical_rates.csv", index=False)

    slice_rows: list[dict[str, object]] = []
    for dimension in contract["slice_dimensions"]:
        add_slice(slice_rows, frame, str(dimension))
    slices = pd.DataFrame(slice_rows).sort_values(["dimension", "season", "value"]).reset_index(drop=True)
    slices.to_csv(OUT / "slice_metrics.csv", index=False)

    cohort = slices.loc[slices["dimension"].eq("f_years_pattern")].reset_index(drop=True)
    cohort.to_csv(OUT / "cohort_metrics.csv", index=False)

    feature_rows: list[dict[str, object]] = []
    feature_columns = sorted(column for column in frame if column.startswith("raw__"))
    for feature in feature_columns:
        target_values = frame.loc[frame["season"].eq(2023), feature].to_numpy(float)
        target_finite = target_values[np.isfinite(target_values)]
        for reference_year in (2022, 2024):
            reference_values = frame.loc[frame["season"].eq(reference_year), feature].to_numpy(float)
            reference_finite = reference_values[np.isfinite(reference_values)]
            pooled_std = float(np.sqrt((np.var(target_finite) + np.var(reference_finite)) / 2.0)) if len(target_finite) and len(reference_finite) else 0.0
            smd = float((np.mean(target_finite) - np.mean(reference_finite)) / pooled_std) if pooled_std > 0 else 0.0
            feature_rows.append({
                "feature": feature.removeprefix("raw__"), "target_year": 2023, "reference_year": reference_year,
                "target_rows": int(len(target_values)), "reference_rows": int(len(reference_values)),
                "target_missing_rate": float(1.0 - len(target_finite) / len(target_values)), "reference_missing_rate": float(1.0 - len(reference_finite) / len(reference_values)),
                "target_mean": float(np.mean(target_finite)), "reference_mean": float(np.mean(reference_finite)),
                "target_median": float(np.median(target_finite)), "reference_median": float(np.median(reference_finite)),
                "target_q10": float(np.quantile(target_finite, .1)), "reference_q10": float(np.quantile(reference_finite, .1)),
                "target_q90": float(np.quantile(target_finite, .9)), "reference_q90": float(np.quantile(reference_finite, .9)),
                "standardized_mean_difference": smd,
            })
    feature_shift = pd.DataFrame(feature_rows).sort_values(["reference_year", "feature"]).reset_index(drop=True)
    feature_shift.to_csv(OUT / "feature_shift.csv", index=False)

    weights = np.asarray(manifest["main_weights"], float)
    stack_coefficients = np.asarray(manifest["stack_coefficients"], float)
    channel_specs = [
        ("v2", "v2_global", "v2_adjusted", stack_coefficients[0] * weights[0] / weights.sum()),
        ("v355", "v355_global", "v355_adjusted", stack_coefficients[0] * weights[1] / weights.sum()),
        ("v330", "v330_global", "v330_adjusted", stack_coefficients[0] * weights[2] / weights.sum()),
        ("risk_middle", "risk_middle_global", "risk_middle_adjusted", stack_coefficients[1]),
        ("risk_wild", "risk_wild_global", "risk_wild_adjusted", stack_coefficients[2]),
        ("risk_reverse", "risk_reverse_global", "risk_reverse_adjusted", stack_coefficients[3]),
    ]
    channel_rows: list[dict[str, object]] = []
    for year in YEARS:
        part = frame.loc[frame["season"].eq(year)]
        y = part["target"].to_numpy(float)
        current_loss = (y - part["prediction_current"].to_numpy(float)) ** 2
        global_loss = (y - part["prediction_global_only"].to_numpy(float)) ** 2
        for name, global_column, adjusted_column, coefficient in channel_specs:
            delta = part[adjusted_column].to_numpy(float) - part[global_column].to_numpy(float)
            contribution = coefficient * delta
            channel_rows.append({
                "season": year, "channel": name, "rows": int(len(part)), "stack_multiplier": float(coefficient),
                "global_mean": float(part[global_column].mean()), "adjusted_mean": float(part[adjusted_column].mean()),
                "raw_delta_mean": float(delta.mean()), "raw_delta_abs_mean": float(np.abs(delta).mean()),
                "prediction_contribution_mean": float(contribution.mean()), "prediction_contribution_abs_mean": float(np.abs(contribution).mean()),
                "correlation_contribution_target": safe_corr(contribution, y),
                "correlation_contribution_current_loss": safe_corr(contribution, current_loss),
                "f_expert_brier_gain_vs_global_only": float(global_loss.mean() - current_loss.mean()),
            })
    channel_summary = pd.DataFrame(channel_rows).sort_values(["season", "channel"]).reset_index(drop=True)
    channel_summary.to_csv(OUT / "channel_summary.csv", index=False)

    overall: dict[str, object] = {}
    for year in YEARS:
        part = frame.loc[frame["season"].eq(year)]
        y = part["target"].to_numpy(float)
        variants = {name: metric(y, part[f"prediction_{name}"].to_numpy(float)) for name in ("current", "no_shift", "global_only", "f075")}
        overall[str(year)] = {
            "metrics": variants,
            "shift_removal_brier_gain": float(variants["current"]["brier"] - variants["no_shift"]["brier"]),
            "global_only_brier_gain": float(variants["current"]["brier"] - variants["global_only"]["brier"]),
            "f075_brier_gain": float(variants["current"]["brier"] - variants["f075"]["brier"]),
        }
    f_history = historical.loc[historical["game_type"].astype(str).eq("F")]
    top_features = feature_shift.assign(abs_smd=feature_shift["standardized_mean_difference"].abs()).sort_values("abs_smd", ascending=False).head(10)
    top_slices = slices.loc[slices["season"].eq(2023) & slices["rows"].ge(500)].sort_values("current_excess_loss_sum", ascending=False).head(10)
    findings = {
        "historical_f_target_rate": {str(int(row.season)): {"rows": int(row.rows), "target_rate": float(row.target_rate)} for row in f_history.itertuples(index=False)},
        "f_2023_target_drop_vs_2022": float(overall["2023"]["metrics"]["current"]["target_rate"] - overall["2022"]["metrics"]["current"]["target_rate"]),  # type: ignore[index]
        "f_2023_target_difference_vs_2024": float(overall["2023"]["metrics"]["current"]["target_rate"] - overall["2024"]["metrics"]["current"]["target_rate"]),  # type: ignore[index]
        "f_2023_current_bias": float(overall["2023"]["metrics"]["current"]["mean_bias"]),  # type: ignore[index]
        "f_2023_shift_removal_gain": float(overall["2023"]["shift_removal_brier_gain"]),
        "f_2023_global_only_gain": float(overall["2023"]["global_only_brier_gain"]),
        "f_2023_f075_gain": float(overall["2023"]["f075_brier_gain"]),
        "top_feature_shifts": top_features[["feature", "reference_year", "standardized_mean_difference"]].to_dict(orient="records"),
        "top_2023_loss_slices": top_slices[["dimension", "value", "rows", "current_excess_loss_sum", "current_mean_bias"]].to_dict(orient="records"),
        "diagnostic_status": "DESCRIPTIVE_NOT_CAUSAL", "automatic_candidate_status": "NO_AUTOMATIC_CANDIDATE",
    }
    output_counts = {
        "f_diagnostic_rows": int(len(frame)), "historical_rate_rows": int(len(historical)),
        "slice_rows": int(len(slices)), "cohort_rows": int(len(cohort)),
        "feature_shift_rows": int(len(feature_shift)), "channel_summary_rows": int(len(channel_summary)),
    }
    result = {
        "experiment_id": EXPERIMENT_ID, "status": "PENDING_AUDIT", "diagnostic_status": "COMPLETE",
        "source_official_score": 1068.25021, "overall": overall, "findings": findings,
        "output_counts": output_counts, "candidate_count": 0, "actual_leaf_count": 0,
        "gate_checks_count": 0, "model_count": 0, "test_inference_performed": False,
        "full_train_performed": False, "zip_created": False, "elapsed_seconds": time.time() - started,
    }
    write_json(OUT / "result.json", result)
    embedded_value = {key: result[key] for key in result if key != "elapsed_seconds"}
    embedded = json.dumps(embedded_value, sort_keys=True)
    lines = [
        f"# {EXPERIMENT_ID}", "", "- status: `PENDING_AUDIT`", "- diagnostic: `COMPLETE`",
        "- candidate/leaf/gate/model: `0/0/0/0`", f"- F diagnostic rows: `{output_counts['f_diagnostic_rows']}`",
        "- test/full-train/ZIP: `false/false/false`", "", "| year | F rows | target rate | current mean | current Brier | global-only gain |", "|---|---:|---:|---:|---:|---:|",
    ]
    for year in YEARS:
        item = overall[str(year)]
        lines.append(f"| {year} | {item['metrics']['current']['rows']} | {item['metrics']['current']['target_rate']:.12g} | {item['metrics']['current']['prediction_mean']:.12g} | {item['metrics']['current']['brier']:.12g} | {item['global_only_brier_gain']:+.12g} |")  # type: ignore[index]
    lines.extend(["", "<!-- RESULT_JSON_BEGIN", embedded, "RESULT_JSON_END -->"])
    (OUT / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    artifact_paths = [
        ROOT / "scripts" / "preflight_ref4_f_drift_diag_035a.py",
        ROOT / "scripts" / "run_ref4_f_drift_diag_035a.py",
        ROOT / "scripts" / "verify_ref4_f_drift_diag_035a.py",
        OUT / "diagnostic_contract.json", OUT / "preflight_report.json", OUT / "preflight_report.md",
        OUT / "f_diagnostic_rows.csv", OUT / "historical_rates.csv", OUT / "slice_metrics.csv",
        OUT / "cohort_metrics.csv", OUT / "feature_shift.csv", OUT / "channel_summary.csv",
        OUT / "result.json", OUT / "result.md", BASE / "audit_manifest.json", BASE / "audit_attestation.json",
        BASE / "oof_2023.csv", BASE / "oof_2024.csv", OOF22 / "audit_manifest.json",
        OOF22 / "audit_attestation.json", OOF22 / "oof_2022.csv", SOURCE / "manifest.json",
        SOURCE / "f_regime_meta.json", TRAIN, ROOT / "start03_reference.md", ROOT / "01_제약과금지사항.md",
        ROOT / "output" / "submit_ref4_champion_030.zip",
    ]
    records = {str(path.relative_to(ROOT)): {"sha256": sha256_path(path), "size": path.stat().st_size} for path in artifact_paths}
    audit = {"experiment_id": EXPERIMENT_ID, "status": "PENDING_VALIDATION", "artifact_count": len(records), "artifacts": records, "candidate_count": 0, "leaf_count": 0, "gate_checks_count": 0, "model_count": 0, "oof_rows": preflight["oof_rows"], "f_diagnostic_rows": len(frame), "output_counts": output_counts}
    write_json(OUT / "audit_manifest.json", audit)
    print(json.dumps({"experiment_id": EXPERIMENT_ID, "status": "PENDING_AUDIT", "candidate_count": 0, "gate_checks_count": 0, "model_count": 0, "oof_rows": preflight["oof_rows"], **output_counts, "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
