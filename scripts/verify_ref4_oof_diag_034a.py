#!/usr/bin/env python3
"""Independent source-level validator for REF4-OOF-DIAG-034A."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "REF4-OOF-DIAG-034A"
OUT = ROOT / "model" / EXPERIMENT_ID
BASE = ROOT / "model" / "REF4-EXACT-OOF-031A"
OOF22 = ROOT / "model" / "REF4-ADAPTIVE-GATE-031B"
SOURCE = ROOT / "model" / "REF4-CHAMPION-STACK-030"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def metric(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | int | None]:
    rate = float(y.mean())
    brier = float(np.mean((y - prediction) ** 2))
    denominator = rate * (1.0 - rate)
    bss = float(1.0 - brier / denominator) if denominator > 0 else None
    return {
        "rows": int(len(y)), "target_rate": rate, "prediction_mean": float(prediction.mean()),
        "mean_bias": float(prediction.mean() - rate), "brier": brier,
        "bss": bss, "local_score": 100000.0 * bss if bss is not None else None,
    }


def independent_prior(raw: pd.DataFrame, year: int) -> pd.Series:
    counts = raw.loc[raw["season"].lt(year)].groupby(
        ["pitcher_id", "season", "game_type"], observed=True
    ).size().rename("n").reset_index()
    dominant = counts.sort_values("n").groupby(["pitcher_id", "season"], observed=True).tail(1)
    latest = dominant.sort_values("season").groupby("pitcher_id", observed=True).tail(1)
    return latest.set_index(latest["pitcher_id"].astype(str))["game_type"].astype(str)


def independent_recombine(oof: pd.DataFrame, regime: dict[str, float], manifest: dict[str, object]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    is_f = oof["game_type"].eq("F").to_numpy()
    global_channels = [
        oof["p_v2_global"].to_numpy(float), oof["p_v3_55_global"].to_numpy(float),
        oof["p_v3_30_global"].to_numpy(float),
    ]
    f_v2 = oof["p_v2_f"].to_numpy(float)
    f_v355 = oof["p_v3_55_f"].to_numpy(float)
    recent = global_channels[2] + regime["v330_recent_inner_scale"] * (
        oof["p_v3_30_f_recent"].to_numpy(float) - global_channels[2]
    )
    f_v330 = regime["v330_all_weight"] * oof["p_v3_30_f_all"].to_numpy(float) + (
        1.0 - regime["v330_all_weight"]
    ) * recent
    adjusted = [
        np.where(is_f, global_channels[0] + regime["v2_scale"] * (f_v2 - global_channels[0]), global_channels[0]),
        np.where(is_f, global_channels[1] + regime["v355_scale"] * (f_v355 - global_channels[1]), global_channels[1]),
        np.where(is_f, global_channels[2] + regime["v330_scale"] * (f_v330 - global_channels[2]), global_channels[2]),
    ]
    risks = []
    for name in ("middle", "wild", "reverse"):
        global_risk = oof[f"risk_{name}_global"].to_numpy(float)
        f_risk = oof[f"risk_{name}_f"].to_numpy(float)
        risks.append(np.where(is_f, global_risk + regime["subtype_scale"] * (f_risk - global_risk), global_risk))
    main = np.average(np.vstack(adjusted), axis=0, weights=np.asarray(manifest["main_weights"], float))
    no_shift = float(manifest["stack_intercept"]) + np.column_stack([main, *risks]) @ np.asarray(manifest["stack_coefficients"], float)
    final = np.clip(no_shift + float(manifest["global_shift"]), 1e-5, 1 - 1e-5)
    return final, {
        "p_v2_adjusted": adjusted[0], "p_v3_55_adjusted": adjusted[1],
        "p_v3_30_adjusted": adjusted[2], "risk_middle_adjusted": risks[0],
        "risk_wild_adjusted": risks[1], "risk_reverse_adjusted": risks[2],
        "prediction_no_shift_raw": no_shift,
    }


def compare_frames(actual: pd.DataFrame, expected: pd.DataFrame) -> dict[str, object]:
    failures: list[str] = []
    maximum = 0.0
    if list(actual.columns) != list(expected.columns):
        failures.append("columns")
        return {"failures": failures, "max_abs_diff": maximum}
    if len(actual) != len(expected):
        failures.append("rows")
        return {"failures": failures, "max_abs_diff": maximum}
    for column in actual.columns:
        if pd.api.types.is_numeric_dtype(expected[column]):
            left = pd.to_numeric(actual[column], errors="coerce").to_numpy(float)
            right = pd.to_numeric(expected[column], errors="coerce").to_numpy(float)
            finite_equal = np.array_equal(np.isfinite(left), np.isfinite(right))
            if not finite_equal:
                failures.append(f"{column}:finite")
                continue
            finite = np.isfinite(right)
            diff = float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else 0.0
            maximum = max(maximum, diff)
            if diff > 1e-10:
                failures.append(f"{column}:{diff}")
        else:
            if not np.array_equal(actual[column].astype(str).to_numpy(), expected[column].astype(str).to_numpy()):
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
            maximum = max(maximum, diff); failures.extend(child)
    elif isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            failures.append(f"{path}:length")
        else:
            for index, (left, right) in enumerate(zip(actual, expected)):
                diff, child = compare_objects(left, right, f"{path}[{index}]")
                maximum = max(maximum, diff); failures.extend(child)
    elif isinstance(actual, (int, float)) and not isinstance(actual, bool) and isinstance(expected, (int, float)) and not isinstance(expected, bool):
        diff = abs(float(actual) - float(expected)); maximum = max(maximum, diff)
        if diff > 1e-10:
            failures.append(f"{path}:{diff}")
    elif actual != expected:
        failures.append(f"{path}:{actual!r}!={expected!r}")
    return maximum, failures


def slice_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dimension in ("game_type", "known_status", "transition", "pitcher_n_bin", "game_month", "trackman_status"):
        for (year, value), part in frame.groupby(["season", dimension], observed=True, sort=True, dropna=False):
            y = part["target"].to_numpy(float)
            current = metric(y, part["prediction_current"].to_numpy(float))
            no_shift = metric(y, part["prediction_no_shift"].to_numpy(float))
            global_only = metric(y, part["prediction_global_only"].to_numpy(float))
            f075 = metric(y, part["prediction_f075"].to_numpy(float))
            rows.append({
                "dimension": dimension, "value": str(value), "season": int(year),
                "rows": int(len(part)), "target_rate": current["target_rate"],
                "current_prediction_mean": current["prediction_mean"],
                "current_brier": current["brier"], "current_bss": current["bss"],
                "no_shift_brier": no_shift["brier"],
                "shift_removal_brier_gain": float(current["brier"] - no_shift["brier"]),
                "global_only_brier": global_only["brier"],
                "global_only_brier_gain": float(current["brier"] - global_only["brier"]),
                "f075_brier": f075["brier"],
                "f075_brier_gain": float(current["brier"] - f075["brier"]),
            })
    return pd.DataFrame(rows).sort_values(["dimension", "season", "value"]).reset_index(drop=True)


def main() -> None:
    checks: list[dict[str, object]] = []
    mismatches: list[str] = []

    def check(name: str, passed: bool, actual: object) -> None:
        checks.append({"name": name, "checked": True, "passed": bool(passed), "actual": actual})
        if not passed:
            mismatches.append(name)

    required = [
        OUT / "audit_manifest.json", OUT / "result.json", OUT / "result.md",
        OUT / "diagnostic_rows.csv", OUT / "slice_metrics.csv",
        OUT / "calibration_bins.csv", OUT / "pitcher_clusters.csv",
    ]
    for path in required:
        check(f"exists:{path.name}", path.is_file(), path.is_file())
    audit = json.loads((OUT / "audit_manifest.json").read_text(encoding="utf-8"))
    execution = json.loads((OUT / "result.json").read_text(encoding="utf-8"))

    artifact_failures: list[str] = []
    for relative, record in audit["artifacts"].items():
        path = ROOT / relative
        if not path.is_file():
            artifact_failures.append(f"missing:{relative}"); continue
        if sha256_path(path) != record["sha256"]:
            artifact_failures.append(f"sha256:{relative}")
        if path.stat().st_size != record["size"]:
            artifact_failures.append(f"size:{relative}")
    check("manifest_artifact_hashes", not artifact_failures, artifact_failures)
    check("manifest_artifact_count", len(audit["artifacts"]) == audit["artifact_count"], {"actual": len(audit["artifacts"]), "recorded": audit["artifact_count"]})

    input_paths = {2022: OOF22 / "oof_2022.csv", 2023: BASE / "oof_2023.csv", 2024: BASE / "oof_2024.csv"}
    mapping_paths = {2022: OOF22 / "fold_2022" / "pitcher_trackman_mapping.csv", 2023: BASE / "fold_2023" / "pitcher_trackman_mapping.csv", 2024: BASE / "fold_2024" / "pitcher_trackman_mapping.csv"}
    for year, path in input_paths.items():
        directory = OOF22 if year == 2022 else BASE
        source_audit = json.loads((directory / "audit_manifest.json").read_text(encoding="utf-8"))
        key = str(path.relative_to(ROOT))
        check(f"source_oof_hash_{year}", sha256_path(path) == source_audit["artifacts"][key]["sha256"], {"actual": sha256_path(path), "recorded": source_audit["artifacts"][key]["sha256"]})

    raw = pd.read_csv(ROOT / "data" / "train.csv", low_memory=False)
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    current_regime = {key: float(value) for key, value in json.loads((SOURCE / "f_regime_meta.json").read_text(encoding="utf-8")).items()}
    global_regime = dict(current_regime); f075_regime = dict(current_regime)
    for key in ("v2_scale", "v355_scale", "v330_scale", "subtype_scale"):
        global_regime[key] = 0.0; f075_regime[key] *= 0.75

    parts = []
    component_summary: dict[str, object] = {}
    formula_differences: dict[str, float] = {}
    for year in (2022, 2023, 2024):
        oof = pd.read_csv(input_paths[year], dtype={"row_id": str, "game_type": str, "pitcher_id": str})
        raw_year = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        check(f"row_order_{year}", np.array_equal(oof["row_id"].astype(str).to_numpy(), raw_year["row_id"].astype(str).to_numpy()), {"oof": len(oof), "raw": len(raw_year)})
        check(f"target_{year}", np.array_equal(oof["target"].to_numpy(float), raw_year["control_success"].to_numpy(float)), int(np.sum(oof["target"].to_numpy(float) != raw_year["control_success"].to_numpy(float))))
        current, components = independent_recombine(oof, current_regime, manifest)
        global_only, _ = independent_recombine(oof, global_regime, manifest)
        f075, _ = independent_recombine(oof, f075_regime, manifest)
        formula_differences[str(year)] = float(np.max(np.abs(current - oof["prediction"].to_numpy(float))))
        prior = independent_prior(raw, year); pitcher = raw_year["pitcher_id"].astype(str)
        prior_type = pitcher.map(prior).fillna("NEW").astype(str)
        mapped = set(pd.read_csv(mapping_paths[year], dtype={"pitcher_id": str})["pitcher_id"].astype(str))
        pitcher_n = pd.to_numeric(raw_year["asof_pitcher_n"], errors="coerce").fillna(0)
        pitcher_n_bin = pd.cut(pitcher_n, [-1, 0, 9, 49, 199, np.inf], labels=["0", "1-9", "10-49", "50-199", "200+"]).astype(str)
        part = pd.DataFrame({
            "row_id": oof["row_id"].astype(str), "season": year,
            "game_type": raw_year["game_type"].astype(str), "pitcher_id": pitcher,
            "target": raw_year["control_success"].to_numpy(float),
            "game_month": pd.to_numeric(raw_year["game_month"], errors="coerce").fillna(-1).astype(int),
            "known_status": np.where(prior_type.eq("NEW"), "NEW", "KNOWN"),
            "prior_type": prior_type, "transition": prior_type + ">" + raw_year["game_type"].astype(str),
            "pitcher_n_bin": pitcher_n_bin, "trackman_status": np.where(pitcher.isin(mapped), "MAPPED", "UNMAPPED"),
            "prediction_current": current,
            "prediction_no_shift": np.clip(oof["prediction_no_shift"].to_numpy(float), 1e-5, 1 - 1e-5),
            "prediction_global_only": global_only, "prediction_f075": f075,
        })
        for name, values in components.items():
            part[name] = values
        parts.append(part)
        channels = np.column_stack([components["p_v2_adjusted"], components["p_v3_55_adjusted"], components["p_v3_30_adjusted"]])
        component_summary[str(year)] = {name: {"mean": float(values.mean()), "std": float(values.std())} for name, values in components.items()}
        component_summary[str(year)]["ensemble_disagreement"] = {
            "std_mean": float(channels.std(axis=1).mean()),
            "std_q95": float(np.quantile(channels.std(axis=1), .95)),
            "range_mean": float((channels.max(axis=1) - channels.min(axis=1)).mean()),
        }
    check("source_formula_current", max(formula_differences.values()) <= 1e-12, formula_differences)
    expected_rows = pd.concat(parts, ignore_index=True)
    stored_rows = pd.read_csv(OUT / "diagnostic_rows.csv", dtype={"row_id": str, "game_type": str, "pitcher_id": str, "known_status": str, "prior_type": str, "transition": str, "pitcher_n_bin": str, "trackman_status": str})
    row_comparison = compare_frames(stored_rows, expected_rows)
    check("diagnostic_rows_recomputed", not row_comparison["failures"], row_comparison)
    check("diagnostic_row_id_unique", stored_rows["row_id"].is_unique, int(stored_rows["row_id"].nunique()))
    prediction_columns = [column for column in stored_rows if column.startswith("prediction_") or column.startswith("p_") or column.startswith("risk_")]
    check("diagnostic_predictions_finite", all(np.isfinite(stored_rows[column].to_numpy(float)).all() for column in prediction_columns), [column for column in prediction_columns if not np.isfinite(stored_rows[column].to_numpy(float)).all()])

    expected_slices = slice_metrics(expected_rows)
    stored_slices = pd.read_csv(OUT / "slice_metrics.csv", dtype={"dimension": str, "value": str})
    slice_comparison = compare_frames(stored_slices, expected_slices)
    check("slice_metrics_recomputed", not slice_comparison["failures"], slice_comparison)

    calibration_rows = []
    for year, part in expected_rows.groupby("season", sort=True):
        bins = np.minimum((part["prediction_current"].to_numpy(float) * 10).astype(int), 9)
        temp = part.assign(calibration_bin=bins)
        for bin_id, group in temp.groupby("calibration_bin", sort=True):
            calibration_rows.append({"season": int(year), "bin": int(bin_id), "rows": int(len(group)), "target_rate": float(group["target"].mean()), "prediction_mean": float(group["prediction_current"].mean()), "mean_bias": float(group["prediction_current"].mean() - group["target"].mean()), "brier": float(np.mean((group["target"] - group["prediction_current"]) ** 2))})
    expected_calibration = pd.DataFrame(calibration_rows)
    stored_calibration = pd.read_csv(OUT / "calibration_bins.csv")
    calibration_comparison = compare_frames(stored_calibration, expected_calibration)
    check("calibration_recomputed", not calibration_comparison["failures"], calibration_comparison)

    cluster_frames = []
    cluster_summary: dict[str, object] = {}
    for year, part in expected_rows.groupby("season", sort=True):
        rate = float(part["target"].mean())
        temp = part.assign(squared_loss=(part["target"] - part["prediction_current"]) ** 2, reference_loss=(part["target"] - rate) ** 2)
        temp["excess_loss"] = temp["squared_loss"] - temp["reference_loss"]
        grouped = temp.groupby("pitcher_id", sort=True).agg(rows=("row_id", "size"), target_rate=("target", "mean"), prediction_mean=("prediction_current", "mean"), squared_loss=("squared_loss", "sum"), reference_loss=("reference_loss", "sum"), excess_loss=("excess_loss", "sum")).reset_index()
        grouped.insert(0, "season", int(year)); cluster_frames.append(grouped)
        positive = grouped.loc[grouped["excess_loss"].gt(0), "excess_loss"].sort_values(ascending=False); total = float(positive.sum())
        cluster_summary[str(year)] = {"clusters": int(len(grouped)), "positive_excess_clusters": int(len(positive)), "total_excess_loss": float(grouped["excess_loss"].sum()), "total_positive_excess_loss": total, "top_10_positive_share": float(positive.head(10).sum() / total) if total > 0 else 0.0, "top_20_positive_share": float(positive.head(20).sum() / total) if total > 0 else 0.0}
    expected_clusters = pd.concat(cluster_frames, ignore_index=True)
    stored_clusters = pd.read_csv(OUT / "pitcher_clusters.csv", dtype={"pitcher_id": str})
    cluster_comparison = compare_frames(stored_clusters, expected_clusters)
    check("pitcher_clusters_recomputed", not cluster_comparison["failures"], cluster_comparison)

    overall: dict[str, object] = {}
    for year in (2022, 2023, 2024):
        part = expected_rows.loc[expected_rows["season"].eq(year)]
        y = part["target"].to_numpy(float)
        metrics = {name: metric(y, part[column].to_numpy(float)) for name, column in (("current", "prediction_current"), ("no_shift", "prediction_no_shift"), ("global_only", "prediction_global_only"), ("f075", "prediction_f075"))}
        overall[str(year)] = {"metrics": metrics, "shift_removal_brier_gain": float(metrics["current"]["brier"] - metrics["no_shift"]["brier"]), "global_only_brier_gain": float(metrics["current"]["brier"] - metrics["global_only"]["brier"]), "f075_brier_gain": float(metrics["current"]["brier"] - metrics["f075"]["brier"]), "f_rows": int(part["game_type"].eq("F").sum())}
    f_rows = expected_slices.loc[(expected_slices["dimension"] == "game_type") & (expected_slices["value"] == "F")]
    f_slice = {str(int(row.season)): {"rows": int(row.rows), "current_brier": float(row.current_brier), "f075_brier": float(row.f075_brier), "f075_brier_gain": float(row.f075_brier_gain)} for row in f_rows.itertuples(index=False)}
    findings = {"shift_removal_positive_years": [year for year in ("2022", "2023", "2024") if overall[year]["shift_removal_brier_gain"] > 0], "global_only_positive_years": [year for year in ("2022", "2023", "2024") if overall[year]["global_only_brier_gain"] > 0], "f075_positive_years": [year for year in ("2022", "2023", "2024") if overall[year]["f075_brier_gain"] > 0], "f075_f_slice_positive_years": [year for year in ("2022", "2023", "2024") if f_slice[year]["f075_brier_gain"] > 0], "root_cause_status": "DIAGNOSTIC_ONLY_NOT_CAUSAL", "next_experiments": ["REF4-SHIFT-034B", "REF4-F-REGIME-032B"]}
    expected_sections = {"overall": overall, "f_slice": f_slice, "component_summary": component_summary, "cluster_summary": cluster_summary, "findings": findings}
    actual_sections = {key: execution[key] for key in expected_sections}
    maximum, result_failures = compare_objects(actual_sections, expected_sections)
    check("result_sections_recomputed", not result_failures, {"max_abs_diff": maximum, "failures": result_failures})
    check("count_contract", execution["candidate_count"] == execution["actual_leaf_count"] == execution["gate_checks_count"] == audit["candidate_count"] == audit["leaf_count"] == audit["gate_checks_count"] == 0, {"candidate": execution["candidate_count"], "leaf": execution["actual_leaf_count"], "gate": execution["gate_checks_count"]})
    check("output_counts", execution["oof_rows"] == audit["oof_rows"] == len(expected_rows) and execution["slice_rows"] == audit["slice_rows"] == len(expected_slices), {"oof": execution["oof_rows"], "slices": execution["slice_rows"]})
    check("no_models", execution["model_count"] == audit["model_count"] == 0 and not list(OUT.glob("**/*.cbm")), {"execution": execution["model_count"], "audit": audit["model_count"], "files": len(list(OUT.glob("**/*.cbm")))})
    check("no_training_test_zip", execution["training_performed"] is False and execution["test_inference_performed"] is False and execution["zip_created"] is False, {"training": execution["training_performed"], "test": execution["test_inference_performed"], "zip": execution["zip_created"]})

    markdown = (OUT / "result.md").read_text(encoding="utf-8")
    match = re.search(r"<!-- RESULT_JSON_BEGIN\n(.*?)\nRESULT_JSON_END -->", markdown, re.DOTALL)
    embedded = json.loads(match.group(1)) if match else None
    expected_embedded = {"candidate_count": 0, "actual_leaf_count": 0, "gate_checks_count": 0, "oof_rows": execution["oof_rows"], "slice_rows": execution["slice_rows"], "overall": execution["overall"], "f_slice": execution["f_slice"], "findings": execution["findings"]}
    check("json_markdown_embedded", embedded == expected_embedded, embedded == expected_embedded)

    status = "AUDIT_VERIFIED" if not mismatches else "FAIL"
    report = {
        "experiment_id": EXPERIMENT_ID, "status": status,
        "diagnostic_status": "COMPLETE" if status == "AUDIT_VERIFIED" else "FAIL",
        "checked_count": len(checks), "passed_count": sum(bool(item["passed"]) for item in checks),
        "mismatch_count": len(mismatches), "mismatches": mismatches,
        "actual_leaf_count": 0, "gate_checks_count": 0, "model_count": 0,
        "oof_rows": len(expected_rows), "slice_rows": len(expected_slices),
        **expected_sections, "checks": checks,
    }
    report_path = OUT / "validation_report.json"
    write_json(report_path, report)
    attestation = {
        "experiment_id": EXPERIMENT_ID, "status": status,
        "diagnostic_status": report["diagnostic_status"],
        "manifest_sha256": sha256_path(OUT / "audit_manifest.json"),
        "validation_report_sha256": sha256_path(report_path),
        "validator_sha256": sha256_path(Path(__file__).resolve()),
        "checked_count": report["checked_count"], "passed_count": report["passed_count"],
        "mismatch_count": report["mismatch_count"], "actual_leaf_count": 0,
        "gate_checks_count": 0, "model_count": 0,
        "oof_rows": report["oof_rows"], "slice_rows": report["slice_rows"],
    }
    write_json(OUT / "audit_attestation.json", attestation)
    print(json.dumps(attestation, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
