#!/usr/bin/env python3
"""Independent source-level validator for REF4-EXACT-OOF-031A outputs."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "REF4-EXACT-OOF-031A"
OUT = ROOT / "model" / EXPERIMENT_ID
RAW_PATH = ROOT / "data" / "train.csv"
SOURCE_MODEL = ROOT / "model" / "REF4-CHAMPION-STACK-030"
VALID_YEARS = (2023, 2024)


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
    return {"rows": int(len(y)), "target_rate": rate, "brier": brier, "bss": bss, "local_score": 100000.0 * bss}


def main() -> None:
    checks: list[dict[str, object]] = []
    mismatches: list[str] = []

    def check(name: str, passed: bool, actual: object) -> None:
        checks.append({"name": name, "checked": True, "passed": bool(passed), "actual": actual})
        if not passed:
            mismatches.append(name)

    audit_manifest_path = OUT / "audit_manifest.json"
    execution_result_path = OUT / "result.json"
    execution_markdown_path = OUT / "result.md"
    oof_path = OUT / "oof_predictions.csv"
    for path in (audit_manifest_path, execution_result_path, execution_markdown_path, oof_path):
        check(f"exists:{path.name}", path.is_file(), path.is_file())

    audit_manifest = json.loads(audit_manifest_path.read_text(encoding="utf-8"))
    execution = json.loads(execution_result_path.read_text(encoding="utf-8"))
    source_manifest = json.loads((SOURCE_MODEL / "manifest.json").read_text(encoding="utf-8"))
    regime = {k: float(v) for k, v in json.loads((SOURCE_MODEL / "f_regime_meta.json").read_text(encoding="utf-8")).items()}

    artifact_failures: list[str] = []
    for relative, recorded in audit_manifest["artifacts"].items():
        path = ROOT / relative
        if not path.is_file():
            artifact_failures.append(f"missing:{relative}")
            continue
        actual_sha = sha256_path(path)
        actual_size = path.stat().st_size
        if actual_sha != recorded["sha256"]:
            artifact_failures.append(f"sha256:{relative}")
        if actual_size != recorded["size"]:
            artifact_failures.append(f"size:{relative}")
    check("manifest_artifact_hashes", not artifact_failures, artifact_failures)
    check("manifest_artifact_count", len(audit_manifest["artifacts"]) == audit_manifest["artifact_count"], {"actual": len(audit_manifest["artifacts"]), "recorded": audit_manifest["artifact_count"]})

    model_files = sorted(OUT.glob("fold_*/models/*.cbm"))
    check("model_count_filesystem", len(model_files) == 110, len(model_files))
    check("model_count_manifest", audit_manifest["model_count"] == len(model_files), audit_manifest["model_count"])
    for valid_year in VALID_YEARS:
        metadata = json.loads((OUT / f"fold_{valid_year}" / "fold_metadata.json").read_text(encoding="utf-8"))
        fold_model_count = len(list((OUT / f"fold_{valid_year}" / "models").glob("*.cbm")))
        check(f"fold_{valid_year}_train_before_valid", metadata["train_max_season"] < valid_year, metadata["train_max_season"])
        check(f"fold_{valid_year}_model_count", metadata["actual_model_count"] == fold_model_count == 55, {"metadata": metadata["actual_model_count"], "filesystem": fold_model_count})
        check(f"fold_{valid_year}_trackman_before_valid", metadata["trackman"]["mapping_source_max_season"] < valid_year and metadata["trackman"]["trackman_source_max_season"] < valid_year, metadata["trackman"])
        check(f"fold_{valid_year}_trackman_keys_unique", metadata["trackman"]["duplicate_keys"] == 0, metadata["trackman"]["duplicate_keys"])
        check(f"fold_{valid_year}_subtype_nonempty", metadata["subtype"]["recovered_rows"] > 0, metadata["subtype"])

    raw = pd.read_csv(RAW_PATH, usecols=["row_id", "season", "game_type", "pitcher_id", "control_success"], low_memory=False)
    expected = raw.loc[raw["season"].isin(VALID_YEARS)].reset_index(drop=True)
    oof = pd.read_csv(oof_path, dtype={"row_id": str, "game_type": str, "pitcher_id": str})
    check("raw_row_id_unique", raw["row_id"].is_unique, int(raw["row_id"].nunique()))
    check("oof_row_id_unique", oof["row_id"].is_unique, int(oof["row_id"].nunique()))
    check("oof_row_count", len(oof) == len(expected) == audit_manifest["oof_rows"], {"oof": len(oof), "expected": len(expected), "manifest": audit_manifest["oof_rows"]})
    expected_ids = expected["row_id"].astype(str).to_numpy()
    actual_ids = oof["row_id"].astype(str).to_numpy()
    check("oof_row_id_order", np.array_equal(actual_ids, expected_ids), int(np.sum(actual_ids != expected_ids)) if len(actual_ids) == len(expected_ids) else "length_mismatch")
    check("oof_row_id_set", set(actual_ids) == set(expected_ids), {"missing": len(set(expected_ids) - set(actual_ids)), "extra": len(set(actual_ids) - set(expected_ids))})
    check("oof_season_match", np.array_equal(oof["season"].to_numpy(int), expected["season"].to_numpy(int)), int(np.sum(oof["season"].to_numpy(int) != expected["season"].to_numpy(int))))
    check("oof_target_match", np.array_equal(oof["target"].to_numpy(float), expected["control_success"].to_numpy(float)), int(np.sum(oof["target"].to_numpy(float) != expected["control_success"].to_numpy(float))))
    check("oof_game_type_match", np.array_equal(oof["game_type"].to_numpy(str), expected["game_type"].astype(str).to_numpy()), int(np.sum(oof["game_type"].to_numpy(str) != expected["game_type"].astype(str).to_numpy())))
    check("oof_pitcher_match", np.array_equal(oof["pitcher_id"].to_numpy(str), expected["pitcher_id"].astype(str).to_numpy()), int(np.sum(oof["pitcher_id"].to_numpy(str) != expected["pitcher_id"].astype(str).to_numpy())))
    target = oof["target"].to_numpy(float)
    check("target_finite_binary", bool(np.isfinite(target).all() and np.isin(target, [0.0, 1.0]).all()), sorted(np.unique(target).tolist()))

    numeric_prediction_columns = [
        column for column in oof.columns
        if column.startswith("p_") or column.startswith("risk_") or column.startswith("prediction")
    ]
    finite_failures = [column for column in numeric_prediction_columns if not np.isfinite(oof[column].to_numpy(float)).all()]
    range_source_columns = [column for column in numeric_prediction_columns if column != "prediction_no_shift"]
    range_failures = [column for column in range_source_columns if not ((oof[column].to_numpy(float) >= 0.0) & (oof[column].to_numpy(float) <= 1.0)).all()]
    check("prediction_columns_finite", not finite_failures, finite_failures)
    check("prediction_columns_range", not range_failures, range_failures)

    futures = oof["game_type"].eq("F").to_numpy()
    p0g = oof["p_v2_global"].to_numpy(float); p0f = oof["p_v2_f"].to_numpy(float)
    p1g = oof["p_v3_55_global"].to_numpy(float); p1f = oof["p_v3_55_f"].to_numpy(float)
    p2g = oof["p_v3_30_global"].to_numpy(float)
    p2a = oof["p_v3_30_f_all"].to_numpy(float); p2r = oof["p_v3_30_f_recent"].to_numpy(float)
    p0 = np.where(futures, p0g + regime["v2_scale"] * (p0f - p0g), p0g)
    p1 = np.where(futures, p1g + regime["v355_scale"] * (p1f - p1g), p1g)
    recent_inner = p2g + regime["v330_recent_inner_scale"] * (p2r - p2g)
    f30 = regime["v330_all_weight"] * p2a + (1.0 - regime["v330_all_weight"]) * recent_inner
    p2 = np.where(futures, p2g + regime["v330_scale"] * (f30 - p2g), p2g)
    adjusted_risks = []
    for name in ("middle", "wild", "reverse"):
        global_risk = oof[f"risk_{name}_global"].to_numpy(float)
        f_risk = oof[f"risk_{name}_f"].to_numpy(float)
        adjusted_risks.append(np.where(futures, global_risk + regime["subtype_scale"] * (f_risk - global_risk), global_risk))
    weight_sum = float(sum(source_manifest["main_weights"]))
    check("main_weights_valid", abs(weight_sum - 1.0) <= 1e-7, {"sum": weight_sum, "abs_error": abs(weight_sum - 1.0)})
    check("transition_scale_zero", regime["transition_scale"] == 0.0, regime["transition_scale"])
    main_prediction = np.average(np.vstack([p0, p1, p2]), axis=0, weights=np.asarray(source_manifest["main_weights"], float))
    recomputed_no_shift = float(source_manifest["stack_intercept"]) + np.column_stack([main_prediction, *adjusted_risks]) @ np.asarray(source_manifest["stack_coefficients"], float)
    recomputed_final = np.clip(recomputed_no_shift + float(source_manifest["global_shift"]), 1e-5, 1 - 1e-5)
    no_shift_diff = float(np.max(np.abs(recomputed_no_shift - oof["prediction_no_shift"].to_numpy(float))))
    final_diff = float(np.max(np.abs(recomputed_final - oof["prediction"].to_numpy(float))))
    check("formula_no_shift", no_shift_diff <= 1e-12, no_shift_diff)
    check("formula_final", final_diff <= 1e-12, final_diff)

    recalculated: dict[str, object] = {}
    for split in ("2023", "2024"):
        mask = oof["season"].eq(int(split)).to_numpy()
        recalculated[split] = metric(target[mask], recomputed_final[mask])
    recalculated["pooled"] = metric(target, recomputed_final)
    recalculated["worst_season_bss"] = min(float(recalculated[split]["bss"]) for split in ("2023", "2024"))  # type: ignore[index]
    recalculated["worst_season_local_score"] = 100000.0 * float(recalculated["worst_season_bss"])
    metric_differences: dict[str, float] = {}
    for split in ("2023", "2024", "pooled"):
        for field in ("rows", "target_rate", "brier", "bss", "local_score"):
            metric_differences[f"{split}.{field}"] = abs(float(recalculated[split][field]) - float(execution["metrics"][split][field]))  # type: ignore[index]
    for field in ("worst_season_bss", "worst_season_local_score"):
        metric_differences[field] = abs(float(recalculated[field]) - float(execution["metrics"][field]))
    check("metrics_recomputed", max(metric_differences.values()) <= 1e-10, {"max_abs_diff": max(metric_differences.values()), "differences": metric_differences})

    markdown = execution_markdown_path.read_text(encoding="utf-8")
    match = re.search(r"<!-- RESULT_JSON_BEGIN\n(.*?)\nRESULT_JSON_END -->", markdown, flags=re.DOTALL)
    embedded = json.loads(match.group(1)) if match else None
    expected_embedded = {"candidate_name": execution["candidate_name"], "candidate_status": execution["candidate_status"], "metrics": execution["metrics"]}
    check("json_markdown_embedded_match", embedded == expected_embedded, embedded == expected_embedded)
    markdown_values_missing = []
    for split in ("2023", "2024", "pooled"):
        item = execution["metrics"][split]
        row = f"| {split} | {item['rows']} | {item['brier']:.12f} | {item['bss']:.12f} | {item['local_score']:.6f} |"
        if row not in markdown:
            markdown_values_missing.append(split)
    check("json_markdown_table_match", not markdown_values_missing, markdown_values_missing)
    check("leaf_count", execution["actual_leaf_count"] == audit_manifest["leaf_count"] == 1, {"execution": execution["actual_leaf_count"], "manifest": audit_manifest["leaf_count"]})
    check("no_test_or_zip", execution["test_inference_performed"] is False and execution["zip_created"] is False, {"test": execution["test_inference_performed"], "zip": execution["zip_created"]})

    status = "AUDIT_VERIFIED" if not mismatches else "FAIL"
    report = {
        "experiment_id": EXPERIMENT_ID, "status": status,
        "candidate_name": execution["candidate_name"],
        "candidate_status": "BASELINE_REGISTERED" if status == "AUDIT_VERIFIED" else "HOLD",
        "checked_count": len(checks), "passed_count": sum(bool(item["passed"]) for item in checks),
        "mismatch_count": len(mismatches), "mismatches": mismatches,
        "actual_leaf_count": 1, "gate_checks_count": 1,
        "registration_gate": status == "AUDIT_VERIFIED",
        "model_count": len(model_files), "oof_rows": len(oof),
        "metrics": recalculated, "formula_max_abs_diff": final_diff,
        "checks": checks,
    }
    report_path = OUT / "validation_report.json"
    write_json(report_path, report)
    validator_path = Path(__file__).resolve()
    attestation = {
        "experiment_id": EXPERIMENT_ID, "status": status,
        "manifest_sha256": sha256_path(audit_manifest_path),
        "validation_report_sha256": sha256_path(report_path),
        "validator_sha256": sha256_path(validator_path),
        "checked_count": report["checked_count"], "passed_count": report["passed_count"],
        "mismatch_count": report["mismatch_count"], "actual_leaf_count": 1,
        "gate_checks_count": 1, "model_count": len(model_files), "oof_rows": len(oof),
    }
    write_json(OUT / "audit_attestation.json", attestation)
    print(json.dumps(attestation, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
