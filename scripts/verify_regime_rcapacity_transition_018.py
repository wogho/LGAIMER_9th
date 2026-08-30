#!/usr/bin/env python3
"""Independently verify hashes, rows, metrics, and gates for transition 018."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric(pred: np.ndarray, target: np.ndarray) -> float:
    return float(1e5 * np.corrcoef(pred, target)[0, 1] ** 2)


def brier(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def close(left: float, right: float, atol: float = 1e-11) -> bool:
    return bool(np.isclose(left, right, rtol=0.0, atol=atol))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    return parser.parse_args()


def resolve(entry: dict[str, object]) -> Path:
    return ROOT / str(entry["path"])


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    hash_checks: list[dict[str, object]] = []
    entries = list(manifest["common_files"])
    for fold in manifest["fold_files"].values():
        entries.extend(fold.values())
    for entry in entries:
        path = resolve(entry)
        actual_sha = sha256_file(path) if path.is_file() else None
        actual_size = path.stat().st_size if path.is_file() else None
        passed = actual_sha == entry["sha256"] and actual_size == entry["size"]
        hash_checks.append(
            {
                "path": entry["path"],
                "expected_sha256": entry["sha256"],
                "actual_sha256": actual_sha,
                "expected_size": entry["size"],
                "actual_size": actual_size,
                "pass": passed,
            }
        )
        if not passed:
            failures.append(f"hash_or_size:{entry['path']}")

    common_entries = {entry["path"]: entry for entry in manifest["common_files"]}
    preserved_zip_checks: list[dict[str, object]] = []
    for relative_path, expected_sha in manifest["preserved_zip_expected_sha256"].items():
        recorded_sha = common_entries[relative_path]["sha256"]
        passed = recorded_sha == expected_sha
        preserved_zip_checks.append(
            {
                "path": relative_path,
                "expected_sha256": expected_sha,
                "manifest_sha256": recorded_sha,
                "pass": passed,
            }
        )
        if not passed:
            failures.append(f"preserved_zip_hash:{relative_path}")

    actual_environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    environment_match = actual_environment == manifest["environment"]
    if not environment_match:
        failures.append("environment_provenance")

    weights = manifest["fixed_blend_weights"]
    weight_sum = float(weights["baseline"] + weights["split_rcapacity"])
    if not close(weight_sum, 1.0, 1e-15):
        failures.append("blend_weight_sum")
    threshold = float(manifest["required_relative_improvement"])
    if not 0.0 <= threshold <= 1.0:
        failures.append("gate_threshold_range")

    train = pd.read_csv(
        ROOT / "data" / "train.csv",
        encoding="utf-8-sig",
        usecols=["row_id", "season", "control_success"],
    )
    baseline = pd.read_csv(
        ROOT / "model" / "COMBO-RESID3-OOF-007" / "oof_predictions.csv",
        usecols=["row_id", "season", "target", "pred"],
    )
    trackman = pd.read_csv(ROOT / "data" / "trackman_history.csv")
    pitcher_map = pd.read_csv(
        ROOT / "model" / "TRACKMAN-MAP-004" / "pitcher_id_map.csv"
    )
    data_integrity = {
        "train_rows": int(len(train)),
        "train_unique_row_ids": int(train.row_id.nunique()),
        "train_row_id_unique": bool(not train.row_id.duplicated().any()),
        "train_target_finite": bool(np.isfinite(train.control_success.to_numpy()).all()),
        "train_target_values": sorted(train.control_success.unique().tolist()),
        "baseline_rows": int(len(baseline)),
        "baseline_unique_row_ids": int(baseline.row_id.nunique()),
        "baseline_row_id_unique": bool(not baseline.row_id.duplicated().any()),
        "baseline_target_finite": bool(np.isfinite(baseline.target.to_numpy()).all()),
        "baseline_target_values": sorted(baseline.target.unique().tolist()),
        "baseline_prediction_finite": bool(np.isfinite(baseline.pred.to_numpy()).all()),
        "baseline_prediction_range": [
            float(baseline.pred.min()),
            float(baseline.pred.max()),
        ],
        "trackman_rows": int(len(trackman)),
        "trackman_column_count": int(len(trackman.columns)),
        "trackman_row_id_present": bool("row_id" in trackman.columns),
        "trackman_row_id_check_applicable": False,
        "pitcher_map_rows": int(len(pitcher_map)),
        "pitcher_map_column_count": int(len(pitcher_map.columns)),
        "pitcher_map_row_id_present": bool("row_id" in pitcher_map.columns),
        "pitcher_map_row_id_check_applicable": False,
    }
    if not data_integrity["train_row_id_unique"]:
        failures.append("train_row_id_unique")
    if not data_integrity["baseline_row_id_unique"]:
        failures.append("baseline_row_id_unique")
    if not data_integrity["train_target_finite"] or data_integrity["train_target_values"] != [0, 1]:
        failures.append("train_target_binary_finite")
    if not data_integrity["baseline_target_finite"] or data_integrity["baseline_target_values"] != [0, 1]:
        failures.append("baseline_target_binary_finite")
    if not data_integrity["baseline_prediction_finite"] or not (
        0.0 <= data_integrity["baseline_prediction_range"][0]
        <= data_integrity["baseline_prediction_range"][1]
        <= 1.0
    ):
        failures.append("baseline_prediction_range")

    fold_checks: list[dict[str, object]] = []
    gate_checks: list[dict[str, object]] = []
    for season in manifest["seasons"]:
        season_key = str(season)
        fold_entries = manifest["fold_files"][season_key]
        report = json.loads(resolve(fold_entries["report"]).read_text(encoding="utf-8"))
        valid = train.loc[train.season.eq(season)].copy()
        base = baseline.loc[baseline.season.eq(season)].copy()
        if season == 2022:
            cand = pd.read_csv(resolve(fold_entries["predictions"]))
            cand = cand.rename(columns={"pred": "candidate_pred"})
            paired = valid.merge(
                base,
                on=["row_id", "season"],
                how="inner",
                validate="one_to_one",
            ).merge(
                cand,
                on="row_id",
                how="inner",
                suffixes=("_base", "_candidate"),
                validate="one_to_one",
            )
            target_match = bool(
                np.array_equal(paired.control_success, paired.target_base)
                and np.array_equal(paired.control_success, paired.target_candidate)
            )
            baseline_pred = paired.pred.to_numpy(float)
            candidate_pred = paired.candidate_pred.to_numpy(float)
        else:
            paired = pd.read_csv(resolve(fold_entries["predictions"]))
            source = valid[["row_id", "control_success"]].merge(
                base[["row_id", "target", "pred"]],
                on="row_id",
                how="inner",
                validate="one_to_one",
            ).merge(
                paired[["row_id", "target", "baseline_pred"]],
                on="row_id",
                how="inner",
                suffixes=("_source", "_paired"),
                validate="one_to_one",
            )
            target_match = bool(
                len(source) == len(valid)
                and np.array_equal(source.control_success, source.target_source)
                and np.array_equal(source.control_success, source.target_paired)
            )
            baseline_value_match = bool(
                np.array_equal(source.pred.to_numpy(), source.baseline_pred.to_numpy())
            )
            baseline_pred = paired.baseline_pred.to_numpy(float)
            candidate_pred = paired.candidate_pred.to_numpy(float)
        blend_pred = (
            float(weights["baseline"]) * baseline_pred
            + float(weights["split_rcapacity"]) * candidate_pred
        )
        target = paired.target.to_numpy(np.int8) if season != 2022 else paired.control_success.to_numpy(np.int8)
        expected_ids = set(valid.row_id)
        actual_ids = set(paired.row_id)
        base_ids = set(base.row_id)
        row_set_match = expected_ids == actual_ids == base_ids
        candidate_finite = bool(np.isfinite(candidate_pred).all())
        blend_finite = bool(np.isfinite(blend_pred).all())
        candidate_range = [float(candidate_pred.min()), float(candidate_pred.max())]
        blend_range = [float(blend_pred.min()), float(blend_pred.max())]
        baseline_metric = metric(baseline_pred, target)
        candidate_metric = metric(candidate_pred, target)
        blend_metric = metric(blend_pred, target)
        baseline_brier = brier(baseline_pred, target)
        candidate_brier = brier(candidate_pred, target)
        blend_brier = brier(blend_pred, target)
        relative_improvement = blend_metric / baseline_metric - 1.0
        gate_pass = bool(relative_improvement >= threshold)
        report_checks: list[bool]
        if season == 2022:
            baseline_value_match = True
            report_checks = [
                close(candidate_metric, float(report["overall_metric"])),
                close(candidate_brier, float(report["overall_brier"])),
            ]
        else:
            report_checks = [
                close(baseline_metric, float(report["metrics"]["baseline"]["metric"])),
                close(candidate_metric, float(report["metrics"]["split_rcapacity_raw"]["metric"])),
                close(blend_metric, float(report["metrics"]["baseline25_split75"]["metric"])),
                close(baseline_brier, float(report["metrics"]["baseline"]["brier"])),
                close(candidate_brier, float(report["metrics"]["split_rcapacity_raw"]["brier"])),
                close(blend_brier, float(report["metrics"]["baseline25_split75"]["brier"])),
                close(relative_improvement, float(report["metrics"]["baseline25_split75"]["relative_improvement"])),
                gate_pass == bool(report["metrics"]["baseline25_split75"]["gate_pass_2pct"]),
                close(float(report["fixed_blend_weights"]["sum"]), 1.0, 1e-15),
                report["predictions_sha256"] == fold_entries["predictions"]["sha256"],
                report["input_hashes"]["train_csv"] == common_entries["data/train.csv"]["sha256"],
                report["input_hashes"]["trackman_csv"] == common_entries["data/trackman_history.csv"]["sha256"],
                report["input_hashes"]["baseline_oof"] == common_entries["model/COMBO-RESID3-OOF-007/oof_predictions.csv"]["sha256"],
                report["input_hashes"]["pitcher_map"] == common_entries["model/TRACKMAN-MAP-004/pitcher_id_map.csv"]["sha256"],
            ]
        fold_pass = bool(
            len(paired) == len(valid)
            and paired.row_id.nunique() == len(valid)
            and row_set_match
            and target_match
            and baseline_value_match
            and candidate_finite
            and blend_finite
            and 0.0 <= candidate_range[0] <= candidate_range[1] <= 1.0
            and 0.0 <= blend_range[0] <= blend_range[1] <= 1.0
            and all(report_checks)
        )
        if not fold_pass:
            failures.append(f"fold_integrity_or_report:{season}")
        fold_checks.append(
            {
                "season": season,
                "expected_rows": int(len(valid)),
                "actual_rows": int(len(paired)),
                "unique_row_ids": int(paired.row_id.nunique()),
                "row_set_match": row_set_match,
                "target_match": target_match,
                "baseline_value_match": baseline_value_match,
                "candidate_prediction_finite": candidate_finite,
                "candidate_prediction_range": candidate_range,
                "blend_prediction_finite": blend_finite,
                "blend_prediction_range": blend_range,
                "baseline_metric": baseline_metric,
                "candidate_metric": candidate_metric,
                "blend_metric": blend_metric,
                "baseline_brier": baseline_brier,
                "candidate_brier": candidate_brier,
                "blend_brier": blend_brier,
                "report_checks_count": len(report_checks),
                "report_checks_passed": int(sum(report_checks)),
                "pass": fold_pass,
            }
        )
        gate_checks.append(
            {
                "leaf_candidate_id": "baseline25_split75",
                "season": season,
                "relative_improvement_full_precision": relative_improvement,
                "required_relative_improvement": threshold,
                "expected_gate": gate_pass,
                "pass": gate_pass,
            }
        )

    checked_file_count = len(hash_checks)
    hash_pass_count = sum(bool(item["pass"]) for item in hash_checks)
    fold_pass_count = sum(bool(item["pass"]) for item in fold_checks)
    gate_pass_count = sum(bool(item["pass"]) for item in gate_checks)
    audit_verified = bool(
        not failures
        and hash_pass_count == checked_file_count
        and fold_pass_count == len(fold_checks)
        and len(manifest["leaf_candidate_ids"]) == 1
        and len(gate_checks) == len(manifest["seasons"])
    )
    validation = {
        "audit_id": manifest["audit_id"],
        "manifest_sha256": sha256_file(manifest_path),
        "checked_file_count": checked_file_count,
        "hash_pass_count": hash_pass_count,
        "hash_mismatch_count": checked_file_count - hash_pass_count,
        "leaf_candidate_count": len(manifest["leaf_candidate_ids"]),
        "fold_checked_count": len(fold_checks),
        "fold_pass_count": fold_pass_count,
        "gate_checked_count": len(gate_checks),
        "gate_pass_count": gate_pass_count,
        "data_integrity": data_integrity,
        "environment_expected": manifest["environment"],
        "environment_actual": actual_environment,
        "environment_match": environment_match,
        "preserved_zip_checks": preserved_zip_checks,
        "hash_checks": hash_checks,
        "fold_checks": fold_checks,
        "gate_checks": gate_checks,
        "failures": failures,
        "unverified_items": [],
        "status": "AUDIT_VERIFIED" if audit_verified else "AUDIT_FAIL",
    }
    output_dir = manifest_path.parent
    stem = manifest_path.name.removesuffix("_manifest.json")
    validation_path = output_dir / f"{stem}_validation_report.json"
    if validation_path.exists():
        raise RuntimeError(f"refusing to overwrite validation report: {validation_path}")
    validation_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    attestation = {
        "audit_id": manifest["audit_id"],
        "manifest_sha256": sha256_file(manifest_path),
        "validation_report_sha256": sha256_file(validation_path),
        "validator_sha256": sha256_file(Path(__file__)),
        "checked_file_count": checked_file_count,
        "hash_pass_count": hash_pass_count,
        "hash_mismatch_count": checked_file_count - hash_pass_count,
        "leaf_candidate_count": len(manifest["leaf_candidate_ids"]),
        "fold_checked_count": len(fold_checks),
        "fold_pass_count": fold_pass_count,
        "gate_checked_count": len(gate_checks),
        "gate_pass_count": gate_pass_count,
        "failure_count": len(failures),
        "unverified_count": len(validation["unverified_items"]),
        "status": validation["status"],
    }
    attestation_path = output_dir / f"{stem}_attestation.json"
    if attestation_path.exists():
        raise RuntimeError(f"refusing to overwrite attestation: {attestation_path}")
    attestation_path.write_text(
        json.dumps(attestation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(attestation, ensure_ascii=False, indent=2))
    if not audit_verified:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
