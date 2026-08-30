#!/usr/bin/env python3
"""Independent source-binding and prediction-parity verifier for 038A."""
from __future__ import annotations

import hashlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import catboost as cb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "REF4-CHAMPION-REPRO-038A"
OUT = ROOT / "model" / EXPERIMENT_ID
ORIGINAL = ROOT / "model" / "REF4-CHAMPION-STACK-030"
TRAIN = ROOT / "data" / "train.csv"
REF4 = ROOT / "github_reference" / "4번 레포"
TOLERANCE = 1e-12
PROBE_SIZE = 512

sys.path.insert(0, str(REF4 / "final" / "training"))
sys.path.insert(0, str(REF4 / "final" / "inference"))
sys.path.insert(0, str(REF4))

from src.preprocessing_v2 import build_v2_features, build_v3_features  # noqa: E402


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_sha256(values: np.ndarray) -> str:
    array = np.asarray(values, dtype="<f8")
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def evenly_spaced(indices: np.ndarray, count: int = PROBE_SIZE) -> np.ndarray:
    if len(indices) <= count:
        return indices.astype(np.int64, copy=True)
    positions = np.linspace(0, len(indices) - 1, count, dtype=np.int64)
    return indices[positions].astype(np.int64, copy=False)


def transition_frame() -> pd.DataFrame:
    frame = pd.DataFrame({
        "game_type": ["R", "F"], "prior_type": ["R", "F"],
        "transition": ["R>R", "R>F"], "count": ["0-0", "1-1"],
        "hand": ["Right-Right", "Left-Left"], "team_type": ["1|R", "2|F"],
        "base_prediction": [0.48, 0.52], "log_pitcher_n": [5.0, 5.0],
        "career": [0.48, 0.48], "recent1": [0.48, 0.48],
        "recent3": [0.48, 0.48], "recent5": [0.48, 0.48],
        "middle": [0.1, 0.1], "reverse": [0.1, 0.1], "li": [1.0, 1.0],
        "inning": [1.0, 1.0], "runners": [0.0, 0.0],
    })
    for column in ("game_type", "prior_type", "transition", "count", "hand", "team_type"):
        frame[column] = frame[column].astype(str)
    return frame


def predict(path: Path, model_kind: str, features: pd.DataFrame) -> tuple[Any, np.ndarray]:
    model = cb.CatBoostClassifier() if model_kind == "classifier" else cb.CatBoostRegressor()
    model.load_model(path)
    if model_kind == "classifier":
        values = model.predict_proba(features)[:, 1]
    else:
        values = model.predict(features)
    return model, np.asarray(values, dtype=np.float64)


def embedded_thread(model: Any) -> int:
    metadata = dict(model.get_metadata())
    params = json.loads(metadata["params"])
    return int(params.get("flat_params", {}).get("thread_count", params.get("system_options", {}).get("thread_count", -1)))


def add_check(checks: list[dict[str, Any]], check_id: str, passed: bool, detail: Any) -> None:
    checks.append({"check_id": check_id, "status": "PASS" if passed else "FAIL", "detail": detail})


def main() -> None:
    if not (OUT / "build_manifest.json").is_file():
        raise RuntimeError("Build manifest is missing; full-train is incomplete")
    prebuild = json.loads((OUT / "prebuild_manifest.json").read_text())
    build = json.loads((OUT / "build_manifest.json").read_text())
    checks: list[dict[str, Any]] = []

    add_check(checks, "experiment_id", prebuild.get("experiment_id") == EXPERIMENT_ID == build.get("experiment_id"), {
        "prebuild": prebuild.get("experiment_id"), "build": build.get("experiment_id")
    })
    add_check(checks, "prebuild_manifest_binding", sha256_file(OUT / "prebuild_manifest.json") == build.get("prebuild_manifest_sha256"), build.get("prebuild_manifest_sha256"))
    add_check(checks, "declared_model_count", prebuild.get("model_count") == 56 and build.get("model_count") == 56, {
        "prebuild": prebuild.get("model_count"), "build": build.get("model_count")
    })

    source_records = prebuild.get("source_inputs", []) + prebuild.get("source_code", [])
    source_mismatches = []
    for record in source_records:
        path = ROOT / record["path"]
        actual = sha256_file(path) if path.is_file() else None
        if actual != record["sha256"] or (path.is_file() and path.stat().st_size != record["size"]):
            source_mismatches.append({"path": record["path"], "expected": record["sha256"], "actual": actual})
    add_check(checks, "source_binding", not source_mismatches, source_mismatches)

    original_mismatches = []
    for record in prebuild.get("original_model_inventory", []):
        path = ROOT / record["path"]
        actual = sha256_file(path) if path.is_file() else None
        if actual != record["sha256"]:
            original_mismatches.append({"path": record["path"], "expected": record["sha256"], "actual": actual})
    add_check(checks, "original_champion_unchanged", not original_mismatches and len(prebuild.get("original_model_inventory", [])) == 56, original_mismatches)

    contract = prebuild.get("model_contract", [])
    expected_names = {row["file"] for row in contract}
    actual_paths = sorted(OUT.glob("*.cbm"))
    add_check(checks, "generated_model_set", len(actual_paths) == 56 and {path.name for path in actual_paths} == expected_names, {
        "actual_count": len(actual_paths), "missing": sorted(expected_names - {p.name for p in actual_paths}),
        "extra": sorted({p.name for p in actual_paths} - expected_names)
    })

    manifest_models = {row["file"]: row for row in build.get("models", [])}
    model_hash_mismatches = []
    for path in actual_paths:
        record = manifest_models.get(path.name)
        actual = sha256_file(path)
        if record is None or record.get("sha256") != actual or record.get("size") != path.stat().st_size:
            model_hash_mismatches.append({"file": path.name, "actual": actual, "record": record})
    add_check(checks, "generated_model_hashes", not model_hash_mismatches and len(manifest_models) == 56, model_hash_mismatches)

    add_check(checks, "build_parameter_mismatches", build.get("parameter_mismatch_count") == 0, build.get("parameter_mismatch_count"))
    derived = build.get("derived_artifact_comparison", {})
    derived_failures = [name for name, record in derived.items() if not record.get("content_equal")]
    add_check(checks, "derived_artifact_content", not derived_failures and len(derived) == 8, derived_failures)
    forbidden = build.get("forbidden_outputs_created", {})
    candidate_exists = (ROOT / "candidate" / EXPERIMENT_ID).exists()
    zip_matches = sorted(str(path.relative_to(ROOT)) for path in (ROOT / "output").glob("*038A*.zip"))
    add_check(checks, "no_candidate", not candidate_exists and not forbidden.get("candidate_directory"), candidate_exists)
    add_check(checks, "no_zip", not zip_matches and not forbidden.get("zip_matches"), zip_matches)
    add_check(checks, "no_test_or_package_path", build.get("test_access", "").startswith("not performed"), build.get("test_access"))

    raw = pd.read_csv(TRAIN, low_memory=False)
    prior = float(raw.control_success.mean())
    all_indices = evenly_spaced(np.arange(len(raw), dtype=np.int64))
    f_indices = evenly_spaced(np.flatnonzero(raw.game_type.eq("F").to_numpy()))
    f2024_indices = evenly_spaced(np.flatnonzero((raw.game_type.eq("F") & raw.season.eq(2024)).to_numpy()))
    union_indices = np.unique(np.concatenate([all_indices, f_indices, f2024_indices]))
    raw_probe = raw.iloc[union_indices].reset_index(drop=True)
    original_position = {int(value): index for index, value in enumerate(union_indices)}
    role_positions = {
        "global": np.asarray([original_position[int(value)] for value in all_indices], dtype=np.int64),
        "F": np.asarray([original_position[int(value)] for value in f_indices], dtype=np.int64),
        "F2024": np.asarray([original_position[int(value)] for value in f2024_indices], dtype=np.int64),
    }
    pitcher = pd.read_pickle(OUT / "pitcher_snapshots.pkl")
    batter = pd.read_pickle(OUT / "batter_snapshots.pkl")
    pitchmix = pd.read_pickle(OUT / "pitchmix_snapshots.pkl")
    x2, _ = build_v2_features(raw_probe, prior, pitcher, str(OUT / "trackman_prior_features.csv"))
    x3, _ = build_v3_features(raw_probe, prior, pitcher, batter, pitchmix, str(OUT / "trackman_prior_features.csv"))
    add_check(checks, "probe_feature_shapes", x2.shape[1] == 181 and x3.shape[1] == 196, {"x2": list(x2.shape), "x3": list(x3.shape)})

    parity_rows: list[dict[str, Any]] = []
    parameter_failures = []
    feature_failures = []
    for spec in contract:
        name = spec["file"]
        if name == "transition_gate.cbm":
            role = "synthetic"
            features = transition_frame()
        else:
            if spec["scope"] == "train_2024_F":
                role = "F2024"
            elif "_F_" in spec["scope"]:
                role = "F"
            else:
                role = "global"
            matrix = x2 if spec["family"] in ("v2_decay55", "f_v2_all") else x3
            features = matrix.iloc[role_positions[role]]
        original_model, original_pred = predict(ORIGINAL / name, spec["model_kind"], features)
        repro_model, repro_pred = predict(OUT / name, spec["model_kind"], features)
        max_abs = float(np.max(np.abs(original_pred - repro_pred)))
        same_features = original_model.feature_names_ == repro_model.feature_names_
        if not same_features:
            feature_failures.append(name)
        params = repro_model.get_all_params()
        actual_thread = embedded_thread(repro_model)
        actual_seed = int(params.get("random_seed", -1))
        if actual_thread != int(spec["thread_count"]) or actual_seed != int(spec["random_seed"]):
            parameter_failures.append(name)
        parity_rows.append({
            "file": name,
            "family": spec["family"],
            "scope": spec["scope"],
            "probe_role": role,
            "probe_rows": len(features),
            "expected_thread_count": spec["thread_count"],
            "actual_thread_count": actual_thread,
            "expected_random_seed": spec["random_seed"],
            "actual_random_seed": actual_seed,
            "feature_names_equal": same_features,
            "original_prediction_sha256": prediction_sha256(original_pred),
            "repro_prediction_sha256": prediction_sha256(repro_pred),
            "max_abs_difference": max_abs,
            "within_tolerance": max_abs <= TOLERANCE,
        })
    parity = pd.DataFrame(parity_rows).sort_values("file").reset_index(drop=True)
    parity.to_csv(OUT / "prediction_parity.csv", index=False)
    add_check(checks, "embedded_thread_and_seed", not parameter_failures, parameter_failures)
    add_check(checks, "feature_name_parity", not feature_failures, feature_failures)
    failed_parity = parity.loc[~parity.within_tolerance, ["file", "max_abs_difference"]].to_dict("records")
    add_check(checks, "prediction_parity_56", not failed_parity and len(parity) == 56, failed_parity)

    failed_checks = [row["check_id"] for row in checks if row["status"] != "PASS"]
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": now_utc(),
        "validator": str(Path(__file__).relative_to(ROOT)),
        "validator_sha256": sha256_file(Path(__file__)),
        "tolerance": TOLERANCE,
        "probe_size_per_role": PROBE_SIZE,
        "probe_source": "deterministic evenly-spaced row positions from source-bound train.csv",
        "candidate_count": 0,
        "actual_leaf_count": 0,
        "gate_count": 0,
        "test_read_performed": False,
        "test_inference_performed": False,
        "zip_created": False,
        "check_count": len(checks),
        "pass_count": len(checks) - len(failed_checks),
        "fail_count": len(failed_checks),
        "mismatch_count": len(failed_checks),
        "failed_checks": failed_checks,
        "max_abs_prediction_difference": float(parity.max_abs_difference.max()),
        "prediction_models_within_tolerance": int(parity.within_tolerance.sum()),
        "status": "AUDIT_VERIFIED" if not failed_checks else "AUDIT_FAIL_REPRODUCTION",
        "checks": checks,
        "artifacts": {
            "prebuild_manifest_sha256": sha256_file(OUT / "prebuild_manifest.json"),
            "build_manifest_sha256": sha256_file(OUT / "build_manifest.json"),
            "prediction_parity_sha256": sha256_file(OUT / "prediction_parity.csv"),
        },
    }
    (OUT / "validation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    attestation = {
        "experiment_id": EXPERIMENT_ID,
        "validation_report_sha256": sha256_file(OUT / "validation_report.json"),
        "validator_sha256": report["validator_sha256"],
        "status": report["status"],
        "check_count": report["check_count"],
        "mismatch_count": report["mismatch_count"],
        "signed_by": "independent deterministic verifier",
        "created_at_utc": now_utc(),
    }
    (OUT / "audit_attestation.json").write_text(json.dumps(attestation, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({
        "status": report["status"], "checks": report["check_count"],
        "passed": report["pass_count"], "mismatches": report["mismatch_count"],
        "models_within_tolerance": report["prediction_models_within_tolerance"],
        "max_abs_prediction_difference": report["max_abs_prediction_difference"],
    }, indent=2))
    if failed_checks:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
