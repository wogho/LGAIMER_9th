#!/usr/bin/env python3
"""Finalize 038A manifests after the non-mutating parameter-reader failure."""
from __future__ import annotations

import hashlib
import json
import pickle
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import catboost as cb
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "REF4-CHAMPION-REPRO-038A"
OUT = ROOT / "model" / EXPERIMENT_ID
ORIGINAL = ROOT / "model" / "REF4-CHAMPION-STACK-030"
DRIVER = ROOT / "scripts" / "run_ref4_champion_repro_038a.py"


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def record(path: Path) -> dict[str, Any]:
    return {"path": str(path.relative_to(ROOT)), "size": path.stat().st_size, "sha256": sha256_file(path)}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def compare_frame(original: Path, repro: Path) -> dict[str, Any]:
    if original.suffix == ".pkl":
        left, right = pd.read_pickle(original), pd.read_pickle(repro)
    else:
        left = pd.read_csv(original, low_memory=False)
        right = pd.read_csv(repro, low_memory=False)
    result: dict[str, Any] = {
        "rows_original": len(left), "rows_repro": len(right),
        "columns_original": list(left.columns), "columns_repro": list(right.columns),
    }
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=True)
        result["content_equal"] = True
    except AssertionError as exc:
        result["content_equal"] = False
        result["difference"] = str(exc)[:2000]
    return result


def embedded_thread(model: cb.CatBoost) -> int:
    metadata = dict(model.get_metadata())
    params = json.loads(metadata["params"])
    return int(params.get("flat_params", {}).get("thread_count", params.get("system_options", {}).get("thread_count", -1)))


def main() -> None:
    started = time.time()
    prebuild = json.loads((OUT / "prebuild_manifest.json").read_text())
    failed_status = json.loads((OUT / "execution_status.json").read_text())
    if failed_status.get("state") != "ERROR" or "actual_thread': -1" not in failed_status.get("error", ""):
        raise RuntimeError("Finalizer only accepts the known non-mutating thread-reader failure")
    driver_record = next(row for row in prebuild["source_code"] if row["path"] == str(DRIVER.relative_to(ROOT)))
    if sha256_file(DRIVER) != driver_record["sha256"]:
        raise RuntimeError("Training driver changed after prebuild binding")

    contract = prebuild["model_contract"]
    expected = {row["file"] for row in contract}
    models = sorted(OUT.glob("*.cbm"))
    if len(models) != 56 or {path.name for path in models} != expected:
        raise RuntimeError("The completed model set is not the fixed 56-model contract")

    contract_by_name = {row["file"]: row for row in contract}
    model_records: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for path in models:
        spec = contract_by_name[path.name]
        model = cb.CatBoost()
        model.load_model(path)
        params = model.get_all_params()
        thread = embedded_thread(model)
        seed = int(params.get("random_seed", -1))
        if thread != spec["thread_count"] or seed != spec["random_seed"]:
            mismatches.append({
                "file": path.name, "expected_thread": spec["thread_count"], "actual_thread": thread,
                "expected_seed": spec["random_seed"], "actual_seed": seed,
            })
        model_records.append({
            **record(path), **spec, "tree_count": int(model.tree_count_),
            "feature_count": len(model.feature_names_),
            "feature_names_sha256": sha256_bytes("\n".join(model.feature_names_).encode()),
            "embedded_thread_count": thread, "embedded_params": params,
            "train_finish_time": dict(model.get_metadata()).get("train_finish_time"),
        })
    if mismatches:
        raise RuntimeError(f"True embedded parameter mismatches: {mismatches}")

    for name in ("pitcher_snapshots.csv", "batter_snapshots.csv", "pitchmix_snapshots.csv", "prior_type.csv"):
        if not (OUT / name).is_file():
            raise RuntimeError(f"Expected post-train metadata file is missing: {name}")

    comparisons: dict[str, Any] = {}
    for name in (
        "pitcher_trackman_mapping.csv", "trackman_prior_features.csv",
        "pitcher_snapshots.pkl", "batter_snapshots.pkl", "pitchmix_snapshots.pkl",
    ):
        comparisons[name] = compare_frame(ORIGINAL / name, OUT / name)
    for name in ("f_regime_meta.json", "manifest.json"):
        comparisons[name] = {
            "content_equal": json.loads((ORIGINAL / name).read_text()) == json.loads((OUT / name).read_text()),
            "original": record(ORIGINAL / name), "repro": record(OUT / name),
        }
    with (ORIGINAL / "prior_type.pkl").open("rb") as handle:
        old_prior = pickle.load(handle)
    with (OUT / "prior_type.pkl").open("rb") as handle:
        new_prior = pickle.load(handle)
    comparisons["prior_type.pkl"] = {
        "content_equal": old_prior == new_prior,
        "entries_original": len(old_prior), "entries_repro": len(new_prior),
    }

    non_models = sorted(path for path in OUT.iterdir() if path.is_file() and path.suffix != ".cbm" and path.name != "build_manifest.json")
    build = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": now_utc(),
        "finalizer": record(Path(__file__).resolve()),
        "recovery_event": {
            "training_completed_before_error": True,
            "failed_postprocess_phase": "thread metadata read via get_all_params",
            "failure_was_model_mutating": False,
            "retraining_performed": False,
            "models_overwritten": False,
            "failed_status_snapshot": failed_status,
        },
        "prebuild_manifest_sha256": sha256_file(OUT / "prebuild_manifest.json"),
        "training_driver_sha256": sha256_file(DRIVER),
        "model_count": len(model_records),
        "parameter_mismatch_count": len(mismatches),
        "models": model_records,
        "derived_artifact_comparison": comparisons,
        "non_model_files": [record(path) for path in non_models],
        "forbidden_outputs_created": {
            "candidate_directory": (ROOT / "candidate" / EXPERIMENT_ID).exists(),
            "zip_matches": [str(path.relative_to(ROOT)) for path in (ROOT / "output").glob("*038A*.zip")],
        },
        "test_access": "not performed; derived trainer statically excludes test.csv and packaging section",
        "finalizer_elapsed_seconds": time.time() - started,
    }
    write_json(OUT / "build_manifest.json", build)
    write_json(OUT / "execution_status.json", {
        "experiment_id": EXPERIMENT_ID, "updated_at_utc": now_utc(),
        "phase": "post_train_finalization", "state": "PASS", "model_count": 56,
        "recovered_from_non_mutating_validator_error": True,
        "retraining_performed": False, "models_overwritten": False,
    })
    print(json.dumps({
        "status": "PASS", "models": len(model_records), "parameter_mismatches": len(mismatches),
        "derived_equal": {name: value["content_equal"] for name, value in comparisons.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
