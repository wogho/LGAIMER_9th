#!/usr/bin/env python3
"""Build REF4-CHAMPION-REPRO-038A without test inference or packaging.

The historical champion trainer is executed only through its training/metadata
boundary.  The ten v2 models use the thread count embedded in the audited
champion files; every other model retains the current source setting.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import catboost as cb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "REF4-CHAMPION-REPRO-038A"
OUT = ROOT / "model" / EXPERIMENT_ID
ORIGINAL_OUT = ROOT / "model" / "REF4-CHAMPION-STACK-030"
TRAINER = ROOT / "scripts" / "train_and_package_ref4_champion_030.py"
TRACKMAN_BUILDER = ROOT / "scripts" / "build_ref4_trackman_030.py"
TRAIN = ROOT / "data" / "train.csv"
TRACKMAN = ROOT / "data" / "trackman_history.csv"
REF4 = ROOT / "github_reference" / "4번 레포"
SEEDS = [260802, 260803, 260804, 260805, 260806, 260807]
PACKAGE_MARKER = "    # 7. Assembling candidate package"


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


def file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def model_contract() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(name: str, family: str, scope: str, threads: int, seed: int, kind: str) -> None:
        rows.append({
            "file": name,
            "family": family,
            "scope": scope,
            "thread_count": threads,
            "random_seed": seed,
            "model_kind": kind,
        })

    for seed in SEEDS:
        add(f"v2_decay55_seed{seed}.cbm", "v2_decay55", "train_2019_2024_all_decay55", 4, seed, "regressor")
    for j in range(4):
        add(f"f_v2_all_{j}.cbm", "f_v2_all", "train_2019_2024_F_decay55", 4, 968000 + j, "regressor")
    for seed in SEEDS:
        add(f"v3_decay55_seed{seed}.cbm", "v3_decay55", "train_2019_2024_all_decay55", 3, seed, "regressor")
    for seed in SEEDS:
        add(f"v3_decay30_seed{seed}.cbm", "v3_decay30", "train_2019_2024_all_decay30", 3, seed, "regressor")
    for j in range(6):
        add(f"f_v355_recent_{j}.cbm", "f_v355_recent", "train_2024_F", 3, 968100 + j, "regressor")
    for j in range(4):
        add(f"f_v330_all_{j}.cbm", "f_v330_all", "train_2019_2024_F_decay30", 3, 968200 + j, "regressor")
    for j in range(2):
        add(f"f_v330_recent_{j}.cbm", "f_v330_recent", "train_2024_F", 3, 968300 + j, "regressor")
    for index, family in enumerate(("middle", "wild", "reverse")):
        for seed in SEEDS:
            add(
                f"subtype_{family}_seed{seed}.cbm",
                f"subtype_{family}",
                "train_2019_2024_recovered_decay30",
                3,
                seed + index * 100,
                "classifier",
            )
    for index, family in enumerate(("middle", "wild", "reverse")):
        add(
            f"f_subtype_{family}.cbm",
            f"f_subtype_{family}",
            "train_2019_2024_F_recovered_decay30",
            3,
            968400 + index,
            "classifier",
        )
    add("transition_gate.cbm", "transition_gate", "fixed_synthetic_2_rows", 3, 0, "regressor")
    if len(rows) != 56 or len({row["file"] for row in rows}) != 56:
        raise RuntimeError("Internal contract must contain exactly 56 unique models")
    return rows


def transformed_trainer_source() -> str:
    source = TRAINER.read_text()
    if source.count(PACKAGE_MARKER) != 1:
        raise RuntimeError("Could not identify unique package boundary")
    source = source.split(PACKAGE_MARKER, 1)[0].rstrip() + "\n"
    old_id = 'EXPERIMENT_ID = "REF4-CHAMPION-STACK-030"'
    new_id = f'EXPERIMENT_ID = "{EXPERIMENT_ID}"'
    if source.count(old_id) != 1:
        raise RuntimeError("Unexpected experiment-id declaration")
    source = source.replace(old_id, new_id)

    forbidden_mkdirs = [
        "    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)\n",
        "    (CANDIDATE_DIR / \"model\").mkdir(parents=True, exist_ok=True)\n",
        "    (CANDIDATE_DIR / \"output\").mkdir(parents=True, exist_ok=True)\n",
        "    (CANDIDATE_DIR / \"src\").mkdir(parents=True, exist_ok=True)\n",
    ]
    for statement in forbidden_mkdirs:
        if source.count(statement) != 1:
            raise RuntimeError(f"Unexpected candidate mkdir statement: {statement.strip()}")
        source = source.replace(statement, "")

    seed_call = "random_seed=s, thread_count=THREAD_COUNT,"
    if source.count(seed_call) != 3:
        raise RuntimeError("Unexpected number of shared-seed training call sites")
    source = source.replace(seed_call, "random_seed=s, thread_count=4,", 1)
    f_v2_call = "random_seed=968000 + j, thread_count=THREAD_COUNT,"
    if source.count(f_v2_call) != 1:
        raise RuntimeError("Unexpected f_v2 thread call site")
    source = source.replace(f_v2_call, "random_seed=968000 + j, thread_count=4,", 1)

    forbidden_tokens = ["test.csv", "submission.csv", "ZipFile(", "Assembling candidate package"]
    for token in forbidden_tokens:
        if token in source:
            raise RuntimeError(f"Forbidden post-training token remains: {token}")
    if source.count("thread_count=4") != 2:
        raise RuntimeError("Derived trainer must contain exactly two v2 thread_count=4 call sites")
    return source


def transformed_trackman_source() -> str:
    source = TRACKMAN_BUILDER.read_text()
    old = 'OUT = ROOT / "model" / "REF4-CHAMPION-STACK-030"'
    new = f'OUT = ROOT / "model" / "{EXPERIMENT_ID}"'
    if source.count(old) != 1:
        raise RuntimeError("Unexpected TrackMan output declaration")
    return source.replace(old, new)


def reference_sources() -> list[Path]:
    paths = [TRAINER, TRACKMAN_BUILDER, Path(__file__).resolve(), REF4 / "requirements.txt"]
    paths.extend(sorted((REF4 / "final" / "training").rglob("*.py")))
    paths.extend(sorted((REF4 / "final" / "inference").rglob("*.py")))
    unique = {path.resolve(): path for path in paths if path.is_file()}
    return [unique[key] for key in sorted(unique, key=str)]


def update_status(phase: str, state: str, **extra: Any) -> None:
    payload = {"experiment_id": EXPERIMENT_ID, "updated_at_utc": now_utc(), "phase": phase, "state": state}
    payload.update(extra)
    write_json(OUT / "execution_status.json", payload)


def compare_frame(left_path: Path, right_path: Path) -> dict[str, Any]:
    if left_path.suffix == ".pkl":
        left = pd.read_pickle(left_path)
        right = pd.read_pickle(right_path)
    else:
        left = pd.read_csv(left_path, low_memory=False)
        right = pd.read_csv(right_path, low_memory=False)
    result: dict[str, Any] = {
        "rows_original": len(left),
        "rows_repro": len(right),
        "columns_original": list(left.columns) if hasattr(left, "columns") else [],
        "columns_repro": list(right.columns) if hasattr(right, "columns") else [],
    }
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=True)
        result["content_equal"] = True
    except AssertionError as exc:
        result["content_equal"] = False
        result["difference"] = str(exc)[:2000]
    return result


def main() -> None:
    started = time.time()
    contract = model_contract()
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Refusing to reuse non-empty reproduction directory: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)

    original_cbms = sorted(ORIGINAL_OUT.glob("*.cbm"))
    expected_names = {row["file"] for row in contract}
    if len(original_cbms) != 56 or {p.name for p in original_cbms} != expected_names:
        raise RuntimeError("Original champion model set does not match the fixed 56-model contract")

    trainer_source = transformed_trainer_source()
    trackman_source = transformed_trackman_source()
    environment = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "catboost": cb.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": importlib.metadata.version("scikit-learn"),
        "scipy": importlib.metadata.version("scipy"),
        "cpu_count": os.cpu_count(),
    }
    prebuild = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": now_utc(),
        "purpose": "provenance-only full-train reproduction; no test, candidate, zip, or submission",
        "source_inputs": [file_record(TRAIN), file_record(TRACKMAN)],
        "source_code": [file_record(path) for path in reference_sources()],
        "derived_training_source_sha256": sha256_bytes(trainer_source.encode()),
        "derived_trackman_source_sha256": sha256_bytes(trackman_source.encode()),
        "transformation_contract": {
            "training_boundary": PACKAGE_MARKER.strip(),
            "removed_operations": ["candidate assembly", "test inference", "submission generation", "ZIP packaging"],
            "thread_override": "only v2_decay55 and f_v2_all call sites use 4; all remaining call sites use 3",
        },
        "environment": environment,
        "model_count": len(contract),
        "model_contract": contract,
        "original_model_inventory": [file_record(path) for path in original_cbms],
    }
    write_json(OUT / "prebuild_manifest.json", prebuild)
    update_status("prebuild", "PASS", model_count=56)
    print(f"[{EXPERIMENT_ID}] PREBUILD PASS: 56-model contract and source binding written", flush=True)

    update_status("trackman", "RUNNING")
    trackman_ns = {"__name__": "ref4_repro_trackman", "__file__": str(TRACKMAN_BUILDER)}
    exec(compile(trackman_source, str(TRACKMAN_BUILDER), "exec"), trackman_ns)
    for name in ("pitcher_trackman_mapping.csv", "trackman_prior_features.csv"):
        if not (OUT / name).is_file():
            raise RuntimeError(f"TrackMan builder did not create {name}")
    update_status("trackman", "PASS")

    update_status("full_train", "RUNNING", expected_models=56)
    trainer_ns = {"__name__": "ref4_repro_trainer", "__file__": str(TRAINER)}
    exec(compile(trainer_source, str(TRAINER), "exec"), trainer_ns)
    trainer_ns["main"]()

    generated = sorted(OUT.glob("*.cbm"))
    if len(generated) != 56 or {p.name for p in generated} != expected_names:
        raise RuntimeError(f"Generated CBM set mismatch: got {len(generated)}, expected 56")

    for stem in ("pitcher_snapshots", "batter_snapshots", "pitchmix_snapshots"):
        frame = pd.read_pickle(OUT / f"{stem}.pkl")
        frame.to_csv(OUT / f"{stem}.csv", index=False)
    import pickle
    with (OUT / "prior_type.pkl").open("rb") as handle:
        prior_lookup = pickle.load(handle)
    pd.Series(prior_lookup, name="prior_type").rename_axis("pitcher_id").reset_index().to_csv(
        OUT / "prior_type.csv", index=False
    )

    contract_by_name = {row["file"]: row for row in contract}
    model_records: list[dict[str, Any]] = []
    parameter_mismatches: list[dict[str, Any]] = []
    for path in generated:
        spec = contract_by_name[path.name]
        model = cb.CatBoostClassifier() if spec["model_kind"] == "classifier" else cb.CatBoostRegressor()
        model.load_model(path)
        params = model.get_all_params()
        actual_thread = int(params.get("thread_count", -1))
        actual_seed = int(params.get("random_seed", -1))
        if actual_thread != spec["thread_count"] or actual_seed != spec["random_seed"]:
            parameter_mismatches.append({
                "file": path.name,
                "expected_thread": spec["thread_count"],
                "actual_thread": actual_thread,
                "expected_seed": spec["random_seed"],
                "actual_seed": actual_seed,
            })
        model_records.append({
            **file_record(path),
            **spec,
            "tree_count": model.tree_count_,
            "feature_count": len(model.feature_names_),
            "feature_names_sha256": sha256_bytes("\n".join(model.feature_names_).encode()),
            "embedded_params": params,
        })
    if parameter_mismatches:
        raise RuntimeError(f"Embedded parameter mismatches: {parameter_mismatches}")

    comparisons: dict[str, Any] = {}
    for name in (
        "pitcher_trackman_mapping.csv",
        "trackman_prior_features.csv",
        "pitcher_snapshots.pkl",
        "batter_snapshots.pkl",
        "pitchmix_snapshots.pkl",
    ):
        comparisons[name] = compare_frame(ORIGINAL_OUT / name, OUT / name)
    for name in ("f_regime_meta.json", "manifest.json"):
        comparisons[name] = {
            "content_equal": json.loads((ORIGINAL_OUT / name).read_text()) == json.loads((OUT / name).read_text()),
            "original": file_record(ORIGINAL_OUT / name),
            "repro": file_record(OUT / name),
        }
    with (ORIGINAL_OUT / "prior_type.pkl").open("rb") as handle:
        original_lookup = pickle.load(handle)
    comparisons["prior_type.pkl"] = {
        "content_equal": original_lookup == prior_lookup,
        "entries_original": len(original_lookup),
        "entries_repro": len(prior_lookup),
    }

    generated_files = sorted(path for path in OUT.iterdir() if path.is_file())
    build_manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": now_utc(),
        "elapsed_seconds": time.time() - started,
        "prebuild_manifest_sha256": sha256_file(OUT / "prebuild_manifest.json"),
        "model_count": len(model_records),
        "parameter_mismatch_count": len(parameter_mismatches),
        "models": model_records,
        "derived_artifact_comparison": comparisons,
        "non_model_files": [file_record(path) for path in generated_files if path.suffix != ".cbm"],
        "forbidden_outputs_created": {
            "candidate_directory": (ROOT / "candidate" / EXPERIMENT_ID).exists(),
            "zip_matches": [str(p.relative_to(ROOT)) for p in (ROOT / "output").glob("*038A*.zip")],
        },
        "test_access": "not performed; derived trainer statically excludes test.csv and packaging section",
    }
    write_json(OUT / "build_manifest.json", build_manifest)
    update_status("full_train", "PASS", model_count=56, elapsed_seconds=time.time() - started)
    print(f"[{EXPERIMENT_ID}] FULL TRAIN PASS: 56/56 models; no candidate or ZIP created", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        if OUT.exists():
            update_status("failed", "ERROR", error_type=type(exc).__name__, error=str(exc))
        raise
