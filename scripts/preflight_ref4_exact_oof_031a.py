#!/usr/bin/env python3
"""Read-only source and bundle preflight for REF4-EXACT-OOF-031A."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "REF4-EXACT-OOF-031A"
OUT = ROOT / "model" / EXPERIMENT_ID
ZIP_PATH = ROOT / "output" / "submit_ref4_champion_030.zip"
CANDIDATE = ROOT / "candidate" / "REF4-CHAMPION-STACK-030"
TRAIN_PATH = ROOT / "data" / "train.csv"
REFERENCE_DOC = ROOT / "start03_reference.md"
ROADMAP_DOC = ROOT / "start04_uptostage.md"
BANNED_CALLS = {"groupby", "rolling", "expanding", "shift", "rank", "quantile", "median", "fit", "fit_transform"}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run_inference(root: Path, frame: pd.DataFrame) -> np.ndarray:
    data_dir = root / "data"
    output_dir = root / "output"
    data_dir.mkdir(exist_ok=True)
    output_dir.mkdir(exist_ok=True)
    frame.to_csv(data_dir / "test.csv", index=False)
    env = dict(os.environ)
    env.update({"OMP_NUM_THREADS": "3", "MKL_NUM_THREADS": "3"})
    completed = subprocess.run(
        [sys.executable, "script.py"], cwd=root, env=env,
        capture_output=True, text=True, timeout=600, check=True,
    )
    result = pd.read_csv(output_dir / "submission.csv")
    if len(result) != len(frame):
        raise AssertionError(f"inference row mismatch: {len(result)} != {len(frame)}; {completed.stdout}")
    return result["control_success"].to_numpy(float)


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, object]] = []
    mismatches: list[str] = []

    def record(name: str, passed: bool, actual: object) -> None:
        checks.append({"name": name, "checked": True, "passed": bool(passed), "actual": actual})
        if not passed:
            mismatches.append(name)

    for required in (ZIP_PATH, CANDIDATE / "script.py", TRAIN_PATH, REFERENCE_DOC, ROADMAP_DOC):
        record(f"exists:{required.relative_to(ROOT)}", required.is_file(), required.is_file())

    reference_text = REFERENCE_DOC.read_text(encoding="utf-8")
    roadmap_text = ROADMAP_DOC.read_text(encoding="utf-8")
    zip_sha = sha256_path(ZIP_PATH)
    record("zip_sha_declared_in_start03", zip_sha in reference_text, zip_sha)
    record("current_score_in_start03", "1068.25021" in reference_text, "1068.25021" in reference_text)
    record("current_score_in_start04", "1068.25021" in roadmap_text, "1068.25021" in roadmap_text)
    record("experiment_contract_declared", EXPERIMENT_ID in roadmap_text, EXPERIMENT_ID in roadmap_text)

    train = pd.read_csv(TRAIN_PATH, usecols=["row_id", "season", "control_success"])
    record("train_row_count", len(train) == 1_475_092, len(train))
    record("train_row_id_unique", train["row_id"].is_unique, int(train["row_id"].nunique()))
    target = train["control_success"].to_numpy(float)
    record("train_target_finite", bool(np.isfinite(target).all()), int(np.isfinite(target).sum()))
    record("train_target_binary", bool(np.isin(target, [0.0, 1.0]).all()), sorted(np.unique(target).tolist()))
    season_counts = {str(int(k)): int(v) for k, v in train.groupby("season", sort=True).size().items()}
    record("train_seasons_exact", set(season_counts) == {"2019", "2020", "2021", "2022", "2023", "2024"}, season_counts)

    manifest = json.loads((CANDIDATE / "model" / "manifest.json").read_text(encoding="utf-8"))
    regime = json.loads((CANDIDATE / "model" / "f_regime_meta.json").read_text(encoding="utf-8"))
    main_weight_sum = float(sum(manifest["main_weights"]))
    record(
        "main_weights_sum",
        abs(main_weight_sum - 1.0) <= 1e-7,
        {"sum": main_weight_sum, "abs_error": abs(main_weight_sum - 1.0), "inference_normalizes": True},
    )
    record("main_weights_count", len(manifest["main_weights"]) == 3, len(manifest["main_weights"]))
    record("stack_coefficients_count", len(manifest["stack_coefficients"]) == 4, len(manifest["stack_coefficients"]))
    record("seed_count", len(manifest["seeds"]) == 6, len(manifest["seeds"]))
    record("global_shift_finite", np.isfinite(float(manifest["global_shift"])), float(manifest["global_shift"]))
    record("transition_disabled", float(regime["transition_scale"]) == 0.0, float(regime["transition_scale"]))
    regime_values = {k: float(v) for k, v in regime.items()}
    record("regime_values_finite", all(np.isfinite(list(regime_values.values()))), regime_values)

    source_paths = [CANDIDATE / "script.py", *sorted((CANDIDATE / "src").glob("*.py"))]
    reachable = {
        "catboost_features.py": {"add_row_features", "build_catboost_features", "attach_trackman_features"},
        "preprocessing_v2.py": {"_safe_logit", "build_v2_features", "build_v3_features"},
        "season_delta_features.py": {"attach_season_delta"},
        "season_history_v3.py": {"attach_entity_season"},
        "league_transition.py": {"transition_features"},
    }
    banned_hits: list[str] = []
    for source_path in source_paths:
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        if source_path.name == "script.py":
            nodes = list(ast.walk(tree))
        else:
            selected = [
                node for node in tree.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name in reachable.get(source_path.name, set())
            ]
            nodes = [nested for node in selected for nested in ast.walk(node)]
        for node in nodes:
            if isinstance(node, ast.Call):
                name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id if isinstance(node.func, ast.Name) else ""
                if name in BANNED_CALLS:
                    banned_hits.append(f"{source_path.relative_to(ROOT)}:{getattr(node, 'lineno', 0)}:{name}")
    record("inference_banned_calls_absent", not banned_hits, banned_hits)

    required_names = {"script.py", "requirements.txt", "model/manifest.json", "model/f_regime_meta.json"}
    zip_fs_mismatches: list[str] = []
    with zipfile.ZipFile(ZIP_PATH) as archive:
        names = archive.namelist()
        record("zip_integrity", archive.testzip() is None, archive.testzip())
        record("zip_names_unique", len(names) == len(set(names)), {"total": len(names), "unique": len(set(names))})
        record("zip_required_names", required_names.issubset(names), sorted(required_names - set(names)))
        for name in names:
            if name.startswith("solution/"):
                local = ROOT / name
            else:
                local = CANDIDATE / name
            if not local.is_file():
                zip_fs_mismatches.append(f"missing:{name}")
                continue
            if sha256_bytes(archive.read(name)) != sha256_path(local):
                zip_fs_mismatches.append(f"sha256:{name}")
    record("zip_matches_candidate", not zip_fs_mismatches, zip_fs_mismatches)

    sample = pd.read_csv(ROOT / "data" / "test.csv", low_memory=False)
    dynamic = {"sample_rows": int(len(sample))}
    with tempfile.TemporaryDirectory(prefix="ref4_031a_preflight_") as tmp:
        sandbox = Path(tmp)
        with zipfile.ZipFile(ZIP_PATH) as archive:
            archive.extractall(sandbox)
        full = run_inference(sandbox, sample)
        singleton = np.array([run_inference(sandbox, sample.iloc[[i]].copy())[0] for i in range(len(sample))])
        permuted_frame = sample.sample(frac=1, random_state=3101)
        permuted = run_inference(sandbox, permuted_frame)
        restored = pd.Series(permuted, index=permuted_frame.index).sort_index().to_numpy()
        augmented_frame = pd.concat([sample, sample.iloc[[0]].copy()], ignore_index=True)
        augmented = run_inference(sandbox, augmented_frame)[:len(sample)]
        dynamic.update({
            "prediction_min": float(full.min()), "prediction_max": float(full.max()),
            "singleton_max_abs_diff": float(np.max(np.abs(full - singleton))),
            "permutation_max_abs_diff": float(np.max(np.abs(full - restored))),
            "augmentation_max_abs_diff": float(np.max(np.abs(full - augmented))),
        })
    record("dynamic_prediction_finite", bool(np.isfinite(full).all()), dynamic)
    record("dynamic_prediction_range", bool(((full >= 0.0) & (full <= 1.0)).all()), dynamic)
    record("dynamic_singleton_independence", dynamic["singleton_max_abs_diff"] <= 1e-12, dynamic["singleton_max_abs_diff"])
    record("dynamic_permutation_independence", dynamic["permutation_max_abs_diff"] <= 1e-12, dynamic["permutation_max_abs_diff"])
    record("dynamic_augmentation_independence", dynamic["augmentation_max_abs_diff"] <= 1e-12, dynamic["augmentation_max_abs_diff"])

    status = "AUDIT_VERIFIED" if not mismatches else "FAIL"
    report = {
        "experiment_id": EXPERIMENT_ID,
        "scope": "source_bundle_preflight",
        "status": status,
        "checked_count": len(checks),
        "passed_count": sum(bool(c["passed"]) for c in checks),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "zip_sha256": zip_sha,
        "zip_file_count": len(names),
        "source_python_file_count": len(source_paths),
        "checks": checks,
        "elapsed_seconds": time.time() - started,
    }
    report_path = OUT / "preflight_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown = [
        f"# {EXPERIMENT_ID} source preflight", "",
        f"- status: `{status}`", f"- checked: `{len(checks)}`",
        f"- mismatch: `{len(mismatches)}`", f"- ZIP files: `{len(names)}`",
        f"- ZIP SHA-256: `{zip_sha}`", "", "## Mismatches", "",
        *(f"- `{item}`" for item in mismatches),
    ]
    if not mismatches:
        markdown.append("- none")
    (OUT / "preflight_report.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "checked_count", "passed_count", "mismatch_count", "zip_file_count", "elapsed_seconds")}, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
