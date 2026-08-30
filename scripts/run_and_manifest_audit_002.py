#!/usr/bin/env python3
"""
RUN-AND-MANIFEST-AUDIT-002:
Automated execution of all 6 Round 2 audit evaluation scripts and complete manifest generation.
Includes 9 model metadata files, audit tools, dynamic environment info, and complete provenance tracking.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import catboost
import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "model" / "ROUND2-AUDIT-FIX-002"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_SCRIPTS = [
    "scripts/diagnose_transfer_002.py",
    "scripts/evaluate_regime_calibration_002.py",
    "scripts/evaluate_catboost_ronly_001.py",
    "scripts/evaluate_regime_confidence_001.py",
    "scripts/evaluate_r_stack_oof_001.py",
    "scripts/evaluate_catboost_d7_001.py",
]

AUDIT_TOOL_SCRIPTS = [
    "scripts/run_and_manifest_audit_002.py",
    "scripts/verify_audit_manifest.py",
]

MODEL_METADATA_FILES = [
    "model/CAT-FE001-RONLY-EW-2022/metadata.json",
    "model/CAT-FE001-RONLY-EW-2023/metadata.json",
    "model/CAT-FE001-RONLY-2024/metadata.json",
    "model/CAT-FE001-RONLY-D7-EW-2022/metadata.json",
    "model/CAT-FE001-RONLY-D7-EW-2023/metadata.json",
    "model/CAT-FE001-RONLY-D7-2024/metadata.json",
    "model/LGBM-FE001-RONLY-EW-2022/metadata.json",
    "model/LGBM-FE001-RONLY-EW-2023/metadata.json",
    "model/LGBM-FE001-RONLY-2024/metadata.json",
]

SUBMISSION_ZIPS = {
    "sub001_rollback_zip": ROOT / "output" / "submit_final_selective.zip",
    "sub002_active_candidate_zip": ROOT / "output" / "candidates" / "submit_regime_r_candidate.zip",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_file_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "path": str(path)}
    return {
        "exists": True,
        "path": str(path.relative_to(ROOT)),
        "size_bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "last_modified": datetime.datetime.fromtimestamp(
            path.stat().st_mtime, tz=datetime.timezone.utc
        ).isoformat(),
    }


def main() -> None:
    manifest: dict[str, Any] = {
        "manifest_id": "ROUND2-AUDIT-FIX-002",
        "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
            "linux_kernel_release": platform.release(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "libraries": {
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit-learn": sklearn.__version__,
                "lightgbm": lgb.__version__,
                "catboost": catboost.__version__,
            },
        },
        "scripts_execution": {},
        "files_manifest": {
            "eval_scripts": {},
            "audit_tools": {},
            "model_metadata": {},
            "source_inputs": {},
            "output_json": {},
            "output_markdown": {},
            "preserved_submission_zips": {},
        },
    }

    print("========================================================================")
    print("ROUND2-AUDIT-FIX-002: Automated Execution of 6 Evaluation Scripts")
    print("========================================================================")

    # 1. Execute all 6 eval scripts
    for script_rel in EVAL_SCRIPTS:
        script_path = ROOT / script_rel
        print(f"\n[RUNNING] {script_rel} ...")
        t0 = time.time()
        res = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(ROOT),
            capture_output=True,
            text=True,
        )
        elapsed = time.time() - t0
        print(f"  Exit code: {res.returncode} | Elapsed: {elapsed:.2f}s")
        if res.returncode != 0:
            print(f"  STDERR: {res.stderr[:500]}")
            raise RuntimeError(f"Script {script_rel} failed with exit code {res.returncode}")

        manifest["scripts_execution"][script_rel] = {
            "exit_code": res.returncode,
            "elapsed_seconds": elapsed,
            "status": "SUCCESS" if res.returncode == 0 else "FAILED",
        }

    # 2. Record eval scripts and audit tools manifest
    for script_rel in EVAL_SCRIPTS:
        manifest["files_manifest"]["eval_scripts"][script_rel] = get_file_metadata(ROOT / script_rel)
    for tool_rel in AUDIT_TOOL_SCRIPTS:
        manifest["files_manifest"]["audit_tools"][tool_rel] = get_file_metadata(ROOT / tool_rel)

    # 3. Record model metadata manifest (9 models)
    for meta_rel in MODEL_METADATA_FILES:
        manifest["files_manifest"]["model_metadata"][meta_rel] = get_file_metadata(ROOT / meta_rel)

    # 4. Record source inputs manifest
    raw_train = ROOT / "data" / "train.csv"
    manifest["files_manifest"]["source_inputs"]["data_train_csv"] = get_file_metadata(raw_train)

    for season in [2022, 2023, 2024]:
        sel_file = (
            ROOT / "model" / "ENS-CATF-LGBMCATR5050-FE001-EW-2022" / f"selective_predictions_{season}.csv"
            if season == 2022
            else ROOT / "model" / "ENS-CATF-LGBMCATR5050-FE001" / f"selective_predictions_{season}.csv"
        )
        lgbm_r = (
            ROOT / "model" / f"LGBM-FE001-RONLY-EW-{season}" / "validation_predictions.csv"
            if season in [2022, 2023]
            else ROOT / "model" / "LGBM-FE001-RONLY-2024" / "validation_predictions.csv"
        )
        cat_r = (
            ROOT / "model" / f"CAT-FE001-RONLY-EW-{season}" / "validation_predictions.csv"
            if season in [2022, 2023]
            else ROOT / "model" / "CAT-FE001-RONLY-2024" / "validation_predictions.csv"
        )
        cat_d7 = (
            ROOT / "model" / f"CAT-FE001-RONLY-D7-EW-{season}" / "validation_predictions.csv"
            if season in [2022, 2023]
            else ROOT / "model" / "CAT-FE001-RONLY-D7-2024" / "validation_predictions.csv"
        )

        manifest["files_manifest"]["source_inputs"][f"selective_preds_{season}"] = get_file_metadata(sel_file)
        manifest["files_manifest"]["source_inputs"][f"lgbm_r_preds_{season}"] = get_file_metadata(lgbm_r)
        manifest["files_manifest"]["source_inputs"][f"cat_r_preds_{season}"] = get_file_metadata(cat_r)
        manifest["files_manifest"]["source_inputs"][f"cat_d7_preds_{season}"] = get_file_metadata(cat_d7)

    # 5. Record outputs manifest
    output_pairs = [
        ("TRANSFER-DIAG-002", "transfer_diag_results.json", "transfer_diag_report.md"),
        ("CAL-REGIME-002", "cal_regime_results.json", "cal_regime_report.md"),
        ("CAT-RONLY-001", "cat_ronly_results.json", "cat_ronly_report.md"),
        ("REGIME-CONFIDENCE-001", "confidence_routing_results.json", "confidence_routing_report.md"),
        ("R-STACK-OOF-001", "stacking_results.json", "stacking_report.md"),
        ("CAT-RONLY-DEPTH7-001", "cat_d7_results.json", "cat_d7_report.md"),
    ]

    for dir_name, json_name, md_name in output_pairs:
        json_p = ROOT / "model" / dir_name / json_name
        md_p = ROOT / "model" / dir_name / md_name
        manifest["files_manifest"]["output_json"][f"{dir_name}/{json_name}"] = get_file_metadata(json_p)
        manifest["files_manifest"]["output_markdown"][f"{dir_name}/{md_name}"] = get_file_metadata(md_p)

    # 6. Record preserved submission ZIPs
    for label, zip_path in SUBMISSION_ZIPS.items():
        manifest["files_manifest"]["preserved_submission_zips"][label] = get_file_metadata(zip_path)

    manifest_path = OUTPUT_DIR / "audit_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_hash = sha256_file(manifest_path)
    print("\n========================================================================")
    print(f"Audit Manifest Generated: {manifest_path}")
    print(f"Manifest SHA-256: {manifest_hash}")
    print("========================================================================")


if __name__ == "__main__":
    main()
