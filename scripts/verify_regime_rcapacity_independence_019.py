#!/usr/bin/env python3
"""Strict Row-Independence and Isolated E2E Verification for REGIME-RCAPACITY-FULL-019."""
from __future__ import annotations

import ast
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_DIR = ROOT / "candidate" / "REGIME-RCAPACITY-FULL-019"
OUT_DIR = ROOT / "model" / "REGIME-RCAPACITY-FULL-019"
PYTHON_BIN = sys.executable

HIGH_RISK_CALLS = {
    "cumcount", "cummax", "cummin", "cumprod", "cumsum", "diff", "expanding",
    "fit", "fit_transform", "groupby", "mean", "median", "mode", "nunique",
    "partial_fit", "pct_change", "pivot_table", "quantile", "rank",
    "resample", "rolling", "shift", "std", "value_counts", "var"
}


def assert_no_high_risk_calls(script_path: Path) -> list[str]:
    source = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(script_path))
    findings = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        call_name = None
        if isinstance(node.func, ast.Attribute):
            call_name = node.func.attr
        elif isinstance(node.func, ast.Name):
            call_name = node.func.id
        if call_name in HIGH_RISK_CALLS:
            findings.append(f"{call_name}() at line {node.lineno}")
    return findings


def run_isolated_inference(package_dir: Path, test_df: pd.DataFrame) -> pd.DataFrame:
    with tempfile.TemporaryDirectory(prefix="regime_rcapacity_iso_") as temp_dir:
        temp_root = Path(temp_dir)
        # Copy candidate files into temporary isolated sandbox
        shutil.copytree(package_dir / "model", temp_root / "model")
        shutil.copy2(package_dir / "script.py", temp_root / "script.py")
        shutil.copy2(package_dir / "requirements.txt", temp_root / "requirements.txt")
        (temp_root / "data").mkdir(exist_ok=True)
        (temp_root / "output").mkdir(exist_ok=True)
        test_df.to_csv(temp_root / "data" / "test.csv", index=False, encoding="utf-8-sig")

        res = subprocess.run(
            [PYTHON_BIN, "script.py"],
            cwd=temp_root,
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        sub_path = temp_root / "output" / "submission.csv"
        if not sub_path.exists():
            raise RuntimeError(f"submission.csv not generated: stdout={res.stdout}, stderr={res.stderr}")
        return pd.read_csv(sub_path, encoding="utf-8")


def main() -> None:
    start_time = time.time()
    print("[INDEPENDENCE-AUDIT] Starting isolated row-independence verification...")

    script_path = CANDIDATE_DIR / "script.py"
    high_risk_findings = assert_no_high_risk_calls(script_path)
    if high_risk_findings:
        raise RuntimeError(f"High risk AST calls detected in script.py: {high_risk_findings}")
    print("  ✅ AST Static Check: No forbidden batch/rolling/shift/groupby calls found.")

    test_raw = pd.read_csv(ROOT / "data" / "test.csv", encoding="utf-8-sig")
    print(f"  Loaded official test.csv: {len(test_raw)} rows.")

    # 1. Base Full Batch Prediction
    print("  Running Test 1: Full Batch Prediction...")
    sub_base = run_isolated_inference(CANDIDATE_DIR, test_raw)
    if not sub_base.row_id.equals(test_raw.row_id):
        raise RuntimeError("Full batch row_id order mismatch")
    p_base = sub_base.set_index("row_id")["control_success"]

    # 2. Singleton Test (Row-by-Row)
    print("  Running Test 2: Singleton (Row-by-Row) Independence...")
    singleton_preds = []
    for idx in range(len(test_raw)):
        single_row_df = test_raw.iloc[[idx]].copy()
        sub_single = run_isolated_inference(CANDIDATE_DIR, single_row_df)
        singleton_preds.append(sub_single)
    sub_singleton = pd.concat(singleton_preds, ignore_index=True)
    p_singleton = sub_singleton.set_index("row_id")["control_success"].reindex(p_base.index)

    diff_singleton = float(np.max(np.abs(p_base - p_singleton)))
    print(f"    Singleton Max Absolute Diff: {diff_singleton:.16f}")
    if diff_singleton > 1e-12:
        raise RuntimeError(f"Singleton test failed: max diff = {diff_singleton}")
    print("  ✅ Singleton Independence: PASS (0.0 diff)")

    # 3. Permutation Test
    print("  Running Test 3: Permutation Independence...")
    perm_indices = [3, 0, 4, 1, 2] if len(test_raw) == 5 else list(reversed(range(len(test_raw))))
    df_perm = test_raw.iloc[perm_indices].copy()
    sub_perm = run_isolated_inference(CANDIDATE_DIR, df_perm)
    p_perm = sub_perm.set_index("row_id")["control_success"].reindex(p_base.index)

    diff_perm = float(np.max(np.abs(p_base - p_perm)))
    print(f"    Permutation Max Absolute Diff: {diff_perm:.16f}")
    if diff_perm > 1e-12:
        raise RuntimeError(f"Permutation test failed: max diff = {diff_perm}")
    print("  ✅ Permutation Independence: PASS (0.0 diff)")

    # 4. Augmentation Test (Adding Duplicate/Dummy Rows)
    print("  Running Test 4: Augmentation Independence...")
    df_aug = pd.concat([test_raw, test_raw.iloc[[0]].assign(row_id=999999), test_raw.iloc[[1]].assign(row_id=999998)], ignore_index=True)
    sub_aug = run_isolated_inference(CANDIDATE_DIR, df_aug)
    p_aug = sub_aug.set_index("row_id")["control_success"].reindex(p_base.index)

    diff_aug = float(np.max(np.abs(p_base - p_aug)))
    print(f"    Augmentation Max Absolute Diff: {diff_aug:.16f}")
    if diff_aug > 1e-12:
        raise RuntimeError(f"Augmentation test failed: max diff = {diff_aug}")
    print("  ✅ Augmentation Independence: PASS (0.0 diff)")

    # 5. Stress Test with Train Sample (10,000 rows pseudo-test)
    print("  Running Test 5: Stress E2E Test on 10,000 pseudo-test rows...")
    raw_train = pd.read_csv(ROOT / "data" / "train.csv", nrows=10000, encoding="utf-8-sig")
    pseudo_test = raw_train.drop(columns=["control_success"], errors="ignore").copy()
    pseudo_test["season"] = 2025

    t0 = time.time()
    sub_stress = run_isolated_inference(CANDIDATE_DIR, pseudo_test)
    t_stress = time.time() - t0
    print(f"    10,000 rows inference completed in {t_stress:.2f}s ({len(pseudo_test)/t_stress:.0f} rows/s).")
    if len(sub_stress) != 10000 or not np.isfinite(sub_stress.control_success).all():
        raise RuntimeError("Stress inference output contract invalid")
    print("  ✅ Stress Test: PASS")

    elapsed = time.time() - start_time
    report = {
        "experiment_id": "REGIME-RCAPACITY-FULL-019",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "static_ast_check": "PASS",
        "singleton_max_diff": diff_singleton,
        "permutation_max_diff": diff_perm,
        "augmentation_max_diff": diff_aug,
        "stress_test": {
            "rows": 10000,
            "elapsed_seconds": t_stress,
            "throughput_rows_per_sec": float(len(pseudo_test) / t_stress),
            "status": "PASS",
        },
        "all_checks_passed": True,
        "elapsed_seconds": elapsed,
        "verdict": "ROW_INDEPENDENCE_AUDIT_VERIFIED",
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report_file = OUT_DIR / "independence_report.json"
    report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\n[INDEPENDENCE-AUDIT] Completed successfully in {elapsed:.1f}s. Verdict: ROW_INDEPENDENCE_AUDIT_VERIFIED.")


if __name__ == "__main__":
    main()
