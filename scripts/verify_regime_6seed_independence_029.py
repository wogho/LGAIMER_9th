#!/usr/bin/env python3
"""Strict Row-Independence and Isolated E2E Verification for REGIME-6SEED-FULL-029."""
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
CANDIDATE_DIR = ROOT / "candidate" / "REGIME-6SEED-FULL-029"
OUT_DIR = ROOT / "model" / "REGIME-6SEED-FULL-029"
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
    with tempfile.TemporaryDirectory(prefix="regime_6seed_iso_") as temp_dir:
        temp_root = Path(temp_dir)
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
    print("[INDEPENDENCE-AUDIT] Starting isolated row-independence verification for 6-Seed Model...")

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

    max_diff_singleton = float(np.max(np.abs(p_base.to_numpy() - p_singleton.to_numpy())))
    print(f"  ✅ Max diff (Full vs Singleton): {max_diff_singleton:.16f}")
    if max_diff_singleton > 1e-12:
        raise RuntimeError(f"Singleton test failed with max diff: {max_diff_singleton}")

    # 3. Permutation Test
    print("  Running Test 3: Permutation Invariance...")
    permuted_indices = np.random.RandomState(42).permutation(len(test_raw))
    test_permuted = test_raw.iloc[permuted_indices].copy()
    sub_perm = run_isolated_inference(CANDIDATE_DIR, test_permuted)
    p_perm = sub_perm.set_index("row_id")["control_success"].reindex(p_base.index)

    max_diff_perm = float(np.max(np.abs(p_base.to_numpy() - p_perm.to_numpy())))
    print(f"  ✅ Max diff (Full vs Permuted): {max_diff_perm:.16f}")
    if max_diff_perm > 1e-12:
        raise RuntimeError(f"Permutation test failed with max diff: {max_diff_perm}")

    # 4. Augmentation Test
    print("  Running Test 4: Augmentation Invariance...")
    dummy_rows = test_raw.iloc[:2].copy()
    dummy_rows["row_id"] = [f"dummy_{i}" for i in range(len(dummy_rows))]
    test_augmented = pd.concat([test_raw, dummy_rows], ignore_index=True)
    sub_aug = run_isolated_inference(CANDIDATE_DIR, test_augmented)
    p_aug = sub_aug.set_index("row_id")["control_success"].reindex(p_base.index)

    max_diff_aug = float(np.max(np.abs(p_base.to_numpy() - p_aug.to_numpy())))
    print(f"  ✅ Max diff (Full vs Augmented): {max_diff_aug:.16f}")
    if max_diff_aug > 1e-12:
        raise RuntimeError(f"Augmentation test failed with max diff: {max_diff_aug}")

    # 5. Throughput Stress Test
    print("  Running Test 5: Synthetic 10,000-row stress test...")
    reps = int(np.ceil(10000 / len(test_raw)))
    synthetic_test = pd.concat([test_raw] * reps, ignore_index=True).iloc[:10000].copy()
    synthetic_test["row_id"] = [f"synth_{i}" for i in range(10000)]

    t0 = time.time()
    sub_synth = run_isolated_inference(CANDIDATE_DIR, synthetic_test)
    stress_elapsed = time.time() - t0
    throughput = 10000 / stress_elapsed
    print(f"  ✅ 10,000 rows inferred in {stress_elapsed:.2f}s ({throughput:.1f} rows/s).")

    report = {
        "candidate_id": "REGIME-6SEED-FULL-029",
        "ast_clean": True,
        "high_risk_findings": [],
        "max_diff_singleton": max_diff_singleton,
        "max_diff_permutation": max_diff_perm,
        "max_diff_augmentation": max_diff_aug,
        "stress_10000_elapsed_sec": stress_elapsed,
        "throughput_rows_per_sec": throughput,
        "all_tests_passed": True,
        "audit_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "audit_status": "ROW_INDEPENDENCE_AUDIT_VERIFIED"
    }

    report_path = OUT_DIR / "row_independence_audit_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n" + "=" * 60)
    print("  >>> [AUDIT_PASS] 6-Seed Model Passed 100% Strict Row-Independence Audit!")
    print(f"Report saved: {report_path}")


if __name__ == "__main__":
    main()
