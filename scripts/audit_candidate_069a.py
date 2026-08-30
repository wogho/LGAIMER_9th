#!/usr/bin/env python3
"""Strict Row-Independence Audit for submit_ref4_adaptive_channel_opt_069."""
import os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / 'output/submit_ref4_adaptive_channel_opt_069.zip'
AUDIT_DIR = ROOT / 'model/REF4-ADAPTIVE-CHANNEL-OPT-069A/audit_sandbox'
PYTHON_BIN = str(ROOT / '.venv-submit/bin/python')

def run_inference(sandbox: Path, test_df: pd.DataFrame) -> pd.DataFrame:
    data_dir = sandbox / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(data_dir / 'test.csv', index=False)
    
    cmd = [PYTHON_BIN, 'script.py']
    p = subprocess.run(cmd, cwd=str(sandbox), capture_output=True, text=True)
    if p.returncode != 0:
        print("STDOUT:", p.stdout)
        print("STDERR:", p.stderr)
        raise RuntimeError("Inference failed in sandbox")
        
    out_csv = sandbox / 'output/submission.csv'
    return pd.read_csv(out_csv)

def main():
    t0 = time.time()
    print(f"=== Unzipping {ZIP_PATH.name} into audit sandbox ===")
    if AUDIT_DIR.exists():
        shutil.rmtree(AUDIT_DIR)
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall(AUDIT_DIR)
        
    # Use 1,500 rows from 2024 validation data as test probe
    raw = pd.read_csv(ROOT / 'data/train.csv', low_memory=False)
    test_probe = raw.loc[raw.season == 2024].iloc[:1500].copy().reset_index(drop=True)
    n_probe = len(test_probe)
    
    print(f"1. Running Baseline inference on {n_probe} probe rows...")
    res_base = run_inference(AUDIT_DIR, test_probe)
    p_base = res_base['control_success'].to_numpy(float)
    
    print(f"2. Running Half-file inference on first 750 rows...")
    res_half = run_inference(AUDIT_DIR, test_probe.iloc[:750].copy().reset_index(drop=True))
    p_half = res_half['control_success'].to_numpy(float)
    diff_half = np.max(np.abs(p_base[:750] - p_half))
    print(f"   Max |diff| (Half vs Full): {diff_half:.3e}")
    
    print(f"3. Running Shuffled-file inference on {n_probe} rows...")
    shuffled_idx = np.random.default_rng(42).permutation(n_probe)
    test_shuffled = test_probe.iloc[shuffled_idx].copy().reset_index(drop=True)
    res_shuffled = run_inference(AUDIT_DIR, test_shuffled)
    unshuffle_order = np.argsort(shuffled_idx)
    p_shuffled_reordered = res_shuffled['control_success'].to_numpy(float)[unshuffle_order]
    diff_shuffled = np.max(np.abs(p_base - p_shuffled_reordered))
    print(f"   Max |diff| (Shuffled vs Full): {diff_shuffled:.3e}")
    
    print(f"4. Running Single-Row isolated inference on 8 diverse test rows...")
    selected_indices = [0, 50, 150, 250, 350, 500, 750, 1499]
    isolated_diffs = []
    for idx in selected_indices:
        single_row_df = test_probe.iloc[[idx]].copy().reset_index(drop=True)
        res_single = run_inference(AUDIT_DIR, single_row_df)
        p_single = float(res_single['control_success'].iloc[0])
        diff_single = abs(p_base[idx] - p_single)
        isolated_diffs.append(diff_single)
        print(f"   Row {idx:4d}: Base={p_base[idx]:.8f}, Single={p_single:.8f}, Diff={diff_single:.3e}")
        
    max_isolated_diff = max(isolated_diffs)
    
    audit_pass = (diff_half < 1e-9) and (diff_shuffled < 1e-9) and (max_isolated_diff < 1e-9)
    print(f"\n=== Row-Independence Audit Summary ===")
    print(f"  Half file diff:     {diff_half:.3e} (PASS: {diff_half < 1e-9})")
    print(f"  Shuffled file diff: {diff_shuffled:.3e} (PASS: {diff_shuffled < 1e-9})")
    print(f"  Isolated row diff:  {max_isolated_diff:.3e} (PASS: {max_isolated_diff < 1e-9})")
    print(f"AUDIT VERDICT: {'PASS' if audit_pass else 'FAIL'}")
    print(f"Completed in {time.time() - t0:.2f}s")
    
    if not audit_pass:
        sys.exit(1)

if __name__ == '__main__':
    main()
