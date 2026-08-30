#!/usr/bin/env python3
"""Test Ensemble Blend between 071A/075A and Repo 1."""
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPO1_DIR = ROOT / 'github_reference/1번 레포/open/baseline_submit'
PYTHON_BIN = str(ROOT / '.venv-submit/bin/python')

def bss(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-5, 1 - 1e-5)
    r = float(y.mean())
    ref = r * (1.0 - r)
    return 1e5 * (1.0 - np.mean((p - y) ** 2) / ref)

def main():
    raw = pd.read_csv(ROOT / 'data/train.csv', low_memory=False)
    oof_069 = pd.read_csv(ROOT / 'model/REF4-ADAPTIVE-CHANNEL-OPT-069A/oof_predictions.csv').set_index('row_id')
    oof_051 = pd.read_csv(ROOT / 'model/REF4-TRAINONLY-R-SPECIFIC-SPLIT-051A/oof_predictions.csv').set_index('row_id')
    
    val_24 = raw.loc[raw.season == 2024].copy().reset_index(drop=True)
    y_24 = val_24.control_success.to_numpy(float)
    row_ids_24 = val_24.row_id.to_numpy()
    
    p_071_24 = oof_069.loc[row_ids_24, 'prediction'].to_numpy(float)
    p_051_24 = oof_051.loc[row_ids_24, 'candidate_prediction'].to_numpy(float)
    
    # Run Repo 1 inference on 2024 data
    print("Running Repo 1 inference on 2024 validation data...")
    test_repo1_dir = REPO1_DIR / 'data'
    test_repo1_dir.mkdir(parents=True, exist_ok=True)
    val_24.to_csv(test_repo1_dir / 'test.csv', index=False)
    pd.DataFrame({'row_id': val_24.row_id, 'control_success': 0.5}).to_csv(test_repo1_dir / 'sample_submission.csv', index=False)
    
    t0 = time.time()
    p = subprocess.run([PYTHON_BIN, 'script.py'], cwd=str(REPO1_DIR), capture_output=True, text=True)
    if p.returncode != 0:
        print("Repo 1 STDERR:", p.stderr)
        print("Repo 1 STDOUT:", p.stdout)
        raise RuntimeError("Repo 1 execution failed!")
        
    res_repo1 = pd.read_csv(REPO1_DIR / 'output/submission.csv')
    p_repo1_24 = res_repo1['control_success'].to_numpy(float)
    print(f"Repo 1 inference completed in {time.time() - t0:.2f}s")
    
    print(f"\n=== Standalone Validation Scores on 2024 ===")
    print(f"051A Baseline BSS: {bss(y_24, p_051_24):.4f}")
    print(f"071A Champion BSS: {bss(y_24, p_071_24):.4f}")
    print(f"Repo 1 Alone BSS:  {bss(y_24, p_repo1_24):.4f}")
    
    print(f"\nCorrelation between 071A and Repo1 predictions: {np.corrcoef(p_071_24, p_repo1_24)[0,1]:.4f}")
    
    print("\n=== Ensemble Blend Grid Search (071A + Repo1) ===")
    best_w = 0.0
    best_score = -9999
    for w in [0.0, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40]:
        p_blend = (1.0 - w) * p_071_24 + w * p_repo1_24
        score = bss(y_24, p_blend)
        diff = score - bss(y_24, p_071_24)
        print(f"Repo1 Weight {w:4.2f} (071: {1-w:4.2f}) -> 2024 BSS: {score:8.4f} (diff vs 071: {diff:+8.4f})")
        if score > best_score:
            best_score = score
            best_w = w
            
    print(f"\nBest Blend Found: Repo1 Weight = {best_w:.2f} -> 2024 BSS = {best_score:.4f} (+{best_score - bss(y_24, p_071_24):.4f} gain!)")

if __name__ == '__main__':
    main()
