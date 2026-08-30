#!/usr/bin/env python3
"""Evaluate Unified 075A (Decoupled 2군 4052) + Repo 1 (1038) Orthogonal Blend."""
import json, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SANDBOX_075 = ROOT / 'model/REF4-MACRO-LEAP-CHAMPION-075A/audit_sandbox'
REPO1_DIR = ROOT / 'github_reference/1번 레포/open/baseline_submit'
PYTHON_BIN = str(ROOT / '.venv-submit/bin/python')

def bss(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-5, 1 - 1e-5)
    r = float(y.mean())
    ref = r * (1.0 - r)
    return 1e5 * (1.0 - np.mean((p - y) ** 2) / ref)

def cluster_bootstrap_gain(y, p_cand, p_ref, clusters, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    unique_clusters = np.unique(clusters)
    n_c = len(unique_clusters)
    cluster_indices = {c: np.where(clusters == c)[0] for c in unique_clusters}
    cand_se = (p_cand - y) ** 2
    ref_se = (p_ref - y) ** 2
    diff_se = ref_se - cand_se
    cluster_means = np.array([np.sum(diff_se[cluster_indices[c]]) for c in unique_clusters])
    cluster_lens = np.array([len(cluster_indices[c]) for c in unique_clusters])
    boot_gains = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample_c = rng.integers(0, n_c, size=n_c)
        tot_diff = np.sum(cluster_means[sample_c])
        tot_rows = np.sum(cluster_lens[sample_c])
        boot_gains[i] = tot_diff / tot_rows
    ci_low, ci_high = np.percentile(boot_gains, [2.5, 97.5])
    return float(np.mean(diff_se)), float(ci_low), float(ci_high)

def main():
    raw = pd.read_csv(ROOT / 'data/train.csv', low_memory=False)
    val_24 = raw.loc[raw.season == 2024].copy().reset_index(drop=True)
    y_24 = val_24.control_success.to_numpy(float)
    pitchers_24 = val_24.pitcher_id.to_numpy()
    
    # 1. Run 075A inference on 2024
    test_dir_075 = SANDBOX_075 / 'data'
    test_dir_075.mkdir(parents=True, exist_ok=True)
    val_24.to_csv(test_dir_075 / 'test.csv', index=False)
    subprocess.run([PYTHON_BIN, 'script.py'], cwd=str(SANDBOX_075), capture_output=True, text=True)
    p_075_24 = pd.read_csv(SANDBOX_075 / 'output/submission.csv')['control_success'].to_numpy(float)
    
    # 2. Run Repo 1 inference on 2024
    test_dir_repo1 = REPO1_DIR / 'data'
    test_dir_repo1.mkdir(parents=True, exist_ok=True)
    val_24.to_csv(test_dir_repo1 / 'test.csv', index=False)
    pd.DataFrame({'row_id': val_24.row_id, 'control_success': 0.5}).to_csv(test_dir_repo1 / 'sample_submission.csv', index=False)
    subprocess.run([PYTHON_BIN, 'script.py'], cwd=str(REPO1_DIR), capture_output=True, text=True)
    p_repo1_24 = pd.read_csv(REPO1_DIR / 'output/submission.csv')['control_success'].to_numpy(float)
    
    oof_069 = pd.read_csv(ROOT / 'model/REF4-ADAPTIVE-CHANNEL-OPT-069A/oof_predictions.csv').set_index('row_id')
    p_071_24 = oof_069.loc[val_24.row_id, 'prediction'].to_numpy(float)
    
    print("=== Standalone Models on 2024 ===")
    print(f"071A Champion (LB 1092): BSS = {bss(y_24, p_071_24):.4f}")
    print(f"075A Macro-Leap Model:   BSS = {bss(y_24, p_075_24):.4f}")
    print(f"Repo 1 Alone:            BSS = {bss(y_24, p_repo1_24):.4f}")
    
    print("\n=== Grid Search: 075A + Repo1 Blend ===")
    for w in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
        p_blend = (1.0 - w) * p_075_24 + w * p_repo1_24
        score = bss(y_24, p_blend)
        gain, ci_low, ci_high = cluster_bootstrap_gain(y_24, p_blend, p_071_24, pitchers_24, n_boot=2000, seed=42)
        print(f"Repo1 {w:4.2f} + 075A {1-w:4.2f} -> BSS: {score:8.4f} (diff vs 071: {score - bss(y_24, p_071_24):+8.4f}) | Gain: {gain:+.8f} (CI: [{ci_low:+.8f}, {ci_high:+.8f}])")

if __name__ == '__main__':
    main()
