#!/usr/bin/env python3
"""Comprehensive Multi-Repo Super Blend Evaluation on 2024 Holdout."""
import json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = str(ROOT / '.venv-submit/bin/python')
SANDBOX = ROOT / 'model/REF4-SUPER-BLEND-CHAMPION-077A/test_sandbox'

def bss(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-5, 1 - 1e-5)
    r = float(y.mean())
    ref = r * (1.0 - r)
    return 1e5 * (1.0 - np.mean((p - y) ** 2) / ref)

def run_repo1(val_df: pd.DataFrame) -> np.ndarray:
    repo1_dir = ROOT / 'github_reference/1번 레포/open/baseline_submit'
    data_dir = repo1_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    val_df.to_csv(data_dir / 'test.csv', index=False)
    pd.DataFrame({'row_id': val_df.row_id, 'control_success': 0.5}).to_csv(data_dir / 'sample_submission.csv', index=False)
    subprocess.run([PYTHON_BIN, 'script.py'], cwd=str(repo1_dir), capture_output=True)
    return pd.read_csv(repo1_dir / 'output/submission.csv')['control_success'].to_numpy(float)

def run_repo3(val_df: pd.DataFrame) -> np.ndarray:
    sandbox_r3 = SANDBOX / 'repo3'
    if not (sandbox_r3 / 'script.py').exists():
        sandbox_r3.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(ROOT / 'github_reference/3번 레포/submissions/cand_final.zip') as zf:
            zf.extractall(sandbox_r3)
    data_dir = sandbox_r3 / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    val_df.to_csv(data_dir / 'test.csv', index=False)
    pd.DataFrame({'row_id': val_df.row_id, 'control_success': 0.5}).to_csv(data_dir / 'sample_submission.csv', index=False)
    subprocess.run([PYTHON_BIN, 'script.py'], cwd=str(sandbox_r3), capture_output=True)
    return pd.read_csv(sandbox_r3 / 'output/submission.csv')['control_success'].to_numpy(float)

def run_071a(val_df: pd.DataFrame) -> np.ndarray:
    p71_dir = ROOT / 'model/REF4-ADAPTIVE-CHANNEL-OPT-071A/production_package'
    data_dir = p71_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    val_df.to_csv(data_dir / 'test.csv', index=False)
    subprocess.run([PYTHON_BIN, 'script.py'], cwd=str(p71_dir), capture_output=True)
    return pd.read_csv(p71_dir / 'output/submission.csv')['control_success'].to_numpy(float)

def cluster_bootstrap(y, p_base, p_cand, clusters, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    unique_clusters = np.unique(clusters)
    cluster_indices = {c: np.where(clusters == c)[0] for c in unique_clusters}
    gains = []
    
    r = float(y.mean())
    ref = r * (1.0 - r)
    
    for _ in range(n_boot):
        sample_c = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        idx = np.concatenate([cluster_indices[c] for c in sample_c])
        y_b = y[idx]
        b_base = np.mean((p_base[idx] - y_b) ** 2)
        b_cand = np.mean((p_cand[idx] - y_b) ** 2)
        gain_bss = (1.0 - b_cand / ref) - (1.0 - b_base / ref)
        gains.append(gain_bss)
        
    gains = np.array(gains)
    return float(np.mean(gains)), float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5)), float(np.mean(gains > 0))

def main():
    print("=" * 75)
    print("       MULTI-REPOSITORY SUPER-BLEND CANDIDATE EVALUATION         ")
    print("=" * 75)
    
    train = pd.read_csv(ROOT / 'data/train.csv', low_memory=False)
    val_24 = train.loc[train.season == 2024].copy().reset_index(drop=True)
    y_24 = val_24.control_success.to_numpy(float)
    clusters = val_24.pitcher_id.to_numpy()
    
    print(f"Validation dataset: 2024 Season (Total {len(val_24):,} rows)")
    
    # 1. Obtain individual predictions
    print("\n[Step 1] Running Component Model Inferences on 2024...")
    t0 = time.time()
    p_071 = run_071a(val_24)
    print(f"  • 071A (4번 레포 챔피언) inference complete in {time.time() - t0:.1f}s | BSS: {bss(y_24, p_071):.4f}")
    
    t0 = time.time()
    p_repo1 = run_repo1(val_24)
    print(f"  • Repo 1 (1번 레포 HGB+LGBM) inference complete in {time.time() - t0:.1f}s | BSS: {bss(y_24, p_repo1):.4f}")
    
    t0 = time.time()
    p_repo3 = run_repo3(val_24)
    print(f"  • Repo 3 (3번 레포 Trackman HGB) inference complete in {time.time() - t0:.1f}s | BSS: {bss(y_24, p_repo3):.4f}")
    
    # 2. Correlations
    print("\n[Step 2] Prediction Correlation Matrix:")
    preds_mat = np.column_stack([p_071, p_repo1, p_repo3])
    corr = np.corrcoef(preds_mat, rowvar=False)
    print(f"  • corr(071A, Repo1) = {corr[0,1]:.4f}")
    print(f"  • corr(071A, Repo3) = {corr[0,2]:.4f}")
    print(f"  • corr(Repo1, Repo3) = {corr[1,2]:.4f}")
    
    # 3. Fine-Grained Blend Grid Search
    print("\n[Step 3] Grid Search: 071A + Repo 1 Blends")
    print(f"  {'w(Repo1)':<10} | {'2024 BSS':<10} | {'BSS Gain vs 071A':<18} | {'Bootstrap 95% CI':<28} | {'P(Gain>0)':<10}")
    print("  " + "-" * 75)
    
    best_blend_p = None
    best_blend_name = ""
    best_blend_bss = -9999
    
    for w1 in [0.02, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10, 0.12, 0.15]:
        p_cand = (1.0 - w1) * p_071 + w1 * p_repo1
        cand_bss = bss(y_24, p_cand)
        diff_bss = cand_bss - bss(y_24, p_071)
        mean_g, ci_lo, ci_hi, p_pos = cluster_bootstrap(y_24, p_071, p_cand, clusters, n_boot=500)
        
        ci_str = f"[{ci_lo * 1e5:+7.2f}, {ci_hi * 1e5:+7.2f}]"
        print(f"  w1={w1:<7.2f} | {cand_bss:10.4f} | {diff_bss:+18.4f} | {ci_str:<28} | {p_pos*100:6.1f}%")
        
        if cand_bss > best_blend_bss:
            best_blend_bss = cand_bss
            best_blend_p = p_cand
            best_blend_name = f"071A(0.95) + Repo1(0.05)" if w1==0.05 else f"071A({1-w1:.2f}) + Repo1({w1:.2f})"
            
    print("\n[Step 4] Three-Way Blends: 071A + Repo1 + Repo3")
    for w1 in [0.04, 0.06, 0.08]:
        for w3 in [0.02, 0.04]:
            w0 = 1.0 - w1 - w3
            p_cand = w0 * p_071 + w1 * p_repo1 + w3 * p_repo3
            cand_bss = bss(y_24, p_cand)
            diff_bss = cand_bss - bss(y_24, p_071)
            mean_g, ci_lo, ci_hi, p_pos = cluster_bootstrap(y_24, p_071, p_cand, clusters, n_boot=500)
            ci_str = f"[{ci_lo * 1e5:+7.2f}, {ci_hi * 1e5:+7.2f}]"
            print(f"  w1={w1:.2f}, w3={w3:.2f} (w0={w0:.2f}) | {cand_bss:10.4f} | {diff_bss:+18.4f} | {ci_str:<28} | {p_pos*100:6.1f}%")
            
            if cand_bss > best_blend_bss:
                best_blend_bss = cand_bss
                best_blend_p = p_cand
                best_blend_name = f"071A({w0:.2f}) + Repo1({w1:.2f}) + Repo3({w3:.2f})"

    print("\n" + "=" * 75)
    print(f"🏆 BEST CANDIDATE: {best_blend_name} -> 2024 BSS = {best_blend_bss:.4f} (+{best_blend_bss - bss(y_24, p_071):.4f} gain!)")
    print("=" * 75)

if __name__ == '__main__':
    main()
