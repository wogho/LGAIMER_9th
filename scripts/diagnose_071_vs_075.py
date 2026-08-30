#!/usr/bin/env python3
"""Forensic comparison between 071A and 075A on historical and holdout splits."""
import numpy as np, pandas as pd, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = str(ROOT / '.venv-submit/bin/python')

def bss(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-5, 1 - 1e-5)
    r = float(y.mean())
    ref = r * (1.0 - r)
    return 1e5 * (1.0 - np.mean((p - y) ** 2) / ref)

def main():
    print("=" * 70, flush=True)
    print("       FORENSIC DIAGNOSIS: 071A (1092.1879) vs 075A (1091.2133)   ", flush=True)
    print("=" * 70, flush=True)
    
    train = pd.read_csv(ROOT / 'data/train.csv', low_memory=False)
    
    for yr in [2024]:
        val = train.loc[train.season == yr].copy().reset_index(drop=True)
        y = val.control_success.to_numpy()
        is_f = (val.game_type == 'F').to_numpy()
        is_r = (val.game_type == 'R').to_numpy()
        
        print(f"\n[Season {yr} Holdout - Total {len(val):,} rows (1군: {is_r.sum():,}, 2군: {is_f.sum():,})]", flush=True)
        
        # Run 071A
        p71_dir = ROOT / 'model/REF4-ADAPTIVE-CHANNEL-OPT-071A/production_package'
        (p71_dir / 'data').mkdir(parents=True, exist_ok=True)
        val.to_csv(p71_dir / 'data/test.csv', index=False)
        print("  Running 071A inference...", flush=True)
        subprocess.run([PYTHON_BIN, 'script.py'], cwd=str(p71_dir), capture_output=True)
        p71 = pd.read_csv(p71_dir / 'output/submission.csv')['control_success'].to_numpy()
        
        # Run 075A
        p75_dir = ROOT / 'model/REF4-MACRO-LEAP-CHAMPION-075A/audit_verification_sandbox'
        (p75_dir / 'data').mkdir(parents=True, exist_ok=True)
        val.to_csv(p75_dir / 'data/test.csv', index=False)
        print("  Running 075A inference...", flush=True)
        subprocess.run([PYTHON_BIN, 'script.py'], cwd=str(p75_dir), capture_output=True)
        p75 = pd.read_csv(p75_dir / 'output/submission.csv')['control_success'].to_numpy()
        
        bss_71_all = bss(y, p71)
        bss_71_r = bss(y[is_r], p71[is_r])
        bss_71_f = bss(y[is_f], p71[is_f])
        
        bss_75_all = bss(y, p75)
        bss_75_r = bss(y[is_r], p75[is_r])
        bss_75_f = bss(y[is_f], p75[is_f])
        
        print(f"\n  • 071A Overall BSS : {bss_71_all:9.4f} | 1군: {bss_71_r:9.4f} | 2군: {bss_71_f:9.4f}", flush=True)
        print(f"  • 075A Overall BSS : {bss_75_all:9.4f} | 1군: {bss_75_r:9.4f} | 2군: {bss_75_f:9.4f}", flush=True)
        print(f"  • 1군 diff (75 - 71): {bss_75_r - bss_71_r:+9.4f}", flush=True)
        print(f"  • 2군 diff (75 - 71): {bss_75_f - bss_71_f:+9.4f}", flush=True)
        
        diff_r_preds = np.max(np.abs(p75[is_r] - p71[is_r]))
        diff_f_preds = np.max(np.abs(p75[is_f] - p71[is_f]))
        print(f"\n  • 1군 Predictions Max |diff|: {diff_r_preds:.3e}", flush=True)
        print(f"  • 2군 Predictions Max |diff|: {diff_f_preds:.3e}", flush=True)
        print(f"  • 2군 Mean Prob: 071A={np.mean(p71[is_f]):.5f}, 075A={np.mean(p75[is_f]):.5f}, GroundTruth={np.mean(y[is_f]):.5f}", flush=True)
        print(f"  • 2군 Std Prob : 071A={np.std(p71[is_f]):.5f}, 075A={np.std(p75[is_f]):.5f}, GroundTruth={np.std(y[is_f]):.5f}", flush=True)
        
        # Grid Search blending 071A and 075A
        print("\n  === Blend Grid Search: w * 075A + (1 - w) * 071A ===", flush=True)
        for w in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
            p_blend = w * p75 + (1 - w) * p71
            score = bss(y, p_blend)
            score_f = bss(y[is_f], p_blend[is_f])
            print(f"    w(075A)={w:4.2f} -> Overall BSS: {score:9.4f} (2군: {score_f:9.4f})", flush=True)

if __name__ == '__main__':
    main()
