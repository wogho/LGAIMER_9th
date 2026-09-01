#!/usr/bin/env python3
"""
Full End-to-End Subprocess Evaluation on 2024 Holdout Dataset.
Evaluates 109C vs 112A vs 112B vs 112C.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/train.csv"

def evaluate_production_package(pkg_path: Path, val_df: pd.DataFrame, n_rows: int = 15000):
    val_sample = val_df.head(n_rows).copy().reset_index(drop=True)
    y_true = val_sample["control_success"].to_numpy(float)
    ref_brier = brier_score_loss(y_true, np.full_like(y_true, 0.523766))
    
    sandbox = Path(tempfile.mkdtemp(prefix="strict_e2e_"))
    try:
        pkg_dir = sandbox / "pkg"
        shutil.copytree(pkg_path, pkg_dir)
        data_dir = pkg_dir / "data"
        out_dir = pkg_dir / "output"
        data_dir.mkdir(exist_ok=True)
        out_dir.mkdir(exist_ok=True)
        
        test_in = val_sample.drop(columns=["control_success"], errors="ignore").copy()
        test_in["row_id"] = [f"r_{i}" for i in range(len(test_in))]
        test_in.to_csv(data_dir / "test.csv", index=False)
        
        t0 = time.time()
        res = subprocess.run([ROOT / ".venv/bin/python", "script.py"], cwd=str(pkg_dir), capture_output=True, text=True)
        elapsed = time.time() - t0
        
        if res.returncode != 0:
            print(f"Error in {pkg_path.name}:", res.stderr)
            return None, None, elapsed
            
        sub = pd.read_csv(out_dir / "submission.csv")
        p_pred = sub["control_success"].to_numpy(float)
        bs = brier_score_loss(y_true, p_pred)
        bss = (1.0 - (bs / ref_brier)) * 100.0
        return bss, bs, elapsed
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)

def main():
    print("=" * 80)
    print("  STRICT REAL END-TO-END EVALUATION: 109C vs 112A vs 112B vs 112C  ")
    print("=" * 80)
    
    train = pd.read_csv(DATA_PATH, low_memory=False)
    val_2024 = train[train["season"] == 2024].copy().reset_index(drop=True)
    N_EVAL = 15000
    print(f"Loaded 2024 Season ({len(val_2024):,} rows). Evaluating End-to-End on {N_EVAL:,} rows...\n")
    
    candidates = [
        ("109C Baseline", ROOT / "model/REF4-SUPER-ENSEMBLE-109C/production_package"),
        ("112A Champion-Restore", ROOT / "model/REF4-CHAMPION-RESTORE-112A/production_package"),
        ("112B Micro Game-Theory", ROOT / "model/REF4-MICRO-GAME-THEORY-112B/production_package"),
        ("112C SLSQP Simplex Brier", ROOT / "model/REF4-DIRECT-BRIER-SIMPLEX-112C/production_package"),
    ]
    
    results = []
    for name, path in candidates:
        print(f"Evaluating {name}...")
        bss, bs, elap = evaluate_production_package(path, val_2024, n_rows=N_EVAL)
        results.append({"name": name, "bss": bss, "brier": bs, "time": elap})
        print(f"  -> BSS: {bss:.4f}% | Brier: {bs:.6f} | Time: {elap:.2f}s")
        
    print("\n" + "=" * 80)
    print("  FINAL COMPARISON SCOREBOARD (TRUE END-TO-END PIPELINE)  ")
    print("=" * 80)
    print(f"{'Candidate':<26} | {'Brier Score':<12} | {'BSS (%)':<10} | {'Delta vs 109C':<14} | {'Speed'}")
    print("-" * 80)
    base_bss = results[0]["bss"]
    for r in results:
        delta = r["bss"] - base_bss
        print(f"{r['name']:<26} | {r['brier']:<12.6f} | {r['bss']:<10.4f} | {delta:<+14.4f} | {r['time']:.1f}s")
    print("=" * 80)

if __name__ == "__main__":
    main()
