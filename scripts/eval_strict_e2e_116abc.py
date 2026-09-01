#!/usr/bin/env python3
"""
Strict Full End-to-End Evaluation on 2024 Holdout Dataset.
Evaluates 113A Champion (1121.90 LB) vs 116A vs 116B vs 116C.
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
    
    sandbox = Path(tempfile.mkdtemp(prefix="strict_116_"))
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
        return bss, bs, elapsed, p_pred.mean()
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)

def main():
    print("=" * 80)
    print("  STRICT REAL END-TO-END EVALUATION: 113A (1121.90 LB) vs 116A vs 116B vs 116C  ")
    print("=" * 80)
    
    train = pd.read_csv(DATA_PATH, low_memory=False)
    val_2024 = train[train["season"] == 2024].copy().reset_index(drop=True)
    N_EVAL = 15000
    print(f"Loaded 2024 Season ({len(val_2024):,} rows). Evaluating End-to-End on {N_EVAL:,} rows...\n")
    
    candidates = [
        ("113A Champion Base", ROOT / "model/REF4-DISJOINT-EB-113A/production_package"),
        ("116A Direct v66 Additive", ROOT / "model/REF4-DIRECT-V66-116A/production_package"),
        ("116B Exposure + Sigmoid", ROOT / "model/REF4-V66-EXPOSURE-SIGMOID-116B/production_package"),
        ("116C Tunneling Specialist", ROOT / "model/REF4-V66-TUNNELING-116C/production_package"),
    ]
    
    results = []
    for name, path in candidates:
        print(f"Evaluating {name}...")
        bss, bs, elap, p_mean = evaluate_production_package(path, val_2024, n_rows=N_EVAL)
        results.append({"name": name, "bss": bss, "brier": bs, "time": elap, "p_mean": p_mean})
        print(f"  -> BSS: {bss:.4f}% | Brier: {bs:.6f} | Mean P: {p_mean:.4f} | Time: {elap:.2f}s")
        
    print("\n" + "=" * 80)
    print("  FINAL 116 SCOREBOARD (TRUE END-TO-END HOLD-OUT EVALUATION)  ")
    print("=" * 80)
    print(f"{'Candidate':<28} | {'Brier Score':<12} | {'BSS (%)':<10} | {'Delta vs 113A':<14} | {'Mean P':<8} | {'Speed'}")
    print("-" * 80)
    base_bss = results[0]["bss"]
    for r in results:
        delta = r["bss"] - base_bss
        print(f"{r['name']:<28} | {r['brier']:<12.6f} | {r['bss']:<10.4f} | {delta:<+14.4f} | {r['p_mean']:<8.4f} | {r['time']:.1f}s")
    print("=" * 80)

if __name__ == "__main__":
    main()
