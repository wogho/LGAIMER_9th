#!/usr/bin/env python3
"""
Test 10th repo's true additive nested deviations on 113A champion.
"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/train.csv"

def main():
    print("=" * 80)
    print("  TESTING DIRECT ADDITIVE 10TH REPO DEVIATIONS ON 113A BASE  ")
    print("=" * 80)
    
    train = pd.read_csv(DATA_PATH, low_memory=False)
    val_2024 = train[train["season"] == 2024].head(15000).copy().reset_index(drop=True)
    y_true = val_2024["control_success"].to_numpy(float)
    ref_brier = brier_score_loss(y_true, np.full_like(y_true, 0.523766))
    
    # Load 10th repo v66 nested deviations
    v66_conf = json.loads((ROOT / "model/frozen_10th_tables/v66_nested_deviations.json").read_text())
    
    pitcher = val_2024["pitcher_id"].astype(str)
    b_hand = val_2024["batter_hand"].astype(str)
    p_hand_key = pitcher + "|" + b_hand
    s = val_2024["strikes_before"].fillna(0).to_numpy(float)
    b = val_2024["balls_before"].fillna(0).to_numpy(float)
    adv = (s > b).astype(int).astype(str)
    p_hand_adv_key = p_hand_key + "|" + adv
    n_run = val_2024["num_runners_on"].fillna(0).to_numpy(float)
    run_gate = (n_run > 0).astype(int).astype(str)
    run_key = p_hand_key + "|" + run_gate
    
    keys_map = {
        "platoon": p_hand_key,
        "advantage": p_hand_adv_key,
        "runner": run_key
    }
    
    v66_delta = np.zeros(len(val_2024), dtype=float)
    for ax in v66_conf["axes"]:
        name = ax["name"]
        lut = dict(zip(ax["keys"], ax["deltas"]))
        w = float(ax["weight"])
        vals = keys_map[name].map(lut).fillna(0.0).to_numpy(float)
        v66_delta += w * vals
        
    print(f"v66_delta mean: {v66_delta.mean():.6f}, min: {v66_delta.min():.4f}, max: {v66_delta.max():.4f}")
    
    # Check baseline 0.523766 + v66_delta
    p_naive = np.full_like(y_true, 0.523766)
    p_v66 = np.clip(p_naive + v66_delta, 0.02, 0.98)
    
    bs_naive = brier_score_loss(y_true, p_naive)
    bs_v66 = brier_score_loss(y_true, p_v66)
    bss_v66 = (1.0 - (bs_v66 / ref_brier)) * 100.0
    
    print(f"Naive Brier: {bs_naive:.6f}")
    print(f"Naive + v66 Delta Brier: {bs_v66:.6f} | BSS: {bss_v66:.4f}% (+{bss_v66:.4f}%)")
    
if __name__ == "__main__":
    main()
