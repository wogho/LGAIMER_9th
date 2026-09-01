#!/usr/bin/env python3
"""
Train dedicated Asymmetric 3-Tier Matchup Empirical Bayes Expert for Candidate 113A.
Trains on 1.47M rows from data/train.csv.
"""
import json
import pickle
import time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/train.csv"
OUT_MODEL_DIR = ROOT / "model/REF4-DISJOINT-EB-113A/production_package/model"

def main():
    print("=" * 80)
    print("  [113A] TRAINING DISJOINT MATCHUP EMPIRICAL BAYES ON 1.47M ROWS  ")
    print("=" * 80)
    t0 = time.time()
    
    print("1. Loading train dataset...")
    train = pd.read_csv(DATA_PATH, low_memory=False)
    N = len(train)
    print(f"   Loaded {N:,} rows.")
    
    y = train["control_success"].to_numpy(float)
    
    # 3-Tier Shrinkage calculations
    n_p = train["asof_pitcher_n"].fillna(0).to_numpy(float)
    n_b = train["asof_batter_n"].fillna(0).to_numpy(float)
    p_rate = train["asof_pitcher_success_rate"].fillna(0.523766).to_numpy(float)
    prev1_p = train["asof_pitcher_prev1_game_success_rate"].fillna(0.523766).to_numpy(float)
    b_rate = train["asof_batter_success_rate"].fillna(0.523766).to_numpy(float)
    b_mid = train["asof_batter_middle_rate"].fillna(0.523766).to_numpy(float)
    
    gamma_p = n_p / (n_p + 20.0)
    gamma_b = n_b / (n_b + 40.0)
    p_shrunk = gamma_p * p_rate + (1.0 - gamma_p) * 0.523766
    b_shrunk = gamma_b * (1.0 - b_rate) + (1.0 - gamma_b) * 0.523766
    
    is_p_left = (train["pitcher_hand"].astype(str) == "L").to_numpy(float)
    is_b_left = (train["batter_hand"].astype(str) == "L").to_numpy(float)
    platoon = (is_p_left != is_b_left).astype(float)
    
    b = train["balls_before"].fillna(0).to_numpy(float)
    s = train["strikes_before"].fillna(0).to_numpy(float)
    li = train["li"].fillna(0.98).to_numpy(float)
    
    # Construct EB features
    X = pd.DataFrame({
        "eb_pitcher_shrunk": p_shrunk,
        "eb_batter_shrunk": b_shrunk,
        "eb_form_gap": prev1_p - (1.0 - b_rate),
        "gamma_p": gamma_p,
        "gamma_b": gamma_b,
        "platoon": platoon,
        "count_pressure": (b - s) * (b + s + 1.0) / 7.0,
        "is_2s": (s == 2).astype(float),
        "is_3b": (b == 3).astype(float),
        "asof_pitcher_success_rate": p_rate,
        "asof_pitcher_prev1_game_success_rate": prev1_p,
        "asof_batter_middle_rate": b_mid,
        "li": li
    })
    
    p_anchor = np.clip(0.70 * p_rate + 0.30 * prev1_p, 0.05, 0.95)
    target_residual = y - p_anchor
    
    OUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    print("2. Fitting CatBoost Disjoint EB Regressor...")
    cb_eb = CatBoostRegressor(
        iterations=220,
        depth=5,
        learning_rate=0.035,
        l2_leaf_reg=30.0,
        loss_function="RMSE",
        random_seed=42,
        verbose=50
    )
    cb_eb.fit(X, target_residual)
    cb_path = OUT_MODEL_DIR / "disjoint_eb_cb.cbm"
    cb_eb.save_model(str(cb_path))
    print(f"   Saved Disjoint EB model to {cb_path}")
    print(f"3. Disjoint EB Training Completed in {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
