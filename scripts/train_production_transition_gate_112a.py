#!/usr/bin/env python3
"""
Real Training of League Transition Gate for Candidate 112A.
Trains on 1.47M rows from data/train.csv.
"""
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/train.csv"
OUT_MODEL_DIR = ROOT / "model/REF4-CHAMPION-RESTORE-112A/production_package/model"

def main():
    print("=" * 80)
    print("  [112A-REAL] TRAINING LEAGUE TRANSITION GATE ON 1.47M ROWS  ")
    print("=" * 80)
    t0 = time.time()
    
    print("1. Loading train dataset...")
    train = pd.read_csv(DATA_PATH, low_memory=False)
    N = len(train)
    print(f"   Loaded {N:,} rows.")
    
    # Identify league transition state for each pitcher
    print("2. Constructing league transition features...")
    train = train.sort_values(["season", "game_month", "game_dayofweek", "pitcher_id"]).reset_index(drop=True)
    
    p_rate = train["asof_pitcher_success_rate"].fillna(0.523766).to_numpy(float)
    prev1_p = train["asof_pitcher_prev1_game_success_rate"].fillna(0.523766).to_numpy(float)
    p_anchor = np.clip(0.70 * p_rate + 0.30 * prev1_p, 0.05, 0.95)
    
    y = train["control_success"].to_numpy(float)
    target_residual = y - p_anchor
    
    # Feature engineering for transition gate
    game_type = train["game_type"].astype(str)
    n_p = train["asof_pitcher_n"].fillna(0).to_numpy(float)
    li = train["li"].fillna(0.98).to_numpy(float)
    b = train["balls_before"].fillna(0).to_numpy(float)
    s = train["strikes_before"].fillna(0).to_numpy(float)
    
    X = pd.DataFrame({
        "game_type": game_type,
        "pitcher_hand": train["pitcher_hand"].astype(str),
        "batter_hand": train["batter_hand"].astype(str),
        "log_pitcher_n": np.log1p(n_p),
        "asof_pitcher_success_rate": p_rate,
        "asof_pitcher_prev1_game_success_rate": prev1_p,
        "momentum_drift": prev1_p - p_rate,
        "li": li,
        "count_pressure": (b - s) / 3.0,
        "p_anchor": p_anchor
    })
    
    cat_features = ["game_type", "pitcher_hand", "batter_hand"]
    
    print("3. Fitting CatBoost Transition Gate Regressor...")
    model = CatBoostRegressor(
        iterations=200,
        depth=4,
        learning_rate=0.03,
        l2_leaf_reg=50.0,
        loss_function="RMSE",
        random_seed=42,
        verbose=50
    )
    model.fit(X, target_residual, cat_features=cat_features)
    
    # Save model
    OUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cbm_path = OUT_MODEL_DIR / "transition_gate.cbm"
    model.save_model(str(cbm_path))
    print(f"4. Saved real transition gate to {cbm_path} ({cbm_path.stat().st_size / 1024:.1f} KB)")
    print(f"   Completed in {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
