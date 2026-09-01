#!/usr/bin/env python3
"""
Train dedicated Platoon Physical Interaction Expert for Candidate 113B.
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
OUT_MODEL_DIR = ROOT / "model/REF4-PLATOON-PHYS-113B/production_package/model"

def main():
    print("=" * 80)
    print("  [113B] TRAINING PLATOON PHYSICAL INTERACTION EXPERT ON 1.47M ROWS  ")
    print("=" * 80)
    t0 = time.time()
    
    train = pd.read_csv(DATA_PATH, low_memory=False)
    y = train["control_success"].to_numpy(float)
    
    p_rate = train["asof_pitcher_success_rate"].fillna(0.523766).to_numpy(float)
    prev1_p = train["asof_pitcher_prev1_game_success_rate"].fillna(0.523766).to_numpy(float)
    p_anchor = np.clip(0.70 * p_rate + 0.30 * prev1_p, 0.05, 0.95)
    target_residual = y - p_anchor
    
    is_p_left = (train["pitcher_hand"].astype(str) == "L").to_numpy(float)
    is_b_left = (train["batter_hand"].astype(str) == "L").to_numpy(float)
    platoon = (is_p_left != is_b_left).astype(float)
    
    fb_rate = train["asof_pitcher_fastball_rate"].fillna(0.50).to_numpy(float)
    brk_rate = train["asof_pitcher_breaking_rate"].fillna(0.30).to_numpy(float)
    off_rate = train["asof_pitcher_offspeed_rate"].fillna(0.20).to_numpy(float)
    
    b = train["balls_before"].fillna(0).to_numpy(float)
    s = train["strikes_before"].fillna(0).to_numpy(float)
    
    X = pd.DataFrame({
        "platoon": platoon,
        "pitcher_hand": train["pitcher_hand"].astype(str),
        "batter_hand": train["batter_hand"].astype(str),
        "platoon_fastball": platoon * fb_rate,
        "platoon_breaking": platoon * brk_rate,
        "platoon_offspeed": platoon * off_rate,
        "count_pressure": (b - s) * (b + s + 1.0) / 7.0,
        "is_2s": (s == 2).astype(float),
        "asof_pitcher_success_rate": p_rate,
        "asof_pitcher_prev1_game_success_rate": prev1_p,
        "asof_batter_middle_rate": train["asof_batter_middle_rate"].fillna(0.523766).to_numpy(float),
        "p_anchor": p_anchor
    })
    
    cat_features = ["pitcher_hand", "batter_hand"]
    
    model = CatBoostRegressor(
        iterations=200,
        depth=5,
        learning_rate=0.03,
        l2_leaf_reg=40.0,
        loss_function="RMSE",
        random_seed=42,
        verbose=50
    )
    model.fit(X, target_residual, cat_features=cat_features)
    
    OUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cbm_path = OUT_MODEL_DIR / "platoon_phys_expert.cbm"
    model.save_model(str(cbm_path))
    print(f"Saved platoon physical model to {cbm_path} in {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
