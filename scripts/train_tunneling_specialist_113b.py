#!/usr/bin/env python3
"""
Train dedicated Pitch-Tunneling & Count-Tension Dual Specialist for Candidate 113B.
Trains on 1.47M rows from data/train.csv.
"""
import json
import time
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/train.csv"
OUT_MODEL_DIR = ROOT / "model/REF4-TUNNELING-GBDT-113B/production_package/model"

def main():
    print("=" * 80)
    print("  [113B] TRAINING PITCH-TUNNELING & COUNT-TENSION SPECIALISTS ON 1.47M ROWS  ")
    print("=" * 80)
    t0 = time.time()
    
    print("1. Loading train dataset...")
    train = pd.read_csv(DATA_PATH, low_memory=False)
    N = len(train)
    print(f"   Loaded {N:,} rows.")
    
    y = train["control_success"].to_numpy(float)
    
    # Extract Physical & Count Dynamics Features
    b = train["balls_before"].fillna(0).astype(int).to_numpy()
    s = train["strikes_before"].fillna(0).astype(int).to_numpy()
    count_code = b * 3 + s
    
    velo = train["release_speed"].fillna(140.0).to_numpy(float) if "release_speed" in train.columns else np.full(N, 140.0)
    spin = train["release_spin_rate"].fillna(2200.0).to_numpy(float) if "release_spin_rate" in train.columns else np.full(N, 2200.0)
    pfx_x = train["pfx_x"].fillna(0.0).to_numpy(float) if "pfx_x" in train.columns else np.zeros(N)
    pfx_z = train["pfx_z"].fillna(0.0).to_numpy(float) if "pfx_z" in train.columns else np.zeros(N)
    rel_x = train["release_pos_x"].fillna(0.0).to_numpy(float) if "release_pos_x" in train.columns else np.zeros(N)
    rel_z = train["release_pos_z"].fillna(1.8).to_numpy(float) if "release_pos_z" in train.columns else np.full(N, 1.8)
    
    movement_mag = np.sqrt(pfx_x**2 + pfx_z**2)
    spin_eff = np.clip(movement_mag / (velo + 1e-5), 0, 1.0)
    
    n_p = train["asof_pitcher_n"].fillna(0).to_numpy(float)
    n_b = train["asof_batter_n"].fillna(0).to_numpy(float)
    p_rate = train["asof_pitcher_success_rate"].fillna(0.523766).to_numpy(float)
    prev1_p = train["asof_pitcher_prev1_game_success_rate"].fillna(0.523766).to_numpy(float)
    b_mid = train["asof_batter_middle_rate"].fillna(0.523766).to_numpy(float)
    b_rate = train["asof_batter_success_rate"].fillna(0.523766).to_numpy(float)
    
    fb_rate = train["asof_pitcher_fastball_rate"].fillna(0.50).to_numpy(float)
    brk_rate = train["asof_pitcher_breaking_rate"].fillna(0.30).to_numpy(float)
    off_rate = train["asof_pitcher_offspeed_rate"].fillna(0.20).to_numpy(float)
    
    is_p_left = (train["pitcher_hand"].astype(str) == "L").to_numpy(float)
    is_b_left = (train["batter_hand"].astype(str) == "L").to_numpy(float)
    platoon = (is_p_left != is_b_left).astype(float)
    
    li = train["li"].fillna(0.98).to_numpy(float)
    inn = train["inning"].fillna(1).to_numpy(float)
    
    # Construct 18-Feature Matrix
    X = pd.DataFrame({
        "count_code": count_code,
        "balls": b,
        "strikes": s,
        "count_pressure": (b - s) * (b + s + 1.0) / 7.0,
        "is_2s": (s == 2).astype(float),
        "is_3b": (b == 3).astype(float),
        "asof_pitcher_success_rate": p_rate,
        "asof_pitcher_prev1_game_success_rate": prev1_p,
        "pitcher_recent_drift": (prev1_p - p_rate) * (n_p / (n_p + 15.0)),
        "asof_batter_middle_rate": b_mid,
        "asof_batter_success_rate": b_rate,
        "platoon": platoon,
        "platoon_fastball": platoon * fb_rate,
        "platoon_breaking": platoon * brk_rate,
        "platoon_offspeed": platoon * off_rate,
        "phys_velo": velo,
        "phys_movement_mag": movement_mag,
        "phys_spin_eff": spin_eff,
        "li": li,
        "inning": inn
    })
    
    OUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    # 2. Train LightGBM Tunneling Booster
    print("2. Fitting LightGBM Tunneling Booster...")
    dtrain = lgb.Dataset(X, label=y)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.035,
        "num_leaves": 35,
        "min_child_samples": 40,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": 42
    }
    lgb_booster = lgb.train(params, dtrain, num_boost_round=220)
    lgb_path = OUT_MODEL_DIR / "tunneling_lgb.txt"
    lgb_booster.save_model(str(lgb_path))
    print(f"   Saved LightGBM model to {lgb_path}")
    
    # 3. Train CatBoost Tunneling Regressor
    print("3. Fitting CatBoost Tunneling Regressor...")
    cb_model = CatBoostRegressor(
        iterations=220,
        depth=5,
        learning_rate=0.035,
        l2_leaf_reg=35.0,
        loss_function="RMSE",
        random_seed=42,
        verbose=50
    )
    cb_model.fit(X, y)
    cb_path = OUT_MODEL_DIR / "tunneling_cb.cbm"
    cb_model.save_model(str(cb_path))
    print(f"   Saved CatBoost model to {cb_path}")
    
    print(f"4. Tunneling Specialist Training Completed in {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
