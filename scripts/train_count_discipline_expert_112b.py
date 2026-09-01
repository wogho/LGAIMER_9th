#!/usr/bin/env python3
"""
Real Training of Micro Game-Theory Count-Discipline Expert Models for Candidate 112B.
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
OUT_MODEL_DIR = ROOT / "model/REF4-MICRO-GAME-THEORY-112B/production_package/model"

def main():
    print("=" * 80)
    print("  [112B-REAL] TRAINING COUNT-DISCIPLINE EXPERT MODELS ON 1.47M ROWS  ")
    print("=" * 80)
    t0 = time.time()
    
    print("1. Loading train dataset...")
    train = pd.read_csv(DATA_PATH, low_memory=False)
    N = len(train)
    print(f"   Loaded {N:,} rows.")
    
    # Target and features
    y = train["control_success"].to_numpy(float)
    
    b = train["balls_before"].fillna(0).astype(int).to_numpy()
    s = train["strikes_before"].fillna(0).astype(int).to_numpy()
    count_code = b * 3 + s  # 0 to 11
    
    n_b = train["asof_batter_n"].fillna(0).to_numpy(float)
    b_rate = train["asof_batter_success_rate"].fillna(0.523766).to_numpy(float)
    b_mid = train["asof_batter_middle_rate"].fillna(0.523766).to_numpy(float)
    
    n_p = train["asof_pitcher_n"].fillna(0).to_numpy(float)
    p_rate = train["asof_pitcher_success_rate"].fillna(0.523766).to_numpy(float)
    fb_rate = train["asof_pitcher_fastball_rate"].fillna(0.50).to_numpy(float)
    brk_rate = train["asof_pitcher_breaking_rate"].fillna(0.30).to_numpy(float)
    off_rate = train["asof_pitcher_offspeed_rate"].fillna(0.20).to_numpy(float)
    
    li = train["li"].fillna(0.98).to_numpy(float)
    inn = train["inning"].fillna(1).to_numpy(float)
    
    is_p_left = (train["pitcher_hand"].astype(str) == "L").to_numpy(float)
    is_b_left = (train["batter_hand"].astype(str) == "L").to_numpy(float)
    platoon = (is_p_left != is_b_left).astype(float)
    
    X = pd.DataFrame({
        "count_code": count_code,
        "balls": b,
        "strikes": s,
        "count_pressure": (b - s) * (b + s + 1.0) / 7.0,
        "is_2s": (s == 2).astype(float),
        "is_3b": (b == 3).astype(float),
        "asof_pitcher_success_rate": p_rate,
        "asof_pitcher_fastball_rate": fb_rate,
        "asof_pitcher_breaking_rate": brk_rate,
        "asof_pitcher_offspeed_rate": off_rate,
        "asof_batter_middle_rate": b_mid,
        "asof_batter_success_rate": b_rate,
        "batter_discipline_pressure": (b_mid - 0.523766) * (n_b / (n_b + 40.0)) * ((3.0 - s) / 3.0),
        "platoon": platoon,
        "li": li,
        "inning": inn
    })
    
    # 2. Train LightGBM Expert
    print("2. Fitting LightGBM Count-Discipline Booster...")
    dtrain = lgb.Dataset(X, label=y)
    params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "learning_rate": 0.04,
        "num_leaves": 31,
        "min_child_samples": 50,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "verbosity": -1,
        "seed": 42
    }
    lgb_booster = lgb.train(params, dtrain, num_boost_round=180)
    
    OUT_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    lgb_path = OUT_MODEL_DIR / "count_discipline_lgb.txt"
    lgb_booster.save_model(str(lgb_path))
    print(f"   Saved LightGBM model to {lgb_path}")
    
    # 3. Train CatBoost Expert
    print("3. Fitting CatBoost Count-Discipline Regressor...")
    cb_model = CatBoostRegressor(
        iterations=200,
        depth=5,
        learning_rate=0.04,
        l2_leaf_reg=30.0,
        loss_function="RMSE",
        random_seed=42,
        verbose=50
    )
    cb_model.fit(X, y)
    cb_path = OUT_MODEL_DIR / "count_discipline_cb.cbm"
    cb_model.save_model(str(cb_path))
    print(f"   Saved CatBoost model to {cb_path}")
    
    print(f"4. Real Count-Discipline Training Completed in {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
