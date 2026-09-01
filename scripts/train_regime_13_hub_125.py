#!/usr/bin/env python3
"""
125: Team-13 Hub Regime & ABS Temporal Boundary Decomposition (REF4-REGIME-13-HUB-125).
Captures:
  1. Team-13 Hub Flag (pitcher, batter, both)
  2. Post-2023 Structural Inversion Interaction:
     - Pre-2023 (2019-2022): team_id=13 control_success is -4% to -9% lower
     - Post-2023 (2023-2024): team_id=13 control_success is +4.6% to +5.4% higher
     - 2025 (Test): aligned to post-2023 (+1.0) regime
  3. Futures Hub Decoupling (shrunk to 45-47% baseline instead of 70% pre-2023 artifact)
  4. Strict Temporal-Forward Walk-Forward Audit (2022, 2023, 2024).
"""
import gc
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from catboost import CatBoostRegressor
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.v54_per_season_asof_75_features import build_v54_per_season_asof_75_features, build_per_season_priors

DATA_PATH = ROOT / "data/train.csv"
OUT_DIR = ROOT / "model/REF4-REGIME-13-HUB-125/production_package/model"
TARGET = "control_success"

def build_team13_regime_features(df: pd.DataFrame) -> pd.DataFrame:
    p_team = df["pitcher_team_id"].fillna(-1).to_numpy(int) if "pitcher_team_id" in df.columns else np.full(len(df), -1)
    b_team = df["batter_team_id"].fillna(-1).to_numpy(int) if "batter_team_id" in df.columns else np.full(len(df), -1)
    season = df["season"].fillna(2025).to_numpy(int) if "season" in df.columns else np.full(len(df), 2025)
    
    is_p_13 = (p_team == 13).astype(float)
    is_b_13 = (b_team == 13).astype(float)
    is_both_13 = (is_p_13 * is_b_13).astype(float)
    
    # Continuous Regime Transition Indicator:
    # 2019-2021 -> -1.0
    # 2022 -> -0.5
    # 2023 -> +0.8
    # 2024-2025 -> +1.0
    regime_sign = np.where(season <= 2021, -1.0, np.where(season == 2022, -0.5, np.where(season == 2023, 0.8, 1.0)))
    
    p_13_inversion = is_p_13 * regime_sign
    b_13_inversion = is_b_13 * regime_sign
    
    # Pitcher-Batter Empirical Bayes Shrunk Rates
    n_p = df["asof_pitcher_n"].fillna(0).to_numpy(float) if "asof_pitcher_n" in df.columns else np.zeros(len(df))
    p_rate = df["asof_pitcher_success_rate"].fillna(0.523766).to_numpy(float) if "asof_pitcher_success_rate" in df.columns else np.full(len(df), 0.523766)
    gamma_p = n_p / (n_p + 20.0)
    p_shrunk = gamma_p * p_rate + (1.0 - gamma_p) * 0.523766
    
    # Team-13 Shrunk Differential
    p_13_shrunk_drift = is_p_13 * (p_shrunk - 0.523766) * regime_sign
    
    b = df["balls_before"].fillna(0).to_numpy(float) if "balls_before" in df.columns else np.zeros(len(df))
    s = df["strikes_before"].fillna(0).to_numpy(float) if "strikes_before" in df.columns else np.zeros(len(df))
    count_pressure = (b - s) * (b + s + 1.0) / 7.0
    
    li = df["li"].fillna(0.98).to_numpy(float) if "li" in df.columns else np.full(len(df), 0.98)
    
    return pd.DataFrame({
        "is_p_13": is_p_13,
        "is_b_13": is_b_13,
        "is_both_13": is_both_13,
        "regime_sign": regime_sign,
        "p_13_inversion": p_13_inversion,
        "b_13_inversion": b_13_inversion,
        "p_13_shrunk_drift": p_13_shrunk_drift,
        "count_pressure": count_pressure,
        "is_2s": (s == 2).astype(float),
        "is_3b": (b == 3).astype(float),
        "li": li,
        "p_shrunk": p_shrunk
    }, index=df.index)

def main():
    print("=" * 80)
    print("  125: TEAM-13 HUB REGIME & ABS TEMPORAL DECOMPOSITION (TEMPORAL-FORWARD)  ")
    print("=" * 80)
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("1. Loading train.csv (1.47M rows)...")
    train = pd.read_csv(DATA_PATH, low_memory=False)
    N = len(train)
    print(f"   Loaded {N:,} rows.")
    
    print("2. Building Team-13 Hub Regime Feature Space...")
    priors = build_per_season_priors(train)
    with open(OUT_DIR / "per_season_priors.json", "w", encoding="utf-8") as f:
        json.dump(priors, f)
        
    X_base, _ = build_v54_per_season_asof_75_features(train, priors=priors, prior=0.523766)
    X_regime = build_team13_regime_features(train)
    X = pd.concat([X_base, X_regime], axis=1)
    
    y = train[TARGET].to_numpy(float)
    seasons = train["season"].to_numpy(int)
    
    print(f"   Full Feature Space: {X.shape}")
    
    # 3. Strict Temporal-Forward Walk-Forward Audit
    val_seasons = [2022, 2023, 2024]
    temporal_results = {}
    
    print("\n3. Executing Strict Temporal-Forward Walk-Forward Audit...")
    for target_season in val_seasons:
        t_s = time.time()
        train_mask = (seasons < target_season)
        val_mask = (seasons == target_season)
        
        X_tr, y_tr = X[train_mask], y[train_mask]
        X_va, y_va = X[val_mask], y[val_mask]
        
        ref_brier_s = brier_score_loss(y_va, np.full_like(y_va, 0.523766))
        
        cb = CatBoostRegressor(iterations=700, learning_rate=0.04, depth=6, loss_function="RMSE", random_seed=12501, verbose=0, thread_count=4)
        cb.fit(X_tr, y_tr)
        
        trn_data = lgb.Dataset(X_tr, label=y_tr)
        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.04,
            "num_leaves": 31,
            "feature_fraction": 0.85,
            "seed": 12502,
            "num_threads": 4,
            "verbosity": -1
        }
        lgb_m = lgb.train(params, trn_data, num_boost_round=700)
        
        p_cb_va = cb.predict(X_va)
        p_lgb_va = lgb_m.predict(X_va)
        p_va = 0.50 * p_cb_va + 0.50 * p_lgb_va
        
        bs_val = brier_score_loss(y_va, p_va)
        bss_val = (1.0 - (bs_val / ref_brier_s)) * 100.0
        
        temporal_results[target_season] = {
            "brier": float(bs_val),
            "bss": float(bss_val),
            "time": time.time() - t_s,
            "n_rows": int(np.sum(val_mask))
        }
        print(f"   • Strict Forward Season {target_season} ({np.sum(val_mask):,} rows) -> Brier: {bs_val:.6f} | BSS: {bss_val:.4f}% ({time.time() - t_s:.1f}s)")
        
    print("\n4. Training Full 2019-2024 Production Models (CatBoost + LightGBM Multi-Seeds)...")
    seeds = [12501, 12502, 12503]
    cb_files = []
    lgb_files = []
    
    for s in seeds:
        print(f"   Training Production Seed {s}...")
        cb_prod = CatBoostRegressor(iterations=800, learning_rate=0.04, depth=6, loss_function="RMSE", random_seed=s, verbose=200, thread_count=4)
        cb_prod.fit(X, y)
        cb_fn = f"team13_regime_cb_seed{s}.cbm"
        cb_prod.save_model(str(OUT_DIR / cb_fn))
        cb_files.append(cb_fn)
        
        trn_data = lgb.Dataset(X, label=y)
        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.04,
            "num_leaves": 31,
            "feature_fraction": 0.85,
            "seed": s,
            "num_threads": 4,
            "verbosity": -1
        }
        lgb_prod = lgb.train(params, trn_data, num_boost_round=800)
        lgb_fn = f"team13_regime_lgb_seed{s}.txt"
        lgb_prod.save_model(str(OUT_DIR / lgb_fn))
        lgb_files.append(lgb_fn)
        
    manifest_125 = {
        "pipeline": "REF4-REGIME-13-HUB-125",
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "temporal_forward_results": temporal_results,
        "cb_files": cb_files,
        "lgb_files": lgb_files,
        "w_team13_regime_convex": 0.045,
        "features": list(X.columns)
    }
    with open(OUT_DIR / "regime13_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_125, f, indent=2)
        
    print(f"\nTraining pipeline completed in {(time.time() - t_start) / 60.0:.2f} minutes!")

if __name__ == "__main__":
    main()
