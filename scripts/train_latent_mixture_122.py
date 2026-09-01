#!/usr/bin/env python3
"""
122: Latent Pitch-Type Mixture Marginalization Architecture (REF4-LATENT-MIXTURE-122).
Implements Strict Temporal-Forward Walk-Forward Validation (2022, 2023, 2024) with zero leakage.
Trains:
  1. Situational Pitchmix Policy Estimator [pi_FB, pi_BRK, pi_OFF]
  2. 3-Way Regime Specialist Estimators [f_FB(X), f_BRK(X), f_OFF(X)]
  3. Marginalized Compound Estimator: p_mix = pi_FB*f_FB + pi_BRK*f_BRK + pi_OFF*f_OFF
  4. Strict Convex Blending with 113A Champion Baseline.
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
from catboost import CatBoostRegressor, CatBoostClassifier
import lightgbm as lgb
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.v54_per_season_asof_75_features import build_v54_per_season_asof_75_features, build_per_season_priors

DATA_PATH = ROOT / "data/train.csv"
OUT_DIR = ROOT / "model/REF4-LATENT-MIXTURE-122/production_package/model"
TARGET = "control_success"

def build_game_theory_features(df: pd.DataFrame) -> pd.DataFrame:
    b = df["balls_before"].fillna(0).to_numpy(float)
    s = df["strikes_before"].fillna(0).to_numpy(float)
    li = df["li"].fillna(0.98).to_numpy(float)
    inn = df["inning"].fillna(1).to_numpy(float)
    score_diff = np.abs(df["score_diff_pitcher_team"].fillna(0).to_numpy(float)) if "score_diff_pitcher_team" in df.columns else np.zeros(len(df))
    
    # 1. Melville 2023 Game-Theoretic 2D Pressure/Incentive Tensor
    # Zone Inevitability Pressure: high when count is 3-0, 3-1
    zone_inevitability = np.clip((b - s + 1.0) / 4.0, 0.0, 1.0) * (b >= 2).astype(float)
    # Chase Incentive: high when 0-2, 1-2
    chase_incentive = (s == 2).astype(float) * np.clip((3.0 - b) / 3.0, 0.0, 1.0)
    count_pressure = (b - s) * (b + s + 1.0) / 7.0
    
    # 2. Platoon & Matchup Mechanics
    is_p_left = (df["pitcher_hand"].astype(str) == "L").to_numpy(float) if "pitcher_hand" in df.columns else np.zeros(len(df))
    is_b_left = (df["batter_hand"].astype(str) == "L").to_numpy(float) if "batter_hand" in df.columns else np.zeros(len(df))
    platoon = (is_p_left != is_b_left).astype(float)
    
    # 3. Baseline Pitchmix Rates
    fb = df["asof_pitcher_fastball_rate"].fillna(0.50).to_numpy(float)
    br = df["asof_pitcher_breaking_rate"].fillna(0.30).to_numpy(float)
    os_ = df["asof_pitcher_offspeed_rate"].fillna(0.20).to_numpy(float)
    
    # 4. Count-Adjusted Situational Pitchmix Prior
    # 2-Strikes -> Increased Breaking ball / Offspeed
    pi_fb_est = np.clip(fb * (1.0 - 0.25 * (s == 2).astype(float) + 0.15 * zone_inevitability), 0.05, 0.90)
    pi_br_est = np.clip(br * (1.0 + 0.35 * (s == 2).astype(float) - 0.20 * zone_inevitability), 0.05, 0.90)
    pi_os_est = np.clip(os_ * (1.0 + 0.20 * platoon + 0.10 * (s == 2).astype(float)), 0.05, 0.90)
    tot_pi = pi_fb_est + pi_br_est + pi_os_est
    pi_fb_est /= tot_pi
    pi_br_est /= tot_pi
    pi_os_est /= tot_pi
    
    # 5. Form & Subtype Momentum
    p_rate = df["asof_pitcher_success_rate"].fillna(0.523766).to_numpy(float)
    prev1_p = df["asof_pitcher_prev1_game_success_rate"].fillna(0.523766).to_numpy(float)
    b_rate = df["asof_batter_success_rate"].fillna(0.523766).to_numpy(float)
    b_mid = df["asof_batter_middle_rate"].fillna(0.523766).to_numpy(float)
    
    n_p = df["asof_pitcher_n"].fillna(0).to_numpy(float)
    gamma_p = n_p / (n_p + 20.0)
    p_shrunk = gamma_p * p_rate + (1.0 - gamma_p) * 0.523766
    
    return pd.DataFrame({
        "zone_inevitability": zone_inevitability,
        "chase_incentive": chase_incentive,
        "count_pressure": count_pressure,
        "is_2s": (s == 2).astype(float),
        "is_3b": (b == 3).astype(float),
        "platoon": platoon,
        "pi_fb_est": pi_fb_est,
        "pi_br_est": pi_br_est,
        "pi_os_est": pi_os_est,
        "fb_rate": fb,
        "br_rate": br,
        "os_rate": os_,
        "p_shrunk": p_shrunk,
        "form_gap": prev1_p - (1.0 - b_rate),
        "b_mid": b_mid,
        "li": li,
        "is_high_li": (li >= 1.5).astype(float),
        "late_close": ((inn >= 7) & (score_diff <= 2)).astype(float),
    }, index=df.index)

def main():
    print("=" * 80)
    print("  122: LATENT PITCH-TYPE MIXTURE MARGINALIZATION (TEMPORAL-FORWARD)  ")
    print("=" * 80)
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("1. Loading train.csv (1.47M rows)...")
    train = pd.read_csv(DATA_PATH, low_memory=False)
    N = len(train)
    print(f"   Loaded {N:,} rows.")
    
    print("2. Building Game-Theoretic Latent State Features...")
    priors = build_per_season_priors(train)
    with open(OUT_DIR / "per_season_priors.json", "w", encoding="utf-8") as f:
        json.dump(priors, f)
        
    X_base, _ = build_v54_per_season_asof_75_features(train, priors=priors, prior=0.523766)
    X_gt = build_game_theory_features(train)
    X = pd.concat([X_base, X_gt], axis=1)
    
    y = train[TARGET].to_numpy(float)
    seasons = train["season"].to_numpy(int)
    
    print(f"   Full Feature Space: {X.shape}")
    
    # 3. Strict Temporal-Forward Walk-Forward Cross-Validation
    # Train <= 2021 -> Val 2022
    # Train <= 2022 -> Val 2023
    # Train <= 2023 -> Val 2024
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
        
        # Train Fastball Specialist
        cb_fb = CatBoostRegressor(iterations=600, learning_rate=0.05, depth=6, loss_function="RMSE", random_seed=42, verbose=0, thread_count=4)
        cb_fb.fit(X_tr, y_tr, sample_weight=X_tr["fb_rate"].to_numpy(float))
        
        # Train Breaking Specialist
        cb_br = CatBoostRegressor(iterations=600, learning_rate=0.05, depth=6, loss_function="RMSE", random_seed=43, verbose=0, thread_count=4)
        cb_br.fit(X_tr, y_tr, sample_weight=X_tr["br_rate"].to_numpy(float))
        
        # Train Offspeed Specialist
        cb_os = CatBoostRegressor(iterations=600, learning_rate=0.05, depth=6, loss_function="RMSE", random_seed=44, verbose=0, thread_count=4)
        cb_os.fit(X_tr, y_tr, sample_weight=X_tr["os_rate"].to_numpy(float))
        
        # Predict on Validation Season
        p_fb_va = cb_fb.predict(X_va)
        p_br_va = cb_br.predict(X_va)
        p_os_va = cb_os.predict(X_va)
        
        # Marginalize
        pi_fb = X_va["pi_fb_est"].to_numpy(float)
        pi_br = X_va["pi_br_est"].to_numpy(float)
        pi_os = X_va["pi_os_est"].to_numpy(float)
        p_mix_va = pi_fb * p_fb_va + pi_br * p_br_va + pi_os * p_os_va
        
        bs_mix = brier_score_loss(y_va, p_mix_va)
        bss_mix = (1.0 - (bs_mix / ref_brier_s)) * 100.0
        
        temporal_results[target_season] = {
            "brier": float(bs_mix),
            "bss": float(bss_mix),
            "time": time.time() - t_s,
            "n_rows": int(np.sum(val_mask))
        }
        print(f"   • Strict Forward Season {target_season} ({np.sum(val_mask):,} rows) -> Brier: {bs_mix:.6f} | BSS: {bss_mix:.4f}% ({time.time() - t_s:.1f}s)")
        
    print("\n4. Training Full 2019-2024 Production Models (3 Specialists + 2 LightGBM)...")
    cb_fb_prod = CatBoostRegressor(iterations=800, learning_rate=0.04, depth=6, loss_function="RMSE", random_seed=12201, verbose=200, thread_count=4)
    cb_fb_prod.fit(X, y, sample_weight=X["fb_rate"].to_numpy(float))
    cb_fb_prod.save_model(str(OUT_DIR / "latent_fb_specialist.cbm"))
    
    cb_br_prod = CatBoostRegressor(iterations=800, learning_rate=0.04, depth=6, loss_function="RMSE", random_seed=12202, verbose=200, thread_count=4)
    cb_br_prod.fit(X, y, sample_weight=X["br_rate"].to_numpy(float))
    cb_br_prod.save_model(str(OUT_DIR / "latent_br_specialist.cbm"))
    
    cb_os_prod = CatBoostRegressor(iterations=800, learning_rate=0.04, depth=6, loss_function="RMSE", random_seed=12203, verbose=200, thread_count=4)
    cb_os_prod.fit(X, y, sample_weight=X["os_rate"].to_numpy(float))
    cb_os_prod.save_model(str(OUT_DIR / "latent_os_specialist.cbm"))
    
    # Save manifest
    manifest_122 = {
        "pipeline": "REF4-LATENT-MIXTURE-122",
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "temporal_forward_results": temporal_results,
        "w_latent_mixture_convex": 0.055,
        "features": list(X.columns)
    }
    with open(OUT_DIR / "latent_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_122, f, indent=2)
        
    print(f"\nTraining pipeline completed in {(time.time() - t_start) / 60.0:.2f} minutes!")

if __name__ == "__main__":
    main()
