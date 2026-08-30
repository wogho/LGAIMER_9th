#!/usr/bin/env python3
"""Evaluate 108C: 1군 TrackMan 3D Mechanical Fatigue Deviation + 2군 Futures 잠재 전이 잔차 융합 Full-Spectrum Booster."""
import gc, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'model/REF4-SUPER-ENSEMBLE-107A/production_package'))

from src.v5_deep_61_features import build_v5_deep_61_features
from scripts.test_108a_multiseason import predict_103a_full, brier_skill_score

def extract_advanced_physics(df: pd.DataFrame, tm_df: pd.DataFrame) -> pd.DataFrame:
    velo = df["release_speed"].fillna(142.0).to_numpy(float) if "release_speed" in df.columns else np.full(len(df), 142.0)
    spin = df["spin_rate"].fillna(2200.0).to_numpy(float) if "spin_rate" in df.columns else np.full(len(df), 2200.0)
    pfx_x = df["pfx_x"].fillna(0.0).to_numpy(float) if "pfx_x" in df.columns else np.zeros(len(df))
    pfx_z = df["pfx_z"].fillna(0.0).to_numpy(float) if "pfx_z" in df.columns else np.zeros(len(df))
    rel_x = df["release_pos_x"].fillna(0.0).to_numpy(float) if "release_pos_x" in df.columns else np.zeros(len(df))
    rel_z = df["release_pos_z"].fillna(1.8).to_numpy(float) if "release_pos_z" in df.columns else np.full(len(df), 1.8)
    
    movement_mag = np.sqrt(pfx_x**2 + pfx_z**2)
    spin_eff_ratio = np.clip(movement_mag / (velo + 1e-5), 0, 1.0)
    release_dist = np.sqrt(rel_x**2 + (rel_z - 1.8)**2)
    
    b = df["balls_before"].fillna(0).to_numpy(float) if "balls_before" in df.columns else np.zeros(len(df))
    s = df["strikes_before"].fillna(0).to_numpy(float) if "strikes_before" in df.columns else np.zeros(len(df))
    is_2s = (s == 2).astype(float)
    is_3b = (b == 3).astype(float)
    count_pressure = (b - s) * (b + s + 1.0) / 7.0
    
    return pd.DataFrame({
        "phys_velo": velo,
        "phys_spin": spin,
        "phys_movement_mag": movement_mag,
        "phys_spin_eff": spin_eff_ratio,
        "phys_release_dist": release_dist,
        "phys_is_2s": is_2s,
        "phys_is_3b": is_3b,
        "phys_count_pressure": count_pressure
    }, index=df.index)

def main():
    print("=== Testing 108C Full-Spectrum Booster (1군 고급물리 + 2군 전이 잔차) ===")
    t0 = time.time()
    raw = pd.read_csv(ROOT / "data/train.csv", low_memory=False)
    prof_path = ROOT / "model/team_asof_profile.json"
    
    MODEL_107 = ROOT / "model/REF4-SUPER-ENSEMBLE-107A/production_package/model"
    meta_107 = json.loads((MODEL_107 / "manifest.json").read_text(encoding="utf-8"))
    priors = json.loads((MODEL_107 / "per_season_priors.json").read_text(encoding="utf-8"))
    ps = pd.read_pickle(MODEL_107 / "pitcher_snapshots.pkl")
    bs = pd.read_pickle(MODEL_107 / "batter_snapshots.pkl")
    ms = pd.read_pickle(MODEL_107 / "pitchmix_snapshots.pkl")
    tm = str(MODEL_107 / "trackman_prior_features.csv")
    seeds = [260802, 260803, 260804, 260805, 260806, 260807]
    regime = json.loads((MODEL_107 / "f_regime_meta.json").read_text())
    
    train_subset = raw[raw["season"] < 2024].copy().reset_index(drop=True)
    p_103_tr = predict_103a_full(train_subset, MODEL_107, meta_107, priors, ps, bs, ms, tm, seeds, regime)
    y_tr = train_subset["control_success"].to_numpy(float)
    resid_tr = y_tr - p_103_tr
    
    # 1군 Training
    reg_mask_tr = train_subset["game_type"] != "F"
    X_tr_61_reg, _ = build_v5_deep_61_features(train_subset[reg_mask_tr].reset_index(drop=True), profile_path=prof_path, prior=0.523766)
    prev1_p_tr = train_subset.loc[reg_mask_tr, "asof_pitcher_prev1_game_success_rate"].fillna(0.523766).to_numpy(float)
    b_rate_tr = train_subset.loc[reg_mask_tr, "asof_batter_success_rate"].fillna(0.523766).to_numpy(float)
    X_tr_61_reg["form_gap"] = prev1_p_tr - b_rate_tr
    X_tr_61_reg["anchor_p"] = p_103_tr[reg_mask_tr]
    phys_tr_reg = extract_advanced_physics(train_subset[reg_mask_tr].reset_index(drop=True), None)
    X_tr_reg_full = pd.concat([X_tr_61_reg, phys_tr_reg], axis=1)
    y_resid_reg_tr = resid_tr[reg_mask_tr]
    
    # 2군 Training (Futures)
    fut_mask_tr = train_subset["game_type"] == "F"
    X_tr_61_fut, _ = build_v5_deep_61_features(train_subset[fut_mask_tr].reset_index(drop=True), profile_path=prof_path, prior=0.523766)
    X_tr_61_fut["anchor_p"] = p_103_tr[fut_mask_tr]
    phys_tr_fut = extract_advanced_physics(train_subset[fut_mask_tr].reset_index(drop=True), None)
    X_tr_fut_full = pd.concat([X_tr_61_fut, phys_tr_fut], axis=1)
    y_resid_fut_tr = resid_tr[fut_mask_tr]
    
    # Train 1군 Boosters (5 CatBoost + 5 LightGBM)
    r_seeds = [42, 1, 2, 3, 4]
    reg_boosters = []
    print("Training 1군 10-Model Boosters...")
    for s in r_seeds:
        cb = CatBoostRegressor(iterations=220, depth=5, learning_rate=0.03, l2_leaf_reg=15.0, random_seed=s, verbose=False, thread_count=-1)
        cb.fit(X_tr_reg_full, y_resid_reg_tr)
        reg_boosters.append(cb)
        
        lgb_m = lgb.LGBMRegressor(n_estimators=160, learning_rate=0.03, num_leaves=31, min_child_samples=50, subsample=0.8, colsample_bytree=0.8, random_state=s, n_jobs=-1, verbose=-1)
        lgb_m.fit(X_tr_reg_full, y_resid_reg_tr)
        reg_boosters.append(lgb_m)
        
    # Train 2군 Boosters (4 CatBoost)
    fut_boosters = []
    print(f"Training 2군 4-Model Boosters on {len(X_tr_fut_full):,} Futures rows...")
    for s in [42, 1, 2, 3]:
        cb_f = CatBoostRegressor(iterations=150, depth=4, learning_rate=0.03, l2_leaf_reg=20.0, random_seed=s, verbose=False, thread_count=-1)
        cb_f.fit(X_tr_fut_full, y_resid_fut_tr)
        fut_boosters.append(cb_f)
        
    print("\n=== Evaluating on Multi-Season Holdouts (2022, 2023, 2024) ===")
    for s in [2022, 2023, 2024]:
        val_df = raw[raw["season"] == s].copy().reset_index(drop=True)
        y_val = val_df["control_success"].to_numpy(float)
        p_103_val = predict_103a_full(val_df, MODEL_107, meta_107, priors, ps, bs, ms, tm, seeds, regime)
        
        reg_val = val_df["game_type"] != "F"
        fut_val = ~reg_val
        
        # Predict 1군
        X_val_61_reg, _ = build_v5_deep_61_features(val_df[reg_val].reset_index(drop=True), profile_path=prof_path, prior=0.523766)
        prev1_p_val = val_df.loc[reg_val, "asof_pitcher_prev1_game_success_rate"].fillna(0.523766).to_numpy(float)
        b_rate_val = val_df.loc[reg_val, "asof_batter_success_rate"].fillna(0.523766).to_numpy(float)
        X_val_61_reg["form_gap"] = prev1_p_val - b_rate_val
        X_val_61_reg["anchor_p"] = p_103_val[reg_val]
        phys_val_reg = extract_advanced_physics(val_df[reg_val].reset_index(drop=True), None)
        X_val_reg_full = pd.concat([X_val_61_reg, phys_val_reg], axis=1)
        p_res_reg = np.mean([m.predict(X_val_reg_full) for m in reg_boosters], axis=0)
        
        # Predict 2군
        p_res_fut = np.zeros(np.sum(fut_val))
        if np.sum(fut_val) > 0:
            X_val_61_fut, _ = build_v5_deep_61_features(val_df[fut_val].reset_index(drop=True), profile_path=prof_path, prior=0.523766)
            X_val_61_fut["anchor_p"] = p_103_val[fut_val]
            phys_val_fut = extract_advanced_physics(val_df[fut_val].reset_index(drop=True), None)
            X_val_fut_full = pd.concat([X_val_61_fut, phys_val_fut], axis=1)
            p_res_fut = np.mean([m.predict(X_val_fut_full) for m in fut_boosters], axis=0)
            
        bss_103 = brier_skill_score(y_val, p_103_val)
        
        p_108c = p_103_val.copy()
        p_108c[reg_val] = p_108c[reg_val] + 0.08 * p_res_reg
        p_108c[fut_val] = p_108c[fut_val] + 0.04 * p_res_fut
        p_108c = np.clip(p_108c, 1e-5, 1 - 1e-5)
        
        bss_108c = brier_skill_score(y_val, p_108c)
        delta = bss_108c - bss_103
        print(f"Season {s} ({len(val_df):,} rows) -> 103A BSS: {bss_103:.4f} | 108C BSS: {bss_108c:.4f} (Δ {delta:+.4f} pt)")

if __name__ == "__main__":
    main()
