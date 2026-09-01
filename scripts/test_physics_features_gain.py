#!/usr/bin/env python3
"""Evaluate adding Physics-Trajectory Baseline Physical/Trajectory/Release features to Residual Booster on 2024 Holdout."""
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

def extract_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Physics-Trajectory Baseline(1038점) 고유 물리 궤적 및 릴리즈 안정성 피처."""
    velo = df["release_speed"].fillna(142.0).to_numpy(float) if "release_speed" in df.columns else np.full(len(df), 142.0)
    spin = df["spin_rate"].fillna(2200.0).to_numpy(float) if "spin_rate" in df.columns else np.full(len(df), 2200.0)
    pfx_x = df["pfx_x"].fillna(0.0).to_numpy(float) if "pfx_x" in df.columns else np.zeros(len(df))
    pfx_z = df["pfx_z"].fillna(0.0).to_numpy(float) if "pfx_z" in df.columns else np.zeros(len(df))
    rel_x = df["release_pos_x"].fillna(0.0).to_numpy(float) if "release_pos_x" in df.columns else np.zeros(len(df))
    rel_z = df["release_pos_z"].fillna(1.8).to_numpy(float) if "release_pos_z" in df.columns else np.full(len(df), 1.8)
    
    # Physical derived features
    movement_mag = np.sqrt(pfx_x**2 + pfx_z**2)
    spin_eff_ratio = np.clip(movement_mag / (velo + 1e-5), 0, 1.0)
    release_dist = np.sqrt(rel_x**2 + (rel_z - 1.8)**2)
    
    # Count pressure geometry
    b = df["balls_before"].fillna(0).to_numpy(float) if "balls_before" in df.columns else np.zeros(len(df))
    s = df["strikes_before"].fillna(0).to_numpy(float) if "strikes_before" in df.columns else np.zeros(len(df))
    is_2s = (s == 2).astype(float)
    is_3b = (b == 3).astype(float)
    
    return pd.DataFrame({
        "phys_velo": velo,
        "phys_spin": spin,
        "phys_movement_mag": movement_mag,
        "phys_spin_eff": spin_eff_ratio,
        "phys_release_dist": release_dist,
        "phys_is_2s": is_2s,
        "phys_is_3b": is_3b
    }, index=df.index)

def main():
    print("=== Testing Physics/Trajectory Features in Residual Booster ===")
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
    
    reg_mask_tr = train_subset["game_type"] != "F"
    X_tr_61, _ = build_v5_deep_61_features(train_subset[reg_mask_tr].reset_index(drop=True), profile_path=prof_path, prior=0.523766)
    
    prev1_p_tr = train_subset.loc[reg_mask_tr, "asof_pitcher_prev1_game_success_rate"].fillna(0.523766).to_numpy(float)
    b_rate_tr = train_subset.loc[reg_mask_tr, "asof_batter_success_rate"].fillna(0.523766).to_numpy(float)
    X_tr_61["form_gap"] = prev1_p_tr - b_rate_tr
    X_tr_61["anchor_p"] = p_103_tr[reg_mask_tr]
    y_resid_reg_tr = resid_tr[reg_mask_tr]
    
    # Add physics features
    phys_tr = extract_physics_features(train_subset[reg_mask_tr].reset_index(drop=True))
    X_tr_physics = pd.concat([X_tr_61, phys_tr], axis=1)
    
    print(f"Features: 61 base -> {X_tr_physics.shape[1]} physics-augmented")
    
    # Train 5 CatBoost + 5 LightGBM with physics
    r_seeds = [42, 1, 2, 3, 4]
    boosters_phys = []
    for s in r_seeds:
        cb = CatBoostRegressor(iterations=220, depth=5, learning_rate=0.03, l2_leaf_reg=15.0, random_seed=s, verbose=False, thread_count=-1)
        cb.fit(X_tr_physics, y_resid_reg_tr)
        boosters_phys.append(cb)
        
        lgb_m = lgb.LGBMRegressor(n_estimators=160, learning_rate=0.03, num_leaves=31, min_child_samples=50, subsample=0.8, colsample_bytree=0.8, random_state=s, n_jobs=-1, verbose=-1)
        lgb_m.fit(X_tr_physics, y_resid_reg_tr)
        boosters_phys.append(lgb_m)
        
    print(f"Trained {len(boosters_phys)} Physics-Augmented Boosters")
    
    val_2024 = raw[raw["season"] == 2024].copy().reset_index(drop=True)
    y_val = val_2024["control_success"].to_numpy(float)
    p_103_val = predict_103a_full(val_2024, MODEL_107, meta_107, priors, ps, bs, ms, tm, seeds, regime)
    reg_val = val_2024["game_type"] != "F"
    
    X_val_61, _ = build_v5_deep_61_features(val_2024[reg_val].reset_index(drop=True), profile_path=prof_path, prior=0.523766)
    prev1_p_val = val_2024.loc[reg_val, "asof_pitcher_prev1_game_success_rate"].fillna(0.523766).to_numpy(float)
    b_rate_val = val_2024.loc[reg_val, "asof_batter_success_rate"].fillna(0.523766).to_numpy(float)
    X_val_61["form_gap"] = prev1_p_val - b_rate_val
    X_val_61["anchor_p"] = p_103_val[reg_val]
    phys_val = extract_physics_features(val_2024[reg_val].reset_index(drop=True))
    X_val_physics = pd.concat([X_val_61, phys_val], axis=1)
    
    p_res_phys = np.mean([m.predict(X_val_physics) for m in boosters_phys], axis=0)
    
    bss_103 = brier_skill_score(y_val, p_103_val)
    print(f"103A Base BSS: {bss_103:.4f}")
    
    for w in [0.05, 0.07, 0.08, 0.10]:
        p_cand = p_103_val.copy()
        p_cand[reg_val] = p_cand[reg_val] + w * p_res_phys
        p_cand = np.clip(p_cand, 1e-5, 1 - 1e-5)
        bss_c = brier_skill_score(y_val, p_cand)
        delta = bss_c - bss_103
        print(f"  Physics Residual Booster (w={w:.2f}) -> BSS: {bss_c:.4f} (Δ {delta:+.4f} pt vs 103A)")

if __name__ == "__main__":
    main()
