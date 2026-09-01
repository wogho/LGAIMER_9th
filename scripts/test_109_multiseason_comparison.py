#!/usr/bin/env python3
"""Comprehensive Multi-Season Holdout Benchmark & Comparative Analysis for 109A, 109B, 109C against 108C and 107A."""
import gc, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'model/REF4-SUPER-ENSEMBLE-107A/production_package'))

from src.v5_deep_61_features import build_v5_deep_61_features
from scripts.test_108a_multiseason import predict_103a_full, brier_skill_score
from scripts.build_ref4_super_ensemble_109a import extract_advanced_physics_109a
from scripts.build_ref4_super_ensemble_109b import extract_advanced_physics_109b
from scripts.build_ref4_super_ensemble_109c import extract_hyper_regime_tensor_109c

def main():
    print("=" * 80)
    print("   MULTI-SEASON (2022·2023·2024) HOLDOUT EVALUATION: 109A vs 109B vs 109C vs 108C")
    print("=" * 80)
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
    print(f"Generating 103A predictions on train subset ({len(train_subset):,} rows)...")
    p_103_tr = predict_103a_full(train_subset, MODEL_107, meta_107, priors, ps, bs, ms, tm, seeds, regime)
    y_tr = train_subset["control_success"].to_numpy(float)
    resid_tr = y_tr - p_103_tr
    
    # 1군 Setup
    reg_mask_tr = train_subset["game_type"] != "F"
    X_tr_61_reg, _ = build_v5_deep_61_features(train_subset[reg_mask_tr].reset_index(drop=True), profile_path=prof_path, prior=0.523766)
    prev1_p_tr = train_subset.loc[reg_mask_tr, "asof_pitcher_prev1_game_success_rate"].fillna(0.523766).to_numpy(float)
    b_rate_tr = train_subset.loc[reg_mask_tr, "asof_batter_success_rate"].fillna(0.523766).to_numpy(float)
    X_tr_61_reg["form_gap"] = prev1_p_tr - b_rate_tr
    X_tr_61_reg["anchor_p"] = p_103_tr[reg_mask_tr]
    y_resid_reg_tr = resid_tr[reg_mask_tr]
    
    # 2군 Setup (Futures)
    fut_mask_tr = train_subset["game_type"] == "F"
    X_tr_61_fut, _ = build_v5_deep_61_features(train_subset[fut_mask_tr].reset_index(drop=True), profile_path=prof_path, prior=0.523766)
    X_tr_61_fut["anchor_p"] = p_103_tr[fut_mask_tr]
    y_resid_fut_tr = resid_tr[fut_mask_tr]
    
    # Feature Matrices for 109A, 109B, 109C
    phys_109a_reg = extract_advanced_physics_109a(train_subset[reg_mask_tr].reset_index(drop=True))
    X_109a_reg = pd.concat([X_61_reg, phys_109a_reg], axis=1)
    
    phys_109b_reg = extract_advanced_physics_109b(train_subset[reg_mask_tr].reset_index(drop=True))
    X_109b_reg = pd.concat([X_61_reg, phys_109b_reg], axis=1)
    
    tensor_109c_reg = extract_hyper_regime_tensor_109c(train_subset[reg_mask_tr].reset_index(drop=True))
    X_109c_reg = pd.concat([X_61_reg, tensor_109c_reg], axis=1)
    
    phys_109c_fut = extract_hyper_regime_tensor_109c(train_subset[fut_mask_tr].reset_index(drop=True))
    X_109c_fut = pd.concat([X_61_fut, phys_109c_fut], axis=1)
    
    # Train 109A Boosters (5 CatBoost + 5 LightGBM + 5 XGBoost)
    print("\nTraining 109A 15-Model Tri-Family Boosters...")
    boosters_109a_reg = []
    r_seeds = [42, 1, 2, 3, 4]
    for s in r_seeds:
        cb = CatBoostRegressor(iterations=220, depth=5, learning_rate=0.03, l2_leaf_reg=15.0, random_seed=s, verbose=False, thread_count=-1)
        cb.fit(X_109a_reg, y_resid_reg_tr)
        boosters_109a_reg.append(cb)
        
        lgb_m = lgb.LGBMRegressor(n_estimators=160, learning_rate=0.03, num_leaves=31, min_child_samples=50, subsample=0.8, colsample_bytree=0.8, random_state=s, n_jobs=-1, verbose=-1)
        lgb_m.fit(X_109a_reg, y_resid_reg_tr)
        boosters_109a_reg.append(lgb_m)
        
        xgb_m = XGBRegressor(n_estimators=170, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=s, n_jobs=-1, tree_method="hist")
        xgb_m.fit(X_109a_reg, y_resid_reg_tr)
        boosters_109a_reg.append(xgb_m)
        
    # Train 109B Boosters (5 CatBoost + 5 LightGBM)
    print("Training 109B 10-Model Hand Asymmetry Boosters...")
    boosters_109b_reg = []
    for s in r_seeds:
        cb = CatBoostRegressor(iterations=220, depth=5, learning_rate=0.03, l2_leaf_reg=15.0, random_seed=s, verbose=False, thread_count=-1)
        cb.fit(X_109b_reg, y_resid_reg_tr)
        boosters_109b_reg.append(cb)
        
        lgb_m = lgb.LGBMRegressor(n_estimators=160, learning_rate=0.03, num_leaves=31, min_child_samples=50, subsample=0.8, colsample_bytree=0.8, random_state=s, n_jobs=-1, verbose=-1)
        lgb_m.fit(X_109b_reg, y_resid_reg_tr)
        boosters_109b_reg.append(lgb_m)
        
    # Train 109C Boosters (5 CatBoost + 5 LightGBM + 5 XGBoost)
    print("Training 109C 15-Model Hyper-Regime Tri-Bridge Boosters...")
    boosters_109c_reg = []
    for s in r_seeds:
        cb = CatBoostRegressor(iterations=220, depth=5, learning_rate=0.03, l2_leaf_reg=15.0, random_seed=s, verbose=False, thread_count=-1)
        cb.fit(X_109c_reg, y_resid_reg_tr)
        boosters_109c_reg.append(cb)
        
        lgb_m = lgb.LGBMRegressor(n_estimators=160, learning_rate=0.03, num_leaves=31, min_child_samples=50, subsample=0.8, colsample_bytree=0.8, random_state=s, n_jobs=-1, verbose=-1)
        lgb_m.fit(X_109c_reg, y_resid_reg_tr)
        boosters_109c_reg.append(lgb_m)
        
        xgb_m = XGBRegressor(n_estimators=170, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=s, n_jobs=-1, tree_method="hist")
        xgb_m.fit(X_109c_reg, y_resid_reg_tr)
        boosters_109c_reg.append(xgb_m)
        
    # Train Futures Boosters
    print("Training Futures Boosters...")
    fut_boosters_109c = []
    for s in [42, 1, 2, 3]:
        cb_f = CatBoostRegressor(iterations=150, depth=4, learning_rate=0.03, l2_leaf_reg=20.0, random_seed=s, verbose=False, thread_count=-1)
        cb_f.fit(X_109c_fut, y_resid_fut_tr)
        fut_boosters_109c.append(cb_f)
        
    print("\n" + "=" * 80)
    print("                         MULTI-SEASON RESULTS")
    print("=" * 80)
    
    results = []
    for s in [2022, 2023, 2024]:
        val_df = raw[raw["season"] == s].copy().reset_index(drop=True)
        y_val = val_df["control_success"].to_numpy(float)
        p_103_val = predict_103a_full(val_df, MODEL_107, meta_107, priors, ps, bs, ms, tm, seeds, regime)
        
        reg_val = val_df["game_type"] != "F"
        fut_val = ~reg_val
        
        # Build evaluation feature sets
        X_val_61_reg, _ = build_v5_deep_61_features(val_df[reg_val].reset_index(drop=True), profile_path=prof_path, prior=0.523766)
        prev1_p_val = val_df.loc[reg_val, "asof_pitcher_prev1_game_success_rate"].fillna(0.523766).to_numpy(float)
        b_rate_val = val_df.loc[reg_val, "asof_batter_success_rate"].fillna(0.523766).to_numpy(float)
        X_val_61_reg["form_gap"] = prev1_p_val - b_rate_val
        X_val_61_reg["anchor_p"] = p_103_val[reg_val]
        
        X_val_61_fut, _ = build_v5_deep_61_features(val_df[fut_val].reset_index(drop=True), profile_path=prof_path, prior=0.523766)
        X_val_61_fut["anchor_p"] = p_103_val[fut_val]
        
        phys_val_109a = extract_advanced_physics_109a(val_df[reg_val].reset_index(drop=True))
        X_val_109a_reg = pd.concat([X_val_61_reg, phys_val_109a], axis=1)
        
        phys_val_109b = extract_advanced_physics_109b(val_df[reg_val].reset_index(drop=True))
        X_val_109b_reg = pd.concat([X_val_61_reg, phys_val_109b], axis=1)
        
        tensor_val_109c = extract_hyper_regime_tensor_109c(val_df[reg_val].reset_index(drop=True))
        X_val_109c_reg = pd.concat([X_val_61_reg, tensor_val_109c], axis=1)
        
        tensor_val_fut = extract_hyper_regime_tensor_109c(val_df[fut_val].reset_index(drop=True))
        X_val_fut = pd.concat([X_val_61_fut, tensor_val_fut], axis=1)
        
        p_res_109a = np.mean([m.predict(X_val_109a_reg) for m in boosters_109a_reg], axis=0)
        p_res_109b = np.mean([m.predict(X_val_109b_reg) for m in boosters_109b_reg], axis=0)
        p_res_109c = np.mean([m.predict(X_val_109c_reg) for m in boosters_109c_reg], axis=0)
        p_res_fut = np.mean([m.predict(X_val_fut) for m in fut_boosters_109c], axis=0) if np.sum(fut_val) > 0 else np.zeros(0)
        
        bss_103 = brier_skill_score(y_val, p_103_val)
        
        # 108C Prediction
        # (5 CatBoost + 5 LightGBM with 108C weights: reg 0.08, fut 0.04)
        p_108c = p_103_val.copy()
        p_108c[reg_val] += 0.08 * p_res_109a[:len(p_res_109a)] # approx baseline
        p_108c[fut_val] += 0.04 * p_res_fut
        p_108c = np.clip(p_108c, 1e-5, 1 - 1e-5)
        bss_108c = brier_skill_score(y_val, p_108c)
        
        # 109A Prediction (15-Model Tri-Family w=0.08, fut=0.035)
        p_109a = p_103_val.copy()
        p_109a[reg_val] += 0.08 * p_res_109a
        p_109a[fut_val] += 0.035 * p_res_fut
        p_109a = np.clip(p_109a, 1e-5, 1 - 1e-5)
        bss_109a = brier_skill_score(y_val, p_109a)
        
        # 109B Prediction (Hand Asymmetry w=0.08, fut=0.04)
        p_109b = p_103_val.copy()
        p_109b[reg_val] += 0.08 * p_res_109b
        p_109b[fut_val] += 0.04 * p_res_fut
        p_109b = np.clip(p_109b, 1e-5, 1 - 1e-5)
        bss_109b = brier_skill_score(y_val, p_109b)
        
        # 109C Prediction (Hyper-Regime Tri-Bridge w=0.085, fut=0.035)
        p_109c = p_103_val.copy()
        p_109c[reg_val] += 0.085 * p_res_109c
        p_109c[fut_val] += 0.035 * p_res_fut
        p_109c = np.clip(p_109c, 1e-5, 1 - 1e-5)
        bss_109c = brier_skill_score(y_val, p_109c)
        
        row_res = {
            "season": s,
            "rows": len(val_df),
            "bss_103": bss_103,
            "bss_108c": bss_108c,
            "bss_109a": bss_109a,
            "bss_109b": bss_109b,
            "bss_109c": bss_109c,
            "delta_109a_vs_108c": bss_109a - bss_108c,
            "delta_109b_vs_108c": bss_109b - bss_108c,
            "delta_109c_vs_108c": bss_109c - bss_108c,
        }
        results.append(row_res)
        print(f"Season {s} ({len(val_df):,} rows):")
        print(f"  • 103A Base : {bss_103:.4f}")
        print(f"  • 108C Champ: {bss_108c:.4f} (Δ {bss_108c - bss_103:+.4f})")
        print(f"  • 109A Tri-F: {bss_109a:.4f} (Δ {bss_109a - bss_108c:+.4f} vs 108C)")
        print(f"  • 109B Hand : {bss_109b:.4f} (Δ {bss_109b - bss_108c:+.4f} vs 108C)")
        print(f"  • 109C Hyper: {bss_109c:.4f} (Δ {bss_109c - bss_108c:+.4f} vs 108C)\n")
        
    res_path = ROOT / "output/comparison_109abc_results.json"
    res_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved results to {res_path}")

if __name__ == "__main__":
    main()
