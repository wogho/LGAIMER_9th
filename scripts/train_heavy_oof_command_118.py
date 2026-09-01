#!/usr/bin/env python3
"""
Heavy 5-Fold OOF Command Outcome Training Pipeline for 118.
Trains 15 Heavy GBDT Models (CatBoost, LightGBM, XGBoost) on 1.47M rows with 5-Fold Group Stratification.
Uses 100% test-compliant 75-feature vector space from src/v54_per_season_asof_75_features.py.
"""
import gc
import json
import os
import pickle
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import brier_score_loss
from catboost import CatBoostRegressor
import lightgbm as lgb
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from src.v54_per_season_asof_75_features import build_v54_per_season_asof_75_features, build_per_season_priors

DATA_PATH = ROOT / "data/train.csv"
OUT_DIR = ROOT / "model/REF4-HEAVY-OOF-COMMAND-118/production_package/model"
TARGET = "control_success"

def main():
    print("=" * 80)
    print("  HEAVY 5-FOLD OOF COMMAND OUTCOME TRAINING PIPELINE (118)  ")
    print("=" * 80)
    t_start = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("1. Loading train.csv (1.47M rows)...")
    train = pd.read_csv(DATA_PATH, low_memory=False)
    N = len(train)
    print(f"   Loaded {N:,} rows.")
    
    print("2. Building Per-Season Priors & 75-Feature Vector Space...")
    priors = build_per_season_priors(train)
    with open(OUT_DIR / "per_season_priors.json", "w", encoding="utf-8") as f:
        json.dump(priors, f)
        
    X, _ = build_v54_per_season_asof_75_features(
        train, priors=priors, prior=0.523766
    )
    
    # Extra Game Context & Count Tension
    b = train["balls_before"].fillna(0).to_numpy(float)
    s = train["strikes_before"].fillna(0).to_numpy(float)
    li = train["li"].fillna(0.98).to_numpy(float)
    inn = train["inning"].fillna(1).to_numpy(float)
    score_diff = np.abs(train["score_diff_pitcher_team"].fillna(0).to_numpy(float))
    
    X["count_pressure"] = (b - s) * (b + s + 1.0) / 7.0
    X["is_2s"] = (s == 2).astype(float)
    X["is_3b"] = (b == 3).astype(float)
    X["is_high_li"] = (li >= 1.5).astype(float)
    X["late_close"] = ((inn >= 7) & (score_diff <= 2)).astype(float)
    
    y = train[TARGET].to_numpy(float)
    seasons = train["season"].to_numpy(int)
    ref_brier = brier_score_loss(y, np.full_like(y, 0.523766))
    
    print(f"   Feature space shape: {X.shape}")
    
    # 5-Fold Stratified K-Fold
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    oof_preds_cb = np.zeros(N, dtype=float)
    oof_preds_lgb = np.zeros(N, dtype=float)
    oof_preds_xgb = np.zeros(N, dtype=float)
    
    cb_models = []
    lgb_models = []
    xgb_models = []
    
    print(f"\n3. Starting 5-Fold Training ({n_splits} Folds x 3 GBDT Architectures = 15 Models)...")
    
    fold_idx = 0
    for train_idx, val_idx in kf.split(X):
        fold_idx += 1
        t_fold = time.time()
        print(f"\n--- [Fold {fold_idx}/{n_splits}] Training on {len(train_idx):,} rows, Validating on {len(val_idx):,} rows ---")
        
        X_tr, y_tr = X.iloc[train_idx], y[train_idx]
        X_va, y_va = X.iloc[val_idx], y[val_idx]
        
        # A. CatBoost
        print(f"   [Fold {fold_idx}] Training CatBoost (depth=6, iters=800, lr=0.05)...")
        cb = CatBoostRegressor(
            iterations=800,
            learning_rate=0.05,
            depth=6,
            loss_function="RMSE",
            random_seed=42 + fold_idx,
            verbose=200,
            thread_count=4
        )
        cb.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=200)
        p_va_cb = cb.predict(X_va)
        oof_preds_cb[val_idx] = p_va_cb
        cb_path = OUT_DIR / f"heavy_cb_fold{fold_idx}.cbm"
        cb.save_model(str(cb_path))
        cb_models.append(cb_path.name)
        
        # B. LightGBM
        print(f"   [Fold {fold_idx}] Training LightGBM (leaves=31, iters=800, lr=0.04)...")
        trn_data = lgb.Dataset(X_tr, label=y_tr)
        val_data = lgb.Dataset(X_va, label=y_va, reference=trn_data)
        params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.04,
            "num_leaves": 31,
            "feature_fraction": 0.85,
            "bagging_fraction": 0.85,
            "bagging_freq": 1,
            "seed": 42 + fold_idx,
            "num_threads": 4,
            "verbosity": -1
        }
        lgbm = lgb.train(
            params,
            trn_data,
            num_boost_round=800,
            valid_sets=[val_data],
            callbacks=[lgb.early_stopping(50, verbose=False)]
        )
        p_va_lgb = lgbm.predict(X_va)
        oof_preds_lgb[val_idx] = p_va_lgb
        lgb_path = OUT_DIR / f"heavy_lgb_fold{fold_idx}.txt"
        lgbm.save_model(str(lgb_path))
        lgb_models.append(lgb_path.name)
        
        # C. XGBoost
        print(f"   [Fold {fold_idx}] Training XGBoost (depth=6, iters=800, lr=0.04)...")
        xgb = XGBRegressor(
            n_estimators=800,
            learning_rate=0.04,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42 + fold_idx,
            n_jobs=4,
            early_stopping_rounds=50,
            tree_method="hist"
        )
        xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=200)
        p_va_xgb = xgb.predict(X_va)
        oof_preds_xgb[val_idx] = p_va_xgb
        xgb_path = OUT_DIR / f"heavy_xgb_fold{fold_idx}.json"
        xgb.save_model(str(xgb_path))
        xgb_models.append(xgb_path.name)
        
        # Fold Ensemble Score
        p_va_ens = (p_va_cb + p_va_lgb + p_va_xgb) / 3.0
        bs_fold = brier_score_loss(y_va, p_va_ens)
        bss_fold = (1.0 - (bs_fold / ref_brier)) * 100.0
        print(f"   >>> Fold {fold_idx} Brier: {bs_fold:.6f} | BSS: {bss_fold:.4f}% | Elapsed: {time.time() - t_fold:.1f}s")
        
    print("\n" + "=" * 80)
    print("  OUT-OF-FOLD MULTI-YEAR VALIDATION RESULTS (1.47M ROWS)  ")
    print("=" * 80)
    
    oof_ens = (oof_preds_cb + oof_preds_lgb + oof_preds_xgb) / 3.0
    bs_all = brier_score_loss(y, oof_ens)
    bss_all = (1.0 - (bs_all / ref_brier)) * 100.0
    print(f"Overall Full 1.47M OOF Brier Score: {bs_all:.6f} | BSS: {bss_all:.4f}%")
    
    for s in sorted(np.unique(seasons)):
        mask_s = (seasons == s)
        bs_s = brier_score_loss(y[mask_s], oof_ens[mask_s])
        ref_s = brier_score_loss(y[mask_s], np.full_like(y[mask_s], 0.523766))
        bss_s = (1.0 - (bs_s / ref_s)) * 100.0
        print(f"  • Season {s} ({np.sum(mask_s):,} rows) -> Brier: {bs_s:.6f} | BSS: {bss_s:.4f}%")
        
    # Save Model Manifest & Artifacts
    manifest_heavy = {
        "pipeline": "REF4-HEAVY-OOF-COMMAND-118",
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_models": 15,
        "cb_models": cb_models,
        "lgb_models": lgb_models,
        "xgb_models": xgb_models,
        "overall_oof_brier": float(bs_all),
        "overall_oof_bss": float(bss_all),
        "features": list(X.columns),
        "w_heavy_command": 0.045
    }
    with open(OUT_DIR / "heavy_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest_heavy, f, indent=2)
        
    print(f"\nTraining pipeline completed in {(time.time() - t_start) / 60.0:.2f} minutes!")

if __name__ == "__main__":
    main()
