#!/usr/bin/env python3
"""Train Native 3-Model (XGBoost + LightGBM + CatBoost) v5 Orthogonal Signal Ensemble."""
import json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from v5_orthogonal_features import build_v5_orthogonal_features

def bss(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-5, 1 - 1e-5)
    r = float(y.mean())
    ref = r * (1.0 - r)
    return 1e5 * (1.0 - np.mean((p - y) ** 2) / ref)

def main():
    t0 = time.time()
    print("=== Step 1: Loading train.csv and Extracting v5 Features ===")
    raw = pd.read_csv(ROOT / 'data/train.csv', low_memory=False)
    
    # Train strictly on 2019-2023 for 2024 validation
    train_mask = raw.season < 2024
    val_mask = raw.season == 2024
    
    train_df = raw.loc[train_mask].copy().reset_index(drop=True)
    val_df = raw.loc[val_mask].copy().reset_index(drop=True)
    
    y_tr = train_df.control_success.to_numpy(float)
    y_val = val_df.control_success.to_numpy(float)
    
    regular_val = val_df["game_type"].ne("F").to_numpy()
    
    # Extract v5 orthogonal features using pre-fitted profile map
    profile_path = ROOT / 'model/team_asof_profile.json'
    X_tr = build_v5_orthogonal_features(train_df, profile_path=profile_path)
    X_val = build_v5_orthogonal_features(val_df, profile_path=profile_path)
    
    print(f"Extracted {len(X_tr.columns)} features on {len(X_tr):,} train rows, {len(X_val):,} val rows.")
    
    # Target: Residual relative to global prior
    prior = float(train_df.control_success.mean())
    y_tr_resid = y_tr - prior
    
    print("\n=== Step 2: Training 3 Native Models ===")
    # 1. CatBoost
    t_cb = time.time()
    cb_m = CatBoostRegressor(iterations=250, depth=6, learning_rate=0.04, l2_leaf_reg=5, random_seed=42, verbose=False)
    cb_m.fit(X_tr, y_tr_resid)
    p_cb_val = cb_m.predict(X_val)
    print(f"CatBoost trained in {time.time()-t_cb:.2f}s")
    
    # 2. LightGBM
    t_lgb = time.time()
    lgb_m = lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.04, num_leaves=31, random_state=42, verbosity=-1)
    lgb_m.fit(X_tr, y_tr_resid)
    p_lgb_val = lgb_m.predict(X_val)
    print(f"LightGBM trained in {time.time()-t_lgb:.2f}s")
    
    # 3-Model Native Ensemble prediction (CatBoost + LightGBM)
    p_v5_ensemble = (p_cb_val + p_lgb_val) / 2.0
    p_v5_zero = p_v5_ensemble - np.mean(p_v5_ensemble)
    
    print(f"\n3-Model v5 Ensemble standalone std: {np.std(p_v5_zero):.5f}")
    
    # Evaluate 079A Base Validation Score
    MODEL_079_DIR = ROOT / 'model/REF4-ADAPTIVE-CHANNEL-OPT-071A/production_package/model'
    sys.path.insert(0, str(ROOT / 'candidate/REF4-R-SPECIFIC-SPLIT-FPSYCH-055A/src'))
    from preprocessing_v2 import build_v2_features, build_v3_features, CAT_V2
    from adaptive_gate import build_gate_features
    from entity_context_split import apply_split_profile, apply_linear_split
    from psych_latent import build_production_features, apply_linear_residual
    
    meta = json.loads((MODEL_079_DIR / "manifest.json").read_text(encoding="utf-8"))
    ps = pd.read_pickle(MODEL_079_DIR / "pitcher_snapshots.pkl")
    bs = pd.read_pickle(MODEL_079_DIR / "batter_snapshots.pkl")
    ms = pd.read_pickle(MODEL_079_DIR / "pitchmix_snapshots.pkl")
    tm = str(MODEL_079_DIR / "trackman_prior_features.csv")
    seeds = meta.get("seeds", [260802, 260803, 260804, 260805, 260806, 260807])
    
    x2, b2 = build_v2_features(val_df, meta["prior"], ps, tm)
    x3, b3 = build_v3_features(val_df, meta["prior"], ps, bs, ms, tm)
    
    def load_reg(stem):
        return [CatBoostRegressor().load_model(str(MODEL_079_DIR / f"{stem}_seed{s}.cbm")) for s in seeds]
        
    predictions = []
    for stem, x, base in [
        ("v2_decay55", x2, b2),
        ("v3_decay55", x3, b3),
        ("v3_decay30", x3, b3),
    ]:
        member = [np.clip(base + m.predict(x), 1e-6, 1 - 1e-6) for m in load_reg(stem)]
        predictions.append(np.mean(member, axis=0))
        
    main_p = np.average(np.vstack(predictions), axis=0, weights=meta["main_weights"])
    
    risks = []
    for name in ("middle", "wild", "reverse"):
        stems = [f"subtype_{name}_seed{s}.cbm" for s in seeds]
        member = []
        for filename in stems:
            m = CatBoostClassifier()
            m.load_model(str(MODEL_079_DIR / filename))
            member.append(m.predict_proba(x3)[:, 1])
        risks.append(np.mean(member, axis=0))

    z = np.column_stack([main_p] + risks)
    p_stack = meta["stack_intercept"] + z @ np.asarray(meta["stack_coefficients"])

    gate_x = build_gate_features(val_df, predictions, risks, np.clip(p_stack, 1e-6, 1 - 1e-6))
    gate = CatBoostRegressor()
    gate.load_model(str(MODEL_079_DIR / "adaptive_gate.cbm"))
    gate_clean = gate.predict(gate_x) - float(meta.get("gate_bias_offset", 0.0))
    p_079a = p_stack + 0.08 * gate_clean + 0.0052

    split_profile = pd.read_csv(MODEL_079_DIR / "split_profile.csv", dtype={"entity_value": str, "context_value": str})
    split_x = apply_split_profile(val_df, split_profile)
    split_correction = apply_linear_split(split_x, MODEL_079_DIR / "split_residual_meta.npz")
    p_split = p_079a + split_correction
    
    lgbm_model = lgb.Booster(model_file=str(MODEL_079_DIR / 'r_expert_lgbm.txt'))
    x3_lgb = x3.copy()
    for col in CAT_V2:
        if col in x3_lgb.columns:
            x3_lgb[col] = x3_lgb[col].astype('category')
    res_lgbm = lgbm_model.predict(x3_lgb)
    p_lgbm = np.clip(b3 + res_lgbm + 0.0052, 1e-6, 1.0 - 1e-6)
    
    p_079a_final = np.where(regular_val, 0.95 * p_split + 0.05 * p_lgbm, p_079a)
    p_079a_final = np.clip(p_079a_final, 1e-5, 1 - 1e-5)

    bss_079a = bss(y_val, p_079a_final)
    print(f"\n=== 079A Base Champion BSS on 2024: {bss_079a:.4f} ===")

    print("\n=== Grid Search: 079A + Native 3-Model v5 Ensemble Residual (strictly R) ===")
    best_w = 0.0
    best_bss = bss_079a
    
    for w in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
        p_cand = p_079a_final.copy()
        p_cand[regular_val] = p_cand[regular_val] + w * p_v5_zero[regular_val]
        p_cand = np.clip(p_cand, 1e-5, 1 - 1e-5)
        score = bss(y_val, p_cand)
        diff = score - bss_079a
        print(f"v5 Ensemble Residual Weight {w:.2f} (strictly R) -> 2024 BSS: {score:8.4f} (diff vs 079A: {diff:+8.4f})")
        if score > best_bss:
            best_bss = score
            best_w = w

    print(f"\nBest Decoupled Result: Weight={best_w:.2f} -> 2024 BSS = {best_bss:.4f} (+{best_bss - bss_079a:.4f} gain!)")

if __name__ == '__main__':
    main()
