#!/usr/bin/env python3
"""Test v5 Orthogonal Signal Blending (080A) on 2024 Holdout OOF."""
import json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
CAND_DIR = ROOT / 'candidate/REF4-R-SPECIFIC-SPLIT-FPSYCH-055A'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'model/REF4-ADAPTIVE-CHANNEL-OPT-071A/production_package'))
sys.path.insert(0, str(CAND_DIR))
sys.path.insert(0, str(ROOT / 'github_reference/4번 레포'))
sys.path.insert(0, str(ROOT / 'github_reference/4번 레포/src'))

from preprocessing_v2 import build_v2_features, build_v3_features, CAT_V2
from adaptive_gate import build_gate_features
from entity_context_split import apply_split_profile, apply_linear_split
from psych_latent import build_production_features, apply_linear_residual
from v5_orthogonal_features import build_v5_orthogonal_features

def bss(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-5, 1 - 1e-5)
    r = float(y.mean())
    ref = r * (1.0 - r)
    return 1e5 * (1.0 - np.mean((p - y) ** 2) / ref)

def main():
    t0 = time.time()
    raw = pd.read_csv(ROOT / 'data/train.csv', low_memory=False)
    
    val_24 = raw.loc[raw.season == 2024].copy().reset_index(drop=True)
    y_24 = val_24.control_success.to_numpy(float)
    futures_24 = val_24["game_type"].eq("F").to_numpy()
    regular_24 = ~futures_24
    
    # 1. Compute 079A Base Predictions on 2024 Holdout
    MODEL = ROOT / 'model/REF4-ADAPTIVE-CHANNEL-OPT-071A/production_package/model'
    meta = json.loads((MODEL / "manifest.json").read_text(encoding="utf-8"))
    ps = pd.read_pickle(MODEL / "pitcher_snapshots.pkl")
    bs = pd.read_pickle(MODEL / "batter_snapshots.pkl")
    ms = pd.read_pickle(MODEL / "pitchmix_snapshots.pkl")
    tm = str(MODEL / "trackman_prior_features.csv")
    seeds = meta.get("seeds", [260802, 260803, 260804, 260805, 260806, 260807])
    
    x2, b2 = build_v2_features(val_24, meta["prior"], ps, tm)
    x3, b3 = build_v3_features(val_24, meta["prior"], ps, bs, ms, tm)
    
    def load_reg(stem):
        return [CatBoostRegressor().load_model(str(MODEL / f"{stem}_seed{s}.cbm")) for s in seeds]
        
    predictions = []
    for stem, x, base in [
        ("v2_decay55", x2, b2),
        ("v3_decay55", x3, b3),
        ("v3_decay30", x3, b3),
    ]:
        member = [np.clip(base + m.predict(x), 1e-6, 1 - 1e-6) for m in load_reg(stem)]
        predictions.append(np.mean(member, axis=0))
        
    regime = json.loads((MODEL / "f_regime_meta.json").read_text())
    
    def f_reg_mean(stem, count, x, base):
        member = []
        for j in range(count):
            m = CatBoostRegressor()
            m.load_model(str(MODEL / f"{stem}_{j}.cbm"))
            member.append(np.clip(base + m.predict(x), 1e-6, 1 - 1e-6))
        return np.mean(member, axis=0)

    if futures_24.any():
        f2 = f_reg_mean("f_v2_all", 4, x2, b2)
        predictions[0] = np.where(futures_24, predictions[0] + regime["v2_scale"] * (f2 - predictions[0]), predictions[0])
        f55 = f_reg_mean("f_v355_recent", 6, x3, b3)
        predictions[1] = np.where(futures_24, predictions[1] + regime["v355_scale"] * (f55 - predictions[1]), predictions[1])
        f30a = f_reg_mean("f_v330_all", 4, x3, b3)
        f30r = f_reg_mean("f_v330_recent", 2, x3, b3)
        recent_inner = predictions[2] + regime["v330_recent_inner_scale"] * (f30r - predictions[2])
        f30 = regime["v330_all_weight"] * f30a + (1 - regime["v330_all_weight"]) * recent_inner
        predictions[2] = np.where(futures_24, predictions[2] + regime["v330_scale"] * (f30 - predictions[2]), predictions[2])

    risks = []
    for name in ("middle", "wild", "reverse"):
        stems = [f"subtype_{name}_seed{s}.cbm" for s in seeds]
        member = []
        for filename in stems:
            m = CatBoostClassifier()
            m.load_model(str(MODEL / filename))
            member.append(m.predict_proba(x3)[:, 1])
        risk = np.mean(member, axis=0)
        if futures_24.any():
            fm = CatBoostClassifier()
            fm.load_model(str(MODEL / f"f_subtype_{name}.cbm"))
            fr = fm.predict_proba(x3)[:, 1]
            risk = np.where(futures_24, risk + regime["subtype_scale"] * (fr - risk), risk)
        risks.append(risk)

    main_p = np.average(np.vstack(predictions), axis=0, weights=meta["main_weights"])
    z = np.column_stack([main_p] + risks)
    p_stack = meta["stack_intercept"] + z @ np.asarray(meta["stack_coefficients"])

    # Adaptive Gate (gate_scale = 0.08)
    gate_x = build_gate_features(val_24, predictions, risks, np.clip(p_stack, 1e-6, 1 - 1e-6))
    gate = CatBoostRegressor()
    gate.load_model(str(MODEL / "adaptive_gate.cbm"))
    gate_pred = gate.predict(gate_x)
    bias_offset = float(meta.get("gate_bias_offset", 0.0))
    gate_clean = gate_pred - bias_offset
    
    p_079a = p_stack + 0.08 * gate_clean + 0.0052
    
    # Futures Psych Latent
    residual_x = build_production_features(val_24, MODEL / "psych_profile.pkl", MODEL / "latent_pitch_context.csv")
    correction_f = apply_linear_residual(residual_x, MODEL / "psych_latent_meta.npz")
    p_079a = np.where(futures_24, p_079a + correction_f, p_078a if 'p_078a' in locals() else p_079a)

    # Regular Split Profile + LightGBM R-Expert (w_lgb=0.05 for 079A)
    split_profile = pd.read_csv(MODEL / "split_profile.csv", dtype={"entity_value": str, "context_value": str})
    split_x = apply_split_profile(val_24, split_profile)
    split_correction = apply_linear_split(split_x, MODEL / "split_residual_meta.npz")
    p_split = p_079a + split_correction
    
    lgbm_model = lgb.Booster(model_file=str(MODEL / 'r_expert_lgbm.txt'))
    x3_lgb = x3.copy()
    for col in CAT_V2:
        if col in x3_lgb.columns:
            x3_lgb[col] = x3_lgb[col].astype('category')
    res_lgbm = lgbm_model.predict(x3_lgb)
    p_lgbm = np.clip(b3 + res_lgbm + 0.0052, 1e-6, 1.0 - 1e-6)
    
    p_079a_final = np.where(regular_24, 0.95 * p_split + 0.05 * p_lgbm, p_079a)
    p_079a_final = np.clip(p_079a_final, 1e-5, 1 - 1e-5)

    bss_079a = bss(y_24, p_079a_final)
    print(f"=== 079A Base Validation Score on 2024 ===")
    print(f"079A Champion BSS: {bss_079a:.4f}")
    
    # 2. Extract v5 Orthogonal Features
    print("\nExtracting v5 Orthogonal Features...")
    v5_feat = build_v5_orthogonal_features(val_24)
    print(f"Extracted {len(v5_feat.columns)} v5 features: {list(v5_feat.columns)}")
    
    # 3. Train v5 Orthogonal Residual Corrector strictly on Regular season (game_type == 'R')
    train_22_23 = raw.loc[raw.season.isin([2022, 2023]) & (raw.game_type != "F")].copy().reset_index(drop=True)
    v5_train = build_v5_orthogonal_features(train_22_23)
    y_tr = train_22_23.control_success.to_numpy(float)
    
    # Simple Ridge / CatBoost orthogonal residual predictor
    ortho_model = CatBoostRegressor(iterations=100, depth=4, learning_rate=0.03, l2_leaf_reg=10, random_seed=42, verbose=False)
    # Target: residual against global prior
    ortho_model.fit(v5_train, y_tr - 0.523766)
    
    v5_val_reg = v5_feat.loc[regular_24]
    p_ortho_reg = ortho_model.predict(v5_val_reg)
    v5_corr = p_ortho_reg - np.mean(p_ortho_reg)
    
    print("\n=== Grid Search: Zero-Centered v5 Orthogonal Residual Blend Weight (080A) ===")
    best_w = 0.0
    best_bss = bss_079a
    
    for w in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10, 0.12, 0.15]:
        p_cand = p_079a_final.copy()
        p_cand[regular_24] = p_cand[regular_24] + w * v5_corr
        p_cand = np.clip(p_cand, 1e-5, 1 - 1e-5)
        score = bss(y_24, p_cand)
        diff = score - bss_079a
        print(f"Zero-Centered v5 Weight {w:.2f} (strictly R) -> 2024 BSS: {score:8.4f} (diff vs 079A: {diff:+8.4f})")
        if score > best_bss:
            best_bss = score
            best_w = w
            
    print(f"\nBest 080A Result: v5 Weight={best_w:.2f} -> 2024 BSS = {best_bss:.4f} (+{best_bss - bss_079a:.4f} gain!)")

if __name__ == '__main__':
    main()
