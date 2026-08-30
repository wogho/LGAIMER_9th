#!/usr/bin/env python3
"""Test Repo 6 Standalone Inference and R-Residual Integration with 079A Backbone."""
import json, os, sys, time, joblib
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
REPO6_DIR = ROOT / 'github_reference/6번 레포'
MODEL_079_DIR = ROOT / 'model/REF4-ADAPTIVE-CHANNEL-OPT-071A/production_package/model'

sys.path.insert(0, str(ROOT / 'github_reference/4번 레포'))
sys.path.insert(0, str(ROOT / 'candidate/REF4-R-SPECIFIC-SPLIT-FPSYCH-055A/src'))
sys.path.insert(0, str(ROOT / 'model/REF4-ADAPTIVE-CHANNEL-OPT-071A/production_package/src'))
sys.path.insert(0, str(ROOT / 'github_reference/4번 레포/src'))

from preprocessing_v2 import build_v2_features, build_v3_features, CAT_V2
from adaptive_gate import build_gate_features
from entity_context_split import apply_split_profile, apply_linear_split
from psych_latent import build_production_features, apply_linear_residual

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
    meta = json.loads((MODEL_079_DIR / "manifest.json").read_text(encoding="utf-8"))
    ps = pd.read_pickle(MODEL_079_DIR / "pitcher_snapshots.pkl")
    bs = pd.read_pickle(MODEL_079_DIR / "batter_snapshots.pkl")
    ms = pd.read_pickle(MODEL_079_DIR / "pitchmix_snapshots.pkl")
    tm = str(MODEL_079_DIR / "trackman_prior_features.csv")
    seeds = meta.get("seeds", [260802, 260803, 260804, 260805, 260806, 260807])
    
    x2, b2 = build_v2_features(val_24, meta["prior"], ps, tm)
    x3, b3 = build_v3_features(val_24, meta["prior"], ps, bs, ms, tm)
    
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
        
    regime = json.loads((MODEL_079_DIR / "f_regime_meta.json").read_text())
    
    def f_reg_mean(stem, count, x, base):
        member = []
        for j in range(count):
            m = CatBoostRegressor()
            m.load_model(str(MODEL_079_DIR / f"{stem}_{j}.cbm"))
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
            m.load_model(str(MODEL_079_DIR / filename))
            member.append(m.predict_proba(x3)[:, 1])
        risk = np.mean(member, axis=0)
        if futures_24.any():
            fm = CatBoostClassifier()
            fm.load_model(str(MODEL_079_DIR / f"f_subtype_{name}.cbm"))
            fr = fm.predict_proba(x3)[:, 1]
            risk = np.where(futures_24, risk + regime["subtype_scale"] * (fr - risk), risk)
        risks.append(risk)

    main_p = np.average(np.vstack(predictions), axis=0, weights=meta["main_weights"])
    z = np.column_stack([main_p] + risks)
    p_stack = meta["stack_intercept"] + z @ np.asarray(meta["stack_coefficients"])

    # Adaptive Gate (gate_scale = 0.08)
    gate_x = build_gate_features(val_24, predictions, risks, np.clip(p_stack, 1e-6, 1 - 1e-6))
    gate = CatBoostRegressor()
    gate.load_model(str(MODEL_079_DIR / "adaptive_gate.cbm"))
    gate_pred = gate.predict(gate_x)
    bias_offset = float(meta.get("gate_bias_offset", 0.0))
    gate_clean = gate_pred - bias_offset
    
    p_079a = p_stack + 0.08 * gate_clean + 0.0052
    
    # Futures Psych Latent
    residual_x = build_production_features(val_24, MODEL_079_DIR / "psych_profile.pkl", MODEL_079_DIR / "latent_pitch_context.csv")
    correction_f = apply_linear_residual(residual_x, MODEL_079_DIR / "psych_latent_meta.npz")
    p_079a = np.where(futures_24, p_079a + correction_f, p_079a)

    # Regular Split Profile + LightGBM R-Expert (w_lgb=0.05)
    split_profile = pd.read_csv(MODEL_079_DIR / "split_profile.csv", dtype={"entity_value": str, "context_value": str})
    split_x = apply_split_profile(val_24, split_profile)
    split_correction = apply_linear_split(split_x, MODEL_079_DIR / "split_residual_meta.npz")
    p_split = p_079a + split_correction
    
    lgbm_model = lgb.Booster(model_file=str(MODEL_079_DIR / 'r_expert_lgbm.txt'))
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

    # 2. Run Repo 6 Model Inference
    print("\nLoading Repo 6 Tuned Models...")
    cb_6 = joblib.load(REPO6_DIR / 'model/catboost_tuned.pkl')
    lgb_6 = joblib.load(REPO6_DIR / 'model/lightgbm_tuned.pkl')
    xgb_6 = joblib.load(REPO6_DIR / 'model/xgboost_tuned.pkl')
    
    print("Pre-processing test features for Repo 6...")
    feat_24 = val_24.copy()
    feat_24["is_same_hand"] = (feat_24["pitcher_hand"] == feat_24["batter_hand"]).astype(int)
    
    p_cb = cb_6.predict_proba(feat_24)[:, 1] if hasattr(cb_6, "predict_proba") else cb_6.predict(feat_24)
    p_lgb = lgb_6.predict_proba(feat_24)[:, 1] if hasattr(lgb_6, "predict_proba") else lgb_6.predict(feat_24)
    p_xgb = xgb_6.predict_proba(feat_24)[:, 1] if hasattr(xgb_6, "predict_proba") else xgb_6.predict(feat_24)
    
    p_repo6 = (p_cb + p_lgb + p_xgb) / 3.0
    print(f"Repo 6 Model Standalone BSS on 2024: {bss(y_24, p_repo6):.4f}")
    
    print(f"Correlation between 079A and Repo 6 predictions: {np.corrcoef(p_079a_final, p_repo6)[0,1]:.4f}")

    # 3. Test R-Decoupled Residual Correction using Repo 6 predictions
    print("\n=== Grid Search: 079A + Repo 6 Orthogonal Residual (strictly R) ===")
    p_repo6_zero = p_repo6 - np.mean(p_repo6)
    
    best_w = 0.0
    best_bss = bss_079a
    
    for w in [0.00, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]:
        p_cand = p_079a_final.copy()
        p_cand[regular_24] = p_cand[regular_24] + w * p_repo6_zero[regular_24]
        p_cand = np.clip(p_cand, 1e-5, 1 - 1e-5)
        score = bss(y_24, p_cand)
        diff = score - bss_079a
        print(f"Repo 6 Residual Weight {w:.2f} (strictly R) -> 2024 BSS: {score:8.4f} (diff vs 079A: {diff:+8.4f})")
        if score > best_bss:
            best_bss = score
            best_w = w

    print(f"\nBest Decoupled Result: Weight={best_w:.2f} -> 2024 BSS = {best_bss:.4f} (+{best_bss - bss_079a:.4f} gain!)")

if __name__ == '__main__':
    main()
