#!/usr/bin/env python3
"""Test F-Regime without psych overcorrection on 2024 validation set."""
import gc, json, os, sys, time
from pathlib import Path
from catboost import CatBoostClassifier, CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CAND_DIR = ROOT / 'candidate/REF4-R-SPECIFIC-SPLIT-FPSYCH-055A'
sys.path.insert(0, str(CAND_DIR))

from src.preprocessing_v2 import build_v2_features, build_v3_features, CAT_V2
from src.entity_context_split import apply_split_profile, apply_linear_split
from src.psych_latent import build_production_features, apply_linear_residual

def bss(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-5, 1 - 1e-5)
    r = float(y.mean())
    ref = r * (1.0 - r)
    return 1e5 * (1.0 - np.mean((p - y) ** 2) / ref)

def main():
    t0 = time.time()
    print("=== Loading 2024 Validation Set ===")
    raw = pd.read_csv(ROOT / 'data/train.csv', low_memory=False)
    val_2024 = raw.loc[raw.season == 2024].copy().reset_index(drop=True)
    y_val = val_2024.control_success.to_numpy(float)
    is_r = (val_2024.game_type == 'R').to_numpy()
    is_f = (val_2024.game_type == 'F').to_numpy()
    
    MODEL = CAND_DIR / 'model'
    meta = json.loads((MODEL / "manifest.json").read_text(encoding="utf-8"))
    ps = pd.read_pickle(MODEL / "pitcher_snapshots.pkl")
    bs = pd.read_pickle(MODEL / "batter_snapshots.pkl")
    ms = pd.read_pickle(MODEL / "pitchmix_snapshots.pkl")
    tm = str(MODEL / "trackman_prior_features.csv")
    
    print("Building features...")
    x2, base2 = build_v2_features(val_2024, meta["prior"], ps, tm)
    x3, base3 = build_v3_features(val_2024, meta["prior"], ps, bs, ms, tm)
    
    seeds = meta.get("seeds", [260802, 260803, 260804, 260805, 260806, 260807])
    
    def load_reg(stem):
        names = [f"{stem}_seed{s}.cbm" for s in seeds]
        out = []
        for filename in names:
            m = CatBoostRegressor()
            m.load_model(str(MODEL / filename))
            out.append(m)
        return out

    print("Computing 6-seed base predictions...")
    predictions = []
    for stem, x, base in [
        ("v2_decay55", x2, base2),
        ("v3_decay55", x3, base3),
        ("v3_decay30", x3, base3),
    ]:
        member = [np.clip(base + m.predict(x), 1e-6, 1 - 1e-6) for m in load_reg(stem)]
        predictions.append(np.mean(member, axis=0))

    regime = json.loads((MODEL / "f_regime_meta.json").read_text())
    futures = is_f
    
    def f_reg_mean(stem, count, x, base):
        member = []
        for j in range(count):
            m = CatBoostRegressor()
            m.load_model(str(MODEL / f"{stem}_{j}.cbm"))
            member.append(np.clip(base + m.predict(x), 1e-6, 1 - 1e-6))
        return np.mean(member, axis=0)

    print("Computing F-regime modifications on F-rows...")
    f2 = f_reg_mean("f_v2_all", 4, x2, base2)
    predictions[0] = np.where(futures, predictions[0] + regime["v2_scale"] * (f2 - predictions[0]), predictions[0])
    f55 = f_reg_mean("f_v355_recent", 6, x3, base3)
    predictions[1] = np.where(futures, predictions[1] + regime["v355_scale"] * (f55 - predictions[1]), predictions[1])
    f30a = f_reg_mean("f_v330_all", 4, x3, base3)
    f30r = f_reg_mean("f_v330_recent", 2, x3, base3)
    recent_inner = predictions[2] + regime["v330_recent_inner_scale"] * (f30r - predictions[2])
    f30 = regime["v330_all_weight"] * f30a + (1 - regime["v330_all_weight"]) * recent_inner
    predictions[2] = np.where(futures, predictions[2] + regime["v330_scale"] * (f30 - predictions[2]), predictions[2])

    print("Computing subtype classifiers...")
    risks = []
    for name in ("middle", "wild", "reverse"):
        stems = [f"subtype_{name}_seed{s}.cbm" for s in seeds]
        member = []
        for filename in stems:
            m = CatBoostClassifier()
            m.load_model(str(MODEL / filename))
            member.append(m.predict_proba(x3)[:, 1])
        risk = np.mean(member, axis=0)
        fm = CatBoostClassifier()
        fm.load_model(str(MODEL / f"f_subtype_{name}.cbm"))
        fr = fm.predict_proba(x3)[:, 1]
        risk = np.where(futures, risk + regime["subtype_scale"] * (fr - risk), risk)
        risks.append(risk)

    print("Computing meta hierarchical stack...")
    main_p = np.average(np.vstack(predictions), axis=0, weights=meta["main_weights"])
    z = np.column_stack([main_p] + risks)
    p_stack = meta["stack_intercept"] + z @ np.asarray(meta["stack_coefficients"])

    # Global shift
    p_stack = p_stack + 0.0052

    # Option 1: 055A / 064A behavior (with psych on F)
    residual_x = build_production_features(val_2024, MODEL / "psych_profile.pkl", MODEL / "latent_pitch_context.csv")
    psych_corr = apply_linear_residual(residual_x, MODEL / "psych_latent_meta.npz")
    
    split_profile = pd.read_csv(MODEL / "split_profile.csv", dtype={"entity_value": str, "context_value": str})
    split_x = apply_split_profile(val_2024, split_profile)
    split_corr = apply_linear_split(split_x, MODEL / "split_residual_meta.npz")

    # LightGBM R-Expert
    lgbm_model = lgb.Booster(model_file=str(ROOT / 'model/REF4-TRAINONLY-R-LGBM-CONSERVATIVE-063A/r_expert_lgbm_2024.txt'))
    for col in CAT_V2:
        if col in x3.columns:
            x3[col] = x3[col].astype('category')
    res_lgbm = lgbm_model.predict(x3)
    p_lgbm_r = np.clip(base3 + res_lgbm + 0.0052, 1e-6, 1.0 - 1e-6)

    # Variant A: Current 064A (psych on F, split on R, LGBM on R)
    p_A = np.where(is_f, p_stack + psych_corr, p_stack + split_corr)
    p_A[is_r] = 0.98 * p_A[is_r] + 0.02 * p_lgbm_r[is_r]
    p_A = np.clip(p_A, 1e-5, 1 - 1e-5)

    # Variant B: Pure F-Regime (NO psych on F, split on R, LGBM on R)
    p_B = np.where(is_f, p_stack, p_stack + split_corr)
    p_B[is_r] = 0.98 * p_B[is_r] + 0.02 * p_lgbm_r[is_r]
    p_B = np.clip(p_B, 1e-5, 1 - 1e-5)

    # Variant C: Scaled psych on F (0.50 * psych on F, split on R, LGBM on R)
    p_C = np.where(is_f, p_stack + 0.50 * psych_corr, p_stack + split_corr)
    p_C[is_r] = 0.98 * p_C[is_r] + 0.02 * p_lgbm_r[is_r]
    p_C = np.clip(p_C, 1e-5, 1 - 1e-5)

    print("\n=== Comparison on 2024 Validation Set ===")
    print(f"Variant A (Current 064A): Overall BSS = {bss(y_val, p_A):.4f} | R: {bss(y_val[is_r], p_A[is_r]):.4f} | F: {bss(y_val[is_f], p_A[is_f]):.4f}")
    print(f"Variant B (Pure F-Regime): Overall BSS = {bss(y_val, p_B):.4f} | R: {bss(y_val[is_r], p_B[is_r]):.4f} | F: {bss(y_val[is_f], p_B[is_f]):.4f}")
    print(f"Variant C (50% Psych F):  Overall BSS = {bss(y_val, p_C):.4f} | R: {bss(y_val[is_r], p_C[is_r]):.4f} | F: {bss(y_val[is_f], p_C[is_f]):.4f}")
    print(f"Mean comparison: True={y_val.mean():.6f}, A={p_A.mean():.6f}, B={p_B.mean():.6f}, C={p_C.mean():.6f}")

if __name__ == '__main__':
    main()
