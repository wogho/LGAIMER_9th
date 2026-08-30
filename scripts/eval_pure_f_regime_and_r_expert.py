#!/usr/bin/env python3
"""Evaluate Pure 16-Model Futures Regime + 051A R-Split + 063A LightGBM R-Expert on 2024."""
import gc, json, os, sys, time
from pathlib import Path
from catboost import CatBoostClassifier, CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CAND_DIR = ROOT / 'candidate/REF4-R-SPECIFIC-SPLIT-FPSYCH-055A'
sys.path.insert(0, str(CAND_DIR))

from src.preprocessing_v2 import build_v2_features, build_v3_features
from src.split_priors import apply_split_priors

def bss(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-5, 1 - 1e-5)
    r = float(y.mean())
    ref = r * (1.0 - r)
    return 1e5 * (1.0 - np.mean((p - y) ** 2) / ref)

def main():
    t0 = time.time()
    print("=== Loading 2024 Validation Rows ===")
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
    
    print("Building v2 and v3 features on 2024...")
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

    print("Computing 6-Seed Base predictions...")
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

    print("Applying 16 Futures CatBoost models...")
    f2 = f_reg_mean("f_v2_all", 4, x2, base2)
    predictions[0] = np.where(futures, predictions[0] + regime["v2_scale"] * (f2 - predictions[0]), predictions[0])
    f55 = f_reg_mean("f_v355_recent", 6, x3, base3)
    predictions[1] = np.where(futures, predictions[1] + regime["v355_scale"] * (f55 - predictions[1]), predictions[1])
    f30a = f_reg_mean("f_v330_all", 4, x3, base3)
    f30r = f_reg_mean("f_v330_recent", 2, x3, base3)
    recent_inner = predictions[2] + regime["v330_recent_inner_scale"] * (f30r - predictions[2])
    f30 = regime["v330_all_weight"] * f30a + (1 - regime["v330_all_weight"]) * recent_inner
    predictions[2] = np.where(futures, predictions[2] + regime["v330_scale"] * (f30 - predictions[2]), predictions[2])

    print("Computing subtype risks with Futures subtype classifiers...")
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

    print("Computing hierarchical meta stack...")
    main_p = np.average(np.vstack(predictions), axis=0, weights=meta["main_weights"])
    z = np.column_stack([main_p] + risks)
    p = meta["stack_intercept"] + z @ np.asarray(meta["stack_coefficients"])

    # 1군: 051A R-Specific Split Priors
    print("Applying 051A R-Specific Split Priors on 1군...")
    tables = pd.read_pickle(MODEL / "split_prior_tables.pkl")
    split_corr = apply_split_priors(val_2024, tables)
    p = p + np.where(is_r, split_corr, 0.0)

    # 1군: 063A LightGBM R-Expert
    print("Applying 063A LightGBM R-Expert on 1군...")
    lgbm_model = lgb.Booster(model_file=str(ROOT / 'model/REF4-TRAINONLY-R-LGBM-CONSERVATIVE-063A/r_expert_lgbm_2024.txt'))
    for col in CAT_V2:
        if col in x3.columns:
            x3[col] = x3[col].astype('category')
    res_lgbm = lgbm_model.predict(x3)
    p_lgbm_r = np.clip(base3 + res_lgbm + 0.0052, 1e-6, 1.0 - 1e-6)
    
    # Global shift
    p_base_shifted = p + 0.0052
    p_final = np.where(is_r, 0.98 * p_base_shifted + 0.02 * p_lgbm_r, p_base_shifted)
    p_final = np.clip(p_final, 1e-5, 1.0 - 1e-5)

    print("\n=== Validation Results on 2024 ===")
    print(f"Overall BSS: {bss(y_val, p_final):.4f} (Brier: {np.mean((p_final - y_val)**2):.7f})")
    print(f"1군 (R) BSS: {bss(y_val[is_r], p_final[is_r]):.4f} (Brier: {np.mean((p_final[is_r] - y_val[is_r])**2):.7f})")
    print(f"2군 (F) BSS: {bss(y_val[is_f], p_final[is_f]):.4f} (Brier: {np.mean((p_final[is_f] - y_val[is_f])**2):.7f})")
    print(f"Mean Prediction: {p_final.mean():.6f} vs True Mean: {y_val.mean():.6f} (Gap: {p_final.mean() - y_val.mean():+.6f})")
    print(f"Completed in {time.time() - t0:.2f}s")

if __name__ == '__main__':
    main()
