#!/usr/bin/env python3
"""Multi-Season (2022, 2023, 2024) Holdout Check for 108A Multi-Model Residual Booster."""
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
from src.v54_per_season_asof_75_features import build_per_season_priors, build_v54_per_season_asof_75_features
from src.preprocessing_v2 import build_v2_features, build_v3_features, CAT_V2
from src.adaptive_gate import build_gate_features

def brier_score(y_true, y_pred):
    return np.mean((y_pred - y_true) ** 2)

def brier_skill_score(y_true, y_pred):
    r = np.mean(y_true)
    bs_ref = r * (1 - r)
    bs = brier_score(y_true, y_pred)
    return max(0.0, 100000.0 * (1.0 - bs / bs_ref))

def build_leverage_features(df: pd.DataFrame) -> pd.DataFrame:
    li = df["li"].fillna(0.98).to_numpy(float) if "li" in df.columns else np.full(len(df), 0.98)
    b = df["balls_before"].fillna(0).to_numpy(float) if "balls_before" in df.columns else np.zeros(len(df))
    s = df["strikes_before"].fillna(0).to_numpy(float) if "strikes_before" in df.columns else np.zeros(len(df))
    count_diff = b - s
    inn = df["inning"].fillna(1).to_numpy(float) if "inning" in df.columns else np.ones(len(df))
    score_diff = np.abs(df["score_diff"].fillna(0).to_numpy(float)) if "score_diff" in df.columns else np.zeros(len(df))
    late_close = ((inn >= 7) & (score_diff <= 2)).astype(float)
    is_high_li = (li >= 1.5).astype(float)
    return pd.DataFrame({
        "is_high_leverage": is_high_li,
        "li_count_diff": li * count_diff,
        "li_late_close": li * late_close
    }, index=df.index)

def predict_103a_full(df, MODEL, meta, priors, ps, bs, ms, tm, seeds, regime):
    x2, base2 = build_v2_features(df, meta["prior"], ps, tm)
    x3, base3 = build_v3_features(df, meta["prior"], ps, bs, ms, tm)
    
    def load_reg(stem):
        return [CatBoostRegressor().load_model(str(MODEL / f"{stem}_seed{s}.cbm")) for s in seeds]
        
    predictions = []
    for stem, x, base in [
        ("v2_decay55", x2, base2),
        ("v3_decay55", x3, base3),
        ("v3_decay30", x3, base3),
    ]:
        member = [np.clip(base + m.predict(x), 1e-6, 1 - 1e-6) for m in load_reg(stem)]
        predictions.append(np.mean(member, axis=0))
        
    futures = df["game_type"].eq("F").to_numpy()
    regular = ~futures
    
    def f_reg_mean(stem, count, x, base):
        member = []
        for j in range(count):
            m = CatBoostRegressor()
            m.load_model(str(MODEL / f"{stem}_{j}.cbm"))
            member.append(np.clip(base + m.predict(x), 1e-6, 1 - 1e-6))
        return np.mean(member, axis=0)
        
    if futures.any():
        f2 = f_reg_mean("f_v2_all", 4, x2, base2)
        predictions[0] = np.where(futures, predictions[0] + regime["v2_scale"] * (f2 - predictions[0]), predictions[0])
        f55 = f_reg_mean("f_v355_recent", 6, x3, base3)
        predictions[1] = np.where(futures, predictions[1] + regime["v355_scale"] * (f55 - predictions[1]), predictions[1])
        f30a = f_reg_mean("f_v330_all", 4, x3, base3)
        f30r = f_reg_mean("f_v330_recent", 2, x3, base3)
        recent_inner = predictions[2] + regime["v330_recent_inner_scale"] * (f30r - predictions[2])
        f30 = regime["v330_all_weight"] * f30a + (1 - regime["v330_all_weight"]) * recent_inner
        predictions[2] = np.where(futures, predictions[2] + regime["v330_scale"] * (f30 - predictions[2]), predictions[2])
        
    risks = []
    for name in ("middle", "wild", "reverse"):
        stems = [f"subtype_{name}_seed{s}.cbm" for s in seeds]
        member = []
        for filename in stems:
            m = CatBoostClassifier()
            m.load_model(str(MODEL / filename))
            member.append(m.predict_proba(x3)[:, 1])
        risk = np.mean(member, axis=0)
        if futures.any():
            fm = CatBoostClassifier()
            fm.load_model(str(MODEL / f"f_subtype_{name}.cbm"))
            fr = fm.predict_proba(x3)[:, 1]
            risk = np.where(futures, risk + regime["subtype_scale"] * (fr - risk), risk)
        risks.append(risk)
        
    main_p = np.average(np.vstack(predictions), axis=0, weights=meta["main_weights"])
    z = np.column_stack([main_p] + risks)
    p = meta["stack_intercept"] + z @ np.asarray(meta["stack_coefficients"])
    
    if meta.get("adaptive_gate", False):
        gate_x = build_gate_features(df, predictions, risks, np.clip(p, 1e-6, 1 - 1e-6))
        gate = CatBoostRegressor()
        gate.load_model(str(MODEL / "adaptive_gate.cbm"))
        gate_pred = gate.predict(gate_x)
        bias_offset = float(meta.get("gate_bias_offset", 0.0))
        p = p + float(meta.get("gate_scale", 0.08)) * (gate_pred - bias_offset)
        
    p = p + float(meta.get("global_shift", 0.0052))
    
    if futures.any():
        from src.psych_latent import build_production_features, apply_linear_residual
        residual_x = build_production_features(
            df, MODEL / "psych_profile.pkl", MODEL / "latent_pitch_context.csv"
        )
        correction = apply_linear_residual(residual_x, MODEL / "psych_latent_meta.npz")
        p = np.where(futures, p + correction, p)
        
    if regular.any():
        from src.entity_context_split import apply_split_profile, apply_linear_split
        split_profile = pd.read_csv(
            MODEL / "split_profile.csv", dtype={"entity_value": str, "context_value": str}
        )
        split_x = apply_split_profile(df, split_profile)
        split_correction = apply_linear_split(split_x, MODEL / "split_residual_meta.npz")
        p_split = p + split_correction
        
        lgbm_model = lgb.Booster(model_file=str(MODEL / 'r_expert_lgbm.txt'))
        x3_lgb = x3.copy()
        for col in CAT_V2:
            if col in x3_lgb.columns:
                x3_lgb[col] = x3_lgb[col].astype('category')
        res_lgbm = lgbm_model.predict(x3_lgb)
        p_lgbm = np.clip(base3 + res_lgbm + 0.0052, 1e-6, 1.0 - 1e-6)
        
        w_lgb = float(meta.get("r_expert_lgbm_weight", 0.05))
        p = np.where(regular, (1.0 - w_lgb) * p_split + w_lgb * p_lgbm, p)
        
    v54_feat, _ = build_v54_per_season_asof_75_features(
        df, profile_path=MODEL / "team_asof_profile.json", priors=priors, prior=float(meta.get("prior", 0.523766))
    )
    lev_df = build_leverage_features(df)
    v54_feat = pd.concat([v54_feat, lev_df], axis=1)
    
    n_p_raw = df["asof_pitcher_n"].fillna(0).to_numpy(float) if "asof_pitcher_n" in df.columns else np.zeros(len(df))
    p_rate_raw = df["asof_pitcher_success_rate"].fillna(0.523766).to_numpy(float) if "asof_pitcher_success_rate" in df.columns else np.full(len(df), 0.523766)
    prev1_raw = df["asof_pitcher_prev1_game_success_rate"].to_numpy(float) if "asof_pitcher_prev1_game_success_rate" in df.columns else p_rate_raw
    prev1_raw = np.where(np.isnan(prev1_raw), p_rate_raw, prev1_raw)
    
    overall_prior = float(meta.get("prior", 0.523766))
    k_prior = 25.0
    rel_weight = n_p_raw / (n_p_raw + k_prior)
    p_shrunk = overall_prior + rel_weight * (p_rate_raw - overall_prior)
    p_hb_hier = np.clip(0.70 * p_shrunk + 0.30 * prev1_raw, 0.05, 0.95)
    
    v54_seeds = meta.get("v54_seeds", [42, 1, 2, 3, 4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150])
    res_preds = []
    for s in v54_seeds:
        cb_m = CatBoostRegressor()
        cb_m.load_model(str(MODEL / f"deep_cb_l2_seed{s}.cbm"))
        res_preds.append(cb_m.predict(v54_feat))
        lgb_m = lgb.Booster(model_file=str(MODEL / f"deep_lgb_l2_seed{s}.txt"))
        res_preds.append(lgb_m.predict(v54_feat))
        
    p_deep_resid = np.mean(res_preds, axis=0)
    p_deep_reconstructed = np.clip(p_hb_hier + p_deep_resid, 1e-5, 1 - 1e-5)
    deep_mean_offset = float(meta.get("deep_mean_offset", 0.514025))
    deep_corr = p_deep_reconstructed - deep_mean_offset
    
    # 103A Refined Call
    cb_call = CatBoostClassifier()
    cb_call.load_model(str(MODEL / "refined_call_expert.cbm"))
    x3_cb = x3.copy()
    for c in CAT_V2:
        if c in x3_cb.columns:
            x3_cb[c] = x3_cb[c].astype(str)
    call_probs = cb_call.predict_proba(x3_cb)
    call_corr = call_probs[:, 0] - float(meta.get("call_mean_offset", 0.523766))
    
    high_pocket_mask = regular & (n_p_raw < 15) & (p_rate_raw > 0.65)
    low_pocket_mask = regular & (n_p_raw < 15) & (p_rate_raw < 0.40)
    
    p_103 = np.where(regular, p + 0.08 * deep_corr + 0.04 * call_corr, p)
    p_103 = np.where(high_pocket_mask, p_103 - 0.012, p_103)
    p_103 = np.where(low_pocket_mask, p_103 + 0.008, p_103)
    p_103 = np.clip(p_103, 1e-5, 1 - 1e-5)
    return p_103

def main():
    print("=== Multi-Season Validation for 108A Multi-Model Residual Booster ===")
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
    
    # Train 9-model ensemble (3 CatBoost + 3 LightGBM + 3 XGBoost)
    r_seeds = [42, 1, 2]
    boosters = []
    for s in r_seeds:
        cb = CatBoostRegressor(iterations=180, depth=5, learning_rate=0.03, l2_leaf_reg=15.0, random_seed=s, verbose=False)
        cb.fit(X_tr_61, y_resid_reg_tr)
        boosters.append(cb)
        
        lgb_m = lgb.LGBMRegressor(n_estimators=140, learning_rate=0.03, num_leaves=31, min_child_samples=50, subsample=0.8, colsample_bytree=0.8, random_state=s, n_jobs=-1, verbose=-1)
        lgb_m.fit(X_tr_61, y_resid_reg_tr)
        boosters.append(lgb_m)
        
        xgb_m = XGBRegressor(n_estimators=150, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=s, n_jobs=-1, tree_method="hist")
        xgb_m.fit(X_tr_61, y_resid_reg_tr)
        boosters.append(xgb_m)
        
    print(f"Trained {len(boosters)} Multi-Model Boosters")
    
    for s in [2022, 2023, 2024]:
        val_df = raw[raw["season"] == s].copy().reset_index(drop=True)
        y_val = val_df["control_success"].to_numpy(float)
        p_103_val = predict_103a_full(val_df, MODEL_107, meta_107, priors, ps, bs, ms, tm, seeds, regime)
        reg_val = val_df["game_type"] != "F"
        
        X_val_61, _ = build_v5_deep_61_features(val_df[reg_val].reset_index(drop=True), profile_path=prof_path, prior=0.523766)
        prev1_p_val = val_df.loc[reg_val, "asof_pitcher_prev1_game_success_rate"].fillna(0.523766).to_numpy(float)
        b_rate_val = val_df.loc[reg_val, "asof_batter_success_rate"].fillna(0.523766).to_numpy(float)
        X_val_61["form_gap"] = prev1_p_val - b_rate_val
        X_val_61["anchor_p"] = p_103_val[reg_val]
        
        p_res = np.mean([m.predict(X_val_61) for m in boosters], axis=0)
        
        bss_103 = brier_skill_score(y_val, p_103_val)
        
        p_108 = p_103_val.copy()
        p_108[reg_val] = p_108[reg_val] + 0.07 * p_res
        p_108 = np.clip(p_108, 1e-5, 1 - 1e-5)
        bss_108 = brier_skill_score(y_val, p_108)
        delta = bss_108 - bss_103
        
        print(f"Season {s} ({len(val_df):,} rows) -> 103A BSS: {bss_103:.4f} | 108A BSS: {bss_108:.4f} (Δ {delta:+.4f} pt)")

if __name__ == "__main__":
    main()
