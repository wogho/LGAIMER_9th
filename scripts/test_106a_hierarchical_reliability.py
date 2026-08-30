#!/usr/bin/env python3
"""Evaluate 6번/7번 레포 1146.3952pt mechanisms on 103A Champion Backbone."""
import gc, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'model/REF4-DEEP-HIERARCHICAL-103A/production_package'))

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

def main():
    print("Loading data for validation...")
    raw = pd.read_csv(ROOT / "data/train.csv", low_memory=False)
    
    val_2024 = raw[raw["season"] == 2024].copy().reset_index(drop=True)
    y_val = val_2024["control_success"].to_numpy(float)
    
    MODEL = ROOT / "model/REF4-DEEP-HIERARCHICAL-103A/production_package/model"
    meta = json.loads((MODEL / "manifest.json").read_text(encoding="utf-8"))
    priors = json.loads((MODEL / "per_season_priors.json").read_text(encoding="utf-8"))
    
    ps = pd.read_pickle(MODEL / "pitcher_snapshots.pkl")
    bs = pd.read_pickle(MODEL / "batter_snapshots.pkl")
    ms = pd.read_pickle(MODEL / "pitchmix_snapshots.pkl")
    tm = str(MODEL / "trackman_prior_features.csv")
    
    print("Computing 103A features on 2024 Holdout...")
    x2, base2 = build_v2_features(val_2024, meta["prior"], ps, tm)
    x3, base3 = build_v3_features(val_2024, meta["prior"], ps, bs, ms, tm)
    
    seeds = meta.get("seeds", [260802, 260803, 260804, 260805, 260806, 260807])
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
        
    regime = json.loads((MODEL / "f_regime_meta.json").read_text())
    futures = val_2024["game_type"].eq("F").to_numpy()
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
        gate_x = build_gate_features(val_2024, predictions, risks, np.clip(p, 1e-6, 1 - 1e-6))
        gate = CatBoostRegressor()
        gate.load_model(str(MODEL / "adaptive_gate.cbm"))
        gate_pred = gate.predict(gate_x)
        bias_offset = float(meta.get("gate_bias_offset", 0.0))
        p = p + float(meta.get("gate_scale", 0.08)) * (gate_pred - bias_offset)
        
    p = p + float(meta.get("global_shift", 0.0052))
    
    if futures.any():
        from src.psych_latent import build_production_features, apply_linear_residual
        residual_x = build_production_features(
            val_2024, MODEL / "psych_profile.pkl", MODEL / "latent_pitch_context.csv"
        )
        correction = apply_linear_residual(residual_x, MODEL / "psych_latent_meta.npz")
        p = np.where(futures, p + correction, p)
        
    if regular.any():
        from src.entity_context_split import apply_split_profile, apply_linear_split
        split_profile = pd.read_csv(
            MODEL / "split_profile.csv", dtype={"entity_value": str, "context_value": str}
        )
        split_x = apply_split_profile(val_2024, split_profile)
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
        
    # Pre-build v54 features & L2 deep models
    v54_feat, _ = build_v54_per_season_asof_75_features(
        val_2024, profile_path=MODEL / "team_asof_profile.json", priors=priors, prior=float(meta.get("prior", 0.523766))
    )
    lev_df = build_leverage_features(val_2024)
    v54_feat = pd.concat([v54_feat, lev_df], axis=1)
    
    n_p_raw = val_2024["asof_pitcher_n"].fillna(0).to_numpy(float) if "asof_pitcher_n" in val_2024.columns else np.zeros(len(val_2024))
    p_rate_raw = val_2024["asof_pitcher_success_rate"].fillna(0.523766).to_numpy(float) if "asof_pitcher_success_rate" in val_2024.columns else np.full(len(val_2024), 0.523766)
    prev1_raw = val_2024["asof_pitcher_prev1_game_success_rate"].to_numpy(float) if "asof_pitcher_prev1_game_success_rate" in val_2024.columns else p_rate_raw
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
    
    p_anchor_102 = np.where(regular, p + 0.08 * deep_corr, p)
    
    # 103A Refined Call
    cb_call = CatBoostClassifier()
    cb_call.load_model(str(MODEL / "refined_call_expert.cbm"))
    x3_cb = x3.copy()
    for c in CAT_V2:
        if c in x3_cb.columns:
            x3_cb[c] = x3_cb[c].astype(str)
    call_probs = cb_call.predict_proba(x3_cb)
    call_corr = call_probs[:, 0] - float(meta.get("call_mean_offset", 0.523766))
    
    p_103 = np.where(regular, p_anchor_102 + 0.04 * call_corr, p_anchor_102)
    
    # Pocket Post-Processor
    high_pocket_mask = regular & (n_p_raw < 15) & (p_rate_raw > 0.65)
    p_103 = np.where(high_pocket_mask, p_103 - 0.012, p_103)
    low_pocket_mask = regular & (n_p_raw < 15) & (p_rate_raw < 0.40)
    p_103 = np.where(low_pocket_mask, p_103 + 0.008, p_103)
    p_103 = np.clip(p_103, 1e-5, 1 - 1e-5)
    
    bss_103 = brier_skill_score(y_val, p_103)
    brier_103 = brier_score(y_val, p_103)
    print(f"\n[Baseline 103A Champion on 2024 Holdout] Brier: {brier_103:.8f} | BSS: {bss_103:.4f}")
    
    # ----------------------------------------------------
    # TEST MECHANISM A: 6번 레포 Sample-Size Reliability Shrinkage Modulated Deep Hierarchical
    # ----------------------------------------------------
    print("\n--- Testing Mechanism A: Sample-Size Reliability Shrinkage Modulation ---")
    best_gain = 0.0
    best_cfg = None
    for k_val in [5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 40.0, 50.0]:
        rel_mod = n_p_raw / (n_p_raw + k_val)
        deep_corr_shrunk = rel_mod * deep_corr
        
        for w_cand in [0.06, 0.08, 0.10, 0.12, 0.14]:
            p_cand = np.where(regular, p + w_cand * deep_corr_shrunk + 0.04 * call_corr, p)
            p_cand = np.where(high_pocket_mask, p_cand - 0.012, p_cand)
            p_cand = np.where(low_pocket_mask, p_cand + 0.008, p_cand)
            p_cand = np.clip(p_cand, 1e-5, 1 - 1e-5)
            
            b_c = brier_score(y_val, p_cand)
            bss_c = brier_skill_score(y_val, p_cand)
            delta_b = brier_103 - b_c
            delta_bss = bss_c - bss_103
            if delta_bss > best_gain:
                best_gain = delta_bss
                best_cfg = (k_val, w_cand, b_c, bss_c)
                print(f"  --> [NEW BEST] K={k_val:4.1f}, w={w_cand:4.2f} -> Brier: {b_c:.8f} (Δ {delta_b:+.8f}) | BSS: {bss_c:.4f} (Δ {delta_bss:+.4f} pt)")

    # ----------------------------------------------------
    # TEST MECHANISM B: 6번 레포 Hierarchical Reliability Sub-Correction
    # ----------------------------------------------------
    print("\n--- Testing Mechanism B: Sample-Size Hierarchical Reliability Sub-Correction ---")
    # delta_rel = (n_p / (n_p + K)) * (p_hb_hier - prior)
    for k_rel in [15.0, 25.0, 35.0, 50.0]:
        rel_term = (n_p_raw / (n_p_raw + k_rel)) * (p_hb_hier - meta["prior"])
        rel_offset = float(np.mean(rel_term[regular]))
        rel_clean = rel_term - rel_offset
        
        for w_rel in [0.01, 0.02, 0.03, 0.04, 0.05]:
            p_cand = p_103 + np.where(regular, w_rel * rel_clean, 0.0)
            p_cand = np.clip(p_cand, 1e-5, 1 - 1e-5)
            
            b_c = brier_score(y_val, p_cand)
            bss_c = brier_skill_score(y_val, p_cand)
            delta_b = brier_103 - b_c
            delta_bss = bss_c - bss_103
            if delta_bss > 0:
                print(f"  --> [PASS B] K={k_rel:4.1f}, w_rel={w_rel:4.2f} -> Brier: {b_c:.8f} (Δ {delta_b:+.8f}) | BSS: {bss_c:.4f} (Δ {delta_bss:+.4f} pt)")

if __name__ == "__main__":
    main()
