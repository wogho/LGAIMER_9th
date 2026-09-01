#!/usr/bin/env python3
"""Evaluate Production Packages 108C, 109A, 109B, 109C directly across Holdout Seasons 2022, 2023, 2024."""
import gc, hashlib, json, os, sys, time
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
from src.v54_per_season_asof_75_features import build_v54_per_season_asof_75_features
from src.preprocessing_v2 import build_v2_features, build_v3_features, CAT_V2
from src.adaptive_gate import build_gate_features

def brier_skill_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    r = np.mean(y_true)
    bs_ref = r * (1.0 - r)
    bs_model = np.mean((y_pred - y_true) ** 2)
    return float(max(0.0, 100000.0 * (1.0 - bs_model / bs_ref)))

def build_leverage_features(df: pd.DataFrame) -> pd.DataFrame:
    li = df["li"].fillna(0.98).to_numpy(float) if "li" in df.columns else np.full(len(df), 0.98)
    b = df["balls_before"].fillna(0).to_numpy(float) if "balls_before" in df.columns else np.zeros(len(df))
    s = df["strikes_before"].fillna(0).to_numpy(float) if "strikes_before" in df.columns else np.zeros(len(df))
    count_diff = b - s
    
    inn = df["inning"].fillna(1).to_numpy(float) if "inning" in df.columns else np.ones(len(df))
    score_diff = np.abs(df["score_diff"].fillna(0).to_numpy(float)) if "score_diff" in df.columns else np.zeros(len(df))
    late_close = ((inn >= 7) & (score_diff <= 2)).astype(float)
    
    is_high_li = (li >= 1.5).astype(float)
    li_count_diff = li * count_diff
    li_late_close = li * late_close
    
    return pd.DataFrame({
        "is_high_leverage": is_high_li,
        "li_count_diff": li_count_diff,
        "li_late_close": li_late_close
    }, index=df.index)

def extract_advanced_physics_108c(df: pd.DataFrame) -> pd.DataFrame:
    velo = df["release_speed"].fillna(142.0).to_numpy(float) if "release_speed" in df.columns else np.full(len(df), 142.0)
    spin = df["spin_rate"].fillna(2200.0).to_numpy(float) if "spin_rate" in df.columns else np.full(len(df), 2200.0)
    pfx_x = df["pfx_x"].fillna(0.0).to_numpy(float) if "pfx_x" in df.columns else np.zeros(len(df))
    pfx_z = df["pfx_z"].fillna(0.0).to_numpy(float) if "pfx_z" in df.columns else np.zeros(len(df))
    rel_x = df["release_pos_x"].fillna(0.0).to_numpy(float) if "release_pos_x" in df.columns else np.zeros(len(df))
    rel_z = df["release_pos_z"].fillna(1.8).to_numpy(float) if "release_pos_z" in df.columns else np.full(len(df), 1.8)
    
    movement_mag = np.sqrt(pfx_x**2 + pfx_z**2)
    spin_eff_ratio = np.clip(movement_mag / (velo + 1e-5), 0, 1.0)
    release_dist = np.sqrt(rel_x**2 + (rel_z - 1.8)**2)
    
    b = df["balls_before"].fillna(0).to_numpy(float) if "balls_before" in df.columns else np.zeros(len(df))
    s = df["strikes_before"].fillna(0).to_numpy(float) if "strikes_before" in df.columns else np.zeros(len(df))
    is_2s = (s == 2).astype(float)
    is_3b = (b == 3).astype(float)
    count_pressure = (b - s) * (b + s + 1.0) / 7.0
    
    return pd.DataFrame({
        "phys_velo": velo,
        "phys_spin": spin,
        "phys_movement_mag": movement_mag,
        "phys_spin_eff": spin_eff_ratio,
        "phys_release_dist": release_dist,
        "phys_is_2s": is_2s,
        "phys_is_3b": is_3b,
        "phys_count_pressure": count_pressure
    }, index=df.index)

def extract_advanced_physics_109a(df: pd.DataFrame) -> pd.DataFrame:
    df_base = extract_advanced_physics_108c(df)
    n_p = df["asof_pitcher_n"].fillna(0).to_numpy(float) if "asof_pitcher_n" in df.columns else np.zeros(len(df))
    n_b = df["asof_batter_n"].fillna(0).to_numpy(float) if "asof_batter_n" in df.columns else np.zeros(len(df))
    df_base["asof_pitcher_log_n"] = np.log1p(np.clip(n_p, 0, None))
    df_base["asof_batter_log_n"] = np.log1p(np.clip(n_b, 0, None))
    return df_base

def extract_advanced_physics_109b(df: pd.DataFrame) -> pd.DataFrame:
    df_base = extract_advanced_physics_108c(df)
    p_hand = df["pitcher_hand"].astype(str) if "pitcher_hand" in df.columns else pd.Series(["R"] * len(df), index=df.index)
    b_hand = df["batter_hand"].astype(str) if "batter_hand" in df.columns else pd.Series(["R"] * len(df), index=df.index)
    is_p_left = (p_hand == "L").astype(float)
    is_b_left = (b_hand == "L").astype(float)
    df_base["hand_platoon"] = (is_p_left != is_b_left).astype(float)
    df_base["hand_match_code"] = (is_p_left * 2.0 + is_b_left).astype(float)
    
    n_p = df["asof_pitcher_n"].fillna(0).to_numpy(float) if "asof_pitcher_n" in df.columns else np.zeros(len(df))
    p_rate = df["asof_pitcher_success_rate"].fillna(0.523766).to_numpy(float) if "asof_pitcher_success_rate" in df.columns else np.full(len(df), 0.523766)
    prev1_p = df["asof_pitcher_prev1_game_success_rate"].fillna(0.523766).to_numpy(float) if "asof_pitcher_prev1_game_success_rate" in df.columns else p_rate
    df_base["recent_drift"] = (prev1_p - p_rate) * (n_p / (n_p + 10.0))
    return df_base

def extract_hyper_regime_tensor_109c(df: pd.DataFrame) -> pd.DataFrame:
    df_base = extract_advanced_physics_108c(df)
    p_hand = df["pitcher_hand"].astype(str) if "pitcher_hand" in df.columns else pd.Series(["R"] * len(df), index=df.index)
    b_hand = df["batter_hand"].astype(str) if "batter_hand" in df.columns else pd.Series(["R"] * len(df), index=df.index)
    is_p_left = (p_hand == "L").astype(float)
    is_b_left = (b_hand == "L").astype(float)
    df_base["hand_platoon"] = (is_p_left != is_b_left).astype(float)
    df_base["hand_match_code"] = (is_p_left * 2.0 + is_b_left).astype(float)
    
    n_p = df["asof_pitcher_n"].fillna(0).to_numpy(float) if "asof_pitcher_n" in df.columns else np.zeros(len(df))
    n_b = df["asof_batter_n"].fillna(0).to_numpy(float) if "asof_batter_n" in df.columns else np.zeros(len(df))
    df_base["asof_pitcher_log_n"] = np.log1p(np.clip(n_p, 0, None))
    df_base["asof_batter_log_n"] = np.log1p(np.clip(n_b, 0, None))
    
    p_rate = df["asof_pitcher_success_rate"].fillna(0.523766).to_numpy(float) if "asof_pitcher_success_rate" in df.columns else np.full(len(df), 0.523766)
    prev1_p = df["asof_pitcher_prev1_game_success_rate"].fillna(0.523766).to_numpy(float) if "asof_pitcher_prev1_game_success_rate" in df.columns else p_rate
    prev5_p = df["asof_pitcher_prev5_game_success_rate"].fillna(0.523766).to_numpy(float) if "asof_pitcher_prev5_game_success_rate" in df.columns else p_rate
    df_base["recent_drift"] = (prev1_p - p_rate) * (n_p / (n_p + 10.0))
    df_base["momentum_trend"] = (prev1_p - prev5_p).astype(float)
    return df_base

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

def evaluate_production_models():
    print("=" * 80)
    print("  PRODUCTION MODELS EVALUATION: 108C vs 109A vs 109B vs 109C (2022·2023·2024)")
    print("=" * 80)
    raw = pd.read_csv(ROOT / "data/train.csv", low_memory=False)
    
    M107 = ROOT / "model/REF4-SUPER-ENSEMBLE-107A/production_package/model"
    M108C = ROOT / "model/REF4-SUPER-ENSEMBLE-108C/production_package/model"
    M109A = ROOT / "model/REF4-SUPER-ENSEMBLE-109A/production_package/model"
    M109B = ROOT / "model/REF4-SUPER-ENSEMBLE-109B/production_package/model"
    M109C = ROOT / "model/REF4-SUPER-ENSEMBLE-109C/production_package/model"
    
    meta_107 = json.loads((M107 / "manifest.json").read_text(encoding="utf-8"))
    priors = json.loads((M107 / "per_season_priors.json").read_text(encoding="utf-8"))
    ps = pd.read_pickle(M107 / "pitcher_snapshots.pkl")
    bs = pd.read_pickle(M107 / "batter_snapshots.pkl")
    ms = pd.read_pickle(M107 / "pitchmix_snapshots.pkl")
    tm = str(M107 / "trackman_prior_features.csv")
    seeds = [260802, 260803, 260804, 260805, 260806, 260807]
    regime = json.loads((M107 / "f_regime_meta.json").read_text())
    
    r_seeds = [42, 1, 2, 3, 4]
    f_seeds = [42, 1, 2, 3]
    
    # Load 108C Boosters
    print("Loading 108C Boosters (10 regular + 4 futures)...")
    cb_108c_reg = [CatBoostRegressor().load_model(str(M108C / f"super_resid_cb_seed{s}.cbm")) for s in r_seeds]
    lgb_108c_reg = [lgb.Booster(model_file=str(M108C / f"super_resid_lgb_seed{s}.txt")) for s in r_seeds]
    cb_108c_fut = [CatBoostRegressor().load_model(str(M108C / f"fut_resid_cb_seed{s}.cbm")) for s in f_seeds]
    
    # Load 109A Boosters (15 regular + 4 futures)
    print("Loading 109A Boosters (15 regular Tri-Family + 4 futures)...")
    cb_109a_reg = [CatBoostRegressor().load_model(str(M109A / f"super_resid_cb_seed{s}.cbm")) for s in r_seeds]
    lgb_109a_reg = [lgb.Booster(model_file=str(M109A / f"super_resid_lgb_seed{s}.txt")) for s in r_seeds]
    xgb_109a_reg = []
    for s in r_seeds:
        m = XGBRegressor()
        m.load_model(str(M109A / f"super_resid_xgb_seed{s}.json"))
        xgb_109a_reg.append(m)
    cb_109a_fut = [CatBoostRegressor().load_model(str(M109A / f"fut_resid_cb_seed{s}.cbm")) for s in f_seeds]
    
    # Load 109B Boosters (10 regular + 4 futures)
    print("Loading 109B Boosters (10 regular + 4 futures)...")
    cb_109b_reg = [CatBoostRegressor().load_model(str(M109B / f"super_resid_cb_seed{s}.cbm")) for s in r_seeds]
    lgb_109b_reg = [lgb.Booster(model_file=str(M109B / f"super_resid_lgb_seed{s}.txt")) for s in r_seeds]
    cb_109b_fut = [CatBoostRegressor().load_model(str(M109B / f"fut_resid_cb_seed{s}.cbm")) for s in f_seeds]
    
    # Load 109C Boosters (15 regular Tri-Family + 4 futures)
    print("Loading 109C Boosters (15 regular Tri-Family + 4 futures)...")
    cb_109c_reg = [CatBoostRegressor().load_model(str(M109C / f"super_resid_cb_seed{s}.cbm")) for s in r_seeds]
    lgb_109c_reg = [lgb.Booster(model_file=str(M109C / f"super_resid_lgb_seed{s}.txt")) for s in r_seeds]
    xgb_109c_reg = []
    for s in r_seeds:
        m = XGBRegressor()
        m.load_model(str(M109C / f"super_resid_xgb_seed{s}.json"))
        xgb_109c_reg.append(m)
    cb_109c_fut = [CatBoostRegressor().load_model(str(M109C / f"fut_resid_cb_seed{s}.cbm")) for s in f_seeds]
    
    print("\nEvaluating all candidates on 2022, 2023, and 2024 seasons (746,504 rows total)...")
    results = []
    
    for s in [2022, 2023, 2024]:
        t_s0 = time.time()
        val_df = raw[raw["season"] == s].copy().reset_index(drop=True)
        y_val = val_df["control_success"].to_numpy(float)
        
        p_103_val = predict_103a_full(val_df, M107, meta_107, priors, ps, bs, ms, tm, seeds, regime)
        
        reg = val_df["game_type"] != "F"
        fut = ~reg
        
        # Build 108C feature sets
        X_61_reg, _ = build_v5_deep_61_features(val_df[reg].reset_index(drop=True), profile_path=M107 / "team_asof_profile.json", prior=0.523766)
        prev1_p_val = val_df.loc[reg, "asof_pitcher_prev1_game_success_rate"].fillna(0.523766).to_numpy(float)
        b_rate_val = val_df.loc[reg, "asof_batter_success_rate"].fillna(0.523766).to_numpy(float)
        X_61_reg["form_gap"] = prev1_p_val - b_rate_val
        X_61_reg["anchor_p"] = p_103_val[reg]
        
        X_61_fut, _ = build_v5_deep_61_features(val_df[fut].reset_index(drop=True), profile_path=M107 / "team_asof_profile.json", prior=0.523766)
        X_61_fut["anchor_p"] = p_103_val[fut]
        
        # 108C Evaluation
        phys_108c_reg = extract_advanced_physics_108c(val_df[reg].reset_index(drop=True))
        X_108c_reg = pd.concat([X_61_reg, phys_108c_reg], axis=1)
        phys_108c_fut = extract_advanced_physics_108c(val_df[fut].reset_index(drop=True))
        X_108c_fut = pd.concat([X_61_fut, phys_108c_fut], axis=1)
        
        cb_p_108c = [m.predict(X_108c_reg) for m in cb_108c_reg]
        lgb_p_108c = [m.predict(X_108c_reg) for m in lgb_108c_reg]
        res_108c_reg = np.mean(cb_p_108c + lgb_p_108c, axis=0)
        res_108c_fut = np.mean([m.predict(X_108c_fut) for m in cb_108c_fut], axis=0) if np.sum(fut) > 0 else np.zeros(0)
        
        p_108c = p_103_val.copy()
        p_108c[reg] += 0.08 * res_108c_reg
        p_108c[fut] += 0.04 * res_108c_fut
        p_108c = np.clip(p_108c, 1e-5, 1 - 1e-5)
        
        # 109A Evaluation
        phys_109a_reg = extract_advanced_physics_109a(val_df[reg].reset_index(drop=True))
        X_109a_reg = pd.concat([X_61_reg, phys_109a_reg], axis=1)
        phys_109a_fut = extract_advanced_physics_109a(val_df[fut].reset_index(drop=True))
        X_109a_fut = pd.concat([X_61_fut, phys_109a_fut], axis=1)
        
        cb_p_109a = [m.predict(X_109a_reg) for m in cb_109a_reg]
        lgb_p_109a = [m.predict(X_109a_reg) for m in lgb_109a_reg]
        xgb_p_109a = [m.predict(X_109a_reg) for m in xgb_109a_reg]
        res_109a_reg = np.mean(cb_p_109a + lgb_p_109a + xgb_p_109a, axis=0)
        res_109a_fut = np.mean([m.predict(X_109a_fut) for m in cb_109a_fut], axis=0) if np.sum(fut) > 0 else np.zeros(0)
        
        p_109a = p_103_val.copy()
        p_109a[reg] += 0.08 * res_109a_reg
        p_109a[fut] += 0.035 * res_109a_fut
        p_109a = np.clip(p_109a, 1e-5, 1 - 1e-5)
        
        # 109B Evaluation
        phys_109b_reg = extract_advanced_physics_109b(val_df[reg].reset_index(drop=True))
        X_109b_reg = pd.concat([X_61_reg, phys_109b_reg], axis=1)
        phys_109b_fut = extract_advanced_physics_109b(val_df[fut].reset_index(drop=True))
        X_109b_fut = pd.concat([X_61_fut, phys_109b_fut], axis=1)
        
        cb_p_109b = [m.predict(X_109b_reg) for m in cb_109b_reg]
        lgb_p_109b = [m.predict(X_109b_reg) for m in lgb_109b_reg]
        res_109b_reg = np.mean(cb_p_109b + lgb_p_109b, axis=0)
        res_109b_fut = np.mean([m.predict(X_109b_fut) for m in cb_109b_fut], axis=0) if np.sum(fut) > 0 else np.zeros(0)
        
        p_109b = p_103_val.copy()
        p_109b[reg] += 0.08 * res_109b_reg
        p_109b[fut] += 0.04 * res_109b_fut
        p_109b = np.clip(p_109b, 1e-5, 1 - 1e-5)
        
        # 109C Evaluation
        tensor_109c_reg = extract_hyper_regime_tensor_109c(val_df[reg].reset_index(drop=True))
        X_109c_reg = pd.concat([X_61_reg, tensor_109c_reg], axis=1)
        tensor_109c_fut = extract_hyper_regime_tensor_109c(val_df[fut].reset_index(drop=True))
        X_109c_fut = pd.concat([X_61_fut, tensor_109c_fut], axis=1)
        
        cb_p_109c = [m.predict(X_109c_reg) for m in cb_109c_reg]
        lgb_p_109c = [m.predict(X_109c_reg) for m in lgb_109c_reg]
        xgb_p_109c = [m.predict(X_109c_reg) for m in xgb_109c_reg]
        res_109c_reg = np.mean(cb_p_109c + lgb_p_109c + xgb_p_109c, axis=0)
        res_109c_fut = np.mean([m.predict(X_109c_fut) for m in cb_109c_fut], axis=0) if np.sum(fut) > 0 else np.zeros(0)
        
        p_109c = p_103_val.copy()
        p_109c[reg] += 0.085 * res_109c_reg
        p_109c[fut] += 0.035 * res_109c_fut
        p_109c = np.clip(p_109c, 1e-5, 1 - 1e-5)
        
        bss_103 = brier_skill_score(y_val, p_103_val)
        bss_108c = brier_skill_score(y_val, p_108c)
        bss_109a = brier_skill_score(y_val, p_109a)
        bss_109b = brier_skill_score(y_val, p_109b)
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
        print(f"\nSeason {s} ({len(val_df):,} rows, evaluated in {time.time() - t_s0:.1f}s):")
        print(f"  • 103A Base         : {bss_103:9.4f} pt")
        print(f"  • 108C Champion     : {bss_108c:9.4f} pt (Δ {bss_108c - bss_103:+7.2f} pt)")
        print(f"  • 109A (Tri-Family) : {bss_109a:9.4f} pt (Δ {bss_109a - bss_108c:+7.2f} pt vs 108C)")
        print(f"  • 109B (Hand/Drift) : {bss_109b:9.4f} pt (Δ {bss_109b - bss_108c:+7.2f} pt vs 108C)")
        print(f"  • 109C (Hyper-Regime): {bss_109c:9.4f} pt (Δ {bss_109c - bss_108c:+7.2f} pt vs 108C)")
        
    out_file = ROOT / "output/comparison_109abc_production_results.json"
    out_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved benchmark results to {out_file}")

if __name__ == "__main__":
    evaluate_production_models()
