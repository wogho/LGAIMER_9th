#!/usr/bin/env python3
"""Multi-Season Full Validation Benchmark: 108C vs 109C vs 110A vs 110B vs 110C."""
import json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model/REF4-SUPER-ENSEMBLE-109C/production_package"))

DATA_PATH = ROOT / "data/train.csv"

def brier_score(y_true, y_prob):
    return np.mean((y_prob - y_true) ** 2)

def brier_skill_score(y_true, y_prob):
    bs = brier_score(y_true, y_prob)
    p_bar = np.mean(y_true)
    bs_ref = p_bar * (1.0 - p_bar)
    return (1.0 - (bs / bs_ref)) * 1000.0

def safe_cb_predict(model, df):
    if len(df) == 0: return np.zeros(0, dtype=float)
    if hasattr(model, "feature_names_") and model.feature_names_:
        cols = [c for c in model.feature_names_ if c in df.columns]
        if len(cols) == len(model.feature_names_):
            sub_df = df[model.feature_names_].copy()
            cat_indices = model.get_cat_feature_indices() if hasattr(model, "get_cat_feature_indices") else []
            for idx in cat_indices:
                col_name = model.feature_names_[idx]
                sub_df[col_name] = sub_df[col_name].fillna("missing").astype(str)
            return model.predict(sub_df)
    return model.predict(df)

def safe_cb_predict_proba(model, df):
    if len(df) == 0: return np.zeros((0, 2), dtype=float)
    if hasattr(model, "feature_names_") and model.feature_names_:
        cols = [c for c in model.feature_names_ if c in df.columns]
        if len(cols) == len(model.feature_names_):
            sub_df = df[model.feature_names_].copy()
            cat_indices = model.get_cat_feature_indices() if hasattr(model, "get_cat_feature_indices") else []
            for idx in cat_indices:
                col_name = model.feature_names_[idx]
                sub_df[col_name] = sub_df[col_name].fillna("missing").astype(str)
            return model.predict_proba(sub_df)
    return model.predict_proba(df)

def safe_lgb_predict(booster, df):
    if len(df) == 0: return np.zeros(0, dtype=float)
    if hasattr(booster, "feature_name"):
        fn = booster.feature_name()
        if fn and all(c in df.columns for c in fn):
            return booster.predict(df[fn])
    return booster.predict(df)

def safe_xgb_predict(model, df):
    if len(df) == 0: return np.zeros(0, dtype=float)
    if hasattr(model, "feature_names_in_"):
        fn = list(model.feature_names_in_)
        if fn and all(c in df.columns for c in fn):
            return model.predict(df[fn])
    return model.predict(df)

def extract_hyper_regime_tensor_109c(df: pd.DataFrame) -> pd.DataFrame:
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
    
    is_p_left = (df["pitcher_hand"].astype(str) == "L").to_numpy(float) if "pitcher_hand" in df.columns else np.zeros(len(df))
    is_b_left = (df["batter_hand"].astype(str) == "L").to_numpy(float) if "batter_hand" in df.columns else np.zeros(len(df))
    hand_platoon = (is_p_left != is_b_left).astype(float)
    hand_match_code = (is_p_left * 2.0 + is_b_left).astype(float)
    
    n_p = df["asof_pitcher_n"].fillna(0).to_numpy(float) if "asof_pitcher_n" in df.columns else np.zeros(len(df))
    n_b = df["asof_batter_n"].fillna(0).to_numpy(float) if "asof_batter_n" in df.columns else np.zeros(len(df))
    p_log_n = np.log1p(np.clip(n_p, 0, None))
    b_log_n = np.log1p(np.clip(n_b, 0, None))
    
    p_rate = df["asof_pitcher_success_rate"].fillna(0.523766).to_numpy(float) if "asof_pitcher_success_rate" in df.columns else np.full(len(df), 0.523766)
    prev1_p = df["asof_pitcher_prev1_game_success_rate"].fillna(0.523766).to_numpy(float) if "asof_pitcher_prev1_game_success_rate" in df.columns else p_rate
    prev5_p = df["asof_pitcher_prev5_game_success_rate"].fillna(0.523766).to_numpy(float) if "asof_pitcher_prev5_game_success_rate" in df.columns else p_rate
    recent_drift = (prev1_p - p_rate) * (n_p / (n_p + 10.0))
    momentum_trend = (prev1_p - prev5_p).astype(float)
    
    return pd.DataFrame({
        "phys_velo": velo,
        "phys_spin": spin,
        "phys_movement_mag": movement_mag,
        "phys_spin_eff": spin_eff_ratio,
        "phys_release_dist": release_dist,
        "phys_is_2s": is_2s,
        "phys_is_3b": is_3b,
        "phys_count_pressure": count_pressure,
        "hand_platoon": hand_platoon,
        "hand_match_code": hand_match_code,
        "asof_pitcher_log_n": p_log_n,
        "asof_batter_log_n": b_log_n,
        "recent_drift": recent_drift,
        "momentum_trend": momentum_trend
    }, index=df.index)

def main():
    print("=" * 80)
    print("  MULTI-SEASON VALIDATION BENCHMARK: 108C vs 109C vs 110A vs 110B vs 110C  ")
    print("=" * 80)
    
    t0 = time.time()
    df = pd.read_csv(DATA_PATH, low_memory=False).reset_index(drop=True)
    df["game_year"] = df["season"].astype(int)
    y_all = df["control_success"].to_numpy(float)
    
    # Validation set: Seasons 2022, 2023, 2024 (746,504 rows)
    val_mask = df["game_year"].isin([2022, 2023, 2024]).to_numpy()
    val_df = df[val_mask].reset_index(drop=True)
    y_val = y_all[val_mask]
    
    print(f"Loaded validation set: {len(val_df):,} rows (Seasons 2022-2024)")
    print(f"  • 2022: {np.sum(val_df.game_year == 2022):,} rows")
    print(f"  • 2023: {np.sum(val_df.game_year == 2023):,} rows")
    print(f"  • 2024: {np.sum(val_df.game_year == 2024):,} rows")
    print(f"  • Regular: {np.sum(val_df.game_type != 'F'):,} | Futures: {np.sum(val_df.game_type == 'F'):,}")
    
    from catboost import CatBoostRegressor, CatBoostClassifier
    import lightgbm as lgb
    from xgboost import XGBRegressor
    from src.preprocessing_v2 import build_v2_features, build_v3_features, CAT_V2
    from src.adaptive_gate import build_gate_features
    from src.v54_per_season_asof_75_features import build_v54_per_season_asof_75_features
    from src.v5_deep_61_features import build_v5_deep_61_features
    
    MODEL_109 = ROOT / "model/REF4-SUPER-ENSEMBLE-109C/production_package/model"
    meta_109 = json.loads((MODEL_109 / "manifest.json").read_text())
    priors_109 = json.loads((MODEL_109 / "per_season_priors.json").read_text())
    ps = pd.read_pickle(MODEL_109 / "pitcher_snapshots.pkl")
    bs = pd.read_pickle(MODEL_109 / "batter_snapshots.pkl")
    ms = pd.read_pickle(MODEL_109 / "pitchmix_snapshots.pkl")
    tm = str(MODEL_109 / "trackman_prior_features.csv")
    
    print("\n[1/5] Extracting base features for validation set...")
    x2, base2 = build_v2_features(val_df, meta_109["prior"], ps, tm)
    x3, base3 = build_v3_features(val_df, meta_109["prior"], ps, bs, ms, tm)
    
    seeds = meta_109.get("seeds", [260802, 260803, 260804, 260805, 260806, 260807])
    def load_reg(stem):
        return [CatBoostRegressor().load_model(str(MODEL_109 / f"{stem}_seed{s}.cbm")) for s in seeds]
        
    preds_base = []
    for stem, x, base in [
        ("v2_decay55", x2, base2),
        ("v3_decay55", x3, base3),
        ("v3_decay30", x3, base3),
    ]:
        m_preds = [np.clip(base + safe_cb_predict(m, x), 1e-6, 1 - 1e-6) for m in load_reg(stem)]
        preds_base.append(np.mean(m_preds, axis=0))
        
    regime = json.loads((MODEL_109 / "f_regime_meta.json").read_text())
    futures = val_df["game_type"].eq("F").to_numpy()
    regular = ~futures
    
    def f_reg_mean(stem, count, x, base):
        m_preds = []
        for j in range(count):
            m = CatBoostRegressor()
            m.load_model(str(MODEL_109 / f"{stem}_{j}.cbm"))
            m_preds.append(np.clip(base + safe_cb_predict(m, x), 1e-6, 1 - 1e-6))
        return np.mean(m_preds, axis=0)
        
    if futures.any():
        f2 = f_reg_mean("f_v2_all", 4, x2, base2)
        preds_base[0] = np.where(futures, preds_base[0] + regime["v2_scale"] * (f2 - preds_base[0]), preds_base[0])
        f55 = f_reg_mean("f_v355_recent", 6, x3, base3)
        preds_base[1] = np.where(futures, preds_base[1] + regime["v355_scale"] * (f55 - preds_base[1]), preds_base[1])
        f30a = f_reg_mean("f_v330_all", 4, x3, base3)
        f30r = f_reg_mean("f_v330_recent", 2, x3, base3)
        recent_inner = preds_base[2] + regime["v330_recent_inner_scale"] * (f30r - preds_base[2])
        f30 = regime["v330_all_weight"] * f30a + (1 - regime["v330_all_weight"]) * recent_inner
        preds_base[2] = np.where(futures, preds_base[2] + regime["v330_scale"] * (f30 - preds_base[2]), preds_base[2])
        
    risks = []
    for name in ("middle", "wild", "reverse"):
        stems = [f"subtype_{name}_seed{s}.cbm" for s in seeds]
        m_preds = []
        for filename in stems:
            m = CatBoostClassifier()
            m.load_model(str(MODEL_109 / filename))
            m_preds.append(safe_cb_predict_proba(m, x3)[:, 1])
        risk = np.mean(m_preds, axis=0)
        if futures.any():
            fm = CatBoostClassifier()
            fm.load_model(str(MODEL_109 / f"f_subtype_{name}.cbm"))
            fr = safe_cb_predict_proba(fm, x3)[:, 1]
            risk = np.where(futures, risk + regime["subtype_scale"] * (fr - risk), risk)
        risks.append(risk)
        
    main_p = np.average(np.vstack(preds_base), axis=0, weights=meta_109["main_weights"])
    z = np.column_stack([main_p] + risks)
    p_103 = meta_109["stack_intercept"] + z @ np.asarray(meta_109["stack_coefficients"])
    
    if meta_109.get("adaptive_gate", False):
        gate_x = build_gate_features(val_df, preds_base, risks, np.clip(p_103, 1e-6, 1 - 1e-6))
        gate = CatBoostRegressor()
        gate.load_model(str(MODEL_109 / "adaptive_gate.cbm"))
        gate_pred = safe_cb_predict(gate, gate_x)
        p_103 = p_103 + float(meta_109.get("gate_scale", 0.08)) * (gate_pred - float(meta_109.get("gate_bias_offset", 0.0)))
        
    p_103 = p_103 + float(meta_109.get("global_shift", 0.0052))
    
    # 107A base pipeline
    p_107 = np.copy(p_103)
    if futures.any():
        from src.psych_latent import build_production_features, apply_linear_residual
        residual_x = build_production_features(val_df, MODEL_109 / "psych_profile.pkl", MODEL_109 / "latent_pitch_context.csv")
        correction = apply_linear_residual(residual_x, MODEL_109 / "psych_latent_meta.npz")
        p_107 = np.where(futures, p_107 + correction, p_107)
        
    if regular.any():
        from src.entity_context_split import apply_split_profile, apply_linear_split
        split_profile = pd.read_csv(MODEL_109 / "split_profile.csv", dtype={"entity_value": str, "context_value": str})
        split_x = apply_split_profile(val_df, split_profile)
        split_correction = apply_linear_split(split_x, MODEL_109 / "split_residual_meta.npz")
        p_split = p_107 + split_correction
        
        lgbm_model = lgb.Booster(model_file=str(MODEL_109 / 'r_expert_lgbm.txt'))
        x3_lgb = x3.copy()
        for col in CAT_V2:
            if col in x3_lgb.columns: x3_lgb[col] = x3_lgb[col].astype('category')
        res_lgbm = safe_lgb_predict(lgbm_model, x3_lgb)
        p_lgbm = np.clip(base3 + res_lgbm + 0.0052, 1e-6, 1.0 - 1e-6)
        p_107 = np.where(regular, 0.95 * p_split + 0.05 * p_lgbm, p_107)
        
    # 108C pipeline
    print("[2/5] Running 108C pipeline inference...")
    v54_feat, _ = build_v54_per_season_asof_75_features(val_df, profile_path=MODEL_109 / "team_asof_profile.json", priors=priors_109, prior=0.523766)
    
    li_val = val_df["li"].fillna(0.98).to_numpy(float) if "li" in val_df.columns else np.full(len(val_df), 0.98)
    b_val = val_df["balls_before"].fillna(0).to_numpy(float) if "balls_before" in val_df.columns else np.zeros(len(val_df))
    s_val = val_df["strikes_before"].fillna(0).to_numpy(float) if "strikes_before" in val_df.columns else np.zeros(len(val_df))
    lev_df = pd.DataFrame({
        "is_high_leverage": (li_val >= 1.5).astype(float),
        "li_count_diff": li_val * (b_val - s_val),
        "li_late_close": np.zeros(len(val_df))
    }, index=val_df.index)
    v54_feat = pd.concat([v54_feat, lev_df], axis=1)
    
    n_p_raw = val_df["asof_pitcher_n"].fillna(0).to_numpy(float)
    p_rate_raw = val_df["asof_pitcher_success_rate"].fillna(0.523766).to_numpy(float)
    prev1_raw = val_df["asof_pitcher_prev1_game_success_rate"].to_numpy(float) if "asof_pitcher_prev1_game_success_rate" in val_df.columns else p_rate_raw
    prev1_raw = np.where(np.isnan(prev1_raw), p_rate_raw, prev1_raw)
    
    rel_weight = n_p_raw / (n_p_raw + 25.0)
    p_shrunk = 0.523766 + rel_weight * (p_rate_raw - 0.523766)
    p_hb_hier = np.clip(0.70 * p_shrunk + 0.30 * prev1_raw, 0.05, 0.95)
    
    v54_seeds = [42, 1, 2, 3, 4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
    res_preds = []
    for s in v54_seeds:
        cb_m = CatBoostRegressor().load_model(str(MODEL_109 / f"deep_cb_l2_seed{s}.cbm"))
        res_preds.append(safe_cb_predict(cb_m, v54_feat))
        lgb_m = lgb.Booster(model_file=str(MODEL_109 / f"deep_lgb_l2_seed{s}.txt"))
        res_preds.append(safe_lgb_predict(lgb_m, v54_feat))
        
    p_deep_resid = np.mean(res_preds, axis=0)
    p_deep_reconstructed = np.clip(p_hb_hier + p_deep_resid, 1e-5, 1 - 1e-5)
    deep_corr = p_deep_reconstructed - 0.514025
    
    p_108 = np.where(regular, p_107 + 0.08 * deep_corr, p_107)
    
    cb_call = CatBoostClassifier().load_model(str(MODEL_109 / "refined_call_expert.cbm"))
    call_probs = safe_cb_predict_proba(cb_call, x3)[:, 0]
    p_108 = np.where(regular, p_108 + 0.04 * (call_probs - 0.523766), p_108)
    
    p_108 = np.where(regular & (n_p_raw < 15) & (p_rate_raw > 0.65), p_108 - 0.012, p_108)
    p_108 = np.where(regular & (n_p_raw < 15) & (p_rate_raw < 0.40), p_108 + 0.008, p_108)
    p_108 = np.clip(p_108, 1e-5, 1 - 1e-5)
    
    # 109C predictions
    print("[3/5] Running 109C pipeline inference...")
    X_61_reg, _ = build_v5_deep_61_features(val_df[regular].reset_index(drop=True), profile_path=MODEL_109 / "team_asof_profile.json", prior=0.523766)
    b_rate_val = val_df.loc[regular, "asof_batter_success_rate"].fillna(0.523766).to_numpy(float)
    X_61_reg["form_gap"] = prev1_raw[regular] - b_rate_val
    X_61_reg["anchor_p"] = p_108[regular]
    tensor_reg = extract_hyper_regime_tensor_109c(val_df[regular].reset_index(drop=True))
    X_reg_full_109 = pd.concat([X_61_reg, tensor_reg], axis=1)
    
    r_seeds = [42, 1, 2, 3, 4]
    cb_p = [safe_cb_predict(CatBoostRegressor().load_model(str(MODEL_109 / f"super_resid_cb_seed{s}.cbm")), X_reg_full_109) for s in r_seeds]
    lgb_p = [safe_lgb_predict(lgb.Booster(model_file=str(MODEL_109 / f"super_resid_lgb_seed{s}.txt")), X_reg_full_109) for s in r_seeds]
    xgb_p = []
    for s in r_seeds:
        m = XGBRegressor()
        m.load_model(str(MODEL_109 / f"super_resid_xgb_seed{s}.json"))
        xgb_p.append(safe_xgb_predict(m, X_reg_full_109))
        
    p_res_reg_109 = np.mean(cb_p + lgb_p + xgb_p, axis=0)
    p_109 = np.copy(p_108)
    p_109[regular] = p_109[regular] + 0.085 * p_res_reg_109
    
    if futures.any():
        X_61_fut, _ = build_v5_deep_61_features(val_df[futures].reset_index(drop=True), profile_path=MODEL_109 / "team_asof_profile.json", prior=0.523766)
        X_61_fut["anchor_p"] = p_108[futures]
        tensor_fut = extract_hyper_regime_tensor_109c(val_df[futures].reset_index(drop=True))
        X_fut_full_109 = pd.concat([X_61_fut, tensor_fut], axis=1)
        cb_f = [safe_cb_predict(CatBoostRegressor().load_model(str(MODEL_109 / f"fut_resid_cb_seed{s}.cbm")), X_fut_full_109) for s in [42, 1, 2, 3]]
        p_109[futures] = p_109[futures] + 0.035 * np.mean(cb_f, axis=0)
    p_109 = np.clip(p_109, 1e-5, 1 - 1e-5)
    
    # 110A predictions (MoE Router)
    print("[4/5] Running 110A pipeline inference...")
    MODEL_110A = ROOT / "model/REF4-SUPER-ENSEMBLE-110A/production_package/model"
    is_p_left = (val_df["pitcher_hand"].astype(str) == "L").to_numpy(float) if "pitcher_hand" in val_df.columns else np.zeros(len(val_df))
    is_b_left = (val_df["batter_hand"].astype(str) == "L").to_numpy(float) if "batter_hand" in val_df.columns else np.zeros(len(val_df))
    hand_matchup = (is_p_left * 2.0 + is_b_left).astype(float)
    count_pressure = (b_val - s_val) * (b_val + s_val + 1.0) / 7.0
    log_pitcher_n = np.log1p(np.clip(n_p_raw, 0, None))
    log_batter_n = np.log1p(np.clip(val_df["asof_batter_n"].fillna(0).to_numpy(float), 0, None)) if "asof_batter_n" in val_df.columns else np.zeros(len(val_df))
    
    diff_108_107 = p_108 - p_107
    diff_109_108 = p_109 - p_108
    stack_p = np.column_stack([p_107, p_108, p_109])
    expert_std = np.std(stack_p, axis=1)
    
    router_df = pd.DataFrame({
        "p107": p_107,
        "p108": p_108,
        "p109": p_109,
        "diff_108_107": diff_108_107,
        "diff_109_108": diff_109_108,
        "expert_std": expert_std,
        "is_futures": futures.astype(float),
        "count_pressure": count_pressure,
        "log_p_n": log_pitcher_n,
        "log_b_n": log_batter_n,
        "hand_matchup": hand_matchup,
        "li": li_val
    }, index=val_df.index)
    
    router_seeds = [42, 1, 2]
    r_preds = []
    for s in router_seeds:
        rm = CatBoostRegressor().load_model(str(MODEL_110A / f"moe_router_cb_seed{s}.cbm"))
        r_preds.append(safe_cb_predict(rm, router_df))
    p_110a = np.clip(p_109 + np.mean(r_preds, axis=0), 1e-5, 1 - 1e-5)
    
    # 110B predictions (Cross-Fitted Residual Boosters)
    print("[5/5] Running 110B & 110C pipeline inference...")
    MODEL_110B = ROOT / "model/REF4-CROSSFIT-SUPER-110B/production_package/model"
    cb_p_b = [safe_cb_predict(CatBoostRegressor().load_model(str(MODEL_110B / f"super_resid_cb_seed{s}.cbm")), X_reg_full_109) for s in r_seeds]
    lgb_p_b = [safe_lgb_predict(lgb.Booster(model_file=str(MODEL_110B / f"super_resid_lgb_seed{s}.txt")), X_reg_full_109) for s in r_seeds]
    xgb_p_b = []
    for s in r_seeds:
        m = XGBRegressor()
        m.load_model(str(MODEL_110B / f"super_resid_xgb_seed{s}.json"))
        xgb_p_b.append(safe_xgb_predict(m, X_reg_full_109))
        
    p_res_reg_b = np.mean(cb_p_b + lgb_p_b + xgb_p_b, axis=0)
    p_110b = np.copy(p_108)
    p_110b[regular] = p_110b[regular] + 0.085 * p_res_reg_b
    if futures.any():
        cb_f_b = [safe_cb_predict(CatBoostRegressor().load_model(str(MODEL_110B / f"fut_resid_cb_seed{s}.cbm")), X_fut_full_109) for s in [42, 1, 2, 3]]
        p_110b[futures] = p_110b[futures] + 0.035 * np.mean(cb_f_b, axis=0)
    p_110b = np.clip(p_110b, 1e-5, 1 - 1e-5)
    
    # 110C predictions (Grand Super Ensemble with Empirical Bayes Hand Pooling)
    platoon_adj = np.where(hand_matchup % 2 != (hand_matchup // 2), -0.0035, +0.0035) * rel_weight
    p_hb_hier_c = np.clip(p_hb_hier + platoon_adj, 0.05, 0.95)
    p_deep_reconstructed_c = np.clip(p_hb_hier_c + p_deep_resid, 1e-5, 1 - 1e-5)
    deep_corr_c = p_deep_reconstructed_c - 0.514025
    
    p_108_c = np.where(regular, p_107 + 0.08 * deep_corr_c, p_107)
    p_108_c = np.where(regular, p_108_c + 0.04 * (call_probs - 0.523766), p_108_c)
    p_108_c = np.where(regular & (n_p_raw < 15) & (p_rate_raw > 0.65), p_108_c - 0.012, p_108_c)
    p_108_c = np.where(regular & (n_p_raw < 15) & (p_rate_raw < 0.40), p_108_c + 0.008, p_108_c)
    p_108_c = np.clip(p_108_c, 1e-5, 1 - 1e-5)
    
    p_110c = np.copy(p_108_c)
    p_110c[regular] = p_110c[regular] + 0.085 * p_res_reg_b
    if futures.any():
        p_110c[futures] = p_110c[futures] + 0.035 * np.mean(cb_f_b, axis=0)
    p_110c = np.clip(p_110c, 1e-5, 1 - 1e-5)
    
    # Calculate exact BSS across all models
    models = {
        "108C (LB: 1120.5651)": p_108,
        "109C (LB: 1120.8914)": p_109,
        "110A (Temporal MoE Router)": p_110a,
        "110B (Leakage-Corrected CrossFit)": p_110b,
        "110C (Grand Super Ensemble)": p_110c,
    }
    
    print("\n" + "=" * 80)
    print("                    FINAL MULTI-SEASON BSS RESULTS                      ")
    print("=" * 80)
    print(f"{'Model Candidate':<35} | {'Overall BSS':<11} | {'2022 BSS':<9} | {'2023 BSS':<9} | {'2024 BSS':<9} | {'Brier Score':<11}")
    print("-" * 95)
    
    results = {}
    for name, p_val in models.items():
        bss_all = brier_skill_score(y_val, p_val)
        bss_22 = brier_skill_score(y_val[val_df.game_year == 2022], p_val[val_df.game_year == 2022])
        bss_23 = brier_skill_score(y_val[val_df.game_year == 2023], p_val[val_df.game_year == 2023])
        bss_24 = brier_skill_score(y_val[val_df.game_year == 2024], p_val[val_df.game_year == 2024])
        bs_all = brier_score(y_val, p_val)
        results[name] = {
            "overall": bss_all, "2022": bss_22, "2023": bss_23, "2024": bss_24, "brier": bs_all
        }
        print(f"{name:<35} | {bss_all:11.4f} | {bss_22:9.4f} | {bss_23:9.4f} | {bss_24:9.4f} | {bs_all:11.6f}")
        
    print("-" * 95)
    
    # Regular vs Futures BSS Breakdown
    print("\n[Segment Breakdown]")
    for name, p_val in models.items():
        bss_reg = brier_skill_score(y_val[regular], p_val[regular])
        bss_fut = brier_skill_score(y_val[futures], p_val[futures])
        print(f"  • {name:<35}: Regular BSS = {bss_reg:.4f} | Futures BSS = {bss_fut:.4f}")
        
    print(f"\nMulti-Season benchmark completed in {time.time() - t0:.1f}s")
    
    # Save results to json
    (ROOT / "output/multiseason_benchmark_110.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

if __name__ == "__main__":
    main()
