#!/usr/bin/env python3
"""
REF4-LATENT-MIXTURE-122 Production Packager & Builder.
Integrates 113A Champion (1121.9040 LB) + 3-Way Latent Pitch-Type Mixture Marginalization (Convex Blend).
Strictly NO zombie components, strictly Convex Blend (w=0.055).
"""
import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
MODEL_113A = ROOT / "model/REF4-DISJOINT-EB-113A/production_package/model"
SRC_113A = ROOT / "model/REF4-DISJOINT-EB-113A/production_package/src"
LATENT_122_MODEL = ROOT / "model/REF4-LATENT-MIXTURE-122/production_package/model"
OUT_DIR = ROOT / "model/REF4-LATENT-MIXTURE-122/production_package"
ZIP_OUT = ROOT / "output/submit_ref4_super_ensemble_122.zip"

def main():
    print("=" * 80)
    print("  PACKAGING CANDIDATE 122: REF4-LATENT-MIXTURE-122  ")
    print("=" * 80)
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_model = OUT_DIR / "model"
    out_src = OUT_DIR / "src"
    out_model.mkdir(exist_ok=True)
    out_src.mkdir(exist_ok=True)
    
    # 1. Copy src modules
    for py_file in (ROOT / "src").glob("*.py"):
        shutil.copy2(py_file, out_src / py_file.name)
    if SRC_113A.exists():
        for py_file in SRC_113A.glob("*.py"):
            shutil.copy2(py_file, out_src / py_file.name)
            
    # 2. Copy all 113A models if not already in out_model
    for asset in MODEL_113A.glob("*"):
        if asset.is_file() and not (out_model / asset.name).exists():
            shutil.copy2(asset, out_model / asset.name)
            
    # 3. Copy latent mixture models
    for m_name in ["latent_fb_specialist.cbm", "latent_br_specialist.cbm", "latent_os_specialist.cbm", "latent_manifest.json", "per_season_priors.json"]:
        if (LATENT_122_MODEL / m_name).exists() and not (out_model / m_name).exists():
            shutil.copy2(LATENT_122_MODEL / m_name, out_model / m_name)
        
    # 4. Write manifest.json
    manifest = json.loads((MODEL_113A / "manifest.json").read_text(encoding="utf-8"))
    manifest["pipeline"] = "REF4-LATENT-MIXTURE-122"
    manifest["description"] = "113A Champion (1121.9040 LB) + Latent Pitch-Type Mixture Marginalization Convex Blend"
    manifest["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest["w_latent_mixture_convex"] = 0.055
    
    with open(out_model / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    # 5. Write requirements.txt
    reqs = (
        "numpy>=1.24.0\n"
        "pandas>=2.0.0\n"
        "scipy>=1.10.0\n"
        "scikit-learn>=1.2.0\n"
        "catboost>=1.2.0\n"
        "lightgbm>=4.0.0\n"
        "xgboost>=1.7.0\n"
    )
    (OUT_DIR / "requirements.txt").write_text(reqs, encoding="utf-8")
    
    # 6. Write production script.py
    script_code = '''# Production inference entry point for REF4-LATENT-MIXTURE-122
import json
import os
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
import lightgbm as lgb
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "model"
SRC = ROOT / "src"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from src.preprocessing_v2 import CAT_V2, build_v2_features, build_v3_features
from src.v5_deep_61_features import build_v5_deep_61_features
from src.v54_per_season_asof_75_features import build_v54_per_season_asof_75_features
from src.adaptive_gate import build_gate_features

def safe_cb_predict(model, df):
    if len(df) == 0:
        return np.zeros(0, dtype=float)
    cols = model.feature_names_
    df_aligned = df[cols].copy()
    try:
        cat_indices = model.get_cat_feature_indices()
        for idx in cat_indices:
            col_name = cols[idx]
            df_aligned[col_name] = df_aligned[col_name].astype(str).replace({"nan": "missing", "None": "missing"})
    except Exception:
        for c in df_aligned.select_dtypes(include=['object', 'category']).columns:
            df_aligned[c] = df_aligned[c].astype(str).replace({"nan": "missing", "None": "missing"})
    return model.predict(df_aligned)

def safe_cb_predict_proba(model, df):
    if len(df) == 0:
        return np.zeros((0, 2), dtype=float)
    cols = model.feature_names_
    df_aligned = df[cols].copy()
    try:
        cat_indices = model.get_cat_feature_indices()
        for idx in cat_indices:
            col_name = cols[idx]
            df_aligned[col_name] = df_aligned[col_name].astype(str).replace({"nan": "missing", "None": "missing"})
    except Exception:
        for c in df_aligned.select_dtypes(include=['object', 'category']).columns:
            df_aligned[c] = df_aligned[c].astype(str).replace({"nan": "missing", "None": "missing"})
    return model.predict_proba(df_aligned)

def safe_lgb_predict(booster, df):
    if len(df) == 0:
        return np.zeros(0, dtype=float)
    cols = booster.feature_name()
    df_aligned = df[cols].copy()
    for col in CAT_V2:
        if col in df_aligned.columns:
            df_aligned[col] = df_aligned[col].astype('category')
    for col in df_aligned.select_dtypes(include=['object']).columns:
        df_aligned[col] = df_aligned[col].astype('category')
    return booster.predict(df_aligned)

def safe_xgb_predict(model, df):
    if len(df) == 0:
        return np.zeros(0, dtype=float)
    cols = list(model.feature_names_in_)
    return model.predict(df[cols])

def build_leverage_features(df: pd.DataFrame) -> pd.DataFrame:
    li = df["li"].fillna(0.98).to_numpy(float) if "li" in df.columns else np.full(len(df), 0.98)
    b = df["balls_before"].fillna(0).to_numpy(float) if "balls_before" in df.columns else np.zeros(len(df))
    s = df["strikes_before"].fillna(0).to_numpy(float) if "strikes_before" in df.columns else np.zeros(len(df))
    inn = df["inning"].fillna(1).to_numpy(float) if "inning" in df.columns else np.ones(len(df))
    score_diff = np.abs(df["score_diff"].fillna(0).to_numpy(float)) if "score_diff" in df.columns else (
        np.abs(df["score_diff_pitcher_team"].fillna(0).to_numpy(float)) if "score_diff_pitcher_team" in df.columns else np.zeros(len(df))
    )
    late_close = ((inn >= 7) & (score_diff <= 2)).astype(float)
    return pd.DataFrame({
        "is_high_leverage": (li >= 1.5).astype(float),
        "li_count_diff": li * (b - s),
        "li_late_close": li * late_close
    }, index=df.index)

def extract_hyper_regime_tensor_109c(df: pd.DataFrame) -> pd.DataFrame:
    velo = df["release_speed"].fillna(140.0).to_numpy(float) if "release_speed" in df.columns else np.full(len(df), 140.0)
    spin = df["release_spin_rate"].fillna(2200.0).to_numpy(float) if "release_spin_rate" in df.columns else np.full(len(df), 2200.0)
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

def predict_latent_mixture_122(df: pd.DataFrame, model_dir: Path, priors: dict) -> np.ndarray:
    feat_base, _ = build_v54_per_season_asof_75_features(df, priors=priors, prior=0.523766)
    
    b = df["balls_before"].fillna(0).to_numpy(float) if "balls_before" in df.columns else np.zeros(len(df))
    s = df["strikes_before"].fillna(0).to_numpy(float) if "strikes_before" in df.columns else np.zeros(len(df))
    li = df["li"].fillna(0.98).to_numpy(float) if "li" in df.columns else np.full(len(df), 0.98)
    inn = df["inning"].fillna(1).to_numpy(float) if "inning" in df.columns else np.ones(len(df))
    score_diff = np.abs(df["score_diff_pitcher_team"].fillna(0).to_numpy(float)) if "score_diff_pitcher_team" in df.columns else np.zeros(len(df))
    
    zone_inevitability = np.clip((b - s + 1.0) / 4.0, 0.0, 1.0) * (b >= 2).astype(float)
    chase_incentive = (s == 2).astype(float) * np.clip((3.0 - b) / 3.0, 0.0, 1.0)
    count_pressure = (b - s) * (b + s + 1.0) / 7.0
    
    is_p_left = (df["pitcher_hand"].astype(str) == "L").to_numpy(float) if "pitcher_hand" in df.columns else np.zeros(len(df))
    is_b_left = (df["batter_hand"].astype(str) == "L").to_numpy(float) if "batter_hand" in df.columns else np.zeros(len(df))
    platoon = (is_p_left != is_b_left).astype(float)
    
    fb = df["asof_pitcher_fastball_rate"].fillna(0.50).to_numpy(float) if "asof_pitcher_fastball_rate" in df.columns else np.full(len(df), 0.50)
    br = df["asof_pitcher_breaking_rate"].fillna(0.30).to_numpy(float) if "asof_pitcher_breaking_rate" in df.columns else np.full(len(df), 0.30)
    os_ = df["asof_pitcher_offspeed_rate"].fillna(0.20).to_numpy(float) if "asof_pitcher_offspeed_rate" in df.columns else np.full(len(df), 0.20)
    
    pi_fb_est = np.clip(fb * (1.0 - 0.25 * (s == 2).astype(float) + 0.15 * zone_inevitability), 0.05, 0.90)
    pi_br_est = np.clip(br * (1.0 + 0.35 * (s == 2).astype(float) - 0.20 * zone_inevitability), 0.05, 0.90)
    pi_os_est = np.clip(os_ * (1.0 + 0.20 * platoon + 0.10 * (s == 2).astype(float)), 0.05, 0.90)
    tot_pi = pi_fb_est + pi_br_est + pi_os_est
    pi_fb_est /= tot_pi
    pi_br_est /= tot_pi
    pi_os_est /= tot_pi
    
    p_rate = df["asof_pitcher_success_rate"].fillna(0.523766).to_numpy(float) if "asof_pitcher_success_rate" in df.columns else np.full(len(df), 0.523766)
    prev1_p = df["asof_pitcher_prev1_game_success_rate"].fillna(0.523766).to_numpy(float) if "asof_pitcher_prev1_game_success_rate" in df.columns else p_rate
    b_rate = df["asof_batter_success_rate"].fillna(0.523766).to_numpy(float) if "asof_batter_success_rate" in df.columns else np.full(len(df), 0.523766)
    b_mid = df["asof_batter_middle_rate"].fillna(0.523766).to_numpy(float) if "asof_batter_middle_rate" in df.columns else np.full(len(df), 0.523766)
    
    n_p = df["asof_pitcher_n"].fillna(0).to_numpy(float) if "asof_pitcher_n" in df.columns else np.zeros(len(df))
    gamma_p = n_p / (n_p + 20.0)
    p_shrunk = gamma_p * p_rate + (1.0 - gamma_p) * 0.523766
    
    feat_gt = pd.DataFrame({
        "zone_inevitability": zone_inevitability,
        "chase_incentive": chase_incentive,
        "count_pressure": count_pressure,
        "is_2s": (s == 2).astype(float),
        "is_3b": (b == 3).astype(float),
        "platoon": platoon,
        "pi_fb_est": pi_fb_est,
        "pi_br_est": pi_br_est,
        "pi_os_est": pi_os_est,
        "fb_rate": fb,
        "br_rate": br,
        "os_rate": os_,
        "p_shrunk": p_shrunk,
        "form_gap": prev1_p - (1.0 - b_rate),
        "b_mid": b_mid,
        "li": li,
        "is_high_li": (li >= 1.5).astype(float),
        "late_close": ((inn >= 7) & (score_diff <= 2)).astype(float),
    }, index=df.index)
    
    X_full = pd.concat([feat_base, feat_gt], axis=1)
    
    cb_fb = CatBoostRegressor()
    cb_fb.load_model(str(model_dir / "latent_fb_specialist.cbm"))
    p_fb = safe_cb_predict(cb_fb, X_full)
    
    cb_br = CatBoostRegressor()
    cb_br.load_model(str(model_dir / "latent_br_specialist.cbm"))
    p_br = safe_cb_predict(cb_br, X_full)
    
    cb_os = CatBoostRegressor()
    cb_os.load_model(str(model_dir / "latent_os_specialist.cbm"))
    p_os = safe_cb_predict(cb_os, X_full)
    
    p_marginalized = pi_fb_est * p_fb + pi_br_est * p_br + pi_os_est * p_os
    return p_marginalized

def main():
    started = time.time()
    meta = json.loads((MODEL / "manifest.json").read_text(encoding="utf-8"))
    priors = json.loads((MODEL / "per_season_priors.json").read_text(encoding="utf-8")) if (MODEL / "per_season_priors.json").exists() else {}
    regime = json.loads((MODEL / "f_regime_meta.json").read_text(encoding="utf-8"))
    
    test = pd.read_csv(ROOT / "data/test.csv", low_memory=False)
    row_id = test["row_id"].copy()
    
    n_p_raw = test["asof_pitcher_n"].fillna(0).to_numpy(float) if "asof_pitcher_n" in test.columns else np.zeros(len(test))
    p_rate_raw = test["asof_pitcher_success_rate"].fillna(0.523766).to_numpy(float) if "asof_pitcher_success_rate" in test.columns else np.full(len(test), 0.523766)
    
    ps = pd.read_pickle(MODEL / "pitcher_snapshots.pkl")
    bs = pd.read_pickle(MODEL / "batter_snapshots.pkl")
    ms = pd.read_pickle(MODEL / "pitchmix_snapshots.pkl")
    tm = str(MODEL / "trackman_prior_features.csv")
    
    x2, base2 = build_v2_features(test, meta["prior"], ps, tm)
    x3, base3 = build_v3_features(test, meta["prior"], ps, bs, ms, tm)
    
    seeds = meta.get("seeds", [260802, 260803, 260804, 260805, 260806, 260807])
    def load_reg(stem):
        return [CatBoostRegressor().load_model(str(MODEL / f"{stem}_seed{s}.cbm")) for s in seeds]
        
    predictions = []
    for stem, x, base in [
        ("v2_decay55", x2, base2),
        ("v3_decay55", x3, base3),
        ("v3_decay30", x3, base3),
    ]:
        member = [np.clip(base + safe_cb_predict(m, x), 1e-6, 1 - 1e-6) for m in load_reg(stem)]
        predictions.append(np.mean(member, axis=0))
        
    futures = test["game_type"].eq("F").to_numpy() if "game_type" in test.columns else np.zeros(len(test), dtype=bool)
    regular = ~futures
    
    def f_reg_mean(stem, count, x, base):
        member = []
        for j in range(count):
            m = CatBoostRegressor()
            m.load_model(str(MODEL / f"{stem}_{j}.cbm"))
            member.append(np.clip(base + safe_cb_predict(m, x), 1e-6, 1 - 1e-6))
        return np.mean(member, axis=0)

    if futures.any():
        f2 = f_reg_mean("f_v2_all", 4, x2, base2)
        predictions[0] = np.where(futures, predictions[0] + regime["v2_scale"] * (f2 - predictions[0]), predictions[0])
        f55 = f_reg_mean("f_v355_recent", 6, x3, base3)
        predictions[1] = np.where(futures, predictions[1] + regime["v355_scale"] * (f55 - predictions[1]), predictions[1])
        f30a = f_reg_mean("f_v330_all", 4, x3, base3)
        f30r = f_reg_mean("f_v330_recent", 2, x3, base3)
        recent_inner = predictions[2] + regime["v330_recent_inner_scale"] * (f30r - predictions[2])
        f30 = regime["v330_all_weight"] * f30a + (1.0 - regime["v330_all_weight"]) * recent_inner
        predictions[2] = np.where(futures, predictions[2] + regime["v330_scale"] * (f30 - predictions[2]), predictions[2])

    risks = []
    for name in ("middle", "wild", "reverse"):
        stems = [f"subtype_{name}_seed{s}.cbm" for s in seeds]
        member = []
        for filename in stems:
            m = CatBoostClassifier()
            m.load_model(str(MODEL / filename))
            member.append(safe_cb_predict_proba(m, x3)[:, 1])
        risk = np.mean(member, axis=0)
        if futures.any():
            fm = CatBoostClassifier()
            fm.load_model(str(MODEL / f"f_subtype_{name}.cbm"))
            fr = safe_cb_predict_proba(fm, x3)[:, 1]
            risk = np.where(futures, risk + regime["subtype_scale"] * (fr - risk), risk)
        risks.append(risk)

    main_p = np.average(np.vstack(predictions), axis=0, weights=meta["main_weights"])
    z = np.column_stack([main_p] + risks)
    p = meta["stack_intercept"] + z @ np.asarray(meta["stack_coefficients"])

    if meta.get("adaptive_gate", False):
        gate_x = build_gate_features(test, predictions, risks, np.clip(p, 1e-6, 1 - 1e-6))
        gate = CatBoostRegressor()
        gate.load_model(str(MODEL / "adaptive_gate.cbm"))
        gate_pred = safe_cb_predict(gate, gate_x)
        bias_offset = float(meta.get("gate_bias_offset", 0.0))
        gate_clean = gate_pred - bias_offset
        p = p + float(meta.get("gate_scale", 0.08)) * gate_clean

    p = p + float(meta.get("global_shift", 0.0052))

    if futures.any():
        from src.psych_latent import build_production_features, apply_linear_residual
        residual_x = build_production_features(
            test, MODEL / "psych_profile.pkl", MODEL / "latent_pitch_context.csv"
        )
        correction = apply_linear_residual(residual_x, MODEL / "psych_latent_meta.npz")
        p = np.where(futures, p + correction, p)

    v54_feat = None
    p_hb_hier = None
    if regular.any():
        test_reg = test[regular].reset_index(drop=True)
        v54_feat, _ = build_v54_per_season_asof_75_features(
            test_reg, profile_path=MODEL / "team_asof_profile.json", priors=priors, prior=float(meta.get("prior", 0.523766))
        )
        lev_df = build_leverage_features(test_reg)
        v54_feat = pd.concat([v54_feat, lev_df], axis=1)

        prev1_raw = test.loc[regular, "asof_pitcher_prev1_game_success_rate"].to_numpy(float) if "asof_pitcher_prev1_game_success_rate" in test.columns else p_rate_raw[regular]
        prev1_raw = np.where(np.isnan(prev1_raw), p_rate_raw[regular], prev1_raw)
        
        # 113A Core Anchor
        p_hb_hier = np.clip(0.70 * p_rate_raw[regular] + 0.30 * prev1_raw, 0.05, 0.95)

    if regular.any():
        from src.entity_context_split import apply_split_profile, apply_linear_split
        split_profile = pd.read_csv(
            MODEL / "split_profile.csv", dtype={"entity_value": str, "context_value": str}
        )
        split_x = apply_split_profile(test[regular].reset_index(drop=True), split_profile)
        split_correction = apply_linear_split(split_x, MODEL / "split_residual_meta.npz")
        p_split = p[regular] + split_correction
        
        lgbm_model = lgb.Booster(model_file=str(MODEL / 'r_expert_lgbm.txt'))
        x3_reg = x3[regular].reset_index(drop=True)
        res_lgbm = safe_lgb_predict(lgbm_model, x3_reg)
        base3_reg = base3[regular]
        p_lgbm = np.clip(base3_reg + res_lgbm + 0.0052, 1e-6, 1.0 - 1e-6)
        
        w_lgb = float(meta.get("r_expert_lgbm_weight", 0.05))
        p[regular] = (1.0 - w_lgb) * p_split + w_lgb * p_lgbm

    w_deep = float(meta.get("w_deep_hierarchical_base", 0.078))
    v54_seeds = meta.get("v54_seeds", [42, 1, 2, 3, 4, 10])
    if regular.any() and w_deep > 0.0:
        res_preds = []
        for s in v54_seeds:
            cb_m = CatBoostRegressor()
            cb_m.load_model(str(MODEL / f"deep_cb_l2_seed{s}.cbm"))
            res_preds.append(safe_cb_predict(cb_m, v54_feat))
            
            lgb_m = lgb.Booster(model_file=str(MODEL / f"deep_lgb_l2_seed{s}.txt"))
            res_preds.append(safe_lgb_predict(lgb_m, v54_feat))
            
        p_deep_resid = np.mean(res_preds, axis=0)
        p_deep_reconstructed = np.clip(p_hb_hier + p_deep_resid, 1e-5, 1 - 1e-5)
        deep_mean_offset = float(meta.get("deep_mean_offset", 0.514025))
        deep_corr = p_deep_reconstructed - deep_mean_offset
        p[regular] = p[regular] + w_deep * deep_corr

    # 113A Disjoint Empirical Bayes Expert
    if regular.any() and (MODEL / "disjoint_eb_cb.cbm").exists():
        cb_eb = CatBoostRegressor()
        cb_eb.load_model(str(MODEL / "disjoint_eb_cb.cbm"))
        
        n_p = test.loc[regular, "asof_pitcher_n"].fillna(0).to_numpy(float) if "asof_pitcher_n" in test.columns else np.zeros(int(np.sum(regular)))
        n_b = test.loc[regular, "asof_batter_n"].fillna(0).to_numpy(float) if "asof_batter_n" in test.columns else np.zeros(int(np.sum(regular)))
        p_rate_reg = p_rate_raw[regular]
        prev1_reg = prev1_raw
        b_rate = test.loc[regular, "asof_batter_success_rate"].fillna(0.523766).to_numpy(float) if "asof_batter_success_rate" in test.columns else np.full(int(np.sum(regular)), 0.523766)
        b_mid = test.loc[regular, "asof_batter_middle_rate"].fillna(0.523766).to_numpy(float) if "asof_batter_middle_rate" in test.columns else np.full(int(np.sum(regular)), 0.523766)
        
        gamma_p = n_p / (n_p + 20.0)
        gamma_b = n_b / (n_b + 40.0)
        p_shrunk = gamma_p * p_rate_reg + (1.0 - gamma_p) * 0.523766
        b_shrunk = gamma_b * (1.0 - b_rate) + (1.0 - gamma_b) * 0.523766
        
        is_p_left = (test.loc[regular, "pitcher_hand"].astype(str) == "L").to_numpy(float) if "pitcher_hand" in test.columns else np.zeros(int(np.sum(regular)))
        is_b_left = (test.loc[regular, "batter_hand"].astype(str) == "L").to_numpy(float) if "batter_hand" in test.columns else np.zeros(int(np.sum(regular)))
        platoon = (is_p_left != is_b_left).astype(float)
        
        b = test.loc[regular, "balls_before"].fillna(0).to_numpy(float) if "balls_before" in test.columns else np.zeros(int(np.sum(regular)))
        s = test.loc[regular, "strikes_before"].fillna(0).to_numpy(float) if "strikes_before" in test.columns else np.zeros(int(np.sum(regular)))
        li = test.loc[regular, "li"].fillna(0.98).to_numpy(float) if "li" in test.columns else np.full(int(np.sum(regular)), 0.98)
        
        eb_X = pd.DataFrame({
            "eb_pitcher_shrunk": p_shrunk,
            "eb_batter_shrunk": b_shrunk,
            "eb_form_gap": prev1_reg - (1.0 - b_rate),
            "gamma_p": gamma_p,
            "gamma_b": gamma_b,
            "platoon": platoon,
            "count_pressure": (b - s) * (b + s + 1.0) / 7.0,
            "is_2s": (s == 2).astype(float),
            "is_3b": (b == 3).astype(float),
            "asof_pitcher_success_rate": p_rate_reg,
            "asof_pitcher_prev1_game_success_rate": prev1_reg,
            "asof_batter_middle_rate": b_mid,
            "li": li
        })
        eb_res = safe_cb_predict(cb_eb, eb_X)
        w_eb = float(meta.get("w_disjoint_eb", 0.035))
        p[regular] = p[regular] + w_eb * eb_res

    # Context-Adaptive Multi-Channel Leverage Gating
    li_val = test.loc[regular, "li"].fillna(0.98).to_numpy(float) if "li" in test.columns else np.full(int(np.sum(regular)), 0.98)
    s_val = test.loc[regular, "strikes_before"].fillna(0).to_numpy(float) if "strikes_before" in test.columns else np.zeros(int(np.sum(regular)))
    is_high_li = (li_val >= 1.5).astype(float)
    is_2s = (s_val == 2).astype(float)

    # SLSQP-Optimized Refined Call with Dynamic 2-Strike Boost
    w_refined_call = float(meta.get("w_refined_call_base", 0.042))
    if regular.any() and w_refined_call > 0.0 and (MODEL / "refined_call_expert.cbm").exists():
        cb_call = CatBoostClassifier()
        cb_call.load_model(str(MODEL / "refined_call_expert.cbm"))
        x3_cb = x3[regular].reset_index(drop=True).copy()
        for c in CAT_V2:
            if c in x3_cb.columns:
                x3_cb[c] = x3_cb[c].astype(str)
        p_call_success = safe_cb_predict_proba(cb_call, x3_cb)[:, 0]
        call_mean_offset = float(meta.get("call_mean_offset", 0.523766))
        call_corr = p_call_success - call_mean_offset
        w_call_dyn = w_refined_call * (1.0 + 0.25 * is_2s)
        p[regular] = p[regular] + w_call_dyn * call_corr

    # Corrected pocket shifts
    high_pocket_mask = regular & (n_p_raw < 15) & (p_rate_raw > 0.65)
    downshift_val = float(meta.get("pocket_downshift_val", -0.012))
    p = np.where(high_pocket_mask, p + downshift_val, p)

    low_pocket_mask = regular & (n_p_raw < 15) & (p_rate_raw < 0.40)
    upshift_val = float(meta.get("pocket_upshift_val", +0.008))
    p = np.where(low_pocket_mask, p + upshift_val, p)

    # SLSQP-Optimized 15-Model Ensemble with High-Leverage Boost
    w_super_resid = float(meta.get("w_super_resid_base", 0.088))
    r_seeds = meta.get("r_seeds", [42, 1, 2, 3, 4])
    if regular.any() and w_super_resid > 0.0:
        X_61_reg, _ = build_v5_deep_61_features(test[regular].reset_index(drop=True), profile_path=MODEL / "team_asof_profile.json", prior=0.523766)
        prev1_p_val = test.loc[regular, "asof_pitcher_prev1_game_success_rate"].to_numpy(float) if "asof_pitcher_prev1_game_success_rate" in test.columns else p_rate_raw[regular]
        prev1_p_val = np.where(np.isnan(prev1_p_val), p_rate_raw[regular], prev1_p_val)
        b_rate_val = test.loc[regular, "asof_batter_success_rate"].fillna(0.523766).to_numpy(float) if "asof_batter_success_rate" in test.columns else np.full(int(np.sum(regular)), 0.523766)
        
        X_61_reg["form_gap"] = prev1_p_val - b_rate_val
        X_61_reg["anchor_p"] = p[regular]
        
        phys_reg = extract_hyper_regime_tensor_109c(test[regular].reset_index(drop=True))
        X_reg_full = pd.concat([X_61_reg, phys_reg], axis=1)
        
        resid_reg_preds = []
        for s in r_seeds:
            cb_path = MODEL / f"super_resid_cb_seed{s}.cbm"
            if cb_path.exists():
                cb_m = CatBoostRegressor()
                cb_m.load_model(str(cb_path))
                resid_reg_preds.append(safe_cb_predict(cb_m, X_reg_full))
                
            lgb_path = MODEL / f"super_resid_lgb_seed{s}.txt"
            if lgb_path.exists():
                lgb_m = lgb.Booster(model_file=str(lgb_path))
                resid_reg_preds.append(safe_lgb_predict(lgb_m, X_reg_full))
                
            xgb_path = MODEL / f"super_resid_xgb_seed{s}.json"
            if xgb_path.exists():
                xgb_m = XGBRegressor()
                xgb_m.load_model(str(xgb_path))
                resid_reg_preds.append(safe_xgb_predict(xgb_m, X_reg_full))
                
        if resid_reg_preds:
            p_res_reg = np.mean(resid_reg_preds, axis=0)
            w_super_dyn = w_super_resid * (1.0 + 0.20 * is_high_li)
            p[regular] = p[regular] + w_super_dyn * p_res_reg

    # 2군 Futures (w=0.035)
    w_fut_resid = float(meta.get("w_fut_resid", 0.035))
    f_seeds = meta.get("f_seeds", [42, 1, 2, 3])
    if futures.any() and w_fut_resid > 0.0:
        X_61_fut, _ = build_v5_deep_61_features(test[futures].reset_index(drop=True), profile_path=MODEL / "team_asof_profile.json", prior=0.523766)
        X_61_fut["anchor_p"] = p[futures]
        phys_fut = extract_hyper_regime_tensor_109c(test[futures].reset_index(drop=True))
        X_fut_full = pd.concat([X_61_fut, phys_fut], axis=1)
        
        resid_fut_preds = []
        for s in f_seeds:
            cb_path = MODEL / f"fut_resid_cb_seed{s}.cbm"
            if cb_path.exists():
                cb_f = CatBoostRegressor()
                cb_f.load_model(str(cb_path))
                resid_fut_preds.append(safe_cb_predict(cb_f, X_fut_full))
                
        if resid_fut_preds:
            p_res_fut = np.mean(resid_fut_preds, axis=0)
            p[futures] = p[futures] + w_fut_resid * p_res_fut

    # =========================================================================
    # 122: STRICT CONVEX BLEND with Latent Mixture Marginalization
    # p_122 = (1 - w) * p_113A + w * p_mixture
    # =========================================================================
    w_mix = float(meta.get("w_latent_mixture_convex", 0.055))
    p_mix = predict_latent_mixture_122(test, MODEL, priors)
    
    p = (1.0 - w_mix) * p + w_mix * p_mix

    # Monotonic bounded calibration
    p = np.clip(p, 0.02, 0.98)
    if len(p) != len(test) or not np.isfinite(p).all():
        raise RuntimeError("invalid predictions")

    out = ROOT / "output"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_id": row_id, "control_success": p}).to_csv(
        out / "submission.csv", index=False
    )
    print(f"predicted {len(p):,} rows in {time.time() - started:.1f}s")

if __name__ == "__main__":
    main()
'''
    (OUT_DIR / "script.py").write_text(script_code, encoding="utf-8")
    
    # 7. Create ZIP Package
    print(f"Packaging {OUT_DIR} -> {ZIP_OUT}...")
    ZIP_OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for file in OUT_DIR.rglob("*"):
            if file.is_file():
                arcname = file.relative_to(OUT_DIR)
                z.write(file, arcname)
                
    zip_mb = ZIP_OUT.stat().st_size / (1024 * 1024)
    print(f"Created {ZIP_OUT} ({zip_mb:.2f} MB)")

if __name__ == "__main__":
    main()
