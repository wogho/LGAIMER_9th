#!/usr/bin/env python3
"""Build Production Package 109B: External Repositories Deep Fusion - Hand Matchup Asymmetry (Asymmetric Matchup Model c4p3) + Recent Form Drift + 3D Physics 10-Model Residual Ensemble."""
import gc, hashlib, json, os, shutil, sys, time, zipfile
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'model/REF4-SUPER-ENSEMBLE-107A/production_package'))

EXP_DIR = ROOT / 'model/REF4-SUPER-ENSEMBLE-109B'
PROD_DIR = EXP_DIR / 'production_package'
MODEL_DIR = PROD_DIR / 'model'
SRC_DIR = PROD_DIR / 'src'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
SRC_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_107_MODEL = ROOT / 'model/REF4-SUPER-ENSEMBLE-107A/production_package/model'
SOURCE_107_SRC = ROOT / 'model/REF4-SUPER-ENSEMBLE-107A/production_package/src'
SOURCE_107_REQS = ROOT / 'model/REF4-SUPER-ENSEMBLE-107A/production_package/requirements.txt'

from src.v5_deep_61_features import build_v5_deep_61_features
from src.v54_per_season_asof_75_features import build_per_season_priors, build_v54_per_season_asof_75_features
from src.preprocessing_v2 import build_v2_features, build_v3_features, CAT_V2
from src.adaptive_gate import build_gate_features

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

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

def extract_advanced_physics_109b(df: pd.DataFrame) -> pd.DataFrame:
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
    
    p_hand = df["pitcher_hand"].astype(str) if "pitcher_hand" in df.columns else pd.Series(["R"] * len(df), index=df.index)
    b_hand = df["batter_hand"].astype(str) if "batter_hand" in df.columns else pd.Series(["R"] * len(df), index=df.index)
    
    is_p_left = (p_hand == "L").astype(float)
    is_b_left = (b_hand == "L").astype(float)
    hand_platoon = (is_p_left != is_b_left).astype(float)
    hand_match_code = (is_p_left * 2.0 + is_b_left).astype(float)
    
    n_p = df["asof_pitcher_n"].fillna(0).to_numpy(float) if "asof_pitcher_n" in df.columns else np.zeros(len(df))
    p_rate = df["asof_pitcher_success_rate"].fillna(0.523766).to_numpy(float) if "asof_pitcher_success_rate" in df.columns else np.full(len(df), 0.523766)
    prev1_p = df["asof_pitcher_prev1_game_success_rate"].fillna(0.523766).to_numpy(float) if "asof_pitcher_prev1_game_success_rate" in df.columns else p_rate
    recent_drift = (prev1_p - p_rate) * (n_p / (n_p + 10.0))
    
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
        "recent_drift": recent_drift
    }, index=df.index)

def predict_103a_full(df, MODEL, meta, priors, ps, bs, ms, tm, seeds, regime):
    cache_path = ROOT / "candidate/p_103_full_train.npy"
    if cache_path.exists():
        print(f"Loading cached 103A predictions from {cache_path}...")
        return np.load(cache_path)
        
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
    
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, p_103)
    print(f"Saved 103A predictions cache to {cache_path}")
    return p_103

SCRIPT_CONTENT_109B = '''"""Offline inference entry point for REF4-SUPER-ENSEMBLE-109B."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from src.preprocessing_v2 import build_v2_features, build_v3_features, CAT_V2
from src.adaptive_gate import build_gate_features
from src.v54_per_season_asof_75_features import build_v54_per_season_asof_75_features
from src.v5_deep_61_features import build_v5_deep_61_features

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "model"


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


def extract_advanced_physics_109b(df: pd.DataFrame) -> pd.DataFrame:
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
    
    p_hand = df["pitcher_hand"].astype(str) if "pitcher_hand" in df.columns else pd.Series(["R"] * len(df), index=df.index)
    b_hand = df["batter_hand"].astype(str) if "batter_hand" in df.columns else pd.Series(["R"] * len(df), index=df.index)
    
    is_p_left = (p_hand == "L").astype(float)
    is_b_left = (b_hand == "L").astype(float)
    hand_platoon = (is_p_left != is_b_left).astype(float)
    hand_match_code = (is_p_left * 2.0 + is_b_left).astype(float)
    
    n_p = df["asof_pitcher_n"].fillna(0).to_numpy(float) if "asof_pitcher_n" in df.columns else np.zeros(len(df))
    p_rate = df["asof_pitcher_success_rate"].fillna(0.523766).to_numpy(float) if "asof_pitcher_success_rate" in df.columns else np.full(len(df), 0.523766)
    prev1_p = df["asof_pitcher_prev1_game_success_rate"].fillna(0.523766).to_numpy(float) if "asof_pitcher_prev1_game_success_rate" in df.columns else p_rate
    recent_drift = (prev1_p - p_rate) * (n_p / (n_p + 10.0))
    
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
        "recent_drift": recent_drift
    }, index=df.index)


def main():
    started = time.time()
    meta = json.loads((MODEL / "manifest.json").read_text(encoding="utf-8"))
    priors = json.loads((MODEL / "per_season_priors.json").read_text(encoding="utf-8")) if (MODEL / "per_season_priors.json").exists() else {}
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
        member = [np.clip(base + m.predict(x), 1e-6, 1 - 1e-6) for m in load_reg(stem)]
        predictions.append(np.mean(member, axis=0))

    regime = json.loads((MODEL / "f_regime_meta.json").read_text())
    futures = test["game_type"].eq("F").to_numpy()
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
        gate_x = build_gate_features(test, predictions, risks, np.clip(p, 1e-6, 1 - 1e-6))
        gate = CatBoostRegressor()
        gate.load_model(str(MODEL / "adaptive_gate.cbm"))
        gate_pred = gate.predict(gate_x)
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
        v54_feat, _ = build_v54_per_season_asof_75_features(
            test, profile_path=MODEL / "team_asof_profile.json", priors=priors, prior=float(meta.get("prior", 0.523766))
        )
        lev_df = build_leverage_features(test)
        v54_feat = pd.concat([v54_feat, lev_df], axis=1)

        prev1_raw = test["asof_pitcher_prev1_game_success_rate"].to_numpy(float) if "asof_pitcher_prev1_game_success_rate" in test.columns else p_rate_raw
        prev1_raw = np.where(np.isnan(prev1_raw), p_rate_raw, prev1_raw)
        
        overall_prior = float(meta.get("prior", 0.523766))
        k_prior = 25.0
        rel_weight = n_p_raw / (n_p_raw + k_prior)
        p_shrunk = overall_prior + rel_weight * (p_rate_raw - overall_prior)
        p_hb_hier = np.clip(0.70 * p_shrunk + 0.30 * prev1_raw, 0.05, 0.95)

    if regular.any():
        from src.entity_context_split import apply_split_profile, apply_linear_split
        split_profile = pd.read_csv(
            MODEL / "split_profile.csv", dtype={"entity_value": str, "context_value": str}
        )
        split_x = apply_split_profile(test, split_profile)
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

    w_deep = float(meta.get("w_deep_hierarchical", 0.08))
    v54_seeds = meta.get("v54_seeds", [42, 1, 2, 3, 4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150])
    if regular.any() and w_deep > 0.0:
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
        p = np.where(regular, p + w_deep * deep_corr, p)

    w_refined_call = float(meta.get("w_refined_call", 0.04))
    if regular.any() and w_refined_call > 0.0 and (MODEL / "refined_call_expert.cbm").exists():
        cb_call = CatBoostClassifier()
        cb_call.load_model(str(MODEL / "refined_call_expert.cbm"))
        
        x3_cb = x3.copy()
        for c in CAT_V2:
            if c in x3_cb.columns:
                x3_cb[c] = x3_cb[c].astype(str)
                
        call_probs = cb_call.predict_proba(x3_cb)
        p_call_success = call_probs[:, 0]
        call_mean_offset = float(meta.get("call_mean_offset", 0.523766))
        call_corr = p_call_success - call_mean_offset
        p = np.where(regular, p + w_refined_call * call_corr, p)

    high_pocket_mask = regular & (n_p_raw < 15) & (p_rate_raw > 0.65)
    downshift_val = float(meta.get("pocket_downshift_val", -0.012))
    p = np.where(high_pocket_mask, p + downshift_val, p)

    low_pocket_mask = regular & (n_p_raw < 15) & (p_rate_raw < 0.40)
    upshift_val = float(meta.get("pocket_upshift_val", +0.008))
    p = np.where(low_pocket_mask, p + upshift_val, p)

    # 109B: 10-Model Residual Boosters with Hand Asymmetry & Form Drift (w=0.08)
    w_super_resid = float(meta.get("w_super_resid", 0.08))
    r_seeds = meta.get("r_seeds", [42, 1, 2, 3, 4])
    if regular.any() and w_super_resid > 0.0:
        X_61_reg, _ = build_v5_deep_61_features(test[regular].reset_index(drop=True), profile_path=MODEL / "team_asof_profile.json", prior=0.523766)
        prev1_p_val = test.loc[regular, "asof_pitcher_prev1_game_success_rate"].to_numpy(float) if "asof_pitcher_prev1_game_success_rate" in test.columns else p_rate_raw[regular]
        prev1_p_val = np.where(np.isnan(prev1_p_val), p_rate_raw[regular], prev1_p_val)
        b_rate_val = test.loc[regular, "asof_batter_success_rate"].fillna(0.523766).to_numpy(float) if "asof_batter_success_rate" in test.columns else np.full(np.sum(regular), 0.523766)
        
        X_61_reg["form_gap"] = prev1_p_val - b_rate_val
        X_61_reg["anchor_p"] = p[regular]
        
        phys_reg = extract_advanced_physics_109b(test[regular].reset_index(drop=True))
        X_reg_full = pd.concat([X_61_reg, phys_reg], axis=1)
        
        resid_reg_preds = []
        for s in r_seeds:
            cb_path = MODEL / f"super_resid_cb_seed{s}.cbm"
            if cb_path.exists():
                cb_m = CatBoostRegressor()
                cb_m.load_model(str(cb_path))
                resid_reg_preds.append(cb_m.predict(X_reg_full))
                
            lgb_path = MODEL / f"super_resid_lgb_seed{s}.txt"
            if lgb_path.exists():
                lgb_m = lgb.Booster(model_file=str(lgb_path))
                resid_reg_preds.append(lgb_m.predict(X_reg_full))
                
        if resid_reg_preds:
            p_res_reg = np.mean(resid_reg_preds, axis=0)
            p[regular] = p[regular] + w_super_resid * p_res_reg

    # 2군 Futures (w=0.04)
    w_fut_resid = float(meta.get("w_fut_resid", 0.04))
    f_seeds = meta.get("f_seeds", [42, 1, 2, 3])
    if futures.any() and w_fut_resid > 0.0:
        X_61_fut, _ = build_v5_deep_61_features(test[futures].reset_index(drop=True), profile_path=MODEL / "team_asof_profile.json", prior=0.523766)
        X_61_fut["anchor_p"] = p[futures]
        phys_fut = extract_advanced_physics_109b(test[futures].reset_index(drop=True))
        X_fut_full = pd.concat([X_61_fut, phys_fut], axis=1)
        
        resid_fut_preds = []
        for s in f_seeds:
            cb_path = MODEL / f"fut_resid_cb_seed{s}.cbm"
            if cb_path.exists():
                cb_f = CatBoostRegressor()
                cb_f.load_model(str(cb_path))
                resid_fut_preds.append(cb_f.predict(X_fut_full))
                
        if resid_fut_preds:
            p_res_fut = np.mean(resid_fut_preds, axis=0)
            p[futures] = p[futures] + w_fut_resid * p_res_fut

    p = np.clip(p, 1e-5, 1 - 1e-5)
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

def build_package():
    t0 = time.time()
    print("=== Step 1: Copying Base Models & Assets from 107A ===")
    for f in SOURCE_107_MODEL.iterdir():
        if f.is_file() and f.name != 'manifest.json':
            shutil.copy2(f, MODEL_DIR / f.name)
            
    for f in SOURCE_107_SRC.iterdir():
        if f.is_file():
            shutil.copy2(f, SRC_DIR / f.name)
            
    shutil.copy2(SOURCE_107_REQS, PROD_DIR / 'requirements.txt')
    
    print("\n=== Step 2: Training 109B 10-Model Boosters with Hand Asymmetry & Form Drift on Train ===")
    raw = pd.read_csv(ROOT / "data/train.csv", low_memory=False)
    priors = build_per_season_priors(raw)
    (MODEL_DIR / 'per_season_priors.json').write_text(json.dumps(priors, indent=2), encoding='utf-8')
    
    ps = pd.read_pickle(SOURCE_107_MODEL / "pitcher_snapshots.pkl")
    bs = pd.read_pickle(SOURCE_107_MODEL / "batter_snapshots.pkl")
    ms = pd.read_pickle(SOURCE_107_MODEL / "pitchmix_snapshots.pkl")
    tm = str(SOURCE_107_MODEL / "trackman_prior_features.csv")
    seeds = [260802, 260803, 260804, 260805, 260806, 260807]
    regime = json.loads((SOURCE_107_MODEL / "f_regime_meta.json").read_text())
    meta_107 = json.loads((SOURCE_107_MODEL / "manifest.json").read_text(encoding='utf-8'))
    
    p_team_map = raw.groupby("pitcher_team_id")["control_success"].mean().to_dict()
    b_team_map = raw.groupby("batter_team_id")["control_success"].mean().to_dict()
    prof_dict = {"p_team": {str(k): float(v) for k, v in p_team_map.items()},
                 "b_team": {str(k): float(v) for k, v in b_team_map.items()},
                 "prior": 0.523766}
    (MODEL_DIR / "team_asof_profile.json").write_text(json.dumps(prof_dict, indent=2), encoding="utf-8")
    
    print("Computing 103A anchor predictions across full training set...")
    p_103_full = predict_103a_full(raw, SOURCE_107_MODEL, meta_107, priors, ps, bs, ms, tm, seeds, regime)
    y_full = raw["control_success"].to_numpy(float)
    anchor_resid = y_full - p_103_full
    
    # 1군 Regular
    reg_mask = raw["game_type"] != "F"
    raw_reg = raw[reg_mask].copy().reset_index(drop=True)
    y_resid_reg = anchor_resid[reg_mask]
    
    X_61_reg, _ = build_v5_deep_61_features(raw_reg, profile_path=MODEL_DIR / "team_asof_profile.json", prior=0.523766)
    prev1_p = raw_reg["asof_pitcher_prev1_game_success_rate"].fillna(0.523766).to_numpy(float) if "asof_pitcher_prev1_game_success_rate" in raw_reg.columns else raw_reg["asof_pitcher_success_rate"].fillna(0.523766).to_numpy(float)
    b_rate = raw_reg["asof_batter_success_rate"].fillna(0.523766).to_numpy(float)
    X_61_reg["form_gap"] = prev1_p - b_rate
    X_61_reg["anchor_p"] = p_103_full[reg_mask]
    
    phys_reg = extract_advanced_physics_109b(raw_reg)
    X_physics_reg = pd.concat([X_61_reg, phys_reg], axis=1)
    
    r_seeds = [42, 1, 2, 3, 4]
    print(f"\nFitting 1군 10 Boosters (5 CatBoost + 5 LightGBM) on {len(X_physics_reg):,} Regular rows...")
    for s in r_seeds:
        cb = CatBoostRegressor(iterations=220, depth=5, learning_rate=0.03, l2_leaf_reg=15.0, random_seed=s, verbose=False, thread_count=-1)
        cb.fit(X_physics_reg, y_resid_reg)
        cb.save_model(str(MODEL_DIR / f"super_resid_cb_seed{s}.cbm"))
        
        lgb_params = {"objective": "regression", "metric": "rmse", "learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "random_seed": s, "verbose": -1, "n_jobs": -1}
        dtrain = lgb.Dataset(X_physics_reg, label=y_resid_reg)
        bst = lgb.train(lgb_params, dtrain, num_boost_round=160)
        bst.save_model(str(MODEL_DIR / f"super_resid_lgb_seed{s}.txt"))
        print(f"  • Regular Seed {s}: CatBoost & LightGBM fitted and saved")
        
    # 2군 Futures
    fut_mask = raw["game_type"] == "F"
    raw_fut = raw[fut_mask].copy().reset_index(drop=True)
    y_resid_fut = anchor_resid[fut_mask]
    
    X_61_fut, _ = build_v5_deep_61_features(raw_fut, profile_path=MODEL_DIR / "team_asof_profile.json", prior=0.523766)
    X_61_fut["anchor_p"] = p_103_full[fut_mask]
    phys_fut = extract_advanced_physics_109b(raw_fut)
    X_physics_fut = pd.concat([X_61_fut, phys_fut], axis=1)
    
    f_seeds = [42, 1, 2, 3]
    print(f"\nFitting 2군 4 Futures Boosters (4 CatBoost) on {len(X_physics_fut):,} Futures rows...")
    for s in f_seeds:
        cb_f = CatBoostRegressor(iterations=150, depth=4, learning_rate=0.03, l2_leaf_reg=20.0, random_seed=s, verbose=False, thread_count=-1)
        cb_f.fit(X_physics_fut, y_resid_fut)
        cb_f.save_model(str(MODEL_DIR / f"fut_resid_cb_seed{s}.cbm"))
        print(f"  • Futures Seed {s}: CatBoost fitted and saved")
        
    print("\n=== Step 3: Updating Manifest for 109B ===")
    manifest_path = MODEL_DIR / 'manifest.json'
    manifest = json.loads((SOURCE_107_MODEL / 'manifest.json').read_text(encoding='utf-8'))
    manifest['version'] = 'REF4-SUPER-ENSEMBLE-109B'
    manifest['w_super_resid'] = 0.08
    manifest['w_fut_resid'] = 0.04
    manifest['r_seeds'] = r_seeds
    manifest['f_seeds'] = f_seeds
    manifest['notes'] = "109B 108C Backbone + Hand Matchup Asymmetry (Asymmetric Matchup Model c4p3) + Form Drift + 3D Physics 10-Model Residual Ensemble"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    
    print("\n=== Step 4: Packaging submit_ref4_super_ensemble_109B.zip ===")
    (PROD_DIR / 'script.py').write_text(SCRIPT_CONTENT_109B, encoding='utf-8')
    
    zip_path = ROOT / 'output/submit_ref4_super_ensemble_109B.zip'
    if zip_path.exists():
        zip_path.unlink()
        
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(PROD_DIR / 'requirements.txt', 'requirements.txt')
        zf.write(PROD_DIR / 'script.py', 'script.py')
        for root_dir, _, files in os.walk(MODEL_DIR):
            for file in files:
                abs_f = Path(root_dir) / file
                rel_f = abs_f.relative_to(PROD_DIR)
                zf.write(abs_f, str(rel_f))
        for root_dir, _, files in os.walk(SRC_DIR):
            for file in files:
                abs_f = Path(root_dir) / file
                rel_f = abs_f.relative_to(PROD_DIR)
                zf.write(abs_f, str(rel_f))
                
    zip_hash = sha256_file(zip_path)
    zip_size = zip_path.stat().st_size
    print(f"\nCreated ZIP: {zip_path.name}")
    print(f"Size: {zip_size:,} bytes ({zip_size / (1024*1024):.2f} MB)")
    print(f"SHA-256: {zip_hash}")
    print(f"Build completed in {time.time() - t0:.1f}s")

if __name__ == '__main__':
    build_package()
