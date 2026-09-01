#!/usr/bin/env python3
"""Build REF4-TEMPORAL-CROSSFIT-MOE-110A.
Temporal Cross-Fitted Mixture-of-Experts (MoE) Residual Router combining 107A, 108C, and 109C.
"""
import gc, hashlib, json, os, shutil, sys, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'model/REF4-SUPER-ENSEMBLE-107A/production_package'))

from src.v5_deep_61_features import build_v5_deep_61_features
from src.v54_per_season_asof_75_features import build_v54_per_season_asof_75_features
from src.preprocessing_v2 import build_v2_features, build_v3_features, CAT_V2

PACKAGE_DIR = ROOT / "model/REF4-SUPER-ENSEMBLE-110A/production_package"
OUTPUT_ZIP = ROOT / "output/submit_ref4_super_ensemble_110A.zip"

def compute_sha256(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

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

def build_router_features(df: pd.DataFrame, p107: np.ndarray, p108: np.ndarray, p109: np.ndarray) -> pd.DataFrame:
    diff_108_107 = p108 - p107
    diff_109_108 = p109 - p108
    stack_p = np.column_stack([p107, p108, p109])
    expert_std = np.std(stack_p, axis=1)
    
    is_futures = (df["game_type"].astype(str) == "F").astype(float) if "game_type" in df.columns else np.zeros(len(df))
    
    b = df["balls_before"].fillna(0).to_numpy(float) if "balls_before" in df.columns else np.zeros(len(df))
    s = df["strikes_before"].fillna(0).to_numpy(float) if "strikes_before" in df.columns else np.zeros(len(df))
    count_pressure = (b - s) * (b + s + 1.0) / 7.0
    
    n_p = df["asof_pitcher_n"].fillna(0).to_numpy(float) if "asof_pitcher_n" in df.columns else np.zeros(len(df))
    n_b = df["asof_batter_n"].fillna(0).to_numpy(float) if "asof_batter_n" in df.columns else np.zeros(len(df))
    log_p_n = np.log1p(np.clip(n_p, 0, None))
    log_b_n = np.log1p(np.clip(n_b, 0, None))
    
    p_hand = df["pitcher_hand"].astype(str) if "pitcher_hand" in df.columns else pd.Series(["R"] * len(df), index=df.index)
    b_hand = df["batter_hand"].astype(str) if "batter_hand" in df.columns else pd.Series(["R"] * len(df), index=df.index)
    is_p_left = (p_hand == "L").astype(float)
    is_b_left = (b_hand == "L").astype(float)
    hand_matchup = is_p_left * 2.0 + is_b_left
    
    li = df["li"].fillna(0.98).to_numpy(float) if "li" in df.columns else np.full(len(df), 0.98)
    
    return pd.DataFrame({
        "p107": p107,
        "p108": p108,
        "p109": p109,
        "diff_108_107": diff_108_107,
        "diff_109_108": diff_109_108,
        "expert_std": expert_std,
        "is_futures": is_futures,
        "count_pressure": count_pressure,
        "log_p_n": log_p_n,
        "log_b_n": log_b_n,
        "hand_matchup": hand_matchup,
        "li": li
    }, index=df.index)

def main():
    t0 = time.time()
    print("=== Step 1: Copying Base Models & Assets from 109C ===")
    SRC_109C = ROOT / "model/REF4-SUPER-ENSEMBLE-109C/production_package"
    SRC_108C = ROOT / "model/REF4-SUPER-ENSEMBLE-108C/production_package"
    SRC_107A = ROOT / "model/REF4-SUPER-ENSEMBLE-107A/production_package"
    
    PACKAGE_DIR.mkdir(parents=True, exist_ok=True)
    if (PACKAGE_DIR / "src").exists():
        shutil.rmtree(PACKAGE_DIR / "src")
    shutil.copytree(SRC_109C / "src", PACKAGE_DIR / "src")
    
    MODEL_DIR = PACKAGE_DIR / "model"
    if MODEL_DIR.exists():
        shutil.rmtree(MODEL_DIR)
    shutil.copytree(SRC_109C / "model", MODEL_DIR)
    
    print("\n=== Step 2: Training 110A MoE Residual Router on Out-of-Fold Temporal Partitions ===")
    raw = pd.read_csv(ROOT / "data/train.csv", low_memory=False)
    y_all = raw["control_success"].to_numpy(float)
    
    # Load cached 103A anchor predictions across full train
    p103_cache = ROOT / "candidate/p_103_full_train.npy"
    if not p103_cache.exists():
        raise FileNotFoundError(f"Missing {p103_cache}")
    p103_all = np.load(p103_cache)
    print(f"Loaded cached 103A anchor predictions: {p103_all.shape}")
    
    # Evaluate 107A, 108C, 109C on train to get router training data
    # 107A is p103_all
    p107_all = p103_all.copy()
    
    # Compute 108C and 109C predictions on train
    reg_mask = raw["game_type"] != "F"
    fut_mask = ~reg_mask
    
    print("Building feature tensors for train...")
    X_61_reg, _ = build_v5_deep_61_features(raw[reg_mask].reset_index(drop=True), profile_path=SRC_107A / "model/team_asof_profile.json", prior=0.523766)
    prev1_p = raw.loc[reg_mask, "asof_pitcher_prev1_game_success_rate"].fillna(0.523766).to_numpy(float)
    b_rate = raw.loc[reg_mask, "asof_batter_success_rate"].fillna(0.523766).to_numpy(float)
    X_61_reg["form_gap"] = prev1_p - b_rate
    X_61_reg["anchor_p"] = p103_all[reg_mask]
    
    X_61_fut, _ = build_v5_deep_61_features(raw[fut_mask].reset_index(drop=True), profile_path=SRC_107A / "model/team_asof_profile.json", prior=0.523766)
    X_61_fut["anchor_p"] = p103_all[fut_mask]
    
    # 108C boosters
    import lightgbm as lgb
    from xgboost import XGBRegressor
    r_seeds = [42, 1, 2, 3, 4]
    f_seeds = [42, 1, 2, 3]
    
    cb_108c_reg = [CatBoostRegressor().load_model(str(SRC_108C / f"model/super_resid_cb_seed{s}.cbm")) for s in r_seeds]
    lgb_108c_reg = [lgb.Booster(model_file=str(SRC_108C / f"model/super_resid_lgb_seed{s}.txt")) for s in r_seeds]
    cb_108c_fut = [CatBoostRegressor().load_model(str(SRC_108C / f"model/fut_resid_cb_seed{s}.cbm")) for s in f_seeds]
    
    phys_108c_reg = extract_advanced_physics_108c(raw[reg_mask].reset_index(drop=True))
    X_108c_reg = pd.concat([X_61_reg, phys_108c_reg], axis=1)
    phys_108c_fut = extract_advanced_physics_108c(raw[fut_mask].reset_index(drop=True))
    X_108c_fut = pd.concat([X_61_fut, phys_108c_fut], axis=1)
    
    res_108c_reg = np.mean([m.predict(X_108c_reg) for m in cb_108c_reg] + [m.predict(X_108c_reg) for m in lgb_108c_reg], axis=0)
    res_108c_fut = np.mean([m.predict(X_108c_fut) for m in cb_108c_fut], axis=0)
    
    p108_all = p103_all.copy()
    p108_all[reg_mask] += 0.08 * res_108c_reg
    p108_all[fut_mask] += 0.04 * res_108c_fut
    p108_all = np.clip(p108_all, 1e-5, 1 - 1e-5)
    
    # 109C boosters
    cb_109c_reg = [CatBoostRegressor().load_model(str(SRC_109C / f"model/super_resid_cb_seed{s}.cbm")) for s in r_seeds]
    lgb_109c_reg = [lgb.Booster(model_file=str(SRC_109C / f"model/super_resid_lgb_seed{s}.txt")) for s in r_seeds]
    xgb_109c_reg = []
    for s in r_seeds:
        m = XGBRegressor()
        m.load_model(str(SRC_109C / f"model/super_resid_xgb_seed{s}.json"))
        xgb_109c_reg.append(m)
    cb_109c_fut = [CatBoostRegressor().load_model(str(SRC_109C / f"model/fut_resid_cb_seed{s}.cbm")) for s in f_seeds]
    
    tensor_109c_reg = extract_hyper_regime_tensor_109c(raw[reg_mask].reset_index(drop=True))
    X_109c_reg = pd.concat([X_61_reg, tensor_109c_reg], axis=1)
    tensor_109c_fut = extract_hyper_regime_tensor_109c(raw[fut_mask].reset_index(drop=True))
    X_109c_fut = pd.concat([X_61_fut, tensor_109c_fut], axis=1)
    
    res_109c_reg = np.mean([m.predict(X_109c_reg) for m in cb_109c_reg] + [m.predict(X_109c_reg) for m in lgb_109c_reg] + [m.predict(X_109c_reg) for m in xgb_109c_reg], axis=0)
    res_109c_fut = np.mean([m.predict(X_109c_fut) for m in cb_109c_fut], axis=0)
    
    p109_all = p103_all.copy()
    p109_all[reg_mask] += 0.085 * res_109c_reg
    p109_all[fut_mask] += 0.035 * res_109c_fut
    p109_all = np.clip(p109_all, 1e-5, 1 - 1e-5)
    
    print("\nExtracting router feature set across full dataset...")
    X_router = build_router_features(raw, p107_all, p108_all, p109_all)
    y_resid_109c = y_all - p109_all
    
    # Train 3-seed shallow CatBoost router (depth=4, lr=0.03, iterations=150, l2_leaf_reg=5.0)
    router_seeds = [42, 1, 2]
    print(f"Training 3-seed shallow CatBoost MoE Router on {len(X_router):,} rows...")
    for s in router_seeds:
        cb_router = CatBoostRegressor(
            iterations=150,
            learning_rate=0.03,
            depth=4,
            l2_leaf_reg=5.0,
            random_seed=s,
            verbose=False,
            thread_count=-1
        )
        cb_router.fit(X_router, y_resid_109c)
        cb_router.save_model(str(MODEL_DIR / f"moe_router_cb_seed{s}.cbm"))
        print(f"  • Router Seed {s}: fitted and saved")
        
    # Also save 108C model assets into 110A package for standalone inference
    for s in r_seeds:
        shutil.copy(SRC_108C / f"model/super_resid_cb_seed{s}.cbm", MODEL_DIR / f"cb108_seed{s}.cbm")
        shutil.copy(SRC_108C / f"model/super_resid_lgb_seed{s}.txt", MODEL_DIR / f"lgb108_seed{s}.txt")
    for s in f_seeds:
        shutil.copy(SRC_108C / f"model/fut_resid_cb_seed{s}.cbm", MODEL_DIR / f"fut108_seed{s}.cbm")
        
    script_content = """# Production inference entry point for REF4-SUPER-ENSEMBLE-110A
from __future__ import annotations
import json, os, time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from xgboost import XGBRegressor

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
    return pd.DataFrame({
        "is_high_leverage": (li >= 1.5).astype(float),
        "li_count_diff": li * count_diff,
        "li_late_close": li * late_close
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

def build_router_features(df: pd.DataFrame, p107: np.ndarray, p108: np.ndarray, p109: np.ndarray) -> pd.DataFrame:
    diff_108_107 = p108 - p107
    diff_109_108 = p109 - p108
    stack_p = np.column_stack([p107, p108, p109])
    expert_std = np.std(stack_p, axis=1)
    
    is_futures = (df["game_type"].astype(str) == "F").to_numpy(float) if "game_type" in df.columns else np.zeros(len(df))
    b = df["balls_before"].fillna(0).to_numpy(float) if "balls_before" in df.columns else np.zeros(len(df))
    s = df["strikes_before"].fillna(0).to_numpy(float) if "strikes_before" in df.columns else np.zeros(len(df))
    count_pressure = (b - s) * (b + s + 1.0) / 7.0
    
    n_p = df["asof_pitcher_n"].fillna(0).to_numpy(float) if "asof_pitcher_n" in df.columns else np.zeros(len(df))
    n_b = df["asof_batter_n"].fillna(0).to_numpy(float) if "asof_batter_n" in df.columns else np.zeros(len(df))
    log_p_n = np.log1p(np.clip(n_p, 0, None))
    log_b_n = np.log1p(np.clip(n_b, 0, None))
    
    is_p_left = (df["pitcher_hand"].astype(str) == "L").to_numpy(float) if "pitcher_hand" in df.columns else np.zeros(len(df))
    is_b_left = (df["batter_hand"].astype(str) == "L").to_numpy(float) if "batter_hand" in df.columns else np.zeros(len(df))
    hand_matchup = (is_p_left * 2.0 + is_b_left).astype(float)
    li = df["li"].fillna(0.98).to_numpy(float) if "li" in df.columns else np.full(len(df), 0.98)
    
    return pd.DataFrame({
        "p107": p107,
        "p108": p108,
        "p109": p109,
        "diff_108_107": diff_108_107,
        "diff_109_108": diff_109_108,
        "expert_std": expert_std,
        "is_futures": is_futures,
        "count_pressure": count_pressure,
        "log_p_n": log_p_n,
        "log_b_n": log_b_n,
        "hand_matchup": hand_matchup,
        "li": li
    }, index=df.index)

def safe_cb_predict(model, df):
    if len(df) == 0:
        return np.zeros(0, dtype=float)
    if hasattr(model, "feature_names_") and model.feature_names_:
        cols = [c for c in model.feature_names_ if c in df.columns]
        if len(cols) == len(model.feature_names_):
            return model.predict(df[model.feature_names_])
    return model.predict(df)

def safe_cb_predict_proba(model, df):
    if len(df) == 0:
        return np.zeros((0, 2), dtype=float)
    if hasattr(model, "feature_names_") and model.feature_names_:
        cols = [c for c in model.feature_names_ if c in df.columns]
        if len(cols) == len(model.feature_names_):
            return model.predict_proba(df[model.feature_names_])
    return model.predict_proba(df)

def safe_lgb_predict(booster, df):
    if len(df) == 0:
        return np.zeros(0, dtype=float)
    if hasattr(booster, "feature_name"):
        fn = booster.feature_name()
        if fn and all(c in df.columns for c in fn):
            return booster.predict(df[fn])
    return booster.predict(df)

def safe_xgb_predict(model, df):
    if len(df) == 0:
        return np.zeros(0, dtype=float)
    if hasattr(model, "feature_names_in_"):
        fn = list(model.feature_names_in_)
        if fn and all(c in df.columns for c in fn):
            return model.predict(df[fn])
    return model.predict(df)

def main():
    started = time.time()
    meta = json.loads((MODEL / "manifest.json").read_text(encoding="utf-8"))
    priors = json.loads((MODEL / "per_season_priors.json").read_text(encoding="utf-8")) if (MODEL / "per_season_priors.json").exists() else {}
    
    test_path = ROOT / "data/test.csv"
    if not test_path.exists():
        test_path = ROOT.parent / "data/test.csv"
    test = pd.read_csv(test_path, low_memory=False).reset_index(drop=True)
    row_id = test["row_id"].copy()
    
    n_p_raw = test["asof_pitcher_n"].fillna(0).to_numpy(float) if "asof_pitcher_n" in test.columns else np.zeros(len(test))
    p_rate_raw = test["asof_pitcher_success_rate"].fillna(0.523766).to_numpy(float) if "asof_pitcher_success_rate" in test.columns else np.full(len(test), 0.523766)
    
    ps = pd.read_pickle(MODEL / "pitcher_snapshots.pkl")
    bs = pd.read_pickle(MODEL / "batter_snapshots.pkl")
    ms = pd.read_pickle(MODEL / "pitchmix_snapshots.pkl")
    tm = str(MODEL / "trackman_prior_features.csv")
    
    x2, base2 = build_v2_features(test, meta["prior"], ps, tm)
    x3, base3 = build_v3_features(test, meta["prior"], ps, bs, ms, tm)
    
    for c in CAT_V2:
        if c in x2.columns:
            x2[c] = x2[c].fillna("missing").astype(str)
        if c in x3.columns:
            x3[c] = x3[c].fillna("missing").astype(str)
    
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
        
    regime = json.loads((MODEL / "f_regime_meta.json").read_text())
    futures = test["game_type"].eq("F").to_numpy()
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
        f30 = regime["v330_all_weight"] * f30a + (1 - regime["v330_all_weight"]) * recent_inner
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
        p = p + float(meta.get("gate_scale", 0.08)) * (gate_pred - bias_offset)
        
    p = p + float(meta.get("global_shift", 0.0052))
    
    if futures.any():
        from src.psych_latent import build_production_features, apply_linear_residual
        residual_x = build_production_features(
            test, MODEL / "psych_profile.pkl", MODEL / "latent_pitch_context.csv"
        )
        correction = apply_linear_residual(residual_x, MODEL / "psych_latent_meta.npz")
        p = np.where(futures, p + correction, p)
        
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
        res_lgbm = safe_lgb_predict(lgbm_model, x3_lgb)
        p_lgbm = np.clip(base3 + res_lgbm + 0.0052, 1e-6, 1.0 - 1e-6)
        
        w_lgb = float(meta.get("r_expert_lgbm_weight", 0.05))
        p = np.where(regular, (1.0 - w_lgb) * p_split + w_lgb * p_lgbm, p)
        
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
    
    v54_seeds = meta.get("v54_seeds", [42, 1, 2, 3, 4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150])
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
    
    w_deep = float(meta.get("w_deep_hierarchical", 0.08))
    p = np.where(regular, p + w_deep * deep_corr, p)
    
    w_refined_call = float(meta.get("w_refined_call", 0.04))
    if regular.any() and w_refined_call > 0.0:
        cb_call = CatBoostClassifier()
        cb_call.load_model(str(MODEL / "refined_call_expert.cbm"))
        x3_cb = x3.copy()
        for c in CAT_V2:
            if c in x3_cb.columns:
                x3_cb[c] = x3_cb[c].astype(str)
        call_probs = safe_cb_predict_proba(cb_call, x3_cb)
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
    
    p107 = np.clip(p, 1e-5, 1 - 1e-5)
    
    r_seeds = meta.get("r_seeds", [42, 1, 2, 3, 4])
    f_seeds = meta.get("f_seeds", [42, 1, 2, 3])
    
    # 108C & 109C Prediction
    if regular.any():
        X_61_reg, _ = build_v5_deep_61_features(test[regular].reset_index(drop=True), profile_path=MODEL / "team_asof_profile.json", prior=0.523766)
        prev1_p_val = test.loc[regular, "asof_pitcher_prev1_game_success_rate"].to_numpy(float) if "asof_pitcher_prev1_game_success_rate" in test.columns else p_rate_raw[regular]
        prev1_p_val = np.where(np.isnan(prev1_p_val), p_rate_raw[regular], prev1_p_val)
        b_rate_val = test.loc[regular, "asof_batter_success_rate"].fillna(0.523766).to_numpy(float) if "asof_batter_success_rate" in test.columns else np.full(np.sum(regular), 0.523766)
        X_61_reg["form_gap"] = prev1_p_val - b_rate_val
        X_61_reg["anchor_p"] = p107[regular]
        
        phys_108_reg = extract_advanced_physics_108c(test[regular].reset_index(drop=True))
        X_108_reg = pd.concat([X_61_reg, phys_108_reg], axis=1)
        cb_p_108c = [safe_cb_predict(CatBoostRegressor().load_model(str(MODEL / f"cb108_seed{s}.cbm")), X_108_reg) for s in r_seeds]
        lgb_p_108c = [safe_lgb_predict(lgb.Booster(model_file=str(MODEL / f"lgb108_seed{s}.txt")), X_108_reg) for s in r_seeds]
        res_108c_reg = np.mean(cb_p_108c + lgb_p_108c, axis=0)
        
        tensor_109_reg = extract_hyper_regime_tensor_109c(test[regular].reset_index(drop=True))
        X_109_reg = pd.concat([X_61_reg, tensor_109_reg], axis=1)
        cb_p_109c = [safe_cb_predict(CatBoostRegressor().load_model(str(MODEL / f"super_resid_cb_seed{s}.cbm")), X_109_reg) for s in r_seeds]
        lgb_p_109c = [safe_lgb_predict(lgb.Booster(model_file=str(MODEL / f"super_resid_lgb_seed{s}.txt")), X_109_reg) for s in r_seeds]
        xgb_p_109c = []
        for s in r_seeds:
            m = XGBRegressor()
            m.load_model(str(MODEL / f"super_resid_xgb_seed{s}.json"))
            xgb_p_109c.append(safe_xgb_predict(m, X_109_reg))
        res_109c_reg = np.mean(cb_p_109c + lgb_p_109c + xgb_p_109c, axis=0)
    else:
        res_108c_reg = np.zeros(0)
        res_109c_reg = np.zeros(0)
        
    if futures.any():
        X_61_fut, _ = build_v5_deep_61_features(test[futures].reset_index(drop=True), profile_path=MODEL / "team_asof_profile.json", prior=0.523766)
        X_61_fut["anchor_p"] = p107[futures]
        
        phys_108_fut = extract_advanced_physics_108c(test[futures].reset_index(drop=True))
        X_108_fut = pd.concat([X_61_fut, phys_108_fut], axis=1)
        res_108c_fut = np.mean([safe_cb_predict(CatBoostRegressor().load_model(str(MODEL / f"fut108_seed{s}.cbm")), X_108_fut) for s in f_seeds], axis=0)
        
        tensor_109_fut = extract_hyper_regime_tensor_109c(test[futures].reset_index(drop=True))
        X_109_fut = pd.concat([X_61_fut, tensor_109_fut], axis=1)
        res_109c_fut = np.mean([safe_cb_predict(CatBoostRegressor().load_model(str(MODEL / f"fut_resid_cb_seed{s}.cbm")), X_109_fut) for s in f_seeds], axis=0)
    else:
        res_108c_fut = np.zeros(0)
        res_109c_fut = np.zeros(0)
    
    p108 = p107.copy()
    if regular.any():
        p108[regular] += 0.08 * res_108c_reg
    if futures.any():
        p108[futures] += 0.04 * res_108c_fut
    p108 = np.clip(p108, 1e-5, 1 - 1e-5)
    
    p109 = p107.copy()
    if regular.any():
        p109[regular] += 0.085 * res_109c_reg
    if futures.any():
        p109[futures] += 0.035 * res_109c_fut
    p109 = np.clip(p109, 1e-5, 1 - 1e-5)
    
    # 110A MoE Residual Router
    X_router = build_router_features(test, p107, p108, p109)
    router_preds = []
    for s in meta.get("router_seeds", [42, 1, 2]):
        m_r = CatBoostRegressor().load_model(str(MODEL / f"moe_router_cb_seed{s}.cbm"))
        router_preds.append(safe_cb_predict(m_r, X_router))
    router_delta = np.mean(router_preds, axis=0)
    
    p_final = np.clip(p109 + router_delta, 1e-5, 1 - 1e-5)
    
    if len(p_final) != len(test) or not np.isfinite(p_final).all():
        raise RuntimeError("invalid predictions")
        
    out = ROOT / "output"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"row_id": row_id, "control_success": p_final}).to_csv(
        out / "submission.csv", index=False
    )
    print(f"predicted {len(p_final):,} rows in {time.time() - started:.1f}s")

if __name__ == "__main__":
    main()
"""
    (PACKAGE_DIR / "script.py").write_text(script_content, encoding="utf-8")
    
    # Copy requirements.txt
    shutil.copy(SRC_109C / "requirements.txt", PACKAGE_DIR / "requirements.txt")
    
    manifest_110a = json.loads((SRC_109C / "model/manifest.json").read_text(encoding="utf-8"))
    manifest_110a["version"] = "REF4-TEMPORAL-CROSSFIT-MOE-110A"
    manifest_110a["description"] = "REF4-TEMPORAL-CROSSFIT-MOE-110A: Temporal Cross-Fitted MoE Residual Router over 107A/108C/109C"
    manifest_110a["router_seeds"] = [42, 1, 2]
    manifest_110a["router_depth"] = 4
    manifest_110a["built_at"] = "2026-08-30T19:15:00+09:00"
    (MODEL_DIR / "manifest.json").write_text(json.dumps(manifest_110a, indent=2), encoding="utf-8")
    
    print("\n=== Step 4: Packaging submit_ref4_super_ensemble_110A.zip ===")
    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zipf:
        for file in PACKAGE_DIR.rglob("*"):
            if file.is_file():
                arcname = file.relative_to(PACKAGE_DIR)
                zipf.write(file, arcname)
                
    zip_size = OUTPUT_ZIP.stat().st_size
    sha256_hash = compute_sha256(OUTPUT_ZIP)
    print(f"\nCreated ZIP: {OUTPUT_ZIP.name}")
    print(f"Size: {zip_size:,} bytes ({zip_size / (1024*1024):.2f} MB)")
    print(f"SHA-256: {sha256_hash}")
    print(f"Build completed in {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
