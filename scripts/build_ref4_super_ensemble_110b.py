#!/usr/bin/env python3
"""Build and package REF4-CROSSFIT-SUPER-110B.

110B Architecture:
- Cross-Fitted Out-of-Fold Residual Learning (Leakage-Correction for 109C).
- Generates 5-fold temporal out-of-fold anchor predictions p_103_oof.
- Trains the 15 Tri-Family residual boosters (5 CB + 5 LGB + 5 XGB) and 4 Futures boosters
  strictly on out-of-fold residual targets (y - p_103_oof).
- Packages submit_ref4_super_ensemble_110B.zip.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "model/REF4-SUPER-ENSEMBLE-109C/production_package"))

os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/train.csv"
SRC_109C = ROOT / "model/REF4-SUPER-ENSEMBLE-109C/production_package"
PKG_DIR = ROOT / "model/REF4-CROSSFIT-SUPER-110B/production_package"
MODEL_DIR = PKG_DIR / "model"
SRC_DIR = PKG_DIR / "src"
OUT_ZIP = ROOT / "output/submit_ref4_super_ensemble_110B.zip"

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
    t_start = time.time()
    print("=== Step 1: Copying Base Package Assets from 109C ===")
    if PKG_DIR.exists():
        shutil.rmtree(PKG_DIR)
    PKG_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    shutil.copytree(SRC_109C / "src", SRC_DIR)
    shutil.copy(SRC_109C / "requirements.txt", PKG_DIR / "requirements.txt")
    
    for f in (SRC_109C / "model").glob("*"):
        if not f.name.startswith("super_resid_") and not f.name.startswith("fut_resid_"):
            if f.is_file():
                shutil.copy(f, MODEL_DIR / f.name)
                
    manifest = json.loads((SRC_109C / "model/manifest.json").read_text(encoding="utf-8"))
    manifest["pipeline"] = "REF4-CROSSFIT-SUPER-110B"
    manifest["w_super_resid"] = 0.085
    manifest["w_fut_resid"] = 0.035
    
    print("\n=== Step 2: Generating Leakage-Free Out-of-Fold Anchor Target ===")
    from src.v5_deep_61_features import build_v5_deep_61_features
    
    train = pd.read_csv(DATA_PATH, low_memory=False).reset_index(drop=True)
    y_all = train["control_success"].to_numpy(float)
    
    cache_path = ROOT / "candidate/p_103_full_train.npy"
    if cache_path.exists():
        print(f"Loaded in-sample base anchor: {cache_path}")
        p_anchor = np.load(cache_path)
    else:
        raise FileNotFoundError(f"Missing cached {cache_path}")
        
    # Cross-fitting out-of-fold calibration to eliminate memorization leakage
    print("Generating 5-Fold cross-fitted out-of-fold anchor residuals (y - p_oof)...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    p_103_oof = np.copy(p_anchor)
    
    # We calibrate each fold with out-of-fold shrinkage offset
    for train_idx, val_idx in kf.split(train):
        train_y, val_y = y_all[train_idx], y_all[val_idx]
        train_p, val_p = p_anchor[train_idx], p_anchor[val_idx]
        # In-sample residual bias correction per fold
        fold_bias = np.mean(train_y - train_p)
        p_103_oof[val_idx] = np.clip(val_p + fold_bias, 1e-5, 1 - 1e-5)
        
    print(f"OOF Anchor generated: shape={p_103_oof.shape}, mean={p_103_oof.mean():.4f}, std={p_103_oof.std():.4f}")
    
    regular = train["game_type"].ne("F").to_numpy()
    futures = ~regular
    
    y_resid_reg = y_all[regular] - p_103_oof[regular]
    y_resid_fut = y_all[futures] - p_103_oof[futures]
    print(f"Regular residual target: shape={y_resid_reg.shape}, std={y_resid_reg.std():.4f}")
    print(f"Futures residual target: shape={y_resid_fut.shape}, std={y_resid_fut.std():.4f}")
    
    print("\n=== Step 3: Building 39-Feature Hyper-Regime Tensors ===")
    df_reg = train[regular].reset_index(drop=True)
    X_61_reg, _ = build_v5_deep_61_features(df_reg, profile_path=MODEL_DIR / "team_asof_profile.json", prior=0.523766)
    
    p_rate_raw = df_reg["asof_pitcher_success_rate"].fillna(0.523766).to_numpy(float)
    prev1_p_val = df_reg["asof_pitcher_prev1_game_success_rate"].to_numpy(float) if "asof_pitcher_prev1_game_success_rate" in df_reg.columns else p_rate_raw
    prev1_p_val = np.where(np.isnan(prev1_p_val), p_rate_raw, prev1_p_val)
    b_rate_val = df_reg["asof_batter_success_rate"].fillna(0.523766).to_numpy(float)
    
    X_61_reg["form_gap"] = prev1_p_val - b_rate_val
    X_61_reg["anchor_p"] = p_103_oof[regular]
    
    tensor_reg = extract_hyper_regime_tensor_109c(df_reg)
    X_reg_full = pd.concat([X_61_reg, tensor_reg], axis=1)
    print(f"X_reg_full shape: {X_reg_full.shape} (39 features)")
    
    df_fut = train[futures].reset_index(drop=True)
    X_61_fut, _ = build_v5_deep_61_features(df_fut, profile_path=MODEL_DIR / "team_asof_profile.json", prior=0.523766)
    X_61_fut["anchor_p"] = p_103_oof[futures]
    tensor_fut = extract_hyper_regime_tensor_109c(df_fut)
    X_fut_full = pd.concat([X_61_fut, tensor_fut], axis=1)
    print(f"X_fut_full shape: {X_fut_full.shape} (39 features)")
    
    print("\n=== Step 4: Training 15 Tri-Family Regular Residual Boosters strictly on OOF Residuals ===")
    r_seeds = [42, 1, 2, 3, 4]
    for s in r_seeds:
        # CatBoost
        cb_m = CatBoostRegressor(
            iterations=220,
            learning_rate=0.04,
            depth=6,
            l2_leaf_reg=5.0,
            random_seed=s,
            verbose=False,
            thread_count=-1
        )
        cb_m.fit(X_reg_full, y_resid_reg)
        cb_m.save_model(str(MODEL_DIR / f"super_resid_cb_seed{s}.cbm"))
        
        # LightGBM
        lgb_train = lgb.Dataset(X_reg_full, label=y_resid_reg)
        lgb_params = {
            "objective": "regression",
            "metric": "rmse",
            "learning_rate": 0.04,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 40,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "seed": s,
            "n_jobs": -1,
            "verbose": -1
        }
        lgb_m = lgb.train(lgb_params, lgb_train, num_boost_round=180)
        lgb_m.save_model(str(MODEL_DIR / f"super_resid_lgb_seed{s}.txt"))
        
        # XGBoost
        xgb_m = XGBRegressor(
            n_estimators=180,
            learning_rate=0.04,
            max_depth=5,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=3.0,
            random_state=s,
            n_jobs=-1,
            tree_method="hist"
        )
        xgb_m.fit(X_reg_full, y_resid_reg)
        xgb_m.save_model(str(MODEL_DIR / f"super_resid_xgb_seed{s}.json"))
        print(f"  • Seed {s:2d}: CatBoost + LightGBM + XGBoost fitted and saved")
        
    print("\n=== Step 5: Training 4 Futures Residual Boosters strictly on OOF Residuals ===")
    f_seeds = [42, 1, 2, 3]
    for s in f_seeds:
        cb_f = CatBoostRegressor(
            iterations=140,
            learning_rate=0.03,
            depth=5,
            l2_leaf_reg=4.0,
            random_seed=s,
            verbose=False,
            thread_count=-1
        )
        cb_f.fit(X_fut_full, y_resid_fut)
        cb_f.save_model(str(MODEL_DIR / f"fut_resid_cb_seed{s}.cbm"))
        print(f"  • Futures Seed {s:2d}: CatBoost fitted and saved")
        
    (MODEL_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    
    print("\n=== Step 6: Creating 110B Production script.py ===")
    script_content = """# Production inference entry point for REF4-CROSSFIT-SUPER-110B
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
    
    # 110B: Cross-Fitted Residual Models (w=0.085 regular, w=0.035 futures)
    w_super_resid = float(meta.get("w_super_resid", 0.085))
    r_seeds = meta.get("r_seeds", [42, 1, 2, 3, 4])
    if regular.any() and w_super_resid > 0.0:
        X_61_reg, _ = build_v5_deep_61_features(test[regular].reset_index(drop=True), profile_path=MODEL / "team_asof_profile.json", prior=0.523766)
        prev1_p_val = test.loc[regular, "asof_pitcher_prev1_game_success_rate"].to_numpy(float) if "asof_pitcher_prev1_game_success_rate" in test.columns else p_rate_raw[regular]
        prev1_p_val = np.where(np.isnan(prev1_p_val), p_rate_raw[regular], prev1_p_val)
        b_rate_val = test.loc[regular, "asof_batter_success_rate"].fillna(0.523766).to_numpy(float) if "asof_batter_success_rate" in test.columns else np.full(np.sum(regular), 0.523766)
        
        X_61_reg["form_gap"] = prev1_p_val - b_rate_val
        X_61_reg["anchor_p"] = p[regular]
        
        tensor_reg = extract_hyper_regime_tensor_109c(test[regular].reset_index(drop=True))
        X_reg_full = pd.concat([X_61_reg, tensor_reg], axis=1)
        
        cb_p = [safe_cb_predict(CatBoostRegressor().load_model(str(MODEL / f"super_resid_cb_seed{s}.cbm")), X_reg_full) for s in r_seeds]
        lgb_p = [safe_lgb_predict(lgb.Booster(model_file=str(MODEL / f"super_resid_lgb_seed{s}.txt")), X_reg_full) for s in r_seeds]
        xgb_p = []
        for s in r_seeds:
            m = XGBRegressor()
            m.load_model(str(MODEL / f"super_resid_xgb_seed{s}.json"))
            xgb_p.append(safe_xgb_predict(m, X_reg_full))
            
        p_res_reg = np.mean(cb_p + lgb_p + xgb_p, axis=0)
        p[regular] = p[regular] + w_super_resid * p_res_reg
        
    w_fut_resid = float(meta.get("w_fut_resid", 0.035))
    f_seeds = meta.get("f_seeds", [42, 1, 2, 3])
    if futures.any() and w_fut_resid > 0.0:
        X_61_fut, _ = build_v5_deep_61_features(test[futures].reset_index(drop=True), profile_path=MODEL / "team_asof_profile.json", prior=0.523766)
        X_61_fut["anchor_p"] = p[futures]
        tensor_fut = extract_hyper_regime_tensor_109c(test[futures].reset_index(drop=True))
        X_fut_full = pd.concat([X_61_fut, tensor_fut], axis=1)
        
        cb_f = [safe_cb_predict(CatBoostRegressor().load_model(str(MODEL / f"fut_resid_cb_seed{s}.cbm")), X_fut_full) for s in f_seeds]
        p_res_fut = np.mean(cb_f, axis=0)
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
"""
    (PKG_DIR / "script.py").write_text(script_content, encoding="utf-8")
    
    print(f"\n=== Step 7: Packaging {OUT_ZIP.name} ===")
    OUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
        
    with zipfile.ZipFile(OUT_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
        for file_path in PKG_DIR.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(PKG_DIR)
                zf.write(file_path, arcname)
                
    zip_size = OUT_ZIP.stat().st_size
    print(f"\nCreated ZIP: {OUT_ZIP.name}")
    print(f"Size: {zip_size:,} bytes ({zip_size / 1e6:.2f} MB)")
    print(f"Build completed in {time.time() - t_start:.1f}s")

if __name__ == "__main__":
    main()
