#!/usr/bin/env python3
"""Build REF4-MACRO-LEAP-CHAMPION-075A with Decoupled 1군 Gate & 2군 4052-BSS F-Regime Engine."""
import gc, hashlib, json, os, shutil, sys, time, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / 'model/REF4-MACRO-LEAP-CHAMPION-075A'
PROD_DIR = EXP_DIR / 'production_package'
MODEL_DIR = PROD_DIR / 'model'
SRC_DIR = PROD_DIR / 'src'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
SRC_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_071_MODEL = ROOT / 'model/REF4-ADAPTIVE-CHANNEL-OPT-071A/production_package/model'
SOURCE_071_SRC = ROOT / 'model/REF4-ADAPTIVE-CHANNEL-OPT-071A/production_package/src'

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    t0 = time.time()
    print("=== Step 1: Copying Assets and Sources ===")
    for f in SOURCE_071_MODEL.iterdir():
        if f.is_file():
            shutil.copy2(f, MODEL_DIR / f.name)
            
    for f in SOURCE_071_SRC.iterdir():
        if f.is_file():
            shutil.copy2(f, SRC_DIR / f.name)
            
    manifest = json.loads((MODEL_DIR / "manifest.json").read_text(encoding="utf-8"))
    manifest["adaptive_gate"] = True
    manifest["gate_scale"] = 0.05
    manifest["gate_bias_offset"] = 0.008486984068236374
    manifest["alpha_shrinkage"] = 1.0  # Preserving full 1092 amplitude (no harmful shrinkage)
    manifest["r_expert_lgbm"] = True
    manifest["r_expert_lgbm_weight"] = 0.02
    manifest["r_split_table"] = True
    manifest["f_psych_latent"] = True
    manifest["f_regime_decoupled"] = True
    manifest["global_shift"] = 0.0052
    (MODEL_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    
    # Write requirements.txt
    req_content = "numpy>=1.24.0\npandas>=2.0.0\ncatboost>=1.2.0\nlightgbm>=4.0.0\n"
    (PROD_DIR / "requirements.txt").write_text(req_content)
    
    print("=== Step 2: Writing Script with Decoupled 1군 Gate & 2군 F-Regime ===")
    script_content = '''"""Offline inference entry point for REF4-MACRO-LEAP-CHAMPION-075A."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from src.preprocessing_v2 import build_v2_features, build_v3_features, CAT_V2
from src.adaptive_gate import build_gate_features

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "model"


RAW_COLS = [
    'row_id', 'season', 'game_month', 'game_dayofweek', 'inning', 'top_bottom', 'game_type',
    'balls_before', 'strikes_before', 'outs_before', 'run_top_before', 'run_bot_before', 'run_total_before',
    'score_diff_home', 'score_diff_pitcher_team', 'runner_on_1b', 'runner_on_2b', 'runner_on_3b',
    'num_runners_on', 'base_state', 'home_win_expectancy', 'away_win_expectancy', 'li',
    'pitcher_id', 'batter_id', 'pitcher_hand', 'batter_hand', 'pitcher_team_id', 'batter_team_id',
    'asof_pitcher_n', 'asof_pitcher_success_rate', 'asof_pitcher_reverse_rate', 'asof_pitcher_middle_rate',
    'asof_pitcher_ball_rate', 'asof_pitcher_strike_rate', 'asof_pitcher_prev1_game_success_rate',
    'asof_pitcher_prev3_game_success_rate', 'asof_pitcher_prev5_game_success_rate',
    'asof_pitcher_prev1_game_middle_rate', 'asof_pitcher_prev3_game_middle_rate',
    'asof_pitcher_prev5_game_middle_rate', 'asof_batter_n', 'asof_batter_success_rate',
    'asof_batter_middle_rate', 'asof_pitcher_pitchmix_n', 'asof_pitcher_fastball_rate',
    'asof_pitcher_breaking_rate', 'asof_pitcher_offspeed_rate'
]

def main():
    started = time.time()
    meta = json.loads((MODEL / "manifest.json").read_text(encoding="utf-8"))
    test = pd.read_csv(ROOT / "data/test.csv", low_memory=False)
    row_id = test["row_id"].copy()
    
    # Enforce standard column ordering in case test.csv columns are permuted
    ordered_cols = [c for c in RAW_COLS if c in test.columns] + [c for c in test.columns if c not in RAW_COLS]
    test = test[ordered_cols]
    
    ps = pd.read_pickle(MODEL / "pitcher_snapshots.pkl")
    bs = pd.read_pickle(MODEL / "batter_snapshots.pkl")
    ms = pd.read_pickle(MODEL / "pitchmix_snapshots.pkl")
    tm = str(MODEL / "trackman_prior_features.csv")
    
    x2, base2 = build_v2_features(test, meta["prior"], ps, tm)
    x3, base3 = build_v3_features(test, meta["prior"], ps, bs, ms, tm)

    seeds = meta.get("seeds", [260802, 260803, 260804, 260805, 260806, 260807])

    def predict_cat_reg(m, df):
        cols = m.feature_names_ if hasattr(m, 'feature_names_') and all(f in df.columns for f in m.feature_names_) else df.columns
        return m.predict(df[cols])

    def predict_cat_clf(m, df):
        cols = m.feature_names_ if hasattr(m, 'feature_names_') and all(f in df.columns for f in m.feature_names_) else df.columns
        return m.predict_proba(df[cols])[:, 1]

    def load_reg(stem):
        return [CatBoostRegressor().load_model(str(MODEL / f"{stem}_seed{s}.cbm")) for s in seeds]

    predictions = []
    for stem, x, base in [
        ("v2_decay55", x2, base2),
        ("v3_decay55", x3, base3),
        ("v3_decay30", x3, base3),
    ]:
        member = [np.clip(base + predict_cat_reg(m, x), 1e-6, 1 - 1e-6) for m in load_reg(stem)]
        predictions.append(np.mean(member, axis=0))

    futures = test["game_type"].eq("F").to_numpy()
    regular = ~futures
    
    # ----------------------------------------------------
    # 1군 Regular 파이프라인 (071A 1092점 베이스 100% 보존)
    # ----------------------------------------------------
    risks_regular = []
    for name in ("middle", "wild", "reverse"):
        stems = [f"subtype_{name}_seed{s}.cbm" for s in seeds]
        member = []
        for filename in stems:
            m = CatBoostClassifier()
            m.load_model(str(MODEL / filename))
            member.append(predict_cat_clf(m, x3))
        risks_regular.append(np.mean(member, axis=0))

    main_p_reg = np.average(np.vstack(predictions), axis=0, weights=meta["main_weights"])
    z_reg = np.column_stack([main_p_reg] + risks_regular)
    p_reg = meta["stack_intercept"] + z_reg @ np.asarray(meta["stack_coefficients"])

    # 1군 Zero-Centered Adaptive Gate (scale=0.05)
    if meta.get("adaptive_gate", False):
        gate_x = build_gate_features(test, predictions, risks_regular, np.clip(p_reg, 1e-6, 1 - 1e-6))
        gate = CatBoostRegressor()
        gate.load_model(str(MODEL / "adaptive_gate.cbm"))
        gate_pred = predict_cat_reg(gate, gate_x)
        bias_offset = float(meta.get("gate_bias_offset", 0.008486984068236374))
        gate_clean = gate_pred - bias_offset
        p_reg = p_reg + float(meta.get("gate_scale", 0.05)) * gate_clean

    p_reg = p_reg + float(meta.get("global_shift", 0.0052))

    if regular.any():
        from src.entity_context_split import apply_split_profile, apply_linear_split
        split_profile = pd.read_csv(
            MODEL / "split_profile.csv", dtype={"entity_value": str, "context_value": str}
        )
        split_x = apply_split_profile(test, split_profile)
        split_correction = apply_linear_split(split_x, MODEL / "split_residual_meta.npz")
        p_split = p_reg + split_correction
        
        # LightGBM R-Expert (w=0.02)
        lgbm_model = lgb.Booster(model_file=str(MODEL / 'r_expert_lgbm.txt'))
        x3_lgb = x3.copy()
        for col in CAT_V2:
            if col in x3_lgb.columns:
                x3_lgb[col] = x3_lgb[col].astype('category')
        res_lgbm = lgbm_model.predict(x3_lgb)
        p_lgbm = np.clip(base3 + res_lgbm + 0.0052, 1e-6, 1.0 - 1e-6)
        
        w_lgb = float(meta.get("r_expert_lgbm_weight", 0.02))
        p_reg_final = (1.0 - w_lgb) * p_split + w_lgb * p_lgbm
    else:
        p_reg_final = p_reg

    # ----------------------------------------------------
    # 2군 Futures 파이프라인 (16-Model BSS 4052.66 F-Regime 엔진)
    # ----------------------------------------------------
    if futures.any():
        regime = json.loads((MODEL / "f_regime_meta.json").read_text())
        def f_reg_mean(stem, count, x, base):
            member = []
            for j in range(count):
                m = CatBoostRegressor()
                m.load_model(str(MODEL / f"{stem}_{j}.cbm"))
                member.append(np.clip(base + predict_cat_reg(m, x), 1e-6, 1 - 1e-6))
            return np.mean(member, axis=0)

        f_preds = [p.copy() for p in predictions]
        f2 = f_reg_mean("f_v2_all", 4, x2, base2)
        f_preds[0] = f_preds[0] + regime["v2_scale"] * (f2 - f_preds[0])
        f55 = f_reg_mean("f_v355_recent", 6, x3, base3)
        f_preds[1] = f_preds[1] + regime["v355_scale"] * (f55 - f_preds[1])
        f30a = f_reg_mean("f_v330_all", 4, x3, base3)
        f30r = f_reg_mean("f_v330_recent", 2, x3, base3)
        recent_inner = f_preds[2] + regime["v330_recent_inner_scale"] * (f30r - f_preds[2])
        f30 = regime["v330_all_weight"] * f30a + (1 - regime["v330_all_weight"]) * recent_inner
        f_preds[2] = f_preds[2] + regime["v330_scale"] * (f30 - f_preds[2])

        f_risks = [r.copy() for r in risks_regular]
        for idx, name in enumerate(("middle", "wild", "reverse")):
            fm = CatBoostClassifier()
            fm.load_model(str(MODEL / f"f_subtype_{name}.cbm"))
            fr = predict_cat_clf(fm, x3)
            f_risks[idx] = f_risks[idx] + regime["subtype_scale"] * (fr - f_risks[idx])

        f_main = np.average(np.vstack(f_preds), axis=0, weights=meta["main_weights"])
        f_z = np.column_stack([f_main] + f_risks)
        p_f_base = meta["stack_intercept"] + f_z @ np.asarray(meta["stack_coefficients"])

        from src.psych_latent import build_production_features, apply_linear_residual
        residual_x = build_production_features(
            test, MODEL / "psych_profile.pkl", MODEL / "latent_pitch_context.csv"
        )
        correction = apply_linear_residual(residual_x, MODEL / "psych_latent_meta.npz")
        p_futures_final = p_f_base + correction + float(meta.get("global_shift", 0.0052))
    else:
        p_futures_final = p_reg_final

    # 1군/2군 분리 결합
    p = np.where(futures, p_futures_final, p_reg_final)
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
    (PROD_DIR / "script.py").write_text(script_content)
    
    print("=== Step 3: Packaging submit_ref4_macro_leap_075.zip ===")
    zip_path = ROOT / 'output/submit_ref4_macro_leap_075.zip'
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
    
    report = {
        'experiment_id': 'REF4-MACRO-LEAP-CHAMPION-075A',
        'status': 'PACKAGED_AND_READY',
        'zip_filename': zip_path.name,
        'zip_size_bytes': zip_size,
        'sha256': zip_hash,
        'gate_scale': 0.05,
        'f_regime_decoupled': True,
        'has_requirements_txt': True,
        'elapsed_seconds': time.time() - t0
    }
    (EXP_DIR / 'production_package_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f"Completed in {time.time() - t0:.2f}s")

if __name__ == '__main__':
    main()
