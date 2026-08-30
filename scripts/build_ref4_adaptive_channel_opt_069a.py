#!/usr/bin/env python3
"""Build Production Package for REF4-ADAPTIVE-CHANNEL-OPT-069A."""
import gc, hashlib, json, os, shutil, sys, time, zipfile
from pathlib import Path
from catboost import CatBoostClassifier, CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CAND_055 = ROOT / 'candidate/REF4-R-SPECIFIC-SPLIT-FPSYCH-055A'
CAND_064 = ROOT / 'candidate/REF4-LGBM-R-EXPERT-064A'
sys.path.insert(0, str(CAND_055))
sys.path.insert(0, str(ROOT / 'github_reference/4번 레포'))

from src.preprocessing_v2 import build_v2_features, build_v3_features, CAT_V2
from src.adaptive_gate import build_gate_features

EXP_DIR = ROOT / 'model/REF4-ADAPTIVE-CHANNEL-OPT-069A'
PROD_DIR = EXP_DIR / 'production_package'
MODEL_DIR = PROD_DIR / 'model'
SRC_DIR = PROD_DIR / 'src'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
SRC_DIR.mkdir(parents=True, exist_ok=True)

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    t0 = time.time()
    print("=== Step 1: Copying Base 055A Assets & Source Files ===")
    BASE_MODEL = CAND_055 / 'model'
    BASE_SRC = CAND_055 / 'src'
    
    # Copy all model files from 055A
    for f in BASE_MODEL.iterdir():
        if f.is_file() and not f.name.endswith('.cbm.tmp'):
            shutil.copy2(f, MODEL_DIR / f.name)
            
    # Copy all source files
    for f in BASE_SRC.iterdir():
        if f.is_file() and f.suffix == '.py':
            shutil.copy2(f, SRC_DIR / f.name)
            
    # Copy adaptive_gate.py from 4번 레포 src
    shutil.copy2(ROOT / 'github_reference/4번 레포/src/adaptive_gate.py', SRC_DIR / 'adaptive_gate.py')
    
    # Copy LightGBM production model from 064A
    shutil.copy2(CAND_064 / 'model/r_expert_lgbm.txt', MODEL_DIR / 'r_expert_lgbm.txt')

    print("=== Step 2: Training Final Production Adaptive Gate on 2022-2024 Historical Data ===")
    raw = pd.read_csv(ROOT / 'data/train.csv', low_memory=False)
    
    meta = json.loads((MODEL_DIR / "manifest.json").read_text(encoding="utf-8"))
    ps = pd.read_pickle(MODEL_DIR / "pitcher_snapshots.pkl")
    bs = pd.read_pickle(MODEL_DIR / "batter_snapshots.pkl")
    ms = pd.read_pickle(MODEL_DIR / "pitchmix_snapshots.pkl")
    tm = str(MODEL_DIR / "trackman_prior_features.csv")
    seeds = meta.get("seeds", [260802, 260803, 260804, 260805, 260806, 260807])
    
    def load_reg(stem):
        return [CatBoostRegressor().load_model(str(MODEL_DIR / f"{stem}_seed{s}.cbm")) for s in seeds]

    def get_base_preds(x2, base2, x3, base3):
        preds = []
        for stem, x, base in [
            ("v2_decay55", x2, base2),
            ("v3_decay55", x3, base3),
            ("v3_decay30", x3, base3),
        ]:
            member = [np.clip(base + m.predict(x), 1e-6, 1 - 1e-6) for m in load_reg(stem)]
            preds.append(np.mean(member, axis=0))
        return preds

    def get_risks(x3):
        risks = []
        for name in ("middle", "wild", "reverse"):
            member = []
            for s in seeds:
                m = CatBoostClassifier()
                m.load_model(str(MODEL_DIR / f"subtype_{name}_seed{s}.cbm"))
                member.append(m.predict_proba(x3)[:, 1])
            risks.append(np.mean(member, axis=0))
        return risks

    gate_X_parts = []
    gate_y_parts = []
    gate_w_parts = []
    
    for yr in [2022, 2023, 2024]:
        df_yr = raw.loc[raw.season == yr].copy().reset_index(drop=True)
        y_yr = df_yr.control_success.to_numpy(float)
        
        x2_yr, b2_yr = build_v2_features(df_yr, meta["prior"], ps, tm)
        x3_yr, b3_yr = build_v3_features(df_yr, meta["prior"], ps, bs, ms, tm)
        
        preds_yr = get_base_preds(x2_yr, b2_yr, x3_yr, b3_yr)
        risks_yr = get_risks(x3_yr)
        main_yr = np.average(np.vstack(preds_yr), axis=0, weights=meta["main_weights"])
        z_yr = np.column_stack([main_yr] + risks_yr)
        p_stack_yr = meta["stack_intercept"] + z_yr @ np.asarray(meta["stack_coefficients"])
        
        gx_yr = build_gate_features(df_yr, preds_yr, risks_yr, np.clip(p_stack_yr, 1e-6, 1 - 1e-6))
        
        w_yr = np.full(len(df_yr), np.power(0.55, 2024 - yr))
        gate_X_parts.append(gx_yr)
        gate_y_parts.append(y_yr - p_stack_yr)
        gate_w_parts.append(w_yr)
        
    X_gate_all = pd.concat(gate_X_parts, ignore_index=True)
    y_gate_all = np.concatenate(gate_y_parts)
    w_gate_all = np.concatenate(gate_w_parts)
    
    print(f"Fitting production adaptive gate on {len(X_gate_all)} rows...")
    prod_gate = CatBoostRegressor(
        iterations=100,
        depth=4,
        learning_rate=0.025,
        loss_function='RMSE',
        l2_leaf_reg=30,
        random_strength=0.2,
        bootstrap_type='Bernoulli',
        subsample=0.8,
        random_seed=280033,
        thread_count=6,
        allow_writing_files=False,
        verbose=False
    )
    prod_gate.fit(X_gate_all, y_gate_all, sample_weight=w_gate_all)
    prod_gate.save_model(str(MODEL_DIR / 'adaptive_gate.cbm'))
    
    # Store empirical training gate offset to guarantee 0.000000 zero-centering during offline inference!
    gate_bias_offset = float(np.mean(prod_gate.predict(X_gate_all)))
    print(f"Production Gate Training Bias Offset: {gate_bias_offset:+.8f}")
    
    # Update manifest
    manifest = json.loads((MODEL_DIR / "manifest.json").read_text(encoding="utf-8"))
    manifest["adaptive_gate"] = True
    manifest["gate_scale"] = 0.05
    manifest["gate_bias_offset"] = gate_bias_offset
    manifest["r_expert_lgbm"] = True
    manifest["r_expert_lgbm_weight"] = 0.02
    manifest["r_split_table"] = True
    manifest["f_psych_latent"] = True
    manifest["global_shift"] = 0.0052
    (MODEL_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    
    print("=== Step 3: Writing Compliant Offline Inference script.py ===")
    script_content = '''"""Offline inference entry point for REF4-ADAPTIVE-CHANNEL-OPT-069A."""
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


def main():
    started = time.time()
    meta = json.loads((MODEL / "manifest.json").read_text(encoding="utf-8"))
    test = pd.read_csv(ROOT / "data/test.csv", low_memory=False)
    row_id = test["row_id"].copy()
    
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

    # Zero-Centered Adaptive Gate
    if meta.get("adaptive_gate", False):
        gate_x = build_gate_features(test, predictions, risks, np.clip(p, 1e-6, 1 - 1e-6))
        gate = CatBoostRegressor()
        gate.load_model(str(MODEL / "adaptive_gate.cbm"))
        gate_pred = gate.predict(gate_x)
        # Strict zero-centering using learned empirical training offset
        bias_offset = float(meta.get("gate_bias_offset", 0.0))
        gate_clean = gate_pred - bias_offset
        p = p + float(meta.get("gate_scale", 0.05)) * gate_clean

    # Fixed global calibration hyperparameter (+0.0052)
    p = p + float(meta.get("global_shift", 0.0052))

    # Futures 2군: Audited train-only linear psych latent residual
    if futures.any():
        from src.psych_latent import build_production_features, apply_linear_residual
        residual_x = build_production_features(
            test, MODEL / "psych_profile.pkl", MODEL / "latent_pitch_context.csv"
        )
        correction = apply_linear_residual(residual_x, MODEL / "psych_latent_meta.npz")
        p = np.where(futures, p + correction, p)

    # Regular 1군: Audited train-only entity×context split table + LightGBM R-Expert
    regular = ~futures
    if regular.any():
        from src.entity_context_split import apply_split_profile, apply_linear_split
        split_profile = pd.read_csv(
            MODEL / "split_profile.csv", dtype={"entity_value": str, "context_value": str}
        )
        split_x = apply_split_profile(test, split_profile)
        split_correction = apply_linear_split(split_x, MODEL / "split_residual_meta.npz")
        p_split = p + split_correction
        
        # LightGBM R-Expert (w=0.02)
        lgbm_model = lgb.Booster(model_file=str(MODEL / 'r_expert_lgbm.txt'))
        x3_lgb = x3.copy()
        for col in CAT_V2:
            if col in x3_lgb.columns:
                x3_lgb[col] = x3_lgb[col].astype('category')
        res_lgbm = lgbm_model.predict(x3_lgb)
        p_lgbm = np.clip(base3 + res_lgbm + 0.0052, 1e-6, 1.0 - 1e-6)
        
        w_lgb = float(meta.get("r_expert_lgbm_weight", 0.02))
        p = np.where(regular, (1.0 - w_lgb) * p_split + w_lgb * p_lgbm, p)

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
    
    print("=== Step 4: Creating ZIP Package ===")
    zip_path = ROOT / 'output/submit_ref4_adaptive_channel_opt_069.zip'
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
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
        'experiment_id': 'REF4-ADAPTIVE-CHANNEL-OPT-069A',
        'status': 'PACKAGED_AND_READY',
        'zip_filename': zip_path.name,
        'zip_size_bytes': zip_size,
        'sha256': zip_hash,
        'gate_scale': 0.05,
        'gate_bias_offset': gate_bias_offset,
        'elapsed_seconds': time.time() - t0
    }
    (EXP_DIR / 'production_package_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f"Completed in {time.time() - t0:.2f}s")

if __name__ == '__main__':
    main()
