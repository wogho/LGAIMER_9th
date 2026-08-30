#!/usr/bin/env python3
"""Build Production Package 085A with 20 Pure L2 Multi-Seed Deep Hierarchical 67-Feature Residual Engine & Pocket Downshift Post-Processor."""
import gc, hashlib, json, os, shutil, sys, time, zipfile
from pathlib import Path
import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / 'model/REF4-DEEP-HIERARCHICAL-085A'
PROD_DIR = EXP_DIR / 'production_package'
MODEL_DIR = PROD_DIR / 'model'
SRC_DIR = PROD_DIR / 'src'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
SRC_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_082_MODEL = ROOT / 'model/REF4-DEEP-HIERARCHICAL-082A/production_package/model'
SOURCE_082_SRC = ROOT / 'model/REF4-DEEP-HIERARCHICAL-082A/production_package/src'
SOURCE_082_REQS = ROOT / 'model/REF4-DEEP-HIERARCHICAL-082A/production_package/requirements.txt'

sys.path.insert(0, str(ROOT / 'src'))
from v52_deep_67_features import build_v52_deep_67_features

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

SCRIPT_CONTENT_085 = '''"""Offline inference entry point for REF4-DEEP-HIERARCHICAL-085A."""
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
from src.v52_deep_67_features import build_v52_deep_67_features

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

    # Adaptive Gate (gate_scale = 0.08)
    if meta.get("adaptive_gate", False):
        gate_x = build_gate_features(test, predictions, risks, np.clip(p, 1e-6, 1 - 1e-6))
        gate = CatBoostRegressor()
        gate.load_model(str(MODEL / "adaptive_gate.cbm"))
        gate_pred = gate.predict(gate_x)
        bias_offset = float(meta.get("gate_bias_offset", 0.0))
        gate_clean = gate_pred - bias_offset
        p = p + float(meta.get("gate_scale", 0.08)) * gate_clean

    # Fixed global calibration hyperparameter (+0.0052)
    p = p + float(meta.get("global_shift", 0.0052))

    # Futures 2군 psych latent
    if futures.any():
        from src.psych_latent import build_production_features, apply_linear_residual
        residual_x = build_production_features(
            test, MODEL / "psych_profile.pkl", MODEL / "latent_pitch_context.csv"
        )
        correction = apply_linear_residual(residual_x, MODEL / "psych_latent_meta.npz")
        p = np.where(futures, p + correction, p)

    # Regular 1군 split profile + LightGBM R-Expert
    if regular.any():
        from src.entity_context_split import apply_split_profile, apply_linear_split
        split_profile = pd.read_csv(
            MODEL / "split_profile.csv", dtype={"entity_value": str, "context_value": str}
        )
        split_x = apply_split_profile(test, split_profile)
        split_correction = apply_linear_split(split_x, MODEL / "split_residual_meta.npz")
        p_split = p + split_correction
        
        # LightGBM R-Expert (w=0.05)
        lgbm_model = lgb.Booster(model_file=str(MODEL / 'r_expert_lgbm.txt'))
        x3_lgb = x3.copy()
        for col in CAT_V2:
            if col in x3_lgb.columns:
                x3_lgb[col] = x3_lgb[col].astype('category')
        res_lgbm = lgbm_model.predict(x3_lgb)
        p_lgbm = np.clip(base3 + res_lgbm + 0.0052, 1e-6, 1.0 - 1e-6)
        
        w_lgb = float(meta.get("r_expert_lgbm_weight", 0.05))
        p = np.where(regular, (1.0 - w_lgb) * p_split + w_lgb * p_lgbm, p)

    # NEW 085A: 20 Pure L2 Multi-Seed Deep Hierarchical Residual Model + Form Gap (67 features) strictly on Regular season
    w_deep = float(meta.get("w_deep_hierarchical", 0.08))
    v52_seeds = meta.get("v52_seeds", [42, 1, 2, 3, 4, 10, 20, 30, 40, 50])
    if regular.any() and w_deep > 0.0:
        v52_feat, p_hb = build_v52_deep_67_features(test, profile_path=MODEL / "team_asof_profile.json", prior=float(meta.get("prior", 0.523766)))
        
        res_preds = []
        for s in v52_seeds:
            cb_m = CatBoostRegressor()
            cb_m.load_model(str(MODEL / f"deep_cb_l2_seed{s}.cbm"))
            res_preds.append(cb_m.predict(v52_feat))
            
            lgb_m = lgb.Booster(model_file=str(MODEL / f"deep_lgb_l2_seed{s}.txt"))
            res_preds.append(lgb_m.predict(v52_feat))
            
        p_deep_resid = np.mean(res_preds, axis=0)
        p_deep_reconstructed = np.clip(p_hb + p_deep_resid, 1e-5, 1 - 1e-5)
        deep_mean_offset = float(meta.get("deep_mean_offset", 0.514025))
        deep_corr = p_deep_reconstructed - deep_mean_offset
        p = np.where(regular, p + w_deep * deep_corr, p)

    # NEW 085A: Brier-Optimized Pocket Downshift Post-Processor (Defends overconfident low-sample pockets)
    n_p_raw = test["asof_pitcher_n"].fillna(0).to_numpy() if "asof_pitcher_n" in test.columns else np.zeros(len(test))
    p_rate_raw = test["asof_pitcher_success_rate"].fillna(0.523766).to_numpy() if "asof_pitcher_success_rate" in test.columns else np.full(len(test), 0.523766)
    
    pocket_mask = regular & (n_p_raw < 15) & (p_rate_raw > 0.65)
    downshift_val = float(meta.get("pocket_downshift_val", -0.012))
    p = np.where(pocket_mask, p + downshift_val, p)

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
    print("=== Step 1: Training 20 Pure L2 Multi-Seed Deep Hierarchical 67-Feature Model with Form Gap ===")
    raw = pd.read_csv(ROOT / 'data/train.csv', low_memory=False)
    train_reg = raw.loc[raw.game_type != "F"].copy().reset_index(drop=True)
    profile_path = ROOT / 'model/team_asof_profile.json'
    
    v52_train_feat, p_hb_tr = build_v52_deep_67_features(train_reg, profile_path=profile_path)
    
    y_tr = train_reg.control_success.to_numpy(float)
    resid_hb_tr = (y_tr - p_hb_tr).astype(np.float32)
    
    v52_seeds = [42, 1, 2, 3, 4, 10, 20, 30, 40, 50]
    tr_res_preds = []
    
    for s in v52_seeds:
        # CatBoost RMSE (Strict L2)
        cb_m = CatBoostRegressor(iterations=260, depth=6, learning_rate=0.04, l2_leaf_reg=5, random_seed=s, verbose=False)
        cb_m.fit(v52_train_feat, resid_hb_tr)
        cb_path = MODEL_DIR / f"deep_cb_l2_seed{s}.cbm"
        cb_m.save_model(str(cb_path))
        tr_res_preds.append(cb_m.predict(v52_train_feat))
        
        # LightGBM L2
        lgb_tr = lgb.Dataset(v52_train_feat, label=resid_hb_tr)
        params = {
            'objective': 'regression',
            'learning_rate': 0.04,
            'max_depth': 6,
            'num_leaves': 31,
            'seed': s,
            'verbosity': -1,
            'n_jobs': -1
        }
        lgb_m = lgb.train(params, lgb_tr, num_boost_round=210)
        lgb_path = MODEL_DIR / f"deep_lgb_l2_seed{s}.txt"
        lgb_m.save_model(str(lgb_path))
        tr_res_preds.append(lgb_m.predict(v52_train_feat))
        
        print(f"  • Seed {s}: Deep Pure L2 CatBoost & LightGBM trained on (y - p_hb) residual with Form Gap.")

    p_tr_deep_resid = np.mean(tr_res_preds, axis=0)
    p_tr_deep_reconstructed = np.clip(p_hb_tr + p_tr_deep_resid, 1e-5, 1 - 1e-5)
    deep_mean_offset = float(np.mean(p_tr_deep_reconstructed))

    print("\n=== Step 2: Copying 082A Assets, team_asof_profile.json, and Sources ===")
    for f in SOURCE_082_MODEL.iterdir():
        if f.is_file() and f.name != 'manifest.json':
            shutil.copy2(f, MODEL_DIR / f.name)
            
    for f in SOURCE_082_SRC.iterdir():
        if f.is_file():
            shutil.copy2(f, SRC_DIR / f.name)
            
    shutil.copy2(ROOT / 'model/team_asof_profile.json', MODEL_DIR / 'team_asof_profile.json')
    shutil.copy2(ROOT / 'src/v52_deep_67_features.py', SRC_DIR / 'v52_deep_67_features.py')
    shutil.copy2(SOURCE_082_REQS, PROD_DIR / 'requirements.txt')
    
    manifest_path = MODEL_DIR / 'manifest.json'
    manifest = json.loads((SOURCE_082_MODEL / 'manifest.json').read_text(encoding='utf-8'))
    manifest['version'] = 'REF4-DEEP-HIERARCHICAL-085A'
    manifest['gate_scale'] = 0.08
    manifest['r_expert_lgbm_weight'] = 0.05
    manifest['w_deep_hierarchical'] = 0.08
    manifest['pocket_downshift_val'] = -0.012
    manifest['deep_mean_offset'] = deep_mean_offset
    manifest['v52_seeds'] = v52_seeds
    manifest['notes'] = "082A Champion Backbone (1104.12 LB) + 20 Pure L2 Multi-Seed Deep Residual Engine (67 feat, w=0.08) + Pocket Downshift (-0.012) strictly on R"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f"Updated manifest.json: w_deep_hierarchical = {manifest['w_deep_hierarchical']}, deep_mean_offset = {deep_mean_offset:.6f}")
    
    print("\n=== Step 3: Writing script.py and Packaging submit_ref4_deep_hierarchical_085.zip ===")
    (PROD_DIR / 'script.py').write_text(SCRIPT_CONTENT_085, encoding='utf-8')
    
    zip_path = ROOT / 'output/submit_ref4_deep_hierarchical_085.zip'
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

if __name__ == '__main__':
    build_package()
