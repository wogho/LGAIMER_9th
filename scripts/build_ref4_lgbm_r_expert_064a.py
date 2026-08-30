#!/usr/bin/env python3
"""Build and package REF4-LGBM-R-EXPERT-064A production candidate."""
import gc, hashlib, json, os, shutil, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
SRC_055 = ROOT / 'candidate/REF4-R-SPECIFIC-SPLIT-FPSYCH-055A'
CAND_064 = ROOT / 'candidate/REF4-LGBM-R-EXPERT-064A'
OUT_ZIP = ROOT / 'output/submit_ref4_lgbm_r_expert_064.zip'

sys.path.insert(0, str(SRC_055))
from src.preprocessing_v2 import CAT_V2, build_v3_features

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    t0 = time.time()
    print("--- Step 1: Fitting Production LightGBM R-Expert on all train R-rows (2019-2024) ---")
    raw = pd.read_csv(ROOT / 'data/train.csv', low_memory=False)
    meta = json.loads((SRC_055 / 'model/manifest.json').read_text(encoding='utf-8'))
    
    ps = pd.read_pickle(SRC_055 / 'model/pitcher_snapshots.pkl')
    bs = pd.read_pickle(SRC_055 / 'model/batter_snapshots.pkl')
    ms = pd.read_pickle(SRC_055 / 'model/pitchmix_snapshots.pkl')
    tm_path = str(SRC_055 / 'model/trackman_prior_features.csv')
    
    print(f"Building v3 features on {len(raw)} train rows...", flush=True)
    X_train_v3, base_train = build_v3_features(raw, meta['prior'], ps, bs, ms, tm_path)
    
    for col in CAT_V2:
        if col in X_train_v3.columns:
            X_train_v3[col] = X_train_v3[col].astype('category')
            
    mask_r = (raw.game_type == 'R').to_numpy()
    y_train = raw.control_success.to_numpy(float)
    res_train = y_train - base_train
    
    s_train = raw.season.to_numpy(int)
    decay_weights = np.power(0.55, int(s_train.max()) - s_train)
    
    print(f"Fitting LightGBM on {mask_r.sum()} R-rows...", flush=True)
    model = lgb.LGBMRegressor(
        n_estimators=300,
        num_leaves=31,
        learning_rate=0.035,
        colsample_bytree=0.8,
        subsample=0.85,
        min_child_samples=50,
        reg_alpha=1.0,
        reg_lambda=5.0,
        random_state=260803,
        n_jobs=3,
        verbose=-1
    )
    model.fit(X_train_v3.loc[mask_r], res_train[mask_r], sample_weight=decay_weights[mask_r])
    print("LightGBM production fit completed.")
    
    # Clean up train memory
    del X_train_v3, base_train, raw, ps, bs, ms
    gc.collect()
    
    print("\n--- Step 2: Creating candidate directory structure ---")
    if CAND_064.exists():
        shutil.rmtree(CAND_064)
    CAND_064.mkdir(parents=True)
    (CAND_064 / 'model').mkdir()
    
    # Copy src/
    shutil.copytree(SRC_055 / 'src', CAND_064 / 'src')
    
    # Copy model files from 055A
    for f in (SRC_055 / 'model').iterdir():
        if f.is_file():
            shutil.copy2(f, CAND_064 / 'model' / f.name)
            
    # Save production LightGBM model
    model.booster_.save_model(str(CAND_064 / 'model/r_expert_lgbm.txt'))
    
    # Save meta
    lgbm_meta = {
        'model_file': 'r_expert_lgbm.txt',
        'blend_weight': 0.02,
        'shift_offset': 0.0052,
        'trained_rows': int(mask_r.sum()),
        'algorithm': 'LightGBM Regressor (Leaf-wise)',
        'parent_champion': '055A'
    }
    (CAND_064 / 'model/lgbm_meta.json').write_text(json.dumps(lgbm_meta, ensure_ascii=False, indent=2) + '\n')
    
    # Write requirements.txt
    (CAND_064 / 'requirements.txt').write_text(
        "numpy>=1.24.0\n"
        "pandas>=2.0.0\n"
        "catboost>=1.2.0\n"
        "lightgbm>=4.0.0\n"
    )
    
    # Write script.py
    script_code = '''"""Offline inference entry point for the adaptive hierarchical residual stack with LightGBM R-Expert."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor

from src.preprocessing_v2 import CAT_V2, build_v2_features, build_v3_features
from src.league_transition import transition_features

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "model"


def main():
    started = time.time()
    meta = json.loads((MODEL / "manifest.json").read_text(encoding="utf-8"))
    lgbm_meta = json.loads((MODEL / "lgbm_meta.json").read_text(encoding="utf-8"))
    test = pd.read_csv(ROOT / "data/test.csv", low_memory=False)
    row_id = test["row_id"].copy()
    def load_table(name):
        pkl_path = MODEL / f"{name}.pkl"
        csv_path = MODEL / f"{name}.csv"
        try:
            return pd.read_pickle(pkl_path)
        except Exception:
            return pd.read_csv(csv_path)

    ps = load_table("pitcher_snapshots")
    bs = load_table("batter_snapshots")
    ms = load_table("pitchmix_snapshots")
    tm = str(MODEL / "trackman_prior_features.csv")
    x2, base2 = build_v2_features(test, meta["prior"], ps, tm)
    x3, base3 = build_v3_features(test, meta["prior"], ps, bs, ms, tm)

    seeds = meta.get("seeds")

    def load_reg(stem):
        names = [f"{stem}_seed{s}.cbm" for s in seeds] if seeds else [f"{stem}.cbm"]
        out = []
        for filename in names:
            model = CatBoostRegressor()
            model.load_model(MODEL / filename)
            out.append(model)
        return out

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
        member=[]
        for j in range(count):
            m=CatBoostRegressor();m.load_model(MODEL / f"{stem}_{j}.cbm")
            member.append(np.clip(base+m.predict(x),1e-6,1-1e-6))
        return np.mean(member,axis=0)
    if futures.any():
        f2=f_reg_mean("f_v2_all",4,x2,base2)
        predictions[0]=np.where(futures,predictions[0]+regime["v2_scale"]*(f2-predictions[0]),predictions[0])
        f55=f_reg_mean("f_v355_recent",6,x3,base3)
        predictions[1]=np.where(futures,predictions[1]+regime["v355_scale"]*(f55-predictions[1]),predictions[1])
        f30a=f_reg_mean("f_v330_all",4,x3,base3);f30r=f_reg_mean("f_v330_recent",2,x3,base3)
        recent_inner=predictions[2]+regime["v330_recent_inner_scale"]*(f30r-predictions[2])
        f30=regime["v330_all_weight"]*f30a+(1-regime["v330_all_weight"])*recent_inner
        predictions[2]=np.where(futures,predictions[2]+regime["v330_scale"]*(f30-predictions[2]),predictions[2])

    risks = []
    for name in ("middle", "wild", "reverse"):
        stems = [f"subtype_{name}_seed{s}.cbm" for s in seeds] if seeds else [f"subtype_{name}.cbm"]
        member = []
        for filename in stems:
            model = CatBoostClassifier()
            model.load_model(MODEL / filename)
            member.append(model.predict_proba(x3)[:, 1])
        risk=np.mean(member, axis=0)
        if futures.any():
            fm=CatBoostClassifier();fm.load_model(MODEL / f"f_subtype_{name}.cbm")
            fr=fm.predict_proba(x3)[:,1]
            risk=np.where(futures,risk+regime["subtype_scale"]*(fr-risk),risk)
        risks.append(risk)

    main_p = np.average(np.vstack(predictions), axis=0, weights=meta["main_weights"])
    z = np.column_stack([main_p] + risks)
    p = meta["stack_intercept"] + z @ np.asarray(meta["stack_coefficients"])

    # Fixed global calibration hyperparameter.
    p = p + float(meta.get("global_shift", 0.0))

    transition = CatBoostRegressor()
    transition.load_model(MODEL / "transition_gate.cbm")
    tx = transition_features(test, p, MODEL / "prior_type.pkl")
    p = p + regime["transition_scale"] * transition.predict(tx)

    # Audited train-only residual; the gate reads only the current row game_type.
    if futures.any():
        from src.psych_latent import build_production_features, apply_linear_residual
        residual_x = build_production_features(
            test, MODEL / "psych_profile.pkl", MODEL / "latent_pitch_context.csv"
        )
        correction = apply_linear_residual(residual_x, MODEL / "psych_latent_meta.npz")
        p = np.where(futures, p + correction, p)

    # Audited train-only entity×context residual; current-row R gate only.
    regular = ~futures
    if regular.any():
        from src.entity_context_split import apply_split_profile, apply_linear_split
        split_profile = pd.read_csv(
            MODEL / "split_profile.csv", dtype={"entity_value": str, "context_value": str}
        )
        split_x = apply_split_profile(test, split_profile)
        split_correction = apply_linear_split(split_x, MODEL / "split_residual_meta.npz")
        p_055 = p + split_correction
        
        # 064A LightGBM R-Expert blend (w=0.02)
        x3_lgb = x3.copy()
        for col in CAT_V2:
            if col in x3_lgb.columns:
                x3_lgb[col] = x3_lgb[col].astype('category')
        lgb_booster = lgb.Booster(model_file=str(MODEL / lgbm_meta["model_file"]))
        r_expert = np.clip(base3 + lgb_booster.predict(x3_lgb) + lgbm_meta["shift_offset"], 1e-6, 1.0 - 1e-6)
        
        w_lgb = float(lgbm_meta["blend_weight"])
        p = np.where(regular, (1.0 - w_lgb) * p_055 + w_lgb * r_expert, p)

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
    (CAND_064 / 'script.py').write_text(script_code)
    print("Candidate bundle created at", CAND_064)
    print(f"Total time elapsed: {time.time() - t0:.2f}s")

if __name__ == '__main__':
    main()
