#!/usr/bin/env python3
"""Build and package REF4-SUPER-CHAMPION-077A production submission zip."""
import hashlib, json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / 'model/REF4-SUPER-CHAMPION-077A'
PROD_ASSETS = EXP_DIR / 'production_assets'
PROD_PKG = EXP_DIR / 'production_package'
BASE_PKG = ROOT / 'model/REF4-ADAPTIVE-CHANNEL-OPT-071A/production_package'
ZIP_OUT = ROOT / 'output/submit_ref4_super_champion_077.zip'
PYTHON_BIN = str(ROOT / '.venv-submit/bin/python')

SCRIPT_CONTENT = '''"""Offline inference entry point for REF4-SUPER-CHAMPION-077A."""
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
from src.stable_experts import context_ridge_correction, platoon_correction
from src.league_transition import transition_features

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
        member = [
            CatBoostClassifier().load_model(str(MODEL / f"subtype_{name}_seed{s}.cbm")).predict_proba(x3)[:, 1]
            for s in seeds
        ]
        risk = np.mean(member, axis=0)
        if futures.any():
            fm = CatBoostClassifier()
            fm.load_model(str(MODEL / f"f_subtype_{name}.cbm"))
            fr = fm.predict_proba(x3)[:, 1]
            risk = np.where(futures, risk + regime["subtype_scale"] * (fr - risk), risk)
        risks.append(risk)

    original_main = np.average(
        np.vstack(predictions), axis=0, weights=[0.27358084, 0.26512224, 0.46129691]
    )
    original_z = np.column_stack([original_main] + risks)
    original_p = 0.0300329767 + original_z @ np.asarray(
        [0.93505266, -0.00520129, 0.01091677, -0.02528331]
    )

    p = original_p.copy()

    # Adaptive Gate (Scale=0.05)
    if meta.get("adaptive_gate", False):
        gate_x = build_gate_features(
            test, predictions, risks, np.clip(original_p, 1e-6, 1 - 1e-6)
        )
        gate = CatBoostRegressor()
        gate.load_model(str(MODEL / "adaptive_gate.cbm"))
        gate_pred = gate.predict(gate_x)
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

    # 6. NEW 077A: Minimax Stable Experts (w=0.20)
    w_minimax = float(meta.get("w_minimax", 0.20))
    if w_minimax != 0.0 and (MODEL / "stable_context_ridge.npz").exists():
        psych_prof = pd.read_pickle(MODEL / "stable_context_profile.pkl")
        corr_ridge = context_ridge_correction(test, psych_prof, str(MODEL / "stable_context_ridge.npz"))
        platoon_m = CatBoostRegressor()
        platoon_m.load_model(str(MODEL / "stable_platoon.cbm"))
        corr_platoon = platoon_correction(test, platoon_m, scale=0.30)
        p = p + w_minimax * (corr_ridge + corr_platoon)

    # 7. NEW 077A: League Transition Gate (w=0.30)
    w_trans = float(meta.get("w_transition", 0.30))
    if w_trans != 0.0 and (MODEL / "transition_gate.cbm").exists():
        trans_m = CatBoostRegressor()
        trans_m.load_model(str(MODEL / "transition_gate.cbm"))
        tx = transition_features(test, p, str(MODEL / "prior_type.pkl"))
        corr_trans = trans_m.predict(tx)
        p = p + w_trans * corr_trans

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

def main():
    print("=" * 75)
    print("      BUILDING REF4-SUPER-CHAMPION-077A PRODUCTION PACKAGE       ")
    print("=" * 75)
    
    if PROD_PKG.exists():
        shutil.rmtree(PROD_PKG)
    PROD_PKG.mkdir(parents=True, exist_ok=True)
    (PROD_PKG / 'model').mkdir(parents=True, exist_ok=True)
    (PROD_PKG / 'src').mkdir(parents=True, exist_ok=True)
    
    # 1. Copy 071A Base Models and source files
    print("\n[Step 1] Copying 071A Base Models and src/...")
    for item in (BASE_PKG / 'model').iterdir():
        if item.is_file():
            shutil.copy2(item, PROD_PKG / 'model' / item.name)
            
    for item in (BASE_PKG / 'src').iterdir():
        if item.is_file():
            shutil.copy2(item, PROD_PKG / 'src' / item.name)
            
    # Copy new src files from 4번 레포
    repo4_src = ROOT / 'github_reference/4번 레포/src'
    for extra_src in ['context_adjusted_psych.py', 'context_pressure_features.py', 'stable_experts.py', 'league_transition.py']:
        if (repo4_src / extra_src).exists():
            shutil.copy2(repo4_src / extra_src, PROD_PKG / 'src' / extra_src)
            print(f"  • Added src/{extra_src}")
            
    # 2. Copy 077A Production Assets
    print("\n[Step 2] Copying 077A Production Assets...")
    for asset in ['stable_context_profile.pkl', 'stable_context_ridge.npz', 'stable_platoon.cbm', 'prior_type.pkl', 'transition_gate.cbm']:
        asset_path = PROD_ASSETS / asset
        if not asset_path.exists():
            raise FileNotFoundError(f"Missing production asset: {asset_path}")
        shutil.copy2(asset_path, PROD_PKG / 'model' / asset)
        print(f"  • Added model/{asset} ({asset_path.stat().st_size:,} bytes)")
        
    # 3. Update manifest.json with 077A hyperparameters
    print("\n[Step 3] Updating manifest.json...")
    manifest = json.loads((PROD_PKG / 'model/manifest.json').read_text())
    manifest['version'] = 'REF4-SUPER-CHAMPION-077A'
    manifest['adaptive_gate'] = True
    manifest['gate_scale'] = 0.05
    manifest['global_shift'] = 0.0052
    manifest['r_expert_lgbm_weight'] = 0.02
    manifest['w_minimax'] = 0.20
    manifest['w_transition'] = 0.30
    (PROD_PKG / 'model/manifest.json').write_text(json.dumps(manifest, indent=2))
    
    # 4. Write script.py and requirements.txt
    print("\n[Step 4] Writing script.py and requirements.txt...")
    (PROD_PKG / 'script.py').write_text(SCRIPT_CONTENT)
    (PROD_PKG / 'requirements.txt').write_text((BASE_PKG / 'requirements.txt').read_text())
    
    # 5. Build ZIP Package
    print("\n[Step 5] Building ZIP Package...")
    if ZIP_OUT.exists():
        ZIP_OUT.unlink()
        
    with zipfile.ZipFile(ZIP_OUT, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for root, dirs, files in os.walk(PROD_PKG):
            for file in sorted(files):
                abs_path = Path(root) / file
                rel_path = abs_path.relative_to(PROD_PKG)
                zf.write(abs_path, arcname=str(rel_path))
                
    zip_size = ZIP_OUT.stat().st_size
    zip_sha = hashlib.sha256(ZIP_OUT.read_bytes()).hexdigest()
    print(f"\nCreated: {ZIP_OUT}")
    print(f"Size:    {zip_size:,} bytes ({zip_size / 1e6:.2f} MB)")
    print(f"SHA-256: {zip_sha}")

if __name__ == '__main__':
    main()
