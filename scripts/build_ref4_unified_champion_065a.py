#!/usr/bin/env python3
"""Build and package REF4-UNIFIED-CHAMPION-065A production candidate."""
import hashlib, json, os, shutil, subprocess, sys, tempfile, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC_030 = ROOT / 'candidate/REF4-CHAMPION-STACK-030'
SRC_055 = ROOT / 'candidate/REF4-R-SPECIFIC-SPLIT-FPSYCH-055A'
SRC_064 = ROOT / 'candidate/REF4-LGBM-R-EXPERT-064A'
CAND_065 = ROOT / 'candidate/REF4-UNIFIED-CHAMPION-065A'
OUT_ZIP = ROOT / 'output/submit_ref4_unified_champion_065.zip'
PYTHON = str(ROOT / '.venv-submit/bin/python')

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def make_eval_frame(n_rows: int = 1500, seed: int = 42) -> pd.DataFrame:
    columns = pd.read_csv(ROOT / "data/test.csv", nrows=1).columns
    train = pd.read_csv(ROOT / "data/train.csv", low_memory=False)
    rows = train[train.season == 2024].sample(n_rows, random_state=seed).reset_index(drop=True)
    rows["season"] = 2025
    rows["row_id"] = [f"TEST_{i:06d}" for i in range(len(rows))]
    return rows[columns]

def predict(workdir: Path, frame: pd.DataFrame) -> pd.Series:
    (workdir / "data").mkdir(exist_ok=True)
    frame.to_csv(workdir / "data/test.csv", index=False)
    done = subprocess.run([PYTHON, "script.py"], cwd=workdir, capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"inference failed:\n{done.stdout}\n{done.stderr}")
    out = pd.read_csv(workdir / "output/submission.csv")
    return out.set_index("row_id").control_success

def test_row_independence(workdir: Path, n_rows: int = 1500, singles: int = 8, tol: float = 1e-5):
    print(f"\n--- Testing Row Independence on {workdir.name} ---")
    frame = make_eval_frame(n_rows, seed=42)
    full = predict(workdir, frame)
    
    checks = []
    half = frame.iloc[: len(frame) // 2]
    checks.append(("half file", full.reindex(half.row_id), predict(workdir, half)))
    
    shuffled = frame.sample(frac=1.0, random_state=43)
    checks.append(("shuffled file", full.reindex(shuffled.row_id), predict(workdir, shuffled)))
    
    picks = frame.iloc[:: max(1, len(frame) // singles)].head(singles)
    alone = pd.concat([predict(workdir, frame.iloc[[i]]) for i in picks.index])
    checks.append((f"{len(picks)} rows alone", full.reindex(alone.index), alone))
    
    worst = 0.0
    for name, ref, other in checks:
        gap = float(np.max(np.abs(ref.to_numpy(float) - other.to_numpy(float))))
        worst = max(worst, gap)
        verdict = "동일" if gap <= 1e-9 else ("부동소수점 오차" if gap <= tol else "누수")
        print(f"  {name:<22} rows={len(other):5d}  max|diff| = {gap:.3e}   {verdict}")
        
    assert worst <= tol, f"FAIL: max diff {worst:.3e} exceeds tolerance {tol}"
    print(f"PASS: Row-independence verified (worst diff: {worst:.3e} <= {tol})\n")
    return worst

def main():
    t0 = time.time()
    print("--- Step 1: Assembling Unified Champion Assets (REF4-UNIFIED-CHAMPION-065A) ---")
    if CAND_065.exists():
        shutil.rmtree(CAND_065)
    CAND_065.mkdir(parents=True)
    (CAND_065 / 'model').mkdir()
    
    # Copy src/ from 055A which has all required modules
    shutil.copytree(SRC_055 / 'src', CAND_065 / 'src')
    
    # Copy all models from 030
    for f in (SRC_030 / 'model').iterdir():
        if f.is_file():
            shutil.copy2(f, CAND_065 / 'model' / f.name)
            
    # Copy R-split profile and LightGBM model from 064A
    shutil.copy2(SRC_064 / 'model/split_profile.csv', CAND_065 / 'model/split_profile.csv')
    shutil.copy2(SRC_064 / 'model/split_residual_meta.npz', CAND_065 / 'model/split_residual_meta.npz')
    shutil.copy2(SRC_064 / 'model/r_expert_lgbm.txt', CAND_065 / 'model/r_expert_lgbm.txt')
    
    # Update manifest.json
    manifest = json.loads((SRC_030 / 'model/manifest.json').read_text(encoding='utf-8'))
    manifest['version'] = 4
    manifest['global_shift'] = 0.0052
    manifest['r_expert_blend_weight'] = 0.02
    manifest['r_expert_shift_offset'] = 0.0052
    manifest['notes'] = "Unified 1126.45 Champion Architecture (56 models + F-regime 0.75) + 051A R-split table + 063A LightGBM R-expert (w=0.02)."
    (CAND_065 / 'model/manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    
    # Update f_regime_meta.json to exact 0.75-scaled winning parameters
    f_regime_meta = {
        "v2_scale": 1.5,
        "v355_scale": 0.375,
        "v330_scale": 0.375,
        "v330_all_weight": 0.25,
        "v330_recent_inner_scale": 0.25,
        "subtype_scale": 0.5625,
        "transition_scale": 0.1125
    }
    (CAND_065 / 'model/f_regime_meta.json').write_text(json.dumps(f_regime_meta, ensure_ascii=False, indent=2) + '\n')
    
    # Write requirements.txt
    (CAND_065 / 'requirements.txt').write_text(
        "numpy>=1.24.0\n"
        "pandas>=2.0.0\n"
        "catboost>=1.2.0\n"
        "lightgbm>=4.0.0\n"
    )
    
    # Write script.py
    script_code = '''"""Offline inference entry point for the Unified 1126.45 Champion Stack with LightGBM R-Expert."""
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
from src.entity_context_split import apply_split_profile, apply_linear_split

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "model"


def main():
    started = time.time()
    meta = json.loads((MODEL / "manifest.json").read_text(encoding="utf-8"))
    regime = json.loads((MODEL / "f_regime_meta.json").read_text(encoding="utf-8"))
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

    futures = test["game_type"].eq("F").to_numpy()
    def f_reg_mean(stem, count, x, base):
        member=[]
        for j in range(count):
            m=CatBoostRegressor();m.load_model(MODEL / f"{stem}_{j}.cbm")
            member.append(np.clip(base+m.predict(x),1e-6,1-1e-6))
        return np.mean(member,axis=0)
        
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
        stems = [f"subtype_{name}_seed{s}.cbm" for s in seeds] if seeds else [f"subtype_{name}.cbm"]
        member = []
        for filename in stems:
            model = CatBoostClassifier()
            model.load_model(MODEL / filename)
            member.append(model.predict_proba(x3)[:, 1])
        risk = np.mean(member, axis=0)
        if futures.any():
            fm = CatBoostClassifier()
            fm.load_model(MODEL / f"f_subtype_{name}.cbm")
            fr = fm.predict_proba(x3)[:, 1]
            risk = np.where(futures, risk + regime["subtype_scale"] * (fr - risk), risk)
        risks.append(risk)

    main_p = np.average(np.vstack(predictions), axis=0, weights=meta["main_weights"])
    z = np.column_stack([main_p] + risks)
    p = meta["stack_intercept"] + z @ np.asarray(meta["stack_coefficients"])

    # Fixed global calibration shift (+0.0052)
    p = p + float(meta.get("global_shift", 0.0))

    # League transition gate
    transition = CatBoostRegressor()
    transition.load_model(MODEL / "transition_gate.cbm")
    tx = transition_features(test, p, MODEL / "prior_type.pkl")
    p = p + regime["transition_scale"] * transition.predict(tx)

    # 1군(Regular) R-specific split prior table + LightGBM expert (w=0.02)
    regular = ~futures
    if regular.any():
        split_profile = pd.read_csv(
            MODEL / "split_profile.csv", dtype={"entity_value": str, "context_value": str}
        )
        split_x = apply_split_profile(test, split_profile)
        split_correction = apply_linear_split(split_x, MODEL / "split_residual_meta.npz")
        p_r_split = p + split_correction
        
        # LightGBM R-Expert
        x3_lgb = x3.copy()
        for col in CAT_V2:
            if col in x3_lgb.columns:
                x3_lgb[col] = x3_lgb[col].astype('category')
        lgb_booster = lgb.Booster(model_file=str(MODEL / "r_expert_lgbm.txt"))
        r_expert = np.clip(base3 + lgb_booster.predict(x3_lgb) + meta.get("r_expert_shift_offset", 0.0052), 1e-6, 1.0 - 1e-6)
        
        w_lgb = float(meta.get("r_expert_blend_weight", 0.02))
        p = np.where(regular, (1.0 - w_lgb) * p_r_split + w_lgb * r_expert, p)

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
    (CAND_065 / 'script.py').write_text(script_code)
    
    # 2. Test candidate in temporary directory
    print("\n--- Step 2: Testing Row-Independence on Candidate Directory ---")
    with tempfile.TemporaryDirectory(prefix="candcheck-") as tmpdir:
        tmp_path = Path(tmpdir)
        for item in CAND_065.iterdir():
            if item.is_dir():
                shutil.copytree(item, tmp_path / item.name)
            else:
                shutil.copy2(item, tmp_path / item.name)
        worst_cand = test_row_independence(tmp_path)
        
    # 3. Package ZIP
    print(f"\n--- Step 3: Packaging ZIP: {OUT_ZIP.name} ---")
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
        
    with zipfile.ZipFile(OUT_ZIP, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for root_dir, dirs, files in os.walk(CAND_065):
            for file in files:
                full_p = Path(root_dir) / file
                rel_p = full_p.relative_to(CAND_065)
                z.write(full_p, arcname=str(rel_p))
                
    zip_size = OUT_ZIP.stat().st_size
    zip_sha = sha256_file(OUT_ZIP)
    print(f"ZIP created: {OUT_ZIP} ({zip_size / (1024*1024):.2f} MB, SHA256={zip_sha})")
    
    # 4. Test ZIP extraction and inference
    print("\n--- Step 4: Testing Row-Independence on Extracted ZIP ---")
    with tempfile.TemporaryDirectory(prefix="zipcheck-") as tmpdir:
        tmp_path = Path(tmpdir)
        with zipfile.ZipFile(OUT_ZIP) as z:
            z.extractall(tmp_path)
        worst_zip = test_row_independence(tmp_path)
        
    report = {
        'candidate_dir': str(CAND_065),
        'zip_file': str(OUT_ZIP),
        'zip_sha256': zip_sha,
        'zip_size_bytes': zip_size,
        'row_independence_worst_diff_candidate': worst_cand,
        'row_independence_worst_diff_zip': worst_zip,
        'row_independence_status': 'PASS',
        'elapsed_seconds': time.time() - t0
    }
    
    out_dir = ROOT / 'model/REF4-UNIFIED-CHAMPION-065A'
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'production_package_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print("Production package report saved at", out_dir / 'production_package_report.json')
    print(f"Completed in {time.time() - t0:.2f}s")

if __name__ == '__main__':
    main()
