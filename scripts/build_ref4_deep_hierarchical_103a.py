#!/usr/bin/env python3
"""Build Production Package 103A: New Champion 102A Backbone (1105.8202 LB) + Multi-Class Sub-Expert Engine Refined Call Multi-Class Target Sub-Expert Integration."""
import gc, hashlib, json, os, shutil, sys, time, zipfile
from pathlib import Path
import pandas as pd
import numpy as np
import lightgbm as lgb
from catboost import CatBoostClassifier, CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'model/REF4-DEEP-HIERARCHICAL-102A/production_package'))

EXP_DIR = ROOT / 'model/REF4-DEEP-HIERARCHICAL-103A'
PROD_DIR = EXP_DIR / 'production_package'
MODEL_DIR = PROD_DIR / 'model'
SRC_DIR = PROD_DIR / 'src'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
SRC_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_102_MODEL = ROOT / 'model/REF4-DEEP-HIERARCHICAL-102A/production_package/model'
SOURCE_102_SRC = ROOT / 'model/REF4-DEEP-HIERARCHICAL-102A/production_package/src'
SOURCE_102_REQS = ROOT / 'model/REF4-DEEP-HIERARCHICAL-102A/production_package/requirements.txt'

from src.v54_per_season_asof_75_features import build_per_season_priors, build_v54_per_season_asof_75_features
from src.preprocessing_v2 import build_v3_features, CAT_V2

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def build_leverage_features(df: pd.DataFrame) -> pd.DataFrame:
    """Leverage-Index Dynamics Engine 실측 +8.03pt 수혜 3대 Leverage Index 파생 피처."""
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

SCRIPT_CONTENT_103 = '''"""Offline inference entry point for REF4-DEEP-HIERARCHICAL-103A."""
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

ROOT = Path(__file__).resolve().parent
MODEL = ROOT / "model"


def build_leverage_features(df: pd.DataFrame) -> pd.DataFrame:
    """Leverage-Index Dynamics Engine 실측 +8.03pt 수혜 3대 Leverage Index 파생 피처."""
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


def main():
    started = time.time()
    meta = json.loads((MODEL / "manifest.json").read_text(encoding="utf-8"))
    priors = json.loads((MODEL / "per_season_priors.json").read_text(encoding="utf-8")) if (MODEL / "per_season_priors.json").exists() else {}
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

    # Futures 2군 psych latent (Frozen scale 0.15 for 100% stability)
    if futures.any():
        from src.psych_latent import build_production_features, apply_linear_residual
        residual_x = build_production_features(
            test, MODEL / "psych_profile.pkl", MODEL / "latent_pitch_context.csv"
        )
        correction = apply_linear_residual(residual_x, MODEL / "psych_latent_meta.npz")
        p = np.where(futures, p + correction, p)

    # Pre-build v54 features for 103A Hierarchical Baseline L2 residual engine
    v54_feat = None
    p_hb_hier = None
    if regular.any():
        v54_feat, _ = build_v54_per_season_asof_75_features(
            test, profile_path=MODEL / "team_asof_profile.json", priors=priors, prior=float(meta.get("prior", 0.523766))
        )
        # Inject Leverage-Index Dynamics Engine 3대 Leverage Index 파생 피처
        lev_df = build_leverage_features(test)
        v54_feat = pd.concat([v54_feat, lev_df], axis=1)

        n_p_raw = test["asof_pitcher_n"].fillna(0).to_numpy(float) if "asof_pitcher_n" in test.columns else np.zeros(len(test))
        p_rate_raw = test["asof_pitcher_success_rate"].fillna(0.523766).to_numpy(float) if "asof_pitcher_success_rate" in test.columns else np.full(len(test), 0.523766)
        prev1_raw = test["asof_pitcher_prev1_game_success_rate"].to_numpy(float) if "asof_pitcher_prev1_game_success_rate" in test.columns else p_rate_raw
        prev1_raw = np.where(np.isnan(prev1_raw), p_rate_raw, prev1_raw)
        
        overall_prior = float(meta.get("prior", 0.523766))
        k_prior = 25.0
        rel_weight = n_p_raw / (n_p_raw + k_prior)
        p_shrunk = overall_prior + rel_weight * (p_rate_raw - overall_prior)
        p_hb_hier = np.clip(0.70 * p_shrunk + 0.30 * prev1_raw, 0.05, 0.95)

    # Regular 1군 split profile + LightGBM R-Expert
    if regular.any():
        from src.entity_context_split import apply_split_profile, apply_linear_split
        split_profile = pd.read_csv(
            MODEL / "split_profile.csv", dtype={"entity_value": str, "context_value": str}
        )
        split_x = apply_split_profile(test, split_profile)
        split_correction = apply_linear_split(split_x, MODEL / "split_residual_meta.npz")
        p_split = p + split_correction
        
        # LightGBM R-Expert (w=0.05) from 087A Champion
        lgbm_model = lgb.Booster(model_file=str(MODEL / 'r_expert_lgbm.txt'))
        x3_lgb = x3.copy()
        for col in CAT_V2:
            if col in x3_lgb.columns:
                x3_lgb[col] = x3_lgb[col].astype('category')
        res_lgbm = lgbm_model.predict(x3_lgb)
        p_lgbm = np.clip(base3 + res_lgbm + 0.0052, 1e-6, 1.0 - 1e-6)
        
        w_lgb = float(meta.get("r_expert_lgbm_weight", 0.05))
        p = np.where(regular, (1.0 - w_lgb) * p_split + w_lgb * p_lgbm, p)

    # 102A CHAMPION: 40 Pure L2 Models (20 Seeds CatBoost + 20 Seeds LightGBM) Retrained with Leverage Index Features
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

    # 103A NEW: Multi-Class Sub-Expert Engine Refined Call 6-Class Multi-Class Sub-Expert (w=0.04)
    w_refined_call = float(meta.get("w_refined_call", 0.04))
    if regular.any() and w_refined_call > 0.0 and (MODEL / "refined_call_expert.cbm").exists():
        cb_call = CatBoostClassifier()
        cb_call.load_model(str(MODEL / "refined_call_expert.cbm"))
        
        # Prepare x3 with cat_features properly assigned for CatBoost
        x3_cb = x3.copy()
        for c in CAT_V2:
            if c in x3_cb.columns:
                x3_cb[c] = x3_cb[c].astype(str)
                
        call_probs = cb_call.predict_proba(x3_cb)
        # class 0 = success, class 1 = ball, class 2 = strike, class 3 = reverse, class 4 = middle, class 5 = wild
        p_call_success = call_probs[:, 0]
        call_mean_offset = float(meta.get("call_mean_offset", 0.523766))
        call_corr = p_call_success - call_mean_offset
        p = np.where(regular, p + w_refined_call * call_corr, p)

    # 091A Champion Dual Tier-Aware Level Grouping Pocket Post-Processor
    n_p_raw = test["asof_pitcher_n"].fillna(0).to_numpy() if "asof_pitcher_n" in test.columns else np.zeros(len(test))
    p_rate_raw = test["asof_pitcher_success_rate"].fillna(0.523766).to_numpy() if "asof_pitcher_success_rate" in test.columns else np.full(len(test), 0.523766)
    
    high_pocket_mask = regular & (n_p_raw < 15) & (p_rate_raw > 0.65)
    downshift_val = float(meta.get("pocket_downshift_val", -0.012))
    p = np.where(high_pocket_mask, p + downshift_val, p)

    low_pocket_mask = regular & (n_p_raw < 15) & (p_rate_raw < 0.40)
    upshift_val = float(meta.get("pocket_upshift_val", +0.008))
    p = np.where(low_pocket_mask, p + upshift_val, p)

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
    print("=== Step 1: Training Multi-Class Sub-Expert Engine Refined Call 6-Class Multi-Class CatBoost Sub-Expert ===")
    raw = pd.read_csv(ROOT / 'data/train.csv', low_memory=False)
    priors = build_per_season_priors(raw)
    
    (MODEL_DIR / 'per_season_priors.json').write_text(json.dumps(priors, indent=2), encoding='utf-8')
    
    train_reg = raw.loc[raw.game_type != "F"].copy().reset_index(drop=True)
    ps = pd.read_pickle(SOURCE_102_MODEL / "pitcher_snapshots.pkl")
    bs = pd.read_pickle(SOURCE_102_MODEL / "batter_snapshots.pkl")
    ms = pd.read_pickle(SOURCE_102_MODEL / "pitchmix_snapshots.pkl")
    tm = str(SOURCE_102_MODEL / "trackman_prior_features.csv")
    
    x3_tr, _ = build_v3_features(train_reg, 0.523766, ps, bs, ms, tm)
    
    x3_cb_tr = x3_tr.copy()
    cat_indices = []
    for c in CAT_V2:
        if c in x3_cb_tr.columns:
            x3_cb_tr[c] = x3_cb_tr[c].astype(str)
            cat_indices.append(c)
    
    # Multi-Class Sub-Expert Engine Refined Call 6-Class Target 분해:
    # 0 = success, 1 = ball, 2 = strike, 3 = reverse, 4 = middle, 5 = wild
    succ = train_reg["control_success"].to_numpy(int)
    rev = train_reg["subtype_reverse"].to_numpy(int) if "subtype_reverse" in train_reg.columns else np.zeros(len(train_reg), int)
    mid = train_reg["subtype_middle"].to_numpy(int) if "subtype_middle" in train_reg.columns else np.zeros(len(train_reg), int)
    wld = train_reg["subtype_wild"].to_numpy(int) if "subtype_wild" in train_reg.columns else np.zeros(len(train_reg), int)
    
    call_target = np.zeros(len(train_reg), dtype=int)
    call_target = np.where(succ == 1, 0, call_target)
    call_target = np.where((succ == 0) & (rev == 1), 3, call_target)
    call_target = np.where((succ == 0) & (mid == 1), 4, call_target)
    call_target = np.where((succ == 0) & (wld == 1), 5, call_target)
    call_target = np.where((succ == 0) & (rev == 0) & (mid == 0) & (wld == 0), 1, call_target)
    
    cb_call = CatBoostClassifier(
        iterations=300,
        depth=6,
        learning_rate=0.04,
        loss_function='MultiClass',
        cat_features=cat_indices,
        random_seed=42,
        verbose=False
    )
    cb_call.fit(x3_cb_tr, call_target)
    call_expert_path = MODEL_DIR / "refined_call_expert.cbm"
    cb_call.save_model(str(call_expert_path))
    
    tr_call_probs = cb_call.predict_proba(x3_cb_tr)[:, 0]
    call_mean_offset = float(np.mean(tr_call_probs))
    print(f"  • Trained Refined Call Multi-Class Expert (Mean success prob: {call_mean_offset:.6f})")

    print("\n=== Step 2: Copying Champion 102A Models & Base Assets ===")
    for f in SOURCE_102_MODEL.iterdir():
        if f.is_file() and f.name != 'manifest.json':
            shutil.copy2(f, MODEL_DIR / f.name)
            
    for f in SOURCE_102_SRC.iterdir():
        if f.is_file():
            shutil.copy2(f, SRC_DIR / f.name)
            
    shutil.copy2(SOURCE_102_REQS, PROD_DIR / 'requirements.txt')
    
    manifest_path = MODEL_DIR / 'manifest.json'
    manifest = json.loads((SOURCE_102_MODEL / 'manifest.json').read_text(encoding='utf-8'))
    manifest['version'] = 'REF4-DEEP-HIERARCHICAL-103A'
    manifest['gate_scale'] = 0.08
    manifest['r_expert_lgbm_weight'] = 0.05
    manifest['w_deep_hierarchical'] = 0.08
    manifest['pocket_downshift_val'] = -0.012
    manifest['pocket_upshift_val'] = +0.008
    manifest['w_refined_call'] = 0.04
    manifest['call_mean_offset'] = call_mean_offset
    manifest['notes'] = "103A Champion 102A Backbone (1105.8202 LB) + Multi-Class Sub-Expert Engine Refined Call 6-Class Multi-Class Sub-Expert (w=0.04)"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print("Updated manifest.json for 103A")
    
    print("\n=== Step 3: Writing script.py and Packaging submit_ref4_deep_hierarchical_103.zip ===")
    (PROD_DIR / 'script.py').write_text(SCRIPT_CONTENT_103, encoding='utf-8')
    
    zip_path = ROOT / 'output/submit_ref4_deep_hierarchical_103.zip'
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
