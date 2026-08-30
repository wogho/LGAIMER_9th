#!/usr/bin/env python3
"""Systematic grid search for gate_scale and channel weights on 2023 and 2024 Forward OOF."""
import json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
CAND_DIR = ROOT / 'candidate/REF4-R-SPECIFIC-SPLIT-FPSYCH-055A'
sys.path.insert(0, str(CAND_DIR))
sys.path.insert(0, str(ROOT / 'github_reference/4번 레포'))

from src.preprocessing_v2 import build_v2_features, build_v3_features
from src.adaptive_gate import build_gate_features

def bss(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-5, 1 - 1e-5)
    r = float(y.mean())
    ref = r * (1.0 - r)
    return 1e5 * (1.0 - np.mean((p - y) ** 2) / ref)

def main():
    t0 = time.time()
    raw = pd.read_csv(ROOT / 'data/train.csv', low_memory=False)
    
    val_23 = raw.loc[raw.season == 2023].copy().reset_index(drop=True)
    val_24 = raw.loc[raw.season == 2024].copy().reset_index(drop=True)
    y_23 = val_23.control_success.to_numpy(float)
    y_24 = val_24.control_success.to_numpy(float)
    
    # Load 051A OOF predictions
    oof_051 = pd.read_csv(ROOT / 'model/REF4-TRAINONLY-R-SPECIFIC-SPLIT-051A/oof_predictions.csv').set_index('row_id')
    p_051_23 = oof_051.loc[val_23.row_id, 'candidate_prediction'].to_numpy(float)
    p_051_24 = oof_051.loc[val_24.row_id, 'candidate_prediction'].to_numpy(float)
    
    # Load trained production adaptive gate
    gate_prod = CatBoostRegressor()
    gate_prod.load_model(str(ROOT / 'model/REF4-ADAPTIVE-CHANNEL-OPT-069A/production_package/model/adaptive_gate.cbm'))
    
    MODEL = CAND_DIR / 'model'
    meta = json.loads((MODEL / "manifest.json").read_text(encoding="utf-8"))
    ps = pd.read_pickle(MODEL / "pitcher_snapshots.pkl")
    bs = pd.read_pickle(MODEL / "batter_snapshots.pkl")
    ms = pd.read_pickle(MODEL / "pitchmix_snapshots.pkl")
    tm = str(MODEL / "trackman_prior_features.csv")
    seeds = meta.get("seeds", [260802, 260803, 260804, 260805, 260806, 260807])
    
    def load_reg(stem):
        return [CatBoostRegressor().load_model(str(MODEL / f"{stem}_seed{s}.cbm")) for s in seeds]

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
                m.load_model(str(MODEL / f"subtype_{name}_seed{s}.cbm"))
                member.append(m.predict_proba(x3)[:, 1])
            risks.append(np.mean(member, axis=0))
        return risks

    print("Building features for 2023 and 2024...")
    x2_23, b2_23 = build_v2_features(val_23, meta["prior"], ps, tm)
    x3_23, b3_23 = build_v3_features(val_23, meta["prior"], ps, bs, ms, tm)
    preds_23 = get_base_preds(x2_23, b2_23, x3_23, b3_23)
    risks_23 = get_risks(x3_23)
    main_23 = np.average(np.vstack(preds_23), axis=0, weights=meta["main_weights"])
    z_23 = np.column_stack([main_23] + risks_23)
    p_stack_23 = meta["stack_intercept"] + z_23 @ np.asarray(meta["stack_coefficients"])
    gate_x_23 = build_gate_features(val_23, preds_23, risks_23, np.clip(p_stack_23, 1e-6, 1 - 1e-6))
    
    x2_24, b2_24 = build_v2_features(val_24, meta["prior"], ps, tm)
    x3_24, b3_24 = build_v3_features(val_24, meta["prior"], ps, bs, ms, tm)
    preds_24 = get_base_preds(x2_24, b2_24, x3_24, b3_24)
    risks_24 = get_risks(x3_24)
    main_24 = np.average(np.vstack(preds_24), axis=0, weights=meta["main_weights"])
    z_24 = np.column_stack([main_24] + risks_24)
    p_stack_24 = meta["stack_intercept"] + z_24 @ np.asarray(meta["stack_coefficients"])
    gate_x_24 = build_gate_features(val_24, preds_24, risks_24, np.clip(p_stack_24, 1e-6, 1 - 1e-6))
    
    gate_p_23 = gate_prod.predict(gate_x_23) - 0.00848698
    gate_p_24 = gate_prod.predict(gate_x_24) - 0.00848698
    
    print("\n=== Grid Search: gate_scale with 051A Baseline ===")
    print(f"051A Base: 2023 BSS={bss(y_23, p_051_23):.4f}, 2024 BSS={bss(y_24, p_051_24):.4f}")
    
    best_pooled_bss = -9999
    best_scale = 0.0
    
    for s in [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.75, 1.0]:
        p_eval_23 = np.clip(p_051_23 + s * gate_p_23, 1e-5, 1 - 1e-5)
        p_eval_24 = np.clip(p_051_24 + s * gate_p_24, 1e-5, 1 - 1e-5)
        
        bss_23 = bss(y_23, p_eval_23)
        bss_24 = bss(y_24, p_eval_24)
        
        y_comb = np.concatenate([y_23, y_24])
        p_comb = np.concatenate([p_eval_23, p_eval_24])
        bss_comb = bss(y_comb, p_comb)
        
        print(f"Scale {s:.2f} -> 2023: {bss_23:8.4f} (diff vs 051: {bss_23 - bss(y_23, p_051_23):+8.4f}) | 2024: {bss_24:8.4f} (diff vs 051: {bss_24 - bss(y_24, p_051_24):+8.4f}) | Combined: {bss_comb:8.4f}")
        
if __name__ == '__main__':
    main()
