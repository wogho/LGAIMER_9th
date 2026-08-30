#!/usr/bin/env python3
"""Mathematical analysis of Optimal Variance Shrinkage (alpha*) on Forward OOF (2023, 2024)."""
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
CAND_069 = ROOT / 'model/REF4-ADAPTIVE-CHANNEL-OPT-069A/production_package'
sys.path.insert(0, str(CAND_069))

from src.preprocessing_v2 import build_v2_features, build_v3_features
from src.adaptive_gate import build_gate_features

def bss(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-5, 1 - 1e-5)
    r = float(y.mean())
    ref = r * (1.0 - r)
    return 1e5 * (1.0 - np.mean((p - y) ** 2) / ref)

def main():
    raw = pd.read_csv(ROOT / 'data/train.csv', low_memory=False)
    
    val_23 = raw.loc[raw.season == 2023].copy().reset_index(drop=True)
    val_24 = raw.loc[raw.season == 2024].copy().reset_index(drop=True)
    y_23 = val_23.control_success.to_numpy(float)
    y_24 = val_24.control_success.to_numpy(float)
    
    # Load 051A OOF predictions
    oof_051 = pd.read_csv(ROOT / 'model/REF4-TRAINONLY-R-SPECIFIC-SPLIT-051A/oof_predictions.csv').set_index('row_id')
    p_051_23 = oof_051.loc[val_23.row_id, 'candidate_prediction'].to_numpy(float)
    p_051_24 = oof_051.loc[val_24.row_id, 'candidate_prediction'].to_numpy(float)
    
    # Load gate
    gate = CatBoostRegressor()
    gate.load_model(str(CAND_069 / 'model/adaptive_gate.cbm'))
    
    MODEL = CAND_069 / 'model'
    meta = json.loads((MODEL / "manifest.json").read_text())
    ps = pd.read_pickle(MODEL / "pitcher_snapshots.pkl")
    bs = pd.read_pickle(MODEL / "batter_snapshots.pkl")
    ms = pd.read_pickle(MODEL / "pitchmix_snapshots.pkl")
    tm = str(MODEL / "trackman_prior_features.csv")
    seeds = meta.get("seeds", [260802, 260803, 260804, 260805, 260806, 260807])
    
    def load_reg(stem):
        return [CatBoostRegressor().load_model(str(MODEL / f"{stem}_seed{s}.cbm")) for s in seeds]

    def get_base_preds(x2, base2, x3, base3):
        preds = []
        for stem, x, base in [("v2_decay55", x2, base2), ("v3_decay55", x3, base3), ("v3_decay30", x3, base3)]:
            member = [np.clip(base + m.predict(x), 1e-6, 1 - 1e-6) for m in load_reg(stem)]
            preds.append(np.mean(member, axis=0))
        return preds

    def get_risks(x3):
        risks = []
        for name in ("middle", "wild", "reverse"):
            member = [CatBoostClassifier().load_model(str(MODEL / f"subtype_{name}_seed{s}.cbm")).predict_proba(x3)[:, 1] for s in seeds]
            risks.append(np.mean(member, axis=0))
        return risks

    x2_23, b2_23 = build_v2_features(val_23, meta["prior"], ps, tm)
    x3_23, b3_23 = build_v3_features(val_23, meta["prior"], ps, bs, ms, tm)
    preds_23 = get_base_preds(x2_23, b2_23, x3_23, b3_23)
    risks_23 = get_risks(x3_23)
    main_23 = np.average(np.vstack(preds_23), axis=0, weights=meta["main_weights"])
    z_23 = np.column_stack([main_23] + risks_23)
    p_stack_23 = meta["stack_intercept"] + z_23 @ np.asarray(meta["stack_coefficients"])
    gx_23 = build_gate_features(val_23, preds_23, risks_23, np.clip(p_stack_23, 1e-6, 1 - 1e-6))
    gate_clean_23 = gate.predict(gx_23) - 0.00848698

    x2_24, b2_24 = build_v2_features(val_24, meta["prior"], ps, tm)
    x3_24, b3_24 = build_v3_features(val_24, meta["prior"], ps, bs, ms, tm)
    preds_24 = get_base_preds(x2_24, b2_24, x3_24, b3_24)
    risks_24 = get_risks(x3_24)
    main_24 = np.average(np.vstack(preds_24), axis=0, weights=meta["main_weights"])
    z_24 = np.column_stack([main_24] + risks_24)
    p_stack_24 = meta["stack_intercept"] + z_24 @ np.asarray(meta["stack_coefficients"])
    gx_24 = build_gate_features(val_24, preds_24, risks_24, np.clip(p_stack_24, 1e-6, 1 - 1e-6))
    gate_clean_24 = gate.predict(gx_24) - 0.00848698

    # Evaluate 071A baseline (gate_scale = 0.05)
    p_071_23 = p_051_23 + 0.05 * gate_clean_23
    p_071_24 = p_051_24 + 0.05 * gate_clean_24
    
    print(f"=== 071A Baseline Performance ===")
    print(f"2023 BSS: {bss(y_23, p_071_23):.4f}, Var(p): {np.var(p_071_23):.6f}, Cov(p, y): {np.cov(p_071_23, y_23)[0,1]:.6f}")
    print(f"2024 BSS: {bss(y_24, p_071_24):.4f}, Var(p): {np.var(p_071_24):.6f}, Cov(p, y): {np.cov(p_071_24, y_24)[0,1]:.6f}")
    
    cov_23 = np.cov(p_071_23, y_23)[0,1]
    var_23 = np.var(p_071_23)
    alpha_opt_23 = cov_23 / var_23
    
    cov_24 = np.cov(p_071_24, y_24)[0,1]
    var_24 = np.var(p_071_24)
    alpha_opt_24 = cov_24 / var_24
    
    print(f"\nOptimal Theoretical Alpha: 2023={alpha_opt_23:.4f}, 2024={alpha_opt_24:.4f}")
    
    print("\n=== Grid Search: Alpha Shrinkage on 071A Baseline ===")
    for alpha in [1.00, 0.98, 0.95, 0.92, 0.90, 0.88, 0.85, 0.80]:
        p_shrunk_23 = np.mean(p_071_23) + alpha * (p_071_23 - np.mean(p_071_23))
        p_shrunk_24 = np.mean(p_071_24) + alpha * (p_071_24 - np.mean(p_071_24))
        
        bss_23 = bss(y_23, p_shrunk_23)
        bss_24 = bss(y_24, p_shrunk_24)
        comb_bss = bss(np.concatenate([y_23, y_24]), np.concatenate([p_shrunk_23, p_shrunk_24]))
        
        print(f"Alpha {alpha:.2f} -> 2023: {bss_23:8.4f} (diff: {bss_23 - bss(y_23, p_071_23):+8.4f}) | 2024: {bss_24:8.4f} (diff: {bss_24 - bss(y_24, p_071_24):+8.4f}) | Comb: {comb_bss:8.4f}")

    print("\n=== Joint Search: gate_scale (0.01 to 0.15) + Alpha Shrinkage ===")
    best_comb = -9999
    best_config = None
    for gs in [0.00, 0.02, 0.04, 0.05, 0.06, 0.08, 0.10]:
        for alpha in [1.00, 0.98, 0.95, 0.92, 0.90]:
            p_cand_23 = p_051_23 + gs * gate_clean_23
            p_cand_24 = p_051_24 + gs * gate_clean_24
            
            p_shrunk_23 = np.mean(p_cand_23) + alpha * (p_cand_23 - np.mean(p_cand_23))
            p_shrunk_24 = np.mean(p_cand_24) + alpha * (p_cand_24 - np.mean(p_cand_24))
            
            bss_23 = bss(y_23, p_shrunk_23)
            bss_24 = bss(y_24, p_shrunk_24)
            comb = bss(np.concatenate([y_23, y_24]), np.concatenate([p_shrunk_23, p_shrunk_24]))
            
            if comb > best_comb:
                best_comb = comb
                best_config = (gs, alpha, bss_23, bss_24, comb)
                
    print(f"\nBest Config Found: gate_scale={best_config[0]:.2f}, alpha={best_config[1]:.2f} -> 2023 BSS={best_config[2]:.4f}, 2024 BSS={best_config[3]:.4f}, Combined={best_config[4]:.4f}")

if __name__ == '__main__':
    main()
