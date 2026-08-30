#!/usr/bin/env python3
"""Strict Forward-OOF test of Adaptive Gate on 2024."""
import gc, json, os, sys, time
from pathlib import Path
from catboost import CatBoostClassifier, CatBoostRegressor
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CAND_DIR = ROOT / 'candidate/REF4-R-SPECIFIC-SPLIT-FPSYCH-055A'
sys.path.insert(0, str(CAND_DIR))
sys.path.insert(0, str(ROOT / 'github_reference/4번 레포'))

from src.preprocessing_v2 import build_v2_features, build_v3_features, CAT_V2
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
    
    # Check 2023 and 2024
    print("Loading 2023 and 2024 data...")
    val_23 = raw.loc[raw.season == 2023].copy().reset_index(drop=True)
    val_24 = raw.loc[raw.season == 2024].copy().reset_index(drop=True)
    
    y_23 = val_23.control_success.to_numpy(float)
    y_24 = val_24.control_success.to_numpy(float)
    
    MODEL = CAND_DIR / 'model'
    meta = json.loads((MODEL / "manifest.json").read_text(encoding="utf-8"))
    ps = pd.read_pickle(MODEL / "pitcher_snapshots.pkl")
    bs = pd.read_pickle(MODEL / "batter_snapshots.pkl")
    ms = pd.read_pickle(MODEL / "pitchmix_snapshots.pkl")
    tm = str(MODEL / "trackman_prior_features.csv")
    
    print("Building features for 2023 and 2024...")
    x2_23, base2_23 = build_v2_features(val_23, meta["prior"], ps, tm)
    x3_23, base3_23 = build_v3_features(val_23, meta["prior"], ps, bs, ms, tm)
    
    x2_24, base2_24 = build_v2_features(val_24, meta["prior"], ps, tm)
    x3_24, base3_24 = build_v3_features(val_24, meta["prior"], ps, bs, ms, tm)
    
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

    print("Computing base predictions and risks...")
    preds_23 = get_base_preds(x2_23, base2_23, x3_23, base3_23)
    risks_23 = get_risks(x3_23)
    
    preds_24 = get_base_preds(x2_24, base2_24, x3_24, base3_24)
    risks_24 = get_risks(x3_24)
    
    main_23 = np.average(np.vstack(preds_23), axis=0, weights=meta["main_weights"])
    z_23 = np.column_stack([main_23] + risks_23)
    p_stack_23 = meta["stack_intercept"] + z_23 @ np.asarray(meta["stack_coefficients"])
    
    main_24 = np.average(np.vstack(preds_24), axis=0, weights=meta["main_weights"])
    z_24 = np.column_stack([main_24] + risks_24)
    p_stack_24 = meta["stack_intercept"] + z_24 @ np.asarray(meta["stack_coefficients"])
    
    print(f"2023 Base Stack BSS: {bss(y_23, p_stack_23 + 0.0052):.4f}")
    print(f"2024 Base Stack BSS: {bss(y_24, p_stack_24 + 0.0052):.4f}")
    
    print("Building gate features for 2023...")
    gate_x_23 = build_gate_features(val_23, preds_23, risks_23, np.clip(p_stack_23, 1e-6, 1 - 1e-6))
    gate_x_24 = build_gate_features(val_24, preds_24, risks_24, np.clip(p_stack_24, 1e-6, 1 - 1e-6))
    
    # Train adaptive gate on 2023
    print("Fitting adaptive gate on 2023 residuals...")
    gate = CatBoostRegressor(iterations=100, depth=4, learning_rate=0.03, loss_function='RMSE', l2_leaf_reg=30, random_strength=0.2, random_seed=280033, verbose=False)
    gate.fit(gate_x_23, y_23 - p_stack_23)
    
    gate_pred_24 = gate.predict(gate_x_24)
    # Zero center the gate output to guarantee 0 shift!
    gate_pred_24_clean = gate_pred_24 - gate_pred_24.mean()
    
    print(f"Gate pred on 2024: raw_mean={gate_pred_24.mean():+.6f}, clean_mean={gate_pred_24_clean.mean():+.6f}, std={gate_pred_24.std():.6f}")
    
    for scale in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        p_gated = p_stack_24 + scale * gate_pred_24_clean + 0.0052
        print(f"  Scale {scale:.1f}: 2024 BSS = {bss(y_24, p_gated):.4f} (Brier: {np.mean((p_gated - y_24)**2):.7f})")
        
if __name__ == '__main__':
    main()
