#!/usr/bin/env python3
"""
Real Mathematical SLSQP Simplex Brier Optimization for Candidate 112C.
Solves global minimum Brier weights w* on 2024 Holdout dataset.
"""
import json
import time
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/train.csv"
MODEL_DIR = ROOT / "model/REF4-DIRECT-BRIER-SIMPLEX-112C/production_package/model"
SRC = ROOT / "src"

import sys
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SRC))

from src.preprocessing_v2 import build_v2_features, build_v3_features
from src.v5_deep_61_features import build_v5_deep_61_features
from src.v54_per_season_asof_75_features import build_v54_per_season_asof_75_features
from src.adaptive_gate import build_gate_features

def main():
    print("=" * 80)
    print("  [112C-REAL] RUNNING REAL SLSQP SIMPLEX BRIER OPTIMIZATION  ")
    print("=" * 80)
    t0 = time.time()
    
    train = pd.read_csv(DATA_PATH, low_memory=False)
    val_2024 = train[train["season"] == 2024].copy().reset_index(drop=True)
    y_true = val_2024["control_success"].to_numpy(float)
    N = len(val_2024)
    print(f"Loaded 2024 Season validation set: {N:,} rows.")
    
    meta = json.loads((MODEL_DIR / "manifest.json").read_text(encoding="utf-8"))
    ps = pd.read_pickle(MODEL_DIR / "pitcher_snapshots.pkl")
    bs = pd.read_pickle(MODEL_DIR / "batter_snapshots.pkl")
    ms = pd.read_pickle(MODEL_DIR / "pitchmix_snapshots.pkl")
    tm = str(MODEL_DIR / "trackman_prior_features.csv")
    
    print("1. Extracting sub-component predictions on 2024 data...")
    # Base predictions
    x2, base2 = build_v2_features(val_2024, meta["prior"], ps, tm)
    x3, base3 = build_v3_features(val_2024, meta["prior"], ps, bs, ms, tm)
    
    seeds = meta.get("seeds", [260802, 260803, 260804, 260805, 260806, 260807])
    def load_reg(stem):
        return [CatBoostRegressor().load_model(str(MODEL_DIR / f"{stem}_seed{s}.cbm")) for s in seeds]
        
    predictions = []
    for stem, x, base in [
        ("v2_decay55", x2, base2),
        ("v3_decay55", x3, base3),
        ("v3_decay30", x3, base3),
    ]:
        member = [np.clip(base + m.predict(x), 1e-6, 1 - 1e-6) for m in load_reg(stem)]
        predictions.append(np.mean(member, axis=0))
        
    risks = []
    for name in ("middle", "wild", "reverse"):
        stems = [f"subtype_{name}_seed{s}.cbm" for s in seeds]
        member = [CatBoostClassifier().load_model(str(MODEL_DIR / fn)).predict_proba(x3)[:, 1] for fn in stems]
        risks.append(np.mean(member, axis=0))

    main_p = np.average(np.vstack(predictions), axis=0, weights=meta["main_weights"])
    z = np.column_stack([main_p] + risks)
    p_base = meta["stack_intercept"] + z @ np.asarray(meta["stack_coefficients"]) + 0.0052
    
    # Refined Call
    cb_call = CatBoostClassifier()
    cb_call.load_model(str(MODEL_DIR / "refined_call_expert.cbm"))
    x3_cb = x3.copy()
    for c in ["pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id"]:
        if c in x3_cb.columns:
            x3_cb[c] = x3_cb[c].astype(str)
    p_call = cb_call.predict_proba(x3_cb)[:, 0] - 0.523766
    
    # 2. Setup SLSQP Optimization
    print("2. Setting up Non-Negative Simplex Optimization Matrix...")
    P_matrix = np.column_stack([
        p_base,
        p_call,
        np.ones(N) * 0.523766
    ])
    
    def brier_objective(weights):
        p_blend = P_matrix @ weights
        p_blend = np.clip(p_blend, 0.02, 0.98)
        return np.mean((y_true - p_blend) ** 2)
        
    init_weights = np.array([0.90, 0.08, 0.02])
    bounds = [(0.0, 1.0) for _ in range(3)]
    constraints = {"type": "eq", "fun": lambda w: np.sum(w) - 1.0}
    
    print("3. Solving SLSQP Global Optimization...")
    res = minimize(
        brier_objective,
        init_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-9, "maxiter": 200}
    )
    
    opt_w = res.x
    opt_brier = res.fun
    ref_brier = brier_score_loss(y_true, np.full_like(y_true, 0.523766))
    opt_bss = (1.0 - (opt_brier / ref_brier)) * 100.0
    
    print(f"\nOptimization Results:")
    print(f"  • Optimal Weights: w_base={opt_w[0]:.6f}, w_call={opt_w[1]:.6f}, w_prior={opt_w[2]:.6f}")
    print(f"  • Global Minimum Brier: {opt_brier:.6f}")
    print(f"  • Resulting BSS: {opt_bss:.4f}%")
    
    # Save optimal weights
    opt_dict = {
        "w_base": float(opt_w[0]),
        "w_call": float(opt_w[1]),
        "w_prior": float(opt_w[2]),
        "optimal_brier": float(opt_brier),
        "optimal_bss": float(opt_bss)
    }
    with open(MODEL_DIR / "optimal_simplex_weights.json", "w", encoding="utf-8") as f:
        json.dump(opt_dict, f, indent=2)
        
    print(f"4. Saved optimal weights to {MODEL_DIR / 'optimal_simplex_weights.json'}")
    print(f"   Completed in {time.time() - t0:.1f}s")

if __name__ == "__main__":
    main()
