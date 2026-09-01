#!/usr/bin/env python3
"""
Strict Out-of-Time (OOT) Evaluation Framework.
- Train Set: Seasons 2019-2023 (1,003,799 rows)
- Validation Set: Season 2024 (253,507 rows, completely unseen)
- Tests:
  1. Base Reference Prior (0.523766)
  2. Raw AS-OF Success Rate (asof_pitcher_success_rate)
  3. Strict Hierarchical Empirical Bayes (EB Pitcher x Hand)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

ROOT = Path(__file__).resolve().parents[1]

def compute_bss(y_true: np.ndarray, y_pred: np.ndarray, ref_p: float = 0.523766) -> tuple[float, float]:
    bs = brier_score_loss(y_true, y_pred)
    bs_ref = brier_score_loss(y_true, np.full_like(y_true, ref_p))
    bss = (1.0 - (bs / bs_ref)) * 100.0
    return bss, bs

def main():
    print("=" * 80)
    print("  STRICT 2024 OUT-OF-TIME (OOT) VALIDATION EXPERIMENT  ")
    print("=" * 80)
    
    train_df = pd.read_csv(ROOT / "data/train.csv", low_memory=False)
    
    past_df = train_df[train_df["season"] < 2024].copy()
    val_df = train_df[train_df["season"] == 2024].copy().reset_index(drop=True)
    y_val = val_df["control_success"].to_numpy(float)
    
    print(f"Past History (2019-2023): {len(past_df):,} rows")
    print(f"2024 Unseen Holdout:       {len(val_df):,} rows")
    
    # 1. Base Reference Prior
    p_ref = np.full(len(val_df), 0.523766)
    bss_ref, bs_ref = compute_bss(y_val, p_ref)
    print(f"\n[1] Global Constant Prior (0.523766):")
    print(f"    • Brier Score: {bs_ref:.6f} | BSS: {bss_ref:.4f}")
    
    # 2. Raw AS-OF Success Rate
    p_asof = val_df["asof_pitcher_success_rate"].fillna(0.523766).to_numpy(float)
    bss_asof, bs_asof = compute_bss(y_val, p_asof)
    print(f"\n[2] Raw AS-OF Success Rate:")
    print(f"    • Brier Score: {bs_asof:.6f} | BSS: {bss_asof:.4f} (Delta BSS: {bss_asof - bss_ref:+.4f})")
    
    # 3. Strict Hierarchical Empirical Bayes (trained only on 2019-2023)
    sys.path.insert(0, str(ROOT / "scripts"))
    from build_eb_lookup import build_hierarchical_eb_tables
    eb_tables = build_hierarchical_eb_tables(past_df)
    
    p_map = eb_tables["pitcher_baseline"]
    ph_map = eb_tables["pitcher_hand_lookup"]
    glob_p = eb_tables["global_prior"]
    
    p_eb = []
    for _, row in val_df.iterrows():
        pid = str(row["pitcher_id"])
        bhand = str(row["batter_hand"])
        pair_key = f"{pid}_{bhand}"
        if pair_key in ph_map:
            p_eb.append(ph_map[pair_key])
        elif pid in p_map:
            p_eb.append(p_map[pid])
        else:
            p_eb.append(glob_p)
            
    p_eb = np.array(p_eb, dtype=float)
    bss_eb, bs_eb = compute_bss(y_val, p_eb)
    print(f"\n[3] Strict Hierarchical Empirical Bayes (2019-2023 -> 2024):")
    print(f"    • Brier Score: {bs_eb:.6f} | BSS: {bss_eb:.4f} (Delta BSS: {bss_eb - bss_ref:+.4f})")
    
    # 4. Composite: Hierarchical EB blended with AS-OF with sample size weighting
    n_p = val_df["asof_pitcher_n"].fillna(0).to_numpy(float)
    gamma = n_p / (n_p + 25.0)
    p_composite = np.clip(gamma * p_asof + (1.0 - gamma) * p_eb, 0.01, 0.99)
    bss_comp, bs_comp = compute_bss(y_val, p_composite)
    print(f"\n[4] Hierarchical EB + AS-OF Dynamic Shrinkage:")
    print(f"    • Brier Score: {bs_comp:.6f} | BSS: {bss_comp:.4f} (Delta BSS: {bss_comp - bss_ref:+.4f})")

if __name__ == "__main__":
    main()
