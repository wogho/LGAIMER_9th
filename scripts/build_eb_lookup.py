#!/usr/bin/env python3
"""
Computes exact Hierarchical Empirical Bayes Pitcher-Hand & Count Tables.
- Train data: 2019-2023 (strictly past) or full train for production
- Outputs static lookup dictionary for O(1) production inference
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def build_hierarchical_eb_tables(df: pd.DataFrame, prior_weight_global: float = 100.0, prior_weight_hand: float = 50.0) -> dict:
    global_mean = float(df["control_success"].mean())
    alpha_0 = global_mean * prior_weight_global
    beta_0 = (1.0 - global_mean) * prior_weight_global
    
    # 1. Pitcher baseline level
    p_grp = df.groupby("pitcher_id")["control_success"].agg(["sum", "count"]).reset_index()
    p_grp["p_baseline"] = (p_grp["sum"] + alpha_0) / (p_grp["count"] + prior_weight_global)
    p_map = dict(zip(p_grp["pitcher_id"].astype(str), p_grp["p_baseline"].astype(float)))
    
    # 2. Pitcher x Hand Matchup level
    df_copy = df.copy()
    df_copy["pitcher_id_str"] = df_copy["pitcher_id"].astype(str)
    df_copy["batter_hand_str"] = df_copy["batter_hand"].astype(str)
    
    ph_grp = df_copy.groupby(["pitcher_id_str", "batter_hand_str"])["control_success"].agg(["sum", "count"]).reset_index()
    
    eb_hand_lookup = {}
    for _, row in ph_grp.iterrows():
        pid = row["pitcher_id_str"]
        bhand = row["batter_hand_str"]
        s_ih = row["sum"]
        n_ih = row["count"]
        
        p_base = p_map.get(pid, global_mean)
        alpha_ih = prior_weight_hand * p_base + s_ih
        beta_ih = prior_weight_hand * (1.0 - p_base) + (n_ih - s_ih)
        p_ih = float(alpha_ih / (n_ih + prior_weight_hand))
        
        eb_hand_lookup[f"{pid}_{bhand}"] = round(p_ih, 6)
        
    return {
        "global_prior": round(global_mean, 6),
        "pitcher_baseline": {k: round(v, 6) for k, v in p_map.items()},
        "pitcher_hand_lookup": eb_hand_lookup
    }

if __name__ == "__main__":
    train = pd.read_csv(ROOT / "data/train.csv", low_memory=False)
    eb_tables = build_hierarchical_eb_tables(train)
    print(f"Global prior: {eb_tables['global_prior']}")
    print(f"Total pitchers: {len(eb_tables['pitcher_baseline'])}")
    print(f"Total pitcher-hand pairs: {len(eb_tables['pitcher_hand_lookup'])}")
