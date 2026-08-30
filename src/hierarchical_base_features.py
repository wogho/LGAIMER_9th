"""Hierarchical Dynamic Shrinkage Base and Domain Features.

Re-engineered from 4th Reference Repo (LG-Aimers-9th, 1126.45 LB).
100% row-independent, leakage-free, official-data-only contract.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_hierarchical_base_features(df: pd.DataFrame, prior: float = 0.48785) -> pd.DataFrame:
    """Compute mathematical hierarchical shrinkage base and interaction features per row."""
    x = df.copy()

    # 1. Numerical extraction
    num = lambda col, d=np.nan: pd.to_numeric(x[col], errors="coerce").fillna(d).to_numpy(float)
    
    n_career = np.clip(num("asof_pitcher_n", 0), 0, None)
    career_rate = num("asof_pitcher_success_rate", prior)

    recent_cols = [f"asof_pitcher_prev{k}_game_success_rate" for k in (1, 3, 5)]
    rec_vals = [num(c, prior) for c in recent_cols]
    recent_matrix = np.vstack(rec_vals)
    recent_mean = np.nanmean(recent_matrix, axis=0)
    recent_std = np.nanstd(recent_matrix, axis=0)
    recent_std = np.nan_to_num(recent_std, nan=0.15)
    recent_std = np.clip(recent_std, 0.0, 0.5)

    # 2. Dynamic smoothing strength based on recent volatility and sample size
    dynamic_strength = np.clip(55.0 + 220.0 * recent_std + 40.0 / (1.0 + np.log1p(n_career)), 50.0, 180.0)
    career_base = (career_rate * n_career + prior * dynamic_strength) / (n_career + dynamic_strength)

    # 3. Season-to-date estimate
    # cur_succ / cur_logn_pitch if available, else career base
    if "cur_succ" in x.columns and "cur_logn_pitch" in x.columns:
        season_raw = num("cur_succ", prior)
        season_n = np.expm1(num("cur_logn_pitch", 0.0)).clip(0, None)
    else:
        season_raw = career_rate
        season_n = n_career

    season_strength = 30.0
    season_est = (season_raw * season_n + prior * season_strength) / (season_n + season_strength)
    season_reliability = season_n / (season_n + 80.0)
    season_weight = 0.15 + 0.30 * season_reliability
    hierarchical_base = career_base + season_weight * (season_est - career_base)

    # 4. Domain and psychological pressure indicators
    balls = num("balls_before", 0)
    strikes = num("strikes_before", 0)
    num_runners = num("num_runners_on", 0)
    li = np.clip(num("li", 0), 0, None)

    pressure_index = ((balls + 1.0) / (strikes + 1.0)) * (1.0 + num_runners) * np.log1p(li)
    pitcher_uncertainty = 1.0 / np.sqrt(n_career + 1.0)
    stable_momentum = (recent_mean - career_base) / (1.0 + 5.0 * recent_std)
    season_recent_gap = season_est - recent_mean

    # Pitch mix entropy
    rates = np.column_stack([
        num("asof_pitcher_fastball_rate", 0.33),
        num("asof_pitcher_breaking_rate", 0.33),
        num("asof_pitcher_offspeed_rate", 0.33)
    ])
    safe_rates = np.clip(rates, 1e-7, 1.0)
    pitchmix_entropy = -np.sum(safe_rates * np.log(safe_rates), axis=1)

    # 5. Append new engineered features
    out = pd.DataFrame(index=x.index)
    out["hierarchical_success_base"] = hierarchical_base.astype(np.float32)
    out["dynamic_smoothing_strength"] = dynamic_strength.astype(np.float32)
    out["career_dynamic_base"] = career_base.astype(np.float32)
    out["season_form_estimate"] = season_est.astype(np.float32)
    out["season_form_reliability"] = season_reliability.astype(np.float32)
    out["recent_success_std"] = recent_std.astype(np.float32)
    out["stable_momentum"] = stable_momentum.astype(np.float32)
    out["season_recent_gap"] = season_recent_gap.astype(np.float32)
    out["pressure_index"] = pressure_index.astype(np.float32)
    out["pitcher_uncertainty"] = pitcher_uncertainty.astype(np.float32)
    out["pitchmix_entropy"] = pitchmix_entropy.astype(np.float32)

    # Categorical contexts
    score_diff = num("score_diff_pitcher_team", 0)
    inning = num("inning", 1)
    
    cat_pressure = pd.cut(li, [-np.inf, 0.75, 1.5, 3.0, np.inf], labels=["low", "normal", "high", "extreme"]).astype(str)
    cat_inning = pd.cut(inning, [-np.inf, 3, 6, 9, np.inf], labels=["early", "middle", "late", "extra"]).astype(str)
    cat_score = pd.cut(score_diff, [-np.inf, -3, -1, 1, 3, np.inf], labels=["far_behind", "behind", "close", "ahead", "far_ahead"]).astype(str)

    p_hand = x["pitcher_hand"].astype(str)
    b_hand = x["batter_hand"].astype(str)
    b_state = x["base_state"].astype(str)
    c_state = x["balls_before"].astype(str) + "-" + x["strikes_before"].astype(str)
    h_matchup = p_hand + "-" + b_hand

    out["count_base_context"] = c_state + "|" + b_state
    out["hand_count_context"] = h_matchup + "|" + c_state
    out["inning_score_context"] = cat_inning + "|" + cat_score
    out["pressure_context"] = cat_pressure + "|" + num_runners.astype(int).astype(str) + "|" + c_state

    return out


HIERARCHICAL_FEATURE_NAMES = [
    "hierarchical_success_base",
    "dynamic_smoothing_strength",
    "career_dynamic_base",
    "season_form_estimate",
    "season_form_reliability",
    "recent_success_std",
    "stable_momentum",
    "season_recent_gap",
    "pressure_index",
    "pitcher_uncertainty",
    "pitchmix_entropy",
    "count_base_context",
    "hand_count_context",
    "inning_score_context",
    "pressure_context",
]
HIERARCHICAL_CAT_COLUMNS = [
    "count_base_context",
    "hand_count_context",
    "inning_score_context",
    "pressure_context",
]
