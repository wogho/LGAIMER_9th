"""Shared row-local features and fixed hierarchical EB for Codex 110 R1."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


PRIOR = 0.523766
RATE_DEFAULTS = {
    "asof_pitcher_success_rate": PRIOR,
    "asof_pitcher_reverse_rate": 0.0,
    "asof_pitcher_middle_rate": 0.0,
    "asof_pitcher_ball_rate": 0.0,
    "asof_pitcher_strike_rate": 0.0,
    "asof_pitcher_prev1_game_success_rate": PRIOR,
    "asof_pitcher_prev3_game_success_rate": PRIOR,
    "asof_pitcher_prev5_game_success_rate": PRIOR,
    "asof_pitcher_prev1_game_middle_rate": 0.0,
    "asof_pitcher_prev3_game_middle_rate": 0.0,
    "asof_pitcher_prev5_game_middle_rate": 0.0,
    "asof_batter_success_rate": PRIOR,
    "asof_batter_middle_rate": 0.0,
    "asof_pitcher_fastball_rate": 0.0,
    "asof_pitcher_breaking_rate": 0.0,
    "asof_pitcher_offspeed_rate": 0.0,
}
RAW_NUMERIC = [
    "game_month", "game_dayofweek", "inning", "balls_before", "strikes_before",
    "outs_before", "run_top_before", "run_bot_before", "run_total_before",
    "score_diff_home", "score_diff_pitcher_team", "runner_on_1b", "runner_on_2b",
    "runner_on_3b", "num_runners_on", "home_win_expectancy", "away_win_expectancy",
    "li", "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id", "asof_pitcher_n", *RATE_DEFAULTS,
    "asof_batter_n", "asof_pitcher_pitchmix_n",
]
EB_COLUMNS = [
    "eb_pitcher", "eb_pitcher_hand", "eb_reliability", "eb_shrunk",
    "eb_vs_asof", "eb_hand_vs_pitcher", "eb_pitcher_weighted_n",
    "eb_hand_weighted_n",
]


def _numeric(df: pd.DataFrame, column: str, default: float = 0.0) -> np.ndarray:
    if column not in df.columns:
        return np.full(len(df), default, dtype=np.float64)
    return pd.to_numeric(df[column], errors="coerce").fillna(default).to_numpy(np.float64)


def build_residual_features(
    df: pd.DataFrame,
    anchor: np.ndarray,
    eb_features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build deterministic row-local numeric features; never aggregates *df*."""
    if len(df) != len(anchor):
        raise ValueError("anchor length mismatch")
    values: dict[str, np.ndarray] = {}
    for column in RAW_NUMERIC:
        default = RATE_DEFAULTS.get(column, 0.0)
        values[column] = _numeric(df, column, default)

    top = df["top_bottom"].astype(str).eq("T").to_numpy(np.float64) if "top_bottom" in df else np.zeros(len(df))
    futures = df["game_type"].astype(str).eq("F").to_numpy(np.float64) if "game_type" in df else np.zeros(len(df))
    values["is_top"] = top
    values["is_futures"] = futures

    balls = values["balls_before"]
    strikes = values["strikes_before"]
    inning = values["inning"]
    li = values["li"]
    score = np.abs(values["score_diff_pitcher_team"])
    n_pitcher = np.clip(values["asof_pitcher_n"], 0.0, None)
    n_batter = np.clip(values["asof_batter_n"], 0.0, None)
    n_mix = np.clip(values["asof_pitcher_pitchmix_n"], 0.0, None)
    p_rate = values["asof_pitcher_success_rate"]
    prev1 = values["asof_pitcher_prev1_game_success_rate"]
    prev3 = values["asof_pitcher_prev3_game_success_rate"]
    prev5 = values["asof_pitcher_prev5_game_success_rate"]
    b_rate = values["asof_batter_success_rate"]
    values.update({
        "count_diff": balls - strikes,
        "count_pressure": (balls - strikes) * (balls + strikes + 1.0) / 7.0,
        "is_two_strike": (strikes == 2).astype(np.float64),
        "is_three_ball": (balls == 3).astype(np.float64),
        "late_close": ((inning >= 7) & (score <= 2)).astype(np.float64),
        "li_count_diff": li * (balls - strikes),
        "log_pitcher_n": np.log1p(n_pitcher),
        "log_batter_n": np.log1p(n_batter),
        "log_pitchmix_n": np.log1p(n_mix),
        "hand_match": (values["pitcher_hand"] == values["batter_hand"]).astype(np.float64),
        "pitcher_recent1_drift": prev1 - p_rate,
        "pitcher_recent3_drift": prev3 - p_rate,
        "pitcher_recent5_drift": prev5 - p_rate,
        "recent_slope": prev1 - prev5,
        "pitcher_batter_gap": p_rate - b_rate,
        "anchor_p": np.asarray(anchor, dtype=np.float64),
        "anchor_logit": np.log(np.clip(anchor, 1e-5, 1 - 1e-5) / np.clip(1 - anchor, 1e-5, 1.0)),
    })
    out = pd.DataFrame(values, index=df.index).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if eb_features is not None:
        if len(eb_features) != len(out):
            raise ValueError("EB feature length mismatch")
        for column in EB_COLUMNS:
            out[column] = pd.to_numeric(eb_features[column], errors="coerce").fillna(0.0).to_numpy(float)
    return out.astype(np.float32)


def build_eb_tables(
    history: pd.DataFrame,
    global_weight: float = 100.0,
    hand_weight: float = 50.0,
    prediction_year: int | None = None,
    recency_weighted: bool = False,
) -> dict[str, Any]:
    y = pd.to_numeric(history["control_success"], errors="coerce")
    if prediction_year is not None and len(history):
        rates = history.loc[history["season"].lt(prediction_year)].groupby("season")["control_success"].mean().sort_index().tail(3)
        if len(rates) > 1:
            slope, intercept = np.polyfit(rates.index.to_numpy(float), rates.to_numpy(float), 1)
            global_prior = float(np.clip(intercept + slope * prediction_year, 0.42, 0.62))
        else:
            global_prior = float(rates.iloc[-1])
    else:
        global_prior = float(y.mean()) if len(history) else PRIOR
    work = history.assign(_pid=history["pitcher_id"].astype(str), _bhand=history["batter_hand"].astype(str), _y=y)
    if recency_weighted:
        if prediction_year is None:
            raise ValueError("prediction_year is required for recency weighting")
        age = prediction_year - pd.to_numeric(work["season"], errors="raise").astype(int)
        work["_w"] = np.select([age.eq(1), age.eq(2), age.eq(3)], [1.0, 0.7, 0.5], default=0.3)
    else:
        work["_w"] = 1.0
    work["_wy"] = work["_w"] * work["_y"]
    pitcher = work.groupby("_pid", sort=False).agg(wsum=("_wy", "sum"), wcount=("_w", "sum"))
    pitcher["posterior"] = (pitcher["wsum"] + global_prior * global_weight) / (pitcher["wcount"] + global_weight)
    pitcher_map = {str(key): float(value) for key, value in pitcher["posterior"].items()}
    pitcher_count_map = {str(key): float(value) for key, value in pitcher["wcount"].items()}
    pairs = work.groupby(["_pid", "_bhand"], sort=False).agg(wsum=("_wy", "sum"), wcount=("_w", "sum"))
    pair_map: dict[str, float] = {}
    pair_count_map: dict[str, float] = {}
    for (pid, hand), row in pairs.iterrows():
        parent = pitcher_map.get(pid, global_prior)
        key = f"{pid}_{hand}"
        pair_map[key] = float((float(row["wsum"]) + parent * hand_weight) / (float(row["wcount"]) + hand_weight))
        pair_count_map[key] = float(row["wcount"])
    return {
        "global_prior": global_prior,
        "global_weight": float(global_weight),
        "hand_weight": float(hand_weight),
        "pitcher": pitcher_map,
        "pitcher_weighted_n": pitcher_count_map,
        "pitcher_hand": pair_map,
        "pitcher_hand_weighted_n": pair_count_map,
        "prediction_year": prediction_year,
        "recency_weighted": bool(recency_weighted),
    }


def apply_eb_tables(df: pd.DataFrame, tables: dict[str, Any], reliability_k: float = 25.0) -> pd.DataFrame:
    pid = df["pitcher_id"].astype(str)
    hand = df["batter_hand"].astype(str)
    global_prior = float(tables["global_prior"])
    p_pitcher = pid.map(tables["pitcher"]).fillna(global_prior).to_numpy(float)
    p_hand = (pid + "_" + hand).map(tables["pitcher_hand"]).fillna(pd.Series(p_pitcher, index=df.index)).to_numpy(float)
    pitcher_weighted_n = pid.map(tables.get("pitcher_weighted_n", {})).fillna(0.0).to_numpy(float)
    hand_weighted_n = (pid + "_" + hand).map(tables.get("pitcher_hand_weighted_n", {})).fillna(0.0).to_numpy(float)
    n = np.clip(_numeric(df, "asof_pitcher_n", 0.0), 0.0, None)
    p_asof = _numeric(df, "asof_pitcher_success_rate", PRIOR)
    reliability = n / (n + float(reliability_k))
    p_shrunk = reliability * p_asof + (1.0 - reliability) * p_hand
    return pd.DataFrame({
        "eb_pitcher": p_pitcher,
        "eb_pitcher_hand": p_hand,
        "eb_reliability": reliability,
        "eb_shrunk": p_shrunk,
        "eb_vs_asof": p_hand - p_asof,
        "eb_hand_vs_pitcher": p_hand - p_pitcher,
        "eb_pitcher_weighted_n": pitcher_weighted_n,
        "eb_hand_weighted_n": hand_weighted_n,
    }, index=df.index)


def router_features(df: pd.DataFrame, experts: np.ndarray) -> pd.DataFrame:
    if experts.shape != (len(df), 3):
        raise ValueError("router experts must have shape (n, 3)")
    e0, e1, e2 = experts.T
    base = build_residual_features(df, e2)
    keep = [
        "game_month", "inning", "balls_before", "strikes_before", "outs_before",
        "num_runners_on", "score_diff_pitcher_team", "li", "pitcher_hand",
        "batter_hand", "asof_pitcher_n", "asof_batter_n", "is_futures",
        "count_pressure", "log_pitcher_n", "log_batter_n", "hand_match",
        "pitcher_recent1_drift", "pitcher_batter_gap",
    ]
    out = base[keep].copy()
    out["expert_107"] = e0
    out["expert_108"] = e1
    out["expert_109"] = e2
    out["expert_108_minus_107"] = e1 - e0
    out["expert_109_minus_108"] = e2 - e1
    out["expert_std"] = experts.std(axis=1)
    out["expert_range"] = experts.max(axis=1) - experts.min(axis=1)
    return out.astype(np.float32)
