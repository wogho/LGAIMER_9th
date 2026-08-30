"""Independent 57-column feature contract for REF-REBUILD-001.

Implemented from the public data description, without importing reference code
or model artifacts. Every transformation is row-local; global_mean is supplied
from the training partition and persisted by the caller.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"
CAT_COLS = [
    "top_bottom", "game_type", "base_state", "pitcher_hand",
    "batter_hand", "pitcher_team_id", "batter_team_id", "count_state",
]

BASE_COLUMNS = [
    "season", "game_month", "game_dayofweek", "inning", "top_bottom", "game_type",
    "balls_before", "strikes_before", "outs_before", "run_top_before", "run_bot_before",
    "run_total_before", "score_diff_home", "score_diff_pitcher_team", "runner_on_1b",
    "runner_on_2b", "runner_on_3b", "num_runners_on", "base_state", "home_win_expectancy",
    "away_win_expectancy", "li", "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id", "asof_pitcher_n", "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate", "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate", "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate", "asof_batter_n", "asof_batter_success_rate",
    "asof_batter_middle_rate", "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",
]
DERIVED_COLUMNS = [
    "smoothed_pitcher_success_rate", "smoothed_batter_success_rate", "platoon_advantage",
    "count_advantage", "count_state", "recent_control_momentum", "form_trend_5_1",
    "is_home", "pitcher_win_expectancy", "is_coldstart_pitcher",
]
FEATURE_COLUMNS = BASE_COLUMNS + DERIVED_COLUMNS


def engineer(frame: pd.DataFrame, global_mean: float, smoothing: float = 30.0) -> pd.DataFrame:
    missing = sorted(set(BASE_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"missing base columns: {missing}")
    d = frame.loc[:, BASE_COLUMNS].copy()
    for who in ("pitcher", "batter"):
        n = d[f"asof_{who}_n"].fillna(0.0)
        rate = d[f"asof_{who}_success_rate"].fillna(global_mean)
        d[f"smoothed_{who}_success_rate"] = (n * rate + smoothing * global_mean) / (n + smoothing)
    d["platoon_advantage"] = (d["pitcher_hand"].astype("string") == d["batter_hand"].astype("string")).astype("int8")
    d["count_advantage"] = d["strikes_before"] - d["balls_before"]
    d["count_state"] = d["balls_before"].astype("string") + "-" + d["strikes_before"].astype("string")
    d["recent_control_momentum"] = d["asof_pitcher_prev1_game_success_rate"] - d["asof_pitcher_success_rate"]
    d["form_trend_5_1"] = d["asof_pitcher_prev1_game_success_rate"] - d["asof_pitcher_prev5_game_success_rate"]
    d["is_home"] = d["top_bottom"].astype("string").eq("T").astype("int8")
    d["pitcher_win_expectancy"] = np.where(d["is_home"].eq(1), d["home_win_expectancy"], d["away_win_expectancy"])
    d["is_coldstart_pitcher"] = d["asof_pitcher_n"].isna().astype("int8")
    if list(d.columns) != FEATURE_COLUMNS:
        raise AssertionError(f"feature order mismatch: {len(d.columns)}")
    return d


def prepare(frame: pd.DataFrame) -> pd.DataFrame:
    d = frame.copy()
    for col in CAT_COLS:
        d[col] = d[col].astype("string").fillna("<NA>").astype(str)
    return d
