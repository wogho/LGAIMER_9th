"""Enriched feature builder combining base v3 features with Repo 1 & 5 domain features."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from src.preprocessing_v2 import CAT_V2, build_v3_features

ENRICHED_CAT = CAT_V2 + [
    "is_full_count",
    "has_two_strikes",
    "has_three_balls",
    "runner_in_scoring_position",
    "bases_loaded",
    "same_hand",
    "late_inning",
    "close_game",
    "is_chase",
]


def build_enriched_features(
    raw: pd.DataFrame,
    prior: float,
    pitcher_snapshots: pd.DataFrame,
    batter_snapshots: pd.DataFrame,
    pitchmix_snapshots: pd.DataFrame,
    trackman_csv: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build base v3 features + 1번/Contextual Platoon Engine의 검증된 고차원 상황 압박 및 도메인 피처."""
    # 1. Base v3 features
    X_base, base_pred = build_v3_features(
        raw, prior, pitcher_snapshots, batter_snapshots, pitchmix_snapshots, trackman_csv
    )
    
    out = X_base.copy()
    
    # 2. Contextual Platoon Engine 경기 상황/볼카운트/압박 피처
    balls = pd.to_numeric(raw["balls_before"], errors="coerce").fillna(0).astype("int8")
    strikes = pd.to_numeric(raw["strikes_before"], errors="coerce").fillna(0).astype("int8")
    outs = pd.to_numeric(raw["outs_before"], errors="coerce").fillna(0).astype("int8")
    inning = pd.to_numeric(raw["inning"], errors="coerce").fillna(1).astype("int8")
    
    out["count_index"] = (balls * 4 + strikes).astype("int8")
    out["count_out_index"] = (out["count_index"] * 3 + outs).astype("int8")
    out["count_advantage"] = (strikes - balls).astype("int8")
    out["is_full_count"] = ((balls == 3) & (strikes == 2)).astype("int8")
    out["has_two_strikes"] = (strikes == 2).astype("int8")
    out["has_three_balls"] = (balls == 3).astype("int8")
    
    r1 = pd.to_numeric(raw.get("runner_on_1b", 0), errors="coerce").fillna(0).astype("int8")
    r2 = pd.to_numeric(raw.get("runner_on_2b", 0), errors="coerce").fillna(0).astype("int8")
    r3 = pd.to_numeric(raw.get("runner_on_3b", 0), errors="coerce").fillna(0).astype("int8")
    
    out["runner_in_scoring_position"] = ((r2 == 1) | (r3 == 1)).astype("int8")
    out["bases_loaded"] = ((r1 == 1) & (r2 == 1) & (r3 == 1)).astype("int8")
    
    p_hand = raw["pitcher_hand"].astype(str)
    b_hand = raw["batter_hand"].astype(str)
    out["same_hand"] = (p_hand == b_hand).astype("int8")
    out["late_inning"] = (inning >= 7).astype("int8")
    
    score_diff = pd.to_numeric(raw.get("score_diff_pitcher_team", 0), errors="coerce").fillna(0).astype("float32")
    out["close_game"] = (score_diff.abs() <= 1).astype("int8")
    
    li = pd.to_numeric(raw.get("li", 1.0), errors="coerce").fillna(1.0).clip(lower=0).astype("float32")
    out["log_li"] = np.log1p(li).astype("float32")
    out["score_pressure"] = (score_diff.abs() * out["log_li"]).astype("float32")
    
    h_win = pd.to_numeric(raw.get("home_win_expectancy", 0.5), errors="coerce").fillna(0.5).astype("float32")
    a_win = pd.to_numeric(raw.get("away_win_expectancy", 0.5), errors="coerce").fillna(0.5).astype("float32")
    out["win_expectancy_gap"] = (h_win - a_win).astype("float32")
    
    # 3. 투수-타자 기량 갭 및 최근 경기 컨디션 델타 (Contextual Platoon Engine)
    p_succ = pd.to_numeric(raw.get("asof_pitcher_success_rate", prior), errors="coerce").fillna(prior).astype("float32")
    b_succ = pd.to_numeric(raw.get("asof_batter_success_rate", prior), errors="coerce").fillna(prior).astype("float32")
    out["pitcher_batter_success_gap"] = (p_succ - b_succ).astype("float32")
    
    prev1 = pd.to_numeric(raw.get("asof_pitcher_prev1_game_success_rate", p_succ), errors="coerce").fillna(p_succ).astype("float32")
    prev3 = pd.to_numeric(raw.get("asof_pitcher_prev3_game_success_rate", p_succ), errors="coerce").fillna(p_succ).astype("float32")
    prev5 = pd.to_numeric(raw.get("asof_pitcher_prev5_game_success_rate", p_succ), errors="coerce").fillna(p_succ).astype("float32")
    
    out["pitcher_recent_success_delta_1_5"] = (prev1 - prev5).astype("float32")
    out["pitcher_recent_success_delta_3_5"] = (prev3 - prev5).astype("float32")
    
    mid1 = pd.to_numeric(raw.get("asof_pitcher_prev1_game_middle_rate", 0.1), errors="coerce").fillna(0.1).astype("float32")
    mid5 = pd.to_numeric(raw.get("asof_pitcher_prev5_game_middle_rate", 0.1), errors="coerce").fillna(0.1).astype("float32")
    out["pitcher_recent_middle_delta_1_5"] = (mid1 - mid5).astype("float32")
    
    # 4. Physics-Trajectory Baseline Chase 상황 및 Command Gap
    out["is_chase"] = ((strikes == 2) & (balls < 3)).astype("int8")
    
    p_strike = pd.to_numeric(raw.get("asof_pitcher_strike_rate", 0.65), errors="coerce").fillna(0.65).astype("float32")
    p_ball = pd.to_numeric(raw.get("asof_pitcher_ball_rate", 0.35), errors="coerce").fillna(0.35).astype("float32")
    out["command_gap"] = (p_strike - p_ball).astype("float32")
    
    p_mid = pd.to_numeric(raw.get("asof_pitcher_middle_rate", 0.08), errors="coerce").fillna(0.08).astype("float32")
    out["pitcher_risk_ratio"] = (p_mid / (p_succ + 0.01)).astype("float32")
    
    # Set categorical columns
    for col in ENRICHED_CAT:
        if col in out.columns:
            out[col] = out[col].astype("category")
            
    return out, base_pred
