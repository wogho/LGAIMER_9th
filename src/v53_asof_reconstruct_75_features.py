"""v5.3 Active Season Reconstruction 75-Feature Extractor.

Reconstructs in-season active states (cur_n, cur_pitcher_rate, cur_batter_rate, cur_form_trend, dx_succ_sh, dx_succ_bs)
by subtracting pre-calculated career priors from cumulative as-of features.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

def build_priors_from_train(train_df: pd.DataFrame) -> dict:
    """Pre-calculate career priors (n and event counts) for pitchers and batters prior to 2025 season."""
    priors = {"pitcher": {}, "batter": {}}
    
    if "pitcher_id" in train_df.columns and "asof_pitcher_n" in train_df.columns:
        p_grp = train_df.groupby("pitcher_id")
        p_n = p_grp["asof_pitcher_n"].max()
        p_rate = p_grp["asof_pitcher_success_rate"].last().fillna(0.523766)
        p_events = p_n * p_rate
        for pid in p_n.index:
            priors["pitcher"][int(pid)] = (float(p_n[pid]), float(p_events[pid]))
            
    if "batter_id" in train_df.columns and "asof_batter_n" in train_df.columns:
        b_grp = train_df.groupby("batter_id")
        b_n = b_grp["asof_batter_n"].max()
        b_rate = b_grp["asof_batter_success_rate"].last().fillna(0.523766)
        b_events = b_n * b_rate
        for bid in b_n.index:
            priors["batter"][int(bid)] = (float(b_n[bid]), float(b_events[bid]))
            
    return priors

def build_v53_asof_reconstruct_75_features(
    df: pd.DataFrame,
    profile_path: Path | str | None = None,
    priors: dict | None = None,
    prior: float = 0.523766
) -> tuple[pd.DataFrame, np.ndarray]:
    """Build 75 features including Active Season Reconstructed States cleanly and row-independently."""
    out = pd.DataFrame(index=df.index)
    
    # 1. Base numeric features from input df (14 features)
    base_cols = [
        "asof_pitcher_n", "asof_pitcher_success_rate",
        "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_middle_rate", "asof_pitcher_reverse_rate", "asof_pitcher_wild_rate",
        "asof_batter_n", "asof_batter_success_rate",
        "asof_batter_prev1_game_success_rate", "asof_batter_prev5_game_success_rate",
        "asof_batter_middle_rate", "asof_batter_reverse_rate", "asof_batter_wild_rate"
    ]
    for col in base_cols:
        if col in df.columns:
            out[col] = df[col].fillna(prior if "rate" in col else 0).astype(np.float32)
            
    # 2. Hand matching (1 feature)
    if "pitcher_hand" in df.columns and "batter_hand" in df.columns:
        out["is_same_hand"] = (df["pitcher_hand"] == df["batter_hand"]).astype(np.float32)
    else:
        out["is_same_hand"] = 0.0

    # 3. Team-level As-of Success Rates & Matchup Gap (3 features)
    p_team_map = {}
    b_team_map = {}
    if profile_path and Path(profile_path).exists():
        try:
            prof = json.loads(Path(profile_path).read_text(encoding="utf-8"))
            p_team_map = prof.get("p_team", {})
            b_team_map = prof.get("b_team", {})
            prior = float(prof.get("prior", prior))
        except Exception:
            pass

    if "pitcher_team_id" in df.columns:
        p_team = df["pitcher_team_id"].map(p_team_map).fillna(prior).astype(np.float32)
    else:
        p_team = pd.Series(prior, index=df.index, dtype=np.float32)
        
    if "batter_team_id" in df.columns:
        b_team = df["batter_team_id"].map(b_team_map).fillna(prior).astype(np.float32)
    else:
        b_team = pd.Series(prior, index=df.index, dtype=np.float32)
        
    out["asof_pitcher_team_success_rate"] = p_team
    out["asof_batter_team_success_rate"] = b_team
    out["team_matchup_gap"] = (p_team - b_team).astype(np.float32)

    # 4. Hierarchical Bayesian Shrinkage Features (10 features)
    n_p = df["asof_pitcher_n"].fillna(0).astype(np.float32) if "asof_pitcher_n" in df.columns else pd.Series(0.0, index=df.index)
    p_rate = df["asof_pitcher_success_rate"].fillna(prior).astype(np.float32) if "asof_pitcher_success_rate" in df.columns else pd.Series(prior, index=df.index)
    prev1 = df["asof_pitcher_prev1_game_success_rate"].fillna(p_rate).astype(np.float32) if "asof_pitcher_prev1_game_success_rate" in df.columns else p_rate
    prev5 = df["asof_pitcher_prev5_game_success_rate"].fillna(p_rate).astype(np.float32) if "asof_pitcher_prev5_game_success_rate" in df.columns else p_rate
    
    reliability = n_p / (n_p + 50.0)
    out["season_pitcher_n"] = n_p
    out["season_success_reliability"] = reliability
    out["dynamic_smoothing_strength"] = 1.0 - reliability
    
    career_base = reliability * p_rate + (1.0 - reliability) * prior
    out["career_dynamic_base"] = career_base.astype(np.float32)
    
    out["season_form_estimate"] = prev1.astype(np.float32)
    out["season_recent_gap"] = (prev1 - p_rate).astype(np.float32)
    out["stable_momentum"] = (prev1 - prev5).astype(np.float32)
    
    if "asof_pitcher_middle_rate" in df.columns and "asof_pitcher_reverse_rate" in df.columns:
        out["recent_success_std"] = (df["asof_pitcher_middle_rate"].fillna(0) - df["asof_pitcher_reverse_rate"].fillna(0)).abs().astype(np.float32)
    else:
        out["recent_success_std"] = 0.0

    p_clip = np.clip(p_rate, 1e-4, 1.0 - 1e-4)
    out["pitcher_success_logit"] = np.log(p_clip / (1.0 - p_clip)).astype(np.float32)
    
    c_clip = np.clip(career_base, 1e-4, 1.0 - 1e-4)
    out["season_success_logit"] = np.log(c_clip / (1.0 - c_clip)).astype(np.float32)

    # 5. Advanced Bayesian & Matchup Advantage Features (6 features)
    n_b = df["asof_batter_n"].fillna(0).astype(np.float32) if "asof_batter_n" in df.columns else pd.Series(0.0, index=df.index)
    b_rate = df["asof_batter_success_rate"].fillna(prior).astype(np.float32) if "asof_batter_success_rate" in df.columns else pd.Series(prior, index=df.index)
    b_prev1 = df["asof_batter_prev1_game_success_rate"].fillna(b_rate).astype(np.float32) if "asof_batter_prev1_game_success_rate" in df.columns else b_rate
    
    out["pitcher_batter_matchup_n"] = (n_p * n_b / (n_p + n_b + 10.0)).astype(np.float32)
    
    p_smoothed = (n_p * p_rate + 25.0 * p_team) / (n_p + 25.0)
    b_smoothed = (n_b * b_rate + 25.0 * b_team) / (n_b + 25.0)
    out["matchup_bayes_smoothed"] = p_smoothed.astype(np.float32)
    out["batter_bayes_smoothed"] = b_smoothed.astype(np.float32)
    out["net_bayes_advantage"] = (p_smoothed - b_smoothed).astype(np.float32)

    out["batter_pitcher_form_gap"] = (prev1 - b_prev1).astype(np.float32)
    out["career_form_gap"] = (p_rate - b_rate).astype(np.float32)

    # 6. ACTIVE SEASON RECONSTRUCTED STATES (8 features -> Total 75 features)
    p_priors = priors.get("pitcher", {}) if priors else {}
    b_priors = priors.get("batter", {}) if priors else {}
    
    if "pitcher_id" in df.columns:
        p_ids = df["pitcher_id"].fillna(-1).astype(int).to_numpy()
        p_prior_n = np.array([p_priors.get(pid, (0.0, 0.0))[0] for pid in p_ids], dtype=np.float32)
        p_prior_ev = np.array([p_priors.get(pid, (0.0, 0.0))[1] for pid in p_ids], dtype=np.float32)
    else:
        p_prior_n = np.zeros(len(df), dtype=np.float32)
        p_prior_ev = np.zeros(len(df), dtype=np.float32)
        
    if "batter_id" in df.columns:
        b_ids = df["batter_id"].fillna(-1).astype(int).to_numpy()
        b_prior_n = np.array([b_priors.get(bid, (0.0, 0.0))[0] for bid in b_ids], dtype=np.float32)
        b_prior_ev = np.array([b_priors.get(bid, (0.0, 0.0))[1] for bid in b_ids], dtype=np.float32)
    else:
        b_prior_n = np.zeros(len(df), dtype=np.float32)
        b_prior_ev = np.zeros(len(df), dtype=np.float32)

    cur_p_n = np.maximum(0.0, n_p - p_prior_n)
    cur_p_ev = (n_p * p_rate) - p_prior_ev
    cur_p_rate = np.where(cur_p_n > 0, cur_p_ev / np.maximum(1.0, cur_p_n), p_rate)
    
    cur_b_n = np.maximum(0.0, n_b - b_prior_n)
    cur_b_ev = (n_b * b_rate) - b_prior_ev
    cur_b_rate = np.where(cur_b_n > 0, cur_b_ev / np.maximum(1.0, cur_b_n), b_rate)
    
    out["cur_logn_pitcher"] = np.log1p(cur_p_n).astype(np.float32)
    out["cur_pitcher_success_rate"] = cur_p_rate.astype(np.float32)
    out["cur_logn_batter"] = np.log1p(cur_b_n).astype(np.float32)
    out["cur_batter_success_rate"] = cur_b_rate.astype(np.float32)
    
    out["cur_form_trend"] = (prev1 - cur_p_rate).astype(np.float32)
    out["cur_matchup_advantage"] = (cur_p_rate - cur_b_rate).astype(np.float32)
    
    sh = out["is_same_hand"].to_numpy()
    st = df["strikes_before"].fillna(0).astype(np.float32) if "strikes_before" in df.columns else np.zeros(len(df))
    bl = df["balls_before"].fillna(0).astype(np.float32) if "balls_before" in df.columns else np.zeros(len(df))
    bs = (bl - st).to_numpy()
    
    out["dx_succ_sh"] = (cur_p_rate * sh).astype(np.float32)
    out["dx_succ_bs"] = (cur_p_rate * bs).astype(np.float32)

    # Compute hierarchical_base cleanly
    hierarchical_base = np.clip(0.70 * career_base + 0.30 * prev1, 0.05, 0.95).to_numpy(float)
    
    return out, hierarchical_base
