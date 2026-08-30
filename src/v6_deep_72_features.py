"""v6 Deep 72 Feature and Hierarchical Base Extractor."""
from pathlib import Path
import json
import numpy as np
import pandas as pd

def build_v6_deep_72_features(df: pd.DataFrame, profile_path: Path | str | None = None, prior: float = 0.523766) -> tuple[pd.DataFrame, np.ndarray]:
    """Build 72 full features and compute hierarchical_base cleanly and row-independently."""
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

    # 5. NEW v6 Advanced Orthogonal Features
    n_b = df["asof_batter_n"].fillna(0).astype(np.float32) if "asof_batter_n" in df.columns else pd.Series(0.0, index=df.index)
    b_rate = df["asof_batter_success_rate"].fillna(prior).astype(np.float32) if "asof_batter_success_rate" in df.columns else pd.Series(prior, index=df.index)
    
    out["pitcher_batter_matchup_n"] = (n_p * n_b / (n_p + n_b + 10.0)).astype(np.float32)
    
    p_mid = df["asof_pitcher_middle_rate"].fillna(0).astype(np.float32) if "asof_pitcher_middle_rate" in df.columns else pd.Series(0.0, index=df.index)
    p_rev = df["asof_pitcher_reverse_rate"].fillna(0).astype(np.float32) if "asof_pitcher_reverse_rate" in df.columns else pd.Series(0.0, index=df.index)
    p_wld = df["asof_pitcher_wild_rate"].fillna(0).astype(np.float32) if "asof_pitcher_wild_rate" in df.columns else pd.Series(0.0, index=df.index)
    out["asof_pitcher_strike_pressure"] = ((p_mid + 0.5 * p_rev) / (p_mid + p_wld + p_rev + 1e-4)).astype(np.float32)
    
    b_mid = df["asof_batter_middle_rate"].fillna(0).astype(np.float32) if "asof_batter_middle_rate" in df.columns else pd.Series(0.0, index=df.index)
    b_rev = df["asof_batter_reverse_rate"].fillna(0).astype(np.float32) if "asof_batter_reverse_rate" in df.columns else pd.Series(0.0, index=df.index)
    b_wld = df["asof_batter_wild_rate"].fillna(0).astype(np.float32) if "asof_batter_wild_rate" in df.columns else pd.Series(0.0, index=df.index)
    out["asof_batter_chase_vulnerability"] = ((b_wld + b_rev) / (b_mid + b_wld + b_rev + 1e-4)).astype(np.float32)
    
    out["pitch_control_decay_ratio"] = (prev1 / (p_rate + 1e-4)).astype(np.float32)
    out["season_trajectory_momentum"] = (prev1 - 2.0 * prev5 + p_rate).astype(np.float32)
    
    p_team_logit = np.log(np.clip(p_team, 1e-4, 1 - 1e-4) / (1 - np.clip(p_team, 1e-4, 1 - 1e-4)))
    b_team_logit = np.log(np.clip(b_team, 1e-4, 1 - 1e-4) / (1 - np.clip(b_team, 1e-4, 1 - 1e-4)))
    out["team_home_away_proxy"] = (p_team_logit - b_team_logit).astype(np.float32)
    
    out["pitcher_reliability_sqrt"] = np.sqrt(reliability).astype(np.float32)
    b_rel = n_b / (n_b + 50.0)
    out["batter_reliability_sqrt"] = np.sqrt(b_rel).astype(np.float32)
    
    p_smoothed = (n_p * p_rate + 25.0 * p_team) / (n_p + 25.0)
    b_smoothed = (n_b * b_rate + 25.0 * b_team) / (n_b + 25.0)
    out["matchup_bayes_smoothed"] = p_smoothed.astype(np.float32)
    out["batter_bayes_smoothed"] = b_smoothed.astype(np.float32)
    out["net_bayes_advantage"] = (p_smoothed - b_smoothed).astype(np.float32)

    # 6. Additional interaction cross-products to complete feature vector cleanly
    out["p_b_rate_ratio"] = (p_rate / (b_rate + 1e-4)).astype(np.float32)
    out["p_b_n_sum"] = (n_p + n_b).astype(np.float32)
    out["p_team_diff"] = (p_rate - p_team).astype(np.float32)
    out["b_team_diff"] = (b_rate - b_team).astype(np.float32)

    # Compute hierarchical_base cleanly
    hierarchical_base = np.clip(0.70 * career_base + 0.30 * prev1, 0.05, 0.95).to_numpy(float)
    
    return out, hierarchical_base
