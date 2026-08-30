"""v5 Orthogonal Feature Extractor implementing row-independent team_asof and Bayesian shrinkage features."""
import json
from pathlib import Path
import numpy as np
import pandas as pd

def build_v5_orthogonal_features(df: pd.DataFrame, profile_path: Path | str | None = None, prior: float = 0.523766) -> pd.DataFrame:
    """Build team_asof, hand matching, and Bayesian shrinkage features cleanly row-independent."""
    out = pd.DataFrame(index=df.index)
    
    # 1. Hand matching (strictly row-local)
    out["is_same_hand"] = (df["pitcher_hand"] == df["batter_hand"]).astype(np.float32)
    
    # 2. Team-level As-of Success Rates & Matchup Gap (using pre-fitted snapshot map for 100% row independence)
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

    p_team = df["pitcher_team_id"].map(p_team_map).fillna(prior).astype(np.float32)
    b_team = df["batter_team_id"].map(b_team_map).fillna(prior).astype(np.float32)
    
    out["asof_pitcher_team_success_rate"] = p_team
    out["asof_batter_team_success_rate"] = b_team
    out["team_matchup_gap"] = (p_team - b_team).astype(np.float32)
    
    # 3. Hierarchical Bayesian Shrinkage Features (strictly row-local calculations)
    n_p = df["asof_pitcher_n"].fillna(0).astype(np.float32)
    p_rate = df["asof_pitcher_success_rate"].fillna(prior).astype(np.float32)
    prev1 = df["asof_pitcher_prev1_game_success_rate"].fillna(p_rate).astype(np.float32)
    prev5 = df["asof_pitcher_prev5_game_success_rate"].fillna(p_rate).astype(np.float32)
    
    # Reliability / smoothing strength
    reliability = n_p / (n_p + 50.0)
    out["season_pitcher_n"] = n_p
    out["season_success_reliability"] = reliability
    out["dynamic_smoothing_strength"] = 1.0 - reliability
    
    # Bayesian shrinkage base estimate
    career_base = reliability * p_rate + (1.0 - reliability) * prior
    out["career_dynamic_base"] = career_base.astype(np.float32)
    
    # Recent form and momentum
    out["season_form_estimate"] = prev1.astype(np.float32)
    out["season_recent_gap"] = (prev1 - p_rate).astype(np.float32)
    out["stable_momentum"] = (prev1 - prev5).astype(np.float32)
    out["recent_success_std"] = (df["asof_pitcher_middle_rate"].fillna(0) - df["asof_pitcher_reverse_rate"].fillna(0)).abs().astype(np.float32)
    
    # Logit domain representations
    p_clip = np.clip(p_rate, 1e-4, 1.0 - 1e-4)
    out["pitcher_success_logit"] = np.log(p_clip / (1.0 - p_clip)).astype(np.float32)
    
    c_clip = np.clip(career_base, 1e-4, 1.0 - 1e-4)
    out["season_success_logit"] = np.log(c_clip / (1.0 - c_clip)).astype(np.float32)
    
    return out
