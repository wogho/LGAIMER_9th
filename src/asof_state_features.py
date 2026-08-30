"""Independent AS-OF cumulative-to-current-season state reconstruction."""
from __future__ import annotations
import numpy as np
import pandas as pd

RATE_SPEC = [
    ("pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate", "cur_succ"),
    ("pitcher_id", "asof_pitcher_n", "asof_pitcher_middle_rate", "cur_mid"),
    ("pitcher_id", "asof_pitcher_n", "asof_pitcher_ball_rate", "cur_ball"),
    ("pitcher_id", "asof_pitcher_n", "asof_pitcher_reverse_rate", "cur_rev"),
    ("pitcher_id", "asof_pitcher_n", "asof_pitcher_strike_rate", "cur_str"),
    ("batter_id", "asof_batter_n", "asof_batter_success_rate", "cur_bsucc"),
    ("batter_id", "asof_batter_n", "asof_batter_middle_rate", "cur_bmid"),
]
STATE_COLUMNS = ["cur_succ", "cur_mid", "cur_ball", "cur_rev", "cur_str", "cur_bsucc", "cur_bmid", "cur_logn_pitch", "cur_logn_mix", "cur_logn_bat"]

def _prior(history: pd.DataFrame, id_col: str, n_col: str, rate_cols: list[str]) -> pd.DataFrame:
    d = history[[id_col, n_col, *rate_cols]].copy()
    d["_n"] = pd.to_numeric(d[n_col], errors="coerce").fillna(0.0)
    out = d.groupby(id_col, sort=False)["_n"].max().to_frame("prior_n")
    for c in rate_cols:
        out[f"prior_{c}"] = (d["_n"] * pd.to_numeric(d[c], errors="coerce").fillna(0.0)).groupby(d[id_col], sort=False).max()
    return out

def add_state_for_cutoff(rows: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Add current-season state using only history strictly before row season."""
    out = rows.copy()
    pitch_rates=[x[2] for x in RATE_SPEC if x[0]=="pitcher_id"]
    bat_rates=[x[2] for x in RATE_SPEC if x[0]=="batter_id"]
    pp=_prior(history,"pitcher_id","asof_pitcher_n",pitch_rates)
    bp=_prior(history,"batter_id","asof_batter_n",bat_rates)
    pid=out.pitcher_id; bid=out.batter_id
    pn=pd.to_numeric(out.asof_pitcher_n,errors="coerce").fillna(0.0); bn=pd.to_numeric(out.asof_batter_n,errors="coerce").fillna(0.0)
    pprior=pid.map(pp.prior_n).fillna(0.0); bprior=bid.map(bp.prior_n).fillna(0.0)
    pc=(pn-pprior).clip(lower=0.0); bc=(bn-bprior).clip(lower=0.0)
    for id_col, _, rc, lb in RATE_SPEC:
        n, cur, tab = (pn,pc,pp) if id_col == "pitcher_id" else (bn,bc,bp)
        prior_events=pid.map(tab[f"prior_{rc}"]).fillna(0.0) if id_col == "pitcher_id" else bid.map(tab[f"prior_{rc}"]).fillna(0.0)
        total=n*pd.to_numeric(out[rc],errors="coerce").fillna(0.0)
        out[lb]=((total-prior_events)/cur).where(cur>0)
    out["cur_logn_pitch"]=np.log1p(pc); out["cur_logn_mix"]=np.log1p((pd.to_numeric(out.asof_pitcher_pitchmix_n,errors="coerce").fillna(0.0)-pid.map(_prior(history,"pitcher_id","asof_pitcher_pitchmix_n",["asof_pitcher_fastball_rate"]).prior_n).fillna(0.0)).clip(lower=0.0)); out["cur_logn_bat"]=np.log1p(bc)
    return out

def add_state_walkforward(frame: pd.DataFrame, target_year: int) -> pd.DataFrame:
    """Build train/validation features with a strict season cutoff per row."""
    parts=[]
    for season, g in frame.groupby("season", sort=True):
        if int(season) >= target_year: continue
        parts.append(add_state_for_cutoff(g, frame[frame.season < season]))
    valid=frame[frame.season == target_year]
    parts.append(add_state_for_cutoff(valid, frame[frame.season < target_year]))
    return pd.concat(parts).sort_index()
