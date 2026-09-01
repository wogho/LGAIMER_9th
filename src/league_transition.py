from __future__ import annotations
import pickle
import numpy as np,pandas as pd

CAT=["game_type","prior_type","transition","count","hand","team_type"]

def transition_features(rows,prediction,lookup_path):
    try:
        with open(lookup_path, "rb") as f:
            lookup = pickle.load(f)
    except Exception:
        csv_p = str(lookup_path).replace(".pkl", ".csv")
        df_p = pd.read_csv(csv_p)
        lookup = dict(zip(df_p.pitcher_id.astype(str), df_p.prior_type.astype(str)))
    pid=rows.pitcher_id.astype(str)
    prev=pid.map(lookup).fillna("NEW").astype(str);current=rows.game_type.astype(str)
    x=pd.DataFrame({
        "game_type":current,"prior_type":prev,"transition":prev+">"+current,
        "count":rows.balls_before.astype(str)+"-"+rows.strikes_before.astype(str),
        "hand":rows.pitcher_hand.astype(str)+"-"+rows.batter_hand.astype(str),
        "team_type":rows.pitcher_team_id.astype(str)+"|"+current,
        "base_prediction":prediction,
        "log_pitcher_n":np.log1p(pd.to_numeric(rows.asof_pitcher_n,errors="coerce").fillna(0)),
        "career":pd.to_numeric(rows.asof_pitcher_success_rate,errors="coerce"),
        "recent1":pd.to_numeric(rows.asof_pitcher_prev1_game_success_rate,errors="coerce"),
        "recent3":pd.to_numeric(rows.asof_pitcher_prev3_game_success_rate,errors="coerce"),
        "recent5":pd.to_numeric(rows.asof_pitcher_prev5_game_success_rate,errors="coerce"),
        "middle":pd.to_numeric(rows.asof_pitcher_middle_rate,errors="coerce"),
        "reverse":pd.to_numeric(rows.asof_pitcher_reverse_rate,errors="coerce"),
        "li":pd.to_numeric(rows.li,errors="coerce"),"inning":pd.to_numeric(rows.inning,errors="coerce"),
        "runners":pd.to_numeric(rows.num_runners_on,errors="coerce")})
    x[CAT]=x[CAT].astype("string").fillna("__MISSING__").astype(str)
    return x
