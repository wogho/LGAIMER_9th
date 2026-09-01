"""Row-local meta features for context-dependent ensemble correction."""
from __future__ import annotations
import numpy as np
import pandas as pd

def build_gate_features(raw: pd.DataFrame, predictions: list[np.ndarray], risks: list[np.ndarray], old_prediction: np.ndarray) -> pd.DataFrame:
    base=np.column_stack(predictions+risks); pred3=base[:,:3]
    x=pd.DataFrame(base,columns=['p_v2','p_v3_55','p_v3_30','risk_middle','risk_wild','risk_reverse'])
    x['ensemble_std']=pred3.std(1);x['ensemble_range']=pred3.max(1)-pred3.min(1);x['old_prediction']=old_prediction
    x['log_pitcher_n']=np.log1p(pd.to_numeric(raw.asof_pitcher_n,errors='coerce').fillna(0).clip(lower=0)).to_numpy()
    x['log_batter_n']=np.log1p(pd.to_numeric(raw.asof_batter_n,errors='coerce').fillna(0).clip(lower=0)).to_numpy()
    x['li']=pd.to_numeric(raw.li,errors='coerce').fillna(0).to_numpy();x['inning']=raw.inning.to_numpy();x['balls']=raw.balls_before.to_numpy();x['strikes']=raw.strikes_before.to_numpy();x['runners']=raw.num_runners_on.to_numpy()
    rec=raw[['asof_pitcher_prev1_game_success_rate','asof_pitcher_prev3_game_success_rate','asof_pitcher_prev5_game_success_rate']].apply(pd.to_numeric,errors='coerce')
    x['recent_std']=rec.std(axis=1).fillna(.15).to_numpy();x['recent_gap']=(rec.mean(axis=1)-pd.to_numeric(raw.asof_pitcher_success_rate,errors='coerce')).fillna(0).to_numpy()
    return x.replace([np.inf,-np.inf],np.nan)
