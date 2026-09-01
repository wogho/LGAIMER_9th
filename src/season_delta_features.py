"""Row-local current-season form recovered from a prior-season train snapshot."""
from __future__ import annotations
import numpy as np
import pandas as pd

RATE_COLS = [
    'asof_pitcher_success_rate','asof_pitcher_reverse_rate',
    'asof_pitcher_middle_rate','asof_pitcher_ball_rate','asof_pitcher_strike_rate',
]

def build_snapshots(history: pd.DataFrame) -> pd.DataFrame:
    """Per-season last cumulative states, keyed by the following target season."""
    rows=[]
    for (pid, season),g in history.groupby(['pitcher_id','season'],sort=False):
        last=g.iloc[-1]; n=float(last['asof_pitcher_n'])
        row={'pitcher_id':str(pid),'season':int(season)+1,'snapshot_n':n+1.0}
        # The final labeled pitch is needed because asof values are pre-pitch.
        row['snapshot_asof_pitcher_success_rate_count'] = n*float(last['asof_pitcher_success_rate'])+float(last['control_success'])
        # For subtype rates the final event is unavailable without a following row;
        # use the pre-pitch state conservatively rather than inventing a label.
        for c in RATE_COLS[1:]: row['snapshot_'+c+'_count']=n*float(last[c])
        rows.append(row)
    direct=pd.DataFrame(rows)
    expanded=[]
    max_target=int(history.season.max())+1
    for pid,g in direct.groupby('pitcher_id',sort=False):
        years=pd.Index(range(int(g.season.min()),max_target+1),name='season')
        h=g.set_index('season').reindex(years).ffill().reset_index()
        h['pitcher_id']=pid; expanded.append(h)
    return pd.concat(expanded,ignore_index=True)

def attach_season_delta(x: pd.DataFrame, raw: pd.DataFrame, snapshots: pd.DataFrame, prior: float) -> pd.DataFrame:
    out=x.copy(); keys=pd.MultiIndex.from_arrays([raw.pitcher_id.astype(str),raw.season.astype(int)])
    snap=snapshots.set_index(['pitcher_id','season'])
    prev_n=snap.snapshot_n.reindex(keys).fillna(0).to_numpy(float)
    n=pd.to_numeric(raw.asof_pitcher_n,errors='coerce').fillna(0).to_numpy(float)
    season_n=np.maximum(n-prev_n,0); out['season_pitcher_n']=season_n
    for c in RATE_COLS:
        total=n*pd.to_numeric(raw[c],errors='coerce').fillna(prior if c.endswith('success_rate') else 0).to_numpy(float)
        key='snapshot_'+c+'_count'; prev=snap[key].reindex(keys).fillna(0).to_numpy(float)
        count=np.maximum(total-prev,0); rate=(count+prior*30)/(season_n+30)
        out['season_'+c.removeprefix('asof_pitcher_')]=rate
        out['season_'+c.removeprefix('asof_pitcher_')+'_raw']=np.divide(count,season_n,out=np.full(len(raw),prior),where=season_n>0)
    out['season_vs_career_success']=out['season_success_rate']-pd.to_numeric(raw.asof_pitcher_success_rate,errors='coerce').fillna(prior).to_numpy()
    return out
