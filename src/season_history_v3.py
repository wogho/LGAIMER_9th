"""Prior-season snapshots for batter exposure and pitcher pitch-mix histories."""
from __future__ import annotations
import numpy as np
import pandas as pd

def build_entity_snapshots(history,id_col,n_col,rate_cols,final_success_col=None):
    rows=[]
    for (entity,season),g in history.groupby([id_col,'season'],sort=False):
        last=g.iloc[-1];n=float(last[n_col]);row={id_col:str(entity),'season':int(season)+1,'snapshot_n':n+1}
        for c in rate_cols:
            count=n*float(last[c]) if pd.notna(last[c]) else 0.0
            if final_success_col is not None and c==rate_cols[0]: count+=float(last[final_success_col])
            row['snapshot_'+c+'_count']=count
        rows.append(row)
    direct=pd.DataFrame(rows);expanded=[];max_target=int(history.season.max())+1
    for entity,g in direct.groupby(id_col,sort=False):
        years=pd.Index(range(int(g.season.min()),max_target+1),name='season')
        h=g.set_index('season').reindex(years).ffill().reset_index();h[id_col]=str(entity);expanded.append(h)
    return pd.concat(expanded,ignore_index=True)

def attach_entity_season(x,raw,snap,id_col,n_col,rate_cols,prefix,prior):
    out=x.copy();keys=pd.MultiIndex.from_arrays([raw[id_col].astype(str),raw.season.astype(int)]);table=snap.set_index([id_col,'season'])
    prev_n=table.snapshot_n.reindex(keys).fillna(0).to_numpy(float);n=pd.to_numeric(raw[n_col],errors='coerce').fillna(0).to_numpy(float);season_n=np.maximum(n-prev_n,0);out[prefix+'_n']=season_n
    for c in rate_cols:
        fallback=prior if 'success' in c else 0.0
        total=n*pd.to_numeric(raw[c],errors='coerce').fillna(fallback).to_numpy(float);prev=table['snapshot_'+c+'_count'].reindex(keys).fillna(0).to_numpy(float);count=np.maximum(total-prev,0)
        raw_rate=np.divide(count,season_n,out=np.full(len(raw),fallback),where=season_n>0)
        out[prefix+'_'+c.replace('asof_batter_','').replace('asof_pitcher_','')+'_raw']=raw_rate
    return out
