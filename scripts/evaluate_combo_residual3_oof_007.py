#!/usr/bin/env python3
"""Evaluate pre-registered residual-3 axes on strict expanding-season OOF."""
from pathlib import Path
import hashlib, json
import numpy as np, pandas as pd

ROOT=Path(__file__).resolve().parents[1]
EXP=ROOT/'model/COMBO-RESID3-OOF-007'
OOF=EXP/'oof_predictions.csv'
AXES=[('hand',['pitcher_id','pitcher_hand','batter_hand'],1000.0),('strikes',['pitcher_id','strikes_before'],1000.0),('runners',['pitcher_id','num_runners_on'],2000.0)]
FIXED_WEIGHT=0.10

def sha(p):
 h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()
def metric(p,y):
 p=np.asarray(p,float); y=np.asarray(y,float); c=np.corrcoef(p,y)[0,1]; return float(1e5*c*c)
def table(cal,key,k):
 g=cal.groupby(key,dropna=False,observed=True)['resid'].agg(['mean','count'])
 g['adj']=g['mean']*g['count']/(g['count']+k)
 return g['adj']
def apply(v,t,key):
 idx=pd.MultiIndex.from_frame(v[key]); a=t.reindex(idx).fillna(0).to_numpy(); return a
def main():
 o=pd.read_csv(OOF)
 if o.row_id.duplicated().any() or not np.isfinite(o.pred).all() or not o.target.isin([0,1]).all(): raise RuntimeError('OOF contract failed')
 rows=[]
 for season in [2022,2023,2024]:
  val=o[o.season.eq(season)].copy(); cal=o[o.season.lt(season)].copy(); cal['resid']=cal.target-cal.pred
  base=metric(val.pred,val.target); out={'season':season,'n':len(val),'base_metric':base,'base_brier':float(np.mean((val.pred-val.target)**2))}
  for name,key,k in AXES:
   t=table(cal,key,k); adj=apply(val,t,key); p=np.clip(val.pred+adj,val.pred*0+0,val.pred*0+1)
   out[name]={'metric':metric(p,val.target),'brier':float(np.mean((p-val.target)**2)),'delta_metric':metric(p,val.target)-base,'delta_brier':float(np.mean((p-val.target)**2)-np.mean((val.pred-val.target)**2)),'table_rows':len(t),'weight':1.0,'k':k}
  # Fixed additive combination; no LB or test tuning.
  p=val.pred.copy()
  for _,key,k in AXES:p=np.clip(p+FIXED_WEIGHT*apply(val,table(cal,key,k),key),0,1)
  out['resid3']={'metric':metric(p,val.target),'brier':float(np.mean((p-val.target)**2)),'delta_metric':metric(p,val.target)-base,'delta_brier':float(np.mean((p-val.target)**2)-np.mean((val.pred-val.target)**2)),'weights':{'hand':FIXED_WEIGHT,'strikes':FIXED_WEIGHT,'runners':FIXED_WEIGHT}}
  rows.append(out)
 gate=all(x['resid3']['delta_metric'] >= 0 and x['resid3']['delta_brier'] <= 0 for x in rows if x['season'] >= 2023)
 result={'experiment_id':'COMBO-RESID3-OOF-007','oof_sha256':sha(OOF),'official_train_only':True,'axes':AXES,'fixed_weight':FIXED_WEIGHT,'season_results':rows,'gate_no_worse_2023_2024':gate,'status':'SCREEN_GATE_PASS' if gate else 'SCREEN_GATE_FAIL'}
 (EXP/'residual3_results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
 print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
