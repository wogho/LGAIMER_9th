import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
o=pd.read_csv(ROOT/'model/COMBO-RESID3-OOF-007/oof_predictions.csv')
ids=pd.read_csv(ROOT/'data/train.csv',usecols=['row_id','batter_id'])
o=o.merge(ids,on='row_id',how='left',validate='one_to_one')
metric=lambda p,y:float(1e5*np.corrcoef(p,y)[0,1]**2)
rows=[]
for season in [2022,2023,2024]:
 v=o[o.season.eq(season)].copy(); c=o[o.season.lt(season)].copy(); c['r']=c.target-c.pred
 base=metric(v.pred,v.target)
 for name,key,k in [('pitcher','pitcher_id',50000.0),('batter','batter_id',20000.0)]:
  g=c.groupby(key,observed=True).r.agg(['mean','count']); g['adj']=g['mean']*g['count']/(g['count']+k)
  a=g['adj'].reindex(v[key]).fillna(0).to_numpy(); p=np.clip(v.pred.to_numpy()+a,0,1)
  rows.append({'season':season,'axis':name,'k':k,'base_metric':base,'metric':metric(p,v.target),'delta_relative':metric(p,v.target)/base-1,'delta_brier':float(np.mean((p-v.target)**2)-np.mean((v.pred-v.target)**2))})
 p=v.pred.to_numpy().copy()
 for key,k in [('pitcher_id',50000.0),('batter_id',20000.0)]:
  g=c.groupby(key,observed=True).r.agg(['mean','count']); g['adj']=g['mean']*g['count']/(g['count']+k); p=np.clip(p+g['adj'].reindex(v[key]).fillna(0).to_numpy(),0,1)
 rows.append({'season':season,'axis':'both','k':'50000/20000','base_metric':base,'metric':metric(p,v.target),'delta_relative':metric(p,v.target)/base-1,'delta_brier':float(np.mean((p-v.target)**2)-np.mean((v.pred-v.target)**2))})
out={'experiment_id':'MAIN-EFFECT-RESIDUAL-010','rows':rows,'status':'SCREEN_COMPLETE'}
(ROOT/'model/MAIN-EFFECT-RESIDUAL-010.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)); print(pd.DataFrame(rows).to_string(index=False))
