import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
o=pd.read_csv(ROOT/'model/COMBO-RESID3-OOF-007/oof_predictions.csv')
raw=pd.read_csv(ROOT/'data/train.csv',usecols=['row_id','batter_id','asof_batter_n','asof_batter_success_rate','season','control_success'])
o=o.drop(columns=['target'],errors='ignore').merge(raw,on=['row_id','season'],how='left',validate='one_to_one')
metric=lambda p,y:float(1e5*np.corrcoef(p,y)[0,1]**2)
rows=[]
for season in [2022,2023,2024]:
 v=o[o.season.eq(season)].copy(); c=o[o.season.lt(season)].copy(); c['r']=c.control_success-c.pred
 prior=c.groupby('batter_id',observed=True).agg(prior_n=('asof_batter_n','max')).reset_index()
 ev=(c['asof_batter_n'].fillna(0)*c['asof_batter_success_rate'].fillna(0)).groupby(c.batter_id).max().rename('prior_evt').reset_index(); prior=prior.merge(ev,on='batter_id',how='left')
 v=v.merge(prior,on='batter_id',how='left'); cur=(v.asof_batter_n.fillna(0)-v.prior_n.fillna(0)).clip(lower=0); v['cur_bsucc']=((v.asof_batter_n.fillna(0)*v.asof_batter_success_rate.fillna(0)-v.prior_evt.fillna(0))/cur).where(cur>0)
 c['key']=c.batter_id.astype(str)+'|'+c.pitcher_hand.astype(str)+'|'+c.batter_hand.astype(str); v['key']=v.batter_id.astype(str)+'|'+v.pitcher_hand.astype(str)+'|'+v.batter_hand.astype(str)
 g=c.groupby('key',observed=True).r.agg(['mean','count']); g['adj']=g['mean']*g['count']/(g['count']+1000.0); a=g.adj.reindex(v.key).fillna(0).to_numpy(); p=np.clip(v.pred.to_numpy()+a,0,1); b=metric(v.pred,v.control_success); rows.append({'season':season,'base_metric':b,'candidate_metric':metric(p,v.control_success),'relative_improvement':metric(p,v.control_success)/b-1,'delta_brier':float(np.mean((p-v.control_success)**2)-np.mean((v.pred-v.control_success)**2)),'table_rows':len(g)})
out={'experiment_id':'BATTER-ASOF-MATCHUP-011','rows':rows,'status':'SCREEN_COMPLETE'}
(ROOT/'model/BATTER-ASOF-MATCHUP-011.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)); print(pd.DataFrame(rows).to_string(index=False))
