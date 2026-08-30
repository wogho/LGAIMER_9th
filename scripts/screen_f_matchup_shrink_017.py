import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
p=pd.read_csv(ROOT/'model/REGIME-SPLIT-015-INTERACTIONS/predictions.csv')
raw=pd.read_csv(ROOT/'data/train.csv',usecols=['row_id','season','game_type','pitcher_id','batter_hand','control_success'])
m=p.merge(raw,on='row_id',validate='one_to_one'); cal=raw[(raw.season<2022)&raw.game_type.eq('F')&raw.season.ne(2020)].copy(); val=m[m.game_type.eq('F')].copy(); y=val.control_success.to_numpy(); base=1e5*np.corrcoef(val.pred,y)[0,1]**2
g=cal.groupby(['pitcher_id','batter_hand'],observed=True).control_success.agg(['mean','count']); gm=float(cal.control_success.mean()); g['rate']=(g['mean']*g['count']+gm*100)/(g['count']+100); rate=g['rate'].reindex(pd.MultiIndex.from_frame(val[['pitcher_id','batter_hand']])).fillna(gm).to_numpy()
rows=[]
for w in [.1,.25,.5,.75]:
 q=(1-w)*val.pred.to_numpy()+w*rate; rows.append({'weight':w,'base_metric':base,'metric':float(1e5*np.corrcoef(q,y)[0,1]**2),'relative_improvement':float((1e5*np.corrcoef(q,y)[0,1]**2)/base-1),'delta_brier':float(np.mean((q-y)**2)-np.mean((val.pred-y)**2))})
out={'experiment_id':'F-MATCHUP-SHRINK-017','calibration_rows':len(cal),'table_rows':len(g),'rows':rows,'status':'SCREEN_COMPLETE'}
(ROOT/'model/F-MATCHUP-SHRINK-017.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)); print(pd.DataFrame(rows).to_string(index=False))
