import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
ROOT=Path(__file__).resolve().parents[1]
o=pd.read_csv(ROOT/'model/COMBO-RESID3-OOF-007/oof_predictions.csv')
metric=lambda p,y:float(1e5*np.corrcoef(p,y)[0,1]**2)
rows=[]
for season in [2022,2023,2024]:
 v=o[o.season.eq(season)]; c=o[o.season.lt(season)]; b=metric(v.pred,v.target)
 if len(c)==0: rows.append({'season':season,'base_metric':b,'candidate_metric':b,'relative_improvement':0.0,'delta_brier':0.0}); continue
 iso=IsotonicRegression(y_min=0,y_max=1,out_of_bounds='clip').fit(c.pred,c.target); p=iso.predict(v.pred); m=metric(p,v.target)
 rows.append({'season':season,'base_metric':b,'candidate_metric':m,'relative_improvement':m/b-1,'delta_brier':float(np.mean((p-v.target)**2)-np.mean((v.pred-v.target)**2))})
out={'experiment_id':'CORR-CALIBRATION-012','rows':rows,'status':'SCREEN_COMPLETE'}
(ROOT/'model/CORR-CALIBRATION-012.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)); print(pd.DataFrame(rows).to_string(index=False))
