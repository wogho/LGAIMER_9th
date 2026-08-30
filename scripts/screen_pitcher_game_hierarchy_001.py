#!/usr/bin/env python3
"""Official-train-only pitcher x game_type hierarchical posterior screen."""
from __future__ import annotations
import json, platform
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"model/PITCHER-GAME-HIER-001"; TRAIN=ROOT/"data/train.csv"
def brier(y,p): return float(np.mean((np.asarray(y)-np.asarray(p))**2))
def forecast(rates, year):
 s=rates.sort_index().tail(4)
 if len(s)<2:return float(s.iloc[-1])
 x=s.index.to_numpy(float); y=s.to_numpy(float); return float(np.clip(np.polyval(np.polyfit(x,y,1),year),.42,.62))
def keys(df): return df.pitcher_id.astype(str)+"|"+df.game_type.astype(str)
def pred(history, rows, q, ap, ac):
 pk=history.pitcher_id.astype(str); ck=keys(history)
 ph=pd.DataFrame({'k':pk,'y':history.control_success.astype(float)}).groupby('k').y.agg(['sum','count'])
 ph['p']=(ph['sum']+ap*q)/(ph['count']+ap)
 ch=pd.DataFrame({'k':ck,'y':history.control_success.astype(float),'parent':pk.map(ph.p).fillna(q)}).groupby('k').agg(sum=('y','sum'),count=('y','count'),parent=('parent','mean'))
 ch['p']=(ch['sum']+ac*ch['parent'])/(ch['count']+ac)
 rpk=rows.pitcher_id.astype(str); rck=keys(rows)
 pp=rpk.map(ph.p).fillna(q); out=rck.map(ch.p).fillna(pp)
 return out.to_numpy(float), rpk.map(ph['count']).fillna(0).to_numpy(float), rck.map(ch['count']).fillna(0).to_numpy(float)
def main():
 df=pd.read_csv(TRAIN,encoding='utf-8-sig'); rates=df.groupby('season').control_success.mean(); baseline_paths={2022:ROOT/'model/ENS-CATF-LGBMCATR5050-FE001-EW-2022/selective_predictions_2022.csv',2023:ROOT/'model/ENS-CATF-LGBMCATR5050-FE001/selective_predictions_2023.csv',2024:ROOT/'model/ENS-CATF-LGBMCATR5050-FE001/selective_predictions_2024.csv'}
 rows=[]
 for year in (2022,2023,2024):
  hist=df[df.season<year]; valid=df[df.season==year]; q=forecast(rates[rates.index<year],year); base=pd.read_csv(baseline_paths[year]).pred_selective.to_numpy(float); y=valid.control_success.to_numpy(float)
  for ap in (50,100,200):
   for ac in (150,300,600):
    p,pc,cc=pred(hist,valid,q,ap,ac)
    for w in (0.25,0.5,0.75,1.0):
     z=(1-w)*base+w*p; rows.append({'candidate':f'ap{ap}_ac{ac}_w{w:.2f}','year':year,'alpha_pitcher':ap,'alpha_context':ac,'hier_weight':w,'brier':brier(y,z),'delta_vs_base':brier(y,z)-brier(y,base),'mean':float(z.mean()),'pitcher_coverage':float((pc>0).mean()),'context_coverage':float((cc>0).mean()),'q':q})
 out=pd.DataFrame(rows); summary=out.groupby('candidate').agg(mean_brier=('brier','mean'),worst_delta=('delta_vs_base','max'),mean_delta=('delta_vs_base','mean'),all_years_improved=('delta_vs_base',lambda x:bool((x<0).all())),min_pitcher_coverage=('pitcher_coverage','min'),min_context_coverage=('context_coverage','min')).reset_index().sort_values(['all_years_improved','mean_delta'],ascending=[False,True])
 report={'experiment_id':'PITCHER-GAME-HIER-001','environment':{'python':platform.python_version()},'official_train_only':True,'test_used':False,'external_data_used':False,'transitions':[2022,2023,2024],'summary_top':summary.head(10).to_dict(orient='records'),'status':'PASS_ALL_YEARS' if bool(summary.all_years_improved.any()) else 'FAIL_FORWARD_SIGN','submission_status':'HOLD'}
 OUT.mkdir(parents=True,exist_ok=True); out.to_csv(OUT/'grid_metrics.csv',index=False); summary.to_csv(OUT/'candidate_summary.csv',index=False); (OUT/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
