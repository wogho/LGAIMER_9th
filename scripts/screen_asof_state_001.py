#!/usr/bin/env python3
from __future__ import annotations
import json, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.asof_state_features import STATE_COLUMNS, add_state_for_cutoff
def main():
 d=pd.read_csv(ROOT/'data/train.csv',encoding='utf-8-sig'); rows=[]
 for year in (2022,2023,2024):
  v=d[d.season==year].copy(); h=d[d.season<year]; s=add_state_for_cutoff(v,h)
  prior=h.groupby('pitcher_id').size(); cur=s.asof_pitcher_n.fillna(0).to_numpy()-s.pitcher_id.map(prior).fillna(0).to_numpy(); rank=v.groupby('pitcher_id',sort=False).cumcount().to_numpy(); rows.append({'year':year,'rows':len(v),'cur_n_exact_rate':float(np.mean(np.isclose(cur,rank))), 'cur_n_max_abs':float(np.max(np.abs(cur-rank))), 'state_columns':len(STATE_COLUMNS),'missing_cur_succ':float(s.cur_succ.isna().mean()),'missing_cur_bsucc':float(s.cur_bsucc.isna().mean())})
 report={'experiment_id':'ASOF-STATE-001','environment':{'python':platform.python_version()},'official_train_only':True,'test_used':False,'external_data_used':False,'state_columns':STATE_COLUMNS,'checks':rows,'status':'PASS_RECONSTRUCTION' if all(r['cur_n_exact_rate']==1.0 and r['cur_n_max_abs']==0.0 for r in rows) else 'FAIL_RECONSTRUCTION','submission_status':'HOLD'}
 out=ROOT/'model/ASOF-STATE-001'; out.mkdir(parents=True,exist_ok=True); (out/'reconstruction_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
