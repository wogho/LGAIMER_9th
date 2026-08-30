#!/usr/bin/env python3
from pathlib import Path
import json,sys
import catboost as cb
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.asof_state_features import add_state_for_cutoff,add_state_walkforward
from src.target_aggregates import build_pitcher_count_state_target_history
from scripts.build_combo_full_candidate_002 import COLS as COMBO_COLS,extras,prep
from scripts.screen_trackman_context_003 import load_tm,attach,NEW
def run(raw,tm,Y):
    tr=raw[raw.season<Y].copy();va=raw[raw.season==Y].copy();
    a=add_state_walkforward(tr.drop(columns=['row_id','control_success']),Y);b=add_state_for_cutoff(va.drop(columns=['row_id','control_success']),tr.drop(columns=['row_id','control_success']))
    at,ate,_,_=build_pitcher_count_state_target_history(tr,va,smoothing=100.0)
    xtr=attach(pd.concat([extras(a.reset_index(drop=True)),at.reset_index(drop=True)],axis=1),tm,Y);xva=attach(pd.concat([extras(b.reset_index(drop=True)),ate.reset_index(drop=True)],axis=1),tm,Y)
    xtr=xtr[COMBO_COLS+NEW];xva=xva[COMBO_COLS+NEW];xtr,cats=prep(xtr);xva,_=prep(xva);y=tr.control_success.to_numpy();yv=va.control_success.to_numpy()
    def fit(cols):
        m=cb.CatBoostClassifier(iterations=200,learning_rate=.05,depth=6,l2_leaf_reg=5,thread_count=-1,random_seed=42,allow_writing_files=False,verbose=False);m.fit(cb.Pool(xtr[cols],label=y,cat_features=cats,feature_names=cols));return m.predict_proba(cb.Pool(xva[cols],cat_features=cats,feature_names=cols))[:,1]
    p0=fit(COMBO_COLS);p1=fit(COMBO_COLS+NEW);b0=float(np.mean((p0-yv)**2));b1=float(np.mean((p1-yv)**2));return {'season':Y,'train_rows':len(tr),'valid_rows':len(va),'base_brier':b0,'trackman_brier':b1,'delta_brier':b1-b0,'coverage':float(xva[NEW].notna().any(axis=1).mean())}
def main():
    raw=pd.read_csv(ROOT/'data/train.csv',encoding='utf-8-sig');tm,m=load_tm();rows=[]
    for Y in (2022,2023,2024):
        r=run(raw,tm,Y);rows.append(r);print(json.dumps(r,ensure_ascii=False),flush=True)
    out={'experiment_id':'TRACKMAN-TRANSITION-005','mapping_source':'model/TRACKMAN-MAP-004/pitcher_id_map.csv','mapping_rows':len(m),'results':rows,'all_better':all(r['delta_brier']<0 for r in rows),'status':'PASS_TRANSITION' if all(r['delta_brier']<0 for r in rows) else 'REJECT','submission_status':'HOLD'};d=ROOT/'model/TRACKMAN-TRANSITION-005';d.mkdir(parents=True,exist_ok=True);(d/'transition_report.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
