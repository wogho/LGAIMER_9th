#!/usr/bin/env python3
"""Long-running capacity screen; one strict 2024 fold before any full build."""
import json,sys,time,os
from pathlib import Path
import catboost as cb
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.asof_state_features import add_state_for_cutoff,add_state_walkforward
from src.target_aggregates import build_pitcher_count_state_target_history
from scripts.build_combo_full_candidate_002 import COLS as BASE_COLS,extras,prep
from scripts.screen_trackman_context_003 import load_tm,context_tables,attach,NEW
def main():
 t0=time.time(); raw=pd.read_csv(ROOT/'data/train.csv',encoding='utf-8-sig');tm,_=load_tm();Y=int(os.environ.get('CAPACITY_SEASON','2024'));tr=raw[raw.season<Y].copy();va=raw[raw.season==Y].copy()
 a=add_state_walkforward(tr.drop(columns=['row_id','control_success']),Y);b=add_state_for_cutoff(va.drop(columns=['row_id','control_success']),tr.drop(columns=['row_id','control_success']))
 at,ate,_,_=build_pitcher_count_state_target_history(tr,va.assign(control_success=0),smoothing=100.0)
 xtr=pd.concat([extras(a.reset_index(drop=True)),at.reset_index(drop=True)],axis=1);xva=pd.concat([extras(b.reset_index(drop=True)),ate.reset_index(drop=True)],axis=1);xtr=attach(xtr,tm,Y);xva=attach(xva,tm,Y);cols=BASE_COLS+NEW;xtr=xtr[cols];xva=xva[cols];xtr,cats=prep(xtr);xva,_=prep(xva);y=va.control_success.to_numpy(np.int8)
 rows=[]
 seeds=[int(x) for x in os.environ.get('CAPACITY_SEEDS','42,2024,7').split(',') if x]
 for seed in seeds:
  m=cb.CatBoostClassifier(iterations=600,learning_rate=.03,depth=7,l2_leaf_reg=20,loss_function='Logloss',thread_count=3,random_seed=seed,allow_writing_files=False,verbose=False,early_stopping_rounds=80);m.fit(cb.Pool(xtr,label=tr.control_success.to_numpy(np.int8),cat_features=cats,feature_names=cols),eval_set=cb.Pool(xva,label=y,cat_features=cats,feature_names=cols),verbose=False);p=m.predict_proba(cb.Pool(xva,cat_features=cats,feature_names=cols))[:,1]; rows.append({'seed':seed,'trees':int(m.get_best_iteration()+1),'brier':float(np.mean((p-y)**2)),'metric':float(1e5*np.corrcoef(p,y)[0,1]**2),'min':float(p.min()),'max':float(p.max())}); print(rows[-1],flush=True)
 out={'experiment_id':'COMBO-CAPACITY-014','season':Y,'feature_count':len(cols),'official_train_only':True,'test_used':False,'external_data_used':False,'seeds':seeds,'results':rows,'elapsed_sec':time.time()-t0,'status':'SCREEN_COMPLETE'};d=ROOT/f'model/COMBO-CAPACITY-014-{Y}';d.mkdir(parents=True,exist_ok=True);(d/'screen_report.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
