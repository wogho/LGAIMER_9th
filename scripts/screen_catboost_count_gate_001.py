#!/usr/bin/env python3
from __future__ import annotations
import json, platform, sys, time
from pathlib import Path
import catboost as cb
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.features import build_features, load_data
from src.target_aggregates import build_pitcher_count_state_target_history
def br(y,p): return float(np.mean((y-p)**2))
def prep(x):
    x=x.copy(); cats=[c for c in x if str(x[c].dtype) in {'category','object'}]
    for c in cats:x[c]=x[c].astype('string').fillna('<NA>').astype(str)
    return x,cats
def main():
    raw=load_data(str(ROOT/'data/train.csv'),is_train=True); y=raw.control_success.to_numpy(np.int8); results={}
    for year in (2022,2023,2024):
        tr=raw[raw.season<year].copy();va=raw[raw.season==year].copy(); ytr=tr.control_success.to_numpy(np.int8);yv=va.control_success.to_numpy(np.int8)
        xb=build_features(tr);xv=build_features(va)
        at,av,lookup,meta=build_pitcher_count_state_target_history(tr,va,smoothing=100.0)
        xa=pd.concat([xb.reset_index(drop=True),at.reset_index(drop=True)],axis=1); xva=pd.concat([xv.reset_index(drop=True),av.reset_index(drop=True)],axis=1)
        gate_support=av['target_hist_pitcher_count_n'].fillna(0).to_numpy(float)
        row={}
        for name,xtrain,xvalid in [('base',xb,xv),('agg',xa,xva)]:
            xtrain,cat=prep(xtrain);xvalid,_=prep(xvalid);cols=list(xtrain.columns)
            tp=cb.Pool(xtrain,label=ytr,cat_features=cat,feature_names=cols);vp=cb.Pool(xvalid,label=yv,cat_features=cat,feature_names=cols)
            m=cb.CatBoostClassifier(iterations=300,learning_rate=.05,depth=6,l2_leaf_reg=3,loss_function='Logloss',eval_metric='Logloss',thread_count=-1,random_seed=42,allow_writing_files=False,verbose=False,early_stopping_rounds=50)
            m.fit(tp,eval_set=vp,verbose=False);p=m.predict_proba(vp)[:,1];row[name]={'feature_count':len(cols),'best_iteration':int(m.get_best_iteration()),'brier':br(yv,p),'pred':p}
        base=row['base']['pred'];agg=row['agg']['pred']; gates=[]
        for t in (0,30,100,300,1000):
            p=np.where(gate_support<=t,agg,base);gates.append({'support_le':t,'brier':br(yv,p),'delta_vs_base':br(yv,p)-row['base']['brier'],'aggregate_rows':int((gate_support<=t).sum())})
        results[str(year)]={'train_rows':len(tr),'valid_rows':len(va),'aggregate_lookup_rows':len(lookup),'base':{k:v for k,v in row['base'].items() if k!='pred'},'aggregate':{k:v for k,v in row['agg'].items() if k!='pred'},'gates':gates}
        print(year,json.dumps(results[str(year)],ensure_ascii=False))
    out=ROOT/'model/CATBOOST-COUNT-GATE-001';out.mkdir(parents=True,exist_ok=True);report={'experiment_id':'CATBOOST-COUNT-GATE-001','environment':{'python':platform.python_version(),'catboost':cb.__version__},'official_train_only':True,'test_used':False,'external_data_used':False,'results':results,'status':'HOLD'};(out/'screen_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
