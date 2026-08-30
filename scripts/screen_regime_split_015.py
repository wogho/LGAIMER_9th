#!/usr/bin/env python3
import json,sys,time,os
from pathlib import Path
import catboost as cb
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.asof_state_features import add_state_for_cutoff,add_state_walkforward
from src.target_aggregates import build_pitcher_count_state_target_history
from scripts.build_combo_full_candidate_002 import COLS as BASE_COLS,extras,prep
from scripts.screen_trackman_context_003 import load_tm,attach,NEW
def main():
 t0=time.time();Y=2022;raw=pd.read_csv(ROOT/'data/train.csv',encoding='utf-8-sig');tm,_=load_tm();tr=raw[raw.season<Y].copy();va=raw[raw.season==Y].copy()
 def f_rate(rows,history):
  out=np.full(len(rows),0.5,float)
  for s in sorted(rows.season.unique()):
   q=history[history.season<s]; q=q[q.game_type.astype(str).eq('F')]
   if len(q)==0: continue
   gm=float(q.control_success.mean()); g=q.groupby('pitcher_id').control_success.agg(['mean','count']); adj=(g['mean']*g['count']+gm*50)/(g['count']+50); ix=rows.season.eq(s).to_numpy(); out[ix]=rows.loc[ix,'pitcher_id'].map(adj).fillna(gm).to_numpy()
  return out
 a=add_state_walkforward(tr.drop(columns=['row_id','control_success']),Y);b=add_state_for_cutoff(va.drop(columns=['row_id','control_success']),tr.drop(columns=['row_id','control_success']));at,ate,_,_=build_pitcher_count_state_target_history(tr,va.assign(control_success=0),smoothing=100.0)
 xtr=pd.concat([extras(a.reset_index(drop=True)),at.reset_index(drop=True)],axis=1);xva=pd.concat([extras(b.reset_index(drop=True)),ate.reset_index(drop=True)],axis=1);xtr=attach(xtr,tm,Y);xva=attach(xva,tm,Y)
 if os.environ.get('REGIME_F_RATE')=='1':
  xtr['f_pitcher_prior_rate']=f_rate(tr,tr); xva['f_pitcher_prior_rate']=f_rate(va,tr); rate_cols=['f_pitcher_prior_rate']
 else: rate_cols=[]
 if os.environ.get('REGIME_INTERACTIONS')=='1':
  for frame in (xtr,xva):
   frame['f_succ_inning']=frame['cur_succ']*frame['inning'];frame['f_mid_outs']=frame['cur_mid']*frame['outs_before'];frame['f_succ_month']=frame['cur_succ']*frame['game_month'];frame['f_succ_bs']=frame['cur_succ']*frame['bs']
  extra_cols=['f_succ_inning','f_mid_outs','f_succ_month','f_succ_bs']
 else: extra_cols=[]
 cols=BASE_COLS+NEW+extra_cols+rate_cols;xtr=xtr[cols];xva=xva[cols];xtr,cats=prep(xtr);xva,_=prep(xva);ytr=tr.control_success.to_numpy(np.int8);yva=va.control_success.to_numpy(np.int8)
 p=np.zeros(len(va),float);rows=[]
 for regime in ['F','R']:
  it=tr.game_type.astype(str).eq(regime).to_numpy();iv=va.game_type.astype(str).eq(regime).to_numpy()
  if os.environ.get('REGIME_EXCLUDE_2020')=='1' and regime=='F': it=it & tr.season.ne(2020).to_numpy()
  sw=np.ones(int(it.sum()),dtype=float)
  if os.environ.get('REGIME_WEIGHTED')=='1' and regime=='F': sw=tr.loc[it,'season'].map({2019:1.2,2020:0.8,2021:1.5}).fillna(1.0).to_numpy()
  cap_f=os.environ.get('REGIME_CAPACITY')=='1' and regime=='F'; cap_r=os.environ.get('REGIME_R_CAPACITY')=='1' and regime=='R'; cap=cap_f or cap_r; m=cb.CatBoostClassifier(iterations=600 if cap else 300,learning_rate=.03 if cap else .05,depth=7 if cap else 6,l2_leaf_reg=20 if cap else 10,loss_function='Logloss',thread_count=3,random_seed=42,allow_writing_files=False,verbose=False,early_stopping_rounds=80 if cap else 50);m.fit(cb.Pool(xtr.loc[it],label=ytr[it],weight=sw,cat_features=cats,feature_names=cols),eval_set=cb.Pool(xva.loc[iv],label=yva[iv],cat_features=cats,feature_names=cols),verbose=False);p[iv]=m.predict_proba(cb.Pool(xva.loc[iv],cat_features=cats,feature_names=cols))[:,1];rows.append({'regime':regime,'train_rows':int(it.sum()),'valid_rows':int(iv.sum()),'trees':int(m.get_best_iteration()+1),'brier':float(np.mean((p[iv]-yva[iv])**2)),'metric':float(1e5*np.corrcoef(p[iv],yva[iv])[0,1]**2),'weighted':bool(os.environ.get('REGIME_WEIGHTED')=='1' and regime=='F'),'capacity':cap})
 suffix=[]
 if os.environ.get('REGIME_CAPACITY')=='1': suffix.append('CAPACITY')
 if os.environ.get('REGIME_R_CAPACITY')=='1': suffix.append('RCAPACITY')
 if os.environ.get('REGIME_EXCLUDE_2020')=='1': suffix.append('EXCLUDE2020')
 if os.environ.get('REGIME_F_RATE')=='1': suffix.append('FRATE')
 if os.environ.get('REGIME_INTERACTIONS')=='1': suffix.append('INTERACTIONS')
 if os.environ.get('REGIME_WEIGHTED')=='1': suffix.append('WEIGHTED')
 eid='REGIME-SPLIT-015-' + ('-'.join(suffix) if suffix else 'BASE')
 report={'experiment_id':eid,'season':Y,'feature_count':len(cols),'official_train_only':True,'test_used':False,'external_data_used':False,'regimes':rows,'overall_brier':float(np.mean((p-yva)**2)),'overall_metric':float(1e5*np.corrcoef(p,yva)[0,1]**2),'elapsed_sec':time.time()-t0,'status':'SCREEN_COMPLETE'};d=ROOT/('model/'+eid);d.mkdir(parents=True,exist_ok=True);pd.DataFrame({'row_id':va.row_id.to_numpy(),'pred':p,'target':yva}).to_csv(d/'predictions.csv',index=False);(d/'screen_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
