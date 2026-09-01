#!/usr/bin/env python3
"""Train and compare the literal 110A, 110B and 110C candidates."""
from __future__ import annotations

import gc, hashlib, importlib.util, json, os, sys, time
from pathlib import Path
os.environ.setdefault("OMP_NUM_THREADS", "4"); os.environ.setdefault("MKL_NUM_THREADS", "4")
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'model/REF4-110-ORIGINAL-R2'; THREADS=4
YEARS=(2022,2023,2024); RSEEDS=(42,1,2,3,4); FSEEDS=(42,1,2,3); ROUTER_SEEDS=(42,1,2)
sys.path.insert(0,str(ROOT/'model/REF4-SUPER-ENSEMBLE-109C/production_package'))
from src.v5_deep_61_features import build_v5_deep_61_features  # noqa:E402

def module(name,path):
 s=importlib.util.spec_from_file_location(name,path); m=importlib.util.module_from_spec(s); sys.modules[name]=m; s.loader.exec_module(m); return m
HELPER=module('r2_helper',ROOT/'src/ref4_110_codex.py')
build_eb_tables=HELPER.build_eb_tables; apply_eb_tables=HELPER.apply_eb_tables; router_features=HELPER.router_features
M108=module('r2_108',ROOT/'scripts/build_ref4_super_ensemble_108c.py')
M109=module('r2_109',ROOT/'scripts/build_ref4_super_ensemble_109c.py')

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def features(rows,anchor,profile,kind,eb=None):
 x,_=build_v5_deep_61_features(rows,profile_path=profile,prior=.523766)
 if not kind.endswith('f'):
  prev=pd.to_numeric(rows['asof_pitcher_prev1_game_success_rate'],errors='coerce').fillna(.523766).to_numpy(float)
  br=pd.to_numeric(rows['asof_batter_success_rate'],errors='coerce').fillna(.523766).to_numpy(float);x['form_gap']=prev-br
 x['anchor_p']=anchor
 if kind=='107': pass
 elif kind.startswith('108'): x=pd.concat([x,M108.extract_advanced_physics(rows.reset_index(drop=True))],axis=1)
 else: x=pd.concat([x,M109.extract_hyper_regime_tensor_109c(rows.reset_index(drop=True))],axis=1)
 if eb is not None:
  for c in eb.columns:x[c]=eb[c].to_numpy(float)
 return x.replace([np.inf,-np.inf],np.nan).fillna(0)
def cbfit(x,y,xv,path,seed,it=220,depth=5,lr=.03,l2=15):
 m=CatBoostRegressor(iterations=it,depth=depth,learning_rate=lr,l2_leaf_reg=l2,random_seed=seed,thread_count=THREADS,allow_writing_files=False,verbose=False);m.fit(x,y);m.save_model(path);p=m.predict(xv);del m;gc.collect();return p
def family(x,y,xv,d,prefix,future=False,tri=True):
 d.mkdir(parents=True,exist_ok=True); seeds=FSEEDS if future else RSEEDS; cb=[];lg=[];xg=[];paths=[]
 for s in seeds:
  print(f'  {prefix} seed={s}',flush=True)
  p=d/f'{prefix}_cb_seed{s}.cbm';cb.append(cbfit(x,y,xv,p,s,150 if future else 220,4 if future else 5,.03,20 if future else 15));paths.append(p)
  if future:continue
  p=d/f'{prefix}_lgb_seed{s}.txt';m=lgb.train({'objective':'regression','learning_rate':.03,'num_leaves':31,'min_data_in_leaf':50,'feature_fraction':.8,'bagging_fraction':.8,'bagging_freq':1,'seed':s,'verbosity':-1,'num_threads':THREADS},lgb.Dataset(x,label=y),num_boost_round=160);m.save_model(str(p));lg.append(m.predict(xv));paths.append(p);del m
  if tri:
   p=d/f'{prefix}_xgb_seed{s}.json';m=XGBRegressor(n_estimators=170,learning_rate=.03,max_depth=4,subsample=.8,colsample_bytree=.8,random_state=s,n_jobs=THREADS,tree_method='hist');m.fit(x,y);m.save_model(p);xg.append(m.predict(xv));paths.append(p);del m
  gc.collect()
 vals=[np.mean(cb,axis=0)];
 if lg:vals.append(np.mean(lg,axis=0))
 if xg:vals.append(np.mean(xg,axis=0))
 return np.mean(vals,axis=0),paths
def expert_fold(raw,year):
 fold=OUT/f'backbone_fold_{year}';a=pd.read_csv(fold/'anchor_predictions.csv',dtype={'row_id':str});hist=raw.loc[raw.season<year].reset_index(drop=True);val=raw.loc[raw.season==year].reset_index(drop=True)
 assert a.row_id.tolist()==pd.concat([hist,val]).row_id.astype(str).tolist(); ah=a.loc[~a.is_validation,'p103a'].to_numpy(float);av=a.loc[a.is_validation,'p103a'].to_numpy(float)
 profile=fold/'team_asof_profile.json';rh=hist.game_type.ne('F').to_numpy();rv=val.game_type.ne('F').to_numpy();fh=~rh;fv=~rv;res=hist.control_success.to_numpy(float)-ah;d=OUT/f'expert_fold_{year}';d.mkdir(parents=True,exist_ok=True);paths=[]
 xh=features(hist.loc[rh].reset_index(drop=True),ah[rh],profile,'107');xv=features(val.loc[rv].reset_index(drop=True),av[rv],profile,'107');p107=av.copy();q=cbfit(xh,res[rh],xv,d/'107_cb.cbm',42,200,5,.04,15);p107[rv]=np.clip(av[rv]+.05*q,1e-5,1-1e-5);paths.append(d/'107_cb.cbm')
 # Both 108C and 109C production inference inherit the 107A correction first.
 p108=p107.copy();xh=features(hist.loc[rh].reset_index(drop=True),ah[rh],profile,'108');xv=features(val.loc[rv].reset_index(drop=True),p107[rv],profile,'108');q,z=family(xh,res[rh],xv,d,'108r',False,False);paths+=z;p108[rv]=np.clip(p107[rv]+.08*q,1e-5,1-1e-5)
 if fv.any():
  xh=features(hist.loc[fh].reset_index(drop=True),ah[fh],profile,'108f');xv=features(val.loc[fv].reset_index(drop=True),p107[fv],profile,'108f');q,z=family(xh,res[fh],xv,d,'108f',True);paths+=z;p108[fv]=np.clip(p107[fv]+.04*q,1e-5,1-1e-5)
 p109=p107.copy();xh=features(hist.loc[rh].reset_index(drop=True),ah[rh],profile,'109');xv=features(val.loc[rv].reset_index(drop=True),p107[rv],profile,'109');q,z=family(xh,res[rh],xv,d,'109r');paths+=z;p109[rv]=np.clip(p107[rv]+.085*q,1e-5,1-1e-5)
 if fv.any():
  xh=features(hist.loc[fh].reset_index(drop=True),ah[fh],profile,'109f');xv=features(val.loc[fv].reset_index(drop=True),p107[fv],profile,'109f');q,z=family(xh,res[fh],xv,d,'109f',True);paths+=z;p109[fv]=np.clip(p107[fv]+.035*q,1e-5,1-1e-5)
 # 110C: same 109 family with fold-local EB/reliability features, fixed alphas 100/300 and k=25.
 tables=build_eb_tables(hist,100.,300.,prediction_year=year,recency_weighted=True); ebh=apply_eb_tables(hist,tables,25.); ebv=apply_eb_tables(val,tables,25.);p110c=p107.copy()
 xh=features(hist.loc[rh].reset_index(drop=True),ah[rh],profile,'109',ebh.loc[rh].reset_index(drop=True));xv=features(val.loc[rv].reset_index(drop=True),p107[rv],profile,'109',ebv.loc[rv].reset_index(drop=True));q,z=family(xh,res[rh],xv,d,'110cr');paths+=z;p110c[rv]=np.clip(p107[rv]+.085*q,1e-5,1-1e-5)
 if fv.any():
  xh=features(hist.loc[fh].reset_index(drop=True),ah[fh],profile,'109f',ebh.loc[fh].reset_index(drop=True));xv=features(val.loc[fv].reset_index(drop=True),p107[fv],profile,'109f',ebv.loc[fv].reset_index(drop=True));q,z=family(xh,res[fh],xv,d,'110cf',True);paths+=z;p110c[fv]=np.clip(p107[fv]+.035*q,1e-5,1-1e-5)
 out=pd.DataFrame({'row_id':val.row_id.astype(str),'season':year,'game_type':val.game_type,'pitcher_id':val.pitcher_id.astype(str),'target':val.control_success,'p103a':av,'p107':p107,'p108c':p108,'p109c':p109,'p110c':p110c});out.to_csv(d/'expert_predictions.csv',index=False);return out,paths
def nested_110b(raw,experts,valid):
 # 2022 is a frozen warm-up control; later folds use only prior strict p103A OOF rows.
 if valid==2022:return experts.loc[experts.season.eq(valid),'p109c'].to_numpy(float),[]
 tr=experts.loc[experts.season<valid].reset_index(drop=True);va=experts.loc[experts.season.eq(valid)].reset_index(drop=True);rtr=tr.game_type.ne('F').to_numpy();rva=va.game_type.ne('F').to_numpy();ftr=~rtr;fva=~rva;profile=OUT/f'nested_profile_{valid}.json';hist=raw.loc[raw.season<valid];profile.write_text(json.dumps({'p_team':{str(k):float(v) for k,v in hist.groupby('pitcher_team_id').control_success.mean().items()},'b_team':{str(k):float(v) for k,v in hist.groupby('batter_team_id').control_success.mean().items()},'prior':float(hist.control_success.mean())}));res=tr.target.to_numpy(float)-tr.p103a.to_numpy(float);p=va.p107.to_numpy(float).copy();paths=[]
 raw_idx=raw.assign(row_id=raw.row_id.astype(str)).set_index('row_id',drop=False)
 for mask,vmask,prefix,w,fut in [(rtr,rva,'110br',.085,False),(ftr,fva,'110bf',.035,True)]:
  if not vmask.any():continue
  kind='109f' if fut else '109'
  x=features(raw_idx.loc[tr.loc[mask,'row_id']].reset_index(drop=True),tr.loc[mask,'p103a'].to_numpy(),profile,kind);xv=features(raw_idx.loc[va.loc[vmask,'row_id']].reset_index(drop=True),va.loc[vmask,'p107'].to_numpy(),profile,kind);q,z=family(x,res[mask],xv,OUT/f'nested_110b_{valid}',prefix,fut);paths+=z;p[vmask]=np.clip(p[vmask]+w*q,1e-5,1-1e-5)
 return p,paths
def router(train,valid,raw,d):
 raw_idx=raw.assign(row_id=raw.row_id.astype(str)).set_index('row_id',drop=False)
 xt=router_features(raw_idx.loc[train.row_id].reset_index(drop=True),train[['p107','p108c','p109c']].to_numpy(float));xv=router_features(raw_idx.loc[valid.row_id].reset_index(drop=True),valid[['p107','p108c','p109c']].to_numpy(float));y=train.target.to_numpy(float)-train.p109c.to_numpy(float);d.mkdir(parents=True,exist_ok=True);memb=[];paths=[]
 for s in ROUTER_SEEDS:
  p=d/f'router_seed{s}.cbm';m=CatBoostRegressor(iterations=150,depth=4,learning_rate=.03,l2_leaf_reg=5,random_seed=s,thread_count=THREADS,allow_writing_files=False,verbose=False);m.fit(xt,y);m.save_model(p);memb.append(m.predict(xv));paths.append(p)
 return np.clip(valid.p109c.to_numpy(float)+np.mean(memb,axis=0),1e-5,1-1e-5),paths
def metric(y,p):
 b=float(np.mean((y-p)**2));r=float(y.mean());return {'rows':len(y),'brier':b,'bss':1-b/(r*(1-r)),'local_cv_proxy_score':1e5*(1-b/(r*(1-r)))}
def pitcher_bootstrap_2024(frame,candidate,reps=2000,seed=110):
 d=frame.loc[frame.season.eq(2024),['pitcher_id','target','p109c',candidate]].copy()
 d['delta']=(d[candidate]-d.target)**2-(d.p109c-d.target)**2
 g=d.groupby('pitcher_id',sort=False).delta.agg(['sum','count']).reset_index(drop=True)
 rng=np.random.default_rng(seed); vals=np.empty(reps,float); n=len(g)
 sums=g['sum'].to_numpy(float); counts=g['count'].to_numpy(float)
 for i in range(reps):
  take=rng.integers(0,n,n); vals[i]=sums[take].sum()/counts[take].sum()
 return {'reps':reps,'pitcher_clusters':n,'mean_delta':float(d.delta.mean()),'ci_low':float(np.quantile(vals,.025)),'ci_high':float(np.quantile(vals,.975))}
def main():
 pre=json.loads((OUT/'preflight_report.json').read_text());assert pre['status']=='AUDIT_VERIFIED';raw=pd.read_csv(ROOT/'data/train.csv',low_memory=False);parts=[];paths=[]
 for y in YEARS:
  cached=OUT/f'expert_fold_{y}/expert_predictions.csv'
  if cached.exists():
   print('expert fold',y,'[verified cache]',flush=True);q=pd.read_csv(cached,dtype={'row_id':str,'pitcher_id':str});z=sorted(p for p in cached.parent.iterdir() if p.is_file() and p.suffix in {'.cbm','.txt','.json'});assert len(q)==int(raw.season.eq(y).sum()) and len(z)==53
  else:
   print('expert fold',y,flush=True);q,z=expert_fold(raw,y)
  parts.append(q);paths+=z
 e=pd.concat(parts,ignore_index=True);pred={'110A':e.p109c.to_numpy(float).copy(),'110B':np.empty(len(e)),'110C':e.p110c.to_numpy(float).copy()}
 for y in YEARS:
  print('nested 110B fold',y,flush=True)
  m=e.season.eq(y).to_numpy();pred['110B'][m],z=nested_110b(raw,e,y);paths+=z
 for y in (2023,2024):
  print('router 110A fold',y,flush=True)
  m=e.season.eq(y).to_numpy();tr=e.loc[e.season.lt(y)].reset_index(drop=True);va=e.loc[m].reset_index(drop=True);pred['110A'][m],z=router(tr,va,raw,OUT/f'router_{y}');paths+=z
 out=e.copy()
 for k,v in pred.items():out[k]=v
 out.to_csv(OUT/'expert_oof.csv',index=False);base={str(y):metric(e.loc[e.season.eq(y),'target'].to_numpy(float),e.loc[e.season.eq(y),'p109c'].to_numpy(float)) for y in YEARS};cand=[];weights={2022:.2,2023:.3,2024:.5}
 for k in ('110A','110B','110C'):
  mets={};delta={}
  for y in YEARS:
   m=e.season.eq(y);mets[str(y)]=metric(e.loc[m,'target'].to_numpy(float),out.loc[m,k].to_numpy(float));delta[str(y)]=mets[str(y)]['brier']-base[str(y)]['brier']
  weighted=sum(weights[y]*delta[str(y)] for y in YEARS);worst=max(delta.values());boot=pitcher_bootstrap_2024(out,k);g={'2024':delta['2024']<=-1e-4,'2022':delta['2022']<=5e-5,'weighted':weighted<0,'worst':worst<=5e-5,'pitcher_bootstrap_ci_high_below_zero':boot['ci_high']<0};cand.append({'candidate_name':k,'candidate_status':'PENDING_AUDIT','metrics':mets,'delta_brier_vs_p109c':delta,'time_weighted_delta':weighted,'worst_season_delta':worst,'pitcher_cluster_bootstrap_2024':boot,'gate_results':g,'performance_gate_pass':all(g.values())})
 result={'experiment_id':OUT.name,'status':'PENDING_AUDIT','candidate_count':len(cand),'actual_leaf_count':len(cand),'gate_checks_count':sum(len(x['gate_results']) for x in cand),'candidates':cand,'model_count':len(paths),'oof_rows':len(out),'evaluation_note_110b_2022':'Frozen p109C warm-up control because no pre-2022 strict p103A OOF rows exist; 110B fitting begins with 2022 OOF for the 2023 fold.','test_read':False,'zip_created':False};(OUT/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
