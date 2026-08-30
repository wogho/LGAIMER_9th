#!/usr/bin/env python3
"""Preflight for the fixed train-only psych/latent residual."""
from __future__ import annotations

import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'model/REF4-TRAINONLY-PSYCH-LATENT-042A'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def main():
 c=json.loads((OUT/'audit_contract.json').read_text()); checks=[]
 def ck(name,ok,actual):checks.append({'name':name,'checked':True,'pass':bool(ok),'actual':actual})
 required=[ROOT/'data/train.csv',ROOT/'data/trackman_history.csv',ROOT/'01_제약과금지사항.md',ROOT/'start04_uptostage.md',ROOT/c['preserve_zip'],ROOT/'model/REF4-TRAINONLY-RECENT-STACK-041A/audit_attestation.json']+[ROOT/p for p in c['base_oof'].values()]+[ROOT/p for p in c['fold_mapping'].values()]
 for p in required:ck('exists:'+str(p.relative_to(ROOT)),p.is_file(),p.stat().st_size if p.is_file() else None)
 prev=json.loads(required[5].read_text())
 ck('041a_audit_verified_performance_fail',prev.get('overall_status')=='AUDIT_VERIFIED' and prev.get('performance_status')=='FAIL' and prev.get('mismatch_count')==0,prev)
 ck('single_hypothesis',isinstance(c.get('single_hypothesis'),str) and len(c['single_hypothesis'])>30,c.get('single_hypothesis'))
 ck('fixed_candidate',c['fixed_candidate']=={'psych_shrinkage_alpha':500.0,'latent_parent_strength':50.0,'latent_context_strength':100.0,'ridge_alpha':10000.0,'ridge_fit_intercept':False,'correction_scale':0.4,'recent_year_weight':1.0,'older_year_weight':0.55,'psych_feature_mode':'history_log_n_plus_all_11_condition_effect_reliability_active_effect_active_reliability','latent_feature_mode':'mapping_similarity_n_four_probabilities_log_n_reliability_entropy_fast_vs_break'},c['fixed_candidate'])
 ck('temporal_protocol',c['temporal_protocol']==[{'fit_seasons':[2022],'validation_season':2023},{'fit_seasons':[2022,2023],'validation_season':2024}],c['temporal_protocol'])
 raw=pd.read_csv(ROOT/'data/train.csv',usecols=['row_id','season','control_success'])
 ck('train_row_id_unique',raw.row_id.is_unique,int(raw.row_id.nunique()))
 ck('train_target_binary_finite',raw.control_success.isin([0,1]).all() and np.isfinite(raw.control_success).all(),{'rows':len(raw),'target_values':sorted(raw.control_success.unique().tolist())})
 ck('train_seasons',set(raw.season.unique())==set(range(2019,2025)),sorted(raw.season.unique().tolist()))
 tm=pd.read_csv(ROOT/'data/trackman_history.csv',usecols=['season'])
 ck('trackman_cutoff',int(tm.season.max())==2024,{'rows':len(tm),'min':int(tm.season.min()),'max':int(tm.season.max())})
 source={}
 for ys,p in c['base_oof'].items():
  y=int(ys); d=pd.read_csv(ROOT/p,usecols=['row_id','season','target','prediction'])
  source[ys]={'rows':len(d),'sha256':sha(ROOT/p)}
  ck('oof_'+ys,d.row_id.is_unique and d.season.eq(y).all() and d.target.isin([0,1]).all() and np.isfinite(d.prediction).all() and d.prediction.between(0,1).all(),source[ys])
  m=pd.read_csv(ROOT/c['fold_mapping'][ys])
  ck('mapping_'+ys,m[['pitcher_id','pitcher_trackman_id']].notna().all().all() and m.pitcher_id.is_unique and m.pitcher_trackman_id.is_unique,{'rows':len(m),'sha256':sha(ROOT/c['fold_mapping'][ys])})
 ck('cross_oof_row_ids_disjoint',sum(source[y]['rows'] for y in source)==len(set().union(*[set(pd.read_csv(ROOT/p,usecols=['row_id']).row_id) for p in c['base_oof'].values()])),source)
 ck('rollback_hash',sha(ROOT/c['preserve_zip'])=='ca6af4bdf26d4bc79d503cdc6a09b0057a7ea9bdd77e29012f9e459a50cd66b8',sha(ROOT/c['preserve_zip']))
 failures=[x['name'] for x in checks if not x['pass']]
 report={'experiment_id':c['experiment_id'],'status':'AUDIT_VERIFIED' if not failures else 'BLOCKED','checked_count':len(checks),'pass_count':len(checks)-len(failures),'fail_count':len(failures),'failures':failures,'checks':checks,'source_summary':source,'test_read':False,'training_performed':False,'zip_created':False}
 (OUT/'preflight_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 (OUT/'preflight_report.md').write_text(f"# {c['experiment_id']} preflight\n\n- status: `{report['status']}`\n- checked: `{len(checks)}`\n- failures: `{len(failures)}`\n")
 print(json.dumps({k:report[k] for k in ('status','checked_count','pass_count','fail_count','failures')},ensure_ascii=False,indent=2))
 if failures:raise SystemExit(2)
if __name__=='__main__':main()
