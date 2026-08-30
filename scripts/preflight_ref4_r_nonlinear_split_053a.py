#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'model/REF4-TRAINONLY-R-NONLINEAR-SPLIT-053A'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 c=json.loads((OUT/'audit_contract.json').read_text());checks=[]
 def ck(n,p,a,e=None):checks.append({'name':n,'checked':True,'pass':bool(p),'actual':a,'expected':e})
 paths=[ROOT/c[k] for k in ('official_train','f_psych_predictions','f_psych_attestation','comparison_predictions','comparison_attestation','preserve_zip')]+[ROOT/'01_제약과금지사항.md',ROOT/'start04_uptostage.md',ROOT/'src/entity_context_split.py']+[ROOT/p for p in c['base_oof'].values()]
 for p in paths:ck('exists:'+str(p.relative_to(ROOT)),p.is_file(),p.stat().st_size if p.is_file() else None)
 for prefix in ('f_psych','comparison'):
  a=json.loads((ROOT/c[prefix+'_attestation']).read_text());ck(prefix+'_audit',a.get('overall_status')=='AUDIT_VERIFIED' and a.get('mismatch_count')==0 and a.get('oof_predictions_sha256')==sha(ROOT/c[prefix+'_predictions']),{'status':a.get('overall_status'),'performance':a.get('performance_status'),'mismatch':a.get('mismatch_count'),'hash_match':a.get('oof_predictions_sha256')==sha(ROOT/c[prefix+'_predictions'])})
 expected={'iterations':180,'depth':3,'learning_rate':.02,'loss_function':'RMSE','l2_leaf_reg':100.0,'random_strength':.2,'bootstrap_type':'Bernoulli','subsample':.8,'random_seed':530000,'thread_count':3};ck('fixed_model',c['model']==expected,c['model'],expected);ck('fixed_candidate',c['candidate_count']==1 and c['parameter_sweep'] is False and c['feature_count']==40 and c['fit_gate']==c['apply_gate']=='game_type == R',{'candidate':c['candidate_count'],'sweep':c['parameter_sweep'],'features':c['feature_count'],'fit':c['fit_gate']});ck('scope',not c['production_fit'] and not c['test_read'] and not c['zip_creation'],{'production':c['production_fit'],'test':c['test_read'],'zip':c['zip_creation']});ck('rollback',sha(ROOT/c['preserve_zip'])=='1f62100e3901410df69390549dbf7d7c80ecb9ae83dc829e0763254bff84bf6a',sha(ROOT/c['preserve_zip']))
 fail=[x['name'] for x in checks if not x['pass']];r={'experiment_id':c['experiment_id'],'status':'AUDIT_VERIFIED' if not fail else 'BLOCKED','checked_count':len(checks),'pass_count':len(checks)-len(fail),'fail_count':len(fail),'failures':fail,'checks':checks,'test_read':False,'training_performed':False,'zip_created':False};(OUT/'preflight_report.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');(OUT/'preflight_report.md').write_text(f"# {c['experiment_id']} preflight\n\n- status: `{r['status']}`\n- checked: `{len(checks)}`\n- failures: `{len(fail)}`\n");print(json.dumps({k:r[k] for k in ('status','checked_count','pass_count','fail_count','failures')},indent=2));
 if fail:raise SystemExit(2)
if __name__=='__main__':main()
