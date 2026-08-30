#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'model/REF4-TRAINONLY-R-SPECIFIC-SPLIT-051A'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 c=json.loads((OUT/'audit_contract.json').read_text());checks=[]
 def ck(n,p,a,e=None):checks.append({'name':n,'checked':True,'pass':bool(p),'actual':a,'expected':e})
 paths=[ROOT/c['official_train'],ROOT/c['f_psych_predictions'],ROOT/c['f_psych_attestation'],ROOT/c['comparison_predictions'],ROOT/c['comparison_attestation'],ROOT/c['preserve_zip'],ROOT/'01_제약과금지사항.md',ROOT/'start04_uptostage.md',ROOT/'src/entity_context_split.py']+[ROOT/p for p in c['base_oof'].values()]
 for p in paths:ck('exists:'+str(p.relative_to(ROOT)),p.is_file(),p.stat().st_size if p.is_file() else None)
 for prefix in ('f_psych','comparison'):
  att=json.loads((ROOT/c[prefix+'_attestation']).read_text());ck(prefix+'_audit',att.get('overall_status')=='AUDIT_VERIFIED' and att.get('mismatch_count')==0 and att.get('oof_predictions_sha256')==sha(ROOT/c[prefix+'_predictions']),{'status':att.get('overall_status'),'performance':att.get('performance_status'),'mismatch':att.get('mismatch_count'),'hash_match':att.get('oof_predictions_sha256')==sha(ROOT/c[prefix+'_predictions'])})
 ck('fixed_contract',c['candidate_count']==1 and c['parameter_sweep'] is False and c['feature_count']==40 and c['ridge_alpha']==10000.0 and c['fit_gate']==c['apply_gate']=='game_type == R',{'candidate_count':c['candidate_count'],'sweep':c['parameter_sweep'],'features':c['feature_count'],'alpha':c['ridge_alpha'],'fit':c['fit_gate'],'apply':c['apply_gate']});ck('scope',c['production_fit'] is False and c['test_read'] is False and c['zip_creation'] is False,{'production':c['production_fit'],'test':c['test_read'],'zip':c['zip_creation']});ck('rollback_hash',sha(ROOT/c['preserve_zip'])=='1f62100e3901410df69390549dbf7d7c80ecb9ae83dc829e0763254bff84bf6a',sha(ROOT/c['preserve_zip']))
 base=pd.concat([pd.read_csv(ROOT/c['base_oof'][str(y)],usecols=['row_id','season','game_type','target','prediction']) for y in (2023,2024)],ignore_index=True);f=pd.read_csv(ROOT/c['f_psych_predictions']);comp=pd.read_csv(ROOT/c['comparison_predictions']);ck('alignment',base.row_id.tolist()==f.row_id.tolist()==comp.row_id.tolist() and float(np.max(np.abs(base.prediction.to_numpy()-f.baseline_prediction.to_numpy())))<=1e-15 and float(np.max(np.abs(base.prediction.to_numpy()-comp.baseline_prediction.to_numpy())))<=1e-15,{'rows':len(base)})
 fail=[x['name'] for x in checks if not x['pass']];r={'experiment_id':c['experiment_id'],'status':'AUDIT_VERIFIED' if not fail else 'BLOCKED','checked_count':len(checks),'pass_count':len(checks)-len(fail),'fail_count':len(fail),'failures':fail,'checks':checks,'test_read':False,'training_performed':False,'zip_created':False};(OUT/'preflight_report.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');(OUT/'preflight_report.md').write_text(f"# {c['experiment_id']} preflight\n\n- status: `{r['status']}`\n- checked: `{len(checks)}`\n- failures: `{len(fail)}`\n");print(json.dumps({k:r[k] for k in ('status','checked_count','pass_count','fail_count','failures')},indent=2));
 if fail:raise SystemExit(2)
if __name__=='__main__':main()
