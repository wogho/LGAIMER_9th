#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'model/REF4-TRAINONLY-DISJOINT-RSPLIT-FPSYCH-048A'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 c=json.loads((OUT/'audit_contract.json').read_text()); checks=[]
 def ck(n,p,a,e=None):checks.append({'name':n,'checked':True,'pass':bool(p),'actual':a,'expected':e})
 paths=[ROOT/c[k] for k in ('r_predictions','r_attestation','f_predictions','f_attestation','preserve_zip')]+[ROOT/'01_제약과금지사항.md',ROOT/'start04_uptostage.md']
 for p in paths:ck('exists:'+str(p.relative_to(ROOT)),p.is_file(),p.stat().st_size if p.is_file() else None)
 for prefix in ('r','f'):
  att=json.loads((ROOT/c[prefix+'_attestation']).read_text()); ck(prefix+'_upstream',att.get('overall_status')=='AUDIT_VERIFIED' and att.get('mismatch_count')==0 and att.get('oof_predictions_sha256')==sha(ROOT/c[prefix+'_predictions']),{'status':att.get('overall_status'),'performance':att.get('performance_status'),'mismatch_count':att.get('mismatch_count'),'hash_match':att.get('oof_predictions_sha256')==sha(ROOT/c[prefix+'_predictions'])})
 r=pd.read_csv(ROOT/c['r_predictions']); f=pd.read_csv(ROOT/c['f_predictions']); cols=['row_id','season','pitcher_id','game_type','target','baseline_prediction']; aligned=r[cols].equals(f[cols]); base_diff=float(np.max(np.abs(r.baseline_prediction.to_numpy()-f.baseline_prediction.to_numpy())))
 ck('upstream_alignment',aligned and base_diff<=1e-15,{'rows_r':len(r),'rows_f':len(f),'base_max_diff':base_diff},{'aligned':True,'base_max_diff':1e-15})
 ck('fixed_contract',c['candidate_count']==1 and c['parameter_sweep'] is False and c['training_performed'] is False and c['combination']=={'R':'r_gated_prediction','F':'f_gated_prediction'},{'candidate_count':c['candidate_count'],'parameter_sweep':c['parameter_sweep'],'training_performed':c['training_performed'],'combination':c['combination']})
 ck('rollback_hash',sha(ROOT/c['preserve_zip'])=='fe5e7eb7731a7b16942a82b5ed144825a87accc05ecd508d69e578c68d288e2a',sha(ROOT/c['preserve_zip']))
 failures=[x['name'] for x in checks if not x['pass']]; report={'experiment_id':c['experiment_id'],'status':'AUDIT_VERIFIED' if not failures else 'BLOCKED','checked_count':len(checks),'pass_count':len(checks)-len(failures),'fail_count':len(failures),'failures':failures,'checks':checks,'test_read':False,'training_performed':False,'zip_created':False}; (OUT/'preflight_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n'); (OUT/'preflight_report.md').write_text(f"# {c['experiment_id']} preflight\n\n- status: `{report['status']}`\n- checked: `{len(checks)}`\n- failures: `{len(failures)}`\n"); print(json.dumps({k:report[k] for k in ('status','checked_count','pass_count','fail_count','failures')},indent=2));
 if failures:raise SystemExit(2)
if __name__=='__main__':main()
