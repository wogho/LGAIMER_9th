#!/usr/bin/env python3
import hashlib, json
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'model/REF4-TRAINONLY-R-GATED-SPLIT-RESIDUAL-047A'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def main():
 c=json.loads((OUT/'audit_contract.json').read_text()); checks=[]
 def ck(n,p,a,e=None): checks.append({'name':n,'checked':True,'pass':bool(p),'actual':a,'expected':e})
 paths=[ROOT/c['upstream_predictions'],ROOT/c['upstream_attestation'],ROOT/c['official_train'],ROOT/c['preserve_zip'],ROOT/'01_제약과금지사항.md',ROOT/'start04_uptostage.md']
 for p in paths: ck('exists:'+str(p.relative_to(ROOT)),p.is_file(),p.stat().st_size if p.is_file() else None)
 att=json.loads(paths[1].read_text()); ck('upstream_verified',att.get('overall_status')=='AUDIT_VERIFIED' and att.get('mismatch_count')==0 and att.get('oof_predictions_sha256')==sha(paths[0]),{'status':att.get('overall_status'),'performance_status':att.get('performance_status'),'mismatch_count':att.get('mismatch_count'),'hash_match':att.get('oof_predictions_sha256')==sha(paths[0])})
 ck('fixed_candidate',c['candidate_count']==1 and c['parameter_sweep'] is False and c['training_performed'] is False,{'candidate_count':c['candidate_count'],'parameter_sweep':c['parameter_sweep'],'training_performed':c['training_performed']})
 ck('fixed_gate',c['row_local_gate']=={'column':'game_type','apply_value':'R','R_prediction':'upstream_candidate_prediction','non_R_prediction':'upstream_baseline_prediction'},c['row_local_gate'])
 p=pd.read_csv(paths[0]); ck('upstream_rows',p.row_id.is_unique and set(p.season.unique())=={2023,2024} and p[['target','baseline_prediction','candidate_prediction']].notna().all().all(),{'rows':len(p),'seasons':sorted(p.season.unique().tolist())})
 raw=pd.read_csv(paths[2],usecols=['row_id','game_type']); m=p[['row_id']].merge(raw,on='row_id',validate='one_to_one'); ck('gate_source',m.game_type.notna().all() and set(m.game_type.unique()) <= {'F','R'},m.game_type.value_counts().to_dict())
 ck('rollback_hash',sha(paths[3])=='fe5e7eb7731a7b16942a82b5ed144825a87accc05ecd508d69e578c68d288e2a',sha(paths[3]))
 failures=[x['name'] for x in checks if not x['pass']]; report={'experiment_id':c['experiment_id'],'status':'AUDIT_VERIFIED' if not failures else 'BLOCKED','checked_count':len(checks),'pass_count':len(checks)-len(failures),'fail_count':len(failures),'failures':failures,'checks':checks,'test_read':False,'training_performed':False,'zip_created':False}
 (OUT/'preflight_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n'); (OUT/'preflight_report.md').write_text(f"# {c['experiment_id']} preflight\n\n- status: `{report['status']}`\n- checked: `{len(checks)}`\n- failures: `{len(failures)}`\n"); print(json.dumps({k:report[k] for k in ('status','checked_count','pass_count','fail_count','failures')},indent=2))
 if failures: raise SystemExit(2)
if __name__=='__main__': main()
