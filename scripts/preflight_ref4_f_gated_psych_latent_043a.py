#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'model/REF4-TRAINONLY-F-GATED-PSYCH-LATENT-043A'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 c=json.loads((OUT/'audit_contract.json').read_text());checks=[]
 def ck(n,p,a):checks.append({'name':n,'checked':True,'pass':bool(p),'actual':a})
 paths=[ROOT/c['upstream_predictions'],ROOT/c['upstream_attestation'],ROOT/'data/train.csv',ROOT/'01_제약과금지사항.md',ROOT/'start04_uptostage.md',ROOT/c['preserve_zip']]
 for p in paths:ck('exists:'+str(p.relative_to(ROOT)),p.is_file(),p.stat().st_size if p.is_file() else None)
 a=json.loads(paths[1].read_text());ck('upstream_verified',a.get('overall_status')=='AUDIT_VERIFIED' and a.get('mismatch_count')==0 and a.get('oof_predictions_sha256')==sha(paths[0]),{'status':a.get('overall_status'),'mismatch_count':a.get('mismatch_count'),'prediction_hash_match':a.get('oof_predictions_sha256')==sha(paths[0])})
 ck('one_candidate',c.get('candidate_count')==1 and c.get('parameter_sweep') is False,{'candidate_count':c.get('candidate_count'),'parameter_sweep':c.get('parameter_sweep')})
 ck('fixed_gate',c.get('row_local_gate')=={'column':'game_type','apply_value':'F','F_prediction':'upstream_candidate_prediction','non_F_prediction':'upstream_baseline_prediction'},c.get('row_local_gate'))
 p=pd.read_csv(paths[0]);ck('prediction_rows',p.row_id.is_unique and set(p.season.unique())=={2023,2024} and p[['target','baseline_prediction','candidate_prediction']].notna().all().all(),{'rows':len(p),'seasons':sorted(p.season.unique().tolist())})
 train=pd.read_csv(paths[2],usecols=['row_id','game_type']);m=p[['row_id']].merge(train,on='row_id',how='left',validate='one_to_one');ck('gate_source_complete',m.game_type.notna().all() and set(m.game_type.unique()).issubset({'F','R'}),m.game_type.value_counts().to_dict())
 ck('rollback_hash',sha(paths[5])=='ca6af4bdf26d4bc79d503cdc6a09b0057a7ea9bdd77e29012f9e459a50cd66b8',sha(paths[5]));fail=[x['name'] for x in checks if not x['pass']];r={'experiment_id':c['experiment_id'],'status':'AUDIT_VERIFIED' if not fail else 'BLOCKED','checked_count':len(checks),'pass_count':len(checks)-len(fail),'fail_count':len(fail),'failures':fail,'checks':checks,'test_read':False,'training_performed':False,'zip_created':False};(OUT/'preflight_report.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');(OUT/'preflight_report.md').write_text(f"# {c['experiment_id']} preflight\n\n- status: `{r['status']}`\n- checked: `{len(checks)}`\n- failures: `{len(fail)}`\n");print(json.dumps({k:r[k] for k in ('status','checked_count','pass_count','fail_count','failures')},indent=2));
 if fail:raise SystemExit(2)
if __name__=='__main__':main()
