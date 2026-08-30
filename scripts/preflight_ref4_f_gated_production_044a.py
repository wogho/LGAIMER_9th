#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'model/REF4-F-GATED-PSYCH-LATENT-PRODUCTION-044A'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 c=json.loads((OUT/'audit_contract.json').read_text());checks=[]
 def ck(n,p,a):checks.append({'name':n,'checked':True,'pass':bool(p),'actual':a})
 paths=[ROOT/c['promotion_attestation'],ROOT/c['source_contract'],ROOT/c['production_mapping'],ROOT/'data/train.csv',ROOT/'data/trackman_history.csv',ROOT/c['preserve_zip']]+[ROOT/p for p in c['oof_sources'].values()]+[ROOT/p for p in c['fold_mapping'].values()]
 for p in paths:ck('exists:'+str(p.relative_to(ROOT)),p.is_file(),p.stat().st_size if p.is_file() else None)
 a=json.loads(paths[0].read_text());ck('promotion_verified',a.get('overall_status')=='AUDIT_VERIFIED' and a.get('performance_status')=='PASS' and a.get('promotion_pass') is True and a.get('mismatch_count')==0,a)
 ck('fixed_fit',c['fixed_fit']=={'fit_seasons':[2022,2023,2024],'season_weights':{'2022':0.3025,'2023':0.55,'2024':1.0},'psych_shrinkage_alpha':500.0,'latent_parent_strength':50.0,'latent_context_strength':100.0,'ridge_alpha':10000.0,'ridge_fit_intercept':False,'correction_scale':0.4,'inference_gate':'game_type == F'},c['fixed_fit'])
 ck('test_candidate_zip_blocked',c['test_read_allowed'] is False and c['candidate_bundle_allowed'] is False and c['zip_allowed'] is False,[c['test_read_allowed'],c['candidate_bundle_allowed'],c['zip_allowed']]);ck('rollback_hash',sha(ROOT/c['preserve_zip'])=='ca6af4bdf26d4bc79d503cdc6a09b0057a7ea9bdd77e29012f9e459a50cd66b8',sha(ROOT/c['preserve_zip']));fail=[x['name'] for x in checks if not x['pass']];r={'experiment_id':c['experiment_id'],'status':'AUDIT_VERIFIED' if not fail else 'BLOCKED','checked_count':len(checks),'pass_count':len(checks)-len(fail),'fail_count':len(fail),'failures':fail,'checks':checks,'test_read':False,'training_performed':False,'zip_created':False};(OUT/'preflight_report.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');(OUT/'preflight_report.md').write_text(f"# {c['experiment_id']} preflight\n\n- status: `{r['status']}`\n- checked: `{len(checks)}`\n- failures: `{len(fail)}`\n");print(json.dumps({k:r[k] for k in ('status','checked_count','pass_count','fail_count','failures')},indent=2));
 if fail:raise SystemExit(2)
if __name__=='__main__':main()
