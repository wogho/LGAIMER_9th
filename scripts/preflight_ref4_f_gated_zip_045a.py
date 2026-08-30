#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'model/REF4-F-GATED-PSYCH-LATENT-ZIP-045A'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 c=json.loads((OUT/'audit_contract.json').read_text());checks=[]
 def ck(n,p,a):checks.append({'name':n,'checked':True,'pass':bool(p),'actual':a})
 paths=[ROOT/c['base_candidate'],ROOT/c['base_zip'],ROOT/c['production_attestation'],ROOT/c['production_assets'],ROOT/'data/test.csv',ROOT/'data/sample_submission.csv',ROOT/'src/psych_latent_f_gated.py']
 for p in paths:ck('exists:'+str(p.relative_to(ROOT)),p.exists(),p.stat().st_size if p.is_file() else ('dir' if p.is_dir() else None))
 a=json.loads(paths[2].read_text());ck('production_verified',a.get('overall_status')=='AUDIT_VERIFIED' and a.get('asset_status')=='PRODUCTION_ASSETS_VERIFIED' and a.get('mismatch_count')==0,a);ck('base_zip_preserved',sha(paths[1])==c['preserve_zip_sha256'],sha(paths[1]));test=pd.read_csv(paths[4]);ck('test_contract',test.row_id.is_unique and test.season.eq(2025).all() and test.game_type.isin(['F','R']).all(),{'rows':len(test),'game_type_counts':test.game_type.value_counts().to_dict()});ck('output_is_new',ROOT/c['output_zip']!=ROOT/c['base_zip'],c['output_zip']);fail=[x['name'] for x in checks if not x['pass']];r={'experiment_id':c['experiment_id'],'status':'AUDIT_VERIFIED' if not fail else 'BLOCKED','checked_count':len(checks),'pass_count':len(checks)-len(fail),'fail_count':len(fail),'failures':fail,'checks':checks,'test_rows':len(test),'test_game_type_counts':test.game_type.value_counts().to_dict(),'training_performed':False,'test_read':True,'test_inference_performed':False,'zip_created':False};(OUT/'preflight_report.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');(OUT/'preflight_report.md').write_text(f"# {c['experiment_id']} preflight\n\n- status: `{r['status']}`\n- checked: `{len(checks)}`\n- failures: `{len(fail)}`\n");print(json.dumps({k:r[k] for k in ('status','checked_count','pass_count','fail_count','failures','test_rows','test_game_type_counts')},indent=2));
 if fail:raise SystemExit(2)
if __name__=='__main__':main()
