#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'model/REF4-R-SPECIFIC-SPLIT-ZIP-055A'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 c=json.loads((OUT/'audit_contract.json').read_text());checks=[]
 def ck(n,p,a,e=None):checks.append({'name':n,'checked':True,'pass':bool(p),'actual':a,'expected':e})
 required=[ROOT/c['base_zip'],ROOT/c['performance_attestation'],ROOT/c['production_attestation'],ROOT/c['production_dir']/'split_profile.csv',ROOT/c['production_dir']/'split_residual_meta.npz',ROOT/c['production_dir']/'production_probe.csv',ROOT/'src/entity_context_split.py',ROOT/'01_제약과금지사항.md',ROOT/'start04_uptostage.md']
 for p in required:ck('exists:'+str(p.relative_to(ROOT)),p.is_file(),p.stat().st_size if p.is_file() else None)
 base=ROOT/c['base_candidate'];files=sorted(p for p in base.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and 'data' not in p.relative_to(base).parts and 'output' not in p.relative_to(base).parts);ck('base_candidate',base.is_dir() and len(files)==84 and (base/'solution/LG_Aimers_솔루션_PPT_Phase2.pptx').is_file(),{'directory':base.is_dir(),'files':len(files),'ppt':(base/'solution/LG_Aimers_솔루션_PPT_Phase2.pptx').is_file()},{'files':84,'ppt':True});ck('base_zip_hash',sha(required[0])==c['base_zip_sha256'],sha(required[0]),c['base_zip_sha256'])
 perf=json.loads(required[1].read_text());prod=json.loads(required[2].read_text());ck('performance',perf.get('overall_status')=='AUDIT_VERIFIED' and perf.get('performance_status')=='PASS' and perf.get('promotion_pass') is True and perf.get('mismatch_count')==0,{'overall':perf.get('overall_status'),'performance':perf.get('performance_status'),'promotion':perf.get('promotion_pass'),'mismatch':perf.get('mismatch_count')});ck('production',prod.get('overall_status')=='AUDIT_VERIFIED' and prod.get('production_status')=='PRODUCTION_ASSETS_VERIFIED' and prod.get('mismatch_count')==0,{'overall':prod.get('overall_status'),'production':prod.get('production_status'),'mismatch':prod.get('mismatch_count')});ck('new_paths_absent',not (ROOT/c['candidate_dir']).exists() and not (ROOT/c['output_zip']).exists(),{'candidate_exists':(ROOT/c['candidate_dir']).exists(),'zip_exists':(ROOT/c['output_zip']).exists()});ck('fixed_candidate',c['candidate_count']==1 and c['training_performed'] is False and c['row_local_gate']=='game_type == R' and len(c['replacement_assets'])==3,{'candidate_count':c['candidate_count'],'training':c['training_performed'],'gate':c['row_local_gate'],'assets':c['replacement_assets']})
 fail=[x['name'] for x in checks if not x['pass']];r={'experiment_id':c['experiment_id'],'status':'AUDIT_VERIFIED' if not fail else 'BLOCKED','checked_count':len(checks),'pass_count':len(checks)-len(fail),'fail_count':len(fail),'failures':fail,'checks':checks,'test_read':False,'training_performed':False,'zip_created':False};(OUT/'preflight_report.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');(OUT/'preflight_report.md').write_text(f"# {c['experiment_id']} preflight\n\n- status: `{r['status']}`\n- checked: `{len(checks)}`\n- failures: `{len(fail)}`\n");print(json.dumps({k:r[k] for k in ('status','checked_count','pass_count','fail_count','failures')},indent=2))
 if fail:raise SystemExit(2)
if __name__=='__main__':main()
