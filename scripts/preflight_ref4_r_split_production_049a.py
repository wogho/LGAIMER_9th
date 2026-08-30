#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'model/REF4-R-SPLIT-PRODUCTION-049A'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 c=json.loads((OUT/'audit_contract.json').read_text());checks=[]
 def ck(n,p,a,e=None):checks.append({'name':n,'checked':True,'pass':bool(p),'actual':a,'expected':e})
 required=[ROOT/c['source_performance_attestation'],ROOT/c['official_train'],ROOT/'01_제약과금지사항.md',ROOT/'start04_uptostage.md',ROOT/'src/entity_context_split.py']+[ROOT/p for p in c['base_oof'].values()]
 for p in required:ck('exists:'+str(p.relative_to(ROOT)),p.is_file(),p.stat().st_size if p.is_file() else None)
 a=json.loads(required[0].read_text());ck('performance_source',a.get('overall_status')=='AUDIT_VERIFIED' and a.get('performance_status')=='PASS' and a.get('mismatch_count')==0 and a.get('promotion_pass') is True,{'status':a.get('overall_status'),'performance':a.get('performance_status'),'mismatch_count':a.get('mismatch_count'),'promotion':a.get('promotion_pass')})
 ck('fixed_fit',c['feature_count']==40 and c['ridge_alpha']==10000.0 and c['season_weights']=={'2022':0.3025,'2023':0.55,'2024':1.0}, {'feature_count':c['feature_count'],'ridge_alpha':c['ridge_alpha'],'season_weights':c['season_weights']})
 ck('scope',c['test_read'] is False and c['candidate_bundle_created'] is False and c['zip_created'] is False,{'test_read':c['test_read'],'candidate':c['candidate_bundle_created'],'zip':c['zip_created']})
 failures=[x['name'] for x in checks if not x['pass']];r={'experiment_id':c['experiment_id'],'status':'AUDIT_VERIFIED' if not failures else 'BLOCKED','checked_count':len(checks),'pass_count':len(checks)-len(failures),'fail_count':len(failures),'failures':failures,'checks':checks,'test_read':False,'training_performed':False,'zip_created':False};(OUT/'preflight_report.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');(OUT/'preflight_report.md').write_text(f"# {c['experiment_id']} preflight\n\n- status: `{r['status']}`\n- checked: `{len(checks)}`\n- failures: `{len(failures)}`\n");print(json.dumps({k:r[k] for k in ('status','checked_count','pass_count','fail_count','failures')},indent=2));
 if failures:raise SystemExit(2)
if __name__=='__main__':main()
