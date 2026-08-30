#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'model/REF4-R-SPECIFIC-SPLIT-PRODUCTION-054A'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 c=json.loads((OUT/'audit_contract.json').read_text());checks=[]
 def ck(n,p,a,e=None):checks.append({'name':n,'checked':True,'pass':bool(p),'actual':a,'expected':e})
 paths=[ROOT/c['source_performance_attestation'],ROOT/c['official_train'],ROOT/'01_제약과금지사항.md',ROOT/'start04_uptostage.md',ROOT/'src/entity_context_split.py']+[ROOT/p for p in c['base_oof'].values()]
 for p in paths:ck('exists:'+str(p.relative_to(ROOT)),p.is_file(),p.stat().st_size if p.is_file() else None)
 a=json.loads(paths[0].read_text());ck('source',a.get('overall_status')=='AUDIT_VERIFIED' and a.get('performance_status')=='PASS' and a.get('promotion_pass') is True and a.get('mismatch_count')==0,{'status':a.get('overall_status'),'performance':a.get('performance_status'),'promotion':a.get('promotion_pass'),'mismatch':a.get('mismatch_count')});ck('fixed_fit',c['fit_gate']=='game_type == R' and c['feature_count']==40 and c['ridge_alpha']==10000.0 and c['season_weights']=={'2022':.3025,'2023':.55,'2024':1.0},{'gate':c['fit_gate'],'features':c['feature_count'],'alpha':c['ridge_alpha'],'weights':c['season_weights']});ck('scope',not c['test_read'] and not c['candidate_bundle_created'] and not c['zip_created'],{'test':c['test_read'],'candidate':c['candidate_bundle_created'],'zip':c['zip_created']});fail=[x['name'] for x in checks if not x['pass']];r={'experiment_id':c['experiment_id'],'status':'AUDIT_VERIFIED' if not fail else 'BLOCKED','checked_count':len(checks),'pass_count':len(checks)-len(fail),'fail_count':len(fail),'failures':fail,'checks':checks,'test_read':False,'training_performed':False,'zip_created':False};(OUT/'preflight_report.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n');(OUT/'preflight_report.md').write_text(f"# {c['experiment_id']} preflight\n\n- status: `{r['status']}`\n- checked: `{len(checks)}`\n- failures: `{len(fail)}`\n");print(json.dumps({k:r[k] for k in ('status','checked_count','pass_count','fail_count','failures')},indent=2));
 if fail:raise SystemExit(2)
if __name__=='__main__':main()
