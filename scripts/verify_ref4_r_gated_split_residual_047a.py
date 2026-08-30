#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'model/REF4-TRAINONLY-R-GATED-SPLIT-RESIDUAL-047A'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def ci(y,b,p,g,reps,seed):
 d=(b-y)**2-(p-y)**2; codes,u=pd.factorize(g,sort=True); sums=np.bincount(codes,weights=d,minlength=len(u)); counts=np.bincount(codes,minlength=len(u)); rng=np.random.default_rng(seed); values=np.empty(reps)
 for i in range(reps):
  q=rng.integers(0,len(u),len(u)); values[i]=sums[q].sum()/counts[q].sum()
 return float(d.mean()),float(np.quantile(values,.025)),float(np.quantile(values,.975)),len(u)
def main():
 c=json.loads((OUT/'audit_contract.json').read_text()); res=json.loads((OUT/'result.json').read_text()); man=json.loads((OUT/'audit_manifest.json').read_text()); checks=[]
 def ck(n,p,a,e=None): checks.append({'name':n,'checked':True,'pass':bool(p),'actual':a,'expected':e})
 bad=[]
 for rel,v in man['artifacts'].items():
  p=ROOT/rel
  if not p.is_file() or sha(p)!=v['sha256'] or p.stat().st_size!=v['size']: bad.append(rel)
 ck('manifest',not bad,bad,[]); ck('manifest_count',man['artifact_count']==len(man['artifacts']),len(man['artifacts']),man['artifact_count'])
 att=json.loads((ROOT/c['upstream_attestation']).read_text()); ck('upstream_binding',att['overall_status']=='AUDIT_VERIFIED' and att['mismatch_count']==0 and att['oof_predictions_sha256']==sha(ROOT/c['upstream_predictions']),{'status':att['overall_status'],'mismatch_count':att['mismatch_count'],'hash_match':att['oof_predictions_sha256']==sha(ROOT/c['upstream_predictions'])})
 up=pd.read_csv(ROOT/c['upstream_predictions']); raw=pd.read_csv(ROOT/c['official_train'],usecols=['row_id','game_type']); d=up.merge(raw,on='row_id',validate='one_to_one'); expected=np.where(d.game_type.eq('R'),d.candidate_prediction,d.baseline_prediction); saved=pd.read_csv(OUT/'oof_predictions.csv'); diff=float(np.max(np.abs(saved.gated_prediction.to_numpy()-expected))); ck('predictions',saved.row_id.tolist()==d.row_id.tolist() and diff<=1e-15,{'rows':len(saved),'max_abs_diff':diff},{'rows':len(d),'max_abs_diff':1e-15})
 fold={}
 for year,z in d.assign(gated=expected).groupby('season'):
  y=z.target.to_numpy(float); b=z.baseline_prediction.to_numpy(float); p=z.gated.to_numpy(float); v=ci(y,b,p,z.pitcher_id.to_numpy(),int(c['promotion_gate']['cluster_bootstrap_repetitions']),int(c['promotion_gate']['cluster_bootstrap_seed_'+str(year)])); ref=float(np.mean((y-y.mean())**2)); bss=(1-np.mean((p-y)**2)/ref)-(1-np.mean((b-y)**2)/ref); r=res['folds'][str(year)]; numeric=max(abs(v[0]-r['brier_gain']),abs(v[1]-r['cluster_ci']['ci_low']),abs(v[2]-r['cluster_ci']['ci_high']),abs(bss-r['bss_gain'])); ck('metrics_'+str(year),numeric<=1e-15,{'max_diff':numeric,'gain':v[0],'ci':v[1:3]}); fold[str(year)]={'gain':v[0],'ci_low':v[1],'bss_gain':bss}
 y=d.target.to_numpy(float); b=d.baseline_prediction.to_numpy(float); pg=float(np.mean((b-y)**2)-np.mean((expected-y)**2)); gates={'2023_brier_gain_positive':fold['2023']['gain']>0,'2024_brier_gain_positive':fold['2024']['gain']>0,'pooled_brier_gain_positive':pg>0,'worst_season_bss_gain_positive':min(v['bss_gain'] for v in fold.values())>0,'2023_cluster_ci_low_positive':fold['2023']['ci_low']>0,'2024_cluster_ci_low_positive':fold['2024']['ci_low']>0}; promotion=all(gates.values()); ck('pooled',abs(pg-res['pooled']['brier_gain'])<=1e-15,pg,res['pooled']['brier_gain']); ck('gate',gates==res['gate_checks'],gates,res['gate_checks']); ck('counts',res['actual_leaf_count']==man['leaf_count']==1 and res['gate_checks_count']==man['gate_count']==len(gates),[res['actual_leaf_count'],man['leaf_count'],res['gate_checks_count'],man['gate_count'],len(gates)]); ck('promotion',res['promotion_pass'] is promotion,res['promotion_pass'],promotion); ck('scope',not any(res[k] for k in ('training_performed','test_read','test_inference_performed','production_assets_created','candidate_bundle_created','zip_created')),{k:res[k] for k in ('training_performed','test_read','test_inference_performed','production_assets_created','candidate_bundle_created','zip_created')}); md=(OUT/'result.md').read_text(); ck('markdown',res['candidate_name'] in md and res['candidate_status'] in md and f'`{str(promotion).lower()}`' in md,res['candidate_name'])
 failures=[x['name'] for x in checks if not x['pass']]; report={'experiment_id':c['experiment_id'],'status':'AUDIT_VERIFIED' if not failures else 'AUDIT_FAIL_REPORT','checked_count':len(checks),'pass_count':len(checks)-len(failures),'fail_count':len(failures),'mismatch_count':len(failures),'failures':failures,'checks':checks,'recomputed_folds':fold,'pooled_brier_gain':pg,'expected_gate':gates,'promotion_pass':promotion}; rp=OUT/'validation_report.json'; rp.write_text(json.dumps(report,ensure_ascii=False,indent=2,default=lambda v:v.item() if hasattr(v,'item') else str(v))+'\n'); a={'experiment_id':c['experiment_id'],'overall_status':report['status'],'performance_status':'PASS' if promotion else 'FAIL','candidate_count':1,'leaf_count':1,'gate_count':len(gates),'checked_count':len(checks),'pass_count':len(checks)-len(failures),'fail_count':len(failures),'mismatch_count':len(failures),'audit_manifest_sha256':sha(OUT/'audit_manifest.json'),'validation_report_sha256':sha(rp),'validator_sha256':sha(Path(__file__)),'result_sha256':sha(OUT/'result.json'),'oof_predictions_sha256':sha(OUT/'oof_predictions.csv'),'upstream_predictions_sha256':sha(ROOT/c['upstream_predictions']),'preserved_zip_sha256':sha(ROOT/c['preserve_zip']),'promotion_pass':promotion,'training_performed':False,'test_read':False,'test_inference_performed':False,'production_assets_created':False,'candidate_bundle_created':False,'zip_created':False}; (OUT/'audit_attestation.json').write_text(json.dumps(a,indent=2)+'\n'); print(json.dumps({k:a[k] for k in ('overall_status','performance_status','checked_count','pass_count','fail_count','mismatch_count','promotion_pass')},indent=2))
 if failures: raise SystemExit(2)
if __name__=='__main__': main()
