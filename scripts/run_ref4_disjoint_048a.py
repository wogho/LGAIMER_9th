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
def metric(y,p):
 b=float(np.mean((p-y)**2));ref=float(np.mean((y-y.mean())**2));q=1-b/ref;return {'rows':len(y),'target_rate':float(y.mean()),'brier':b,'bss':q,'local_score':1e5*q}
def boot(y,b,p,g,reps,seed):
 d=(b-y)**2-(p-y)**2;codes,u=pd.factorize(g,sort=True);s=np.bincount(codes,weights=d,minlength=len(u));n=np.bincount(codes,minlength=len(u));rng=np.random.default_rng(seed);v=np.empty(reps)
 for i in range(reps):q=rng.integers(0,len(u),len(u));v[i]=s[q].sum()/n[q].sum()
 return {'clusters':len(u),'repetitions':reps,'seed':seed,'brier_gain':float(d.mean()),'ci_low':float(np.quantile(v,.025)),'ci_high':float(np.quantile(v,.975))}
def main():
 c=json.loads((OUT/'audit_contract.json').read_text());pre=json.loads((OUT/'preflight_report.json').read_text());assert pre['status']=='AUDIT_VERIFIED' and pre['fail_count']==0;r=pd.read_csv(ROOT/c['r_predictions']);f=pd.read_csv(ROOT/c['f_predictions']);d=r[['row_id','season','pitcher_id','game_type','target','baseline_prediction']].copy();d['r_prediction']=r.gated_prediction;d['f_prediction']=f.gated_prediction;d['combined_prediction']=np.where(d.game_type.eq('R'),d.r_prediction,d.f_prediction);folds={}
 for year,z in d.groupby('season',sort=True):
  y=z.target.to_numpy(float);b=z.baseline_prediction.to_numpy(float);p=z.combined_prediction.to_numpy(float);bm=metric(y,b);cm=metric(y,p);ci=boot(y,b,p,z.pitcher_id.to_numpy(),int(c['promotion_gate']['cluster_bootstrap_repetitions']),int(c['promotion_gate']['cluster_bootstrap_seed_'+str(year)]));folds[str(year)]={'rows':len(z),'R_rows':int(z.game_type.eq('R').sum()),'F_rows':int(z.game_type.eq('F').sum()),'baseline':bm,'candidate':cm,'brier_gain':bm['brier']-cm['brier'],'bss_gain':cm['bss']-bm['bss'],'cluster_ci':ci}
 y=d.target.to_numpy(float);b=d.baseline_prediction.to_numpy(float);p=d.combined_prediction.to_numpy(float);pb=metric(y,b);pc=metric(y,p);pg=pb['brier']-pc['brier'];g={'2023_brier_gain_positive':folds['2023']['brier_gain']>0,'2024_brier_gain_positive':folds['2024']['brier_gain']>0,'pooled_brier_gain_positive':pg>0,'worst_season_bss_gain_positive':min(folds[k]['bss_gain'] for k in ('2023','2024'))>0,'2023_cluster_ci_low_positive':folds['2023']['cluster_ci']['ci_low']>0,'2024_cluster_ci_low_positive':folds['2024']['cluster_ci']['ci_low']>0};promotion=all(g.values());result={'experiment_id':c['experiment_id'],'candidate_name':'disjoint_R_split_F_psych','candidate_status':'PENDING_AUDIT_PASS' if promotion else 'PENDING_AUDIT_FAIL','folds':folds,'pooled':{'baseline':pb,'candidate':pc,'brier_gain':pg},'gate_checks':g,'gate_checks_count':len(g),'promotion_pass':promotion,'actual_leaf_count':1,'training_performed':False,'test_read':False,'test_inference_performed':False,'production_assets_created':False,'candidate_bundle_created':False,'zip_created':False};d.to_csv(OUT/'oof_predictions.csv',index=False);(OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');lines=[f"# {c['experiment_id']}",'',f"- candidate: `{result['candidate_name']}`",f"- status: `{result['candidate_status']}`",f"- promotion pass: `{str(promotion).lower()}`",'','| validation | baseline Brier | candidate Brier | gain | CI low | CI high |','|---|---:|---:|---:|---:|---:|']
 for year in ('2023','2024'):
  z=folds[year];lines.append(f"| {year} | {z['baseline']['brier']:.12f} | {z['candidate']['brier']:.12f} | {z['brier_gain']:.12f} | {z['cluster_ci']['ci_low']:.12f} | {z['cluster_ci']['ci_high']:.12f} |")
 lines.append(f"| pooled | {pb['brier']:.12f} | {pc['brier']:.12f} | {pg:.12f} |  |  | ");(OUT/'result.md').write_text('\n'.join(lines)+'\n');paths=[OUT/'audit_contract.json',OUT/'preflight_report.json',OUT/'preflight_report.md',OUT/'result.json',OUT/'result.md',OUT/'oof_predictions.csv',ROOT/c['r_predictions'],ROOT/c['r_attestation'],ROOT/c['f_predictions'],ROOT/c['f_attestation'],ROOT/c['preserve_zip'],ROOT/'01_제약과금지사항.md',ROOT/'scripts/preflight_ref4_disjoint_048a.py',ROOT/'scripts/run_ref4_disjoint_048a.py',ROOT/'scripts/verify_ref4_disjoint_048a.py'];arts={str(p.relative_to(ROOT)):{'sha256':sha(p),'size':p.stat().st_size} for p in paths};(OUT/'audit_manifest.json').write_text(json.dumps({'experiment_id':c['experiment_id'],'status':'PENDING_VALIDATION','artifact_count':len(arts),'artifacts':arts,'leaf_count':1,'gate_count':len(g),'oof_rows':len(d)},ensure_ascii=False,indent=2)+'\n');print(json.dumps({'candidate_status':result['candidate_status'],'promotion_pass':promotion,'gate_checks':g,'pooled_brier_gain':pg},indent=2))
if __name__=='__main__':main()
