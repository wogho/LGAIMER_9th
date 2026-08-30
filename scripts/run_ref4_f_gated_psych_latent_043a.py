#!/usr/bin/env python3
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'model/REF4-TRAINONLY-F-GATED-PSYCH-LATENT-043A'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for q in iter(lambda:f.read(1<<20),b''):h.update(q)
 return h.hexdigest()
def metric(y,p):
 b=float(np.mean((p-y)**2));ref=float(np.mean((y-y.mean())**2));return {'rows':len(y),'target_rate':float(y.mean()),'brier':b,'bss':1-b/ref,'local_score':1e5*(1-b/ref)}
def boot(y,b,p,g,reps,seed):
 d=(b-y)**2-(p-y)**2;codes,u=pd.factorize(g,sort=True);s=np.bincount(codes,weights=d,minlength=len(u));n=np.bincount(codes,minlength=len(u));rng=np.random.default_rng(seed);v=[]
 for _ in range(reps):
  q=rng.integers(0,len(u),len(u));v.append(s[q].sum()/n[q].sum())
 return {'clusters':len(u),'repetitions':reps,'seed':seed,'brier_gain':float(d.mean()),'ci_low':float(np.quantile(v,.025)),'ci_high':float(np.quantile(v,.975))}
def main():
 c=json.loads((OUT/'audit_contract.json').read_text());pre=json.loads((OUT/'preflight_report.json').read_text());assert pre['status']=='AUDIT_VERIFIED' and pre['fail_count']==0;up=pd.read_csv(ROOT/c['upstream_predictions']);train=pd.read_csv(ROOT/'data/train.csv',usecols=['row_id','game_type']);d=up.merge(train,on='row_id',how='left',validate='one_to_one');d['gated_prediction']=np.where(d.game_type.eq('F'),d.candidate_prediction,d.baseline_prediction);folds={}
 for y,z in d.groupby('season',sort=True):
  target=z.target.to_numpy(float);base=z.baseline_prediction.to_numpy(float);cand=z.gated_prediction.to_numpy(float);bm=metric(target,base);cm=metric(target,cand);cl=boot(target,base,cand,z.pitcher_id.to_numpy(),int(c['promotion_gate']['cluster_bootstrap_repetitions']),int(c['promotion_gate']['cluster_bootstrap_seed_'+str(y)]));folds[str(y)]={'rows':len(z),'F_rows':int(z.game_type.eq('F').sum()),'R_rows':int(z.game_type.ne('F').sum()),'baseline':bm,'candidate':cm,'brier_gain':bm['brier']-cm['brier'],'bss_gain':cm['bss']-bm['bss'],'cluster_ci':cl}
 ya=d.target.to_numpy(float);ba=d.baseline_prediction.to_numpy(float);ca=d.gated_prediction.to_numpy(float);pb=metric(ya,ba);pc=metric(ya,ca);pg=pb['brier']-pc['brier'];checks={'2023_brier_gain':folds['2023']['brier_gain']>0,'2024_brier_gain':folds['2024']['brier_gain']>0,'pooled_brier_gain':pg>0,'worst_season_bss_gain':min(folds[y]['bss_gain'] for y in ('2023','2024'))>0,'2023_cluster_ci_low':folds['2023']['cluster_ci']['ci_low']>0,'2024_cluster_ci_low':folds['2024']['cluster_ci']['ci_low']>0};passed=all(checks.values());result={'experiment_id':c['experiment_id'],'candidate_name':'fixed_042a_residual_F_rows_only','candidate_status':'PENDING_AUDIT_PASS' if passed else 'PENDING_AUDIT_FAIL','folds':folds,'pooled':{'baseline':pb,'candidate':pc,'brier_gain':pg},'gate_checks':checks,'gate_checks_count':len(checks),'promotion_pass':passed,'actual_leaf_count':1,'training_performed':False,'test_read':False,'test_inference_performed':False,'production_assets_created':False,'candidate_bundle_created':False,'zip_created':False};d[['row_id','season','pitcher_id','game_type','target','baseline_prediction','candidate_prediction','gated_prediction']].to_csv(OUT/'oof_predictions.csv',index=False);(OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');lines=[f"# {c['experiment_id']}",'',f"- candidate: `{result['candidate_name']}`",f"- status: `{result['candidate_status']}`",f"- promotion pass: `{str(passed).lower()}`",'','| validation | F rows | baseline Brier | candidate Brier | gain | CI low | CI high |','|---|---:|---:|---:|---:|---:|---:|']
 for y in ('2023','2024'):
  z=folds[y];lines.append(f"| {y} | {z['F_rows']} | {z['baseline']['brier']:.12f} | {z['candidate']['brier']:.12f} | {z['brier_gain']:.12f} | {z['cluster_ci']['ci_low']:.12f} | {z['cluster_ci']['ci_high']:.12f} |")
 lines.append(f"| pooled | {sum(folds[y]['F_rows'] for y in folds)} | {pb['brier']:.12f} | {pc['brier']:.12f} | {pg:.12f} |  |  |");(OUT/'result.md').write_text('\n'.join(lines)+'\n');paths=[OUT/'audit_contract.json',OUT/'preflight_report.json',OUT/'preflight_report.md',OUT/'result.json',OUT/'result.md',OUT/'oof_predictions.csv',ROOT/c['upstream_predictions'],ROOT/c['upstream_attestation'],ROOT/'data/train.csv',ROOT/'01_제약과금지사항.md',ROOT/'start04_uptostage.md',ROOT/c['preserve_zip'],ROOT/'scripts/preflight_ref4_f_gated_psych_latent_043a.py',ROOT/'scripts/run_ref4_f_gated_psych_latent_043a.py',ROOT/'scripts/verify_ref4_f_gated_psych_latent_043a.py'];arts={str(p.relative_to(ROOT)):{'sha256':sha(p),'size':p.stat().st_size} for p in paths};(OUT/'audit_manifest.json').write_text(json.dumps({'experiment_id':c['experiment_id'],'status':'PENDING_VALIDATION','artifact_count':len(arts),'artifacts':arts,'leaf_count':1,'gate_count':len(checks),'oof_rows':len(d)},ensure_ascii=False,indent=2)+'\n');print(json.dumps({'candidate_status':result['candidate_status'],'promotion_pass':passed,'gate_checks':checks,'brier_gains':{y:folds[y]['brier_gain'] for y in ('2023','2024')},'pooled_brier_gain':pg},indent=2))
if __name__=='__main__':main()
