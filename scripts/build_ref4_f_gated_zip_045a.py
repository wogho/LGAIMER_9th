#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,os,shutil,subprocess,sys,tempfile,zipfile
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'model/REF4-F-GATED-PSYCH-LATENT-ZIP-045A'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def ignore(_dir,names):return {n for n in names if n=='__pycache__' or n.endswith('.pyc') or n=='output' or n=='data'}
def run_bundle(source,test):
 with tempfile.TemporaryDirectory(prefix='ref4_045_run_') as td:
  run=Path(td)/'bundle';shutil.copytree(source,run,copy_function=os.link,ignore=ignore);(run/'data').mkdir();test.to_csv(run/'data/test.csv',index=False);proc=subprocess.run([sys.executable,'script.py'],cwd=run,text=True,capture_output=True,timeout=600)
  if proc.returncode:raise RuntimeError({'returncode':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr})
  pred=pd.read_csv(run/'output/submission.csv');return pred,{'stdout':proc.stdout.strip(),'stderr':proc.stderr.strip(),'returncode':proc.returncode}
def write_zip(candidate,target):
 files=sorted(p for p in candidate.rglob('*') if p.is_file() and '__pycache__' not in p.parts and p.suffix!='.pyc' and 'output' not in p.relative_to(candidate).parts and 'data' not in p.relative_to(candidate).parts)
 with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
  for p in files:
   rel=p.relative_to(candidate).as_posix();info=zipfile.ZipInfo(rel,date_time=(2020,1,1,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o100644<<16;z.writestr(info,p.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=6)
 return files
def main():
 c=json.loads((OUT/'audit_contract.json').read_text());pre=json.loads((OUT/'preflight_report.json').read_text());assert pre['status']=='AUDIT_VERIFIED' and pre['fail_count']==0;base=ROOT/c['base_candidate'];cand=ROOT/c['candidate_dir'];target=ROOT/c['output_zip'];prod=ROOT/c['production_assets']
 if cand.exists() or target.exists():raise RuntimeError('new candidate or ZIP path already exists')
 shutil.copytree(base,cand,copy_function=os.link,ignore=ignore)
 # Break the two hardlinks before editing so the preserved 030 candidate stays byte-identical.
 for relative in ('script.py','model/manifest.json'):
  path=cand/relative;content=path.read_bytes();path.unlink();path.write_bytes(content)
 (cand/'solution').mkdir(exist_ok=True)
 shutil.copy2(ROOT/'solution/LG_Aimers_솔루션_PPT_Phase2.pptx',cand/'solution/LG_Aimers_솔루션_PPT_Phase2.pptx')
 for name in ('psych_profile.pkl','latent_pitch_context.csv','psych_latent_meta.npz'):shutil.copy2(prod/name,cand/'model'/name)
 shutil.copy2(ROOT/'src/psych_latent_f_gated.py',cand/'src/psych_latent.py')
 script=(cand/'script.py').read_text();needle='    p = p + regime["transition_scale"] * transition.predict(tx)\n\n    p = np.clip(p, 1e-5, 1 - 1e-5)';replacement='    p = p + regime["transition_scale"] * transition.predict(tx)\n\n    # Audited train-only residual; the gate reads only the current row game_type.\n    if futures.any():\n        from src.psych_latent import build_production_features, apply_linear_residual\n        residual_x = build_production_features(\n            test, MODEL / "psych_profile.pkl", MODEL / "latent_pitch_context.csv"\n        )\n        correction = apply_linear_residual(residual_x, MODEL / "psych_latent_meta.npz")\n        p = np.where(futures, p + correction, p)\n\n    p = np.clip(p, 1e-5, 1 - 1e-5)'
 if script.count(needle)!=1:raise RuntimeError('inference insertion point mismatch')
 (cand/'script.py').write_text(script.replace(needle,replacement))
 meta=json.loads((cand/'model/manifest.json').read_text());meta.update({'f_gated_psych_latent':True,'psych_latent_gate':'game_type == F','psych_latent_scale':0.4,'psych_latent_experiment':'REF4-F-GATED-PSYCH-LATENT-PRODUCTION-044A'});(cand/'model/manifest.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
 test=pd.read_csv(ROOT/'data/test.csv');base_pred,base_log=run_bundle(base,test);full,full_log=run_bundle(cand,test);base_pred.to_csv(OUT/'base_predictions.csv',index=False);full.to_csv(OUT/'full_predictions.csv',index=False)
 if full.row_id.tolist()!=test.row_id.tolist() or base_pred.row_id.tolist()!=test.row_id.tolist():raise RuntimeError('full row_id mismatch')
 p=full.control_success.to_numpy(float);bp=base_pred.control_success.to_numpy(float);fmask=test.game_type.eq('F').to_numpy();r_diff=float(np.max(np.abs(p[~fmask]-bp[~fmask]))) if (~fmask).any() else 0.0;f_diff=float(np.max(np.abs(p[fmask]-bp[fmask]))) if fmask.any() else None
 single,single_log=run_bundle(cand,test.iloc[[0]].copy());perm_test=test.sample(frac=1,random_state=45045).reset_index(drop=True);perm,perm_log=run_bundle(cand,perm_test);aug_test=test.copy();extra=test.iloc[[0]].copy();extra['row_id']=extra.row_id.astype(str)+'_AUG';aug_test=pd.concat([aug_test,extra],ignore_index=True);aug,aug_log=run_bundle(cand,aug_test)
 full_map=full.set_index('row_id').control_success;singleton_diff=float(abs(single.control_success.iloc[0]-full_map.loc[single.row_id.iloc[0]]));perm_diff=float(np.max(np.abs(perm.set_index('row_id').loc[full.row_id].control_success.to_numpy()-p)));aug_diff=float(np.max(np.abs(aug.set_index('row_id').loc[full.row_id].control_success.to_numpy()-p)))
 files=write_zip(cand,target)
 with tempfile.TemporaryDirectory(prefix='ref4_045_zip_') as td:
  extracted=Path(td)/'bundle';zipfile.ZipFile(target).extractall(extracted);zip_pred,zip_log=run_bundle(extracted,test)
 zip_diff=float(np.max(np.abs(zip_pred.control_success.to_numpy(float)-p)));inventory=[]
 for f in files:
  rel=f.relative_to(cand).as_posix();inventory.append({'member':rel,'size':f.stat().st_size,'sha256':sha(f)})
 pd.DataFrame(inventory).to_csv(OUT/'package_inventory.csv',index=False)
 checks={'full_row_id_exact':full.row_id.tolist()==test.row_id.tolist(),'full_finite':bool(np.isfinite(p).all()),'full_range':bool(((p>=0)&(p<=1)).all()),'R_unchanged_vs_030':r_diff<=1e-12,'singleton_equivalence':singleton_diff<=1e-12,'permutation_equivalence':perm_diff<=1e-12,'augmentation_equivalence':aug_diff<=1e-12,'zip_full_equivalence':zip_diff<=1e-12,'base_zip_preserved':sha(ROOT/c['base_zip'])==c['preserve_zip_sha256']};passed=all(checks.values());result={'experiment_id':c['experiment_id'],'status':'PENDING_AUDIT_PASS' if passed else 'PENDING_AUDIT_FAIL','candidate_dir':c['candidate_dir'],'output_zip':c['output_zip'],'candidate_file_count':len(files),'zip_member_count':len(zipfile.ZipFile(target).infolist()),'zip_size':target.stat().st_size,'zip_sha256':sha(target),'test_rows':len(test),'test_game_type_counts':test.game_type.value_counts().to_dict(),'prediction_min':float(p.min()),'prediction_max':float(p.max()),'prediction_mean':float(p.mean()),'prediction_std':float(p.std()),'R_max_abs_change_vs_030':r_diff,'F_max_abs_change_vs_030':f_diff,'singleton_max_abs_diff':singleton_diff,'permutation_max_abs_diff':perm_diff,'augmentation_max_abs_diff':aug_diff,'zip_full_max_abs_diff':zip_diff,'checks':checks,'checked_count':len(checks),'pass_count':sum(checks.values()),'fail_count':len(checks)-sum(checks.values()),'logs':{'base':base_log,'full':full_log,'singleton':single_log,'permutation':perm_log,'augmentation':aug_log,'zip':zip_log},'training_performed':False,'test_read':True,'test_inference_performed':True,'production_assets_created':False,'candidate_bundle_created':True,'zip_created':True};(OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');(OUT/'result.md').write_text(f"# {c['experiment_id']}\n\n- status: `{result['status']}`\n- ZIP: `{c['output_zip']}`\n- ZIP SHA-256: `{result['zip_sha256']}`\n- test rows: `{len(test)}`\n- test F rows: `{int(fmask.sum())}`\n- row-independence max diff: `{max(singleton_diff,perm_diff,aug_diff):.17g}`\n- candidate↔ZIP max diff: `{zip_diff:.17g}`\n")
 paths=[OUT/'audit_contract.json',OUT/'preflight_report.json',OUT/'preflight_report.md',OUT/'base_predictions.csv',OUT/'full_predictions.csv',OUT/'package_inventory.csv',OUT/'result.json',OUT/'result.md',target,ROOT/c['base_zip'],ROOT/c['production_attestation'],ROOT/'scripts/preflight_ref4_f_gated_zip_045a.py',ROOT/'scripts/build_ref4_f_gated_zip_045a.py',ROOT/'scripts/verify_ref4_f_gated_zip_045a.py',ROOT/'src/psych_latent_f_gated.py',ROOT/'01_제약과금지사항.md',ROOT/'start04_uptostage.md']+files;arts={str(q.relative_to(ROOT)):{'sha256':sha(q),'size':q.stat().st_size} for q in paths};(OUT/'audit_manifest.json').write_text(json.dumps({'experiment_id':c['experiment_id'],'status':'PENDING_VALIDATION','artifact_count':len(arts),'artifacts':arts,'candidate_file_count':len(files),'zip_member_count':len(zipfile.ZipFile(target).infolist()),'dynamic_check_count':len(checks),'test_rows':len(test)},ensure_ascii=False,indent=2)+'\n');print(json.dumps({k:result[k] for k in ('status','zip_size','zip_sha256','test_rows','test_game_type_counts','R_max_abs_change_vs_030','F_max_abs_change_vs_030','singleton_max_abs_diff','permutation_max_abs_diff','augmentation_max_abs_diff','zip_full_max_abs_diff','checks')},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
