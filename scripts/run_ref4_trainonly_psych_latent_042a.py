#!/usr/bin/env python3
"""Run one fixed, strictly temporal psych/latent residual candidate."""
from __future__ import annotations

import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'model/REF4-TRAINONLY-PSYCH-LATENT-042A'
CONDS=('high_li','extreme_li','traffic','risp','late','close','behind','three_ball','two_strike','full_count','compound_pressure')
GROUPS=('fastball','breaking','offspeed','other')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def conditions(d):
 li=pd.to_numeric(d.li,errors='coerce').fillna(0);inn=pd.to_numeric(d.inning,errors='coerce').fillna(0);score=pd.to_numeric(d.score_diff_pitcher_team,errors='coerce').fillna(0);balls=pd.to_numeric(d.balls_before,errors='coerce').fillna(0);strikes=pd.to_numeric(d.strikes_before,errors='coerce').fillna(0);runners=pd.to_numeric(d.num_runners_on,errors='coerce').fillna(0);risp=d.runner_on_2b.fillna(0).astype(bool)|d.runner_on_3b.fillna(0).astype(bool)
 return pd.DataFrame({'high_li':li>=1.5,'extreme_li':li>=3,'traffic':runners>0,'risp':risp,'late':inn>=7,'close':score.abs()<=1,'behind':score<0,'three_ball':balls==3,'two_strike':strikes==2,'full_count':(balls==3)&(strikes==2),'compound_pressure':(li>=1.5)&(risp|(balls==3))&(score.abs()<=2)},index=d.index).astype('int8')
def psych(rows,history,alpha):
 h=history.reset_index(drop=True);cond=conditions(h);y=h.control_success.astype(float);pitch=h.pitcher_id.astype(str);g=float(y.mean());overall=h.assign(_y=y).groupby(pitch,sort=False)._y.agg(['sum','count']);rate=(overall['sum']+alpha*g)/(overall['count']+alpha);profile=pd.DataFrame(index=overall.index);profile['psych_history_log_n']=np.log1p(overall['count']);league={}
 for name in CONDS:
  mask=cond[name].astype(bool);ctx=float(y[mask].mean()) if mask.any() else g;league[name]=ctx-g;agg=h.loc[mask].assign(_y=y[mask]).groupby(pitch[mask],sort=False)._y.agg(['sum','count']);s=agg['sum'].reindex(profile.index).fillna(0);n=agg['count'].reindex(profile.index).fillna(0);r=(s+alpha*rate)/(n+alpha);profile['psych_'+name+'_effect']=r-rate;profile['psych_'+name+'_reliability']=n/(n+alpha)
 joined=pd.DataFrame({'pitcher_id':rows.pitcher_id.astype(str).to_numpy()}).merge(profile.reset_index(names='pitcher_id'),on='pitcher_id',how='left',sort=False).drop(columns='pitcher_id');joined['psych_history_log_n']=joined.psych_history_log_n.fillna(0);active=conditions(rows.reset_index(drop=True))
 for name in CONDS:
  e=joined['psych_'+name+'_effect'].fillna(league[name]);r=joined['psych_'+name+'_reliability'].fillna(0);a=active[name].to_numpy(float);joined['psych_'+name+'_active_effect']=e*a;joined['psych_'+name+'_active_reliability']=r*a;joined['psych_'+name+'_effect']=e;joined['psych_'+name+'_reliability']=r
 return joined.astype('float32')
def latent_table(target,mapping,tm,parent_strength,context_strength):
 m=mapping.copy();m['pitcher_id']=m.pitcher_id.astype(str);h=tm.loc[tm.season.lt(target)].merge(m[['pitcher_id','pitcher_trackman_id','mapping_similarity']],on='pitcher_trackman_id',how='inner');h['pitcher_id']=h.pitcher_id.astype(str);h['batter_hand']=h.batter_hand.astype(str);overall=h.groupby(['pitcher_id','pitch_type_group']).size().unstack(fill_value=0).reindex(columns=GROUPS,fill_value=0);total=overall.sum(1);prior=(overall+parent_strength*np.array([.5,.3,.15,.05]))/(total.to_numpy()[:,None]+parent_strength);ctx=h.groupby(['pitcher_id','balls_before','strikes_before','batter_hand','pitch_type_group']).size().unstack(fill_value=0).reindex(columns=GROUPS,fill_value=0);idx=ctx.index;par=prior.reindex(idx.get_level_values('pitcher_id')).to_numpy();n=ctx.sum(1).to_numpy();prob=(ctx.to_numpy()+context_strength*par)/(n[:,None]+context_strength);out=idx.to_frame(index=False);out['latent_pitch_n']=n;sim=m.set_index('pitcher_id').mapping_similarity;out['latent_mapping_similarity']=sim.reindex(out.pitcher_id).to_numpy()
 for j,g in enumerate(GROUPS):out['latent_'+g+'_prob']=prob[:,j]
 return out
def latent(rows,table):
 keys=pd.DataFrame({'pitcher_id':rows.pitcher_id.astype(str),'balls_before':rows.balls_before.astype(int),'strikes_before':rows.strikes_before.astype(int),'batter_hand':rows.batter_hand.map({1:'Left',2:'Right'}).fillna(rows.batter_hand.astype(str))});z=keys.merge(table,on=['pitcher_id','balls_before','strikes_before','batter_hand'],how='left',sort=False).drop(columns=['pitcher_id','balls_before','strikes_before','batter_hand']);n=pd.to_numeric(z.latent_pitch_n,errors='coerce').fillna(0);z['latent_log_n']=np.log1p(n);z['latent_reliability']=n/(n+100);probs=z[['latent_'+g+'_prob' for g in GROUPS]].clip(1e-6,1);z['latent_entropy']=-(probs*np.log(probs)).sum(1);z['latent_fast_vs_break']=probs.latent_fastball_prob-probs.latent_breaking_prob;return z.replace([np.inf,-np.inf],np.nan)
def standardize(train,valid):
 mean=train.mean();std=train.std().replace(0,1).fillna(1);return ((train-mean)/std).fillna(0),((valid-mean)/std).fillna(0),mean,std
def skill(y,p):
 b=float(np.mean((y-p)**2));ref=float(np.mean((y-y.mean())**2));return {'rows':len(y),'target_rate':float(y.mean()),'brier':b,'bss':1-b/ref,'local_score':1e5*(1-b/ref)}
def ci(y,b,c,g,reps,seed):
 gain=(b-y)**2-(c-y)**2;codes,u=pd.factorize(g,sort=True);s=np.bincount(codes,weights=gain,minlength=len(u));n=np.bincount(codes,minlength=len(u));rng=np.random.default_rng(seed);v=np.empty(reps)
 for i in range(reps):
  q=rng.integers(0,len(u),len(u));v[i]=s[q].sum()/n[q].sum()
 return {'clusters':len(u),'repetitions':reps,'seed':seed,'brier_gain':float(gain.mean()),'ci_low':float(np.quantile(v,.025)),'ci_high':float(np.quantile(v,.975))}
def main():
 c=json.loads((OUT/'audit_contract.json').read_text());pre=json.loads((OUT/'preflight_report.json').read_text())
 if pre['status']!='AUDIT_VERIFIED' or pre['fail_count']:raise RuntimeError('preflight not verified')
 raw=pd.read_csv(ROOT/'data/train.csv',low_memory=False);tm=pd.read_csv(ROOT/'data/trackman_history.csv',usecols=['pitcher_trackman_id','season','balls_before','strikes_before','batter_hand','pitch_type_group']);byid=raw.set_index('row_id',drop=False);frames={};features={};audits=[];cfg=c['fixed_candidate']
 for ys,p in c['base_oof'].items():
  year=int(ys);o=pd.read_csv(ROOT/p);rows=byid.loc[o.row_id].reset_index(drop=True);assert rows.row_id.tolist()==o.row_id.tolist() and rows.season.eq(year).all() and np.array_equal(rows.control_success.to_numpy(),o.target.to_numpy());mapping=pd.read_csv(ROOT/c['fold_mapping'][ys]);table=latent_table(year,mapping,tm,float(cfg['latent_parent_strength']),float(cfg['latent_context_strength']));x=pd.concat([psych(rows,raw.loc[raw.season.lt(year)],float(cfg['psych_shrinkage_alpha'])),latent(rows,table)],axis=1);frames[year]=(o,rows);features[year]=x;audits.append({'season':year,'rows':len(rows),'history_max_season':int(raw.loc[raw.season.lt(year), 'season'].max()),'trackman_max_season':int(tm.loc[tm.season.lt(year),'season'].max()),'mapping_rows':len(mapping),'latent_table_rows':len(table),'feature_rows':len(x),'feature_count':len(x.columns),'feature_names':x.columns.tolist(),'finite_after_standardization_required':True})
 preds=[];folds={};alpha=float(cfg['ridge_alpha']);scale=float(cfg['correction_scale'])
 for protocol in c['temporal_protocol']:
  vy=int(protocol['validation_season']);years=[int(x) for x in protocol['fit_seasons']];xtr=pd.concat([features[y] for y in years],ignore_index=True);ytr=np.concatenate([frames[y][0].target.to_numpy(float)-frames[y][0].prediction.to_numpy(float) for y in years]);weights=np.concatenate([np.full(len(frames[y][0]),float(cfg['recent_year_weight']) if y==max(years) else float(cfg['older_year_weight'])) for y in years]);ztr,zv,mean,std=standardize(xtr,features[vy]);model=Ridge(alpha=alpha,fit_intercept=False).fit(ztr,ytr,sample_weight=weights);base=frames[vy][0].prediction.to_numpy(float);target=frames[vy][0].target.to_numpy(float);cand=np.clip(base+scale*model.predict(zv),1e-5,1-1e-5);bm=skill(target,base);cm=skill(target,cand);cl=ci(target,base,cand,frames[vy][1].pitcher_id.to_numpy(),int(c['promotion_gate']['cluster_bootstrap_repetitions']),int(c['promotion_gate']['cluster_bootstrap_seed_'+str(vy)]));folds[str(vy)]={'fit_seasons':years,'baseline':bm,'candidate':cm,'brier_gain':bm['brier']-cm['brier'],'bss_gain':cm['bss']-bm['bss'],'cluster_ci':cl,'feature_count':len(xtr.columns),'ridge_coef':model.coef_.tolist(),'standardization_mean':mean.tolist(),'standardization_std':std.tolist()};preds.append(pd.DataFrame({'row_id':frames[vy][0].row_id,'season':vy,'pitcher_id':frames[vy][1].pitcher_id,'target':target,'baseline_prediction':base,'candidate_prediction':cand}))
 pred=pd.concat(preds,ignore_index=True);y=pred.target.to_numpy();b=pred.baseline_prediction.to_numpy();p=pred.candidate_prediction.to_numpy();pb=skill(y,b);pc=skill(y,p);pg=pb['brier']-pc['brier'];checks={'2023_brier_gain':folds['2023']['brier_gain']>0,'2024_brier_gain':folds['2024']['brier_gain']>0,'pooled_brier_gain':pg>0,'worst_season_bss_gain':min(folds[y]['bss_gain'] for y in ('2023','2024'))>0,'2023_cluster_ci_low':folds['2023']['cluster_ci']['ci_low']>0,'2024_cluster_ci_low':folds['2024']['cluster_ci']['ci_low']>0};passed=all(checks.values());result={'experiment_id':c['experiment_id'],'candidate_name':'fixed_psych_latent_ridge10000_scale040','candidate_status':'PENDING_AUDIT_PASS' if passed else 'PENDING_AUDIT_FAIL','source_audits':audits,'folds':folds,'pooled':{'baseline':pb,'candidate':pc,'brier_gain':pg},'gate_checks':checks,'gate_checks_count':len(checks),'promotion_pass':passed,'actual_leaf_count':1,'test_read':False,'test_inference_performed':False,'production_assets_created':False,'candidate_bundle_created':False,'zip_created':False};pred.to_csv(OUT/'oof_predictions.csv',index=False);(OUT/'result.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');lines=[f"# {c['experiment_id']}",'',f"- candidate: `{result['candidate_name']}`",f"- status: `{result['candidate_status']}`",f"- promotion pass: `{str(passed).lower()}`",'', '| validation | baseline Brier | candidate Brier | gain | CI low | CI high |','|---|---:|---:|---:|---:|---:|']
 for ys in ('2023','2024'):
  d=folds[ys];lines.append(f"| {ys} | {d['baseline']['brier']:.12f} | {d['candidate']['brier']:.12f} | {d['brier_gain']:.12f} | {d['cluster_ci']['ci_low']:.12f} | {d['cluster_ci']['ci_high']:.12f} |")
 lines.append(f"| pooled | {pb['brier']:.12f} | {pc['brier']:.12f} | {pg:.12f} |  |  |");(OUT/'result.md').write_text('\n'.join(lines)+'\n')
 paths=[OUT/'audit_contract.json',OUT/'preflight_report.json',OUT/'preflight_report.md',OUT/'result.json',OUT/'result.md',OUT/'oof_predictions.csv',ROOT/'data/train.csv',ROOT/'data/trackman_history.csv',ROOT/'01_제약과금지사항.md',ROOT/'start04_uptostage.md',ROOT/c['preserve_zip'],ROOT/'scripts/preflight_ref4_trainonly_psych_latent_042a.py',ROOT/'scripts/run_ref4_trainonly_psych_latent_042a.py',ROOT/'scripts/verify_ref4_trainonly_psych_latent_042a.py']+[ROOT/p for p in c['base_oof'].values()]+[ROOT/p for p in c['fold_mapping'].values()];arts={str(q.relative_to(ROOT)):{'sha256':sha(q),'size':q.stat().st_size} for q in paths};(OUT/'audit_manifest.json').write_text(json.dumps({'experiment_id':c['experiment_id'],'status':'PENDING_VALIDATION','artifact_count':len(arts),'artifacts':arts,'leaf_count':1,'gate_count':len(checks),'oof_rows':len(pred)},ensure_ascii=False,indent=2)+'\n');print(json.dumps({'candidate_status':result['candidate_status'],'promotion_pass':passed,'gate_checks':checks,'brier_gains':{y:folds[y]['brier_gain'] for y in ('2023','2024')},'pooled_brier_gain':pg},indent=2))
if __name__=='__main__':main()
