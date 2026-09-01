"""Row-local production features for the audited F-only psych/latent residual."""
from __future__ import annotations
import pickle
import numpy as np
import pandas as pd

CONDITIONS=('high_li','extreme_li','traffic','risp','late','close','behind','three_ball','two_strike','full_count','compound_pressure')

def condition_frame(raw):
    li=pd.to_numeric(raw.li,errors='coerce').fillna(0);inn=pd.to_numeric(raw.inning,errors='coerce').fillna(0)
    score=pd.to_numeric(raw.score_diff_pitcher_team,errors='coerce').fillna(0);balls=pd.to_numeric(raw.balls_before,errors='coerce').fillna(0);strikes=pd.to_numeric(raw.strikes_before,errors='coerce').fillna(0);runners=pd.to_numeric(raw.num_runners_on,errors='coerce').fillna(0)
    risp=raw.runner_on_2b.fillna(0).astype(bool)|raw.runner_on_3b.fillna(0).astype(bool)
    return pd.DataFrame({'high_li':li>=1.5,'extreme_li':li>=3,'traffic':runners>0,'risp':risp,'late':inn>=7,'close':score.abs()<=1,'behind':score<0,'three_ball':balls==3,'two_strike':strikes==2,'full_count':(balls==3)&(strikes==2),'compound_pressure':(li>=1.5)&(risp|(balls==3))&(score.abs()<=2)},index=raw.index).astype('int8')

def build_production_features(raw,psych_path,latent_path):
    with open(psych_path,'rb') as f: bundle=pickle.load(f)
    profile=bundle['profile'];league=bundle['league'];cond=condition_frame(raw.reset_index(drop=True))
    psych=pd.DataFrame({'pitcher_id':raw.pitcher_id.astype(str).to_numpy()}).merge(profile,on='pitcher_id',how='left',sort=False).drop(columns='pitcher_id')
    psych['psych_history_log_n']=psych.psych_history_log_n.fillna(0)
    for name in CONDITIONS:
        effect=psych[f'psych_{name}_effect'].fillna(league[name]);rel=psych[f'psych_{name}_reliability'].fillna(0);active=cond[name].to_numpy(float)
        psych[f'psych_{name}_active_effect']=effect*active;psych[f'psych_{name}_active_reliability']=rel*active
        psych[f'psych_{name}_effect']=effect;psych[f'psych_{name}_reliability']=rel
    latent=pd.read_csv(latent_path,dtype={'pitcher_id':str,'batter_hand':str})
    keys=pd.DataFrame({'pitcher_id':raw.pitcher_id.astype(str),'season':raw.season.astype(int),'balls_before':raw.balls_before.astype(int),'strikes_before':raw.strikes_before.astype(int),'batter_hand':raw.batter_hand.map({1:'Left',2:'Right'}).fillna(raw.batter_hand.astype(str))})
    z=keys.merge(latent,on=['pitcher_id','season','balls_before','strikes_before','batter_hand'],how='left',sort=False).drop(columns=['pitcher_id','season','balls_before','strikes_before','batter_hand'])
    n=pd.to_numeric(z.latent_pitch_n,errors='coerce').fillna(0);z['latent_log_n']=np.log1p(n);z['latent_reliability']=n/(n+100)
    probs=z[[f'latent_{g}_prob' for g in ('fastball','breaking','offspeed','other')]].clip(1e-6,1);z['latent_entropy']=-(probs*np.log(probs)).sum(1);z['latent_fast_vs_break']=probs.latent_fastball_prob-probs.latent_breaking_prob
    return pd.concat([psych,z],axis=1).replace([np.inf,-np.inf],np.nan)

def apply_linear_residual(features,meta_path):
    z=np.load(meta_path,allow_pickle=False);cols=z['columns'].tolist();x=features.reindex(columns=cols).astype(float)
    standardized=((x-z['mean'])/z['std']).fillna(0).to_numpy()
    return standardized@z['coef']*float(z['scale'])
