#!/usr/bin/env python3
"""Research-only screen: train-only Trackman conditional deviations on COMBO-73."""
from pathlib import Path
import json, sys
import catboost as cb
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.asof_state_features import add_state_for_cutoff, add_state_walkforward
from src.target_aggregates import build_pitcher_count_state_target_history
from scripts.build_combo_full_candidate_002 import BASE, STATE_COLUMNS, EXTRAS, COLS as COMBO_COLS, extras, prep
TM_COLS=['season','pitcher_trackman_id','pitch_type_group','balls_before','strikes_before','batter_hand','rel_speed','spin_rate','induced_vert_break','horz_break','extension','rel_height','rel_side']
MAP=ROOT/'model/TRACKMAN-MAP-004/pitcher_id_map.csv'
PHYS=['rel_speed','spin_rate','induced_vert_break','horz_break','extension','rel_height','rel_side']
KINDS=['breaking','fastball','offspeed']
NEW=[f'tmc_{k}_dev' for k in KINDS]+['tmc_speed_dev']+[f'tmh_{k}_dev' for k in KINDS]+['tmh_speed_dev']
CAAFE=['cf_same_hand','cf_form5','cf_share_reverse','cf_share_ball','cf_mix_entropy','cf_log_pn','cf_trend13','cf_trend35','cf_midform1','cf_midtrend13','cf_ball_minus_strike']
def caafe(x):
    x=x.copy(); eps=1e-6
    ff=x.asof_pitcher_fastball_rate.fillna(0).clip(eps,1); fb=x.asof_pitcher_breaking_rate.fillna(0).clip(eps,1); fo=x.asof_pitcher_offspeed_rate.fillna(0).clip(eps,1); mix=pd.concat([ff,fb,fo],axis=1); mix=mix.div(mix.sum(axis=1),axis=0)
    return pd.DataFrame({'cf_same_hand':(x.pitcher_hand.astype(str)==x.batter_hand.astype(str)).astype(float),'cf_form5':x.asof_pitcher_prev5_game_success_rate-x.asof_pitcher_success_rate,'cf_share_reverse':x.asof_pitcher_reverse_rate/(1-x.asof_pitcher_success_rate).clip(eps),'cf_share_ball':x.asof_pitcher_ball_rate/(1-x.asof_pitcher_success_rate).clip(eps),'cf_mix_entropy':-(mix*np.log(mix)).sum(axis=1),'cf_log_pn':np.log1p(x.asof_pitcher_n.fillna(0)),'cf_trend13':x.asof_pitcher_prev1_game_success_rate-x.asof_pitcher_prev3_game_success_rate,'cf_trend35':x.asof_pitcher_prev3_game_success_rate-x.asof_pitcher_prev5_game_success_rate,'cf_midform1':x.asof_pitcher_prev1_game_middle_rate-x.asof_pitcher_middle_rate,'cf_midtrend13':x.asof_pitcher_prev1_game_middle_rate-x.asof_pitcher_prev3_game_middle_rate,'cf_ball_minus_strike':x.asof_pitcher_ball_rate-x.asof_pitcher_strike_rate},index=x.index)
def context_tables(tm, upto):
    past=tm[tm.season<upto]
    base=past.groupby('pitcher_id',observed=True)[PHYS].mean()
    out=[]
    for keys,prefix,min_n in [(['pitcher_id','balls_before','strikes_before'],'tmc',30),(['pitcher_id','batter_hand'],'tmh',50)]:
        g=past.groupby(keys,observed=True); q=g[PHYS].mean(); q['_n']=g.size(); q=q[q['_n']>=min_n].join(base,on='pitcher_id',rsuffix='_all')
        d={}
        for k in KINDS:d[f'{prefix}_{k}_dev']=q[f'frac_{k}']-q[f'frac_{k}_all'] if f'frac_{k}' in q else np.nan
        # proportions are computed from grouped counts below; physical speed uses mean
        for k in KINDS:
            # placeholder overwritten after count table construction
            pass
        counts=g['pitch_type_group'].value_counts().unstack(fill_value=0); counts=counts.reindex(q.index).fillna(0); total=counts.sum(axis=1).replace(0,np.nan)
        for k in KINDS:d[f'{prefix}_{k}_dev']=counts.get(k,pd.Series(0,index=counts.index))/total - (past.groupby('pitcher_id',observed=True)['pitch_type_group'].value_counts().unstack(fill_value=0).reindex(q.index).fillna(0).get(k,pd.Series(0,index=q.index))/past.groupby('pitcher_id',observed=True).size().reindex(q.index)).fillna(0).to_numpy()
        d[f'{prefix}_speed_dev']=q['rel_speed']-q['rel_speed_all']
        tab=pd.DataFrame(d,index=q.index);out.append(tab)
    return out
def load_tm():
    tm=pd.read_csv(ROOT/'data/trackman_history.csv',usecols=TM_COLS,encoding='utf-8-sig'); m=pd.read_csv(MAP);m=m[m.conf>=.9]
    tm=tm.merge(m[['pitcher_id','pitcher_trackman_id']],on='pitcher_trackman_id',how='inner',validate='many_to_one');tm['batter_hand']=tm['batter_hand'].map({1:'Left',2:'Right'}).fillna(tm['batter_hand'].astype(str));return tm,m
def attach(frame, tm, season):
    c,h=context_tables(tm,season); x=frame.copy(); x['__hand']=x['batter_hand'].map({1:'Left',2:'Right'}).fillna(x['batter_hand'].astype(str));
    x=x.join(c,on=['pitcher_id','balls_before','strikes_before']).join(h,on=['pitcher_id','__hand']).drop(columns='__hand');return x
def main():
    raw=pd.read_csv(ROOT/'data/train.csv',encoding='utf-8-sig');tm,m=load_tm();Y=2024
    tr=raw[raw.season<Y].copy();va=raw[raw.season==Y].copy(); hist=raw.drop(columns=['row_id','control_success']);
    a=add_state_walkforward(tr.drop(columns=['row_id','control_success']),Y); b=add_state_for_cutoff(va.drop(columns=['row_id','control_success']),tr.drop(columns=['row_id','control_success']));
    at,ate,lk,_=build_pitcher_count_state_target_history(raw[raw.season<Y],va,smoothing=100.0)
    xtr=pd.concat([extras(a.reset_index(drop=True)),at.reset_index(drop=True)],axis=1);xva=pd.concat([extras(b.reset_index(drop=True)),ate.reset_index(drop=True)],axis=1)
    xtr=attach(xtr,tm,Y);xva=attach(xva,tm,Y); xtr=pd.concat([xtr,caafe(xtr)],axis=1); xva=pd.concat([xva,caafe(xva)],axis=1); cols=COMBO_COLS+NEW; cols2=cols+CAAFE; xtr=xtr[cols2];xva=xva[cols2];xtr,cats=prep(xtr);xva,_=prep(xva);y=va.control_success.to_numpy();
    m1=cb.CatBoostClassifier(iterations=300,learning_rate=.05,depth=6,l2_leaf_reg=5,thread_count=-1,random_seed=42,allow_writing_files=False,verbose=False);m1.fit(cb.Pool(xtr,label=tr.control_success.to_numpy(),cat_features=cats,feature_names=cols2));p1=m1.predict_proba(cb.Pool(xva,cat_features=cats,feature_names=cols2))[:,1]
    def bs(p):return float(np.mean((p-y)**2))
    base_brier=float(json.loads((ROOT/'model/TRACKMAN-CONTEXT-003/screen_report.json').read_text())['base_brier']); r={'experiment_id':'TRACKMAN-CONTEXT-003-CAAFE','mapping_rows':len(m),'mapping_conf_ge_90_rows':int((m.conf>=.9).sum()),'trackman_rows':len(tm),'train_rows':len(tr),'valid_rows':len(va),'combo_feature_count':len(COMBO_COLS),'context_feature_count':len(NEW),'caafe_feature_count':len(CAAFE),'base_brier':base_brier,'context_caafe_brier':bs(p1),'delta_brier':bs(p1)-base_brier,'valid_context_coverage':float(xva[NEW].notna().any(axis=1).mean()),'status':'PASS_SCREEN' if bs(p1)<base_brier else 'REJECT','submission_status':'HOLD'}
    out=ROOT/'model/TRACKMAN-CONTEXT-003';out.mkdir(parents=True,exist_ok=True);(out/'screen_report.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps(r,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
