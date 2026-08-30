#!/usr/bin/env python3
from pathlib import Path
import os
import json,sys
import catboost as cb
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.asof_state_features import add_state_for_cutoff,add_state_walkforward
from src.target_aggregates import build_pitcher_count_state_target_history
from scripts.build_combo_full_candidate_002 import COLS,extras,prep
def key(frame,cols): return pd.MultiIndex.from_frame(frame[cols],names=cols)
def dev_lookup(tr,va,parent,child,k):
    p=tr.groupby(parent,observed=True).control_success.mean(); g=tr.groupby(child,observed=True).control_success.agg(['mean','count']); par=key(g.reset_index(),parent if False else parent)
    # map parent values from child key columns without relying on arithmetic IDs
    child_df=g.reset_index(); child_df['_parent_mean']=child_df[parent].apply(lambda r: p.get(tuple(r) if len(parent)>1 else r.iloc[0],np.nan),axis=1); child_df['_dev']=child_df['count']*(child_df['mean']-child_df['_parent_mean'])/(child_df['count']+k); lut=child_df.set_index(child)[['_dev']]
    vals=va.merge(lut,left_on=child,right_index=True,how='left',sort=False,validate='many_to_one')['_dev'].fillna(0).to_numpy();return vals
def main():
    raw=pd.read_csv(ROOT/'data/train.csv',encoding='utf-8-sig');Y=int(os.environ.get('VALID_SEASON','2024'));tr=raw[raw.season<Y].copy();va=raw[raw.season==Y].copy();a=add_state_walkforward(tr.drop(columns=['row_id','control_success']),Y);b=add_state_for_cutoff(va.drop(columns=['row_id','control_success']),tr.drop(columns=['row_id','control_success']));at,ate,_,_=build_pitcher_count_state_target_history(tr,va,smoothing=100.0);xtr=pd.concat([extras(a.reset_index(drop=True)),at.reset_index(drop=True)],axis=1);xva=pd.concat([extras(b.reset_index(drop=True)),ate.reset_index(drop=True)],axis=1);xtr,cats=prep(xtr[COLS]);xva,_=prep(xva[COLS]);y=tr.control_success.to_numpy();yv=va.control_success.to_numpy();m=cb.CatBoostClassifier(iterations=200,learning_rate=.05,depth=6,l2_leaf_reg=5,thread_count=-1,random_seed=42,allow_writing_files=False,verbose=False);m.fit(cb.Pool(xtr,label=y,cat_features=cats,feature_names=COLS));p=m.predict_proba(cb.Pool(xva,cat_features=cats,feature_names=COLS))[:,1];
    # Four train-only nested conditionals: pitcher×hand, pitcher×ahead×count, pitcher×hand×count, pitcher×hand×runner.
    tr2=tr.reset_index(drop=True);va2=va.reset_index(drop=True);tr2['hand_key']=tr2.pitcher_hand.astype(str)+'|'+tr2.batter_hand.astype(str);va2['hand_key']=va2.pitcher_hand.astype(str)+'|'+va2.batter_hand.astype(str);tr2['ahead_key']=(tr2.strikes_before>tr2.balls_before).astype(int);va2['ahead_key']=(va2.strikes_before>va2.balls_before).astype(int);tr2['count_key']=tr2.balls_before.astype(str)+'|'+tr2.strikes_before.astype(str);va2['count_key']=va2.balls_before.astype(str)+'|'+va2.strikes_before.astype(str);tr2['runner_key']=(tr2.num_runners_on>0).astype(int);va2['runner_key']=(va2.num_runners_on>0).astype(int)
    d1=dev_lookup(tr2,va2,['pitcher_id'],['pitcher_id','hand_key'],300);d2=dev_lookup(tr2,va2,['pitcher_id','hand_key'],['pitcher_id','hand_key','ahead_key','count_key'],800);d3=dev_lookup(tr2,va2,['pitcher_id','ahead_key'],['pitcher_id','ahead_key','count_key'],800);d4=dev_lookup(tr2,va2,['pitcher_id','hand_key'],['pitcher_id','hand_key','runner_key'],2000);post=.20*d1+.825*d2+.280*d3+.45*d4;raw_b=float(np.mean((p-yv)**2));post_b=float(np.mean((np.clip(p+post,0,1)-yv)**2));r={'experiment_id':'NESTED-TARGET-POST-007','valid_season':Y,'train_rows':len(tr),'valid_rows':len(va),'feature_count':len(COLS),'base_brier':raw_b,'post_brier':post_b,'delta_brier':post_b-raw_b,'post_min':float(post.min()),'post_max':float(post.max()),'status':'PASS_SCREEN' if post_b<raw_b else 'REJECT','submission_status':'HOLD'};out=ROOT/'model/NESTED-TARGET-POST-007';out.mkdir(parents=True,exist_ok=True);(out/'screen_report.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps(r,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
