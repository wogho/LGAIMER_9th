#!/usr/bin/env python3
from pathlib import Path
import json,sys
import catboost as cb
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.asof_state_features import STATE_COLUMNS,add_state_for_cutoff,add_state_walkforward
from src.target_aggregates import build_pitcher_count_state_target_history
BASE=["season","game_month","game_dayofweek","inning","top_bottom","game_type","balls_before","strikes_before","outs_before","run_top_before","run_bot_before","run_total_before","score_diff_home","score_diff_pitcher_team","runner_on_1b","runner_on_2b","runner_on_3b","num_runners_on","base_state","home_win_expectancy","away_win_expectancy","li","pitcher_id","batter_id","pitcher_hand","batter_hand","pitcher_team_id","batter_team_id","asof_pitcher_n","asof_pitcher_success_rate","asof_pitcher_reverse_rate","asof_pitcher_middle_rate","asof_pitcher_ball_rate","asof_pitcher_strike_rate","asof_pitcher_prev1_game_success_rate","asof_pitcher_prev3_game_success_rate","asof_pitcher_prev5_game_success_rate","asof_pitcher_prev1_game_middle_rate","asof_pitcher_prev3_game_middle_rate","asof_pitcher_prev5_game_middle_rate","asof_batter_n","asof_batter_success_rate","asof_batter_middle_rate","asof_pitcher_pitchmix_n","asof_pitcher_fastball_rate","asof_pitcher_breaking_rate","asof_pitcher_offspeed_rate"]
EXTRAS=[f'x_{r}_{c}' for r in ('cur_succ','cur_mid') for c in ('adv','onb','sh','bs')]+[f'h1_{r}_{c}' for r in ('cur_ball','cur_rev','cur_str') for c in ('sh','bs')]
FK2=[f'f_{r}{w}' for r in ('succ','mid') for w in (1,3,5)]+[f'k2_{r}_{w}' for r in ('ball','rev') for w in ('two_strike','two_strike_low')]
COLS=BASE+STATE_COLUMNS+EXTRAS+FK2+['target_hist_pitcher_count_n','target_hist_pitcher_count_delta']
def prep(x):
 x=x.copy();cats=[c for c in x if str(x[c].dtype) in {'category','object'}]
 for c in cats:x[c]=x[c].astype('string').fillna('<NA>').astype(str)
 return x,cats
def extras(frame):
 frame=frame.copy();frame['adv']=(frame.strikes_before>frame.balls_before).astype(float);frame['onb']=(frame.num_runners_on>0).astype(float);frame['sh']=(frame.pitcher_hand.astype(str)==frame.batter_hand.astype(str)).astype(float);frame['bs']=frame.balls_before-frame.strikes_before
 for f in EXTRAS:
  z=f.split('_');frame[f]=frame['_'.join(z[1:3])]*frame[z[3]]
 return frame
def fk2(frame):
 frame=frame.copy()
 for r in ('succ','mid'):
  cur=frame[f'cur_{r}']
  for w in (1,3,5): frame[f'f_{r}{w}']=frame[f'asof_pitcher_prev{w}_game_{"success" if r=="succ" else "middle"}_rate']-cur
 two=(frame.strikes_before==2); low=(two & (frame.balls_before<=1))
 for r in ('ball','rev'):
  frame[f'k2_{r}_two_strike']=frame[f'cur_{r}']*two
  frame[f'k2_{r}_two_strike_low']=frame[f'cur_{r}']*low
 return frame
def prior_bundle(raw,out):
 specs=[('pitcher_id','asof_pitcher_n',['asof_pitcher_success_rate','asof_pitcher_middle_rate','asof_pitcher_ball_rate','asof_pitcher_reverse_rate','asof_pitcher_strike_rate']),('batter_id','asof_batter_n',['asof_batter_success_rate','asof_batter_middle_rate'])]
 for who,ncol,rates in specs:
  g=raw.groupby(who,sort=False)[ncol].max().rename('prior_n').reset_index()
  for rc in rates:
   ev=(raw[ncol].fillna(0)*raw[rc].fillna(0)).groupby(raw[who],sort=False).max().rename('prior_'+rc).reset_index();g=g.merge(ev,on=who,how='left',validate='one_to_one')
  g.to_csv(out/f'asof_{who}_prior.csv',index=False)
 raw.groupby('pitcher_id',sort=False)['asof_pitcher_pitchmix_n'].max().rename('prior_pitchmix_n').reset_index().to_csv(out/'asof_pitchmix_prior.csv',index=False)
def main():
 raw=pd.read_csv(ROOT/'data/train.csv',encoding='utf-8-sig');test=pd.read_csv(ROOT/'data/test.csv',encoding='utf-8-sig');dummy=test.copy();dummy['control_success']=0
 walk=fk2(extras(add_state_walkforward(raw.drop(columns=['row_id','control_success']),2025)));st=fk2(extras(add_state_for_cutoff(dummy.drop(columns=['row_id','control_success']),raw.drop(columns=['row_id','control_success']))))
 at,ate,lookup,_=build_pitcher_count_state_target_history(raw,dummy,smoothing=100.0);xtrain=pd.concat([walk.reset_index(drop=True),at.reset_index(drop=True)],axis=1)[COLS];xtest=pd.concat([st.reset_index(drop=True),ate.reset_index(drop=True)],axis=1)[COLS];xtrain,cat=prep(xtrain);xtest,_=prep(xtest);y=raw.control_success.to_numpy(np.int8)
 m=cb.CatBoostClassifier(iterations=100,learning_rate=.05,depth=6,l2_leaf_reg=5,loss_function='Logloss',thread_count=1,random_seed=42,allow_writing_files=False,verbose=False);m.fit(cb.Pool(xtrain,label=y,cat_features=cat,feature_names=COLS),verbose=False);out=ROOT/'model/COMBO-FK2-006';out.mkdir(parents=True,exist_ok=True);m.save_model(out/'model.cbm');(out/'feature_columns.json').write_text(json.dumps(COLS,ensure_ascii=False,indent=2));lookup.to_csv(out/'pitcher_count_lookup.csv',index=False);prior_bundle(raw,out);pred=m.predict_proba(cb.Pool(xtest,cat_features=cat,feature_names=COLS))[:,1];pd.DataFrame({'row_id':test.row_id,'control_success':pred}).to_csv(out/'test_predictions.csv',index=False);report={'experiment_id':'COMBO-FK2-006','official_train_only':True,'test_used_for_training':False,'external_data_used':False,'train_rows':len(raw),'test_rows':len(test),'feature_count':len(COLS),'tree_count':m.tree_count_,'lookup_rows':len(lookup),'prediction_min':float(pred.min()),'prediction_max':float(pred.max()),'status':'PASS_SCREEN_MODEL','submission_status':'HOLD'};(out/'full_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
