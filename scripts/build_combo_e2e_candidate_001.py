#!/usr/bin/env python3
from pathlib import Path
import json,platform,sys
import catboost as cb
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.asof_state_features import STATE_COLUMNS,add_state_for_cutoff,add_state_walkforward
from src.target_aggregates import build_pitcher_count_state_target_history
BASE=["season","game_month","game_dayofweek","inning","top_bottom","game_type","balls_before","strikes_before","outs_before","run_top_before","run_bot_before","run_total_before","score_diff_home","score_diff_pitcher_team","runner_on_1b","runner_on_2b","runner_on_3b","num_runners_on","base_state","home_win_expectancy","away_win_expectancy","li","pitcher_id","batter_id","pitcher_hand","batter_hand","pitcher_team_id","batter_team_id","asof_pitcher_n","asof_pitcher_success_rate","asof_pitcher_reverse_rate","asof_pitcher_middle_rate","asof_pitcher_ball_rate","asof_pitcher_strike_rate","asof_pitcher_prev1_game_success_rate","asof_pitcher_prev3_game_success_rate","asof_pitcher_prev5_game_success_rate","asof_pitcher_prev1_game_middle_rate","asof_pitcher_prev3_game_middle_rate","asof_pitcher_prev5_game_middle_rate","asof_batter_n","asof_batter_success_rate","asof_batter_middle_rate","asof_pitcher_pitchmix_n","asof_pitcher_fastball_rate","asof_pitcher_breaking_rate","asof_pitcher_offspeed_rate"]
EXTRAS=[f'x_{r}_{c}' for r in ('cur_succ','cur_mid') for c in ('adv','onb','sh','bs')]+[f'h1_{r}_{c}' for r in ('cur_ball','cur_rev','cur_str') for c in ('sh','bs')]
COLS=BASE+STATE_COLUMNS+EXTRAS+['target_hist_pitcher_count_n','target_hist_pitcher_count_delta']
def prep(x):
 x=x.copy();cats=[c for c in x if str(x[c].dtype) in {'category','object'}]
 for c in cats:x[c]=x[c].astype('string').fillna('<NA>').astype(str)
 return x,cats
def state_features(frame,history):
 out=add_state_for_cutoff(frame.drop(columns=['row_id','control_success'],errors='ignore'),history.drop(columns=['row_id','control_success'],errors='ignore')).reset_index(drop=True)
 out['adv']=(out.strikes_before>out.balls_before).astype(float);out['onb']=(out.num_runners_on>0).astype(float);out['sh']=(out.pitcher_hand.astype(str)==out.batter_hand.astype(str)).astype(float);out['bs']=out.balls_before-out.strikes_before
 for f in EXTRAS:
  z=f.split('_');out[f]=out['_'.join(z[1:3])]*out[z[3]]
 return out
def main():
 raw=pd.read_csv(ROOT/'data/train.csv',encoding='utf-8-sig');test=pd.read_csv(ROOT/'data/test.csv',encoding='utf-8-sig');train=raw[raw.season<2024].copy();valid=raw[raw.season==2024].copy();y=train.control_success.to_numpy(np.int8);yv=valid.control_success.to_numpy(np.int8)
 walk=add_state_walkforward(raw.drop(columns=['row_id','control_success']),2024)
 state_train=walk.loc[train.index].reset_index(drop=True)
 state_valid=walk.loc[valid.index].reset_index(drop=True)
 for frame in (state_train,state_valid):
  frame['adv']=(frame.strikes_before>frame.balls_before).astype(float);frame['onb']=(frame.num_runners_on>0).astype(float);frame['sh']=(frame.pitcher_hand.astype(str)==frame.batter_hand.astype(str)).astype(float);frame['bs']=frame.balls_before-frame.strikes_before
  for f in EXTRAS:
   z=f.split('_');frame[f]=frame['_'.join(z[1:3])]*frame[z[3]]
 dummy=test.copy();dummy['control_success']=0;state_test=state_features(dummy,raw)
 at,av,lookup,_=build_pitcher_count_state_target_history(train,valid,smoothing=100.0);_,ate,test_lookup,_=build_pitcher_count_state_target_history(train,dummy,smoothing=100.0)
 xtrain=pd.concat([state_train.reset_index(drop=True),at.reset_index(drop=True)],axis=1)[COLS];xvalid=pd.concat([state_valid.reset_index(drop=True),av.reset_index(drop=True)],axis=1)[COLS];xtest=pd.concat([state_test.reset_index(drop=True),ate.reset_index(drop=True)],axis=1)[COLS]
 xtrain,cat=prep(xtrain);xvalid,_=prep(xvalid);xtest,_=prep(xtest);tp=cb.Pool(xtrain,label=y,cat_features=cat,feature_names=COLS);vp=cb.Pool(xvalid,label=yv,cat_features=cat,feature_names=COLS);m=cb.CatBoostClassifier(iterations=300,learning_rate=.05,depth=6,l2_leaf_reg=5,loss_function='Logloss',eval_metric='Logloss',thread_count=-1,random_seed=42,allow_writing_files=False,verbose=False,early_stopping_rounds=50);m.fit(tp,eval_set=vp,verbose=False);pv=m.predict_proba(vp)[:,1];pt=m.predict_proba(cb.Pool(xtest,cat_features=cat,feature_names=COLS))[:,1]
 out=ROOT/'model/COMBO-E2E-001';out.mkdir(parents=True,exist_ok=True);m.save_model(out/'model.cbm');(out/'feature_columns.json').write_text(json.dumps(COLS,ensure_ascii=False,indent=2));pd.DataFrame({'row_id':test.row_id,'prediction':pt}).to_csv(out/'test_predictions.csv',index=False);lookup.to_csv(out/'pitcher_count_lookup.csv',index=False);test_lookup.to_csv(out/'test_lookup_debug.csv',index=False)
 report={'experiment_id':'COMBO-E2E-001','environment':{'python':platform.python_version(),'catboost':cb.__version__},'official_train_only':True,'test_used_for_training':False,'external_data_used':False,'train_rows':len(train),'valid_rows':len(valid),'test_rows':len(test),'feature_count':len(COLS),'best_iteration':int(m.get_best_iteration()),'valid_brier':float(np.mean((yv-pv)**2)),'test_prediction_min':float(pt.min()),'test_prediction_max':float(pt.max()),'lookup_rows':len(lookup),'test_lookup_rows':len(test_lookup),'status':'PASS_E2E_MODEL','submission_status':'HOLD'};(out/'e2e_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
