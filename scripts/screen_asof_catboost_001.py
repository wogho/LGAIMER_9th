#!/usr/bin/env python3
from __future__ import annotations
import json, platform, sys
from pathlib import Path
import catboost as cb
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.asof_state_features import STATE_COLUMNS, add_state_walkforward
BASE=["season","game_month","game_dayofweek","inning","top_bottom","game_type","balls_before","strikes_before","outs_before","run_top_before","run_bot_before","run_total_before","score_diff_home","score_diff_pitcher_team","runner_on_1b","runner_on_2b","runner_on_3b","num_runners_on","base_state","home_win_expectancy","away_win_expectancy","li","pitcher_id","batter_id","pitcher_hand","batter_hand","pitcher_team_id","batter_team_id","asof_pitcher_n","asof_pitcher_success_rate","asof_pitcher_reverse_rate","asof_pitcher_middle_rate","asof_pitcher_ball_rate","asof_pitcher_strike_rate","asof_pitcher_prev1_game_success_rate","asof_pitcher_prev3_game_success_rate","asof_pitcher_prev5_game_success_rate","asof_pitcher_prev1_game_middle_rate","asof_pitcher_prev3_game_middle_rate","asof_pitcher_prev5_game_middle_rate","asof_batter_n","asof_batter_success_rate","asof_batter_middle_rate","asof_pitcher_pitchmix_n","asof_pitcher_fastball_rate","asof_pitcher_breaking_rate","asof_pitcher_offspeed_rate"]
def brier(y,p): return float(np.mean((np.asarray(y)-np.asarray(p))**2))
def bss(y,p):
 r=float(np.mean(y)); return float(100000*(1-brier(y,p)/(r*(1-r))))
def prep(x,cols):
 d=x[cols].copy(); cats=[c for c in ["top_bottom","game_type","base_state","pitcher_hand","batter_hand"] if c in d]
 for c in cats: d[c]=d[c].astype("string").fillna("<NA>").astype(str)
 return d,[d.columns.get_loc(c) for c in cats]
def main():
 raw=pd.read_csv(ROOT/'data/train.csv',encoding='utf-8-sig'); y=raw.control_success.to_numpy(np.int8); tr=raw.season<2024; va=raw.season==2024
 state=add_state_walkforward(raw.drop(columns=['row_id','control_success']),2024).sort_index(); results={}
 for name,cols in [('base',BASE),('asof',BASE+STATE_COLUMNS)]:
  x,ci=prep(state,cols); tp=cb.Pool(x.loc[tr],label=y[tr],cat_features=ci,feature_names=cols); vp=cb.Pool(x.loc[va],label=y[va],cat_features=ci,feature_names=cols)
  m=cb.CatBoostClassifier(iterations=350,learning_rate=.05,depth=6,l2_leaf_reg=5,loss_function='Logloss',eval_metric='Logloss',thread_count=12,random_seed=42,allow_writing_files=False,verbose=False,early_stopping_rounds=60); m.fit(tp,eval_set=vp,verbose=False); p=m.predict_proba(vp)[:,1]; results[name]={'feature_count':len(cols),'best_iteration':int(m.get_best_iteration()),'brier':brier(y[va],p),'bss':bss(y[va],p),'mean':float(p.mean())}
 report={'experiment_id':'ASOF-CATBOOST-001','environment':{'python':platform.python_version(),'catboost':cb.__version__},'train_rows':int(tr.sum()),'valid_rows':int(va.sum()),'official_train_only':True,'test_used':False,'external_data_used':False,'state_columns':STATE_COLUMNS,'results':results,'delta_asof_minus_base':results['asof']['brier']-results['base']['brier'],'status':'PASS_SCREEN' if results['asof']['brier']<results['base']['brier'] else 'FAIL_SCREEN','submission_status':'HOLD'}
 out=ROOT/'model/ASOF-CATBOOST-001';out.mkdir(parents=True,exist_ok=True);(out/'screen_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
