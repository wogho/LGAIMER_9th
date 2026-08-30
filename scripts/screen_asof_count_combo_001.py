#!/usr/bin/env python3
from __future__ import annotations
import json, platform, sys
from pathlib import Path
import catboost as cb
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.asof_state_features import STATE_COLUMNS, add_state_walkforward
from src.target_aggregates import build_pitcher_count_state_target_history
BASE=["season","game_month","game_dayofweek","inning","top_bottom","game_type","balls_before","strikes_before","outs_before","run_top_before","run_bot_before","run_total_before","score_diff_home","score_diff_pitcher_team","runner_on_1b","runner_on_2b","runner_on_3b","num_runners_on","base_state","home_win_expectancy","away_win_expectancy","li","pitcher_id","batter_id","pitcher_hand","batter_hand","pitcher_team_id","batter_team_id","asof_pitcher_n","asof_pitcher_success_rate","asof_pitcher_reverse_rate","asof_pitcher_middle_rate","asof_pitcher_ball_rate","asof_pitcher_strike_rate","asof_pitcher_prev1_game_success_rate","asof_pitcher_prev3_game_success_rate","asof_pitcher_prev5_game_success_rate","asof_pitcher_prev1_game_middle_rate","asof_pitcher_prev3_game_middle_rate","asof_pitcher_prev5_game_middle_rate","asof_batter_n","asof_batter_success_rate","asof_batter_middle_rate","asof_pitcher_pitchmix_n","asof_pitcher_fastball_rate","asof_pitcher_breaking_rate","asof_pitcher_offspeed_rate"]
def br(y,p): return float(np.mean((np.asarray(y)-np.asarray(p))**2))
def prep(x):
    x=x.copy(); cats=[c for c in x if str(x[c].dtype) in {'category','object'}]
    for c in cats: x[c]=x[c].astype('string').fillna('<NA>').astype(str)
    return x,cats
def main():
    raw=pd.read_csv(ROOT/'data/train.csv',encoding='utf-8-sig'); y=raw.control_success.to_numpy(np.int8); year=2024; tr=raw.season<year; va=raw.season==year
    state=add_state_walkforward(raw.drop(columns=['row_id','control_success']),year).sort_index(); state['adv']=(state.strikes_before>state.balls_before).astype(float); state['onb']=(state.num_runners_on>0).astype(float); state['sh']=(state.pitcher_hand.astype(str)==state.batter_hand.astype(str)).astype(float); state['bs']=state.balls_before-state.strikes_before
    extras=[f'x_{r}_{c}' for r in ('cur_succ','cur_mid') for c in ('adv','onb','sh','bs')]+[f'h1_{r}_{c}' for r in ('cur_ball','cur_rev','cur_str') for c in ('sh','bs')]
    for f in extras:
        z=f.split('_'); state[f]=state['_'.join(z[1:3])]*state[z[3]]
    at,av,lookup,meta=build_pitcher_count_state_target_history(raw.loc[tr].copy(),raw.loc[va].copy(),smoothing=100.0); state=pd.concat([state.reset_index(drop=True),pd.concat([at,av],ignore_index=True).reset_index(drop=True)],axis=1)
    variants={'asof':BASE+STATE_COLUMNS,'combo':BASE+STATE_COLUMNS+extras+['target_hist_pitcher_count_n','target_hist_pitcher_count_delta']}; results={}
    for name,cols in variants.items():
        x,ci=prep(state[cols]); tp=cb.Pool(x.loc[tr],label=y[tr],cat_features=ci,feature_names=cols); vp=cb.Pool(x.loc[va],label=y[va],cat_features=ci,feature_names=cols)
        m=cb.CatBoostClassifier(iterations=300,learning_rate=.05,depth=6,l2_leaf_reg=5,loss_function='Logloss',eval_metric='Logloss',thread_count=-1,random_seed=42,allow_writing_files=False,verbose=False,early_stopping_rounds=50); m.fit(tp,eval_set=vp,verbose=False); p=m.predict_proba(vp)[:,1]; results[name]={'feature_count':len(cols),'best_iteration':int(m.get_best_iteration()),'brier':br(y[va],p),'mean':float(p.mean())}
    report={'experiment_id':'ASOF-COUNT-COMBO-001','environment':{'python':platform.python_version(),'catboost':cb.__version__},'official_train_only':True,'test_used':False,'external_data_used':False,'train_rows':int(tr.sum()),'valid_rows':int(va.sum()),'lookup_rows':len(lookup),'results':results,'delta_combo_vs_asof':results['combo']['brier']-results['asof']['brier'],'status':'PASS_SCREEN' if results['combo']['brier']<results['asof']['brier'] else 'FAIL_SCREEN','submission_status':'HOLD'}
    out=ROOT/'model/ASOF-COUNT-COMBO-001'; out.mkdir(parents=True,exist_ok=True); (out/'screen_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
