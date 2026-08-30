#!/usr/bin/env python3
from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.asof_state_features import STATE_COLUMNS, add_state_for_cutoff
from src.target_aggregates import build_pitcher_count_state_target_history
BASE=["season","game_month","game_dayofweek","inning","top_bottom","game_type","balls_before","strikes_before","outs_before","run_top_before","run_bot_before","run_total_before","score_diff_home","score_diff_pitcher_team","runner_on_1b","runner_on_2b","runner_on_3b","num_runners_on","base_state","home_win_expectancy","away_win_expectancy","li","pitcher_id","batter_id","pitcher_hand","batter_hand","pitcher_team_id","batter_team_id","asof_pitcher_n","asof_pitcher_success_rate","asof_pitcher_reverse_rate","asof_pitcher_middle_rate","asof_pitcher_ball_rate","asof_pitcher_strike_rate","asof_pitcher_prev1_game_success_rate","asof_pitcher_prev3_game_success_rate","asof_pitcher_prev5_game_success_rate","asof_pitcher_prev1_game_middle_rate","asof_pitcher_prev3_game_middle_rate","asof_pitcher_prev5_game_middle_rate","asof_batter_n","asof_batter_success_rate","asof_batter_middle_rate","asof_pitcher_pitchmix_n","asof_pitcher_fastball_rate","asof_pitcher_breaking_rate","asof_pitcher_offspeed_rate"]
EXTRAS=[f'x_{r}_{c}' for r in ('cur_succ','cur_mid') for c in ('adv','onb','sh','bs')]+[f'h1_{r}_{c}' for r in ('cur_ball','cur_rev','cur_str') for c in ('sh','bs')]
def make(frame,history):
    out=add_state_for_cutoff(frame.drop(columns=['row_id']),history.drop(columns=['row_id','control_success'])).reset_index(drop=True)
    out['adv']=(out.strikes_before>out.balls_before).astype(float);out['onb']=(out.num_runners_on>0).astype(float);out['sh']=(out.pitcher_hand.astype(str)==out.batter_hand.astype(str)).astype(float);out['bs']=out.balls_before-out.strikes_before
    for f in EXTRAS:
        z=f.split('_');out[f]=out['_'.join(z[1:3])]*out[z[3]]
    dummy=frame.copy();dummy['control_success']=0
    _,av,lookup,_=build_pitcher_count_state_target_history(history,dummy,smoothing=100.0)
    out=pd.concat([out.reset_index(drop=True),av.reset_index(drop=True)],axis=1)
    cols=BASE+STATE_COLUMNS+EXTRAS+['target_hist_pitcher_count_n','target_hist_pitcher_count_delta']
    return out[cols],lookup
def main():
    train=pd.read_csv(ROOT/'data/train.csv',encoding='utf-8-sig');test=pd.read_csv(ROOT/'data/test.csv',encoding='utf-8-sig')
    assert test.row_id.is_unique and test.season.eq(2025).all()
    a,lookup=make(test,train);perm=test.sample(frac=1.0,random_state=7).reset_index(drop=True);b,_=make(perm,train)
    ka=pd.DataFrame({'row_id':test.row_id});kb=pd.DataFrame({'row_id':perm.row_id});a.index=ka.row_id;b.index=kb.row_id;b=b.loc[ka.row_id]
    numeric=[c for c in a.columns if c not in ['top_bottom','game_type','base_state','pitcher_hand','batter_hand']];maxdiff=float(np.nanmax(np.abs(a[numeric].to_numpy(float)-b[numeric].to_numpy(float))))
    assert a.shape==(len(test),73) and maxdiff==0.0 and not lookup.duplicated(['pitcher_id','count_state']).any()
    report={'experiment_id':'COMBO-INFERENCE-CONTRACT-001','official_train_only':True,'test_used_for_lookup':False,'external_data_used':False,'test_rows':len(test),'feature_count':a.shape[1],'lookup_rows':len(lookup),'row_id_unique':True,'permutation_max_numeric_diff':maxdiff,'status':'PASS_CONTRACT','submission_status':'HOLD'}
    out=ROOT/'model/COMBO-INFERENCE-CONTRACT-001';out.mkdir(parents=True,exist_ok=True);(out/'contract_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
