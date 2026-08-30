#!/usr/bin/env python3
from pathlib import Path
import json
import catboost as cb
import numpy as np
import pandas as pd
_HERE=Path(__file__).resolve().parent
ROOT=_HERE if (_HERE/'model').exists() else _HERE.parent
BUNDLE=ROOT/'model/COMBO-E2E-001'
BASE=["season","game_month","game_dayofweek","inning","top_bottom","game_type","balls_before","strikes_before","outs_before","run_top_before","run_bot_before","run_total_before","score_diff_home","score_diff_pitcher_team","runner_on_1b","runner_on_2b","runner_on_3b","num_runners_on","base_state","home_win_expectancy","away_win_expectancy","li","pitcher_id","batter_id","pitcher_hand","batter_hand","pitcher_team_id","batter_team_id","asof_pitcher_n","asof_pitcher_success_rate","asof_pitcher_reverse_rate","asof_pitcher_middle_rate","asof_pitcher_ball_rate","asof_pitcher_strike_rate","asof_pitcher_prev1_game_success_rate","asof_pitcher_prev3_game_success_rate","asof_pitcher_prev5_game_success_rate","asof_pitcher_prev1_game_middle_rate","asof_pitcher_prev3_game_middle_rate","asof_pitcher_prev5_game_middle_rate","asof_batter_n","asof_batter_success_rate","asof_batter_middle_rate","asof_pitcher_pitchmix_n","asof_pitcher_fastball_rate","asof_pitcher_breaking_rate","asof_pitcher_offspeed_rate"]
STATE=['cur_succ','cur_mid','cur_ball','cur_rev','cur_str','cur_bsucc','cur_bmid','cur_logn_pitch','cur_logn_mix','cur_logn_bat']
EXTRAS=[f'x_{r}_{c}' for r in ('cur_succ','cur_mid') for c in ('adv','onb','sh','bs')]+[f'h1_{r}_{c}' for r in ('cur_ball','cur_rev','cur_str') for c in ('sh','bs')]
COLS=BASE+STATE+EXTRAS+['target_hist_pitcher_count_n','target_hist_pitcher_count_delta']
def main():
    t=pd.read_csv(ROOT/'data/test.csv',encoding='utf-8-sig');out=t.copy();pprior=pd.read_csv(BUNDLE/'asof_pitcher_id_prior.csv');bprior=pd.read_csv(BUNDLE/'asof_batter_id_prior.csv');mprior=pd.read_csv(BUNDLE/'asof_pitchmix_prior.csv')
    out=out.merge(pprior,on='pitcher_id',how='left',validate='many_to_one').merge(bprior,on='batter_id',how='left',validate='many_to_one').merge(mprior,on='pitcher_id',how='left',validate='many_to_one')
    curp=(out.asof_pitcher_n.fillna(0)-out.prior_n_x.fillna(0)).clip(lower=0);curb=(out.asof_batter_n.fillna(0)-out.prior_n_y.fillna(0)).clip(lower=0)
    out['cur_logn_pitch']=np.log1p(curp);out['cur_logn_bat']=np.log1p(curb);out['cur_logn_mix']=np.log1p((out.asof_pitcher_pitchmix_n.fillna(0)-out.prior_pitchmix_n.fillna(0)).clip(lower=0))
    specs=[('asof_pitcher_success_rate','cur_succ','prior_asof_pitcher_success_rate',curp,out.asof_pitcher_n),('asof_pitcher_middle_rate','cur_mid','prior_asof_pitcher_middle_rate',curp,out.asof_pitcher_n),('asof_pitcher_ball_rate','cur_ball','prior_asof_pitcher_ball_rate',curp,out.asof_pitcher_n),('asof_pitcher_reverse_rate','cur_rev','prior_asof_pitcher_reverse_rate',curp,out.asof_pitcher_n),('asof_pitcher_strike_rate','cur_str','prior_asof_pitcher_strike_rate',curp,out.asof_pitcher_n),('asof_batter_success_rate','cur_bsucc','prior_asof_batter_success_rate',curb,out.asof_batter_n),('asof_batter_middle_rate','cur_bmid','prior_asof_batter_middle_rate',curb,out.asof_batter_n)]
    for rc,lb,pc,cur,n in specs:out[lb]=((n.fillna(0)*out[rc].fillna(0)-out[pc].fillna(0))/cur).where(cur>0)
    out['adv']=(out.strikes_before>out.balls_before).astype(float);out['onb']=(out.num_runners_on>0).astype(float);out['sh']=(out.pitcher_hand.astype(str)==out.batter_hand.astype(str)).astype(float);out['bs']=out.balls_before-out.strikes_before
    full=(out.balls_before==3)&(out.strikes_before==2);pa=((out.balls_before==0)&out.strikes_before.isin([1,2]))|((out.balls_before==1)&(out.strikes_before==2));ba=((out.balls_before==2)&(out.strikes_before==0))|((out.balls_before==3)&out.strikes_before.isin([0,1]));out['count_state']=np.select([full,pa,ba],['full_count','ahead_pitcher','ahead_batter'],default='neutral')
    for f in EXTRAS:
        z=f.split('_');out[f]=out['_'.join(z[1:3])]*out[z[3]]
    lk=pd.read_csv(BUNDLE/'pitcher_count_lookup.csv');out=out.merge(lk,on=['pitcher_id','count_state'],how='left',validate='many_to_one');out[['target_hist_pitcher_count_n','target_hist_pitcher_count_delta']]=out[['target_hist_pitcher_count_n','target_hist_pitcher_count_delta']].fillna(0)
    x=out[COLS].copy();model=cb.CatBoostClassifier();model.load_model(BUNDLE/'model.cbm');cats=[COLS[i] for i in model.get_cat_feature_indices()]
    for c in cats:x[c]=x[c].astype(str)
    pred=model.predict_proba(cb.Pool(x,cat_features=cats,feature_names=COLS))[:,1];result=pd.DataFrame({'row_id':t.row_id,'control_success':pred});result.to_csv(BUNDLE/'isolated_submission.csv',index=False);(ROOT/'output').mkdir(exist_ok=True);result.to_csv(ROOT/'output/submission.csv',index=False);assert len(x.columns)==73 and np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all();print(json.dumps({'rows':len(result),'features':len(x.columns),'categorical':cats,'min':float(pred.min()),'max':float(pred.max()),'status':'PASS_ISOLATED_INFERENCE'},ensure_ascii=False))
if __name__=='__main__':main()
