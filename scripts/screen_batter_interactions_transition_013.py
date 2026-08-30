import json,sys
from pathlib import Path
import catboost as cb
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.asof_state_features import STATE_COLUMNS,add_state_walkforward
BASE=['season','game_month','game_dayofweek','inning','top_bottom','game_type','balls_before','strikes_before','outs_before','run_top_before','run_bot_before','run_total_before','score_diff_home','score_diff_pitcher_team','runner_on_1b','runner_on_2b','runner_on_3b','num_runners_on','base_state','home_win_expectancy','away_win_expectancy','li','pitcher_id','batter_id','pitcher_hand','batter_hand','pitcher_team_id','batter_team_id','asof_pitcher_n','asof_pitcher_success_rate','asof_pitcher_reverse_rate','asof_pitcher_middle_rate','asof_pitcher_ball_rate','asof_pitcher_strike_rate','asof_pitcher_prev1_game_success_rate','asof_pitcher_prev3_game_success_rate','asof_pitcher_prev5_game_success_rate','asof_pitcher_prev1_game_middle_rate','asof_pitcher_prev3_game_middle_rate','asof_pitcher_prev5_game_middle_rate','asof_batter_n','asof_batter_success_rate','asof_batter_middle_rate','asof_pitcher_pitchmix_n','asof_pitcher_fastball_rate','asof_pitcher_breaking_rate','asof_pitcher_offspeed_rate']
def main():
 raw=pd.read_csv(ROOT/'data/train.csv',encoding='utf-8-sig'); y=raw.control_success.to_numpy(np.int8); out=[]
 for year in (2022,2023,2024):
  va=raw.season==year; tr=raw.season<year; s=add_state_walkforward(raw.drop(columns=['row_id','control_success']),year).sort_index()
  s['adv']=(s.strikes_before>s.balls_before).astype(float);s['onb']=(s.num_runners_on>0).astype(float);s['sh']=(s.pitcher_hand.astype(str)==s.batter_hand.astype(str)).astype(float);s['bs']=s.balls_before-s.strikes_before
  extras=[]
  for r in ('cur_bsucc','cur_bmid'):
   for c in ('adv','onb','sh','bs'):
    f=f'bx_{r}_{c}';s[f]=s[r]*s[c];extras.append(f)
  cols=BASE+STATE_COLUMNS+extras; x=s[cols].copy(); cats=[c for c in ['top_bottom','game_type','base_state','pitcher_hand','batter_hand'] if c in x]
  for c in cats:x[c]=x[c].astype('string').fillna('<NA>').astype(str)
  tp=cb.Pool(x.loc[tr],label=y[tr],cat_features=cats,feature_names=cols);vp=cb.Pool(x.loc[va],label=y[va],cat_features=cats,feature_names=cols)
  m=cb.CatBoostClassifier(iterations=300,learning_rate=.05,depth=6,l2_leaf_reg=5,loss_function='Logloss',thread_count=3,random_seed=42,allow_writing_files=False,verbose=False,early_stopping_rounds=50);m.fit(tp,eval_set=vp,verbose=False);p=m.predict_proba(vp)[:,1]
  corr=1e5*np.corrcoef(p,y[va])[0,1]**2; out.append({'season':year,'feature_count':len(cols),'brier':float(np.mean((p-y[va])**2)),'metric':float(corr),'mean':float(p.mean()),'best_iteration':int(m.get_best_iteration())});print(out[-1],flush=True)
 report={'experiment_id':'BATTER-INTERACTIONS-TRANSITION-013','official_train_only':True,'test_used':False,'external_data_used':False,'results':out,'status':'SCREEN_COMPLETE'}; d=ROOT/'model/BATTER-INTERACTIONS-TRANSITION-013';d.mkdir(parents=True,exist_ok=True);(d/'screen_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
