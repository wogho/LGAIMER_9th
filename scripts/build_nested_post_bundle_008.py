#!/usr/bin/env python3
from pathlib import Path
import json,shutil
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'model/COMBO-TM-FULL-006';OUT=ROOT/'model/COMBO-TM-POST-FULL-008'
def make(tr,parent,child,k):
 p=tr.groupby(parent,observed=True).control_success.mean().rename('_parent');g=tr.groupby(child,observed=True).control_success.agg(['mean','count']).reset_index();
 def get(row):
  key=tuple(row[c] for c in parent) if len(parent)>1 else row[parent[0]];return p.get(key,0.0)
 g['_parent']=g.apply(get,axis=1);g['dev']=g['count']*(g['mean']-g['_parent'])/(g['count']+k);return g[child+['dev']]
def main():
 tr=pd.read_csv(ROOT/'data/train.csv',encoding='utf-8-sig');OUT.mkdir(parents=True,exist_ok=True)
 for f in ['model.cbm','feature_columns.json','pitcher_count_lookup.csv','asof_pitcher_id_prior.csv','asof_batter_id_prior.csv','asof_pitchmix_prior.csv','trackman_count_lookup.csv','trackman_hand_lookup.csv'] :shutil.copy2(SRC/f,OUT/f)
 tr=tr.copy();tr['hand_key']=tr.pitcher_hand.astype(str)+'|'+tr.batter_hand.astype(str);tr['ahead_key']=(tr.strikes_before>tr.balls_before).astype(int);tr['count_key']=tr.balls_before.astype(str)+'|'+tr.strikes_before.astype(str);tr['runner_key']=(tr.num_runners_on>0).astype(int)
 specs=[('post_ph.csv',['pitcher_id'],['pitcher_id','hand_key'],300),('post_phac.csv',['pitcher_id','hand_key'],['pitcher_id','hand_key','ahead_key','count_key'],800),('post_pac.csv',['pitcher_id','ahead_key'],['pitcher_id','ahead_key','count_key'],800),('post_phr.csv',['pitcher_id','hand_key'],['pitcher_id','hand_key','runner_key'],2000)]
 for fn,p,c,k in specs:make(tr,p,c,k).to_csv(OUT/fn,index=False)
 report={'experiment_id':'COMBO-TM-POST-FULL-008','source_model':'COMBO-TM-FULL-006','official_train_only':True,'post_weights':[0.20,0.825,0.280,0.45],'lookup_files':[x[0] for x in specs],'status':'PASS_BUNDLE','submission_status':'HOLD'};(OUT/'post_bundle_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
