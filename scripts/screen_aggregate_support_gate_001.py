#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def brier(y,p): return float(np.mean((y-p)**2))
def run(year,kind):
    tr=pd.read_csv(ROOT/'data/train.csv',usecols=['row_id','season','pitcher_id','balls_before','strikes_before','runner_on_2b','runner_on_3b','control_success'],encoding='utf-8-sig');tr=tr[tr.season.eq(year)].copy()
    base=pd.read_csv(ROOT/'model'/f'BASE-LGBM-{year}-001/validation_predictions.csv');agg=pd.read_csv(ROOT/'model'/f'AGG-{kind.upper()}-{year}-001/validation_predictions.csv')
    assert base.row_id.equals(agg.row_id) and len(tr)==len(base)
    keyname='count_state' if kind=='count' else 'scoring_pos_runner'
    if kind=='count':
        full=tr.balls_before.eq(3)&tr.strikes_before.eq(2);pa=((tr.balls_before.eq(0)&tr.strikes_before.isin([1,2]))|(tr.balls_before.eq(1)&tr.strikes_before.eq(2)));ba=((tr.balls_before.eq(2)&tr.strikes_before.eq(0))|(tr.balls_before.eq(3)&tr.strikes_before.isin([0,1])));tr['key']=np.select([full,pa,ba],['full_count','ahead_pitcher','ahead_batter'],default='neutral');lkname='pitcher_count_target_lookup.csv';ncol='target_hist_pitcher_count_n'
    else: tr['key']=((tr.runner_on_2b.eq(1))|(tr.runner_on_3b.eq(1))).astype(int);lkname='pitcher_scoring_pos_target_lookup.csv';ncol='target_hist_pitcher_scoring_pos_n'
    lk=pd.read_csv(ROOT/'model'/f'AGG-{kind.upper()}-{year}-001/{lkname}')[['pitcher_id',keyname,ncol]];tr=tr.merge(lk.rename(columns={keyname:'key'}),on=['pitcher_id','key'],how='left',validate='many_to_one');s=tr[ncol].fillna(0).to_numpy();y=base.target.to_numpy();bp=base.pred_lgbm.to_numpy();ap=agg.pred_lgbm.to_numpy();out=[]
    for t in (0,30,100,300,1000):
        p=np.where(s<=t,ap,bp);out.append({'aggregate_if_support_le':t,'brier':brier(y,p),'delta_vs_base':brier(y,p)-brier(y,bp),'aggregate_rows':int((s<=t).sum())})
    return {'year':year,'kind':kind,'base_brier':brier(y,bp),'aggregate_brier':brier(y,ap),'gates':out}
def main():
    d={'experiment_id':'AGGREGATE-SUPPORT-GATE-001','official_train_only':True,'test_used':False,'external_data_used':False,'results':[run(y,k) for y in (2022,2023,2024) for k in ('count','scorepos')]};o=ROOT/'model/AGGREGATE-SUPPORT-GATE-001';o.mkdir(parents=True,exist_ok=True);(o/'gate_report.json').write_text(json.dumps(d,ensure_ascii=False,indent=2));print(json.dumps(d,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
