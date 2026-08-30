#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
TARGET='control_success'

def brier(y,p): return float(np.mean((y-p)**2))

def one(year, kind):
    base=f'BASE-LGBM-{year}-001'; agg=f'AGG-{kind.upper()}-{year}-001'
    tr=pd.read_csv(ROOT/'data/train.csv', usecols=['row_id','season','pitcher_id','balls_before','strikes_before','runner_on_2b','runner_on_3b',TARGET], encoding='utf-8-sig')
    tr=tr[tr.season.eq(year)].copy()
    bv=pd.read_csv(ROOT/'model'/base/'validation_predictions.csv')
    av=pd.read_csv(ROOT/'model'/agg/'validation_predictions.csv')
    assert len(bv)==len(tr)==len(av) and bv.row_id.equals(av.row_id)
    key='count_state' if kind=='count' else 'scoring_pos_runner'
    if kind=='count':
        full=tr.balls_before.eq(3)&tr.strikes_before.eq(2)
        pa=((tr.balls_before.eq(0)&tr.strikes_before.isin([1,2]))|(tr.balls_before.eq(1)&tr.strikes_before.eq(2)))
        ba=((tr.balls_before.eq(2)&tr.strikes_before.eq(0))|(tr.balls_before.eq(3)&tr.strikes_before.isin([0,1])))
        tr[key]=np.select([full,pa,ba],['full_count','ahead_pitcher','ahead_batter'],default='neutral')
        lookup='pitcher_count_target_lookup.csv'
        ncol='target_hist_pitcher_count_n'
    else:
        tr[key]=((tr.runner_on_2b.eq(1))|(tr.runner_on_3b.eq(1))).astype(int)
        lookup='pitcher_scoring_pos_target_lookup.csv'
        ncol='target_hist_pitcher_scoring_pos_n'
    lk=pd.read_csv(ROOT/'model'/agg/lookup)
    tr=tr.merge(lk[['pitcher_id',key,ncol]],on=['pitcher_id',key],how='left',validate='many_to_one')
    tr['support']=tr[ncol].fillna(0).astype(int)
    tr['bin']=pd.cut(tr.support,bins=[-1,0,30,100,300,1000,np.inf],labels=['unseen','1-30','31-100','101-300','301-1000','1001+'])
    out=[]
    for label,g in tr.groupby('bin',observed=False):
        idx=g.index.to_numpy(); y=g[TARGET].to_numpy(float); bp=bv.loc[idx,'pred_lgbm'].to_numpy(); ap=av.loc[idx,'pred_lgbm'].to_numpy()
        out.append({'support_bin':str(label),'rows':int(len(g)),'baseline_brier':brier(y,bp),'aggregate_brier':brier(y,ap),'delta':brier(y,ap)-brier(y,bp)})
    return {'year':year,'kind':kind,'rows':len(tr),'overall':{'baseline_brier':brier(bv['target'].to_numpy(),bv.pred_lgbm.to_numpy()),'aggregate_brier':brier(av['target'].to_numpy(),av.pred_lgbm.to_numpy())},'by_support':out}

def main():
    results=[one(y,k) for y in (2022,2024) for k in ('count','scorepos')]
    report={'experiment_id':'AGGREGATE-SUPPORT-001','official_train_only':True,'test_used':False,'external_data_used':False,'results':results}
    out=ROOT/'model/AGGREGATE-SUPPORT-001';out.mkdir(parents=True,exist_ok=True);(out/'support_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8');print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
