#!/usr/bin/env python3
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
RATE=[('pitcher_id','asof_pitcher_n','asof_pitcher_success_rate'),('pitcher_id','asof_pitcher_n','asof_pitcher_middle_rate'),('pitcher_id','asof_pitcher_n','asof_pitcher_ball_rate'),('pitcher_id','asof_pitcher_n','asof_pitcher_reverse_rate'),('pitcher_id','asof_pitcher_n','asof_pitcher_strike_rate'),('batter_id','asof_batter_n','asof_batter_success_rate'),('batter_id','asof_batter_n','asof_batter_middle_rate')]
def main():
    raw=pd.read_csv(ROOT/'data/train.csv',encoding='utf-8-sig');out=ROOT/'model/COMBO-E2E-001';out.mkdir(parents=True,exist_ok=True)
    for who,ncol in [('pitcher_id','asof_pitcher_n'),('batter_id','asof_batter_n')]:
        g=raw.groupby(who,sort=False)[ncol].max().rename('prior_n').reset_index()
        for ident,n,c in RATE:
            if ident!=who: continue
            ev=(raw[n].fillna(0)*raw[c].fillna(0)).groupby(raw[who],sort=False).max().rename('prior_'+c).reset_index();g=g.merge(ev,on=who,how='left',validate='one_to_one')
        g.to_csv(out/f'asof_{who}_prior.csv',index=False)
    mix=raw.groupby('pitcher_id',sort=False)['asof_pitcher_pitchmix_n'].max().rename('prior_pitchmix_n').reset_index();mix.to_csv(out/'asof_pitchmix_prior.csv',index=False)
    (out/'bundle_metadata.json').write_text(json.dumps({'bundle_id':'COMBO-E2E-BUNDLE-001','source':'official train.csv only','prior_history_max_season':int(raw.season.max()),'files':['asof_pitcher_id_prior.csv','asof_batter_id_prior.csv','asof_pitchmix_prior.csv','pitcher_count_lookup.csv','model.cbm','feature_columns.json']},ensure_ascii=False,indent=2))
    print((out/'bundle_metadata.json').read_text())
if __name__=='__main__':main()
