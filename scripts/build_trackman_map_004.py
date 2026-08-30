#!/usr/bin/env python3
"""Recreate pitcher_id↔pitcher_trackman_id from official CSVs only."""
from pathlib import Path
from collections import Counter, defaultdict
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def train_units(seasons):
    d=pd.read_csv(ROOT/'data/train.csv',usecols=['season','game_month','game_dayofweek','pitcher_team_id','batter_team_id','inning','pitcher_id'],encoding='utf-8-sig');d=d[d.season.isin(seasons)].reset_index(drop=True);key=['season','game_month','game_dayofweek','pitcher_team_id','batter_team_id'];d['_new']=(d.groupby(key,sort=False).inning.diff().fillna(0)<0);d['_seg']=d['_new'].groupby([d[c] for c in key],sort=False).cumsum();d['unit']=list(zip(*[d[c] for c in key+['_seg']]));return d
def tm_units(seasons):
    d=pd.read_csv(ROOT/'data/trackman_history.csv',usecols=['season','game_date','trackman_game_id','pitch_no','pitcher_trackman_id','pitcher_team','batter_team'],encoding='utf-8-sig');d=d[d.season.isin(seasons)].copy();dt=pd.to_datetime(d.game_date,format='mixed');d['game_month']=dt.dt.month;d['game_dayofweek']=dt.dt.dayofweek;d=d.sort_values(['trackman_game_id','pitch_no'],kind='stable');d['unit']=list(zip(d.trackman_game_id,d.pitcher_team));return d
def unit_table(d,pteam,bteam):
    g=d.groupby('unit',sort=False);return pd.DataFrame({'n':g.size(),'season':g.season.first(),'month':g.game_month.first(),'dow':g.game_dayofweek.first(),'pteam':g[pteam].first(),'bteam':g[bteam].first()})
def orders(d,idcol):
    out={}
    for u,s in d.groupby('unit',sort=False):
        seen={}
        for v in s[idcol]:seen[v]=seen.get(v,0)+1
        out[u]=list(seen.items())
    return out
def main():
    seasons=list(range(2019,2025));tr=train_units(seasons);tm=tm_units(seasons);A=unit_table(tr,'pitcher_team_id','batter_team_id');B=unit_table(tm,'pitcher_team','batter_team');sig=['season','month','dow','n'];a=A.groupby(sig).filter(lambda g:len(g)==1);b=B.groupby(sig).filter(lambda g:len(g)==1);m=a.reset_index().merge(b.reset_index(),on=sig,suffixes=('_a','_b'));votes=defaultdict(Counter)
    for r in m.itertuples():votes[r.pteam_a][r.pteam_b]+=1;votes[r.bteam_a][r.bteam_b]+=1
    tmap={k:v.most_common(1)[0][0] for k,v in votes.items()};A2=A.copy();A2['pteam']=A2.pteam.map(tmap);A2['bteam']=A2.bteam.map(tmap);full=['season','month','dow','pteam','bteam','n'];a2=A2.dropna(subset=['pteam','bteam']).groupby(full).filter(lambda g:len(g)==1);b2=B.groupby(full).filter(lambda g:len(g)==1);m2=a2.reset_index().merge(b2.reset_index(),on=full,suffixes=('_a','_b'));pa=orders(tr,'pitcher_id');pb=orders(tm,'pitcher_trackman_id');votes2=defaultdict(Counter)
    for r in m2.itertuples():
        la,lb=pa[r.unit_a],pb[r.unit_b]
        if len(la)!=len(lb):continue
        for (ia,na),(ib,nb) in zip(la,lb):
            if na==nb:votes2[ia][ib]+=1
    rows=[]
    for pid,c in votes2.items():tid,k=c.most_common(1)[0];rows.append({'pitcher_id':pid,'pitcher_trackman_id':tid,'votes':k,'total':sum(c.values()),'conf':k/sum(c.values())})
    out=pd.DataFrame(rows).sort_values('total',ascending=False).reset_index(drop=True);outdir=ROOT/'model/TRACKMAN-MAP-004';outdir.mkdir(parents=True,exist_ok=True);out.to_csv(outdir/'pitcher_id_map.csv',index=False);report={'experiment_id':'TRACKMAN-MAP-004','official_sources':['data/train.csv','data/trackman_history.csv'],'train_unit_count':len(A),'trackman_unit_count':len(B),'unique_stage_matches':len(m),'team_map_count':len(tmap),'full_stage_matches':len(m2),'mapped_pitchers':len(out),'train_pitchers':int(tr.pitcher_id.nunique()),'conf_ge_90':int((out.conf>=.9).sum()),'trackman_id_duplicates':int(out.pitcher_trackman_id.duplicated().sum()),'status':'PASS_MAP_REBUILT' if len(out) and out.pitcher_trackman_id.is_unique else 'FAIL','submission_status':'HOLD'};(outdir/'map_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
