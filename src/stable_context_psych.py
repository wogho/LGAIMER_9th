"""Leakage-safe context-adjusted pitcher pressure features with season stability."""
from __future__ import annotations

import numpy as np
import pandas as pd


CONDITIONS = ("high_li", "extreme_li", "traffic", "risp", "two_out_risp", "three_ball", "full_count", "late_high_li", "close_high_li", "compound_crisis")


def context_frame(data: pd.DataFrame):
    r=data.reset_index(drop=True);balls=pd.to_numeric(r.balls_before,errors='coerce').fillna(0).astype(int);strikes=pd.to_numeric(r.strikes_before,errors='coerce').fillna(0).astype(int);outs=pd.to_numeric(r.outs_before,errors='coerce').fillna(0).astype(int);inning=pd.to_numeric(r.inning,errors='coerce').fillna(0);li=pd.to_numeric(r.li,errors='coerce').fillna(0);score=pd.to_numeric(r.score_diff_pitcher_team,errors='coerce').fillna(0);runners=pd.to_numeric(r.num_runners_on,errors='coerce').fillna(0);risp=r.runner_on_2b.eq(1)|r.runner_on_3b.eq(1)
    x=pd.DataFrame(index=r.index);x['count']=balls.astype(str)+'-'+strikes.astype(str);x['outs']=outs.astype(str);x['base']=r.base_state.astype(str);x['hand']=r.pitcher_hand.astype(str)+'-'+r.batter_hand.astype(str);x['inning_bin']=pd.cut(inning,[-np.inf,3,6,9,np.inf],labels=['early','middle','late','extra']).astype(str);x['li_bin']=pd.cut(li,[-np.inf,.75,1.5,3,np.inf],labels=['low','normal','high','extreme']).astype(str);x['home']=r.top_bottom.astype(str).eq('T').astype(int).astype(str);x['score_bin']=pd.cut(score,[-np.inf,-3,-1,1,3,np.inf],labels=False).fillna(-1).astype(int).astype(str)
    c=pd.DataFrame({'high_li':li>=1.5,'extreme_li':li>=3,'traffic':runners>0,'risp':risp,'two_out_risp':(outs==2)&risp,'three_ball':balls==3,'full_count':(balls==3)&(strikes==2),'late_high_li':(inning>=7)&(li>=1.5),'close_high_li':(score.abs()<=1)&(li>=1.5),'compound_crisis':(li>=1.5)&risp&(balls>=2)},index=r.index).astype('int8');return x,c


def context_adjusted_residual(history: pd.DataFrame, alpha: float=400.0):
    h=history.reset_index(drop=True);ctx,_=context_frame(h);y=h.control_success.to_numpy(float);prior=float(np.mean(y));levels=[['count','outs','base','hand','inning_bin','li_bin','home'],['count','outs','base','hand'],['count','outs','base'],['count','hand']];estimates=[];reliabilities=[];temp=ctx.copy();temp['_y']=y
    for keys in levels:
        agg=temp.groupby(keys,dropna=False)['_y'].agg(['sum','count']);idx=pd.MultiIndex.from_frame(ctx[keys]);s=agg['sum'].reindex(idx).to_numpy();n=agg['count'].reindex(idx).to_numpy();rate=(s-y+alpha*prior)/(np.maximum(n-1,0)+alpha);estimates.append(rate);reliabilities.append(np.maximum(n-1,0)/(np.maximum(n-1,0)+alpha))
    e=np.column_stack(estimates);w=np.column_stack(reliabilities);expected=(e*(.25+w)).sum(1)/(.25+w).sum(1);return y-expected


def build_profile(history: pd.DataFrame, alpha_context: float=400.0, alpha_pitcher: float=100.0):
    h=history.reset_index(drop=True);residual=context_adjusted_residual(h,alpha_context);_,cond=context_frame(h);pid=h.pitcher_id.astype(str);global_res=float(residual.mean());overall=pd.DataFrame({'pid':pid,'res':residual}).groupby('pid').res.agg(['sum','count']);overall_rate=(overall['sum']+alpha_pitcher*global_res)/(overall['count']+alpha_pitcher);out=pd.DataFrame(index=overall.index);out['stable_history_log_n']=np.log1p(overall['count'])
    for name in CONDITIONS:
        mask=cond[name].to_numpy(bool);frame=pd.DataFrame({'pid':pid[mask],'res':residual[mask]});agg=frame.groupby('pid').res.agg(['sum','count']);n=agg['count'].reindex(out.index).fillna(0);rate=(agg['sum'].reindex(out.index).fillna(0)+alpha_pitcher*overall_rate)/(n+alpha_pitcher);effect=rate-overall_rate;per=[]
        for _,idx in h.groupby('season').groups.items():
            ii=np.asarray(list(idx));m=cond.loc[ii,name].to_numpy(bool);sub_pid=pid.iloc[ii].to_numpy()[m];sub_res=residual[ii][m];a=pd.DataFrame({'pid':sub_pid,'res':sub_res}).groupby('pid').res.mean();per.append(a.reindex(out.index))
        s=pd.concat(per,axis=1) if per else pd.DataFrame(index=out.index);stability=np.abs(np.sign(s).mean(1)).fillna(0);out[f'stable_{name}_effect']=effect*stability;out[f'stable_{name}_reliability']=n/(n+alpha_pitcher)
    return out.reset_index(names='pitcher_id')


def apply_profile(rows: pd.DataFrame, profile: pd.DataFrame):
    _,cond=context_frame(rows);x=pd.DataFrame({'pitcher_id':rows.pitcher_id.astype(str).to_numpy()}).merge(profile,on='pitcher_id',how='left',sort=False).drop(columns='pitcher_id').fillna(0);out={'stable_history_log_n':x.stable_history_log_n.to_numpy(float)}
    for name in CONDITIONS:
        active=cond[name].to_numpy(float);effect=x[f'stable_{name}_effect'].to_numpy(float);reliability=x[f'stable_{name}_reliability'].to_numpy(float);out[f'stable_{name}_effect']=effect;out[f'stable_{name}_active_effect']=effect*active;out[f'stable_{name}_active_reliability']=reliability*active
    return pd.DataFrame(out,index=rows.index).astype('float32')


def build_stable_context_features(history: pd.DataFrame, rows: pd.DataFrame, alpha_context: float=400.0, alpha_pitcher: float=100.0):
    return apply_profile(rows,build_profile(history,alpha_context,alpha_pitcher))
