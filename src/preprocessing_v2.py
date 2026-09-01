"""Leakage-safe second-generation features for forward-season prediction."""
from __future__ import annotations
import numpy as np
import pandas as pd
from src.catboost_features import CAT_COLS,build_catboost_features,attach_trackman_features
from src.season_delta_features import attach_season_delta

CAT_V2 = CAT_COLS + [
    'count_base_context','hand_count_context','inning_score_context',
    'pressure_context','recent_regime','trackman_state',
]

CAT_V4 = CAT_V2 + [
    'month_context','inning_count_context','team_hand_context',
    'leverage_score_context','form_direction_context',
]

CAT_V5 = CAT_V4 + [
    'pitcher_batterhand_context','pitcher_count_context','pitcher_base_context',
    'pitcher_pressure_context','pitcher_game_context','pitcher_inning_context',
    'batter_pitcherhand_context','team_count_context',
]

def _safe_logit(s):
    p=pd.to_numeric(s,errors='coerce').clip(1e-4,1-1e-4)
    return np.log(p/(1-p))

def build_v2_features(raw: pd.DataFrame, prior: float, snapshots: pd.DataFrame,
                      trackman_path: str, level=None) -> tuple[pd.DataFrame,np.ndarray]:
    # `level` is a per-row estimate of the season's league rate. Where a scalar
    # prior was only ever a shrinkage target, a thin sample should be pulled
    # toward THIS season rather than toward the train period, whose mean is 4.5
    # points of probability away by 2024.
    lvl = np.full(len(raw), prior, dtype=float) if level is None else np.asarray(level, dtype=float)
    x=attach_trackman_features(build_catboost_features(raw,prior),trackman_path)
    x=attach_season_delta(x,raw.reset_index(drop=True),snapshots,prior)

    # Missingness and reliability are data, especially because only ~63% of rows
    # have a conservative Trackman mapping.
    x['has_trackman']=x['tm_n'].notna().astype('int8')
    x['trackman_reliability']=(x['tm_n'].fillna(0)/(x['tm_n'].fillna(0)+300)).astype('float32')
    x['trackman_mapping_reliable']=((x['tm_mapping_similarity'].fillna(0)>=.95)&(x['tm_n'].fillna(0)>=300)).astype('int8')
    x['trackman_state']=np.select(
        [x['has_trackman'].eq(0),x['trackman_mapping_reliable'].eq(1)],
        ['missing','reliable'],default='weak')

    recent_cols=[f'asof_pitcher_prev{k}_game_success_rate' for k in (1,3,5)]
    rv=raw[recent_cols].apply(pd.to_numeric,errors='coerce')
    recent_mean=rv.mean(axis=1).fillna(pd.Series(lvl,index=rv.index))
    recent_std=rv.std(axis=1).fillna(.15).clip(0,.5)
    n=pd.to_numeric(raw.asof_pitcher_n,errors='coerce').fillna(0).clip(lower=0)
    career_rate=pd.to_numeric(raw.asof_pitcher_success_rate,errors='coerce').fillna(pd.Series(lvl,index=raw.index))
    # Stable recent form earns a little more trust; volatile/cold-start histories
    # are shrunk harder toward a train-only prior.
    dynamic_strength=(55+220*recent_std+40/(1+np.log1p(n))).clip(50,180)
    career_base=(career_rate*n+lvl*dynamic_strength)/(n+dynamic_strength)
    season_n=x['season_pitcher_n'].clip(lower=0)
    season_raw=x['season_success_rate_raw'].fillna(pd.Series(lvl,index=x.index))
    season_strength=30.0
    season_est=(season_raw*season_n+lvl*season_strength)/(season_n+season_strength)
    season_reliability=season_n/(season_n+80)
    season_weight=.15+.30*season_reliability
    hierarchical_base=career_base+season_weight*(season_est-career_base)

    x['recent_success_std']=recent_std.to_numpy()
    x['dynamic_smoothing_strength']=dynamic_strength.to_numpy()
    x['career_dynamic_base']=career_base.to_numpy()
    x['season_form_estimate']=season_est.to_numpy()
    x['season_form_reliability']=season_reliability.to_numpy()
    x['hierarchical_success_base']=hierarchical_base.to_numpy()
    if level is not None:
        # Only present when a per-row level was supplied; the production models
        # were fitted without these columns and must keep their exact schema.
        x['league_level']=lvl
        x['league_minus_prior']=lvl-prior
    x['season_recent_gap']=(season_est-recent_mean).to_numpy()
    x['stable_momentum']=((recent_mean-career_base)/(1+5*recent_std)).to_numpy()
    x['pitcher_success_logit']=_safe_logit(career_rate).to_numpy()
    x['season_success_logit']=_safe_logit(season_est).to_numpy()

    # Explicit low-cardinality interactions save shallow trees several splits.
    pressure=pd.cut(pd.to_numeric(raw.li,errors='coerce').fillna(0),[-np.inf,.75,1.5,3,np.inf],labels=['low','normal','high','extreme']).astype(str)
    inning=pd.cut(pd.to_numeric(raw.inning,errors='coerce').fillna(0),[-np.inf,3,6,9,np.inf],labels=['early','middle','late','extra']).astype(str)
    score=pd.cut(pd.to_numeric(raw.score_diff_pitcher_team,errors='coerce').fillna(0),[-np.inf,-3,-1,1,3,np.inf],labels=['far_behind','behind','close','ahead','far_ahead']).astype(str)
    x['count_base_context']=x['count_state'].astype(str)+'|'+raw.base_state.astype(str)
    x['hand_count_context']=x['hand_matchup'].astype(str)+'|'+x['count_state'].astype(str)
    x['inning_score_context']=inning+'|'+score
    x['pressure_context']=pressure+'|'+raw.num_runners_on.astype(str)+'|'+x['count_state'].astype(str)
    x['recent_regime']=np.select([recent_std<.04,recent_std<.10],['stable','normal'],default='volatile')
    for c in CAT_V2: x[c]=x[c].astype('string').fillna('__MISSING__').astype(str)
    x=x.replace([np.inf,-np.inf],np.nan)
    return x,hierarchical_base.to_numpy(np.float32)

def build_v3_features(raw: pd.DataFrame, prior: float, snapshots: pd.DataFrame,
                      batter_snapshots: pd.DataFrame, pitchmix_snapshots: pd.DataFrame,
                      trackman_path: str, level=None) -> tuple[pd.DataFrame,np.ndarray]:
    from src.season_history_v3 import attach_entity_season
    x,base=build_v2_features(raw,prior,snapshots,trackman_path,level)
    rr=raw.reset_index(drop=True)
    x=attach_entity_season(x,rr,batter_snapshots,'batter_id','asof_batter_n',
        ['asof_batter_success_rate','asof_batter_middle_rate'],'batter_season',prior)
    x=attach_entity_season(x,rr,pitchmix_snapshots,'pitcher_id','asof_pitcher_pitchmix_n',
        ['asof_pitcher_fastball_rate','asof_pitcher_breaking_rate','asof_pitcher_offspeed_rate'],
        'pitchmix_season',prior)
    bn=x['batter_season_n'];bs=x['batter_season_success_rate_raw']
    x['batter_season_success_smoothed']=(bs*bn+prior*40)/(bn+40)
    x['batter_season_reliability']=bn/(bn+80)
    pn=x['pitchmix_season_n']
    for c in ['fastball_rate','breaking_rate','offspeed_rate']:
        raw_col='pitchmix_season_'+c+'_raw';career='asof_pitcher_'+c
        x['pitchmix_season_'+c+'_smoothed']=(x[raw_col]*pn+pd.to_numeric(rr[career],errors='coerce').fillna(0)*50)/(pn+50)
    rates=x[[f'pitchmix_season_{c}_smoothed' for c in ['fastball_rate','breaking_rate','offspeed_rate']]].clip(1e-7,1)
    x['pitchmix_season_entropy']=-(rates*np.log(rates)).sum(axis=1)
    x['pitchmix_season_reliability']=pn/(pn+100)
    x['batter_pitcher_form_gap']=x['hierarchical_success_base']-x['batter_season_success_smoothed']
    return x.replace([np.inf,-np.inf],np.nan),base

def build_v4_features(raw: pd.DataFrame, prior: float, snapshots: pd.DataFrame,
                      batter_snapshots: pd.DataFrame, pitchmix_snapshots: pd.DataFrame,
                      trackman_path: str) -> tuple[pd.DataFrame,np.ndarray]:
    """High-signal explicit interactions on top of the validated v3 representation."""
    x,base=build_v3_features(raw,prior,snapshots,batter_snapshots,pitchmix_snapshots,trackman_path)
    rr=raw.reset_index(drop=True)
    month=pd.to_numeric(rr.game_month,errors='coerce').fillna(0).astype(int).astype(str)
    inning=pd.cut(pd.to_numeric(rr.inning,errors='coerce').fillna(0),[-np.inf,3,6,9,np.inf],labels=['early','middle','late','extra']).astype(str)
    leverage=pd.cut(pd.to_numeric(rr.li,errors='coerce').fillna(0),[-np.inf,.75,1.5,3,np.inf],labels=['low','normal','high','extreme']).astype(str)
    score=pd.cut(pd.to_numeric(rr.score_diff_pitcher_team,errors='coerce').fillna(0),[-np.inf,-3,-1,1,3,np.inf],labels=['far_behind','behind','close','ahead','far_ahead']).astype(str)
    x['month_context']=month+'|'+rr.game_type.astype(str)
    x['inning_count_context']=inning+'|'+x['count_state'].astype(str)
    x['team_hand_context']=rr.pitcher_team_id.astype(str)+'|'+rr.pitcher_hand.astype(str)+'|'+rr.batter_hand.astype(str)
    x['leverage_score_context']=leverage+'|'+score
    recent=x[['asof_pitcher_prev1_game_success_rate','asof_pitcher_prev3_game_success_rate','asof_pitcher_prev5_game_success_rate']]
    direction=np.select([recent.iloc[:,0]>recent.iloc[:,2]+.03,recent.iloc[:,0]<recent.iloc[:,2]-.03],['hot','cold'],default='flat')
    x['form_direction_context']=pd.Series(direction,index=x.index).astype(str)+'|'+leverage
    x['recent_weighted_success']=(.55*recent.iloc[:,0]+.30*recent.iloc[:,1]+.15*recent.iloc[:,2])
    x['recent_acceleration']=recent.iloc[:,0]-2*recent.iloc[:,1]+recent.iloc[:,2]
    middle=x[['asof_pitcher_prev1_game_middle_rate','asof_pitcher_prev3_game_middle_rate','asof_pitcher_prev5_game_middle_rate']]
    x['recent_middle_weighted']=(.55*middle.iloc[:,0]+.30*middle.iloc[:,1]+.15*middle.iloc[:,2])
    x['failure_profile_sum']=x['asof_pitcher_middle_rate']+x['asof_pitcher_reverse_rate']
    x['success_failure_margin']=x['asof_pitcher_success_rate']-x['failure_profile_sum']
    x['batter_career_gap']=x['asof_pitcher_success_rate']-x['asof_batter_success_rate']
    x['exposure_ratio']=np.log1p(x['asof_batter_n'].clip(lower=0))-np.log1p(x['asof_pitcher_n'].clip(lower=0))
    x['pressure_form']=x['pressure_index']*(x['recent_weighted_success']-x['career_dynamic_base'])
    x['leverage_form']=pd.to_numeric(rr.li,errors='coerce').fillna(0)*(x['season_form_estimate']-x['career_dynamic_base'])
    for metric in ['rel_speed','spin_rate','induced_vert_break','horz_break','extension','rel_height','rel_side','zone_speed']:
        mean=x[f'tm_{metric}_mean'].abs().clip(lower=1e-3)
        x[f'tm_{metric}_cv']=x[f'tm_{metric}_std']/mean
    for c in CAT_V4: x[c]=x[c].astype('string').fillna('__MISSING__').astype(str)
    return x.replace([np.inf,-np.inf],np.nan),base

def build_v5_features(raw: pd.DataFrame, prior: float, snapshots: pd.DataFrame,
                      batter_snapshots: pd.DataFrame, pitchmix_snapshots: pd.DataFrame,
                      trackman_path: str) -> tuple[pd.DataFrame,np.ndarray]:
    """Explicit high-cardinality context crosses missed by default CTR complexity."""
    x,base=build_v4_features(raw,prior,snapshots,batter_snapshots,pitchmix_snapshots,trackman_path)
    rr=raw.reset_index(drop=True);pid=rr.pitcher_id.astype(str);bid=rr.batter_id.astype(str)
    inning=pd.cut(pd.to_numeric(rr.inning,errors='coerce').fillna(0),[-np.inf,3,6,9,np.inf],labels=['early','middle','late','extra']).astype(str)
    pressure=pd.cut(pd.to_numeric(rr.li,errors='coerce').fillna(0),[-np.inf,.75,1.5,3,np.inf],labels=['low','normal','high','extreme']).astype(str)
    x['pitcher_batterhand_context']=pid+'|'+rr.batter_hand.astype(str)
    x['pitcher_count_context']=pid+'|'+x['count_state'].astype(str)
    x['pitcher_base_context']=pid+'|'+rr.base_state.astype(str)
    x['pitcher_pressure_context']=pid+'|'+pressure
    x['pitcher_game_context']=pid+'|'+rr.game_type.astype(str)
    x['pitcher_inning_context']=pid+'|'+inning
    x['batter_pitcherhand_context']=bid+'|'+rr.pitcher_hand.astype(str)
    x['team_count_context']=rr.pitcher_team_id.astype(str)+'|'+x['count_state'].astype(str)
    for c in CAT_V5:x[c]=x[c].astype('string').fillna('__MISSING__').astype(str)
    return x.replace([np.inf,-np.inf],np.nan),base

CAT_V5_HAND = CAT_V2 + ['pitcher_batterhand_context']

def build_v5_hand_features(raw: pd.DataFrame, prior: float, snapshots: pd.DataFrame,
                           batter_snapshots: pd.DataFrame, pitchmix_snapshots: pd.DataFrame,
                           trackman_path: str) -> tuple[pd.DataFrame,np.ndarray]:
    x,base=build_v3_features(raw,prior,snapshots,batter_snapshots,pitchmix_snapshots,trackman_path)
    rr=raw.reset_index(drop=True)
    x['pitcher_batterhand_context']=(rr.pitcher_id.astype(str)+'|'+rr.batter_hand.astype(str)).fillna('__MISSING__')
    x['pitcher_batterhand_context']=x['pitcher_batterhand_context'].astype(str)
    return x.replace([np.inf,-np.inf],np.nan),base

def build_v6_context_features(raw: pd.DataFrame, prior: float, snapshots: pd.DataFrame,
                              batter_snapshots: pd.DataFrame, pitchmix_snapshots: pd.DataFrame,
                              trackman_path: str) -> tuple[pd.DataFrame,np.ndarray]:
    """Validated v3 representation plus explicit baseball context candidates."""
    from src.context_pressure_features import build_context_pressure_features
    x,base=build_v3_features(raw,prior,snapshots,batter_snapshots,pitchmix_snapshots,trackman_path)
    extra=build_context_pressure_features(raw)
    return pd.concat([x.reset_index(drop=True),extra],axis=1),base

def attach_v7_context_adjusted_psych(x: pd.DataFrame, raw: pd.DataFrame,
                                     prior_history: pd.DataFrame) -> pd.DataFrame:
    """Research-only psych preprocessing; prior_history must end before raw season."""
    from src.context_adjusted_psych import attach_context_adjusted_psych
    if len(prior_history) and int(prior_history.season.max()) >= int(raw.season.min()):
        raise ValueError('prior_history must contain only seasons before the target rows')
    psych=attach_context_adjusted_psych(raw,prior_history,alpha_context=400.,alpha_pitcher=100.)
    return pd.concat([x.reset_index(drop=True),psych],axis=1)
