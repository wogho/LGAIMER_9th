import gc, os, sys, importlib.util
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'candidate/REF4-CHAMPION-STACK-030'))

spec = importlib.util.spec_from_file_location("entity_context_split", str(ROOT / "src/entity_context_split.py"))
ecs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ecs)
build_split_features = ecs.build_split_features

from src.preprocessing_v2 import CAT_V2, build_v3_features
from src.season_delta_features import build_snapshots
from src.season_history_v3 import build_entity_snapshots

train = pd.read_csv(ROOT / 'data/train.csv', low_memory=False)
oof_2022 = pd.read_csv(ROOT / 'model/REF4-ADAPTIVE-GATE-031B/oof_2022.csv')
oof_2023 = pd.read_csv(ROOT / 'model/REF4-EXACT-OOF-031A/oof_2023.csv')
oof_2024 = pd.read_csv(ROOT / 'model/REF4-EXACT-OOF-031A/oof_2024.csv')
oof_dict = {2022: oof_2022, 2023: oof_2023, 2024: oof_2024}

df_051 = pd.read_csv(ROOT / 'model/REF4-TRAINONLY-R-SPECIFIC-SPLIT-051A/oof_predictions.csv')

# For 2022 051A prediction:
h22 = train.loc[train.season < 2022].reset_index(drop=True)
r22 = oof_2022[['row_id']].merge(train, on='row_id', validate='one_to_one')
sfeat_2022 = build_split_features(h22, r22)
mask_r22 = (oof_2022.game_type == 'R').to_numpy()
s_mean = sfeat_2022.loc[mask_r22].mean().to_numpy(float)
s_std = sfeat_2022.loc[mask_r22].std().replace(0, 1).fillna(1).to_numpy(float)
y_res_2022 = oof_2022.target.to_numpy(float)[mask_r22] - oof_2022.prediction.to_numpy(float)[mask_r22]
reg22 = Ridge(alpha=10000.0, fit_intercept=False).fit(np.nan_to_num((sfeat_2022.loc[mask_r22].to_numpy(float) - s_mean)/s_std), y_res_2022)
corr_2022 = reg22.predict(np.nan_to_num((sfeat_2022.to_numpy(float) - s_mean)/s_std))
pred_051_2022 = np.where(mask_r22, np.clip(oof_2022.prediction.to_numpy(float) + corr_2022, 1e-6, 1-1e-6), oof_2022.prediction.to_numpy(float))

pred_051_dict = {
    2022: pred_051_2022,
    2023: df_051.loc[df_051.season == 2023, 'candidate_prediction'].to_numpy(float),
    2024: df_051.loc[df_051.season == 2024, 'candidate_prediction'].to_numpy(float)
}

print('Extracting v3 features...')
v3_feat = {}
for yr in (2022, 2023, 2024):
    th = train.loc[train.season < yr].reset_index(drop=True)
    vh = train.loc[train.season == yr].reset_index(drop=True)
    prior = float(th.control_success.mean())
    ps = build_snapshots(th)
    bs = build_entity_snapshots(th, 'batter_id', 'asof_batter_n', ['asof_batter_success_rate', 'asof_batter_middle_rate'], 'control_success')
    ms = build_entity_snapshots(th, 'pitcher_id', 'asof_pitcher_pitchmix_n', ['asof_pitcher_fastball_rate', 'asof_pitcher_breaking_rate', 'asof_pitcher_offspeed_rate'])
    tm_path = ROOT / f"model/REF4-EXACT-OOF-031A/fold_{yr}/trackman_prior_features.csv" if yr in (2023, 2024) else ROOT / "model/REF4-ADAPTIVE-GATE-031B/fold_2022/trackman_prior_features.csv"
    xv, _ = build_v3_features(vh, prior, ps, bs, ms, str(tm_path))
    for col in CAT_V2:
        if col in xv.columns:
            xv[col] = xv[col].astype('category')
    v3_feat[yr] = xv
    print(f'Year {yr} v3 features ready: {xv.shape}')

# Test fitting LightGBM on 051A residual
for lr in [0.005, 0.01]:
    for nl in [7, 15]:
        print(f'\n--- Testing LightGBM params: num_leaves={nl}, lr={lr} ---')
        for valid_yr, fit_yrs in [(2023, [2022]), (2024, [2022, 2023])]:
            fit_x, fit_y, fit_w = [], [], []
            for yr in fit_yrs:
                mask = (oof_dict[yr].game_type == 'R').to_numpy()
                fit_x.append(v3_feat[yr].loc[mask])
                res_051 = oof_dict[yr].target.to_numpy(float)[mask] - pred_051_dict[yr][mask]
                fit_y.append(res_051)
                fit_w.append(np.full(mask.sum(), 0.55 if valid_yr == 2024 and yr == 2022 else 1.0))
            
            X_fit = pd.concat(fit_x, ignore_index=True)
            y_fit = np.concatenate(fit_y)
            w_fit = np.concatenate(fit_w)
            
            model = lgb.LGBMRegressor(
                n_estimators=50,
                num_leaves=nl,
                learning_rate=lr,
                colsample_bytree=0.8,
                subsample=0.85,
                min_child_samples=300,
                reg_alpha=20.0,
                reg_lambda=100.0,
                random_state=260803,
                n_jobs=3,
                verbose=-1
            )
            model.fit(X_fit, y_fit, sample_weight=w_fit)
            
            mask_valid_r = (oof_dict[valid_yr].game_type == 'R').to_numpy()
            delta = model.predict(v3_feat[valid_yr])
            
            p051_v = pred_051_dict[valid_yr]
            cand_v = np.where(mask_valid_r, np.clip(p051_v + delta, 1e-6, 1-1e-6), p051_v)
            
            target_v = oof_dict[valid_yr].target.to_numpy(float)
            brier_051 = np.mean((p051_v - target_v)**2)
            brier_cand = np.mean((cand_v - target_v)**2)
            gain = brier_051 - brier_cand
            print(f'Fold {valid_yr}: Brier Gain vs 051 = {gain:+.8f}')
