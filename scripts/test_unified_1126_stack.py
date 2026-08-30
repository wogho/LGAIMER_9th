import gc, importlib.util, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
import lightgbm as lgb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'candidate/REF4-CHAMPION-STACK-030'))

spec = importlib.util.spec_from_file_location("entity_context_split", str(ROOT / "src/entity_context_split.py"))
ecs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ecs)
apply_split_profile = ecs.apply_split_profile
apply_linear_split = ecs.apply_linear_split

from src.preprocessing_v2 import CAT_V2, build_v2_features, build_v3_features
from src.league_transition import transition_features

MODEL = ROOT / 'candidate/REF4-LGBM-R-EXPERT-064A/model'
meta = json.loads((MODEL / 'manifest.json').read_text())
lgbm_meta = json.loads((MODEL / 'lgbm_meta.json').read_text())
regime = json.loads((MODEL / 'f_regime_meta.json').read_text())

train = pd.read_csv(ROOT / 'data/train.csv', low_memory=False)
val24 = train.loc[train.season == 2024].sample(10000, random_state=42).reset_index(drop=True)
val24['season'] = 2025
val24['row_id'] = [f'TEST_{i:06d}' for i in range(len(val24))]

ps = pd.read_pickle(MODEL / 'pitcher_snapshots.pkl')
bs = pd.read_pickle(MODEL / 'batter_snapshots.pkl')
ms = pd.read_pickle(MODEL / 'pitchmix_snapshots.pkl')
tm = str(MODEL / 'trackman_prior_features.csv')

print('Building features on 10,000 test sample...')
x2, base2 = build_v2_features(val24, meta['prior'], ps, tm)
x3, base3 = build_v3_features(val24, meta['prior'], ps, bs, ms, tm)

seeds = meta.get('seeds')
def load_reg(stem):
    return [CatBoostRegressor().load_model(str(MODEL / f'{stem}_seed{s}.cbm')) for s in seeds]

predictions = []
for stem, x, base in [('v2_decay55', x2, base2), ('v3_decay55', x3, base3), ('v3_decay30', x3, base3)]:
    member = [np.clip(base + m.predict(x), 1e-6, 1 - 1e-6) for m in load_reg(stem)]
    predictions.append(np.mean(member, axis=0))

futures = val24['game_type'].eq('F').to_numpy()
print(f'Sample distribution: R={sum(~futures)}, F={sum(futures)}')

# F-Regime scale 0.75
f_scale = 0.75
def f_reg_mean(stem, count, x, base):
    member = []
    for j in range(count):
        m = CatBoostRegressor()
        m.load_model(str(MODEL / f'{stem}_{j}.cbm'))
        member.append(np.clip(base + m.predict(x), 1e-6, 1 - 1e-6))
    return np.mean(member, axis=0)

if futures.any():
    f2 = f_reg_mean('f_v2_all', 4, x2, base2)
    predictions[0] = np.where(futures, predictions[0] + (f_scale * regime['v2_scale']) * (f2 - predictions[0]), predictions[0])
    f55 = f_reg_mean('f_v355_recent', 6, x3, base3)
    predictions[1] = np.where(futures, predictions[1] + (f_scale * regime['v355_scale']) * (f55 - predictions[1]), predictions[1])
    f30a = f_reg_mean('f_v330_all', 4, x3, base3)
    f30r = f_reg_mean('f_v330_recent', 2, x3, base3)
    recent_inner = predictions[2] + regime['v330_recent_inner_scale'] * (f30r - predictions[2])
    f30 = regime['v330_all_weight'] * f30a + (1 - regime['v330_all_weight']) * recent_inner
    predictions[2] = np.where(futures, predictions[2] + (f_scale * regime['v330_scale']) * (f30 - predictions[2]), predictions[2])

risks = []
for name in ('middle', 'wild', 'reverse'):
    member = [CatBoostClassifier().load_model(str(MODEL / f'subtype_{name}_seed{s}.cbm')).predict_proba(x3)[:, 1] for s in seeds]
    risk = np.mean(member, axis=0)
    if futures.any():
        fm = CatBoostClassifier()
        fm.load_model(str(MODEL / f'f_subtype_{name}.cbm'))
        fr = fm.predict_proba(x3)[:, 1]
        risk = np.where(futures, risk + (f_scale * regime['subtype_scale']) * (fr - risk), risk)
    risks.append(risk)

main_p = np.average(np.vstack(predictions), axis=0, weights=meta['main_weights'])
z = np.column_stack([main_p] + risks)
p = meta['stack_intercept'] + z @ np.asarray(meta['stack_coefficients'])

# Fixed global calibration shift (+0.0052)
p = p + float(meta.get('global_shift', 0.0))

# Transition gate
transition = CatBoostRegressor()
transition.load_model(str(MODEL / 'transition_gate.cbm'))
tx = transition_features(val24, p, str(MODEL / 'prior_type.pkl'))
p = p + (f_scale * 0.15) * transition.predict(tx)

# 1군 (Regulars) Post-Processing: R-specific split profile + 0.02 LightGBM expert
regular = ~futures
if regular.any():
    split_profile = pd.read_csv(MODEL / 'split_profile.csv', dtype={'entity_value': str, 'context_value': str})
    split_x = apply_split_profile(val24, split_profile)
    split_correction = apply_linear_split(split_x, str(MODEL / 'split_residual_meta.npz'))
    p_055 = p + split_correction
    
    x3_lgb = x3.copy()
    for col in CAT_V2:
        if col in x3_lgb.columns:
            x3_lgb[col] = x3_lgb[col].astype('category')
    lgb_booster = lgb.Booster(model_file=str(MODEL / lgbm_meta['model_file']))
    r_expert = np.clip(base3 + lgb_booster.predict(x3_lgb) + lgbm_meta['shift_offset'], 1e-6, 1.0 - 1e-6)
    w_lgb = float(lgbm_meta['blend_weight'])
    p = np.where(regular, (1.0 - w_lgb) * p_055 + w_lgb * r_expert, p)

p = np.clip(p, 1e-5, 1.0 - 1e-5)
target = val24['control_success'].to_numpy(float)
ref = float(target.mean() * (1 - target.mean()))
bss = 1e5 * (1.0 - np.mean((p - target)**2) / ref)

r_bss = 1e5*(1 - np.mean((p[regular]-target[regular])**2)/(target[regular].mean()*(1-target[regular].mean())))
f_bss = 1e5*(1 - np.mean((p[futures]-target[futures])**2)/(target[futures].mean()*(1-target[futures].mean())))

print(f'\n=== UNIFIED 1126 STACK VALIDATION ===')
print(f'Overall Local BSS Score: {bss:.4f}')
print(f'  - Regular (1군) BSS: {r_bss:.4f}')
print(f'  - Futures (2군) BSS: {f_bss:.4f}')
