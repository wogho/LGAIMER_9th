import pandas as pd
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
c = pd.read_csv(ROOT / 'model/COMBO-RESID3-OOF-007/oof_predictions.csv', usecols=['row_id','season','target','pred'])
l = pd.read_csv(ROOT / 'model/CAL-FE001-TEMPORAL-OOF/expanding_oof_predictions.csv', usecols=['row_id','season','target','pred_lgbm'])
m = c.merge(l, on=['row_id','season','target'], how='inner', validate='one_to_one')
metric = lambda p, y: float(1e5 * np.corrcoef(p, y)[0, 1] ** 2)
rows = []
for season in [2022, 2023, 2024]:
    v = m[m.season.eq(season)]
    base = metric(v.pred, v.target)
    for w in [0.1, 0.25, 0.5, 0.75, 0.9]:
        p = (1 - w) * v.pred.to_numpy() + w * v.pred_lgbm.to_numpy()
        rows.append({'season': season, 'weight_lgbm': w, 'base_metric': base, 'candidate_metric': metric(p, v.target), 'relative_improvement': metric(p, v.target) / base - 1, 'delta_brier': float(np.mean((p - v.target) ** 2) - np.mean((v.pred - v.target) ** 2))})
out = pd.DataFrame(rows)
out.to_csv(ROOT / 'model/COMBO-BLEND-009-screen.csv', index=False)
print(out.to_string(index=False))
