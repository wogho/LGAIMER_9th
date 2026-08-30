import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
CAND_069 = ROOT / 'model/REF4-ADAPTIVE-CHANNEL-OPT-069A/production_package'
sys.path.insert(0, str(CAND_069))

# Load gate
gate = CatBoostRegressor()
gate.load_model(str(CAND_069 / 'model/adaptive_gate.cbm'))

raw = pd.read_csv(ROOT / 'data/train.csv', low_memory=False)
raw_24 = raw.loc[raw.season == 2024].copy().reset_index(drop=True)

from src.preprocessing_v2 import build_v2_features, build_v3_features
from src.adaptive_gate import build_gate_features

MODEL = CAND_069 / 'model'
meta = json.loads((MODEL / "manifest.json").read_text())
ps = pd.read_pickle(MODEL / "pitcher_snapshots.pkl")
bs = pd.read_pickle(MODEL / "batter_snapshots.pkl")
ms = pd.read_pickle(MODEL / "pitchmix_snapshots.pkl")
tm = str(MODEL / "trackman_prior_features.csv")

x2, b2 = build_v2_features(raw_24, meta["prior"], ps, tm)
x3, b3 = build_v3_features(raw_24, meta["prior"], ps, bs, ms, tm)

seeds = meta.get("seeds", [260802, 260803, 260804, 260805, 260806, 260807])
def load_reg(stem):
    return [CatBoostRegressor().load_model(str(MODEL / f"{stem}_seed{s}.cbm")) for s in seeds]

preds = []
for stem, x, base in [("v2_decay55", x2, b2), ("v3_decay55", x3, b3), ("v3_decay30", x3, b3)]:
    member = [np.clip(base + m.predict(x), 1e-6, 1 - 1e-6) for m in load_reg(stem)]
    preds.append(np.mean(member, axis=0))

risks = []
for name in ("middle", "wild", "reverse"):
    member = [CatBoostClassifier().load_model(str(MODEL / f"subtype_{name}_seed{s}.cbm")).predict_proba(x3)[:, 1] for s in seeds]
    risks.append(np.mean(member, axis=0))

main_p = np.average(np.vstack(preds), axis=0, weights=meta["main_weights"])
z = np.column_stack([main_p] + risks)
p = meta["stack_intercept"] + z @ np.asarray(meta["stack_coefficients"])

gx = build_gate_features(raw_24, preds, risks, np.clip(p, 1e-6, 1 - 1e-6))
gate_raw = gate.predict(gx)

print(f"Gate raw on 2024: min={gate_raw.min():.6f}, max={gate_raw.max():.6f}, mean={gate_raw.mean():.6f}, std={gate_raw.std():.6f}")
print(f"Gate offset used in 072: +0.00848698")
clean_gate = gate_raw - 0.00848698
print(f"Clean Gate: min={clean_gate.min():.6f}, max={clean_gate.max():.6f}, mean={clean_gate.mean():.6f}, std={clean_gate.std():.6f}")

p_005 = p + 0.05 * clean_gate + 0.0052
p_075 = p + 0.75 * clean_gate + 0.0052

print(f"\nP_005: min={p_005.min():.6f}, max={p_005.max():.6f}, mean={p_005.mean():.6f}, std={p_005.std():.6f}, Var={np.var(p_005):.6f}")
print(f"P_075: min={p_075.min():.6f}, max={p_075.max():.6f}, mean={p_075.mean():.6f}, std={p_075.std():.6f}, Var={np.var(p_075):.6f}")

y_24 = raw_24.control_success.to_numpy(float)
print(f"2024 True Mean: {np.mean(y_24):.6f}")
print(f"Mean error 0.05: {np.mean(p_005) - np.mean(y_24):+.6f}")
print(f"Mean error 0.75: {np.mean(p_075) - np.mean(y_24):+.6f}")
print(f"Gate clean mean on 2024: {np.mean(clean_gate):+.6f}")
print(f"Variance increase with 0.75: {np.var(p_075) - np.var(p_005):+.6f}")

