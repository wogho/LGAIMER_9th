import json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
CAND_DIR = ROOT / 'candidate/REF4-R-SPECIFIC-SPLIT-FPSYCH-055A'
sys.path.insert(0, str(CAND_DIR))
sys.path.insert(0, str(ROOT / 'github_reference/4번 레포'))

from src.preprocessing_v2 import build_v2_features, build_v3_features
from src.adaptive_gate import build_gate_features

def bss(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-5, 1 - 1e-5)
    r = float(y.mean())
    ref = r * (1.0 - r)
    return 1e5 * (1.0 - np.mean((p - y) ** 2) / ref)

def main():
    raw = pd.read_csv(ROOT / 'data/train.csv', low_memory=False)
    
    # 2022 Train, 2023 Test
    df_22 = raw.loc[raw.season == 2022].copy().reset_index(drop=True)
    df_23 = raw.loc[raw.season == 2023].copy().reset_index(drop=True)
    df_24 = raw.loc[raw.season == 2024].copy().reset_index(drop=True)
    
    MODEL = CAND_DIR / 'model'
    meta = json.loads((MODEL / "manifest.json").read_text(encoding="utf-8"))
    ps = pd.read_pickle(MODEL / "pitcher_snapshots.pkl")
    bs = pd.read_pickle(MODEL / "batter_snapshots.pkl")
    ms = pd.read_pickle(MODEL / "pitchmix_snapshots.pkl")
    tm = str(MODEL / "trackman_prior_features.csv")
    seeds = meta.get("seeds", [260802, 260803, 260804, 260805, 260806, 260807])
    
    def load_reg(stem):
        return [CatBoostRegressor().load_model(str(MODEL / f"{stem}_seed{s}.cbm")) for s in seeds]

    def get_base_preds(x2, base2, x3, base3):
        preds = []
        for stem, x, base in [
            ("v2_decay55", x2, base2),
            ("v3_decay55", x3, base3),
            ("v3_decay30", x3, base3),
        ]:
            member = [np.clip(base + m.predict(x), 1e-6, 1 - 1e-6) for m in load_reg(stem)]
            preds.append(np.mean(member, axis=0))
        return preds

    def get_risks(x3):
        risks = []
        for name in ("middle", "wild", "reverse"):
            member = []
            for s in seeds:
                m = CatBoostClassifier()
                m.load_model(str(MODEL / f"subtype_{name}_seed{s}.cbm"))
                member.append(m.predict_proba(x3)[:, 1])
            risks.append(np.mean(member, axis=0))
        return risks

    # Train on 2022+2023, test strictly on 2024
    df_train = pd.concat([df_22, df_23], ignore_index=True)
    y_train = df_train.control_success.to_numpy(float)
    y_24 = df_24.control_success.to_numpy(float)
    
    x2_tr, b2_tr = build_v2_features(df_train, meta["prior"], ps, tm)
    x3_tr, b3_tr = build_v3_features(df_train, meta["prior"], ps, bs, ms, tm)
    preds_tr = get_base_preds(x2_tr, b2_tr, x3_tr, b3_tr)
    risks_tr = get_risks(x3_tr)
    main_tr = np.average(np.vstack(preds_tr), axis=0, weights=meta["main_weights"])
    z_tr = np.column_stack([main_tr] + risks_tr)
    p_stack_tr = meta["stack_intercept"] + z_tr @ np.asarray(meta["stack_coefficients"])
    gx_tr = build_gate_features(df_train, preds_tr, risks_tr, np.clip(p_stack_tr, 1e-6, 1 - 1e-6))
    
    x2_24, b2_24 = build_v2_features(df_24, meta["prior"], ps, tm)
    x3_24, b3_24 = build_v3_features(df_24, meta["prior"], ps, bs, ms, tm)
    preds_24 = get_base_preds(x2_24, b2_24, x3_24, b3_24)
    risks_24 = get_risks(x3_24)
    main_24 = np.average(np.vstack(preds_24), axis=0, weights=meta["main_weights"])
    z_24 = np.column_stack([main_24] + risks_24)
    p_stack_24 = meta["stack_intercept"] + z_24 @ np.asarray(meta["stack_coefficients"])
    gx_24 = build_gate_features(df_24, preds_24, risks_24, np.clip(p_stack_24, 1e-6, 1 - 1e-6))
    
    # Train gate strictly on 2022+2023
    gate = CatBoostRegressor(iterations=100, depth=4, learning_rate=0.025, loss_function='RMSE', l2_leaf_reg=30, random_strength=0.2, bootstrap_type='Bernoulli', subsample=0.8, random_seed=280033, thread_count=6, verbose=False)
    gate.fit(gx_tr, y_train - p_stack_tr)
    
    gate_p_24 = gate.predict(gx_24)
    offset_tr = float(np.mean(gate.predict(gx_tr)))
    gate_clean_24 = gate_p_24 - offset_tr
    
    oof_051 = pd.read_csv(ROOT / 'model/REF4-TRAINONLY-R-SPECIFIC-SPLIT-051A/oof_predictions.csv').set_index('row_id')
    p_051_24 = oof_051.loc[df_24.row_id, 'candidate_prediction'].to_numpy(float)
    
    print(f"--- TRUE OOF Evaluation on 2024 (Trained on 2022+2023) ---")
    print(f"051A Baseline 2024 BSS: {bss(y_24, p_051_24):.4f}")
    
    for s in [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10, 0.20, 0.50, 0.75, 1.0]:
        p_cand = np.clip(p_051_24 + s * gate_clean_24, 1e-5, 1 - 1e-5)
        score = bss(y_24, p_cand)
        diff = score - bss(y_24, p_051_24)
        print(f"  Scale {s:.2f} -> 2024 BSS: {score:8.4f} (diff vs 051: {diff:+8.4f})")
        
if __name__ == '__main__':
    main()
