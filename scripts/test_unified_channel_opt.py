#!/usr/bin/env python3
"""Simulation and validation of Unified Channel Optimum + Futures 16-Model Ensemble."""
import gc, json, os, sys, time
from pathlib import Path
from catboost import CatBoostClassifier, CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CAND_DIR = ROOT / 'candidate/REF4-R-SPECIFIC-SPLIT-FPSYCH-055A'
sys.path.insert(0, str(CAND_DIR))

from src.preprocessing_v2 import build_v2_features, build_v3_features, CAT_V2
from src.features_v355 import build_v355_features, CAT_COLS_V355

def bss(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-5, 1 - 1e-5)
    r = float(y.mean())
    ref = r * (1.0 - r)
    return 1e5 * (1.0 - np.mean((p - y) ** 2) / ref)

def metric(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-5, 1 - 1e-5)
    brier = float(np.mean((p - y) ** 2))
    r = float(y.mean())
    ref = r * (1.0 - r)
    bss_score = 1e5 * (1.0 - brier / ref)
    return {'brier': brier, 'bss': bss_score}

def main():
    print("=== Testing Unified Multi-Channel Architecture ===")
    raw = pd.read_csv(ROOT / 'data/train.csv', low_memory=False)
    val_2024 = raw.loc[raw.season == 2024].copy().reset_index(drop=True)
    y_2024 = val_2024.control_success.to_numpy(float)
    is_r = (val_2024.game_type == 'R').to_numpy()
    is_f = (val_2024.game_type == 'F').to_numpy()
    
    oof_063 = pd.read_csv(ROOT / 'model/REF4-TRAINONLY-R-LGBM-CONSERVATIVE-063A/oof_predictions.csv').set_index('row_id')
    oof_051 = pd.read_csv(ROOT / 'model/REF4-TRAINONLY-R-SPECIFIC-SPLIT-051A/oof_predictions.csv').set_index('row_id')
    
    valid_row_ids = val_2024.row_id.to_numpy()
    p_063 = oof_063.loc[valid_row_ids, 'candidate_prediction'].to_numpy(float)
    p_051 = oof_051.loc[valid_row_ids, 'candidate_prediction'].to_numpy(float)
    base_exact = oof_051.loc[valid_row_ids, 'baseline_prediction'].to_numpy(float)
    
    print(f"2024 Total Rows: {len(val_2024)} (1군 R: {is_r.sum()}, 2군 F: {is_f.sum()})")
    print(f"2024 Base Exact BSS: {bss(y_2024, base_exact):.4f} (R: {bss(y_2024[is_r], base_exact[is_r]):.4f}, F: {bss(y_2024[is_f], base_exact[is_f]):.4f})")
    print(f"2024 051A BSS:       {bss(y_2024, p_051):.4f} (R: {bss(y_2024[is_r], p_051[is_r]):.4f}, F: {bss(y_2024[is_f], p_051[is_f]):.4f})")
    print(f"2024 063A BSS:       {bss(y_2024, p_063):.4f} (R: {bss(y_2024[is_r], p_063[is_r]):.4f}, F: {bss(y_2024[is_f], p_063[is_f]):.4f})")
    
if __name__ == '__main__':
    main()
