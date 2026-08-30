import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
CAND_069 = ROOT / 'model/REF4-ADAPTIVE-CHANNEL-OPT-069A/production_package'
sys.path.insert(0, str(CAND_069))

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
    val_24 = raw.loc[raw.season == 2024].copy().reset_index(drop=True)
    y_24 = val_24.control_success.to_numpy(float)
    is_f_24 = (val_24.game_type == 'F').to_numpy()
    is_r_24 = (val_24.game_type == 'R').to_numpy()
    
    oof_051 = pd.read_csv(ROOT / 'model/REF4-TRAINONLY-R-SPECIFIC-SPLIT-051A/oof_predictions.csv').set_index('row_id')
    p_051_24 = oof_051.loc[val_24.row_id, 'candidate_prediction'].to_numpy(float)
    
    oof_069 = pd.read_csv(ROOT / 'model/REF4-ADAPTIVE-CHANNEL-OPT-069A/oof_predictions.csv').set_index('row_id')
    p_071_24 = oof_069.loc[val_24.row_id, 'prediction'].to_numpy(float)
    
    print(f"051A Base 2024 BSS: {bss(y_24, p_051_24):.4f}")
    print(f"071A Champion 2024 BSS: {bss(y_24, p_071_24):.4f}")
    
    # Let's inspect 2군 vs 1군 separate performance under 071A
    print(f"  1군 071A BSS: {bss(y_24[is_r_24], p_071_24[is_r_24]):.4f}")
    print(f"  2군 071A BSS: {bss(y_24[is_f_24], p_071_24[is_f_24]):.4f}")

if __name__ == '__main__':
    main()
