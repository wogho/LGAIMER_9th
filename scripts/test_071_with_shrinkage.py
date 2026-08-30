#!/usr/bin/env python3
"""Strict Forward-OOF test of 071A + Alpha Shrinkage across 2023 and 2024."""
import json, sys, time
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

def metric(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-5, 1 - 1e-5)
    brier = float(np.mean((p - y) ** 2))
    r = float(y.mean())
    ref = r * (1.0 - r)
    bss_score = 1e5 * (1.0 - brier / ref)
    return {'brier': brier, 'bss': bss_score}

def cluster_bootstrap_brier_gain(y, p_cand, p_ref, clusters, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    unique_clusters = np.unique(clusters)
    n_c = len(unique_clusters)
    cluster_indices = {c: np.where(clusters == c)[0] for c in unique_clusters}
    cand_se = (p_cand - y) ** 2
    ref_se = (p_ref - y) ** 2
    diff_se = ref_se - cand_se
    cluster_means = np.array([np.sum(diff_se[cluster_indices[c]]) for c in unique_clusters])
    cluster_lens = np.array([len(cluster_indices[c]) for c in unique_clusters])
    boot_gains = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample_c = rng.integers(0, n_c, size=n_c)
        tot_diff = np.sum(cluster_means[sample_c])
        tot_rows = np.sum(cluster_lens[sample_c])
        boot_gains[i] = tot_diff / tot_rows
    ci_low, ci_high = np.percentile(boot_gains, [2.5, 97.5])
    return float(np.mean(diff_se)), float(ci_low), float(ci_high)

def main():
    raw = pd.read_csv(ROOT / 'data/train.csv', low_memory=False)
    oof_051 = pd.read_csv(ROOT / 'model/REF4-TRAINONLY-R-SPECIFIC-SPLIT-051A/oof_predictions.csv').set_index('row_id')
    oof_069 = pd.read_csv(ROOT / 'model/REF4-ADAPTIVE-CHANNEL-OPT-069A/oof_predictions.csv').set_index('row_id')
    
    y_all = raw.loc[raw.season.isin([2023, 2024]), 'control_success'].to_numpy(float)
    row_ids = raw.loc[raw.season.isin([2023, 2024]), 'row_id'].to_numpy()
    pitchers = raw.loc[raw.season.isin([2023, 2024]), 'pitcher_id'].to_numpy()
    
    p_051 = oof_051.loc[row_ids, 'candidate_prediction'].to_numpy(float)
    p_071 = oof_069.loc[row_ids, 'prediction'].to_numpy(float)
    
    print(f"051A Baseline Pooled BSS: {bss(y_all, p_051):.4f}")
    print(f"071A Champion Pooled BSS: {bss(y_all, p_071):.4f}")
    
    for alpha in [1.00, 0.99, 0.98, 0.97, 0.96, 0.95, 0.94, 0.92, 0.90]:
        # Variance shrinkage centered around empirical mean
        p_shrunk = np.mean(p_071) + alpha * (p_071 - np.mean(p_071))
        gain, ci_low, ci_high = cluster_bootstrap_brier_gain(y_all, p_shrunk, p_071, pitchers, n_boot=2000, seed=42)
        gain_vs_051, ci_low_051, ci_high_051 = cluster_bootstrap_brier_gain(y_all, p_shrunk, p_051, pitchers, n_boot=2000, seed=100)
        
        m = metric(y_all, p_shrunk)
        print(f"Alpha {alpha:.2f} -> Brier: {m['brier']:.7f}, BSS: {m['bss']:8.4f} | Gain vs 071: {gain:+.8f} (CI: [{ci_low:+.8f}, {ci_high:+.8f}]) | Gain vs 051: {gain_vs_051:+.8f}")

if __name__ == '__main__':
    main()
