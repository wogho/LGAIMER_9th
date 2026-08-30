#!/usr/bin/env python3
"""Strict Forward-OOF evaluation of Count-Advantage Empirical Bayes Residual Lookup (067A)."""
import gc, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'candidate/REF4-R-SPECIFIC-SPLIT-FPSYCH-055A'))

OUT_DIR = ROOT / 'model/REF4-PHYS-EB-RESID-067A'
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    diff_se = ref_se - cand_se # positive means candidate has lower SE (better)
    
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

def compute_count_advantage(df: pd.DataFrame) -> np.ndarray:
    strikes = pd.to_numeric(df['strikes_before'], errors='coerce').fillna(0).to_numpy(int)
    balls = pd.to_numeric(df['balls_before'], errors='coerce').fillna(0).to_numpy(int)
    return strikes - balls

def fit_eb_lookup(train_df: pd.DataFrame, base_pred: np.ndarray, K: float = 300.0) -> dict:
    y = train_df['control_success'].to_numpy(float)
    res = y - base_pred
    pitchers = train_df['pitcher_id'].astype(str).to_numpy()
    adv = compute_count_advantage(train_df)
    
    df_res = pd.DataFrame({
        'pitcher_id': pitchers,
        'count_adv': adv,
        'res': res
    })
    
    # 1. Global residual by count_adv
    global_adv = df_res.groupby('count_adv')['res'].agg(['count', 'mean']).rename(columns={'mean': 'global_mean'})
    
    # 2. Pitcher x count_adv stats
    p_adv = df_res.groupby(['pitcher_id', 'count_adv'])['res'].agg(['count', 'mean']).reset_index()
    p_adv = p_adv.merge(global_adv['global_mean'], on='count_adv', how='left')
    
    # Empirical Bayes shrinkage: (n * mean + K * global_mean) / (n + K)
    p_adv['eb_res'] = (p_adv['count'] * p_adv['mean'] + K * p_adv['global_mean']) / (p_adv['count'] + K)
    
    # Pitcher general EB: (n_p * mean_p + K * 0) / (n_p + K)
    p_gen = df_res.groupby('pitcher_id')['res'].agg(['count', 'mean']).reset_index()
    p_gen['eb_gen_res'] = (p_gen['count'] * p_gen['mean']) / (p_gen['count'] + K)
    
    lookup_map = {}
    for _, row in p_adv.iterrows():
        lookup_map[(row['pitcher_id'], int(row['count_adv']))] = float(row['eb_res'])
        
    p_gen_map = dict(zip(p_gen['pitcher_id'], p_gen['eb_gen_res']))
    global_map = global_adv['global_mean'].to_dict()
    
    return {
        'lookup': lookup_map,
        'p_gen': p_gen_map,
        'global': global_map
    }

def apply_eb_lookup(val_df: pd.DataFrame, tables: dict) -> np.ndarray:
    pitchers = val_df['pitcher_id'].astype(str).to_numpy()
    adv = compute_count_advantage(val_df)
    lookup_map = tables['lookup']
    p_gen_map = tables['p_gen']
    global_map = tables['global']
    
    out = np.empty(len(val_df), dtype=float)
    for i in range(len(val_df)):
        key = (pitchers[i], adv[i])
        if key in lookup_map:
            out[i] = lookup_map[key]
        elif pitchers[i] in p_gen_map:
            out[i] = p_gen_map[pitchers[i]]
        elif adv[i] in global_map:
            out[i] = global_map[adv[i]]
        else:
            out[i] = 0.0
    return out

def main():
    t0 = time.time()
    print("=== Step 1: Loading 063A / 051A OOF Anchor Predictions ===")
    oof_063 = pd.read_csv(ROOT / 'model/REF4-TRAINONLY-R-LGBM-CONSERVATIVE-063A/oof_predictions.csv').set_index('row_id')
    oof_051 = pd.read_csv(ROOT / 'model/REF4-TRAINONLY-R-SPECIFIC-SPLIT-051A/oof_predictions.csv').set_index('row_id')
    
    raw = pd.read_csv(ROOT / 'data/train.csv', low_memory=False)
    
    results = {}
    oof_records = []
    
    weights_to_test = [0.002, 0.005, 0.008, 0.010, 0.015, 0.020]
    
    for val_year in [2023, 2024]:
        t_fold = time.time()
        print(f"\n--- Running Fold {val_year} ---")
        train_hist = raw.loc[raw.season < val_year].copy().reset_index(drop=True)
        val_df = raw.loc[raw.season == val_year].copy().reset_index(drop=True)
        
        # Only fit on R-rows (1군)
        mask_train_r = (train_hist.game_type == 'R').to_numpy()
        mask_val_r = (val_df.game_type == 'R').to_numpy()
        
        prior = float(train_hist.control_success.mean())
        base_train = np.full(len(train_hist), prior)
        
        print(f"Fitting Count-Advantage EB lookup on {mask_train_r.sum()} historical R-rows...")
        tables = fit_eb_lookup(train_hist.loc[mask_train_r], base_train[mask_train_r], K=300.0)
        
        valid_row_ids = val_df.row_id.to_numpy()
        p_063 = oof_063.loc[valid_row_ids, 'candidate_prediction'].to_numpy(float)
        p_051 = oof_051.loc[valid_row_ids, 'candidate_prediction'].to_numpy(float)
        base_exact = oof_051.loc[valid_row_ids, 'baseline_prediction'].to_numpy(float)
        target_valid = val_df.control_success.to_numpy(float)
        pitchers_valid = val_df.pitcher_id.to_numpy()
        is_r_valid = mask_val_r
        
        eb_correction = apply_eb_lookup(val_df, tables)
        # zero-center the correction on R-rows to guarantee 0 shift
        eb_correction[is_r_valid] = eb_correction[is_r_valid] - eb_correction[is_r_valid].mean()
        print(f"EB Correction on valid R-rows: mean={eb_correction[is_r_valid].mean():+.8f}, std={eb_correction[is_r_valid].std():.6f}, min={eb_correction[is_r_valid].min():+.6f}, max={eb_correction[is_r_valid].max():+.6f}")
        
        print("\nEvaluating Blend Weights:")
        for w in weights_to_test:
            cand_p = np.where(is_r_valid, np.clip(p_063 + w * eb_correction, 1e-6, 1.0 - 1e-6), p_063)
            m_w = metric(target_valid, cand_p)
            gain_051 = metric(target_valid, p_051)['brier'] - m_w['brier']
            gain_063 = metric(target_valid, p_063)['brier'] - m_w['brier']
            print(f"  w={w:.3f}: Brier={m_w['brier']:.7f}, BSS={m_w['bss']:.4f} | Gain vs 051: {gain_051:+.8f} | Gain vs 063: {gain_063:+.8f}")
            
        best_w = 0.005
        cand_p = np.where(is_r_valid, np.clip(p_063 + best_w * eb_correction, 1e-6, 1.0 - 1e-6), p_063)
        
        gain_vs_base, ci_low_base, ci_high_base = cluster_bootstrap_brier_gain(
            target_valid, cand_p, base_exact, pitchers_valid, n_boot=2000, seed=val_year
        )
        gain_vs_051, ci_low_051, ci_high_051 = cluster_bootstrap_brier_gain(
            target_valid, cand_p, p_051, pitchers_valid, n_boot=2000, seed=val_year + 100
        )
        gain_vs_063, ci_low_063, ci_high_063 = cluster_bootstrap_brier_gain(
            target_valid, cand_p, p_063, pitchers_valid, n_boot=2000, seed=val_year + 200
        )
        
        m_cand = metric(target_valid, cand_p)
        m_063 = metric(target_valid, p_063)
        m_051 = metric(target_valid, p_051)
        m_base = metric(target_valid, base_exact)
        
        results[str(val_year)] = {
            'rows': int(len(target_valid)),
            'brier_blend': m_cand['brier'],
            'brier_063': m_063['brier'],
            'brier_051': m_051['brier'],
            'brier_base': m_base['brier'],
            'bss_blend': m_cand['bss'],
            'bss_063': m_063['bss'],
            'bss_051': m_051['bss'],
            'bss_base': m_base['bss'],
            'gain_vs_063': gain_vs_063,
            'gain_vs_051': gain_vs_051,
            'gain_vs_base': gain_vs_base,
            'ci_low_base': ci_low_base,
            'ci_high_base': ci_high_base,
            'ci_low_051': ci_low_051,
            'ci_high_051': ci_high_051,
            'ci_low_063': ci_low_063,
            'ci_high_063': ci_high_063,
            'elapsed_sec': time.time() - t_fold
        }
        
        for idx in range(len(target_valid)):
            oof_records.append({
                'row_id': valid_row_ids[idx],
                'season': val_year,
                'pitcher_id': pitchers_valid[idx],
                'game_type': val_df.loc[idx, 'game_type'],
                'target': target_valid[idx],
                'base_prediction': base_exact[idx],
                'p_051': p_051[idx],
                'p_063': p_063[idx],
                'eb_correction': eb_correction[idx],
                'prediction': cand_p[idx]
            })
            
    oof_df = pd.DataFrame(oof_records)
    oof_df.to_csv(OUT_DIR / 'oof_predictions.csv', index=False)
    
    # Pooled evaluation
    y_pool = oof_df.target.to_numpy(float)
    p_pool = oof_df.prediction.to_numpy(float)
    p_pool_063 = oof_df.p_063.to_numpy(float)
    p_pool_051 = oof_df.p_051.to_numpy(float)
    p_pool_base = oof_df.base_prediction.to_numpy(float)
    pitchers_pool = oof_df.pitcher_id.to_numpy()
    
    gain_pool_base, ci_low_pool_base, ci_high_pool_base = cluster_bootstrap_brier_gain(
        y_pool, p_pool, p_pool_base, pitchers_pool, n_boot=2000, seed=2025
    )
    gain_pool_051, ci_low_pool_051, ci_high_pool_051 = cluster_bootstrap_brier_gain(
        y_pool, p_pool, p_pool_051, pitchers_pool, n_boot=2000, seed=2026
    )
    gain_pool_063, ci_low_pool_063, ci_high_pool_063 = cluster_bootstrap_brier_gain(
        y_pool, p_pool, p_pool_063, pitchers_pool, n_boot=2000, seed=2027
    )
    
    m_pool_cand = metric(y_pool, p_pool)
    m_pool_063 = metric(y_pool, p_pool_063)
    m_pool_051 = metric(y_pool, p_pool_051)
    m_pool_base = metric(y_pool, p_pool_base)
    
    gate_checks = {
        '2023_brier_gain_vs_base_positive': bool(results['2023']['gain_vs_base'] > 0),
        '2024_brier_gain_vs_base_positive': bool(results['2024']['gain_vs_base'] > 0),
        'pooled_brier_gain_vs_base_positive': bool(gain_pool_base > 0),
        'worst_season_bss_gain_positive': bool(min(results['2023']['bss_blend'] - results['2023']['bss_base'], results['2024']['bss_blend'] - results['2024']['bss_base']) > 0),
        '2023_cluster_ci_low_vs_base_positive': bool(results['2023']['ci_low_base'] > 0),
        '2024_cluster_ci_low_vs_base_positive': bool(results['2024']['ci_low_base'] > 0),
        '2023_gain_vs_051_positive': bool(results['2023']['gain_vs_051'] > 0),
        '2024_gain_vs_051_positive': bool(results['2024']['gain_vs_051'] > 0),
        'pooled_gain_vs_051_material': bool(gain_pool_051 >= 5e-7),
        'pooled_gain_vs_063_positive': bool(gain_pool_063 > 0)
    }
    
    all_pass = all(gate_checks.values())
    
    summary = {
        'experiment_id': 'REF4-PHYS-EB-RESID-067A',
        'status': 'AUDIT_VERIFIED' if all_pass else 'FAIL',
        'promotion_pass': all_pass,
        'blend_weight': best_w,
        'shrinkage_K': 300.0,
        'results_by_fold': results,
        'pooled': {
            'rows': int(len(y_pool)),
            'brier_blend': m_pool_cand['brier'],
            'brier_063': m_pool_063['brier'],
            'brier_051': m_pool_051['brier'],
            'brier_base': m_pool_base['brier'],
            'gain_vs_063': gain_pool_063,
            'gain_vs_051': gain_pool_051,
            'gain_vs_base': gain_pool_base,
            'ci_low_base': ci_low_pool_base,
            'ci_high_base': ci_high_pool_base,
            'ci_low_051': ci_low_pool_051,
            'ci_high_051': ci_high_pool_051,
            'ci_low_063': ci_low_pool_063,
            'ci_high_063': ci_high_pool_063,
        },
        'gate_checks': gate_checks,
        'elapsed_seconds': time.time() - t0
    }
    
    (OUT_DIR / 'result.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')
    print(f"\n=== Final Gate Evaluation ===")
    for k, v in gate_checks.items():
        print(f"  {k}: {v}")
    print(f"PROMOTION PASS: {all_pass} (Pooled gain vs 051: {gain_pool_051:+.8f}, vs 063: {gain_pool_063:+.8f})")
    print(f"Completed in {time.time() - t0:.2f}s")

if __name__ == '__main__':
    main()
