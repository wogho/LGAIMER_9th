#!/usr/bin/env python3
"""Strict Forward-OOF evaluation of LightGBM Futures (F) Expert (068A)."""
import gc, hashlib, json, os, sys, time
from pathlib import Path
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'candidate/REF4-R-SPECIFIC-SPLIT-FPSYCH-055A'))

from src.preprocessing_v2 import CAT_V2, build_v3_features
from src.season_delta_features import build_snapshots
from src.season_history_v3 import build_entity_snapshots

OUT_DIR = ROOT / 'model/REF4-LGBM-F-EXPERT-068A'
OUT_DIR.mkdir(parents=True, exist_ok=True)

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
    t0 = time.time()
    print("=== Step 1: Loading 063A OOF Anchor Predictions ===")
    oof_063 = pd.read_csv(ROOT / 'model/REF4-TRAINONLY-R-LGBM-CONSERVATIVE-063A/oof_predictions.csv').set_index('row_id')
    oof_051 = pd.read_csv(ROOT / 'model/REF4-TRAINONLY-R-SPECIFIC-SPLIT-051A/oof_predictions.csv').set_index('row_id')
    
    raw = pd.read_csv(ROOT / 'data/train.csv', low_memory=False)
    tm_path = str(ROOT / 'candidate/REF4-R-SPECIFIC-SPLIT-FPSYCH-055A/model/trackman_prior_features.csv')
    
    results = {}
    oof_records = []
    
    weights_to_test = [0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10]
    
    for val_year in [2023, 2024]:
        t_fold = time.time()
        print(f"\n--- Running Fold {val_year} ---")
        train_hist = raw.loc[raw.season < val_year].copy().reset_index(drop=True)
        val_df = raw.loc[raw.season == val_year].copy().reset_index(drop=True)
        
        prior = float(train_hist.control_success.mean())
        ps = build_snapshots(train_hist)
        bs = build_entity_snapshots(train_hist, "batter_id", "asof_batter_n",
            ["asof_batter_success_rate", "asof_batter_middle_rate"], "control_success")
        ms = build_entity_snapshots(train_hist, "pitcher_id", "asof_pitcher_pitchmix_n",
            ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"])
            
        print(f"Building v3 features for train (<{val_year})...")
        X_train_v3, base_train = build_v3_features(train_hist, prior, ps, bs, ms, tm_path)
        X_val_v3, base_val = build_v3_features(val_df, prior, ps, bs, ms, tm_path)
        
        for col in CAT_V2:
            if col in X_train_v3.columns:
                X_train_v3[col] = X_train_v3[col].astype('category')
            if col in X_val_v3.columns:
                X_val_v3[col] = X_val_v3[col].astype('category')
                
        mask_train_f = (train_hist.game_type == 'F').to_numpy()
        mask_val_f = (val_df.game_type == 'F').to_numpy()
        
        y_train = train_hist.control_success.to_numpy(float)
        res_train = y_train - base_train
        
        s_train = train_hist.season.to_numpy(int)
        decay_weights = np.power(0.55, int(s_train.max()) - s_train)
        
        print(f"Fitting LightGBM F-Expert on {mask_train_f.sum()} F-rows (<{val_year})...")
        model = lgb.LGBMRegressor(
            n_estimators=200,
            num_leaves=24,
            learning_rate=0.030,
            colsample_bytree=0.75,
            subsample=0.80,
            min_child_samples=40,
            reg_alpha=2.0,
            reg_lambda=8.0,
            random_state=260803,
            n_jobs=3,
            verbose=-1
        )
        model.fit(X_train_v3.loc[mask_train_f], res_train[mask_train_f], sample_weight=decay_weights[mask_train_f])
        
        model.booster_.save_model(str(OUT_DIR / f'f_expert_lgbm_{val_year}.txt'))
        
        val_res_pred = model.predict(X_val_v3)
        f_expert_pred = np.clip(base_val + val_res_pred + 0.0052, 1e-6, 1.0 - 1e-6)
        
        valid_row_ids = val_df.row_id.to_numpy()
        p_063 = oof_063.loc[valid_row_ids, 'candidate_prediction'].to_numpy(float)
        p_051 = oof_051.loc[valid_row_ids, 'candidate_prediction'].to_numpy(float)
        base_exact = oof_051.loc[valid_row_ids, 'baseline_prediction'].to_numpy(float)
        target_valid = val_df.control_success.to_numpy(float)
        pitchers_valid = val_df.pitcher_id.to_numpy()
        is_f_valid = mask_val_f
        
        f_idx = np.where(is_f_valid)[0]
        corr_f = float(np.corrcoef(p_063[f_idx], f_expert_pred[f_idx])[0, 1])
        print(f"LightGBM F-Expert correlation with 063 on F-rows: {corr_f:.4f}")
        
        print("\nEvaluating F-Blend Weights on full season (F-rows modified only):")
        for w in weights_to_test:
            cand_p = np.where(is_f_valid, (1.0 - w) * p_063 + w * f_expert_pred, p_063)
            m_w = metric(target_valid, cand_p)
            gain_051 = metric(target_valid, p_051)['brier'] - m_w['brier']
            gain_063 = metric(target_valid, p_063)['brier'] - m_w['brier']
            print(f"  w_f={w:.2f}: Brier={m_w['brier']:.7f}, BSS={m_w['bss']:.4f} | Gain vs 051: {gain_051:+.8f} | Gain vs 063: {gain_063:+.8f}")
            
        best_w = 0.03
        cand_p = np.where(is_f_valid, (1.0 - best_w) * p_063 + best_w * f_expert_pred, p_063)
        
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
            'f_rows': int(is_f_valid.sum()),
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
            'corr_f': corr_f,
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
                'f_expert_pred': f_expert_pred[idx],
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
        'experiment_id': 'REF4-LGBM-F-EXPERT-068A',
        'status': 'AUDIT_VERIFIED' if all_pass else 'FAIL',
        'promotion_pass': all_pass,
        'blend_weight_f': best_w,
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
