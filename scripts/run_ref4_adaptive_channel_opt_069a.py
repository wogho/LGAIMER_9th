#!/usr/bin/env python3
"""Strict Forward-OOF evaluation of Zero-Centered Adaptive Gate + Multi-Channel Ensemble (069A)."""
import gc, hashlib, json, os, sys, time
from pathlib import Path
from catboost import CatBoostClassifier, CatBoostRegressor
import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CAND_DIR = ROOT / 'candidate/REF4-R-SPECIFIC-SPLIT-FPSYCH-055A'
sys.path.insert(0, str(CAND_DIR))
sys.path.insert(0, str(ROOT / 'github_reference/4번 레포'))

from src.preprocessing_v2 import build_v2_features, build_v3_features, CAT_V2
from src.adaptive_gate import build_gate_features

OUT_DIR = ROOT / 'model/REF4-ADAPTIVE-CHANNEL-OPT-069A'
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
    print("=== Step 1: Loading Data and 051A Anchor Predictions ===")
    oof_051 = pd.read_csv(ROOT / 'model/REF4-TRAINONLY-R-SPECIFIC-SPLIT-051A/oof_predictions.csv').set_index('row_id')
    oof_063 = pd.read_csv(ROOT / 'model/REF4-TRAINONLY-R-LGBM-CONSERVATIVE-063A/oof_predictions.csv').set_index('row_id')
    raw = pd.read_csv(ROOT / 'data/train.csv', low_memory=False)
    
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

    results = {}
    oof_records = []
    
    for val_year in [2023, 2024]:
        t_fold = time.time()
        print(f"\n--- Running Fold {val_year} ---")
        # Training set for adaptive gate: seasons before val_year
        train_df = raw.loc[raw.season == (val_year - 1)].copy().reset_index(drop=True)
        val_df = raw.loc[raw.season == val_year].copy().reset_index(drop=True)
        
        y_train = train_df.control_success.to_numpy(float)
        y_val = val_df.control_success.to_numpy(float)
        
        print(f"Building features for train ({val_year - 1})...")
        x2_tr, b2_tr = build_v2_features(train_df, meta["prior"], ps, tm)
        x3_tr, b3_tr = build_v3_features(train_df, meta["prior"], ps, bs, ms, tm)
        
        print(f"Building features for valid ({val_year})...")
        x2_val, b2_val = build_v2_features(val_df, meta["prior"], ps, tm)
        x3_val, b3_val = build_v3_features(val_df, meta["prior"], ps, bs, ms, tm)
        
        preds_tr = get_base_preds(x2_tr, b2_tr, x3_tr, b3_tr)
        risks_tr = get_risks(x3_tr)
        main_tr = np.average(np.vstack(preds_tr), axis=0, weights=meta["main_weights"])
        z_tr = np.column_stack([main_tr] + risks_tr)
        p_stack_tr = meta["stack_intercept"] + z_tr @ np.asarray(meta["stack_coefficients"])
        
        preds_val = get_base_preds(x2_val, b2_val, x3_val, b3_val)
        risks_val = get_risks(x3_val)
        main_val = np.average(np.vstack(preds_val), axis=0, weights=meta["main_weights"])
        z_val = np.column_stack([main_val] + risks_val)
        p_stack_val = meta["stack_intercept"] + z_val @ np.asarray(meta["stack_coefficients"])
        
        print(f"Building gate features for fold {val_year}...")
        gate_x_tr = build_gate_features(train_df, preds_tr, risks_tr, np.clip(p_stack_tr, 1e-6, 1 - 1e-6))
        gate_x_val = build_gate_features(val_df, preds_val, risks_val, np.clip(p_stack_val, 1e-6, 1 - 1e-6))
        
        print("Fitting adaptive gate...")
        gate = CatBoostRegressor(iterations=75, depth=4, learning_rate=0.03, loss_function='RMSE', l2_leaf_reg=30, random_strength=0.2, random_seed=280033, verbose=False)
        gate.fit(gate_x_tr, y_train - p_stack_tr)
        
        gate_pred_val = gate.predict(gate_x_val)
        # Strictly zero-center the gate prediction on valid set
        gate_pred_val_clean = gate_pred_val - gate_pred_val.mean()
        
        # Save gate model
        gate.save_model(str(OUT_DIR / f'adaptive_gate_{val_year}.cbm'))
        
        valid_row_ids = val_df.row_id.to_numpy()
        p_063 = oof_063.loc[valid_row_ids, 'candidate_prediction'].to_numpy(float)
        p_051 = oof_051.loc[valid_row_ids, 'candidate_prediction'].to_numpy(float)
        base_exact = oof_051.loc[valid_row_ids, 'baseline_prediction'].to_numpy(float)
        pitchers_val = val_df.pitcher_id.to_numpy()
        is_r = (val_df.game_type == 'R').to_numpy()
        
        # Blend adaptive gate with our 063A champion
        gate_scale = 0.05
        cand_p = np.clip(p_063 + gate_scale * gate_pred_val_clean, 1e-5, 1 - 1e-5)
        
        gain_vs_base, ci_low_base, ci_high_base = cluster_bootstrap_brier_gain(
            y_val, cand_p, base_exact, pitchers_val, n_boot=2000, seed=val_year
        )
        gain_vs_051, ci_low_051, ci_high_051 = cluster_bootstrap_brier_gain(
            y_val, cand_p, p_051, pitchers_val, n_boot=2000, seed=val_year + 100
        )
        gain_vs_063, ci_low_063, ci_high_063 = cluster_bootstrap_brier_gain(
            y_val, cand_p, p_063, pitchers_val, n_boot=2000, seed=val_year + 200
        )
        
        m_cand = metric(y_val, cand_p)
        m_063 = metric(y_val, p_063)
        m_051 = metric(y_val, p_051)
        m_base = metric(y_val, base_exact)
        
        print(f"Fold {val_year} results (gate_scale={gate_scale}):")
        print(f"  Brier Blend: {m_cand['brier']:.7f} (BSS: {m_cand['bss']:.4f})")
        print(f"  Brier 063A:  {m_063['brier']:.7f} (BSS: {m_063['bss']:.4f})")
        print(f"  Brier 051A:  {m_051['brier']:.7f} (BSS: {m_051['bss']:.4f})")
        print(f"  Brier Base:  {m_base['brier']:.7f} (BSS: {m_base['bss']:.4f})")
        print(f"  Gain vs 063A: {gain_vs_063:+.8f} (CI: [{ci_low_063:+.8f}, {ci_high_063:+.8f}])")
        print(f"  Gain vs 051A: {gain_vs_051:+.8f} (CI: [{ci_low_051:+.8f}, {ci_high_051:+.8f}])")
        print(f"  Gain vs Base: {gain_vs_base:+.8f} (CI: [{ci_low_base:+.8f}, {ci_high_base:+.8f}])")
        
        results[str(val_year)] = {
            'rows': int(len(y_val)),
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
        
        for idx in range(len(y_val)):
            oof_records.append({
                'row_id': valid_row_ids[idx],
                'season': val_year,
                'pitcher_id': pitchers_val[idx],
                'game_type': val_df.loc[idx, 'game_type'],
                'target': y_val[idx],
                'base_prediction': base_exact[idx],
                'p_051': p_051[idx],
                'p_063': p_063[idx],
                'gate_correction': gate_pred_val_clean[idx],
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
        'experiment_id': 'REF4-ADAPTIVE-CHANNEL-OPT-069A',
        'status': 'AUDIT_VERIFIED' if all_pass else 'FAIL',
        'promotion_pass': all_pass,
        'gate_scale': gate_scale,
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
