#!/usr/bin/env python3
"""Run REF4-TRAINONLY-R-MULTISEED-LGBM-062A forward OOF training and evaluation."""
import gc, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")
try:
    os.nice(10)
except OSError:
    pass

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'model/REF4-TRAINONLY-R-MULTISEED-LGBM-062A'
CANDIDATE = ROOT / 'candidate/REF4-CHAMPION-STACK-030'
sys.path.insert(0, str(CANDIDATE))

from src.preprocessing_v2 import CAT_V2, build_v3_features
from src.season_delta_features import build_snapshots
from src.season_history_v3 import build_entity_snapshots

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def metric(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    b = float(np.mean((p - y) ** 2))
    ref = float(np.mean((y - y.mean()) ** 2))
    q = 1.0 - b / ref
    return {'rows': len(y), 'target_rate': float(y.mean()), 'brier': b, 'bss': q, 'local_score': 1e5 * q}

def boot(y: np.ndarray, b: np.ndarray, p: np.ndarray, g: np.ndarray, reps: int, seed: int) -> dict[str, object]:
    d = (b - y) ** 2 - (p - y) ** 2
    codes, u = pd.factorize(g, sort=True)
    s = np.bincount(codes, weights=d, minlength=len(u))
    n = np.bincount(codes, minlength=len(u))
    rng = np.random.default_rng(seed)
    v = np.empty(reps)
    for i in range(reps):
        q = rng.integers(0, len(u), len(u))
        v[i] = s[q].sum() / n[q].sum()
    return {
        'clusters': len(u), 'repetitions': reps, 'seed': seed,
        'brier_gain': float(d.mean()),
        'ci_low': float(np.quantile(v, 0.025)),
        'ci_high': float(np.quantile(v, 0.975))
    }

def main():
    t0 = time.time()
    c = json.loads((OUT / 'audit_contract.json').read_text(encoding='utf-8'))
    pre = json.loads((OUT / 'preflight_report.json').read_text(encoding='utf-8'))
    assert pre['status'] == 'AUDIT_VERIFIED' and pre['fail_count'] == 0
    
    raw = pd.read_csv(ROOT / c['official_train'], low_memory=False)
    p_051_df = pd.read_csv(ROOT / c['curr_051_predictions']).set_index('row_id')
    p_f_df = pd.read_csv(ROOT / c['f_psych_predictions']).set_index('row_id')
    
    weights_grid = c['blend_weights_grid']
    primary_w = c['primary_candidate_weight']
    seeds = c['r_expert_params']['seeds']
    
    folds = {}
    oof_rows = []
    
    for valid_year in (2023, 2024):
        t_fold = time.time()
        print(f"\n--- Starting Fold {valid_year} ---", flush=True)
        train_hist = raw.loc[raw.season < valid_year].reset_index(drop=True)
        valid_hist = raw.loc[raw.season == valid_year].reset_index(drop=True)
        
        prior = float(train_hist.control_success.mean())
        ps = build_snapshots(train_hist)
        bs = build_entity_snapshots(train_hist, 'batter_id', 'asof_batter_n',
                                   ['asof_batter_success_rate', 'asof_batter_middle_rate'], 'control_success')
        ms = build_entity_snapshots(train_hist, 'pitcher_id', 'asof_pitcher_pitchmix_n',
                                   ['asof_pitcher_fastball_rate', 'asof_pitcher_breaking_rate', 'asof_pitcher_offspeed_rate'])
        
        tm_path = ROOT / f"model/REF4-EXACT-OOF-031A/fold_{valid_year}/trackman_prior_features.csv"
        assert tm_path.exists(), f"Trackman prior missing for {valid_year}"
        
        print(f"Building v3 features for train ({len(train_hist)}) and valid ({len(valid_hist)})...", flush=True)
        X_train_v3, base_train = build_v3_features(train_hist, prior, ps, bs, ms, str(tm_path))
        X_valid_v3, base_valid = build_v3_features(valid_hist, prior, ps, bs, ms, str(tm_path))
        
        # Prepare categoricals for LightGBM
        for col in CAT_V2:
            if col in X_train_v3.columns:
                X_train_v3[col] = X_train_v3[col].astype('category')
                X_valid_v3[col] = X_valid_v3[col].astype('category')
                
        mask_r = (train_hist.game_type == 'R').to_numpy()
        y_train = train_hist.control_success.to_numpy(float)
        res_train = y_train - base_train
        
        s_train = train_hist.season.to_numpy(int)
        decay_weights = np.power(c['r_expert_params']['decay'], int(s_train.max()) - s_train)
        
        seed_predictions = []
        p = c['r_expert_params']
        
        for s_idx, seed_val in enumerate(seeds):
            print(f"Fitting LightGBM Seed {s_idx+1}/{len(seeds)} (seed={seed_val}) on {mask_r.sum()} R-rows...", flush=True)
            model = lgb.LGBMRegressor(
                n_estimators=p['n_estimators'],
                num_leaves=p['num_leaves'],
                learning_rate=p['learning_rate'],
                colsample_bytree=p['colsample_bytree'],
                subsample=p['subsample'],
                min_child_samples=p['min_child_samples'],
                reg_alpha=p['reg_alpha'],
                reg_lambda=p['reg_lambda'],
                random_state=seed_val,
                n_jobs=p['n_jobs'],
                verbose=-1
            )
            model.fit(X_train_v3.loc[mask_r], res_train[mask_r], sample_weight=decay_weights[mask_r])
            model_path = OUT / f"r_expert_lgbm_{valid_year}_seed_{seed_val}.txt"
            model.booster_.save_model(str(model_path))
            
            res_pred = model.predict(X_valid_v3)
            seed_predictions.append(res_pred)
            del model
            gc.collect()
            
        r_res_ensemble = np.mean(seed_predictions, axis=0)
        r_pred_valid = np.clip(base_valid + r_res_ensemble + 0.0052, 1e-6, 1.0 - 1e-6)
        
        # Merge with 051A
        valid_row_ids = valid_hist.row_id.to_numpy()
        pred_051 = p_051_df.loc[valid_row_ids, 'candidate_prediction'].to_numpy(float)
        base_exact = p_051_df.loc[valid_row_ids, 'baseline_prediction'].to_numpy(float)
        pred_f = p_f_df.loc[valid_row_ids, 'gated_prediction'].to_numpy(float)
        target_valid = valid_hist.control_success.to_numpy(float)
        is_r_valid = (valid_hist.game_type == 'R').to_numpy()
        
        # Correlation diagnostic
        r_idx = np.where(is_r_valid)[0]
        corr_with_051 = float(np.corrcoef(pred_051[r_idx], r_pred_valid[r_idx])[0, 1])
        print(f"3-Seed Ensemble correlation with 051 on R-rows: {corr_with_051:.4f}", flush=True)
        
        # Blend candidates
        blend_results = {}
        for w in weights_grid:
            cand_p = np.where(is_r_valid, (1.0 - w) * pred_051 + w * r_pred_valid, pred_f)
            m_cand = metric(target_valid, cand_p)
            gain_vs_base = metric(target_valid, base_exact)['brier'] - m_cand['brier']
            gain_vs_051 = metric(target_valid, pred_051)['brier'] - m_cand['brier']
            blend_results[f"w_{w:.2f}"] = {
                'brier': m_cand['brier'],
                'bss': m_cand['bss'],
                'gain_vs_base': gain_vs_base,
                'gain_vs_051': gain_vs_051
            }
        
        # Primary candidate (w = primary_w)
        primary_cand = np.where(is_r_valid, (1.0 - primary_w) * pred_051 + primary_w * r_pred_valid, pred_f)
        m_base = metric(target_valid, base_exact)
        m_051 = metric(target_valid, pred_051)
        m_prim = metric(target_valid, primary_cand)
        
        ci_base = boot(target_valid, base_exact, primary_cand, valid_hist.pitcher_id.to_numpy(),
                       int(c['promotion_gate']['cluster_bootstrap_repetitions']),
                       int(c['promotion_gate'][f'cluster_bootstrap_seed_{valid_year}']))
        ci_051 = boot(target_valid, pred_051, primary_cand, valid_hist.pitcher_id.to_numpy(),
                      int(c['promotion_gate']['cluster_bootstrap_repetitions']),
                      int(c['promotion_gate'][f'cluster_bootstrap_seed_{valid_year}']))
        
        folds[str(valid_year)] = {
            'valid_rows': len(valid_hist),
            'valid_R_rows': int(is_r_valid.sum()),
            'valid_F_rows': int((~is_r_valid).sum()),
            'seeds_trained': seeds,
            'corr_with_051_on_R': corr_with_051,
            'baseline': m_base,
            'curr_051': m_051,
            'primary_candidate': m_prim,
            'brier_gain_vs_base': m_base['brier'] - m_prim['brier'],
            'bss_gain_vs_base': m_prim['bss'] - m_base['bss'],
            'brier_gain_vs_051': m_051['brier'] - m_prim['brier'],
            'cluster_ci_vs_base': ci_base,
            'cluster_ci_vs_051': ci_051,
            'weights_grid_evaluation': blend_results,
            'fold_elapsed_seconds': time.time() - t_fold
        }
        print(f"Fold {valid_year} done: Gain vs Base = {folds[str(valid_year)]['brier_gain_vs_base']:.8f}, Gain vs 051 = {folds[str(valid_year)]['brier_gain_vs_051']:.8f}")
        
        fold_df = pd.DataFrame({
            'row_id': valid_row_ids,
            'season': valid_year,
            'pitcher_id': valid_hist.pitcher_id,
            'game_type': valid_hist.game_type,
            'target': target_valid,
            'baseline_prediction': base_exact,
            'curr_051_prediction': pred_051,
            'r_expert_multiseed_lgbm_prediction': r_pred_valid,
            'candidate_prediction': primary_cand
        })
        oof_rows.append(fold_df)
        
        del X_train_v3, X_valid_v3, train_hist, valid_hist, ps, bs, ms
        gc.collect()

    all_oof = pd.concat(oof_rows, ignore_index=True)
    y_all = all_oof.target.to_numpy(float)
    base_all = all_oof.baseline_prediction.to_numpy(float)
    p051_all = all_oof.curr_051_prediction.to_numpy(float)
    cand_all = all_oof.candidate_prediction.to_numpy(float)
    
    pb = metric(y_all, base_all)
    p51 = metric(y_all, p051_all)
    pc = metric(y_all, cand_all)
    
    pooled_gain_vs_base = pb['brier'] - pc['brier']
    pooled_gain_vs_051 = p51['brier'] - pc['brier']
    
    gates = {
        '2023_brier_gain_vs_base_positive': folds['2023']['brier_gain_vs_base'] > 0,
        '2024_brier_gain_vs_base_positive': folds['2024']['brier_gain_vs_base'] > 0,
        'pooled_brier_gain_vs_base_positive': pooled_gain_vs_base > 0,
        'worst_season_bss_gain_positive': min(folds[k]['bss_gain_vs_base'] for k in ('2023', '2024')) > 0,
        '2023_cluster_ci_low_vs_base_positive': folds['2023']['cluster_ci_vs_base']['ci_low'] > 0,
        '2024_cluster_ci_low_vs_base_positive': folds['2024']['cluster_ci_vs_base']['ci_low'] > 0,
        '2023_gain_vs_051_positive': folds['2023']['brier_gain_vs_051'] > 0,
        '2024_gain_vs_051_positive': folds['2024']['brier_gain_vs_051'] > 0,
        'pooled_gain_vs_051_material': pooled_gain_vs_051 >= c['promotion_gate']['min_gain_vs_051']
    }
    promotion = all(gates.values())
    
    result = {
        'experiment_id': c['experiment_id'],
        'candidate_name': f'R_multiseed_lgbm_expert_3seeds_blend_w{primary_w:.2f}_plus_fixed_051_and_F_psych',
        'candidate_status': 'PENDING_AUDIT_PASS' if promotion else 'PENDING_AUDIT_FAIL',
        'folds': folds,
        'pooled': {
            'baseline': pb,
            'curr_051': p51,
            'candidate': pc,
            'brier_gain_vs_base': pooled_gain_vs_base,
            'brier_gain_vs_051': pooled_gain_vs_051,
            'min_required_gain_vs_051': c['promotion_gate']['min_gain_vs_051']
        },
        'gate_checks': gates,
        'gate_checks_count': len(gates),
        'promotion_pass': promotion,
        'actual_leaf_count': 1,
        'training_performed': True,
        'test_read': False,
        'test_inference_performed': False,
        'production_assets_created': False,
        'candidate_bundle_created': False,
        'zip_created': False,
        'elapsed_seconds': time.time() - t0
    }
    
    all_oof.to_csv(OUT / 'oof_predictions.csv', index=False)
    (OUT / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    (OUT / 'result.md').write_text(
        f"# {c['experiment_id']}\n\n"
        f"- status: `{result['candidate_status']}`\n"
        f"- promotion pass: `{str(promotion).lower()}`\n"
        f"- pooled gain vs base: `{pooled_gain_vs_base:.17g}`\n"
        f"- pooled gain vs 051: `{pooled_gain_vs_051:.17g}`\n"
    )
    
    # Manifest
    paths = [
        OUT / 'audit_contract.json', OUT / 'preflight_report.json', OUT / 'preflight_report.md',
        OUT / 'result.json', OUT / 'result.md', OUT / 'oof_predictions.csv',
        ROOT / c['official_train'], ROOT / c['curr_051_predictions'], ROOT / c['curr_051_attestation'],
        ROOT / c['f_psych_predictions'], ROOT / c['f_psych_attestation'], ROOT / c['preserve_zip'],
        ROOT / '01_제약과금지사항.md',
        ROOT / 'scripts/preflight_ref4_r_multiseed_lgbm_062a.py',
        ROOT / 'scripts/run_ref4_r_multiseed_lgbm_062a.py',
        ROOT / 'scripts/verify_ref4_r_multiseed_lgbm_062a.py'
    ] + [ROOT / p for p in c['base_oof'].values()]
    
    for yr in (2023, 2024):
        for s_val in seeds:
            paths.append(OUT / f"r_expert_lgbm_{yr}_seed_{s_val}.txt")
            
    artifacts = {str(p.relative_to(ROOT)): {'sha256': sha256_file(p), 'size': p.stat().st_size} for p in paths if p.exists()}
    manifest = {
        'experiment_id': c['experiment_id'],
        'status': 'PENDING_VALIDATION',
        'artifact_count': len(artifacts),
        'artifacts': artifacts,
        'leaf_count': 1,
        'gate_count': len(gates),
        'oof_rows': len(all_oof)
    }
    (OUT / 'audit_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(f"\n062A Execution finished in {time.time()-t0:.2f}s. Result: {result['candidate_status']}, Promotion: {promotion}")
    print(json.dumps({'pooled_gain_vs_base': pooled_gain_vs_base, 'pooled_gain_vs_051': pooled_gain_vs_051, 'gates': gates}, indent=2))

if __name__ == '__main__':
    main()
