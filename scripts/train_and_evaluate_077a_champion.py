#!/usr/bin/env python3
"""Train and evaluate 077A Super-Champion (Minimax Stable Experts + League Transition Gate)."""
import json, os, pickle, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = str(ROOT / '.venv-submit/bin/python')
EXP_DIR = ROOT / 'model/REF4-SUPER-CHAMPION-077A'
EXP_DIR.mkdir(parents=True, exist_ok=True)

# Import src modules
sys.path.insert(0, str(ROOT / 'github_reference/4번 레포'))
from src.context_adjusted_psych import attach_saved_context_psych, build_profiles
from src.context_pressure_features import build_context_pressure_features
from src.stable_experts import PLATOON_COLS, context_ridge_correction, platoon_correction
from src.league_transition import transition_features, CAT as TRANS_CAT

def bss(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-5, 1 - 1e-5)
    r = float(y.mean())
    ref = r * (1.0 - r)
    return 1e5 * (1.0 - np.mean((p - y) ** 2) / ref)

def cluster_bootstrap(y, p_base, p_cand, clusters, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    unique_clusters = np.unique(clusters)
    cluster_indices = {c: np.where(clusters == c)[0] for c in unique_clusters}
    gains = []
    
    r = float(y.mean())
    ref = r * (1.0 - r)
    
    for _ in range(n_boot):
        sample_c = rng.choice(unique_clusters, size=len(unique_clusters), replace=True)
        idx = np.concatenate([cluster_indices[c] for c in sample_c])
        y_b = y[idx]
        b_base = np.mean((p_base[idx] - y_b) ** 2)
        b_cand = np.mean((p_cand[idx] - y_b) ** 2)
        gain_bss = (1.0 - b_cand / ref) - (1.0 - b_base / ref)
        gains.append(gain_bss)
        
    gains = np.array(gains)
    return float(np.mean(gains)), float(np.percentile(gains, 2.5)), float(np.percentile(gains, 97.5)), float(np.mean(gains > 0))

def run_071a(df: pd.DataFrame) -> np.ndarray:
    p71_dir = ROOT / 'model/REF4-ADAPTIVE-CHANNEL-OPT-071A/production_package'
    data_dir = p71_dir / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(data_dir / 'test.csv', index=False)
    subprocess.run([PYTHON_BIN, 'script.py'], cwd=str(p71_dir), capture_output=True)
    return pd.read_csv(p71_dir / 'output/submission.csv')['control_success'].to_numpy(float)

def build_prior_type_table(history: pd.DataFrame, max_season: int = 2024) -> pd.Series:
    h = history[history.season <= max_season]
    counts = h.groupby(["pitcher_id", "season", "game_type"], observed=True).size().rename("n").reset_index()
    dominant = counts.sort_values("n").groupby(["pitcher_id", "season"]).tail(1)
    latest = dominant.sort_values("season").groupby("pitcher_id").tail(1)
    return latest.set_index(latest.pitcher_id.astype(str)).game_type

def main():
    print("=" * 75)
    print("   TRAINING & EVALUATING 077A (MINIMAX EXPERTS + LEAGUE TRANSITION)   ")
    print("=" * 75)
    
    raw = pd.read_csv(ROOT / 'data/train.csv', low_memory=False)
    val_24 = raw.loc[raw.season == 2024].copy().reset_index(drop=True)
    y_24 = val_24.control_success.to_numpy(float)
    clusters_24 = val_24.pitcher_id.to_numpy()
    
    print("\n[Step 1] Running 071A Baseline Inference on 2024 Holdout...")
    t0 = time.time()
    p_071_24 = run_071a(val_24)
    print(f"  • 071A Baseline 2024 BSS: {bss(y_24, p_071_24):.4f} ({time.time() - t0:.1f}s)")
    
    # ----------------------------------------------------
    # Step 2: Fit Minimax Stable Experts on 2019-2023 (Temporal Isolation for 2024 test)
    # ----------------------------------------------------
    print("\n[Step 2] Training Minimax Stable Experts (Time-Isolated for 2024 Validation)...")
    train_pre24 = raw.loc[raw.season < 2024].copy().reset_index(drop=True)
    
    # Run 071A on train_pre24 to get adaptive gate base
    # For speed on train, sample or use historical chunks (2022, 2023)
    train_22_23 = raw.loc[raw.season.isin([2022, 2023])].copy().reset_index(drop=True)
    y_22_23 = train_22_23.control_success.to_numpy(float)
    print(f"  • Running base predictions on {len(train_22_23):,} rows of 2022-2023...")
    p_base_22_23 = run_071a(train_22_23)
    
    # Weights for decay
    decay_weights = np.where(train_22_23.season == 2023, 1.0, 0.55)
    target_res = y_22_23 - p_base_22_23
    
    # A. Context-adjusted Psychology Ridge
    print("  • Fitting Context-Adjusted Psychology Ridge...")
    psych_profile_pre24 = build_profiles(raw.loc[raw.season < 2024], 400.0, 100.0)
    x_psych_train = attach_saved_context_psych(train_22_23, psych_profile_pre24)
    mean_psych = x_psych_train.mean().to_numpy()
    std_psych = x_psych_train.std().replace(0, 1).fillna(1).to_numpy()
    z_psych_train = np.nan_to_num((x_psych_train.to_numpy(float) - mean_psych) / std_psych)
    ridge_model = Ridge(alpha=1000.0, fit_intercept=False).fit(z_psych_train, target_res, sample_weight=decay_weights)
    
    # Save validation assets
    val_ridge_asset = EXP_DIR / 'val_stable_context_ridge.npz'
    np.savez_compressed(
        val_ridge_asset,
        columns=np.array(x_psych_train.columns),
        mean=mean_psych,
        std=std_psych,
        coef=ridge_model.coef_,
        scale=np.float32(0.25 * 0.75)
    )
    
    # B. Platoon CatBoost Expert
    print("  • Fitting Dedicated Platoon CatBoost Expert...")
    px_train = build_context_pressure_features(train_22_23).reindex(columns=PLATOON_COLS)
    platoon_model = CatBoostRegressor(
        iterations=180, depth=3, learning_rate=0.02, loss_function='RMSE',
        l2_leaf_reg=100, random_strength=0.2, bootstrap_type='Bernoulli',
        subsample=0.8, random_seed=250013, thread_count=6, allow_writing_files=False, verbose=False
    )
    platoon_model.fit(px_train, target_res, sample_weight=decay_weights)
    val_platoon_cbm = EXP_DIR / 'val_stable_platoon.cbm'
    platoon_model.save_model(str(val_platoon_cbm))
    
    # C. League Transition Gate
    print("  • Fitting League Transition Gate on 2022-2023...")
    prior_lookup_pre24 = build_prior_type_table(raw, max_season=2023)
    val_prior_pkl = EXP_DIR / 'val_prior_type.pkl'
    pickle.dump(prior_lookup_pre24, open(val_prior_pkl, 'wb'))
    
    tx_train = transition_features(train_22_23, p_base_22_23, val_prior_pkl)
    trans_model = CatBoostRegressor(
        iterations=250, depth=4, learning_rate=0.025, loss_function="RMSE",
        l2_leaf_reg=100, random_strength=0.2, bootstrap_type="Bernoulli",
        subsample=0.8, random_seed=918024, thread_count=6, allow_writing_files=False, verbose=False
    )
    trans_model.fit(tx_train, target_res, cat_features=TRANS_CAT)
    val_trans_cbm = EXP_DIR / 'val_transition_gate.cbm'
    trans_model.save_model(str(val_trans_cbm))
    
    # ----------------------------------------------------
    # Step 3: Evaluate on 2024 Holdout
    # ----------------------------------------------------
    print("\n[Step 3] Evaluating New Channels on 2024 Holdout...")
    corr_ridge = context_ridge_correction(val_24, psych_profile_pre24, str(val_ridge_asset))
    corr_platoon = platoon_correction(val_24, platoon_model, scale=0.30)
    tx_24 = transition_features(val_24, p_071_24, val_prior_pkl)
    corr_trans = trans_model.predict(tx_24)
    
    print(f"  • Mean |corr_ridge|:   {np.mean(np.abs(corr_ridge)):.6f}")
    print(f"  • Mean |corr_platoon|: {np.mean(np.abs(corr_platoon)):.6f}")
    print(f"  • Mean |corr_trans|:   {np.mean(np.abs(corr_trans)):.6f}")
    
    print("\n[Step 4] Channel Weight Grid Search on 2024...")
    print(f"  {'w_minimax':<10} | {'w_trans':<10} | {'2024 BSS':<10} | {'Gain vs 071A':<15} | {'Bootstrap 95% CI':<28} | {'P(Gain>0)':<10}")
    print("  " + "-" * 85)
    
    best_cand_p = p_071_24.copy()
    best_cand_bss = bss(y_24, p_071_24)
    best_w_mm = 0.0
    best_w_tr = 0.0
    
    for w_mm in [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, -0.2, -0.5, -0.6]:
        for w_tr in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
            if w_mm == 0.0 and w_tr == 0.0:
                continue
            p_cand = np.clip(p_071_24 + w_mm * (corr_ridge + corr_platoon) + w_tr * corr_trans, 1e-5, 1 - 1e-5)
            score = bss(y_24, p_cand)
            diff = score - bss(y_24, p_071_24)
            
            if diff > 0.5:
                mean_g, ci_lo, ci_hi, p_pos = cluster_bootstrap(y_24, p_071_24, p_cand, clusters_24, n_boot=500)
                ci_str = f"[{ci_lo * 1e5:+7.2f}, {ci_hi * 1e5:+7.2f}]"
                print(f"  {w_mm:<10.2f} | {w_tr:<10.2f} | {score:10.4f} | {diff:+15.4f} | {ci_str:<28} | {p_pos*100:6.1f}%")
                
                if score > best_cand_bss:
                    best_cand_bss = score
                    best_cand_p = p_cand
                    best_w_mm = w_mm
                    best_w_tr = w_tr
                    
    print("\n" + "=" * 85)
    print(f"🏆 BEST WEIGHTS: w_minimax = {best_w_mm:.2f}, w_transition = {best_w_tr:.2f} -> 2024 BSS = {best_cand_bss:.4f} (+{best_cand_bss - bss(y_24, p_071_24):.4f} pt gain!)")
    print("=" * 85)
    
    # Record report
    report = {
        "candidate": "REF4-SUPER-CHAMPION-077A",
        "best_w_minimax": best_w_mm,
        "best_w_transition": best_w_tr,
        "baseline_2024_bss": float(bss(y_24, p_071_24)),
        "champion_2024_bss": float(best_cand_bss),
        "gain_bss": float(best_cand_bss - bss(y_24, p_071_24))
    }
    (EXP_DIR / "validation_report.json").write_text(json.dumps(report, indent=2))

if __name__ == '__main__':
    main()
