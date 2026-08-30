#!/usr/bin/env python3
"""Train Production Assets for 077A Super-Champion (Minimax Experts + Transition Gate)."""
import json, os, pickle, shutil, subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
PYTHON_BIN = str(ROOT / '.venv-submit/bin/python')
PROD_DIR = ROOT / 'model/REF4-SUPER-CHAMPION-077A/production_assets'
PROD_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / 'github_reference/4번 레포'))
from src.context_adjusted_psych import attach_saved_context_psych, build_profiles
from src.context_pressure_features import build_context_pressure_features
from src.stable_experts import PLATOON_COLS
from src.league_transition import transition_features, CAT as TRANS_CAT

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
    print("      TRAINING PRODUCTION ASSETS FOR 077A SUPER-CHAMPION        ")
    print("=" * 75)
    
    raw = pd.read_csv(ROOT / 'data/train.csv', low_memory=False)
    print(f"Total training data loaded: {len(raw):,} rows (Seasons 2019-2024)")
    
    # 1. Base Predictions on 2022-2024 for Production Target
    train_recent = raw.loc[raw.season.isin([2022, 2023, 2024])].copy().reset_index(drop=True)
    y_recent = train_recent.control_success.to_numpy(float)
    print(f"\n[Step 1] Computing 071A Base Predictions on {len(train_recent):,} rows (2022-2024)...")
    t0 = time.time()
    p_base_recent = run_071a(train_recent)
    print(f"  • Base predictions computed in {time.time() - t0:.1f}s")
    
    # Recency decay weights (2024=1.0, 2023=0.55, 2022=0.55^2)
    decay_map = {2024: 1.0, 2023: 0.55, 2022: 0.55**2}
    decay_weights = train_recent.season.map(decay_map).to_numpy(float)
    target_res = y_recent - p_base_recent
    
    # 2. Build Production Profiles & Context Ridge
    print("\n[Step 2] Building Production Psychology Profile and Context Ridge...")
    prod_psych_profile = build_profiles(raw, 400.0, 100.0)
    pickle.dump(prod_psych_profile, open(PROD_DIR / 'stable_context_profile.pkl', 'wb'))
    
    x_psych_prod = attach_saved_context_psych(train_recent, prod_psych_profile)
    mean_psych = x_psych_prod.mean().to_numpy()
    std_psych = x_psych_prod.std().replace(0, 1).fillna(1).to_numpy()
    z_psych_prod = np.nan_to_num((x_psych_prod.to_numpy(float) - mean_psych) / std_psych)
    
    ridge_model = Ridge(alpha=1000.0, fit_intercept=False).fit(z_psych_prod, target_res, sample_weight=decay_weights)
    np.savez_compressed(
        PROD_DIR / 'stable_context_ridge.npz',
        columns=np.array(x_psych_prod.columns),
        mean=mean_psych,
        std=std_psych,
        coef=ridge_model.coef_,
        scale=np.float32(0.25 * 0.75)
    )
    print("  • Saved stable_context_profile.pkl & stable_context_ridge.npz")
    
    # 3. Fit Production Platoon CatBoost Expert
    print("\n[Step 3] Fitting Production Platoon CatBoost Expert...")
    px_prod = build_context_pressure_features(train_recent).reindex(columns=PLATOON_COLS)
    platoon_model = CatBoostRegressor(
        iterations=180, depth=3, learning_rate=0.02, loss_function='RMSE',
        l2_leaf_reg=100, random_strength=0.2, bootstrap_type='Bernoulli',
        subsample=0.8, random_seed=250013, thread_count=6, allow_writing_files=False, verbose=False
    )
    platoon_model.fit(px_prod, target_res, sample_weight=decay_weights)
    platoon_model.save_model(str(PROD_DIR / 'stable_platoon.cbm'))
    print("  • Saved stable_platoon.cbm")
    
    # 4. Build Production Prior Type Lookup & Transition Gate
    print("\n[Step 4] Fitting Production League Transition Gate...")
    prior_lookup_prod = build_prior_type_table(raw, max_season=2024)
    pickle.dump(prior_lookup_prod, open(PROD_DIR / 'prior_type.pkl', 'wb'))
    
    tx_prod = transition_features(train_recent, p_base_recent, PROD_DIR / 'prior_type.pkl')
    trans_model = CatBoostRegressor(
        iterations=250, depth=4, learning_rate=0.025, loss_function="RMSE",
        l2_leaf_reg=100, random_strength=0.2, bootstrap_type="Bernoulli",
        subsample=0.8, random_seed=918025, thread_count=6, allow_writing_files=False, verbose=False
    )
    trans_model.fit(tx_prod, target_res, sample_weight=decay_weights, cat_features=TRANS_CAT)
    trans_model.save_model(str(PROD_DIR / 'transition_gate.cbm'))
    print("  • Saved prior_type.pkl & transition_gate.cbm")
    
    print("\n" + "=" * 75)
    print("🏆 ALL PRODUCTION ASSETS SUCCESSFULLY TRAINED & SAVED!")
    print("=" * 75)

if __name__ == '__main__':
    main()
