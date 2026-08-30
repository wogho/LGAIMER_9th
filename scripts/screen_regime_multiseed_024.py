#!/usr/bin/env python3
"""EXP-024-REGIME-MULTISEED-6: 6-Seed Multi-Seed Variance Reduction on Regime-RCapacity Model.

Preregistered protocol:
1. Architecture: REGIME-RCAPACITY-FULL-019 (81 features, F-regime ex2020, R-regime depth 7, 600 trees).
2. Seeds: [42, 7, 2024, 99, 1, 123] (6 fixed random seeds).
3. For each sub-model (Baseline, F-regime, R-regime), train 6 seeds and average predictions.
4. Evaluate 0.25 : 0.75 blend on 2022, 2023, 2024 temporal walk-forward.
5. Strict row-independence, official train-only.
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import catboost as cb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_combo_full_candidate_002 import COLS as BASE_COLS, extras, prep
from scripts.screen_trackman_context_003 import NEW as TM_COLS, attach, load_tm
from src.asof_state_features import add_state_for_cutoff, add_state_walkforward
from src.target_aggregates import build_pitcher_count_state_target_history

EXPERIMENT_ID = "EXP-024-REGIME-MULTISEED-6"
OUT_DIR = ROOT / "model" / EXPERIMENT_ID
BASELINE_OOF = ROOT / "model" / "COMBO-RESID3-OOF-007" / "oof_predictions.csv"
FEATURE_COLUMNS_81 = BASE_COLS + TM_COLS
SEEDS = [42, 7, 2024, 99, 1, 123]
BASELINE_WEIGHT = 0.25
CANDIDATE_WEIGHT = 0.75


def metric(pred: np.ndarray, target: np.ndarray) -> float:
    return float(1e5 * np.corrcoef(pred, target)[0, 1] ** 2)


def brier(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def main() -> None:
    start_time = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[{EXPERIMENT_ID}] Starting 6-Seed Multi-Seed Walk-Forward Screening (Seeds: {SEEDS})...")

    raw = pd.read_csv(ROOT / "data" / "train.csv", encoding="utf-8-sig")
    trackman, _ = load_tm()

    results = []
    for valid_season in (2022, 2023, 2024):
        t0 = time.time()
        print(f"\n--- Processing Valid Season {valid_season} ---")
        train_df = raw.loc[raw.season < valid_season].copy()
        valid_df = raw.loc[raw.season == valid_season].copy()

        train_state = add_state_walkforward(
            train_df.drop(columns=["row_id", "control_success"]), valid_season
        )
        valid_state = add_state_for_cutoff(
            valid_df.drop(columns=["row_id", "control_success"]),
            train_df.drop(columns=["row_id", "control_success"])
        )

        train_hist, valid_hist, _, _ = build_pitcher_count_state_target_history(
            train_df, valid_df.assign(control_success=0), smoothing=100.0
        )

        xt = pd.concat([extras(train_state.reset_index(drop=True)), train_hist.reset_index(drop=True)], axis=1)
        xv = pd.concat([extras(valid_state.reset_index(drop=True)), valid_hist.reset_index(drop=True)], axis=1)

        xt = attach(xt, trackman, valid_season)[FEATURE_COLUMNS_81]
        xv = attach(xv, trackman, valid_season)[FEATURE_COLUMNS_81]

        xt, cats = prep(xt)
        xv, _ = prep(xv)

        y_train = train_df.control_success.to_numpy(np.int8)
        y_valid = valid_df.control_success.to_numpy(np.int8)

        mask_f_train = train_df.game_type.astype(str).eq("F").to_numpy() & train_df.season.ne(2020).to_numpy()
        mask_f_valid = valid_df.game_type.astype(str).eq("F").to_numpy()

        mask_r_train = train_df.game_type.astype(str).eq("R").to_numpy()
        mask_r_valid = valid_df.game_type.astype(str).eq("R").to_numpy()

        # 1. 6-Seed Baseline COMBO Predictions
        print(f"  [Step 1] Training 6 Seeds Baseline COMBO ({len(xt):,} rows)...")
        base_preds = []
        for seed in SEEDS:
            m = cb.CatBoostClassifier(
                iterations=300, learning_rate=0.05, depth=6, l2_leaf_reg=5,
                loss_function="Logloss", thread_count=3, random_seed=seed,
                allow_writing_files=False, verbose=False
            )
            m.fit(cb.Pool(xt, label=y_train, cat_features=cats, feature_names=FEATURE_COLUMNS_81))
            base_preds.append(m.predict_proba(cb.Pool(xv, cat_features=cats, feature_names=FEATURE_COLUMNS_81))[:, 1])
        p_base_mean = np.mean(base_preds, axis=0)

        # 2. 6-Seed F-Regime Predictions
        print(f"  [Step 2] Training 6 Seeds F-Regime ({int(mask_f_train.sum()):,} rows)...")
        f_preds = []
        for seed in SEEDS:
            m = cb.CatBoostClassifier(
                iterations=300, learning_rate=0.05, depth=6, l2_leaf_reg=10,
                loss_function="Logloss", thread_count=3, random_seed=seed,
                allow_writing_files=False, verbose=False
            )
            m.fit(cb.Pool(xt.loc[mask_f_train], label=y_train[mask_f_train], cat_features=cats, feature_names=FEATURE_COLUMNS_81))
            f_preds.append(m.predict_proba(cb.Pool(xv.loc[mask_f_valid], cat_features=cats, feature_names=FEATURE_COLUMNS_81))[:, 1])
        p_f_mean = np.mean(f_preds, axis=0) if mask_f_valid.any() else np.array([])

        # 3. 6-Seed R-Regime Predictions (Depth 7, 600 trees)
        print(f"  [Step 3] Training 6 Seeds R-Regime ({int(mask_r_train.sum()):,} rows, depth 7, 600 trees)...")
        r_preds = []
        for seed in SEEDS:
            m = cb.CatBoostClassifier(
                iterations=600, learning_rate=0.03, depth=7, l2_leaf_reg=20,
                loss_function="Logloss", thread_count=3, random_seed=seed,
                allow_writing_files=False, verbose=False
            )
            m.fit(cb.Pool(xt.loc[mask_r_train], label=y_train[mask_r_train], cat_features=cats, feature_names=FEATURE_COLUMNS_81))
            r_preds.append(m.predict_proba(cb.Pool(xv.loc[mask_r_valid], cat_features=cats, feature_names=FEATURE_COLUMNS_81))[:, 1])
        p_r_mean = np.mean(r_preds, axis=0) if mask_r_valid.any() else np.array([])

        # Split prediction assembly
        p_split_mean = np.zeros(len(valid_df), dtype=float)
        if mask_f_valid.any():
            p_split_mean[mask_f_valid] = p_f_mean
        if mask_r_valid.any():
            p_split_mean[mask_r_valid] = p_r_mean

        # 0.25 : 0.75 Blend of 6-Seed ensembles
        p_blend_6seed = BASELINE_WEIGHT * p_base_mean + CANDIDATE_WEIGHT * p_split_mean

        cand_metric = metric(p_blend_6seed, y_valid)
        cand_brier = brier(p_blend_6seed, y_valid)

        # Baseline comparison (81-feat COMBO baseline)
        base_df = pd.read_csv(BASELINE_OOF, usecols=["row_id", "season", "target", "pred"])
        base_valid = base_df.loc[base_df.season == valid_season]
        base_metric = metric(base_valid.pred.to_numpy(), y_valid)
        base_brier = brier(base_valid.pred.to_numpy(), y_valid)

        rel_imp = (cand_metric - base_metric) / base_metric
        delta_brier = cand_brier - base_brier

        res = {
            "season": valid_season,
            "train_rows": len(train_df),
            "valid_rows": len(valid_df),
            "seeds_count": len(SEEDS),
            "baseline_metric": base_metric,
            "candidate_metric": cand_metric,
            "relative_improvement": rel_imp,
            "relative_improvement_pct": rel_imp * 100.0,
            "baseline_brier": base_brier,
            "candidate_brier": cand_brier,
            "delta_brier": delta_brier,
            "elapsed_seconds": time.time() - t0,
        }
        results.append(res)
        print(f"  Season {valid_season} Result (6-Seed Ensemble):")
        print(f"    Baseline Metric : {base_metric:.4f} | Cand Metric: {cand_metric:.4f} ({rel_imp*100:+.2f}%)")
        print(f"    Baseline Brier  : {base_brier:.8f} | Cand Brier : {cand_brier:.8f} (Delta: {delta_brier:+.8f})")

        del xt, xv, base_preds, f_preds, r_preds
        gc.collect()

    all_passed_2pct = all(r["relative_improvement"] >= 0.02 for r in results)
    all_positive = all(r["relative_improvement"] > 0 for r in results)

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "official_train_only": True,
        "test_used": False,
        "external_data_used": False,
        "seeds": SEEDS,
        "results_by_season": results,
        "all_positive_sign": all_positive,
        "all_passed_2pct_gate": all_passed_2pct,
        "total_elapsed_seconds": time.time() - start_time,
        "status": "PASS_SCREEN" if all_positive else "FAIL_SCREEN",
    }

    report_path = OUT_DIR / "screen_report.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n" + "=" * 60)
    print(f"[{EXPERIMENT_ID}] 6-Seed Multi-Seed Screening Completed!")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
