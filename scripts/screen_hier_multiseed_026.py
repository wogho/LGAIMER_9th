#!/usr/bin/env python3
"""EXP-026-HIER-MULTISEED: Pure Hierarchical Dynamic Base (96 features) with 6-Seed Ensemble.

Preregistered protocol:
1. 96 features = 81 Base/Trackman + 15 Hierarchical Dynamic Shrinkage & Context features.
2. No auxiliary risk classifier outputs (eliminates 2023 calibration shift).
3. Regime Partition:
   - Baseline COMBO: 96 features, 300 trees, lr=0.05, l2=5 (6 seeds)
   - F-Regime Model: 96 features, excluding 2020, 300 trees, lr=0.05, l2=10 (6 seeds, 0.75 scale)
   - R-Regime Model: 96 features, depth 7, 600 trees, lr=0.03, l2=20 (6 seeds)
4. Evaluate on 2022, 2023, 2024 temporal walk-forward.
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
from src.hierarchical_base_features import (
    HIERARCHICAL_CAT_COLUMNS,
    HIERARCHICAL_FEATURE_NAMES,
    compute_hierarchical_base_features,
)
from src.target_aggregates import build_pitcher_count_state_target_history

EXPERIMENT_ID = "EXP-026-HIER-MULTISEED"
OUT_DIR = ROOT / "model" / EXPERIMENT_ID
BASELINE_OOF = ROOT / "model" / "COMBO-RESID3-OOF-007" / "oof_predictions.csv"
SEEDS = [42, 7, 2024, 99, 1, 123]
BASELINE_WEIGHT = 0.25
CANDIDATE_WEIGHT = 0.75
F_REGIME_SCALE = 0.75


def metric(pred: np.ndarray, target: np.ndarray) -> float:
    return float(1e5 * np.corrcoef(pred, target)[0, 1] ** 2)


def brier(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def main() -> None:
    start_time = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[{EXPERIMENT_ID}] Starting Pure Hierarchical 96-Feat 6-Seed Walk-Forward Screening...")

    raw = pd.read_csv(ROOT / "data" / "train.csv", encoding="utf-8-sig")
    trackman, _ = load_tm()

    results = []
    for valid_season in (2022, 2023, 2024):
        t0 = time.time()
        print(f"\n========================================================")
        print(f"--- Processing Valid Season {valid_season} ---")
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

        feature_cols_81 = BASE_COLS + TM_COLS
        xt = attach(xt, trackman, valid_season)[feature_cols_81]
        xv = attach(xv, trackman, valid_season)[feature_cols_81]

        # Attach 15 Hierarchical Base & Context Features
        print(f"  Computing Hierarchical Base Features (Repo 4)...")
        train_hier = compute_hierarchical_base_features(train_state)
        valid_hier = compute_hierarchical_base_features(valid_state)

        xt = pd.concat([xt.reset_index(drop=True), train_hier.reset_index(drop=True)], axis=1)
        xv = pd.concat([xv.reset_index(drop=True), valid_hier.reset_index(drop=True)], axis=1)

        final_feature_cols = feature_cols_81 + HIERARCHICAL_FEATURE_NAMES
        xt_prep, cats = prep(xt[final_feature_cols])
        xv_prep, _ = prep(xv[final_feature_cols])
        cats = list(set(cats + HIERARCHICAL_CAT_COLUMNS))

        y_train = train_df.control_success.to_numpy(np.int8)
        y_valid = valid_df.control_success.to_numpy(np.int8)

        mask_f_train = train_df.game_type.astype(str).eq("F").to_numpy() & train_df.season.ne(2020).to_numpy()
        mask_f_valid = valid_df.game_type.astype(str).eq("F").to_numpy()

        mask_r_train = train_df.game_type.astype(str).eq("R").to_numpy()
        mask_r_valid = valid_df.game_type.astype(str).eq("R").to_numpy()

        print(f"  Training 6-Seed Ensembles on {len(final_feature_cols)} features...")

        # 1. Baseline COMBO (6 seeds)
        print(f"    [Model 1] Baseline COMBO (6 seeds)...")
        base_preds = []
        for s in SEEDS:
            m = cb.CatBoostClassifier(
                iterations=300, learning_rate=0.05, depth=6, l2_leaf_reg=5,
                loss_function="Logloss", thread_count=3, random_seed=s,
                allow_writing_files=False, verbose=False
            )
            m.fit(cb.Pool(xt_prep, label=y_train, cat_features=cats, feature_names=final_feature_cols))
            base_preds.append(m.predict_proba(cb.Pool(xv_prep, cat_features=cats, feature_names=final_feature_cols))[:, 1])
        p_base_mean = np.mean(base_preds, axis=0)

        # 2. F-Regime Model (6 seeds)
        print(f"    [Model 2] F-Regime Model (6 seeds)...")
        f_preds = []
        for s in SEEDS:
            m = cb.CatBoostClassifier(
                iterations=300, learning_rate=0.05, depth=6, l2_leaf_reg=10,
                loss_function="Logloss", thread_count=3, random_seed=s,
                allow_writing_files=False, verbose=False
            )
            m.fit(cb.Pool(xt_prep.loc[mask_f_train], label=y_train[mask_f_train], cat_features=cats, feature_names=final_feature_cols))
            f_preds.append(m.predict_proba(cb.Pool(xv_prep.loc[mask_f_valid], cat_features=cats, feature_names=final_feature_cols))[:, 1])
        p_f_mean = np.mean(f_preds, axis=0) if mask_f_valid.any() else np.array([])

        # 3. R-Regime Model (depth 7, 600 trees, 6 seeds)
        print(f"    [Model 3] R-Regime Capacity Model (depth 7, 600 trees, 6 seeds)...")
        r_preds = []
        for s in SEEDS:
            m = cb.CatBoostClassifier(
                iterations=600, learning_rate=0.03, depth=7, l2_leaf_reg=20,
                loss_function="Logloss", thread_count=3, random_seed=s,
                allow_writing_files=False, verbose=False
            )
            m.fit(cb.Pool(xt_prep.loc[mask_r_train], label=y_train[mask_r_train], cat_features=cats, feature_names=final_feature_cols))
            r_preds.append(m.predict_proba(cb.Pool(xv_prep.loc[mask_r_valid], cat_features=cats, feature_names=final_feature_cols))[:, 1])
        p_r_mean = np.mean(r_preds, axis=0) if mask_r_valid.any() else np.array([])

        # Assemble with 0.75 F-regime relaxation
        p_split_mean = np.zeros(len(valid_df), dtype=float)
        if mask_f_valid.any():
            p_f_relaxed = p_base_mean[mask_f_valid] + F_REGIME_SCALE * (p_f_mean - p_base_mean[mask_f_valid])
            p_split_mean[mask_f_valid] = p_f_relaxed
        if mask_r_valid.any():
            p_split_mean[mask_r_valid] = p_r_mean

        p_final_blend = BASELINE_WEIGHT * p_base_mean + CANDIDATE_WEIGHT * p_split_mean

        cand_metric = metric(p_final_blend, y_valid)
        cand_brier = brier(p_final_blend, y_valid)

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
            "feature_count": len(final_feature_cols),
            "seeds_count": len(SEEDS),
            "f_regime_scale": F_REGIME_SCALE,
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
        print(f"\n  >>> Season {valid_season} Result (Pure Hierarchical 96-Feat 6-Seed Stack):")
        print(f"      Baseline Metric : {base_metric:.4f} | Cand Metric: {cand_metric:.4f} ({rel_imp*100:+.2f}%)")
        print(f"      Baseline Brier  : {base_brier:.8f} | Cand Brier : {cand_brier:.8f} (Delta: {delta_brier:+.8f})")

        del xt, xv, xt_prep, xv_prep, base_preds, f_preds, r_preds
        gc.collect()

    all_passed_2pct = all(r["relative_improvement"] >= 0.02 for r in results)
    all_positive = all(r["relative_improvement"] > 0 for r in results)

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "official_train_only": True,
        "test_used": False,
        "external_data_used": False,
        "seeds": SEEDS,
        "features_total": 96,
        "f_regime_scale": F_REGIME_SCALE,
        "results_by_season": results,
        "all_positive_sign": all_positive,
        "all_passed_2pct_gate": all_passed_2pct,
        "total_elapsed_seconds": time.time() - start_time,
        "status": "PASS_SCREEN" if all_positive else "FAIL_SCREEN",
    }

    report_path = OUT_DIR / "screen_report.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n" + "=" * 60)
    print(f"[{EXPERIMENT_ID}] Pure Hierarchical 6-Seed Screening Completed!")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
