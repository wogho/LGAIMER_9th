#!/usr/bin/env python3
"""EXP-028-RESIDUAL-FAST: Ultra-fast 1-Seed Temporal Walk-Forward Screening (~5 mins).

Preregistered protocol:
1. Target: residual = control_success - hierarchical_base.
2. Feature set: 96 features (81 Base/TM + 15 Hierarchical Dynamic Shrinkage & Context).
3. 1-Seed (seed=42), all-core threading (thread_count=-1).
4. Sub-models:
   - Baseline COMBO Residual Regressor (300 trees, depth 6, lr 0.06, l2 10)
   - F-Regime Residual Regressor (excluding 2020, 300 trees, 0.75 scale)
   - R-Regime Capacity Residual Regressor (depth 6, 400 trees, lr 0.05, l2 15)
5. 0.25 : 0.75 blend on 2022, 2023, 2024 temporal walk-forward.
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

EXPERIMENT_ID = "EXP-028-RESIDUAL-FAST"
OUT_DIR = ROOT / "model" / EXPERIMENT_ID
BASELINE_OOF = ROOT / "model" / "COMBO-RESID3-OOF-007" / "oof_predictions.csv"
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
    print(f"[{EXPERIMENT_ID}] Starting Ultra-Fast 1-Seed Walk-Forward Screening...")

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

        # Attach Hierarchical Base & Features
        train_hier = compute_hierarchical_base_features(train_state)
        valid_hier = compute_hierarchical_base_features(valid_state)

        base_t = train_hier["hierarchical_success_base"].to_numpy(dtype=float)
        base_v = valid_hier["hierarchical_success_base"].to_numpy(dtype=float)

        xt = pd.concat([xt.reset_index(drop=True), train_hier.reset_index(drop=True)], axis=1)
        xv = pd.concat([xv.reset_index(drop=True), valid_hier.reset_index(drop=True)], axis=1)

        final_feature_cols = feature_cols_81 + HIERARCHICAL_FEATURE_NAMES
        xt_prep, cats = prep(xt[final_feature_cols])
        xv_prep, _ = prep(xv[final_feature_cols])
        cats = list(set(cats + HIERARCHICAL_CAT_COLUMNS))

        y_train = train_df.control_success.to_numpy(float)
        y_valid = valid_df.control_success.to_numpy(float)

        res_train = y_train - base_t

        mask_f_train = train_df.game_type.astype(str).eq("F").to_numpy() & train_df.season.ne(2020).to_numpy()
        mask_f_valid = valid_df.game_type.astype(str).eq("F").to_numpy()

        mask_r_train = train_df.game_type.astype(str).eq("R").to_numpy()
        mask_r_valid = valid_df.game_type.astype(str).eq("R").to_numpy()

        # 1. Baseline COMBO Residual Regressor (1 seed, multithreaded)
        print(f"  [Model 1] Baseline COMBO Residual Regressor ({len(xt_prep):,} rows)...")
        m_base = cb.CatBoostRegressor(
            iterations=300, learning_rate=0.06, depth=6, l2_leaf_reg=10,
            loss_function="RMSE", thread_count=-1, random_seed=42,
            allow_writing_files=False, verbose=False
        )
        m_base.fit(cb.Pool(xt_prep, label=res_train, cat_features=cats, feature_names=final_feature_cols))
        res_base = m_base.predict(cb.Pool(xv_prep, cat_features=cats, feature_names=final_feature_cols))

        # 2. F-Regime Residual Regressor
        print(f"  [Model 2] F-Regime Residual Regressor ({int(mask_f_train.sum()):,} rows)...")
        m_f = cb.CatBoostRegressor(
            iterations=300, learning_rate=0.06, depth=6, l2_leaf_reg=10,
            loss_function="RMSE", thread_count=-1, random_seed=42,
            allow_writing_files=False, verbose=False
        )
        m_f.fit(cb.Pool(xt_prep.loc[mask_f_train], label=res_train[mask_f_train], cat_features=cats, feature_names=final_feature_cols))
        res_f = m_f.predict(cb.Pool(xv_prep.loc[mask_f_valid], cat_features=cats, feature_names=final_feature_cols)) if mask_f_valid.any() else np.array([])

        # 3. R-Regime Residual Regressor
        print(f"  [Model 3] R-Regime Residual Regressor ({int(mask_r_train.sum()):,} rows)...")
        m_r = cb.CatBoostRegressor(
            iterations=400, learning_rate=0.05, depth=6, l2_leaf_reg=15,
            loss_function="RMSE", thread_count=-1, random_seed=42,
            allow_writing_files=False, verbose=False
        )
        m_r.fit(cb.Pool(xt_prep.loc[mask_r_train], label=res_train[mask_r_train], cat_features=cats, feature_names=final_feature_cols))
        res_r = m_r.predict(cb.Pool(xv_prep.loc[mask_r_valid], cat_features=cats, feature_names=final_feature_cols)) if mask_r_valid.any() else np.array([])

        # Assemble split residual with 0.75 F-regime relaxation
        res_split = np.zeros(len(valid_df), dtype=float)
        if mask_f_valid.any():
            res_f_relaxed = res_base[mask_f_valid] + F_REGIME_SCALE * (res_f - res_base[mask_f_valid])
            res_split[mask_f_valid] = res_f_relaxed
        if mask_r_valid.any():
            res_split[mask_r_valid] = res_r

        res_final = BASELINE_WEIGHT * res_base + CANDIDATE_WEIGHT * res_split
        p_final = np.clip(base_v + res_final, 1e-5, 1.0 - 1e-5)

        cand_metric = metric(p_final, y_valid)
        cand_brier = brier(p_final, y_valid)

        base_df = pd.read_csv(BASELINE_OOF, usecols=["row_id", "season", "target", "pred"])
        base_valid = base_df.loc[base_df.season == valid_season]
        base_metric = metric(base_valid.pred.to_numpy(), y_valid)
        base_brier = brier(base_valid.pred.to_numpy(), y_valid)

        rel_imp = (cand_metric - base_metric) / base_metric
        delta_brier = cand_brier - base_brier
        el = time.time() - t0

        res = {
            "season": valid_season,
            "train_rows": len(train_df),
            "valid_rows": len(valid_df),
            "feature_count": len(final_feature_cols),
            "baseline_metric": base_metric,
            "candidate_metric": cand_metric,
            "relative_improvement": rel_imp,
            "relative_improvement_pct": rel_imp * 100.0,
            "baseline_brier": base_brier,
            "candidate_brier": cand_brier,
            "delta_brier": delta_brier,
            "elapsed_seconds": el,
        }
        results.append(res)
        print(f"\n  >>> Season {valid_season} Result ({el:.1f}s):")
        print(f"      Baseline Metric : {base_metric:.4f} | Cand Metric: {cand_metric:.4f} ({rel_imp*100:+.2f}%)")
        print(f"      Baseline Brier  : {base_brier:.8f} | Cand Brier : {cand_brier:.8f} (Delta: {delta_brier:+.8f})")

        del xt, xv, xt_prep, xv_prep, m_base, m_f, m_r
        gc.collect()

    all_positive = all(r["relative_improvement"] > 0 for r in results)

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "official_train_only": True,
        "test_used": False,
        "external_data_used": False,
        "features_total": 96,
        "f_regime_scale": F_REGIME_SCALE,
        "results_by_season": results,
        "all_positive_sign": all_positive,
        "total_elapsed_seconds": time.time() - start_time,
        "status": "PASS_SCREEN" if all_positive else "FAIL_SCREEN",
    }

    report_path = OUT_DIR / "screen_report.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n" + "=" * 60)
    print(f"[{EXPERIMENT_ID}] Ultra-Fast Screening Completed in {time.time() - start_time:.1f}s!")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
