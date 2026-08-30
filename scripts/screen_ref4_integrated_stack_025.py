#!/usr/bin/env python3
"""EXP-025-REF4-INTEGRATED-STACK: 4th Reference Repo Architecture Integration.

Integrates 4 key breakthroughs from 1126.45 LB Champion (Repo 4):
1. Hierarchical Dynamic Shrinkage Base & Domain Contexts (15 features).
2. 3-Subtype Failure Risks (middle, ball, reverse) trained from train-only recovered labels.
3. F-Regime 0.75 relaxation + R-Regime capacity (depth 7, 600 trees).
4. Multi-Seed 6-Seed Variance Reduction (seeds: 42, 7, 2024, 99, 1, 123).

Strict row-independence and 2022-2024 temporal walk-forward validation.
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

EXPERIMENT_ID = "EXP-025-REF4-INTEGRATED-STACK"
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
    print(f"[{EXPERIMENT_ID}] Starting 4th Repo Architecture Integration Screening...")

    raw = pd.read_csv(ROOT / "data" / "train.csv", encoding="utf-8-sig")
    recovered = pd.read_csv(ROOT / "model" / "REF-AUX-LABEL-001" / "recovered_labels.csv.gz")
    raw = raw.merge(recovered[["row_id", "middle", "ball", "reverse"]], on="row_id", how="left")

    trackman, _ = load_tm()

    results = []
    for valid_season in (2022, 2023, 2024):
        t0 = time.time()
        print(f"\n========================================================")
        print(f"--- Processing Valid Season {valid_season} ---")
        train_df = raw.loc[raw.season < valid_season].copy()
        valid_df = raw.loc[raw.season == valid_season].copy()

        # 1. Base AS-OF State features
        train_state = add_state_walkforward(
            train_df.drop(columns=["row_id", "control_success", "middle", "ball", "reverse"]),
            valid_season
        )
        valid_state = add_state_for_cutoff(
            valid_df.drop(columns=["row_id", "control_success", "middle", "ball", "reverse"]),
            train_df.drop(columns=["row_id", "control_success", "middle", "ball", "reverse"])
        )

        # 2. Target aggregate features
        train_hist, valid_hist, _, _ = build_pitcher_count_state_target_history(
            train_df, valid_df.assign(control_success=0), smoothing=100.0
        )

        xt = pd.concat([extras(train_state.reset_index(drop=True)), train_hist.reset_index(drop=True)], axis=1)
        xv = pd.concat([extras(valid_state.reset_index(drop=True)), valid_hist.reset_index(drop=True)], axis=1)

        # 3. Trackman features (81 Base)
        feature_cols_81 = BASE_COLS + TM_COLS
        xt = attach(xt, trackman, valid_season)[feature_cols_81]
        xv = attach(xv, trackman, valid_season)[feature_cols_81]

        # 4. Hierarchical Base & Domain features (15 features)
        print(f"  Computing Hierarchical Base Features (Repo 4)...")
        train_hier = compute_hierarchical_base_features(train_state)
        valid_hier = compute_hierarchical_base_features(valid_state)

        xt = pd.concat([xt.reset_index(drop=True), train_hier.reset_index(drop=True)], axis=1)
        xv = pd.concat([xv.reset_index(drop=True), valid_hier.reset_index(drop=True)], axis=1)

        # 5. 3-Subtype Failure Risk Prediction (middle, ball, reverse)
        print(f"  Training 3-Subtype Failure Risk Classifiers (middle, ball, reverse)...")
        valid_middle = train_df.middle.notna() & train_df.middle.isin([0, 1])
        valid_ball = train_df.ball.notna() & train_df.ball.isin([0, 1])
        valid_reverse = train_df.reverse.notna() & train_df.reverse.isin([0, 1])

        xt_base_prep, cats_base = prep(xt[feature_cols_81 + HIERARCHICAL_FEATURE_NAMES])
        xv_base_prep, _ = prep(xv[feature_cols_81 + HIERARCHICAL_FEATURE_NAMES])
        cats_base = list(set(cats_base + HIERARCHICAL_CAT_COLUMNS))

        risk_cols = ["p_risk_middle", "p_risk_ball", "p_risk_reverse"]
        risk_train = np.zeros((len(train_df), 3), dtype=np.float32)
        risk_valid = np.zeros((len(valid_df), 3), dtype=np.float32)

        for r_idx, (r_name, r_mask) in enumerate([
            ("middle", valid_middle.to_numpy()),
            ("ball", valid_ball.to_numpy()),
            ("reverse", valid_reverse.to_numpy())
        ]):
            m_risk = cb.CatBoostClassifier(
                iterations=200, learning_rate=0.06, depth=5, l2_leaf_reg=5,
                loss_function="Logloss", thread_count=3, random_seed=42,
                allow_writing_files=False, verbose=False
            )
            y_r = train_df[r_name].to_numpy(np.int8)[r_mask]
            m_risk.fit(cb.Pool(xt_base_prep.loc[r_mask], label=y_r, cat_features=cats_base, feature_names=list(xt_base_prep.columns)))
            risk_train[:, r_idx] = m_risk.predict_proba(cb.Pool(xt_base_prep, cat_features=cats_base, feature_names=list(xt_base_prep.columns)))[:, 1]
            risk_valid[:, r_idx] = m_risk.predict_proba(cb.Pool(xv_base_prep, cat_features=cats_base, feature_names=list(xv_base_prep.columns)))[:, 1]

        for r_idx, r_col in enumerate(risk_cols):
            xt[r_col] = risk_train[:, r_idx]
            xv[r_col] = risk_valid[:, r_idx]

        final_feature_cols = feature_cols_81 + HIERARCHICAL_FEATURE_NAMES + risk_cols
        all_cats = list(set(cats_base))

        xt_final, all_cats = prep(xt[final_feature_cols])
        xv_final, _ = prep(xv[final_feature_cols])

        y_train = train_df.control_success.to_numpy(np.int8)
        y_valid = valid_df.control_success.to_numpy(np.int8)

        mask_f_train = train_df.game_type.astype(str).eq("F").to_numpy() & train_df.season.ne(2020).to_numpy()
        mask_f_valid = valid_df.game_type.astype(str).eq("F").to_numpy()

        mask_r_train = train_df.game_type.astype(str).eq("R").to_numpy()
        mask_r_valid = valid_df.game_type.astype(str).eq("R").to_numpy()

        # 6. Multi-Seed Training (6 Seeds: [42, 7, 2024, 99, 1, 123])
        print(f"  Training 6-Seed Ensembles on {len(final_feature_cols)} features...")
        
        # 6.1 Baseline COMBO (6 seeds)
        print(f"    [Model 1] Baseline COMBO (6 seeds)...")
        base_preds = []
        for s in SEEDS:
            m = cb.CatBoostClassifier(
                iterations=300, learning_rate=0.05, depth=6, l2_leaf_reg=5,
                loss_function="Logloss", thread_count=3, random_seed=s,
                allow_writing_files=False, verbose=False
            )
            m.fit(cb.Pool(xt_final, label=y_train, cat_features=all_cats, feature_names=final_feature_cols))
            base_preds.append(m.predict_proba(cb.Pool(xv_final, cat_features=all_cats, feature_names=final_feature_cols))[:, 1])
        p_base_mean = np.mean(base_preds, axis=0)

        # 6.2 F-Regime Model with 0.75 relaxation (6 seeds)
        print(f"    [Model 2] F-Regime Model (6 seeds)...")
        f_preds = []
        for s in SEEDS:
            m = cb.CatBoostClassifier(
                iterations=300, learning_rate=0.05, depth=6, l2_leaf_reg=10,
                loss_function="Logloss", thread_count=3, random_seed=s,
                allow_writing_files=False, verbose=False
            )
            m.fit(cb.Pool(xt_final.loc[mask_f_train], label=y_train[mask_f_train], cat_features=all_cats, feature_names=final_feature_cols))
            f_preds.append(m.predict_proba(cb.Pool(xv_final.loc[mask_f_valid], cat_features=all_cats, feature_names=final_feature_cols))[:, 1])
        p_f_mean = np.mean(f_preds, axis=0) if mask_f_valid.any() else np.array([])

        # 6.3 R-Regime Model (depth 7, 600 trees, 6 seeds)
        print(f"    [Model 3] R-Regime Capacity Model (depth 7, 600 trees, 6 seeds)...")
        r_preds = []
        for s in SEEDS:
            m = cb.CatBoostClassifier(
                iterations=600, learning_rate=0.03, depth=7, l2_leaf_reg=20,
                loss_function="Logloss", thread_count=3, random_seed=s,
                allow_writing_files=False, verbose=False
            )
            m.fit(cb.Pool(xt_final.loc[mask_r_train], label=y_train[mask_r_train], cat_features=all_cats, feature_names=final_feature_cols))
            r_preds.append(m.predict_proba(cb.Pool(xv_final.loc[mask_r_valid], cat_features=all_cats, feature_names=final_feature_cols))[:, 1])
        p_r_mean = np.mean(r_preds, axis=0) if mask_r_valid.any() else np.array([])

        # Assemble split predictions with F-regime 0.75 relaxation
        p_split_mean = np.zeros(len(valid_df), dtype=float)
        if mask_f_valid.any():
            # 0.75 F-regime relaxation: pull 25% back toward base prediction
            p_f_relaxed = p_base_mean[mask_f_valid] + F_REGIME_SCALE * (p_f_mean - p_base_mean[mask_f_valid])
            p_split_mean[mask_f_valid] = p_f_relaxed
        if mask_r_valid.any():
            p_split_mean[mask_r_valid] = p_r_mean

        # Final 0.25 : 0.75 Blend
        p_final_blend = BASELINE_WEIGHT * p_base_mean + CANDIDATE_WEIGHT * p_split_mean

        cand_metric = metric(p_final_blend, y_valid)
        cand_brier = brier(p_final_blend, y_valid)

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
        print(f"\n  >>> Season {valid_season} Final Result (Integrated 4th Repo Stack):")
        print(f"      Baseline Metric : {base_metric:.4f} | Cand Metric: {cand_metric:.4f} ({rel_imp*100:+.2f}%)")
        print(f"      Baseline Brier  : {base_brier:.8f} | Cand Brier : {cand_brier:.8f} (Delta: {delta_brier:+.8f})")

        del xt, xv, xt_final, xv_final, base_preds, f_preds, r_preds
        gc.collect()

    all_passed_2pct = all(r["relative_improvement"] >= 0.02 for r in results)
    all_positive = all(r["relative_improvement"] > 0 for r in results)

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "official_train_only": True,
        "test_used": False,
        "external_data_used": False,
        "seeds": SEEDS,
        "features_total": 99,
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
    print(f"[{EXPERIMENT_ID}] Integrated 4th Repo Screening Completed!")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
