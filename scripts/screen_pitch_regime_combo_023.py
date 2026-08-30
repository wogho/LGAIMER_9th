#!/usr/bin/env python3
"""EXP-023-PITCH-REGIME-COMBO: Combine 84 Pitch Intent Features with Regime Capacity Split Model.

Preregistered protocol:
1. 84 features (81 Base/Trackman + 3 Pitch Intent Probabilities p_fastball, p_breaking, p_offspeed).
2. Regime Partition:
   - F-regime: game_type=='F', excluding season 2020, iterations=300, depth=6, lr=0.05, l2=10
   - R-regime: game_type=='R', iterations=600, depth=7, lr=0.03, l2=20
3. 0.25 Baseline COMBO + 0.75 Split(F/R) Blend.
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
from src.target_aggregates import build_pitcher_count_state_target_history

EXPERIMENT_ID = "EXP-023-PITCH-REGIME-COMBO"
OUT_DIR = ROOT / "model" / EXPERIMENT_ID
BASELINE_OOF = ROOT / "model" / "COMBO-RESID3-OOF-007" / "oof_predictions.csv"
FEATURE_COLUMNS_81 = BASE_COLS + TM_COLS
BASELINE_WEIGHT = 0.25
CANDIDATE_WEIGHT = 0.75


def metric(pred: np.ndarray, target: np.ndarray) -> float:
    return float(1e5 * np.corrcoef(pred, target)[0, 1] ** 2)


def brier(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def main() -> None:
    start_time = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[{EXPERIMENT_ID}] Starting 2022-2024 temporal walk-forward screening...")

    raw = pd.read_csv(ROOT / "data" / "train.csv", encoding="utf-8-sig")
    recovered = pd.read_csv(ROOT / "model" / "REF-AUX-LABEL-001" / "recovered_labels.csv.gz")
    raw = raw.merge(recovered[["row_id", "fastball", "breaking", "offspeed"]], on="row_id", how="left")

    trackman, _ = load_tm()

    results = []
    for valid_season in (2022, 2023, 2024):
        t0 = time.time()
        print(f"\n--- Processing Valid Season {valid_season} ---")
        train_df = raw.loc[raw.season < valid_season].copy()
        valid_df = raw.loc[raw.season == valid_season].copy()

        # 1. AS-OF State features
        train_state = add_state_walkforward(
            train_df.drop(columns=["row_id", "control_success", "fastball", "breaking", "offspeed"]),
            valid_season
        )
        valid_state = add_state_for_cutoff(
            valid_df.drop(columns=["row_id", "control_success", "fastball", "breaking", "offspeed"]),
            train_df.drop(columns=["row_id", "control_success", "fastball", "breaking", "offspeed"])
        )

        # 2. Target aggregate features
        train_hist, valid_hist, _, _ = build_pitcher_count_state_target_history(
            train_df, valid_df.assign(control_success=0), smoothing=100.0
        )

        xt = pd.concat([extras(train_state.reset_index(drop=True)), train_hist.reset_index(drop=True)], axis=1)
        xv = pd.concat([extras(valid_state.reset_index(drop=True)), valid_hist.reset_index(drop=True)], axis=1)

        # 3. Trackman features
        xt = attach(xt, trackman, valid_season)[FEATURE_COLUMNS_81]
        xv = attach(xv, trackman, valid_season)[FEATURE_COLUMNS_81]

        xt, cats = prep(xt)
        xv, _ = prep(xv)

        y_train = train_df.control_success.to_numpy(np.int8)
        y_valid = valid_df.control_success.to_numpy(np.int8)

        # -------------------------------------------------------------
        # Stage 1: Pitch Type Intention Multiclass Model
        # -------------------------------------------------------------
        has_mix = train_df[["fastball", "breaking", "offspeed"]].notna().all(axis=1)
        mix_train_mask = has_mix.to_numpy()
        y_mix_train = np.select(
            [train_df.fastball == 1, train_df.breaking == 1, train_df.offspeed == 1],
            [0, 1, 2],
            default=-1
        )[mix_train_mask]

        valid_mix_train = y_mix_train >= 0
        x_mix_train = xt.loc[mix_train_mask].loc[valid_mix_train]
        y_mix_train = y_mix_train[valid_mix_train]

        print(f"  [Stage 1] Training Pitch Type CatBoost on {len(x_mix_train):,} rows...")
        m_pitch = cb.CatBoostClassifier(
            iterations=250,
            learning_rate=0.06,
            depth=5,
            l2_leaf_reg=5,
            loss_function="MultiClass",
            thread_count=3,
            random_seed=42,
            allow_writing_files=False,
            verbose=False,
        )
        m_pitch.fit(cb.Pool(x_mix_train, label=y_mix_train, cat_features=cats, feature_names=FEATURE_COLUMNS_81))

        p_pitch_train = m_pitch.predict_proba(cb.Pool(xt, cat_features=cats, feature_names=FEATURE_COLUMNS_81))
        p_pitch_valid = m_pitch.predict_proba(cb.Pool(xv, cat_features=cats, feature_names=FEATURE_COLUMNS_81))

        xt_84 = xt.copy()
        xv_84 = xv.copy()
        for idx, col in enumerate(["p_fastball", "p_breaking", "p_offspeed"]):
            xt_84[col] = p_pitch_train[:, idx]
            xv_84[col] = p_pitch_valid[:, idx]

        features_84 = FEATURE_COLUMNS_81 + ["p_fastball", "p_breaking", "p_offspeed"]

        # -------------------------------------------------------------
        # Stage 2.1: Baseline COMBO Model on 84 Features
        # -------------------------------------------------------------
        print(f"  [Stage 2.1] Baseline COMBO (84 feats, 300 trees)...")
        m_base = cb.CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=5,
            loss_function="Logloss",
            thread_count=3,
            random_seed=42,
            allow_writing_files=False,
            verbose=False,
        )
        m_base.fit(cb.Pool(xt_84, label=y_train, cat_features=cats, feature_names=features_84))
        p_base_valid = m_base.predict_proba(cb.Pool(xv_84, cat_features=cats, feature_names=features_84))[:, 1]

        # -------------------------------------------------------------
        # Stage 2.2: F-Regime Model (2020 excluded, 84 feats)
        # -------------------------------------------------------------
        mask_f_train = train_df.game_type.astype(str).eq("F").to_numpy() & train_df.season.ne(2020).to_numpy()
        mask_f_valid = valid_df.game_type.astype(str).eq("F").to_numpy()
        print(f"  [Stage 2.2] F-Regime Model on {int(mask_f_train.sum()):,} rows...")
        m_f = cb.CatBoostClassifier(
            iterations=300,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=10,
            loss_function="Logloss",
            thread_count=3,
            random_seed=42,
            allow_writing_files=False,
            verbose=False,
        )
        m_f.fit(cb.Pool(xt_84.loc[mask_f_train], label=y_train[mask_f_train], cat_features=cats, feature_names=features_84))

        # -------------------------------------------------------------
        # Stage 2.3: R-Regime Capacity Model (depth 7, 600 trees, 84 feats)
        # -------------------------------------------------------------
        mask_r_train = train_df.game_type.astype(str).eq("R").to_numpy()
        mask_r_valid = valid_df.game_type.astype(str).eq("R").to_numpy()
        print(f"  [Stage 2.3] R-Regime Capacity Model on {int(mask_r_train.sum()):,} rows (depth 7, 600 trees)...")
        m_r = cb.CatBoostClassifier(
            iterations=600,
            learning_rate=0.03,
            depth=7,
            l2_leaf_reg=20,
            loss_function="Logloss",
            thread_count=3,
            random_seed=42,
            allow_writing_files=False,
            verbose=False,
        )
        m_r.fit(cb.Pool(xt_84.loc[mask_r_train], label=y_train[mask_r_train], cat_features=cats, feature_names=features_84))

        # Split prediction
        p_split_valid = np.zeros(len(valid_df), dtype=float)
        if mask_f_valid.any():
            p_split_valid[mask_f_valid] = m_f.predict_proba(cb.Pool(xv_84.loc[mask_f_valid], cat_features=cats, feature_names=features_84))[:, 1]
        if mask_r_valid.any():
            p_split_valid[mask_r_valid] = m_r.predict_proba(cb.Pool(xv_84.loc[mask_r_valid], cat_features=cats, feature_names=features_84))[:, 1]

        # 0.25 : 0.75 Blend
        p_blend_valid = BASELINE_WEIGHT * p_base_valid + CANDIDATE_WEIGHT * p_split_valid

        cand_metric = metric(p_blend_valid, y_valid)
        cand_brier = brier(p_blend_valid, y_valid)

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
            "feature_count": len(features_84),
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
        print(f"  Season {valid_season} Result:")
        print(f"    Baseline Metric : {base_metric:.4f} | Cand Metric: {cand_metric:.4f} ({rel_imp*100:+.2f}%)")
        print(f"    Baseline Brier  : {base_brier:.8f} | Cand Brier : {cand_brier:.8f} (Delta: {delta_brier:+.8f})")

        del m_pitch, m_base, m_f, m_r, xt, xv, xt_84, xv_84
        gc.collect()

    all_passed_2pct = all(r["relative_improvement"] >= 0.02 for r in results)
    all_positive = all(r["relative_improvement"] > 0 for r in results)

    summary = {
        "experiment_id": EXPERIMENT_ID,
        "official_train_only": True,
        "test_used": False,
        "external_data_used": False,
        "results_by_season": results,
        "all_positive_sign": all_positive,
        "all_passed_2pct_gate": all_passed_2pct,
        "total_elapsed_seconds": time.time() - start_time,
        "status": "PASS_SCREEN" if all_positive else "FAIL_SCREEN",
    }

    report_path = OUT_DIR / "screen_report.json"
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n" + "=" * 60)
    print(f"[{EXPERIMENT_ID}] Screening Completed!")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
