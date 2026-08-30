#!/usr/bin/env python3
"""Full-training production pipeline for REGIME-RCAPACITY-FULL-019.

Preregistered settings:
1. Baseline COMBO Model: 2019-2024 all train, iterations=300, lr=0.05, depth=6, l2=5, seed=42
2. F-Regime Model: 2019-2024 train excluding 2020, iterations=300, lr=0.05, depth=6, l2=10, seed=42
3. R-Regime Model: 2019-2024 train (game_type=='R'), iterations=600, lr=0.03, depth=7, l2=20, seed=42
4. Fixed blend: 0.25 * baseline + 0.75 * split(F/R)
"""
from __future__ import annotations

import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import catboost as cb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_combo_full_candidate_002 import COLS as BASE_COLS, extras, prep, prior_bundle
from scripts.screen_trackman_context_003 import NEW, attach, context_tables, load_tm
from src.asof_state_features import add_state_for_cutoff, add_state_walkforward
from src.target_aggregates import build_pitcher_count_state_target_history

EXPERIMENT_ID = "REGIME-RCAPACITY-FULL-019"
OUT_DIR = ROOT / "model" / EXPERIMENT_ID
FEATURE_COLUMNS = BASE_COLS + NEW
BASELINE_WEIGHT = 0.25
CANDIDATE_WEIGHT = 0.75


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    start_time = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[{EXPERIMENT_ID}] Step 1: Loading raw datasets...")
    raw = pd.read_csv(ROOT / "data" / "train.csv", encoding="utf-8-sig")
    test = pd.read_csv(ROOT / "data" / "test.csv", encoding="utf-8-sig")
    if raw.row_id.duplicated().any():
        raise RuntimeError("train.csv row_id is not unique")
    if test.row_id.duplicated().any():
        raise RuntimeError("test.csv row_id is not unique")

    trackman, map_df = load_tm()

    print(f"[{EXPERIMENT_ID}] Step 2: Engineering AS-OF state features...")
    train_base = add_state_walkforward(raw.drop(columns=["row_id", "control_success"]), 2025)
    test_base = add_state_for_cutoff(test.drop(columns=["row_id"]), raw.drop(columns=["row_id", "control_success"]))

    print(f"[{EXPERIMENT_ID}] Step 3: Engineering pitcher x count_state target aggregates...")
    at, ate, lookup, _ = build_pitcher_count_state_target_history(
        raw, test.assign(control_success=0), smoothing=100.0
    )

    xt = pd.concat([extras(train_base.reset_index(drop=True)), at.reset_index(drop=True)], axis=1)
    xe = pd.concat([extras(test_base.reset_index(drop=True)), ate.reset_index(drop=True)], axis=1)

    print(f"[{EXPERIMENT_ID}] Step 4: Attaching Trackman context tables...")
    parts = []
    for s in sorted(raw.season.unique()):
        ix = raw.season.eq(s)
        parts.append(attach(xt.loc[ix.to_numpy()].copy(), trackman, int(s)))
    xt = pd.concat(parts, axis=0).sort_index()

    c25, h25 = context_tables(trackman, 2025)
    xe = xe.copy()
    xe["__hand"] = xe["batter_hand"].map({1: "Left", 2: "Right"}).fillna(xe["batter_hand"].astype(str))
    xe = xe.join(c25, on=["pitcher_id", "balls_before", "strikes_before"]).join(h25, on=["pitcher_id", "__hand"]).drop(columns="__hand")

    xt = xt[FEATURE_COLUMNS]
    xe = xe[FEATURE_COLUMNS]

    xt, categorical_columns = prep(xt)
    xe, _ = prep(xe)

    y_train = raw.control_success.to_numpy(np.int8)

    # -------------------------------------------------------------
    # Model 1: Baseline COMBO-TM Model (all train)
    # -------------------------------------------------------------
    print(f"[{EXPERIMENT_ID}] Step 5.1: Training Baseline COMBO Model (1,475,092 rows, depth 6, 300 trees)...")
    m_baseline = cb.CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=5,
        thread_count=3,
        random_seed=42,
        loss_function="Logloss",
        allow_writing_files=False,
        verbose=False,
    )
    m_baseline.fit(
        cb.Pool(xt, label=y_train, cat_features=categorical_columns, feature_names=FEATURE_COLUMNS)
    )
    m_baseline.save_model(OUT_DIR / "model_baseline_combo.cbm")
    pred_baseline_test = m_baseline.predict_proba(
        cb.Pool(xe, cat_features=categorical_columns, feature_names=FEATURE_COLUMNS)
    )[:, 1]

    # -------------------------------------------------------------
    # Model 2: F-Regime Model (game_type=='F' and season!=2020)
    # -------------------------------------------------------------
    mask_f = raw.game_type.astype(str).eq("F").to_numpy() & raw.season.ne(2020).to_numpy()
    print(f"[{EXPERIMENT_ID}] Step 5.2: Training F-Regime Model ({int(mask_f.sum())} rows, depth 6, 300 trees)...")
    m_f = cb.CatBoostClassifier(
        iterations=300,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=10,
        thread_count=3,
        random_seed=42,
        loss_function="Logloss",
        allow_writing_files=False,
        verbose=False,
    )
    m_f.fit(
        cb.Pool(xt.loc[mask_f], label=y_train[mask_f], cat_features=categorical_columns, feature_names=FEATURE_COLUMNS)
    )
    m_f.save_model(OUT_DIR / "model_regime_f.cbm")

    # -------------------------------------------------------------
    # Model 3: R-Regime Model (game_type=='R', depth 7, 600 trees)
    # -------------------------------------------------------------
    mask_r = raw.game_type.astype(str).eq("R").to_numpy()
    print(f"[{EXPERIMENT_ID}] Step 5.3: Training R-Regime Capacity Model ({int(mask_r.sum())} rows, depth 7, 600 trees)...")
    m_r = cb.CatBoostClassifier(
        iterations=600,
        learning_rate=0.03,
        depth=7,
        l2_leaf_reg=20,
        thread_count=3,
        random_seed=42,
        loss_function="Logloss",
        allow_writing_files=False,
        verbose=False,
    )
    m_r.fit(
        cb.Pool(xt.loc[mask_r], label=y_train[mask_r], cat_features=categorical_columns, feature_names=FEATURE_COLUMNS)
    )
    m_r.save_model(OUT_DIR / "model_regime_r.cbm")

    # -------------------------------------------------------------
    # Step 6: Compute Test Predictions with 25:75 Blend
    # -------------------------------------------------------------
    print(f"[{EXPERIMENT_ID}] Step 6: Generating test predictions with 0.25:0.75 blend...")
    test_mask_f = test.game_type.astype(str).eq("F").to_numpy()
    test_mask_r = test.game_type.astype(str).eq("R").to_numpy()

    pred_split_test = np.zeros(len(test), dtype=float)
    if test_mask_f.any():
        pred_split_test[test_mask_f] = m_f.predict_proba(
            cb.Pool(xe.loc[test_mask_f], cat_features=categorical_columns, feature_names=FEATURE_COLUMNS)
        )[:, 1]
    if test_mask_r.any():
        pred_split_test[test_mask_r] = m_r.predict_proba(
            cb.Pool(xe.loc[test_mask_r], cat_features=categorical_columns, feature_names=FEATURE_COLUMNS)
        )[:, 1]

    pred_blend_test = BASELINE_WEIGHT * pred_baseline_test + CANDIDATE_WEIGHT * pred_split_test

    if not np.isfinite(pred_blend_test).all() or not ((pred_blend_test >= 0) & (pred_blend_test <= 1)).all():
        raise RuntimeError("test blend prediction range/finite check failed")

    # -------------------------------------------------------------
    # Step 7: Saving Artifacts and Lookups
    # -------------------------------------------------------------
    print(f"[{EXPERIMENT_ID}] Step 7: Saving lookups, priors, and manifests...")
    prior_bundle(raw, OUT_DIR)
    (OUT_DIR / "feature_columns.json").write_text(json.dumps(FEATURE_COLUMNS, ensure_ascii=False, indent=2))
    lookup.to_csv(OUT_DIR / "pitcher_count_lookup.csv", index=False)
    c25.reset_index().to_csv(OUT_DIR / "trackman_count_lookup.csv", index=False)
    h25.reset_index().to_csv(OUT_DIR / "trackman_hand_lookup.csv", index=False)
    map_df.to_csv(OUT_DIR / "pitcher_id_map_audit.csv", index=False)

    df_sub = pd.DataFrame({"row_id": test.row_id, "control_success": pred_blend_test})
    df_sub.to_csv(OUT_DIR / "test_predictions.csv", index=False)

    elapsed = time.time() - start_time
    report = {
        "experiment_id": EXPERIMENT_ID,
        "official_train_only": True,
        "test_used_for_training": False,
        "external_data_used": False,
        "mapping_source": "model/TRACKMAN-MAP-004/pitcher_id_map.csv",
        "mapping_rows": len(map_df),
        "train_rows_total": len(raw),
        "test_rows_total": len(test),
        "feature_count": len(FEATURE_COLUMNS),
        "categorical_columns": categorical_columns,
        "blend_weights": {
            "baseline_combo": BASELINE_WEIGHT,
            "regime_split": CANDIDATE_WEIGHT,
        },
        "models": {
            "baseline_combo": {
                "file": "model_baseline_combo.cbm",
                "train_rows": len(raw),
                "trees": int(m_baseline.tree_count_),
                "params": {"iterations": 300, "learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 5},
            },
            "regime_f": {
                "file": "model_regime_f.cbm",
                "train_rows": int(mask_f.sum()),
                "excluded_seasons": [2020],
                "trees": int(m_f.tree_count_),
                "params": {"iterations": 300, "learning_rate": 0.05, "depth": 6, "l2_leaf_reg": 10},
            },
            "regime_r": {
                "file": "model_regime_r.cbm",
                "train_rows": int(mask_r.sum()),
                "trees": int(m_r.tree_count_),
                "params": {"iterations": 600, "learning_rate": 0.03, "depth": 7, "l2_leaf_reg": 20},
            },
        },
        "test_prediction_stats": {
            "count": len(df_sub),
            "min": float(pred_blend_test.min()),
            "max": float(pred_blend_test.max()),
            "mean": float(pred_blend_test.mean()),
            "std": float(pred_blend_test.std()),
        },
        "elapsed_seconds": elapsed,
        "status": "PASS_FULL_TRAIN_MODEL",
        "submission_status": "HOLD",
    }

    (OUT_DIR / "full_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"[{EXPERIMENT_ID}] Full training complete in {elapsed:.1f}s.")


if __name__ == "__main__":
    main()
