#!/usr/bin/env python3
"""EXP-029-REGIME-6SEED-FULL: 6-Seed Production Full-Train on 2019-2024 Official Data.

Preregistered protocol:
1. Data: Official train.csv (1,475,092 rows) and test.csv (246,478 rows).
2. Features: 81 features (Base 73 + Trackman 8).
3. 6 Fixed Seeds: [42, 7, 2024, 99, 1, 123].
4. Models:
   - Baseline COMBO: 1,475,092 rows, 300 trees, depth 6, lr 0.05, l2 5 (6 seeds)
   - F-Regime Model: 137,791 rows (excluding 2020), 300 trees, depth 6, lr 0.05, l2 10 (6 seeds)
   - R-Regime Capacity Model: 1,314,088 rows, 600 trees, depth 7, lr 0.03, l2 20 (6 seeds)
5. Blending: 0.25 * Baseline_6seed + 0.75 * Split_6seed.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import catboost as cb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_combo_full_candidate_002 import COLS as BASE_COLS, extras, prep
from scripts.screen_trackman_context_003 import NEW as TM_COLS, attach, context_tables, load_tm
from src.asof_state_features import add_state_for_cutoff, add_state_walkforward
from src.target_aggregates import build_pitcher_count_state_target_history

EXPERIMENT_ID = "REGIME-6SEED-FULL-029"
OUT_DIR = ROOT / "model" / EXPERIMENT_ID
FEATURE_COLUMNS = BASE_COLS + TM_COLS
SEEDS = [42, 7, 2024, 99, 1, 123]
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
    # 1. Train 6-Seed Baseline COMBO Models
    # -------------------------------------------------------------
    print(f"\n--- [1/3] Training 6-Seed Baseline COMBO Models (1,475,092 rows) ---")
    base_preds_test = []
    base_paths = []
    for s_idx, s in enumerate(SEEDS, 1):
        t0 = time.time()
        m_base = cb.CatBoostClassifier(
            iterations=300, learning_rate=0.05, depth=6, l2_leaf_reg=5,
            loss_function="Logloss", thread_count=-1, random_seed=s,
            allow_writing_files=False, verbose=False
        )
        m_base.fit(cb.Pool(xt, label=y_train, cat_features=categorical_columns, feature_names=FEATURE_COLUMNS))
        p_path = OUT_DIR / f"baseline_combo_seed_{s}.cbm"
        m_base.save_model(p_path)
        base_paths.append(str(p_path.relative_to(ROOT)))
        pred_t = m_base.predict_proba(cb.Pool(xe, cat_features=categorical_columns, feature_names=FEATURE_COLUMNS))[:, 1]
        base_preds_test.append(pred_t)
        print(f"  [Base Seed {s_idx}/6 ({s})] Completed in {time.time() - t0:.1f}s")

    pred_baseline_6seed = np.mean(base_preds_test, axis=0)

    # -------------------------------------------------------------
    # 2. Train 6-Seed F-Regime Models
    # -------------------------------------------------------------
    mask_f_train = raw.game_type.astype(str).eq("F").to_numpy() & raw.season.ne(2020).to_numpy()
    mask_f_test = test.game_type.astype(str).eq("F").to_numpy()
    print(f"\n--- [2/3] Training 6-Seed F-Regime Models ({int(mask_f_train.sum()):,} rows) ---")
    f_preds_test = []
    f_paths = []
    x_f = xt.loc[mask_f_train]
    y_f = y_train[mask_f_train]
    xe_f = xe.loc[mask_f_test]
    for s_idx, s in enumerate(SEEDS, 1):
        t0 = time.time()
        m_f = cb.CatBoostClassifier(
            iterations=300, learning_rate=0.05, depth=6, l2_leaf_reg=10,
            loss_function="Logloss", thread_count=-1, random_seed=s,
            allow_writing_files=False, verbose=False
        )
        m_f.fit(cb.Pool(x_f, label=y_f, cat_features=categorical_columns, feature_names=FEATURE_COLUMNS))
        p_path = OUT_DIR / f"f_regime_seed_{s}.cbm"
        m_f.save_model(p_path)
        f_paths.append(str(p_path.relative_to(ROOT)))
        pred_f = m_f.predict_proba(cb.Pool(xe_f, cat_features=categorical_columns, feature_names=FEATURE_COLUMNS))[:, 1]
        f_preds_test.append(pred_f)
        print(f"  [F-Regime Seed {s_idx}/6 ({s})] Completed in {time.time() - t0:.1f}s")

    pred_f_6seed = np.mean(f_preds_test, axis=0) if mask_f_test.any() else np.array([])

    # -------------------------------------------------------------
    # 3. Train 6-Seed R-Regime Models (Depth 7, 600 trees)
    # -------------------------------------------------------------
    mask_r_train = raw.game_type.astype(str).eq("R").to_numpy()
    mask_r_test = test.game_type.astype(str).eq("R").to_numpy()
    print(f"\n--- [3/3] Training 6-Seed R-Regime Models ({int(mask_r_train.sum()):,} rows, depth 7, 600 trees) ---")
    r_preds_test = []
    r_paths = []
    x_r = xt.loc[mask_r_train]
    y_r = y_train[mask_r_train]
    xe_r = xe.loc[mask_r_test]
    for s_idx, s in enumerate(SEEDS, 1):
        t0 = time.time()
        m_r = cb.CatBoostClassifier(
            iterations=600, learning_rate=0.03, depth=7, l2_leaf_reg=20,
            loss_function="Logloss", thread_count=-1, random_seed=s,
            allow_writing_files=False, verbose=False
        )
        m_r.fit(cb.Pool(x_r, label=y_r, cat_features=categorical_columns, feature_names=FEATURE_COLUMNS))
        p_path = OUT_DIR / f"r_regime_seed_{s}.cbm"
        m_r.save_model(p_path)
        r_paths.append(str(p_path.relative_to(ROOT)))
        pred_r = m_r.predict_proba(cb.Pool(xe_r, cat_features=categorical_columns, feature_names=FEATURE_COLUMNS))[:, 1]
        r_preds_test.append(pred_r)
        print(f"  [R-Regime Seed {s_idx}/6 ({s})] Completed in {time.time() - t0:.1f}s")

    pred_r_6seed = np.mean(r_preds_test, axis=0) if mask_r_test.any() else np.array([])

    # -------------------------------------------------------------
    # 4. Assemble Split & Final Blended Predictions
    # -------------------------------------------------------------
    pred_split_6seed = np.zeros(len(test), dtype=float)
    if mask_f_test.any():
        pred_split_6seed[mask_f_test] = pred_f_6seed
    if mask_r_test.any():
        pred_split_6seed[mask_r_test] = pred_r_6seed

    pred_blend_6seed = BASELINE_WEIGHT * pred_baseline_6seed + CANDIDATE_WEIGHT * pred_split_6seed

    # Save validation artifacts
    np.save(OUT_DIR / "pred_baseline_6seed_test.npy", pred_baseline_6seed)
    np.save(OUT_DIR / "pred_split_6seed_test.npy", pred_split_6seed)
    np.save(OUT_DIR / "pred_blend_6seed_test.npy", pred_blend_6seed)

    # Save lookup and assets
    lookup.to_csv(OUT_DIR / "pitcher_count_state_lookup.csv", index=False)
    c25.reset_index().to_csv(OUT_DIR / "context_2025.csv", index=False)
    h25.reset_index().to_csv(OUT_DIR / "hand_2025.csv", index=False)

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "official_train_only": True,
        "test_used_for_training": False,
        "external_data_used": False,
        "train_rows_total": len(raw),
        "test_rows_total": len(test),
        "seeds": SEEDS,
        "feature_count": len(FEATURE_COLUMNS),
        "categorical_columns": categorical_columns,
        "baseline_models": base_paths,
        "f_regime_models": f_paths,
        "r_regime_models": r_paths,
        "blend_weights": {
            "baseline": BASELINE_WEIGHT,
            "split": CANDIDATE_WEIGHT
        },
        "test_pred_mean": float(np.mean(pred_blend_6seed)),
        "test_pred_std": float(np.std(pred_blend_6seed)),
        "total_elapsed_seconds": time.time() - start_time,
        "status": "FULL_TRAIN_SUCCESS"
    }

    manifest_path = OUT_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("\n" + "=" * 60)
    print(f"[{EXPERIMENT_ID}] Full-Train Complete in {time.time() - start_time:.1f}s!")
    print(f"Test prediction mean: {np.mean(pred_blend_6seed):.6f}, std: {np.std(pred_blend_6seed):.6f}")
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
