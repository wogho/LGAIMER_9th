#!/usr/bin/env python3
"""Train the independently rebuilt multi-seed reference architecture.

Official train only; no test rows, external constants, or public artifacts.
This produces validation predictions and metrics, not a submission ZIP.
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

import catboost as cb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ref_rebuild_features import CAT_COLS, FEATURE_COLUMNS, engineer, prepare

OUT = ROOT / "model" / "REF-REBUILD-001"
TRAIN = ROOT / "data" / "train.csv"
LABELS = ROOT / "model" / "REF-AUX-LABEL-001" / "recovered_labels.csv.gz"
SEEDS_SUCCESS = [42, 7, 2024, 99, 1, 123, 777]
SEEDS_AUX = [42, 7, 2024]


def brier(y, p):
    return float(np.mean((np.asarray(y) - np.asarray(p)) ** 2))


def bss(y, p):
    r = float(np.mean(y))
    return float(100000.0 * (1.0 - brier(y, p) / (r * (1.0 - r))))


def main():
    raw = pd.read_csv(TRAIN, encoding="utf-8-sig")
    labels = pd.read_csv(LABELS, usecols=["row_id", "middle", "reverse"])
    if not raw["row_id"].equals(labels["row_id"]):
        raise AssertionError("train/label row_id order mismatch")
    y_success = raw["control_success"].to_numpy(dtype=np.int8)
    valid_label = labels[["middle", "reverse"]].notna().all(axis=1).to_numpy()
    y_mr = ((labels["middle"].eq(1)) | (labels["reverse"].eq(1))).to_numpy(dtype=np.int8)
    y_wayoff = ((y_success == 0) & (y_mr == 0)).astype(np.int8)
    train_mask = raw["season"].lt(2024).to_numpy() & valid_label
    valid_mask = raw["season"].eq(2024).to_numpy()
    global_mean = float(y_success[train_mask].mean())
    x = prepare(engineer(raw.drop(columns=["row_id", "control_success"]), global_mean))
    cat_idx = [x.columns.get_loc(c) for c in CAT_COLS]
    train_x, valid_x = x.loc[train_mask], x.loc[valid_mask]
    train_pool = cb.Pool(train_x, cat_features=cat_idx, feature_names=FEATURE_COLUMNS)
    valid_pool = cb.Pool(valid_x, cat_features=cat_idx, feature_names=FEATURE_COLUMNS)
    targets = {"success": y_success, "mr": y_mr, "wayoff": y_wayoff}
    seeds = {"success": SEEDS_SUCCESS, "mr": SEEDS_AUX, "wayoff": SEEDS_AUX}
    iters = {"success": 260, "mr": 330, "wayoff": 60}
    pred = {name: [] for name in targets}
    fit_meta = {name: [] for name in targets}
    for name, target in targets.items():
        for seed in seeds[name]:
            params = dict(iterations=iters[name], learning_rate=0.05, depth=6,
                          loss_function="Logloss", eval_metric="Logloss",
                          thread_count=12, random_seed=seed,
                          allow_writing_files=False, verbose=False,
                          early_stopping_rounds=60)
            model = cb.CatBoostClassifier(**params)
            train_target_pool = cb.Pool(train_x, label=target[train_mask], cat_features=cat_idx, feature_names=FEATURE_COLUMNS)
            valid_target_pool = cb.Pool(valid_x, label=target[valid_mask], cat_features=cat_idx, feature_names=FEATURE_COLUMNS)
            model.fit(train_target_pool, eval_set=valid_target_pool, verbose=False)
            p = model.predict_proba(valid_pool)[:, 1]
            pred[name].append(p)
            fit_meta[name].append({"seed": seed, "best_iteration": int(model.get_best_iteration()), "tree_count": int(model.tree_count_)})
    mean_pred = {name: np.mean(vals, axis=0) for name, vals in pred.items()}
    results = {}
    for name, target in targets.items():
        y = target[valid_mask]
        results[name] = {"seeds": seeds[name], "brier": brier(y, mean_pred[name]),
                         "bss": bss(y, mean_pred[name]), "mean": float(mean_pred[name].mean()),
                         "fits": fit_meta[name]}
    out = {"experiment_id": "REF-REBUILD-001", "environment": {"python": platform.python_version(), "catboost": cb.__version__},
           "feature_count": len(FEATURE_COLUMNS), "categorical_features": CAT_COLS,
           "train_rows": int(train_mask.sum()), "valid_rows": int(valid_mask.sum()),
           "official_train_only": True, "test_used": False, "external_data_used": False,
           "results": results, "status": "MULTISEED_COMPLETE", "submission_status": "HOLD"}
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "multiseed_valid_predictions.npz", **mean_pred)
    (OUT / "multiseed_report.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
