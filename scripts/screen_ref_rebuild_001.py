#!/usr/bin/env python3
"""One-seed temporal screen for the independently rebuilt 57-feature contract."""
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
SEED = 42


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
    valid_label = labels[["middle", "reverse"]].notna().all(axis=1).to_numpy()
    y_success = raw["control_success"].to_numpy(dtype=np.int8)
    mr = ((labels["middle"].eq(1)) | (labels["reverse"].eq(1))).to_numpy(dtype=np.int8)
    wayoff = ((y_success == 0) & (mr == 0)).astype(np.int8)
    train_mask = raw["season"].lt(2024).to_numpy() & valid_label
    valid_mask = raw["season"].eq(2024).to_numpy()
    global_mean = float(y_success[train_mask].mean())
    x = engineer(raw.drop(columns=["row_id", "control_success"]), global_mean)
    x = prepare(x)
    cat_idx = [x.columns.get_loc(c) for c in CAT_COLS]
    train_pool_x = cb.Pool(x.loc[train_mask], cat_features=cat_idx, feature_names=FEATURE_COLUMNS)
    valid_pool_x = cb.Pool(x.loc[valid_mask], cat_features=cat_idx, feature_names=FEATURE_COLUMNS)
    params = dict(iterations=600, learning_rate=0.05, depth=6, loss_function="Logloss", eval_metric="Logloss", thread_count=6, random_seed=SEED, allow_writing_files=False, verbose=False, early_stopping_rounds=60)
    predictions = {}
    results = {}
    for name, target in (("success", y_success), ("mr", mr), ("wayoff", wayoff)):
        model = cb.CatBoostClassifier(**params)
        model.fit(
            cb.Pool(x.loc[train_mask], label=target[train_mask], cat_features=cat_idx, feature_names=FEATURE_COLUMNS),
            eval_set=cb.Pool(x.loc[valid_mask], label=target[valid_mask], cat_features=cat_idx, feature_names=FEATURE_COLUMNS),
            verbose=False,
        )
        p = model.predict_proba(valid_pool_x)[:, 1]
        predictions[name] = p
        results[name] = {"best_iteration": int(model.get_best_iteration()), "tree_count": int(model.tree_count_), "brier": brier(target[valid_mask], p), "bss": bss(target[valid_mask], p), "mean": float(p.mean())}
    # A fixed half-split diagnostic only; no submission coefficient is produced.
    y = y_success[valid_mask].astype(float)
    half = np.arange(len(y)) % 2 == 0
    z = np.column_stack([np.log(np.clip(predictions["success"], 1e-6, 1 - 1e-6) / np.clip(1 - predictions["success"], 1e-6, 1 - 1e-6)), np.log(np.clip(predictions["mr"], 1e-6, 1 - 1e-6) / np.clip(1 - predictions["mr"], 1e-6, 1 - 1e-6)), np.log(np.clip(predictions["wayoff"], 1e-6, 1 - 1e-6) / np.clip(1 - predictions["wayoff"], 1e-6, 1 - 1e-6))])
    from scipy.optimize import minimize
    mu = z[half, 1:].mean(axis=0)
    u = z[:, 1] - mu[0]
    v = z[:, 2] - mu[1]
    def nll(w):
        q = 1.0 / (1.0 + np.exp(-(z[:, 0] + w[0] * u + w[1] * v)))
        q = np.clip(q, 1e-8, 1 - 1e-8)
        return float(-np.mean(y[half] * np.log(q[half]) + (1 - y[half]) * np.log(1 - q[half])))
    coef = minimize(nll, [0.0, 0.0], method="Nelder-Mead").x
    adjusted = 1.0 / (1.0 + np.exp(-(z[:, 0] + coef[0] * u + coef[1] * v)))
    results["fixed_half_offset_diagnostic"] = {"b": float(coef[0]), "c": float(coef[1]), "delta_brier_holdout_half": float(brier(y[~half], adjusted[~half]) - brier(y[~half], predictions["success"][~half]))}
    report = {"experiment_id": "REF-REBUILD-001", "environment": {"python": platform.python_version(), "catboost": cb.__version__}, "feature_count": len(FEATURE_COLUMNS), "categorical_features": CAT_COLS, "train_rows": int(train_mask.sum()), "valid_rows": int(valid_mask.sum()), "global_mean_train": global_mean, "official_train_only": True, "test_used": False, "results": results, "status": "SCREEN_COMPLETE", "submission_status": "HOLD"}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "screen_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
