#!/usr/bin/env python3
"""Train an XGBoost FE-001 candidate with strict season Holdout validation."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import (
    TARGET_COL,
    brier_score,
    brier_skill_score,
    build_features,
    hackathon_score,
    load_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train XGBoost with the FE-001 season Holdout contract"
    )
    parser.add_argument("--train-path", type=Path, default=ROOT / "data" / "train.csv")
    parser.add_argument("--exp-id", default="XGB-FE001-EW-2023")
    parser.add_argument("--train-seasons", default="2019,2020,2021,2022")
    parser.add_argument("--valid-season", type=int, default=2023)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--min-child-weight", type=float, default=25.0)
    parser.add_argument("--max-bin", type=int, default=256)
    parser.add_argument("--n-estimators", type=int, default=1500)
    parser.add_argument("--early-stopping", type=int, default=50)
    return parser.parse_args()


def peak_memory_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / 1024.0 if sys.platform != "darwin" else value / (1024.0**2)


def main() -> None:
    args = parse_args()
    start = time.time()
    train_seasons = [
        int(value.strip()) for value in args.train_seasons.split(",") if value.strip()
    ]
    if len(train_seasons) != len(set(train_seasons)):
        raise ValueError("학습 시즌이 중복되었습니다")
    if sorted(train_seasons) != train_seasons:
        raise ValueError("학습 시즌은 오름차순이어야 합니다")
    if max(train_seasons) >= args.valid_season:
        raise ValueError("모든 학습 시즌은 검증 시즌보다 앞서야 합니다")
    if args.max_depth < 1 or args.min_child_weight <= 0 or args.max_bin < 2:
        raise ValueError("XGBoost 트리 파라미터 범위가 잘못되었습니다")

    print("=" * 72)
    print(f"XGBoost FE-001 season Holdout [{args.exp_id}]")
    print(f"train={train_seasons} valid={args.valid_season}")
    print(f"xgboost={xgb.__version__} python={platform.python_version()}")
    print("=" * 72)

    raw = load_data(str(args.train_path), is_train=True)
    train = raw.loc[raw["season"].isin(train_seasons)].copy()
    valid = raw.loc[raw["season"].eq(args.valid_season)].copy()
    if train.empty or valid.empty:
        raise ValueError("학습 또는 검증 데이터가 비어 있습니다")
    if sorted(int(value) for value in train["season"].unique()) != train_seasons:
        raise ValueError("요청한 학습 시즌 중 원본 데이터에 없는 시즌이 있습니다")

    y_train = train[TARGET_COL].to_numpy(dtype=np.int8)
    y_valid = valid[TARGET_COL].to_numpy(dtype=np.int8)
    x_train = build_features(train)
    x_valid = build_features(valid)
    feature_columns = list(x_train.columns)
    if len(feature_columns) != 60 or list(x_valid.columns) != feature_columns:
        raise ValueError("FE-001 60피처 학습·검증 계약이 일치하지 않습니다")
    categorical_features = [
        column
        for column in feature_columns
        if str(x_train[column].dtype) in {"category", "object"}
    ]
    for column in categorical_features:
        x_train[column] = x_train[column].astype("category")
        x_valid[column] = x_valid[column].astype("category")
        if list(x_train[column].cat.categories) != list(
            x_valid[column].cat.categories
        ):
            raise ValueError(f"범주형 category 계약이 다릅니다: {column}")

    print(
        f"train={len(x_train):,}x{len(feature_columns)} "
        f"valid={len(x_valid):,}x{len(feature_columns)}"
    )
    print(f"categorical={categorical_features}")
    dtrain = xgb.QuantileDMatrix(
        x_train,
        label=y_train,
        enable_categorical=True,
        max_bin=args.max_bin,
    )
    dvalid = xgb.QuantileDMatrix(
        x_valid,
        label=y_valid,
        enable_categorical=True,
        max_bin=args.max_bin,
        ref=dtrain,
    )
    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "tree_method": "hist",
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "min_child_weight": args.min_child_weight,
        "max_bin": args.max_bin,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "seed": args.seed,
        "nthread": -1,
    }
    evaluations: dict = {}
    train_start = time.time()
    booster = xgb.train(
        params,
        dtrain,
        num_boost_round=args.n_estimators,
        evals=[(dtrain, "train"), (dvalid, "valid")],
        early_stopping_rounds=args.early_stopping,
        evals_result=evaluations,
        verbose_eval=50,
    )
    train_seconds = time.time() - train_start
    if not hasattr(booster, "best_iteration"):
        raise RuntimeError("XGBoost early stopping best_iteration이 없습니다")
    best_iteration_index = int(booster.best_iteration)
    best_rounds = best_iteration_index + 1
    prediction = booster.predict(dvalid, iteration_range=(0, best_rounds))
    model_brier = float(brier_score(y_valid, prediction))
    model_bss = float(brier_skill_score(y_valid, prediction))
    model_score = float(hackathon_score(y_valid, prediction))
    train_mean = float(y_train.mean())
    constant_prediction = np.full(len(y_valid), train_mean, dtype=np.float64)
    constant_brier = float(brier_score(y_valid, constant_prediction))

    experiment_dir = ROOT / "model" / args.exp_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    model_path = experiment_dir / "model.json"
    features_path = experiment_dir / "feature_columns.json"
    metadata_path = experiment_dir / "metadata.json"
    predictions_path = experiment_dir / "validation_predictions.csv"
    booster.save_model(model_path)
    features_path.write_text(
        json.dumps(feature_columns, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    prediction_frame = pd.DataFrame(
        {
            "row_id": valid["row_id"].to_numpy(),
            "season": valid["season"].to_numpy(),
            "target": y_valid,
            "pred_constant": constant_prediction,
            "pred_xgboost": prediction,
        }
    )
    prediction_frame.to_csv(predictions_path, index=False)
    metadata = {
        "exp_id": args.exp_id,
        "model_family": "xgboost",
        "artifact_role": "validation_candidate",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "xgboost": xgb.__version__,
        },
        "train_seasons": train_seasons,
        "valid_season": args.valid_season,
        "train_rows": int(len(x_train)),
        "valid_rows": int(len(x_valid)),
        "feature_count": len(feature_columns),
        "features": feature_columns,
        "categorical_features": categorical_features,
        "hyperparameters": params,
        "n_estimators_limit": args.n_estimators,
        "early_stopping_rounds": args.early_stopping,
        "best_iteration_index": best_iteration_index,
        "best_rounds": best_rounds,
        "metrics": {
            "constant_baseline": {
                "brier_score": constant_brier,
                "prediction_mean": train_mean,
            },
            "xgboost": {
                "brier_score": model_brier,
                "bss": model_bss,
                "score": model_score,
            },
        },
        "training_time_sec": round(train_seconds, 2),
        "total_time_sec": round(time.time() - start, 2),
        "peak_memory_mb": round(peak_memory_mb(), 1),
        "model_file": model_path.name,
        "feature_columns_file": features_path.name,
        "validation_predictions_file": predictions_path.name,
        "active_model_sync": False,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("=" * 72)
    print(
        f"best_rounds={best_rounds} brier={model_brier:.9f} "
        f"bss={model_bss:.9f} time={train_seconds:.1f}s"
    )
    print(f"saved={experiment_dir}")
    print("active model unchanged")


if __name__ == "__main__":
    main()
