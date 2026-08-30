#!/usr/bin/env python3
"""Train a CatBoost FE-001 candidate with strict season Holdout validation."""

from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path

import catboost as cb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import (
    TARGET_COL,
    brier_score,
    brier_skill_score,
    build_features,
    build_features_fe002,
    hackathon_score,
    load_data,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CatBoost with the FE-001 season Holdout contract"
    )
    parser.add_argument("--train-path", type=Path, default=ROOT / "data" / "train.csv")
    parser.add_argument("--exp-id", default="CAT-FE001-EW-2023")
    parser.add_argument("--train-seasons", default="2019,2020,2021,2022")
    parser.add_argument("--valid-season", type=int, default=2023)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--n-estimators", type=int, default=1500)
    parser.add_argument("--early-stopping", type=int, default=50)
    parser.add_argument(
        "--game-type-filter",
        default="",
        help="학습·검증 행을 단일 game_type(F 또는 R)으로 제한",
    )
    parser.add_argument(
        "--fe002-groups",
        default="",
        help="현재 행 기반 FE-002 그룹 쉼표 목록: state,form,support",
    )
    return parser.parse_args()


def peak_memory_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / 1024.0 if sys.platform != "darwin" else value / (1024.0**2)


def normalize_categorical(
    frame: pd.DataFrame, categorical_features: list[str]
) -> pd.DataFrame:
    normalized = frame.copy()
    for column in categorical_features:
        normalized[column] = (
            normalized[column].astype("string").fillna("<NA>").astype(str)
        )
    return normalized


def main() -> None:
    args = parse_args()
    start = time.time()
    train_seasons = [
        int(value.strip()) for value in args.train_seasons.split(",") if value.strip()
    ]
    fe002_groups = [
        group.strip() for group in args.fe002_groups.split(",") if group.strip()
    ]
    if len(fe002_groups) != len(set(fe002_groups)):
        raise ValueError("--fe002-groups에 중복 그룹이 있습니다")
    if len(train_seasons) != len(set(train_seasons)):
        raise ValueError("학습 시즌이 중복되었습니다")
    if sorted(train_seasons) != train_seasons:
        raise ValueError("학습 시즌은 오름차순이어야 합니다")
    if max(train_seasons) >= args.valid_season:
        raise ValueError("모든 학습 시즌은 검증 시즌보다 앞서야 합니다")
    if args.depth < 1 or args.n_estimators < 1 or args.early_stopping < 1:
        raise ValueError("CatBoost 학습 파라미터 범위가 잘못되었습니다")

    print("=" * 72)
    print(f"CatBoost FE-001 season Holdout [{args.exp_id}]")
    print(f"train={train_seasons} valid={args.valid_season}")
    print(f"game_type_filter={args.game_type_filter or 'none'}")
    print(f"fe002_groups={fe002_groups or 'none'}")
    print(f"catboost={cb.__version__} python={platform.python_version()}")
    print("=" * 72)

    raw = load_data(str(args.train_path), is_train=True)
    train = raw.loc[raw["season"].isin(train_seasons)].copy()
    valid = raw.loc[raw["season"].eq(args.valid_season)].copy()
    if args.game_type_filter:
        train = train.loc[train["game_type"].astype("string").eq(args.game_type_filter)].copy()
        valid = valid.loc[valid["game_type"].astype("string").eq(args.game_type_filter)].copy()
    if train.empty or valid.empty:
        raise ValueError("학습 또는 검증 데이터가 비어 있습니다")
    y_train = train[TARGET_COL].to_numpy(dtype=np.int8)
    y_valid = valid[TARGET_COL].to_numpy(dtype=np.int8)
    if fe002_groups:
        x_train = build_features_fe002(train, fe002_groups)
        x_valid = build_features_fe002(valid, fe002_groups)
    else:
        x_train = build_features(train)
        x_valid = build_features(valid)
    feature_columns = list(x_train.columns)
    expected_feature_count = 60 + sum(
        {"state": 5, "form": 4, "support": 5}[group]
        for group in fe002_groups
    )
    if (
        len(feature_columns) != expected_feature_count
        or list(x_valid.columns) != feature_columns
    ):
        raise ValueError("FE-001/FE-002 학습·검증 피처 계약이 일치하지 않습니다")
    categorical_features = [
        column
        for column in feature_columns
        if str(x_train[column].dtype) in {"category", "object"}
    ]
    x_train = normalize_categorical(x_train, categorical_features)
    x_valid = normalize_categorical(x_valid, categorical_features)
    print(
        f"train={len(x_train):,}x{len(feature_columns)} "
        f"valid={len(x_valid):,}x{len(feature_columns)}"
    )
    print(f"categorical={categorical_features}")

    train_pool = cb.Pool(
        x_train,
        label=y_train,
        cat_features=categorical_features,
        feature_names=feature_columns,
    )
    valid_pool = cb.Pool(
        x_valid,
        label=y_valid,
        cat_features=categorical_features,
        feature_names=feature_columns,
    )
    params = {
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "iterations": args.n_estimators,
        "learning_rate": args.learning_rate,
        "depth": args.depth,
        "l2_leaf_reg": 3.0,
        "random_strength": 1.0,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.8,
        "rsm": 0.8,
        "border_count": 254,
        "random_seed": args.seed,
        "thread_count": -1,
        "allow_writing_files": False,
    }
    model = cb.CatBoostClassifier(**params)
    train_start = time.time()
    model.fit(
        train_pool,
        eval_set=valid_pool,
        use_best_model=True,
        early_stopping_rounds=args.early_stopping,
        verbose=50,
    )
    train_seconds = time.time() - train_start
    best_iteration_index = int(model.get_best_iteration())
    best_rounds = int(model.tree_count_)
    if best_iteration_index < 0 or best_rounds != best_iteration_index + 1:
        raise RuntimeError("CatBoost best iteration과 저장 tree 수가 일치하지 않습니다")
    prediction = model.predict_proba(valid_pool)[:, 1]
    model_brier = float(brier_score(y_valid, prediction))
    model_bss = float(brier_skill_score(y_valid, prediction))
    model_score = float(hackathon_score(y_valid, prediction))
    train_mean = float(y_train.mean())
    constant_prediction = np.full(len(y_valid), train_mean, dtype=np.float64)
    constant_brier = float(brier_score(y_valid, constant_prediction))

    experiment_dir = ROOT / "model" / args.exp_id
    experiment_dir.mkdir(parents=True, exist_ok=True)
    model_path = experiment_dir / "model.cbm"
    features_path = experiment_dir / "feature_columns.json"
    metadata_path = experiment_dir / "metadata.json"
    predictions_path = experiment_dir / "validation_predictions.csv"
    model.save_model(model_path, format="cbm")
    features_path.write_text(
        json.dumps(feature_columns, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        {
            "row_id": valid["row_id"].to_numpy(),
            "season": valid["season"].to_numpy(),
            "target": y_valid,
            "pred_constant": constant_prediction,
            "pred_catboost": prediction,
        }
    ).to_csv(predictions_path, index=False)
    metadata = {
        "exp_id": args.exp_id,
        "model_family": "catboost",
        "artifact_role": "validation_candidate",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "catboost": cb.__version__,
        },
        "train_seasons": train_seasons,
        "valid_season": args.valid_season,
        "train_rows": int(len(x_train)),
        "valid_rows": int(len(x_valid)),
        "feature_count": len(feature_columns),
        "feature_contract": "FE-002" if fe002_groups else "FE-001",
        "fe002_groups": fe002_groups,
        "features": feature_columns,
        "categorical_features": categorical_features,
        "categorical_missing_token": "<NA>",
        "hyperparameters": params,
        "early_stopping_rounds": args.early_stopping,
        "best_iteration_index": best_iteration_index,
        "best_rounds": best_rounds,
        "metrics": {
            "constant_baseline": {
                "brier_score": constant_brier,
                "prediction_mean": train_mean,
            },
            "catboost": {
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
