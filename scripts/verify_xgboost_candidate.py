#!/usr/bin/env python3
"""Rebuild and verify an XGBoost FE-001 validation candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import brier_score, build_features, load_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify XGBoost FE-001 artifacts")
    parser.add_argument("experiment_dir", type=Path)
    parser.add_argument("--train-path", type=Path, default=ROOT / "data" / "train.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = json.loads(
        (args.experiment_dir / "metadata.json").read_text(encoding="utf-8")
    )
    feature_columns = json.loads(
        (args.experiment_dir / "feature_columns.json").read_text(encoding="utf-8")
    )
    stored = pd.read_csv(args.experiment_dir / "validation_predictions.csv")
    if metadata["model_family"] != "xgboost" or metadata["feature_count"] != 60:
        raise ValueError("XGBoost FE-001 metadata 계약이 아닙니다")
    if metadata["features"] != feature_columns or len(feature_columns) != 60:
        raise ValueError("저장 피처 계약이 일치하지 않습니다")

    raw = load_data(str(args.train_path), is_train=True)
    train = raw.loc[raw["season"].isin(metadata["train_seasons"])].copy()
    valid = raw.loc[raw["season"].eq(metadata["valid_season"])].copy()
    train_features = build_features(train)
    features = build_features(valid)
    if list(features.columns) != feature_columns:
        raise ValueError("재구축한 검증 피처 순서가 저장 계약과 다릅니다")
    for column in metadata["categorical_features"]:
        train_features[column] = train_features[column].astype("category")
        features[column] = features[column].astype("category")
    train_matrix = xgb.QuantileDMatrix(
        train_features,
        enable_categorical=True,
        max_bin=metadata["hyperparameters"]["max_bin"],
    )
    matrix = xgb.QuantileDMatrix(
        features,
        enable_categorical=True,
        max_bin=metadata["hyperparameters"]["max_bin"],
        ref=train_matrix,
    )
    booster = xgb.Booster()
    booster.load_model(args.experiment_dir / metadata["model_file"])
    if booster.feature_names != feature_columns or len(booster.feature_names) != 60:
        raise ValueError("XGBoost model.json의 피처 계약이 일치하지 않습니다")
    rebuilt = booster.predict(
        matrix, iteration_range=(0, metadata["best_rounds"])
    )
    saved_prediction = stored["pred_xgboost"].to_numpy(dtype=np.float64)
    max_prediction_diff = float(np.abs(rebuilt - saved_prediction).max())
    if not np.allclose(
        rebuilt,
        saved_prediction,
        rtol=0.0,
        atol=5e-8,
    ):
        raise AssertionError("저장 검증 예측과 재구축 예측이 다릅니다")
    rebuilt_brier = float(brier_score(stored["target"], rebuilt))
    stored_brier = float(metadata["metrics"]["xgboost"]["brier_score"])
    brier_diff = abs(rebuilt_brier - stored_brier)
    if not np.isclose(rebuilt_brier, stored_brier, rtol=0.0, atol=5e-8):
        raise AssertionError("저장 Brier와 재구축 Brier가 다릅니다")
    if not np.isfinite(rebuilt).all() or not (
        (rebuilt >= 0.0) & (rebuilt <= 1.0)
    ).all():
        raise AssertionError("XGBoost 예측 확률 범위가 잘못되었습니다")
    print(f"PASS model JSON load ({xgb.__version__})")
    print(f"PASS FE-001 feature contract ({len(feature_columns)} features)")
    print(f"PASS validation prediction rebuild ({len(valid):,} rows)")
    print(f"PASS float32 CSV tolerance (max abs diff={max_prediction_diff:.3e})")
    print(
        f"PASS stored Brier tolerance ({rebuilt_brier:.9f}, "
        f"abs diff={brier_diff:.3e})"
    )


if __name__ == "__main__":
    main()
