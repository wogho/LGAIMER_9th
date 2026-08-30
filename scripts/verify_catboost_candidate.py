#!/usr/bin/env python3
"""Rebuild and verify a CatBoost FE-001 validation candidate."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import catboost as cb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import brier_score, build_features, load_data
from scripts.train_catboost import normalize_categorical


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify CatBoost FE-001 artifacts")
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
    if metadata["model_family"] != "catboost" or metadata["feature_count"] != 60:
        raise ValueError("CatBoost FE-001 metadata 계약이 아닙니다")
    if metadata["features"] != feature_columns or len(feature_columns) != 60:
        raise ValueError("저장 피처 계약이 일치하지 않습니다")

    raw = load_data(str(args.train_path), is_train=True)
    valid = raw.loc[raw["season"].eq(metadata["valid_season"])].copy()
    features = build_features(valid)
    if list(features.columns) != feature_columns:
        raise ValueError("재구축한 검증 피처 순서가 저장 계약과 다릅니다")
    features = normalize_categorical(features, metadata["categorical_features"])
    pool = cb.Pool(
        features,
        cat_features=metadata["categorical_features"],
        feature_names=feature_columns,
    )
    model = cb.CatBoostClassifier()
    model.load_model(args.experiment_dir / metadata["model_file"], format="cbm")
    if model.feature_names_ != feature_columns or model.tree_count_ != metadata[
        "best_rounds"
    ]:
        raise ValueError("CatBoost CBM 모델의 피처 또는 tree 계약이 다릅니다")
    rebuilt = model.predict_proba(pool)[:, 1]
    saved_prediction = stored["pred_catboost"].to_numpy(dtype=np.float64)
    max_prediction_diff = float(np.abs(rebuilt - saved_prediction).max())
    if not np.allclose(rebuilt, saved_prediction, rtol=0.0, atol=1e-15):
        raise AssertionError("저장 검증 예측과 재구축 예측이 다릅니다")
    rebuilt_brier = float(brier_score(stored["target"], rebuilt))
    stored_brier = float(metadata["metrics"]["catboost"]["brier_score"])
    if not np.isclose(rebuilt_brier, stored_brier, rtol=0.0, atol=1e-15):
        raise AssertionError("저장 Brier와 재구축 Brier가 다릅니다")
    if not np.isfinite(rebuilt).all() or not (
        (rebuilt >= 0.0) & (rebuilt <= 1.0)
    ).all():
        raise AssertionError("CatBoost 예측 확률 범위가 잘못되었습니다")
    print(f"PASS native CBM load ({cb.__version__})")
    print(f"PASS FE-001 feature contract ({len(feature_columns)} features)")
    print(f"PASS validation prediction rebuild ({len(valid):,} rows)")
    print(f"PASS prediction exact match (max abs diff={max_prediction_diff:.3e})")
    print(f"PASS stored Brier exact match ({rebuilt_brier:.9f})")


if __name__ == "__main__":
    main()
