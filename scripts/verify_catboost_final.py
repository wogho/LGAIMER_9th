#!/usr/bin/env python3
"""Verify an isolated full-data CatBoost candidate against its locked policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import catboost as cb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_catboost import normalize_categorical  # noqa: E402
from scripts.train_catboost_final import prediction_sha256, sha256_file  # noqa: E402
from src.features import build_features, load_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify final CatBoost candidate")
    parser.add_argument(
        "experiment_dir",
        type=Path,
        nargs="?",
        default=ROOT / "model" / "CAT-FE001-FINAL-2019-2024-R259",
    )
    parser.add_argument("--train-path", type=Path, default=ROOT / "data" / "train.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = args.experiment_dir / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    features_path = args.experiment_dir / metadata["feature_columns_file"]
    model_path = args.experiment_dir / metadata["model_file"]
    fingerprint_path = (
        args.experiment_dir / metadata["training_prediction_fingerprint_file"]
    )
    feature_columns = json.loads(features_path.read_text(encoding="utf-8"))
    fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))

    if metadata["artifact_role"] != "isolated_final_candidate":
        raise ValueError("격리된 최종 후보 metadata가 아닙니다")
    if metadata["active_model_sync"] is not False:
        raise ValueError("최종 후보가 활성 모델로 동기화되어 있습니다")
    if metadata["validation_set"] is not None or metadata["early_stopping"]:
        raise ValueError("최종 후보에 validation 또는 early stopping이 사용됐습니다")
    if metadata["use_best_model"] or metadata["test_predictions_computed"]:
        raise ValueError("최종 후보 학습 금지 옵션 계약이 깨졌습니다")
    if sha256_file(Path(metadata["policy_file"])) != metadata["policy_sha256"]:
        raise ValueError("반복 수 정책 해시가 metadata와 다릅니다")
    if sha256_file(model_path) != metadata["model_sha256"]:
        raise ValueError("CBM 해시가 metadata와 다릅니다")
    if sha256_file(features_path) != metadata["feature_columns_sha256"]:
        raise ValueError("피처 JSON 해시가 metadata와 다릅니다")
    if sha256_file(fingerprint_path) != metadata["training_prediction_fingerprint_sha256"]:
        raise ValueError("예측 지문 파일 해시가 metadata와 다릅니다")

    raw = load_data(str(args.train_path), is_train=True)
    if sorted(int(value) for value in raw["season"].unique()) != metadata["train_seasons"]:
        raise ValueError("재검증 데이터 시즌이 최종 학습 계약과 다릅니다")
    features = build_features(raw)
    if list(features.columns) != feature_columns or len(feature_columns) != 60:
        raise ValueError("재구축 FE-001 피처 계약이 다릅니다")
    features = normalize_categorical(features, metadata["categorical_features"])
    pool = cb.Pool(
        features,
        cat_features=metadata["categorical_features"],
        feature_names=feature_columns,
    )
    model = cb.CatBoostClassifier()
    model.load_model(model_path, format="cbm")
    if model.tree_count_ != metadata["iterations"] or model.feature_names_ != feature_columns:
        raise ValueError("native CBM tree 또는 피처 계약이 다릅니다")
    prediction = model.predict_proba(pool)[:, 1]
    rebuilt_sha256 = prediction_sha256(prediction)
    if rebuilt_sha256 != fingerprint["prediction_sha256"]:
        raise AssertionError("전체 학습행 예측 지문이 저장 값과 다릅니다")
    if len(prediction) != fingerprint["rows"] or not np.isfinite(prediction).all():
        raise AssertionError("전체 학습행 수 또는 예측 유한성 계약이 다릅니다")
    if not ((prediction >= 0.0) & (prediction <= 1.0)).all():
        raise AssertionError("최종 후보 예측 확률 범위가 잘못되었습니다")

    print(f"PASS policy hash ({metadata['policy_id']})")
    print(f"PASS native CBM load ({cb.__version__}, {model.tree_count_} trees)")
    print(f"PASS FE-001 feature contract ({len(feature_columns)} features)")
    print(f"PASS full training prediction fingerprint ({len(prediction):,} rows)")
    print(f"PASS prediction SHA-256 ({rebuilt_sha256})")
    print("PASS isolated candidate; active model unchanged")


if __name__ == "__main__":
    main()
