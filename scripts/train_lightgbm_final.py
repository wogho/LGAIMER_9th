#!/usr/bin/env python3
"""Train an isolated full-data LightGBM candidate from an approved policy JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import TARGET_COL, build_features, load_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train final LightGBM candidate with a locked iteration policy"
    )
    parser.add_argument("--train-path", type=Path, default=ROOT / "data" / "train.csv")
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=(
            ROOT
            / "model"
            / "LGBM-FE001-FINAL-POLICY-001"
            / "lightgbm_iteration_policy.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "model" / "LGBM-FE001-FINAL-2019-2024-R100",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_sha256(prediction: np.ndarray) -> str:
    canonical = np.asarray(prediction, dtype="<f8")
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def peak_memory_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / 1024.0 if sys.platform != "darwin" else value / (1024.0**2)


def validate_policy(policy: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    if policy.get("policy_gate_pass") is not True:
        raise ValueError("승인된 LightGBM 반복 수 정책이 아닙니다")
    if policy.get("model_family") != "lightgbm" or policy.get("feature_contract") != "FE-001-60":
        raise ValueError("LightGBM FE-001 정책 계약이 아닙니다")
    contract = policy["final_training_contract"]
    if contract["validation_set"] is not None or contract["early_stopping"]:
        raise ValueError("최종 학습 정책은 validation/early stopping을 금지해야 합니다")
    if contract["active_model_sync"]:
        raise ValueError("격리 후보 정책에서 활성 모델 동기화가 허용됐습니다")
    if int(contract["n_estimators"]) != int(policy["selected_iterations"]):
        raise ValueError("정책 iteration과 최종 학습 계약이 일치하지 않습니다")
    if policy.get("iteration_scaling_after_full_data") is not False:
        raise ValueError("전체 데이터 iteration 배율 조정이 금지되지 않았습니다")

    selected_source = policy["selected_source"]
    source_dir = Path(selected_source["directory"])
    metadata_path = source_dir / "metadata.json"
    features_path = source_dir / "feature_columns.json"
    if sha256_file(metadata_path) != selected_source["metadata_sha256"]:
        raise ValueError("선택 source metadata 해시가 정책과 다릅니다")
    if sha256_file(features_path) != selected_source["feature_columns_sha256"]:
        raise ValueError("선택 source 피처 해시가 정책과 다릅니다")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if int(metadata["best_iteration"]) != int(policy["selected_iterations"]):
        raise ValueError("선택 source best iteration이 정책과 다릅니다")
    return contract, metadata


def main() -> None:
    args = parse_args()
    start = time.time()
    policy = json.loads(args.policy_path.read_text(encoding="utf-8"))
    contract, source_metadata = validate_policy(policy)
    train_seasons = [int(value) for value in contract["train_seasons"]]
    iterations = int(contract["n_estimators"])

    print("=" * 76)
    print("LightGBM FE-001 isolated final candidate")
    print(f"policy={policy['policy_id']} train={train_seasons} trees={iterations}")
    print(f"lightgbm={lgb.__version__} python={platform.python_version()}")
    print("validation=None early_stopping=False season_weight=None active_sync=False")
    print("=" * 76)

    raw = load_data(str(args.train_path), is_train=True)
    available_seasons = sorted(int(value) for value in raw["season"].unique())
    if available_seasons != train_seasons:
        raise ValueError(
            f"전체 학습 시즌이 정책과 다릅니다: data={available_seasons}, policy={train_seasons}"
        )
    target = raw[TARGET_COL].to_numpy(dtype=np.int8)
    features = build_features(raw)
    feature_columns = list(features.columns)
    if len(feature_columns) != 60 or feature_columns != source_metadata["features"]:
        raise ValueError("최종 FE-001 60피처 순서가 source 계약과 다릅니다")
    categorical_features = source_metadata["categorical_features"]
    for column in categorical_features:
        features[column] = features[column].astype("category")

    dataset = lgb.Dataset(
        features,
        label=target,
        categorical_feature=categorical_features,
        free_raw_data=False,
    )
    params = dict(source_metadata["hyperparameters"])
    train_start = time.time()
    booster = lgb.train(
        params,
        dataset,
        num_boost_round=iterations,
        callbacks=[lgb.log_evaluation(period=25)],
    )
    training_time = time.time() - train_start
    if booster.num_trees() != iterations or booster.feature_name() != feature_columns:
        raise RuntimeError("학습된 LightGBM tree 또는 피처 계약이 다릅니다")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "model.txt"
    features_path = args.output_dir / "feature_columns.json"
    fingerprint_path = args.output_dir / "training_prediction_fingerprint.json"
    metadata_path = args.output_dir / "metadata.json"
    booster.save_model(str(model_path), num_iteration=iterations)
    features_path.write_text(
        json.dumps(feature_columns, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    prediction = booster.predict(features, num_iteration=iterations)
    if not np.isfinite(prediction).all() or not (
        (prediction >= 0.0) & (prediction <= 1.0)
    ).all():
        raise AssertionError("최종 후보 예측 확률 범위가 잘못되었습니다")
    reloaded = lgb.Booster(model_file=str(model_path))
    rebuilt = reloaded.predict(features, num_iteration=iterations)
    max_reload_diff = float(np.abs(prediction - rebuilt).max())
    if not np.allclose(prediction, rebuilt, rtol=0.0, atol=1e-15):
        raise AssertionError("저장 native model 재로드 예측이 학습 직후 예측과 다릅니다")
    fingerprint = {
        "algorithm": "sha256_float64_little_endian_c_order",
        "rows": int(len(rebuilt)),
        "prediction_sha256": prediction_sha256(rebuilt),
        "finite_probability_range_pass": True,
        "native_reload_match_within_atol_1e-15": True,
        "maximum_reload_absolute_difference": max_reload_diff,
    }
    fingerprint_path.write_text(
        json.dumps(fingerprint, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metadata = {
        "experiment_id": "LGBM-FE001-FINAL-2019-2024-R100",
        "model_family": "lightgbm",
        "artifact_role": "isolated_final_candidate",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "lightgbm": lgb.__version__,
        },
        "policy_id": policy["policy_id"],
        "policy_file": str(args.policy_path),
        "policy_sha256": sha256_file(args.policy_path),
        "selection_method": policy["selection_method"],
        "train_seasons": train_seasons,
        "future_season": int(policy["future_season"]),
        "train_rows": int(len(features)),
        "feature_count": len(feature_columns),
        "categorical_features": categorical_features,
        "excluded_features": [],
        "season_weight_decay": 1.0,
        "sample_weight_used": False,
        "hyperparameters": params,
        "n_estimators": iterations,
        "validation_set": None,
        "early_stopping": False,
        "training_metrics_computed": False,
        "test_predictions_computed": False,
        "training_time_sec": round(training_time, 2),
        "total_time_sec": round(time.time() - start, 2),
        "peak_memory_mb": round(peak_memory_mb(), 1),
        "model_file": model_path.name,
        "model_sha256": sha256_file(model_path),
        "feature_columns_file": features_path.name,
        "feature_columns_sha256": sha256_file(features_path),
        "training_prediction_fingerprint_file": fingerprint_path.name,
        "training_prediction_fingerprint_sha256": sha256_file(fingerprint_path),
        "native_reload_match_within_atol_1e-15": True,
        "active_model_sync": False,
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("=" * 76)
    print(f"trained rows={len(features):,} features={len(feature_columns)} trees={iterations}")
    print(f"training_time={training_time:.1f}s peak_memory={peak_memory_mb():.1f}MB")
    print(f"prediction_fingerprint={fingerprint['prediction_sha256']}")
    print(f"saved={args.output_dir}")
    print("active model unchanged")


if __name__ == "__main__":
    main()
