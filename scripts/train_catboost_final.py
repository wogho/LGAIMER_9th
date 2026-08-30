#!/usr/bin/env python3
"""Train an isolated full-data CatBoost candidate from an approved policy JSON."""

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

import catboost as cb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_catboost import normalize_categorical  # noqa: E402
from src.features import TARGET_COL, build_features, load_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train final CatBoost candidate with a locked iteration policy"
    )
    parser.add_argument("--train-path", type=Path, default=ROOT / "data" / "train.csv")
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=(
            ROOT
            / "model"
            / "CAT-FE001-FINAL-POLICY-001"
            / "catboost_iteration_policy.json"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "model" / "CAT-FE001-FINAL-2019-2024-R259",
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


def validate_policy(policy: dict[str, Any], policy_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    if policy.get("policy_gate_pass") is not True:
        raise ValueError("승인된 CatBoost 반복 수 정책이 아닙니다")
    if policy.get("model_family") != "catboost" or policy.get("feature_contract") != "FE-001-60":
        raise ValueError("CatBoost FE-001 정책 계약이 아닙니다")
    contract = policy["final_training_contract"]
    if contract["validation_set"] is not None:
        raise ValueError("최종 학습 정책에 validation set이 지정되어 있습니다")
    if contract["early_stopping"] or contract["use_best_model"]:
        raise ValueError("최종 학습 정책은 early stopping/use_best_model을 금지해야 합니다")
    if contract["active_model_sync"]:
        raise ValueError("격리 후보 정책에서 활성 모델 동기화가 허용됐습니다")
    if int(contract["iterations"]) != int(policy["selected_rounds"]):
        raise ValueError("정책 rounds와 최종 학습 계약이 일치하지 않습니다")
    if policy.get("round_scaling_after_full_data") is not False:
        raise ValueError("전체 데이터 rounds 배율 조정이 금지되지 않았습니다")

    selected_source = policy["selected_source"]
    source_dir = Path(selected_source["directory"])
    source_metadata_path = source_dir / "metadata.json"
    source_features_path = source_dir / "feature_columns.json"
    if sha256_file(source_metadata_path) != selected_source["metadata_sha256"]:
        raise ValueError("선택 source metadata 해시가 정책과 다릅니다")
    if sha256_file(source_features_path) != selected_source["feature_columns_sha256"]:
        raise ValueError("선택 source 피처 해시가 정책과 다릅니다")
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    if int(source_metadata["best_rounds"]) != int(policy["selected_rounds"]):
        raise ValueError("선택 source best rounds가 정책과 다릅니다")
    return contract, source_metadata


def main() -> None:
    args = parse_args()
    start = time.time()
    policy = json.loads(args.policy_path.read_text(encoding="utf-8"))
    contract, source_metadata = validate_policy(policy, args.policy_path)
    train_seasons = [int(value) for value in contract["train_seasons"]]
    iterations = int(contract["iterations"])

    print("=" * 76)
    print("CatBoost FE-001 isolated final candidate")
    print(f"policy={policy['policy_id']} train={train_seasons} iterations={iterations}")
    print(f"catboost={cb.__version__} python={platform.python_version()}")
    print("validation=None early_stopping=False use_best_model=False active_sync=False")
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
    features = normalize_categorical(features, categorical_features)
    pool = cb.Pool(
        features,
        label=target,
        cat_features=categorical_features,
        feature_names=feature_columns,
    )

    params = dict(source_metadata["hyperparameters"])
    params["iterations"] = iterations
    model = cb.CatBoostClassifier(**params)
    train_start = time.time()
    model.fit(pool, use_best_model=False, verbose=50)
    training_time = time.time() - train_start
    if model.tree_count_ != iterations or model.feature_names_ != feature_columns:
        raise RuntimeError("학습된 CatBoost tree 또는 피처 계약이 다릅니다")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "model.cbm"
    features_path = args.output_dir / "feature_columns.json"
    fingerprint_path = args.output_dir / "training_prediction_fingerprint.json"
    metadata_path = args.output_dir / "metadata.json"
    model.save_model(model_path, format="cbm")
    features_path.write_text(
        json.dumps(feature_columns, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    prediction = model.predict_proba(pool)[:, 1]
    if not np.isfinite(prediction).all() or not (
        (prediction >= 0.0) & (prediction <= 1.0)
    ).all():
        raise AssertionError("최종 후보 예측 확률 범위가 잘못되었습니다")
    reloaded = cb.CatBoostClassifier()
    reloaded.load_model(model_path, format="cbm")
    rebuilt = reloaded.predict_proba(pool)[:, 1]
    if not np.array_equal(prediction, rebuilt):
        raise AssertionError("저장 CBM 재로드 예측이 학습 직후 예측과 다릅니다")
    fingerprint = {
        "algorithm": "sha256_float64_little_endian_c_order",
        "rows": int(len(prediction)),
        "prediction_sha256": prediction_sha256(prediction),
        "finite_probability_range_pass": True,
        "native_reload_exact_match": True,
        "maximum_reload_absolute_difference": float(np.abs(prediction - rebuilt).max()),
    }
    fingerprint_path.write_text(
        json.dumps(fingerprint, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metadata = {
        "experiment_id": "CAT-FE001-FINAL-2019-2024-R259",
        "model_family": "catboost",
        "artifact_role": "isolated_final_candidate",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "catboost": cb.__version__,
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
        "categorical_missing_token": "<NA>",
        "hyperparameters": params,
        "iterations": iterations,
        "validation_set": None,
        "early_stopping": False,
        "use_best_model": False,
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
        "native_reload_exact_match": True,
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
