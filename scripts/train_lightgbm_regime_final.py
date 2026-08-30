#!/usr/bin/env python3
"""Train the isolated full-data R-only LightGBM candidate."""

from __future__ import annotations

import hashlib
import json
import platform
import resource
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import TARGET_COL, build_features, load_data  # noqa: E402

POLICY_PATH = ROOT / "model" / "REGIME-R-FINAL-POLICY-001" / "regime_r_policy.json"
OUTPUT_DIR = ROOT / "model" / "LGBM-FE001-RONLY-FINAL-2019-2024-R110"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_sha256(prediction: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(prediction, dtype="<f8").tobytes()).hexdigest()


def peak_memory_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / 1024.0 if sys.platform != "darwin" else value / (1024.0**2)


def main() -> None:
    start = time.time()
    policy = json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    if policy.get("policy_gate_pass") is not True:
        raise ValueError("승인된 REGIME-R 정책이 아닙니다")
    contract = policy["final_training_contract"]
    if contract["game_type_filter"] != "R" or contract["early_stopping"]:
        raise ValueError("R-only 최종 학습 계약이 다릅니다")
    source = Path(policy["selected_source"]["directory"])
    source_metadata_path = source / "metadata.json"
    source_features_path = source / "feature_columns.json"
    if sha256_file(source_metadata_path) != policy["selected_source"]["metadata_sha256"]:
        raise ValueError("정책 source metadata 해시가 다릅니다")
    if sha256_file(source_features_path) != policy["selected_source"]["feature_columns_sha256"]:
        raise ValueError("정책 source 피처 해시가 다릅니다")
    source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
    source_features = json.loads(source_features_path.read_text(encoding="utf-8"))
    trees = int(contract["n_estimators"])

    raw = load_data(str(ROOT / "data" / "train.csv"), is_train=True)
    filtered = raw.loc[raw["game_type"].astype("string").eq("R")].copy()
    target = filtered[TARGET_COL].to_numpy(dtype=np.int8)
    features = build_features(filtered)
    if list(features.columns) != source_features or len(source_features) != 60:
        raise ValueError("최종 R-only FE-001 피처 계약이 다릅니다")
    categorical = source_metadata["categorical_features"]
    for column in categorical:
        features[column] = features[column].astype("category")
    dataset = lgb.Dataset(
        features,
        label=target,
        categorical_feature=categorical,
        free_raw_data=False,
    )
    params = dict(source_metadata["hyperparameters"])
    train_start = time.time()
    booster = lgb.train(
        params,
        dataset,
        num_boost_round=trees,
        callbacks=[lgb.log_evaluation(period=25)],
    )
    training_time = time.time() - train_start
    if booster.num_trees() != trees or booster.feature_name() != source_features:
        raise RuntimeError("최종 R-only 모델 tree/피처 계약이 다릅니다")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = OUTPUT_DIR / "model.txt"
    features_path = OUTPUT_DIR / "feature_columns.json"
    fingerprint_path = OUTPUT_DIR / "training_prediction_fingerprint.json"
    metadata_path = OUTPUT_DIR / "metadata.json"
    booster.save_model(str(model_path), num_iteration=trees)
    features_path.write_text(json.dumps(source_features, ensure_ascii=False, indent=2), encoding="utf-8")
    prediction = booster.predict(features, num_iteration=trees)
    reloaded = lgb.Booster(model_file=str(model_path))
    rebuilt = reloaded.predict(features, num_iteration=trees)
    max_diff = float(np.abs(prediction - rebuilt).max())
    if not np.allclose(prediction, rebuilt, rtol=0.0, atol=1e-15):
        raise AssertionError("R-only native reload 예측이 다릅니다")
    fingerprint = {
        "rows": int(len(prediction)),
        "prediction_sha256": prediction_sha256(prediction),
        "maximum_reload_absolute_difference": max_diff,
        "native_reload_match_within_atol_1e-15": True,
    }
    fingerprint_path.write_text(json.dumps(fingerprint, ensure_ascii=False, indent=2), encoding="utf-8")
    metadata = {
        "experiment_id": "LGBM-FE001-RONLY-FINAL-2019-2024-R110",
        "model_family": "lightgbm",
        "artifact_role": "isolated_regime_final_candidate",
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "lightgbm": lgb.__version__,
        },
        "policy_id": policy["policy_id"],
        "policy_sha256": sha256_file(POLICY_PATH),
        "train_seasons": contract["train_seasons"],
        "game_type_filter": "R",
        "train_rows": int(len(features)),
        "feature_count": 60,
        "categorical_features": categorical,
        "hyperparameters": params,
        "n_estimators": trees,
        "validation_set": None,
        "early_stopping": False,
        "training_time_sec": round(training_time, 2),
        "total_time_sec": round(time.time() - start, 2),
        "peak_memory_mb": round(peak_memory_mb(), 1),
        "model_file": model_path.name,
        "model_sha256": sha256_file(model_path),
        "feature_columns_sha256": sha256_file(features_path),
        "training_prediction_fingerprint_sha256": sha256_file(fingerprint_path),
        "active_model_sync": False,
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"rows={len(features):,} trees={trees} training_time={training_time:.1f}s")
    print(f"model_sha256={metadata['model_sha256']}")
    print(f"saved={OUTPUT_DIR}")
    print("active model unchanged")


if __name__ == "__main__":
    main()
