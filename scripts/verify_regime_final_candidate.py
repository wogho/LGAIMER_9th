#!/usr/bin/env python3
"""Verify the isolated REGIME-R-001 final candidate end to end."""

from __future__ import annotations

import hashlib
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_catboost import normalize_categorical  # noqa: E402
from src.features import DTYPE_MAP, TARGET_COL, build_features  # noqa: E402

R_MODEL_DIR = ROOT / "model" / "LGBM-FE001-RONLY-FINAL-2019-2024-R110"
CAT_MODEL_DIR = ROOT / "model" / "CAT-FE001-FINAL-2019-2024-R259"
REGIME_DIR = ROOT / "model" / "REGIME-R-001"
OUTPUT_DIR = ROOT / "model" / "REGIME-R-001-FINAL-E2E"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def peak_memory_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / 1024.0 if sys.platform != "darwin" else value / (1024.0**2)


def load_contract() -> dict[str, Any]:
    regime = json.loads((REGIME_DIR / "regime_results.json").read_text(encoding="utf-8"))
    calibrator = json.loads((REGIME_DIR / "future_global_platt.json").read_text(encoding="utf-8"))
    r_metadata = json.loads((R_MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
    cat_metadata = json.loads((CAT_MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
    feature_columns = json.loads((R_MODEL_DIR / "feature_columns.json").read_text(encoding="utf-8"))
    cat_features = json.loads((CAT_MODEL_DIR / "feature_columns.json").read_text(encoding="utf-8"))
    if regime["selected_candidate"] != "r_lgbm_0.75_catboost_0.25":
        raise ValueError("승인된 REGIME-R 가중치가 아닙니다")
    if feature_columns != cat_features or len(feature_columns) != 60:
        raise ValueError("두 최종 모델의 FE-001 계약이 다릅니다")
    if r_metadata["n_estimators"] != 110 or cat_metadata["iterations"] != 259:
        raise ValueError("최종 모델 tree 계약이 다릅니다")
    if calibrator["source_candidate"] != regime["selected_candidate"]:
        raise ValueError("미래 보정기 source candidate가 다릅니다")
    return {
        "schema_version": 1,
        "experiment_id": "REGIME-R-001-FINAL-E2E",
        "future_season": 2025,
        "feature_count": 60,
        "feature_columns": feature_columns,
        "categorical_features": cat_metadata["categorical_features"],
        "selection_rule": {
            "F": {"catboost": 1.0},
            "R": {"lightgbm_r_only": 0.75, "catboost": 0.25},
        },
        "calibration": {
            "method": "platt_logit",
            "scope": "all_rows",
            "clip_epsilon": 1e-6,
            "coefficient": calibrator["coefficient"],
            "intercept": calibrator["intercept"],
        },
        "tree_counts": {"lightgbm_r_only": 110, "catboost": 259},
        "model_files": {
            "lightgbm_r_only": str(R_MODEL_DIR / "model.txt"),
            "catboost": str(CAT_MODEL_DIR / "model.cbm"),
            "feature_columns": str(R_MODEL_DIR / "feature_columns.json"),
        },
        "model_sha256": {
            "lightgbm_r_only": sha256_file(R_MODEL_DIR / "model.txt"),
            "catboost": sha256_file(CAT_MODEL_DIR / "model.cbm"),
            "feature_columns": sha256_file(R_MODEL_DIR / "feature_columns.json"),
        },
        "source_contract_sha256": {
            "regime_results": sha256_file(REGIME_DIR / "regime_results.json"),
            "future_global_platt": sha256_file(REGIME_DIR / "future_global_platt.json"),
            "r_model_metadata": sha256_file(R_MODEL_DIR / "metadata.json"),
            "cat_model_metadata": sha256_file(CAT_MODEL_DIR / "metadata.json"),
        },
        "test_distribution_used": False,
        "external_data_used": False,
        "active_model_sync": False,
    }


def load_models(contract: dict[str, Any]) -> tuple[lgb.Booster, cb.CatBoostClassifier]:
    r_model = lgb.Booster(model_file=contract["model_files"]["lightgbm_r_only"])
    cat_model = cb.CatBoostClassifier()
    cat_model.load_model(contract["model_files"]["catboost"], format="cbm")
    if r_model.num_trees() != 110 or cat_model.tree_count_ != 259:
        raise ValueError("재로드 모델 tree 수가 계약과 다릅니다")
    return r_model, cat_model


def apply_platt(prediction: np.ndarray, contract: dict[str, Any]) -> np.ndarray:
    calibration = contract["calibration"]
    clipped = np.clip(
        np.asarray(prediction, dtype=np.float64),
        calibration["clip_epsilon"],
        1 - calibration["clip_epsilon"],
    )
    logit = np.log(clipped / (1 - clipped))
    z = calibration["coefficient"] * logit + calibration["intercept"]
    return 1 / (1 + np.exp(-z))


def predict(
    frame: pd.DataFrame,
    contract: dict[str, Any],
    r_model: lgb.Booster,
    cat_model: cb.CatBoostClassifier,
) -> np.ndarray:
    if not frame["season"].eq(2025).all():
        raise ValueError("후보 추론 입력에 2025 이외 시즌이 있습니다")
    game_type = frame["game_type"].astype("string")
    if not set(game_type).issubset({"F", "R"}):
        raise ValueError("지원하지 않는 game_type이 있습니다")
    features = build_features(frame).loc[:, contract["feature_columns"]]
    cat_frame = normalize_categorical(features, contract["categorical_features"])
    cat_pool = cb.Pool(
        cat_frame,
        cat_features=contract["categorical_features"],
        feature_names=contract["feature_columns"],
    )
    pred_cat = cat_model.predict_proba(cat_pool)[:, 1]
    raw = pred_cat.copy()
    r_mask = game_type.eq("R").to_numpy(dtype=bool)
    if r_mask.any():
        lgb_frame = features.loc[r_mask].copy()
        for column in contract["categorical_features"]:
            lgb_frame[column] = lgb_frame[column].astype("category")
        pred_r = r_model.predict(lgb_frame, num_iteration=110)
        raw[r_mask] = 0.75 * pred_r + 0.25 * pred_cat[r_mask]
    calibrated = apply_platt(raw, contract)
    if not np.isfinite(calibrated).all() or not ((calibrated >= 0) & (calibrated <= 1)).all():
        raise ValueError("REGIME-R 후보 확률 범위가 잘못되었습니다")
    return calibrated


def representative_rows() -> pd.DataFrame:
    test = pd.read_csv(
        ROOT / "data" / "test.csv", encoding="utf-8-sig", dtype=DTYPE_MAP
    )
    train_dtypes = dict(DTYPE_MAP)
    train_dtypes[TARGET_COL] = "int8"
    train = pd.read_csv(
        ROOT / "data" / "train.csv", encoding="utf-8-sig", dtype=train_dtypes
    )
    f_rows = train.loc[train["game_type"].eq("F")].head(5).drop(columns=["control_success"])
    r_rows = train.loc[train["game_type"].eq("R")].head(5).drop(columns=["control_success"])
    representative = pd.concat([test, f_rows, r_rows], ignore_index=True)
    representative["season"] = 2025
    representative["row_id"] = [f"VERIFY_{idx:04d}" for idx in range(len(representative))]
    return representative


def main() -> None:
    contract = load_contract()
    r_model, cat_model = load_models(contract)
    representative = representative_rows()
    batch = predict(representative, contract, r_model, cat_model)
    singleton = np.array(
        [
            predict(representative.iloc[[index]], contract, r_model, cat_model)[0]
            for index in range(len(representative))
        ]
    )
    singleton_diff = float(np.max(np.abs(batch - singleton)))
    permutation = representative.sample(frac=1.0, random_state=42)
    permuted_prediction = predict(permutation, contract, r_model, cat_model)
    restored = pd.Series(permuted_prediction, index=permutation["row_id"]).loc[
        representative["row_id"]
    ].to_numpy()
    permutation_diff = float(np.max(np.abs(batch - restored)))
    if singleton_diff > 1e-15 or permutation_diff > 1e-15:
        raise AssertionError("REGIME-R 후보 행 독립성 검사가 실패했습니다")

    test = pd.read_csv(
        ROOT / "data" / "test.csv", encoding="utf-8-sig", dtype=DTYPE_MAP
    )
    repeats = int(np.ceil(245789 / len(test)))
    benchmark = pd.concat([test] * repeats, ignore_index=True).iloc[:245789].copy()
    benchmark["row_id"] = [f"BENCH_{index:07d}" for index in range(len(benchmark))]
    start = time.time()
    benchmark_prediction = predict(benchmark, contract, r_model, cat_model)
    benchmark_seconds = time.time() - start
    if benchmark_seconds > 600 or peak_memory_mb() > 28 * 1024:
        raise RuntimeError("REGIME-R 후보 시간 또는 메모리 제한을 초과했습니다")

    sample = pd.read_csv(ROOT / "data" / "sample_submission.csv", encoding="utf-8-sig")
    test_prediction = predict(test, contract, r_model, cat_model)
    submission = sample[["row_id"]].copy()
    submission["control_success"] = pd.Series(
        test_prediction, index=test["row_id"]
    ).reindex(submission["row_id"]).to_numpy()
    if submission["control_success"].isna().any():
        raise ValueError("REGIME-R sample submission 결합에 실패했습니다")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    contract_path = OUTPUT_DIR / "candidate_contract.json"
    report_path = OUTPUT_DIR / "verification_report.json"
    submission_path = OUTPUT_DIR / "candidate_submission.csv"
    contract_path.write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    submission.to_csv(submission_path, index=False)
    report = {
        "experiment_id": contract["experiment_id"],
        "representative_rows": int(len(representative)),
        "representative_game_types": representative["game_type"].value_counts().to_dict(),
        "singleton_max_absolute_difference": singleton_diff,
        "permutation_max_absolute_difference": permutation_diff,
        "benchmark_rows": int(len(benchmark)),
        "benchmark_seconds": benchmark_seconds,
        "peak_memory_mb": peak_memory_mb(),
        "benchmark_prediction_min": float(benchmark_prediction.min()),
        "benchmark_prediction_max": float(benchmark_prediction.max()),
        "candidate_submission_sha256": sha256_file(submission_path),
        "contract_sha256": sha256_file(contract_path),
        "gate_pass": True,
        "active_model_sync": False,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"report_sha256={sha256_file(report_path)}")


if __name__ == "__main__":
    main()
