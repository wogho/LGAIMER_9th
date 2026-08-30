#!/usr/bin/env python3
"""Build and verify the isolated 2025 selective-ensemble candidate output."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
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
from src.features import build_features, load_data  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify final F=CatBoost, R=50:50 candidate E2E"
    )
    parser.add_argument("--test-path", type=Path, default=ROOT / "data" / "test.csv")
    parser.add_argument(
        "--sample-submission-path",
        type=Path,
        default=ROOT / "data" / "sample_submission.csv",
    )
    parser.add_argument(
        "--lightgbm-dir",
        type=Path,
        default=ROOT / "model" / "LGBM-FE001-FINAL-2019-2024-R100",
    )
    parser.add_argument(
        "--catboost-dir",
        type=Path,
        default=ROOT / "model" / "CAT-FE001-FINAL-2019-2024-R259",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "model" / "ENS-CATF-LGBMCATR5050-FINAL-2025-E2E",
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


def load_candidate_contract(directory: Path, expected_family: str) -> tuple[dict[str, Any], list[str]]:
    metadata_path = directory / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    model_path = directory / metadata["model_file"]
    feature_path = directory / metadata["feature_columns_file"]
    feature_columns = json.loads(feature_path.read_text(encoding="utf-8"))
    if metadata["model_family"] != expected_family:
        raise ValueError(f"예상 model family가 아닙니다: {directory}")
    if metadata["artifact_role"] != "isolated_final_candidate":
        raise ValueError(f"격리된 최종 후보가 아닙니다: {directory}")
    if metadata["train_seasons"] != [2019, 2020, 2021, 2022, 2023, 2024]:
        raise ValueError(f"전체 학습 시즌 계약이 다릅니다: {directory}")
    if metadata["future_season"] != 2025 or metadata["feature_count"] != 60:
        raise ValueError(f"미래 시즌 또는 FE-001 계약이 다릅니다: {directory}")
    if metadata["active_model_sync"] is not False:
        raise ValueError(f"활성 모델로 동기화된 후보입니다: {directory}")
    if sha256_file(model_path) != metadata["model_sha256"]:
        raise ValueError(f"모델 해시가 metadata와 다릅니다: {directory}")
    if sha256_file(feature_path) != metadata["feature_columns_sha256"]:
        raise ValueError(f"피처 해시가 metadata와 다릅니다: {directory}")
    if len(feature_columns) != 60:
        raise ValueError(f"피처 목록이 60개가 아닙니다: {directory}")
    return metadata, feature_columns


def apply_selective_rule(
    game_type: np.ndarray, pred_lightgbm: np.ndarray, pred_catboost: np.ndarray
) -> np.ndarray:
    game_type = np.asarray(game_type, dtype=str)
    pred_lightgbm = np.asarray(pred_lightgbm, dtype=np.float64)
    pred_catboost = np.asarray(pred_catboost, dtype=np.float64)
    if not (len(game_type) == len(pred_lightgbm) == len(pred_catboost)):
        raise ValueError("선택 규칙 입력 길이가 다릅니다")
    unexpected = sorted(set(game_type) - {"F", "R"})
    if unexpected:
        raise ValueError(f"지원하지 않는 game_type입니다: {unexpected}")
    if not np.isfinite(pred_lightgbm).all() or not np.isfinite(pred_catboost).all():
        raise ValueError("단일 모델 예측에 비유한 값이 있습니다")
    prediction = np.where(
        game_type == "F",
        pred_catboost,
        0.5 * pred_lightgbm + 0.5 * pred_catboost,
    )
    if not np.isfinite(prediction).all() or not (
        (prediction >= 0.0) & (prediction <= 1.0)
    ).all():
        raise ValueError("선택형 예측 확률 범위가 잘못되었습니다")
    return prediction


def build_model_features(
    raw: pd.DataFrame,
    feature_columns: list[str],
    categorical_features: list[str],
    family: str,
) -> pd.DataFrame:
    features = build_features(raw)
    if list(features.columns) != feature_columns:
        raise ValueError(f"{family} 재구축 피처 이름·순서가 계약과 다릅니다")
    if family == "lightgbm":
        for column in categorical_features:
            features[column] = features[column].astype("category")
        return features
    return normalize_categorical(features, categorical_features)


def predict_components(
    raw: pd.DataFrame,
    feature_columns: list[str],
    lgbm_metadata: dict[str, Any],
    cat_metadata: dict[str, Any],
    lgbm_model: lgb.Booster,
    cat_model: cb.CatBoostClassifier,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    lgbm_features = build_model_features(
        raw,
        feature_columns,
        lgbm_metadata["categorical_features"],
        "lightgbm",
    )
    cat_features = build_model_features(
        raw,
        feature_columns,
        cat_metadata["categorical_features"],
        "catboost",
    )
    pred_lgbm = np.asarray(
        lgbm_model.predict(lgbm_features, num_iteration=lgbm_metadata["n_estimators"]),
        dtype=np.float64,
    )
    cat_pool = cb.Pool(
        cat_features,
        cat_features=cat_metadata["categorical_features"],
        feature_names=feature_columns,
    )
    pred_cat = np.asarray(cat_model.predict_proba(cat_pool)[:, 1], dtype=np.float64)
    pred_selective = apply_selective_rule(
        raw["game_type"].astype(str).to_numpy(), pred_lgbm, pred_cat
    )
    return pred_lgbm, pred_cat, pred_selective


def render_markdown(contract: dict[str, Any]) -> str:
    checks = contract["checks"]
    lines = [
        "# 최종 선택형 앙상블 2025 test E2E 계약",
        "",
        "- 고정 규칙: `F=CatBoost`, `R=LightGBM 0.5 + CatBoost 0.5`",
        "- test 예측 분포를 이용한 선택·가중치·보정 변경 없음",
        f"- test 행 수: `{contract['test_rows']}`",
        f"- 실제 game_type: `{contract['test_game_type_counts']}`",
        "",
        "| 검증 | 결과 |",
        "|---|---|",
    ]
    for name, value in checks.items():
        lines.append(f"| {name} | `{'PASS' if value else 'FAIL'}` |")
    lines.extend(
        [
            "",
            "## 지문",
            "",
            f"- LightGBM 예측 SHA-256: `{contract['prediction_fingerprints']['lightgbm']}`",
            f"- CatBoost 예측 SHA-256: `{contract['prediction_fingerprints']['catboost']}`",
            f"- 선택형 예측 SHA-256: `{contract['prediction_fingerprints']['selective']}`",
            f"- 후보 submission SHA-256: `{contract['candidate_submission_sha256']}`",
            "",
            "- 실제 test에는 F 행이 없어 F 분기는 고정 입력 단위 테스트로 검증했다.",
            "- 후보 산출물은 활성 submission으로 복사하지 않았다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    lgbm_metadata, lgbm_features = load_candidate_contract(args.lightgbm_dir, "lightgbm")
    cat_metadata, cat_features = load_candidate_contract(args.catboost_dir, "catboost")
    if lgbm_features != cat_features:
        raise ValueError("두 최종 모델의 60피처 계약이 다릅니다")
    if lgbm_metadata["categorical_features"] != cat_metadata["categorical_features"]:
        raise ValueError("두 최종 모델의 categorical 계약이 다릅니다")

    test = load_data(str(args.test_path), is_train=False)
    sample = pd.read_csv(args.sample_submission_path, encoding="utf-8-sig")
    if list(sample.columns) != ["row_id", "control_success"]:
        raise ValueError("sample_submission 스키마가 다릅니다")
    if test["row_id"].duplicated().any() or sample["row_id"].duplicated().any():
        raise ValueError("test 또는 sample_submission row_id가 중복되었습니다")
    if not test["season"].eq(2025).all():
        raise ValueError("test에 2025가 아닌 시즌이 있습니다")
    if set(test["row_id"]) != set(sample["row_id"]):
        raise ValueError("test와 sample_submission row_id 집합이 다릅니다")

    lgbm_model = lgb.Booster(model_file=str(args.lightgbm_dir / lgbm_metadata["model_file"]))
    cat_model = cb.CatBoostClassifier()
    cat_model.load_model(args.catboost_dir / cat_metadata["model_file"], format="cbm")
    if lgbm_model.num_trees() != lgbm_metadata["n_estimators"]:
        raise ValueError("LightGBM tree 수가 metadata와 다릅니다")
    if cat_model.tree_count_ != cat_metadata["iterations"]:
        raise ValueError("CatBoost tree 수가 metadata와 다릅니다")
    if lgbm_model.feature_name() != lgbm_features or cat_model.feature_names_ != cat_features:
        raise ValueError("native model 피처 계약이 다릅니다")

    pred_lgbm, pred_cat, pred_selective = predict_components(
        test,
        lgbm_features,
        lgbm_metadata,
        cat_metadata,
        lgbm_model,
        cat_model,
    )
    game_type = test["game_type"].astype(str).to_numpy()
    f_mask = game_type == "F"
    r_mask = game_type == "R"
    f_identity = bool(
        np.array_equal(pred_selective[f_mask], pred_cat[f_mask])
    )
    r_identity = bool(
        np.array_equal(
            pred_selective[r_mask],
            0.5 * pred_lgbm[r_mask] + 0.5 * pred_cat[r_mask],
        )
    )

    self_test = apply_selective_rule(
        np.array(["F", "R"]),
        np.array([0.2, 0.4]),
        np.array([0.8, 0.6]),
    )
    branch_self_test = bool(np.array_equal(self_test, np.array([0.8, 0.5])))

    single_lgbm: list[float] = []
    single_cat: list[float] = []
    single_selective: list[float] = []
    for index in range(len(test)):
        one_lgbm, one_cat, one_selective = predict_components(
            test.iloc[[index]].copy(),
            lgbm_features,
            lgbm_metadata,
            cat_metadata,
            lgbm_model,
            cat_model,
        )
        single_lgbm.append(float(one_lgbm[0]))
        single_cat.append(float(one_cat[0]))
        single_selective.append(float(one_selective[0]))
    row_independence = bool(
        np.allclose(pred_lgbm, single_lgbm, rtol=0.0, atol=1e-15)
        and np.allclose(pred_cat, single_cat, rtol=0.0, atol=1e-15)
        and np.allclose(pred_selective, single_selective, rtol=0.0, atol=1e-15)
    )

    reversed_test = test.iloc[::-1].copy()
    rev_lgbm, rev_cat, rev_selective = predict_components(
        reversed_test,
        lgbm_features,
        lgbm_metadata,
        cat_metadata,
        lgbm_model,
        cat_model,
    )
    reverse_order = np.argsort(reversed_test.index.to_numpy())
    permutation_invariance = bool(
        np.allclose(pred_lgbm, rev_lgbm[reverse_order], rtol=0.0, atol=1e-15)
        and np.allclose(pred_cat, rev_cat[reverse_order], rtol=0.0, atol=1e-15)
        and np.allclose(pred_selective, rev_selective[reverse_order], rtol=0.0, atol=1e-15)
    )

    prediction_by_id = pd.Series(pred_selective, index=test["row_id"].to_numpy())
    candidate = sample[["row_id"]].copy()
    candidate["control_success"] = candidate["row_id"].map(prediction_by_id)
    schema_pass = bool(
        list(candidate.columns) == ["row_id", "control_success"]
        and len(candidate) == len(sample)
        and candidate["row_id"].equals(sample["row_id"])
        and candidate["control_success"].notna().all()
        and np.isfinite(candidate["control_success"]).all()
        and candidate["control_success"].between(0.0, 1.0).all()
    )

    checks = {
        "source_model_hashes": True,
        "shared_feature_contract": True,
        "future_season_2025": True,
        "selection_branch_self_test": branch_self_test,
        "actual_f_catboost_identity_if_present": f_identity,
        "actual_r_50_50_identity": r_identity,
        "single_row_batch_independence": row_independence,
        "row_permutation_invariance": permutation_invariance,
        "sample_submission_schema_and_order": schema_pass,
        "finite_probability_range": True,
    }
    if not all(checks.values()):
        raise AssertionError(f"E2E 계약 검증 실패: {checks}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = args.output_dir / "candidate_submission.csv"
    fingerprint_path = args.output_dir / "prediction_fingerprint.json"
    contract_path = args.output_dir / "e2e_contract.json"
    report_path = args.output_dir / "e2e_report.md"
    candidate.to_csv(candidate_path, index=False)
    fingerprints = {
        "algorithm": "sha256_float64_little_endian_c_order",
        "lightgbm": prediction_sha256(pred_lgbm),
        "catboost": prediction_sha256(pred_cat),
        "selective": prediction_sha256(pred_selective),
    }
    fingerprint_path.write_text(
        json.dumps(fingerprints, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    contract = {
        "experiment_id": "ENS-CATF-LGBMCATR5050-FINAL-2025-E2E",
        "selection_rule": {
            "F": {"lightgbm": 0.0, "catboost": 1.0},
            "R": {"lightgbm": 0.5, "catboost": 0.5},
        },
        "selection_rule_status": "locked_from_2022_2024_validation_not_changed_on_test",
        "test_prediction_distribution_inspected_for_selection": False,
        "test_prediction_distribution_used_for_weight_or_calibration": False,
        "test_rows": int(len(test)),
        "test_game_type_counts": {
            key: int(value)
            for key, value in test["game_type"].astype(str).value_counts().sort_index().items()
        },
        "actual_f_rows": int(f_mask.sum()),
        "actual_r_rows": int(r_mask.sum()),
        "source_files": {
            "test_sha256": sha256_file(args.test_path),
            "sample_submission_sha256": sha256_file(args.sample_submission_path),
            "lightgbm_model_sha256": lgbm_metadata["model_sha256"],
            "catboost_model_sha256": cat_metadata["model_sha256"],
            "feature_columns_sha256": lgbm_metadata["feature_columns_sha256"],
        },
        "prediction_fingerprints": fingerprints,
        "candidate_submission_sha256": sha256_file(candidate_path),
        "checks": checks,
        "candidate_only_not_active_submission": True,
        "active_model_sync": False,
    }
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path.write_text(render_markdown(contract), encoding="utf-8")
    print(report_path.read_text(encoding="utf-8"))
    print(f"Saved candidate: {candidate_path}")
    print("active submission unchanged")


if __name__ == "__main__":
    main()
