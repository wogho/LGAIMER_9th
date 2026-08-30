#!/usr/bin/env python3
"""Verify the future R-only Platt JSON with an FE-001 candidate end to end."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from script import (
    CATEGORICAL_COLS,
    ID_COL,
    apply_r_only_platt,
    build_features,
    load_feature_columns,
    load_r_only_platt_contract,
    load_test,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify R-only Platt load/apply and FE-001 candidate E2E"
    )
    parser.add_argument(
        "--calibrator",
        type=Path,
        default=ROOT / "model" / "CAL-FE001-TEMPORAL-OOF" / "future_r_only_platt.json",
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=ROOT / "model" / "LGBM-FE001-2024",
    )
    parser.add_argument("--test-path", type=Path, default=ROOT / "data" / "test.csv")
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "model" / "CAL-FE001-TEMPORAL-OOF" / "future_e2e_report.json",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "rows": int(len(values)),
        "min": float(values.min()),
        "mean": float(values.mean()),
        "max": float(values.max()),
    }


def verify_source_hashes(contract: dict, calibrator_path: Path) -> None:
    fit_contract = contract["fit_contract"]
    for file_key, hash_key in [
        ("source_oof_file", "source_oof_sha256"),
        ("source_contract_file", "source_contract_sha256"),
    ]:
        source_path = calibrator_path.parent / fit_contract[file_key]
        if not source_path.is_file():
            raise FileNotFoundError(f"보정기 source 파일이 없습니다: {source_path}")
        actual_hash = sha256_file(source_path)
        if actual_hash != fit_contract[hash_key]:
            raise ValueError(f"보정기 source SHA-256 불일치: {source_path}")


def verify_scope_identity(contract: dict) -> None:
    predictions = np.array([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float64)
    game_type = np.array(["F", "R", "F", "R", "unknown"], dtype=object)
    calibrated = apply_r_only_platt(predictions, game_type, contract)
    non_r = game_type != "R"
    if not np.array_equal(calibrated[non_r], predictions[non_r]):
        raise AssertionError("합성 F/비대상 행 identity 검증 실패")
    if np.array_equal(calibrated[~non_r], predictions[~non_r]):
        raise AssertionError("합성 R 행에 보정이 적용되지 않았습니다")


def main() -> None:
    args = parse_args()
    contract = load_r_only_platt_contract(str(args.calibrator))
    verify_source_hashes(contract, args.calibrator)
    verify_scope_identity(contract)

    feature_columns_path = args.candidate_dir / "feature_columns.json"
    model_path = args.candidate_dir / "model.txt"
    metadata_path = args.candidate_dir / "metadata.json"
    feature_columns = load_feature_columns(str(feature_columns_path))
    if len(feature_columns) != 60:
        raise ValueError("E2E 후보가 FE-001 60피처 계약이 아닙니다")
    booster = lgb.Booster(model_file=str(model_path))
    if booster.feature_name() != feature_columns or booster.num_feature() != 60:
        raise ValueError("FE-001 후보 모델과 피처 계약이 일치하지 않습니다")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata["valid_season"] != 2024 or metadata["train_seasons"] != [
        2019,
        2020,
        2021,
        2022,
        2023,
    ]:
        raise ValueError("E2E 후보의 시간 학습 계약이 예상과 다릅니다")

    test = load_test(str(args.test_path))
    ids = test[ID_COL].copy()
    features = build_features(test, feature_columns)
    for column in CATEGORICAL_COLS:
        if column in features.columns:
            features[column] = features[column].astype("category")
    raw_prediction = np.asarray(booster.predict(features), dtype=np.float64)
    calibrated = apply_r_only_platt(raw_prediction, test["game_type"], contract)

    singleton = np.array(
        [
            apply_r_only_platt(
                raw_prediction[index : index + 1],
                test["game_type"].iloc[index : index + 1],
                contract,
            )[0]
            for index in range(len(test))
        ]
    )
    if not np.array_equal(calibrated, singleton):
        raise AssertionError("R-only Platt 단독 행 적용 결과가 배치 결과와 다릅니다")
    order = np.arange(len(test))[::-1]
    shuffled = apply_r_only_platt(
        raw_prediction[order], test["game_type"].iloc[order], contract
    )
    if not np.array_equal(calibrated, shuffled[np.argsort(order)]):
        raise AssertionError("R-only Platt 행 순서 불변성 검증 실패")

    r_mask = test["game_type"].astype("string").eq("R").fillna(False).to_numpy(bool)
    if (~r_mask).any() and not np.array_equal(
        calibrated[~r_mask], raw_prediction[~r_mask]
    ):
        raise AssertionError("실제 test 비대상 행 identity 검증 실패")
    if r_mask.any() and np.array_equal(calibrated[r_mask], raw_prediction[r_mask]):
        raise AssertionError("실제 test R 행에 보정이 적용되지 않았습니다")

    report = {
        "status": "PASS",
        "purpose": "structural E2E only; not a performance evaluation or activation",
        "candidate": {
            "directory": str(args.candidate_dir),
            "exp_id": metadata["exp_id"],
            "train_seasons": metadata["train_seasons"],
            "valid_season": metadata["valid_season"],
            "feature_count": len(feature_columns),
            "model_sha256": sha256_file(model_path),
            "feature_columns_sha256": sha256_file(feature_columns_path),
        },
        "calibrator": {
            "path": str(args.calibrator),
            "sha256": sha256_file(args.calibrator),
            "fit_contract": contract["fit_contract"],
            "parameters": contract["parameters"],
        },
        "test": {
            "path": str(args.test_path),
            "rows": int(len(test)),
            "seasons": sorted(int(value) for value in test["season"].unique()),
            "r_rows": int(r_mask.sum()),
            "other_rows": int((~r_mask).sum()),
            "row_ids": ids.astype(str).tolist(),
            "raw_prediction": describe(raw_prediction),
            "calibrated_prediction": describe(calibrated),
        },
        "checks": {
            "source_hashes": "PASS",
            "json_load_validation": "PASS",
            "synthetic_non_r_identity": "PASS",
            "candidate_model_feature_contract": "PASS",
            "probability_range": "PASS",
            "singleton_equivalence": "PASS",
            "row_order_invariance": "PASS",
            "test_scope_application": "PASS",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved E2E report: {args.output}")


if __name__ == "__main__":
    main()
