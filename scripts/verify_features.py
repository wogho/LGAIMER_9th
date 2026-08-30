#!/usr/bin/env python3
"""학습·추론 피처 계약과 피처 단계 행 독립성을 Fail-Fast로 검증한다."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys

import pandas as pd


def load_script_module(script_file: str):
    spec = importlib.util.spec_from_file_location("submission_script", script_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"script.py 로드 스펙 생성 실패: {script_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def assert_feature_independence(name, build_features, sample_input):
    original = sample_input.copy(deep=True)
    full = build_features(sample_input.copy(deep=True))

    singleton = pd.concat(
        [
            build_features(sample_input.iloc[[position]].copy(deep=True))
            for position in range(len(sample_input))
        ]
    ).loc[full.index]

    permuted_input = sample_input.sample(frac=1.0, random_state=42)
    permuted = build_features(permuted_input.copy(deep=True)).loc[full.index]

    pd.testing.assert_frame_equal(
        full,
        singleton,
        check_dtype=True,
        check_exact=True,
        obj=f"{name}: 전체 배치 vs 단독 행",
    )
    pd.testing.assert_frame_equal(
        full,
        permuted,
        check_dtype=True,
        check_exact=True,
        obj=f"{name}: 전체 배치 vs 순서 변경",
    )
    pd.testing.assert_frame_equal(
        sample_input,
        original,
        check_dtype=True,
        check_exact=True,
        obj=f"{name}: 입력 데이터 변경 여부",
    )
    print(f"  ✅ {name}: 단독 행·순서 변경 피처 불변성 통과")


def parse_args(root: str):
    parser = argparse.ArgumentParser(description="Verify feature and model contracts")
    parser.add_argument(
        "--train-path",
        default=os.path.join(root, "data", "train.csv"),
    )
    parser.add_argument(
        "--test-path",
        default=os.path.join(root, "data", "test.csv"),
    )
    parser.add_argument(
        "--script-file",
        default=os.path.join(root, "script.py"),
    )
    parser.add_argument(
        "--feature-columns",
        default=os.path.join(root, "model", "feature_columns.json"),
    )
    parser.add_argument(
        "--model-path",
        default=os.path.join(root, "model", "lightgbm_model.txt"),
    )
    parser.add_argument(
        "--catboost-model-path",
        default=os.path.join(root, "model", "catboost_model.cbm"),
    )
    parser.add_argument(
        "--ensemble-contract",
        default=os.path.join(root, "model", "ensemble_contract.json"),
    )
    return parser.parse_args()


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)

    args = parse_args(root)
    train_path = args.train_path
    test_path = args.test_path
    script_file = args.script_file
    feature_columns_file = args.feature_columns
    model_file = args.model_path
    catboost_model_file = args.catboost_model_path
    ensemble_contract_file = args.ensemble_contract

    print("=" * 68)
    print("  Feature Contract & Row Independence Test")
    print("=" * 68)

    try:
        from src.features import DTYPE_MAP, build_features as src_build_features

        script_module = load_script_module(script_file)
        if not hasattr(script_module, "build_features"):
            raise AttributeError("script.py에 build_features()가 없음")
        if not hasattr(script_module, "load_feature_columns"):
            raise AttributeError("script.py에 load_feature_columns()가 없음")

        contract = script_module.load_contract(ensemble_contract_file)
        persisted_columns = script_module.load_feature_columns(
            feature_columns_file,
            contract["feature_count"],
        )
        script_build_features = lambda frame: script_module.build_features(
            frame,
            persisted_columns,
        )

        if os.path.isfile(train_path):
            train_dtype = dict(DTYPE_MAP)
            train_dtype["control_success"] = "int8"
            sample_raw = pd.read_csv(
                train_path,
                nrows=50,
                encoding="utf-8-sig",
                dtype=train_dtype,
            )
            sample_input = sample_raw.drop(
                columns=["control_success"],
                errors="ignore",
            )
        else:
            sample_input = pd.read_csv(test_path, encoding="utf-8-sig")

        if len(sample_input) < 2:
            raise ValueError("피처 불변성 검사에는 최소 2개 행이 필요함")

        source_all_features = src_build_features(sample_input.copy(deep=True))
        missing_source = [
            column for column in persisted_columns
            if column not in source_all_features.columns
        ]
        if missing_source:
            raise AssertionError(
                f"src.features에 저장 계약 피처가 없음: {missing_source[:10]}"
            )
        source_features = source_all_features.loc[:, persisted_columns].copy()
        script_features = script_build_features(sample_input.copy(deep=True))

        if list(source_features.columns) != persisted_columns:
            raise AssertionError(
                "src.features의 피처 순서와 feature_columns.json이 다름: "
                f"source={list(source_features.columns)[:5]}, "
                f"contract={persisted_columns[:5]}"
            )

        pd.testing.assert_frame_equal(
            source_features,
            script_features,
            check_dtype=True,
            check_exact=True,
            obj="src/features.py vs script.py",
        )
        print(
            "✅ 학습·추론 피처의 컬럼·순서·dtype·값 일치 "
            f"(rows={len(source_features)}, features={source_features.shape[1]})"
        )

        # 입력 CSV 컬럼 순서가 달라도 저장된 학습 순서로 복원해야 한다.
        shuffled_columns = list(reversed(sample_input.columns))
        shuffled_input = sample_input.loc[:, shuffled_columns]
        shuffled_features = script_build_features(shuffled_input)
        pd.testing.assert_frame_equal(
            script_features,
            shuffled_features,
            check_dtype=True,
            check_exact=True,
            obj="저장 피처 계약 기반 컬럼 순서 복원",
        )
        print("  ✅ 입력 컬럼 순서 변경 시 저장된 학습 피처 순서로 복원")

        # 누락·추가 피처는 조용히 통과하지 않고 즉시 실패해야 한다.
        missing_input = sample_input.drop(columns=[persisted_columns[0]])
        try:
            script_build_features(missing_input)
        except ValueError:
            pass
        else:
            raise AssertionError("누락 피처 입력이 Fail-Fast 되지 않음")

        unexpected_input = sample_input.assign(__unexpected_feature__=0)
        try:
            script_build_features(unexpected_input)
        except ValueError:
            pass
        else:
            raise AssertionError("추가 피처 입력이 Fail-Fast 되지 않음")
        print("  ✅ 누락·추가 피처 Fail-Fast 통과")

        import lightgbm as lgb

        booster = lgb.Booster(model_file=model_file)
        if booster.feature_name() != persisted_columns:
            raise AssertionError("LightGBM 모델과 저장 피처 계약의 이름 또는 순서가 다름")
        if booster.num_feature() != len(persisted_columns):
            raise AssertionError("LightGBM 모델과 저장 피처 계약의 피처 수가 다름")
        if booster.num_trees() != contract["tree_counts"]["lightgbm"]:
            raise AssertionError("LightGBM 모델 tree 수가 앙상블 계약과 다름")
        print("  ✅ LightGBM 모델·tree 수와 저장 피처 계약 일치")

        import catboost as cb

        catboost_model = cb.CatBoostClassifier()
        catboost_model.load_model(catboost_model_file, format="cbm")
        if catboost_model.feature_names_ != persisted_columns:
            raise AssertionError("CatBoost 모델과 저장 피처 계약의 이름 또는 순서가 다름")
        if catboost_model.tree_count_ != contract["tree_counts"]["catboost"]:
            raise AssertionError("CatBoost 모델 tree 수가 앙상블 계약과 다름")
        print("  ✅ CatBoost 모델·tree 수와 저장 피처 계약 일치")

        contract_hashes = contract["model_sha256"]
        actual_hashes = {
            "lightgbm": script_module.sha256_file(model_file),
            "catboost": script_module.sha256_file(catboost_model_file),
            "feature_columns": script_module.sha256_file(feature_columns_file),
        }
        if actual_hashes != contract_hashes:
            raise AssertionError(
                "두 모델·피처 파일 해시가 앙상블 계약과 다름: "
                f"actual={json.dumps(actual_hashes, sort_keys=True)}"
            )
        print("  ✅ 두 native 모델·피처 파일 SHA-256 계약 일치")

        assert_feature_independence(
            "src.features.build_features",
            src_build_features,
            sample_input,
        )
        assert_feature_independence(
            "script.build_features",
            script_build_features,
            sample_input,
        )
    except Exception as error:
        print(f"❌ Feature Contract 검증 실패: {error}")
        sys.exit(1)

    print("=" * 68)
    print("  ✅ Feature Contract & Row Independence ALL PASS")
    print("=" * 68)


if __name__ == "__main__":
    main()
