#!/usr/bin/env python3
"""학습·추론 피처 계약과 피처 단계 행 독립성을 Fail-Fast로 검증한다."""
from __future__ import annotations

import importlib.util
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


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)

    train_path = os.path.join(root, "data", "train.csv")
    test_path = os.path.join(root, "data", "test.csv")
    script_file = os.path.join(root, "script.py")

    print("=" * 68)
    print("  Feature Contract & Row Independence Test")
    print("=" * 68)

    try:
        from src.features import build_features as src_build_features

        script_module = load_script_module(script_file)
        if not hasattr(script_module, "build_features"):
            raise AttributeError("script.py에 build_features()가 없음")
        script_build_features = script_module.build_features

        if os.path.isfile(train_path):
            sample_raw = pd.read_csv(train_path, nrows=50, encoding="utf-8-sig")
            sample_input = sample_raw.drop(
                columns=["control_success"],
                errors="ignore",
            )
        else:
            sample_input = pd.read_csv(test_path, encoding="utf-8-sig")

        if len(sample_input) < 2:
            raise ValueError("피처 불변성 검사에는 최소 2개 행이 필요함")

        source_features = src_build_features(sample_input.copy(deep=True))
        script_features = script_build_features(sample_input.copy(deep=True))

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
