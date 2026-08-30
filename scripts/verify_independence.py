#!/usr/bin/env python3
"""
실제 제출 script.py를 여러 평가 배치에서 반복 실행하여 행 독립성을 검증한다.

검사 변형:
  - 전체 샘플
  - 각 행 단독 입력
  - 행 순서 변경
  - 동일 피처의 무관한 행(새 row_id) 추가

어떤 변형에서도 기존 행의 예측값이 허용 오차 밖으로 달라지면 실패한다.
"""
from __future__ import annotations

import argparse
import ast
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"
EXPECTED_COLUMNS = [ID_COL, TARGET_COL]
SELECTIVE_MODEL_FILES = {
    "lightgbm_model.txt",
    "catboost_model.cbm",
    "feature_columns.json",
    "ensemble_contract.json",
}
HIGH_RISK_CALLS = {
    "cumcount",
    "cummax",
    "cummin",
    "cumprod",
    "cumsum",
    "diff",
    "expanding",
    "fit",
    "fit_transform",
    "groupby",
    "mean",
    "median",
    "mode",
    "nunique",
    "partial_fit",
    "pct_change",
    "pivot_table",
    "quantile",
    "rank",
    "resample",
    "rolling",
    "shift",
    "std",
    "value_counts",
    "var",
}


class IndependenceError(RuntimeError):
    """행 독립성 또는 제출 출력 계약 위반."""


def _canonical_ids(values: pd.Series) -> list[str]:
    return values.astype("string").tolist()


def _assert_no_high_risk_calls(script_path: Path) -> None:
    """추론 스크립트에서 명백한 배치 집계·재학습 호출을 AST로 차단."""
    source = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(script_path))
    findings = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Attribute):
            call_name = node.func.attr
        elif isinstance(node.func, ast.Name):
            call_name = node.func.id
        else:
            continue
        if call_name in HIGH_RISK_CALLS:
            findings.append(f"{call_name}() at line {node.lineno}")

    if findings:
        joined = ", ".join(findings)
        raise IndependenceError(
            "script.py에 평가 배치 참조 또는 추론 중 학습 위험 호출이 있음: "
            f"{joined}"
        )
    print("  ✅ script.py 고위험 배치 연산·재학습 호출 없음")


def _write_case(submission_root: Path, test_df: pd.DataFrame) -> None:
    data_dir = submission_root / "data"
    output_dir = submission_root / "output"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "submission.csv"
    if output_path.exists():
        output_path.unlink()

    test_df.to_csv(data_dir / "test.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            ID_COL: test_df[ID_COL],
            TARGET_COL: np.full(len(test_df), 0.5, dtype=np.float64),
        }
    ).to_csv(
        data_dir / "sample_submission.csv",
        index=False,
        encoding="utf-8-sig",
    )


def _run_case(
    submission_root: Path,
    test_df: pd.DataFrame,
    case_name: str,
    timeout: int,
    python_executable: str,
) -> pd.Series:
    _write_case(submission_root, test_df)

    result = subprocess.run(
        [python_executable, "script.py"],
        cwd=submission_root,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if result.returncode != 0:
        details = result.stderr.strip() or result.stdout.strip()
        raise IndependenceError(
            f"{case_name}: script.py 실패(exit={result.returncode})\n{details}"
        )

    output_path = submission_root / "output" / "submission.csv"
    if not output_path.is_file():
        raise IndependenceError(f"{case_name}: output/submission.csv가 생성되지 않음")

    output = pd.read_csv(output_path, encoding="utf-8-sig")
    if list(output.columns) != EXPECTED_COLUMNS:
        raise IndependenceError(
            f"{case_name}: 출력 컬럼 {list(output.columns)} != {EXPECTED_COLUMNS}"
        )
    if len(output) != len(test_df):
        raise IndependenceError(
            f"{case_name}: 출력 행 수 {len(output)} != 입력 행 수 {len(test_df)}"
        )
    if output[ID_COL].isna().any() or output[ID_COL].duplicated().any():
        raise IndependenceError(f"{case_name}: 출력 row_id에 결측 또는 중복 존재")

    expected_ids = _canonical_ids(test_df[ID_COL])
    actual_ids = _canonical_ids(output[ID_COL])
    if actual_ids != expected_ids:
        raise IndependenceError(f"{case_name}: 출력 row_id 값 또는 순서 불일치")

    predictions = pd.to_numeric(output[TARGET_COL], errors="coerce").to_numpy()
    if not np.isfinite(predictions).all():
        raise IndependenceError(f"{case_name}: 예측에 NaN/Inf 또는 비수치 값 존재")
    if ((predictions < 0.0) | (predictions > 1.0)).any():
        raise IndependenceError(f"{case_name}: 예측 확률이 [0, 1] 범위를 벗어남")

    return pd.Series(
        predictions,
        index=pd.Index(expected_ids, name=ID_COL),
        name=TARGET_COL,
    )


def _new_probe_id(ids: pd.Series):
    if pd.api.types.is_numeric_dtype(ids.dtype):
        used = set(pd.to_numeric(ids, errors="raise").tolist())
        candidate = max(used) + 1 if used else 1
        while candidate in used:
            candidate += 1
        return candidate

    used = set(ids.astype("string"))
    candidate = "__row_independence_probe__"
    suffix = 0
    while candidate in used:
        suffix += 1
        candidate = f"__row_independence_probe_{suffix}__"
    return candidate


def _assert_same(
    baseline: pd.Series,
    candidate: pd.Series,
    ids: list[str],
    case_name: str,
    atol: float,
    rtol: float,
) -> float:
    expected = baseline.loc[ids].to_numpy()
    actual = candidate.loc[ids].to_numpy()
    diff = np.abs(expected - actual)
    max_diff = float(diff.max(initial=0.0))

    if not np.allclose(expected, actual, atol=atol, rtol=rtol):
        worst = int(np.argmax(diff))
        raise IndependenceError(
            f"{case_name}: 행 {ids[worst]}의 예측이 배치 구성에 따라 변경됨 "
            f"(baseline={expected[worst]:.17g}, candidate={actual[worst]:.17g}, "
            f"abs_diff={diff[worst]:.3e})"
        )
    return max_diff


def verify_submission_independence(
    submission_root: str | os.PathLike,
    source_test_path: str | os.PathLike,
    sample_rows: int = 5,
    timeout: int = 120,
    atol: float = 1e-12,
    rtol: float = 1e-12,
    python_executable: str | os.PathLike | None = None,
) -> dict[str, float]:
    """이미 격리된 제출 루트에서 행 독립성 검사를 수행한다."""
    root = Path(submission_root).resolve()
    test_path = Path(source_test_path).resolve()
    runtime_python = str(python_executable or sys.executable)

    script_path = root / "script.py"
    if not script_path.is_file():
        raise IndependenceError(f"script.py 없음: {script_path}")
    if not (root / "model").is_dir():
        raise IndependenceError(f"model/ 없음: {root / 'model'}")
    if not test_path.is_file():
        raise IndependenceError(f"검사용 test.csv 없음: {test_path}")
    if sample_rows < 2:
        raise IndependenceError("sample_rows는 2 이상이어야 함")

    _assert_no_high_risk_calls(script_path)
    source = pd.read_csv(test_path, encoding="utf-8-sig")
    if ID_COL not in source.columns:
        raise IndependenceError(f"검사용 test.csv에 {ID_COL} 컬럼이 없음")
    if TARGET_COL in source.columns:
        source = source.drop(columns=[TARGET_COL])
    if source[ID_COL].isna().any() or source[ID_COL].duplicated().any():
        raise IndependenceError("검사용 test.csv의 row_id에 결측 또는 중복 존재")
    if len(source) < 2:
        raise IndependenceError("행 독립성 검사는 최소 2개 입력 행이 필요함")

    sample = source.iloc[: min(sample_rows, len(source))].copy()
    original_ids = _canonical_ids(sample[ID_COL])

    print(
        f"\n🔒 [행 독립성 검증] 실제 script.py로 {len(sample)}개 대표 행 검사"
    )
    print(f"  ℹ️ 추론 Python: {runtime_python}")
    baseline = _run_case(root, sample, "전체 배치", timeout, runtime_python)

    singleton_values = []
    for position in range(len(sample)):
        one = sample.iloc[[position]].copy()
        one_prediction = _run_case(
            root,
            one,
            f"단독 행 {position + 1}/{len(sample)}",
            timeout,
            runtime_python,
        )
        singleton_values.append(one_prediction.iloc[0])
    singleton = pd.Series(
        singleton_values,
        index=pd.Index(original_ids, name=ID_COL),
        name=TARGET_COL,
    )

    permuted_df = sample.sample(frac=1.0, random_state=42)
    permuted = _run_case(
        root,
        permuted_df,
        "행 순서 변경",
        timeout,
        runtime_python,
    )

    probe = sample.iloc[[0]].copy()
    probe.loc[:, ID_COL] = _new_probe_id(sample[ID_COL])
    augmented_df = pd.concat([sample, probe], ignore_index=True)
    augmented = _run_case(
        root,
        augmented_df,
        "무관한 행 추가",
        timeout,
        runtime_python,
    )

    results = {
        "singleton_max_abs_diff": _assert_same(
            baseline, singleton, original_ids, "각 행 단독 입력", atol, rtol
        ),
        "permutation_max_abs_diff": _assert_same(
            baseline, permuted, original_ids, "행 순서 변경", atol, rtol
        ),
        "augmentation_max_abs_diff": _assert_same(
            baseline, augmented, original_ids, "무관한 행 추가", atol, rtol
        ),
    }

    for name, value in results.items():
        print(f"  ✅ {name}: {value:.3e}")
    print("  ✅ 평가 배치의 크기·구성·순서에 대한 예측 불변성 통과")
    return results


def resolve_submission_python(project_root: str | os.PathLike) -> str:
    """전용 Python 3.11 제출 환경이 있으면 우선 사용한다."""
    root = Path(project_root).resolve()
    candidate = root / ".venv-submit" / "bin" / "python"
    return str(candidate) if candidate.is_file() else sys.executable


def verify_project_independence(
    project_root: str | os.PathLike,
    source_test_path: str | os.PathLike | None = None,
    **kwargs,
) -> dict[str, float]:
    """프로젝트 파일을 임시 디렉토리에 복사한 뒤 안전하게 검사한다."""
    root = Path(project_root).resolve()
    test_path = (
        Path(source_test_path).resolve()
        if source_test_path is not None
        else root / "data" / "test.csv"
    )

    if not kwargs.get("python_executable"):
        kwargs["python_executable"] = resolve_submission_python(root)

    with tempfile.TemporaryDirectory(prefix="lg_row_independence_") as temp_dir:
        sandbox = Path(temp_dir)
        shutil.copy2(root / "script.py", sandbox / "script.py")
        sandbox_model = sandbox / "model"
        sandbox_model.mkdir()
        source_model = root / "model"
        if (source_model / "ensemble_contract.json").is_file():
            missing = [
                name
                for name in sorted(SELECTIVE_MODEL_FILES)
                if not (source_model / name).is_file()
            ]
            if missing:
                raise IndependenceError(
                    f"선택형 활성 모델 파일이 누락됐습니다: {missing}"
                )
            for name in sorted(SELECTIVE_MODEL_FILES):
                shutil.copy2(source_model / name, sandbox_model / name)
        else:
            for name in ("model.txt", "feature_columns.json"):
                source = source_model / name
                if not source.is_file():
                    raise IndependenceError(f"활성 모델 파일이 없습니다: {source}")
                shutil.copy2(source, sandbox_model / name)
        return verify_submission_independence(
            sandbox,
            test_path,
            **kwargs,
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="제출 script.py의 평가 행 독립성 검증"
    )
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1]),
        help="script.py와 model/이 있는 프로젝트 루트",
    )
    parser.add_argument("--test-path", default=None, help="검사용 test.csv 경로")
    parser.add_argument("--sample-rows", type=int, default=5)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--atol", type=float, default=1e-12)
    parser.add_argument("--rtol", type=float, default=1e-12)
    parser.add_argument("--python", default=None, help="추론에 사용할 Python 실행 파일")
    parser.add_argument(
        "--train-probe-rows",
        type=int,
        default=0,
        help="공식 train 선두 N행을 정답 제거·2025 시즌화해 확대 독립성 검사",
    )
    args = parser.parse_args()

    try:
        if args.train_probe_rows:
            if args.train_probe_rows < 2:
                raise IndependenceError("train-probe-rows는 2 이상이어야 합니다")
            root = Path(args.root).resolve()
            train_path = root / "data" / "train.csv"
            probe = pd.read_csv(
                train_path,
                nrows=args.train_probe_rows,
                encoding="utf-8-sig",
            ).drop(columns=[TARGET_COL], errors="ignore")
            if len(probe) != args.train_probe_rows:
                raise IndependenceError(
                    f"확대 검사 행이 부족합니다: {len(probe)} != {args.train_probe_rows}"
                )
            probe.loc[:, "season"] = 2025
            with tempfile.TemporaryDirectory(prefix="lg_independence_probe_") as temp_dir:
                probe_path = Path(temp_dir) / "test_probe.csv"
                probe.to_csv(probe_path, index=False, encoding="utf-8-sig")
                verify_project_independence(
                    root,
                    probe_path,
                    sample_rows=args.train_probe_rows,
                    timeout=args.timeout,
                    atol=args.atol,
                    rtol=args.rtol,
                    python_executable=args.python,
                )
        else:
            verify_project_independence(
                args.root,
                args.test_path,
                sample_rows=args.sample_rows,
                timeout=args.timeout,
                atol=args.atol,
                rtol=args.rtol,
                python_executable=args.python,
            )
    except (IndependenceError, subprocess.TimeoutExpired) as error:
        print(f"❌ 행 독립성 검증 실패: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
