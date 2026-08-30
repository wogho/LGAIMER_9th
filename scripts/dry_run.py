#!/usr/bin/env python3
"""
dry_run.py — 로컬 제출 시뮬레이션 (평가 서버 모사 및 스트레스 테스트)

실행:
  python scripts/dry_run.py              # 기본 (data/test.csv 5행 검증)
  python scripts/dry_run.py --benchmark  # 대용량 가상 test (245,789행) 시간 및 메모리 벤치마크

평가 서버와 동일한 방식으로 script.py를 실행하고
시간/메모리/출력 형식을 엄격하게 검증합니다.
"""
import argparse
import os
import sys
import time
import subprocess
import shutil
import tempfile
import pandas as pd
import numpy as np

from verify_independence import (
    IndependenceError,
    resolve_submission_python,
    verify_project_independence,
)

ACTIVE_MODEL_FILES = (
    "lightgbm_model.txt",
    "catboost_model.cbm",
    "feature_columns.json",
    "ensemble_contract.json",
)


def get_process_memory_mb():
    try:
        import resource
        # ru_maxrss is in kilobytes on Linux
        return resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1024.0
    except Exception:
        return 0.0


def validate_submission_df(sub_path, sample_path):
    """제출 결과 파일의 규격 및 무결성 검증 (엄격한 Fail-Fast)"""
    if not os.path.isfile(sub_path):
        print(f"❌ [검증 실패] 출력 파일 없음: {sub_path}")
        return False

    try:
        sub = pd.read_csv(sub_path)
    except Exception as e:
        print(f"❌ [검증 실패] CSV 읽기 오류: {e}")
        return False

    try:
        sample = pd.read_csv(sample_path, encoding="utf-8-sig")
    except Exception as e:
        print(f"❌ [검증 실패] 샘플 제출 파일 읽기 오류: {e}")
        return False

    errors = []

    # 1. 컬럼 검증
    expected_cols = ["row_id", "control_success"]
    if list(sub.columns) != expected_cols:
        errors.append(f"컬럼 불일치: 실제 {list(sub.columns)} != 기대 {expected_cols}")

    # 2. 행 수 검증
    if len(sub) != len(sample):
        errors.append(f"행 수 불일치: 실제 {len(sub)} != 기대 {len(sample)}")

    # 3. row_id 중복 검증
    if sub["row_id"].duplicated().any():
        dup_count = sub["row_id"].duplicated().sum()
        errors.append(f"row_id 중복 {dup_count}건 발견")

    # 4. row_id 순서 및 일치 검증
    if len(sub) == len(sample):
        if not (sub["row_id"].values == sample["row_id"].values).all():
            mismatch_count = (sub["row_id"].values != sample["row_id"].values).sum()
            errors.append(f"row_id 순서/값 불일치 {mismatch_count}건")

    # 5. 확률값 범위 및 NaN/Inf 검증
    preds = sub["control_success"]
    nan_count = preds.isna().sum()
    if nan_count > 0:
        errors.append(f"NaN 결측치 {nan_count}건 존재")

    if not np.issubdtype(preds.dtype, np.number):
        errors.append(f"control_success 컬럼이 수치형이 아님 (dtype: {preds.dtype})")
    else:
        inf_count = np.isinf(preds).sum()
        if inf_count > 0:
            errors.append(f"Inf 무한대 값 {inf_count}건 존재")

        min_val, max_val = preds.min(), preds.max()
        print(f"  예측값 범위: [{min_val:.6f}, {max_val:.6f}]")
        if min_val < 0.0 or max_val > 1.0:
            errors.append(f"확률 범위 [0.0, 1.0] 벗어남: min={min_val}, max={max_val}")

    if errors:
        print(f"\n❌ [검증 실패] 총 {len(errors)}건의 무결성 오류:")
        for err in errors:
            print(f"  - {err}")
        return False

    print("  ✅ 컬럼, 행 수, row_id 순서, 확률 유효 범위 전체 정상")
    return True


def run_benchmark_test(root, script_path, python_executable):
    """train.csv에서 245,789행을 추출하여 평가 서버 규모 벤치마크 수행"""
    train_path = os.path.join(root, "data", "train.csv")
    if not os.path.isfile(train_path):
        print("❌ [벤치마크 실패] data/train.csv 파일이 없습니다.")
        sys.exit(1)

    bench_dir = tempfile.mkdtemp(prefix="lg_bench_")
    bench_data_dir = os.path.join(bench_dir, "data")
    bench_output_dir = os.path.join(bench_dir, "output")
    bench_model_dir = os.path.join(bench_dir, "model")
    os.makedirs(bench_data_dir, exist_ok=True)
    os.makedirs(bench_output_dir, exist_ok=True)

    # 활성 선택형 모델 파일만 복사해 과거 실험 산출물 혼입을 차단한다.
    os.makedirs(bench_model_dir, exist_ok=True)
    for model_filename in ACTIVE_MODEL_FILES:
        source = os.path.join(root, "model", model_filename)
        if not os.path.isfile(source):
            print(f"❌ [벤치마크 실패] 활성 모델 파일이 없습니다: {source}")
            shutil.rmtree(bench_dir, ignore_errors=True)
            sys.exit(1)
        shutil.copy2(source, os.path.join(bench_model_dir, model_filename))
    # script.py 복사
    shutil.copy2(script_path, os.path.join(bench_dir, "script.py"))

    TARGET_ROWS = 245789
    print(f"\n⚙️ 24.5만 행 가상 평가셋 생성 중 ({TARGET_ROWS:,} 행)...")
    df_train = pd.read_csv(train_path, nrows=TARGET_ROWS)

    # test.csv 형식으로 가공 (control_success 제거)
    test_df = df_train.drop(columns=["control_success"], errors="ignore")
    test_df.loc[:, "season"] = 2025
    bench_test_path = os.path.join(bench_data_dir, "test.csv")
    test_df.to_csv(bench_test_path, index=False, encoding="utf-8-sig")

    # sample_submission.csv 생성
    sample_sub = pd.DataFrame({
        "row_id": test_df["row_id"],
        "control_success": 0.5
    })
    bench_sub_path = os.path.join(bench_data_dir, "sample_submission.csv")
    sample_sub.to_csv(bench_sub_path, index=False, encoding="utf-8-sig")

    print(f"🚀 [24.5만 행 벤치마크] script.py 실행 시작...")
    start_time = time.time()
    result = subprocess.run(
        [python_executable, "script.py"],
        cwd=bench_dir,
        capture_output=True,
        text=True,
        timeout=660,
    )
    elapsed = time.time() - start_time
    max_rss = get_process_memory_mb()

    print(f"\n--- stdout ---\n{result.stdout.strip()}")
    if result.stderr:
        print(f"--- stderr ---\n{result.stderr.strip()}")

    if result.returncode != 0:
        print(f"❌ script.py 벤치마크 실패 (exit code: {result.returncode})")
        shutil.rmtree(bench_dir, ignore_errors=True)
        sys.exit(1)

    print(f"\n⏱️  실행 시간: {elapsed:.2f}초 ({elapsed/60:.2f}분) / 제한: 600.0초 (10분)")
    if max_rss > 0:
        print(f"💾 최대 자식 프로세스 메모리(RSS): {max_rss:.1f} MB / 제한: 28,000 MB")

    if elapsed > 600:
        print("❌ [시간 초과] 10분 제한을 초과했습니다!")
        shutil.rmtree(bench_dir, ignore_errors=True)
        sys.exit(1)

    # 출력 검증
    output_path = os.path.join(bench_output_dir, "submission.csv")
    is_valid = validate_submission_df(output_path, bench_sub_path)
    shutil.rmtree(bench_dir, ignore_errors=True)

    if not is_valid:
        print("❌ 벤치마크 출력 검증 실패!")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  🎉 24.5만 행 대규모 스트레스 테스트 통과 (ALL PASS)")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Dry run submission script simulation")
    parser.add_argument("--benchmark", action="store_true", help="Run 245,789 rows full-scale benchmark")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runtime_python = resolve_submission_python(root)
    script_path = os.path.join(root, "script.py")
    test_path = os.path.join(root, "data", "test.csv")
    sample_sub_path = os.path.join(root, "data", "sample_submission.csv")
    output_path = os.path.join(root, "output", "submission.csv")

    print("=" * 60)
    print("  DRY RUN — 제출 시뮬레이션")
    print("=" * 60)
    print(f"ℹ️ 제출 검증 Python: {runtime_python}")

    # 1. 파일 존재 확인
    checks = [
        ("script.py", script_path),
        ("test.csv", test_path),
        ("sample_submission.csv", sample_sub_path),
    ]
    for label, path in checks:
        if not os.path.isfile(path):
            print(f"❌ {label} 없음: {path}")
            sys.exit(1)
        print(f"✅ {label} 확인")

    # 출력/성능 검사와 별개로 실제 script.py의 행 독립성을 항상 확인한다.
    try:
        verify_project_independence(
            root,
            test_path,
            sample_rows=5,
            timeout=120,
            python_executable=runtime_python,
        )
    except (IndependenceError, subprocess.TimeoutExpired) as error:
        print(f"❌ 행 독립성 검증 실패: {error}")
        sys.exit(1)

    if args.benchmark:
        run_benchmark_test(root, script_path, runtime_python)
        return

    # 2. script.py 실행 (기본 모드)
    print(f"\n🚀 script.py 실행 중...")
    start = time.time()
    result = subprocess.run(
        [runtime_python, script_path],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=660,
    )
    elapsed = time.time() - start

    print(f"\n--- stdout ---\n{result.stdout.strip()}")
    if result.stderr:
        print(f"--- stderr ---\n{result.stderr.strip()}")

    if result.returncode != 0:
        print(f"❌ script.py 실행 실패 (exit code: {result.returncode})")
        sys.exit(1)

    print(f"⏱️  실행 시간: {elapsed:.2f}초 ({elapsed/60:.2f}분)")
    if elapsed > 600:
        print("  ❌ 10분 초과 에러!")
        sys.exit(1)

    # 3. 출력 검증 (Fail-Fast)
    print(f"\n[검증] 출력 파일: {output_path}")
    is_valid = validate_submission_df(output_path, sample_sub_path)
    if not is_valid:
        print("❌ DRY RUN 검증 실패!")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("  ✅ DRY RUN 완료 (검증 통과)")
    print("=" * 60)


if __name__ == "__main__":
    main()
