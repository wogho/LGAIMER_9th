#!/usr/bin/env python3
"""
dry_run.py — 로컬 제출 시뮬레이션 (평가 서버 모사)

실행: python scripts/dry_run.py

평가 서버와 동일한 방식으로 script.py를 실행하고
시간/메모리/출력 형식을 검증합니다.
"""
import os
import sys
import time
import subprocess
import pandas as pd
import numpy as np


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    script_path = os.path.join(root, "script.py")
    test_path = os.path.join(root, "data", "test.csv")
    sample_sub_path = os.path.join(root, "data", "sample_submission.csv")
    output_path = os.path.join(root, "output", "submission.csv")

    print("=" * 60)
    print("  DRY RUN — 제출 시뮬레이션")
    print("=" * 60)

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

    # 2. script.py 실행
    print(f"\n🚀 script.py 실행 중...")
    start = time.time()
    result = subprocess.run(
        [sys.executable, script_path],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=660,  # 11분 (10분 + 1분 여유)
    )
    elapsed = time.time() - start

    print(f"\n--- stdout ---\n{result.stdout}")
    if result.stderr:
        print(f"--- stderr ---\n{result.stderr}")

    if result.returncode != 0:
        print(f"❌ script.py 실행 실패 (exit code: {result.returncode})")
        sys.exit(1)

    print(f"⏱️  실행 시간: {elapsed:.1f}초 ({elapsed/60:.1f}분)")
    if elapsed > 600:
        print("  ⚠️ 경고: 10분 초과!")

    # 3. 출력 검증
    if not os.path.isfile(output_path):
        print(f"❌ 출력 파일 없음: {output_path}")
        sys.exit(1)

    sub = pd.read_csv(output_path)
    sample = pd.read_csv(sample_sub_path, encoding="utf-8-sig")

    print(f"\n[검증] 출력 파일: {output_path}")
    print(f"  행 수: {len(sub)} (기대: {len(sample)})")
    print(f"  컬럼: {list(sub.columns)}")

    # 컬럼 확인
    expected_cols = ["row_id", "control_success"]
    if list(sub.columns) != expected_cols:
        print(f"  ❌ 컬럼 불일치! 기대: {expected_cols}")
    else:
        print(f"  ✅ 컬럼 OK")

    # 행 수 확인
    if len(sub) != len(sample):
        print(f"  ❌ 행 수 불일치!")
    else:
        print(f"  ✅ 행 수 OK")

    # row_id 일치
    if set(sub["row_id"]) == set(sample["row_id"]):
        print(f"  ✅ row_id 일치")
    else:
        print(f"  ❌ row_id 불일치!")

    # 확률 범위
    preds = sub["control_success"]
    print(f"  예측값 범위: [{preds.min():.6f}, {preds.max():.6f}]")
    if preds.isna().any():
        print(f"  ❌ NaN 값 {preds.isna().sum()}개 발견!")
    elif preds.min() < 0 or preds.max() > 1:
        print(f"  ❌ [0, 1] 범위 초과!")
    else:
        print(f"  ✅ 확률값 유효")

    print("\n" + "=" * 60)
    print("  ✅ DRY RUN 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()
