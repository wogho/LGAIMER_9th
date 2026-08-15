#!/usr/bin/env python3
"""
build_submission.py — 제출용 zip 빌드 및 격리 환경 E2E 무결성 검증 스크립트

실행:
  python scripts/build_submission.py [--name EXP_NAME]

생성:
  output/submit_{EXP_NAME}_{timestamp}.zip
    ├── script.py
    ├── requirements.txt
    └── model/
        └── (학습된 모델 파일들)
"""
import argparse
import os
import sys
import zipfile
import shutil
import tempfile
import subprocess
from datetime import datetime

from verify_independence import (
    IndependenceError,
    resolve_submission_python,
    verify_submission_independence,
)


def validate_zip_structure(zip_path):
    """대회 규정에 명시된 submit.zip 디렉토리 구조 검증"""
    allowed_roots = {"model/", "script.py", "requirements.txt"}
    errors = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        if "script.py" not in namelist:
            errors.append("최상위에 script.py가 없습니다.")
        if "requirements.txt" not in namelist:
            errors.append("최상위에 requirements.txt가 없습니다.")

        has_model = any(name.startswith("model/") for name in namelist)
        if not has_model:
            errors.append("model/ 디렉토리 또는 모델 가중치 파일이 없습니다.")

        # 최상위 불법 폴더/파일 검사 (예: src/, notebooks/, .git 등 금지)
        for name in namelist:
            root_item = name.split("/")[0]
            if "/" in name:
                if root_item != "model":
                    errors.append(f"허용되지 않은 최상위 디렉토리 발견: '{root_item}/' (대회 규정 위반)")
            else:
                if root_item not in {"script.py", "requirements.txt"}:
                    errors.append(f"허용되지 않은 최상위 파일 발견: '{root_item}'")

    return errors


def test_zip_in_sandbox(
    zip_path,
    original_test_path,
    original_sub_path,
    python_executable,
):
    """임시 격리 디렉토리에서 zip을 압축해제하고 평가 서버와 동일하게 실행 검증"""
    print("\n🧪 [격리 E2E 검증] 임시 샌드박스에서 제출 zip 테스트 중...")
    sandbox_dir = tempfile.mkdtemp(prefix="submit_test_")
    sandbox_data_dir = os.path.join(sandbox_dir, "data")
    sandbox_out_dir = os.path.join(sandbox_dir, "output")
    os.makedirs(sandbox_data_dir, exist_ok=True)
    os.makedirs(sandbox_out_dir, exist_ok=True)

    try:
        # 1. Zip 압축 해제
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(sandbox_dir)

        # 2. 평가 서버가 제공하는 data/ 환경 모사
        shutil.copy2(original_test_path, os.path.join(sandbox_data_dir, "test.csv"))
        shutil.copy2(original_sub_path, os.path.join(sandbox_data_dir, "sample_submission.csv"))

        # 3. script.py 실행
        result = subprocess.run(
            [python_executable, "script.py"],
            cwd=sandbox_dir,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            print(f"❌ 샌드박스 실행 실패 (exit code: {result.returncode})")
            if result.stderr:
                print(f"stderr:\n{result.stderr}")
            return False

        # 4. submission.csv 생성 검증
        out_sub_path = os.path.join(sandbox_out_dir, "submission.csv")
        if not os.path.isfile(out_sub_path):
            print("❌ output/submission.csv 가 생성되지 않았습니다.")
            return False

        import pandas as pd
        sub = pd.read_csv(out_sub_path)
        sample = pd.read_csv(original_sub_path)

        if list(sub.columns) != ["row_id", "control_success"]:
            print(f"❌ 컬럼 불일치: {list(sub.columns)}")
            return False
        if len(sub) != len(sample):
            print(f"❌ 행 수 불일치: {len(sub)} != {len(sample)}")
            return False
        if not (sub["row_id"].values == sample["row_id"].values).all():
            print("❌ row_id 순서 불일치")
            return False
        if sub["control_success"].isna().any():
            print("❌ NaN 결측치 발견")
            return False

        print("  ✅ 샌드박스 E2E 추론 및 출력 규격 검증 통과")

        # 5. 동일 제출물에 대한 평가 행 독립성 필수 검증
        try:
            verify_submission_independence(
                sandbox_dir,
                original_test_path,
                sample_rows=5,
                timeout=120,
                python_executable=python_executable,
            )
        except (IndependenceError, subprocess.TimeoutExpired) as error:
            print(f"❌ 행 독립성 검증 실패: {error}")
            return False

        return True

    finally:
        shutil.rmtree(sandbox_dir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser(description="Build and verify submission zip")
    parser.add_argument("--name", default="exp", help="Experiment name tag")
    parser.add_argument("--script", default="script.py", help="Inference script path")
    parser.add_argument("--requirements", default="requirements_submit.txt", help="Requirements file path (default: lightweight submit version)")
    parser.add_argument("--model-dir", default="model", help="Model directory")
    parser.add_argument("--output-dir", default="output", help="Output directory for zip")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    runtime_python = resolve_submission_python(root)

    script_path = os.path.join(root, args.script)
    req_path = os.path.join(root, args.requirements)
    model_dir = os.path.join(root, args.model_dir)
    output_dir = os.path.join(root, args.output_dir)
    test_path = os.path.join(root, "data", "test.csv")
    sub_path = os.path.join(root, "data", "sample_submission.csv")

    print("=" * 60)
    print("  제출용 ZIP 빌드 및 무결성 검증")
    print("=" * 60)
    print(f"ℹ️ 제출 검증 Python: {runtime_python}")

    # 1. 파일 검증
    errors = []
    if not os.path.isfile(script_path):
        errors.append(f"script.py 없음: {script_path}")
    if not os.path.isfile(req_path):
        errors.append(f"requirements.txt 없음: {req_path}")
    if not os.path.isdir(model_dir) or not os.listdir(model_dir):
        errors.append(f"model/ 디렉토리가 비어있거나 없음: {model_dir}")

    if errors:
        print("❌ 제출 파일 빌드 실패:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    # 2. Zip 생성
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"submit_{args.name}_{ts}.zip"
    zip_path = os.path.join(output_dir, zip_name)
    os.makedirs(output_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(script_path, "script.py")
        zf.write(req_path, "requirements.txt")

        for dirpath, dirnames, filenames in os.walk(model_dir):
            for fn in filenames:
                # 임시 파일이나 숨김 파일 제외
                if fn.startswith(".") or fn.endswith(".tmp"):
                    continue
                full_path = os.path.join(dirpath, fn)
                arcname = os.path.join("model", os.path.relpath(full_path, model_dir))
                zf.write(full_path, arcname)

    size_mb = os.path.getsize(zip_path) / 1e6
    print(f"✅ 제출 파일 생성 완료: {zip_path}")
    print(f"   용량: {size_mb:.2f} MB (제한: 10,000 MB)")

    if size_mb > 10_000:
        print(f"   ❌ 경고: 10GB 제한 초과! ({size_mb:.0f} MB)")
        sys.exit(1)

    # 3. Zip 구조 검사
    struct_errors = validate_zip_structure(zip_path)
    if struct_errors:
        print("❌ [규격 오류] submit.zip 구조가 대회 규칙과 일치하지 않습니다:")
        for err in struct_errors:
            print(f"  - {err}")
        sys.exit(1)

    print("\n   📦 Zip 구조 (검증 완료):")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            print(f"      {info.filename:40s}  {info.file_size / 1e6:.2f} MB")

    # 4. 샌드박스 E2E + 행 독립성 테스트 (제출 빌드 필수 게이트)
    success = test_zip_in_sandbox(
        zip_path,
        test_path,
        sub_path,
        runtime_python,
    )
    if not success:
        print("❌ 제출 검증 실패! 생성한 zip을 삭제합니다.")
        os.remove(zip_path)
        sys.exit(1)

    print("\n" + "=" * 60)
    print(f"  🎉 최종 제출 zip 빌드 및 무결성 검증 완료: {zip_name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
