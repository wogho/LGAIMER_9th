#!/usr/bin/env python3
"""
build_submission.py — 제출용 zip 빌드 및 격리 환경 E2E 무결성 검증 스크립트

실행:
  python scripts/build_submission.py [--name EXP_NAME]

생성:
  output/submit_{EXP_NAME}.zip
    ├── script.py
    ├── requirements.txt
    └── model/
        ├── lightgbm_model.txt
        ├── catboost_model.cbm
        └── feature_columns.json
        └── ensemble_contract.json
"""
import argparse
import hashlib
import json
import os
import sys
import zipfile
import shutil
import tempfile
import subprocess

from verify_independence import (
    IndependenceError,
    resolve_submission_python,
    verify_submission_independence,
)

REQUIRED_MODEL_FILES = (
    "lightgbm_model.txt",
    "catboost_model.cbm",
    "feature_columns.json",
    "ensemble_contract.json",
)
EXPECTED_ARCHIVE_FILES = {
    "script.py",
    "requirements.txt",
    "model/lightgbm_model.txt",
    "model/catboost_model.cbm",
    "model/feature_columns.json",
    "model/ensemble_contract.json",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_deterministic(zf, archive_name, source_path):
    info = zipfile.ZipInfo(archive_name, date_time=(2020, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    with open(source_path, "rb") as source:
        zf.writestr(info, source.read())


def validate_zip_structure(zip_path):
    """대회 규정에 명시된 submit.zip 디렉토리 구조 검증"""
    errors = []

    with zipfile.ZipFile(zip_path, "r") as zf:
        namelist = zf.namelist()
        actual = set(namelist)
        missing = sorted(EXPECTED_ARCHIVE_FILES - actual)
        unexpected = sorted(actual - EXPECTED_ARCHIVE_FILES)
        if missing:
            errors.append(f"필수 제출 파일 누락: {missing}")
        if unexpected:
            errors.append(f"허용 목록 외 제출 파일 발견: {unexpected}")

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
    parser.add_argument("--name", default="final_selective", help="Experiment name tag")
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
    for model_filename in REQUIRED_MODEL_FILES:
        model_path = os.path.join(model_dir, model_filename)
        if not os.path.isfile(model_path):
            errors.append(f"필수 모델 파일 없음: {model_path}")

    if errors:
        print("❌ 제출 파일 빌드 실패:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    activation_record_path = os.path.join(
        root, "output", "candidates", "selective_activation.json"
    )
    if not os.path.isfile(activation_record_path):
        print(f"❌ 활성 전환 기록 없음: {activation_record_path}")
        sys.exit(1)
    with open(activation_record_path, "r", encoding="utf-8") as file:
        activation_record = json.load(file)
    if activation_record.get("active_submission_sync") is not True:
        print("❌ 선택형 후보가 활성 상태가 아닙니다.")
        sys.exit(1)

    # 2. 고정 순서·timestamp·권한으로 결정론적 Zip 생성
    safe_name = "".join(
        character if character.isalnum() or character in {"-", "_"} else "_"
        for character in args.name
    )
    zip_name = f"submit_{safe_name}.zip"
    zip_path = os.path.join(output_dir, zip_name)
    os.makedirs(output_dir, exist_ok=True)

    sources = {
        "script.py": script_path,
        "requirements.txt": req_path,
    }
    sources.update(
        {
            f"model/{model_filename}": os.path.join(model_dir, model_filename)
            for model_filename in REQUIRED_MODEL_FILES
        }
    )
    if set(sources) != EXPECTED_ARCHIVE_FILES:
        print("❌ 활성 제출 source 목록이 화이트리스트와 다릅니다.")
        sys.exit(1)
    with zipfile.ZipFile(zip_path, "w") as zf:
        for archive_name in sorted(sources):
            add_deterministic(zf, archive_name, sources[archive_name])

    size_mb = os.path.getsize(zip_path) / 1e6
    print(f"✅ 제출 파일 생성 완료: {zip_path}")
    print(f"   용량: {size_mb:.2f} MB (제한: 10,000 MB)")

    if size_mb > 10_000:
        print(f"   ❌ 경고: 10GB 제한 초과! ({size_mb:.0f} MB)")
        sys.exit(1)

    zip_sha256 = sha256_file(zip_path)
    expected_handover_sha256 = activation_record.get("candidate_archive_sha256")
    if zip_sha256 != expected_handover_sha256:
        print(
            "❌ 활성 최종 ZIP이 검증·활성화한 후보 ZIP과 다릅니다: "
            f"{zip_sha256} != {expected_handover_sha256}"
        )
        os.remove(zip_path)
        sys.exit(1)
    print(f"   SHA-256: {zip_sha256}")
    print("   ✅ handover 후보 ZIP과 byte-identical")

    # 3. Zip 구조 검사
    struct_errors = validate_zip_structure(zip_path)
    if struct_errors:
        print("❌ [규격 오류] submit.zip 구조가 대회 규칙과 일치하지 않습니다:")
        for err in struct_errors:
            print(f"  - {err}")
        os.remove(zip_path)
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

    build_record = {
        "schema_version": 1,
        "experiment_id": "ENS-CATF-LGBMCATR5050-FINAL-ACTIVE",
        "archive_path": os.path.relpath(zip_path, root),
        "archive_sha256": zip_sha256,
        "archive_size_bytes": os.path.getsize(zip_path),
        "archive_files": sorted(EXPECTED_ARCHIVE_FILES),
        "handover_candidate_byte_identical": True,
        "sandbox_e2e_pass": True,
        "row_independence_pass": True,
        "active_submission_sync": True,
    }
    build_record_path = os.path.join(output_dir, "final_selective_build.json")
    with open(build_record_path, "w", encoding="utf-8") as file:
        json.dump(build_record, file, ensure_ascii=False, indent=2)
        file.write("\n")
    print(f"   Build record: {build_record_path}")

    print("\n" + "=" * 60)
    print(f"  🎉 최종 제출 zip 빌드 및 무결성 검증 완료: {zip_name}")
    print("=" * 60)


if __name__ == "__main__":
    main()
