#!/usr/bin/env python3
"""
build_submission.py — 제출용 zip 빌드 스크립트

실행: python scripts/build_submission.py [--name EXP_NAME]

생성:
  output/submit_{EXP_NAME}_{timestamp}.zip
    ├── script.py
    ├── requirements.txt
    └── model/
        └── (학습된 모델 파일들)
"""
import argparse
import os
import zipfile
import shutil
from datetime import datetime


def main():
    parser = argparse.ArgumentParser(description="Build submission zip")
    parser.add_argument("--name", default="exp", help="Experiment name tag")
    parser.add_argument("--script", default="script.py", help="Inference script path")
    parser.add_argument("--requirements", default="requirements_submit.txt", help="Requirements file path (default: lightweight submit version)")
    parser.add_argument("--model-dir", default="model", help="Model directory")
    parser.add_argument("--output-dir", default="output", help="Output directory for zip")
    args = parser.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    script_path = os.path.join(root, args.script)
    req_path = os.path.join(root, args.requirements)
    model_dir = os.path.join(root, args.model_dir)
    output_dir = os.path.join(root, args.output_dir)

    # 검증
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
        return

    # Zip 생성
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    zip_name = f"submit_{args.name}_{ts}.zip"
    zip_path = os.path.join(output_dir, zip_name)
    os.makedirs(output_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(script_path, "script.py")
        zf.write(req_path, "requirements.txt")

        for dirpath, dirnames, filenames in os.walk(model_dir):
            for fn in filenames:
                full_path = os.path.join(dirpath, fn)
                arcname = os.path.join("model", os.path.relpath(full_path, model_dir))
                zf.write(full_path, arcname)

    size_mb = os.path.getsize(zip_path) / 1e6
    print(f"✅ 제출 파일 생성: {zip_path}")
    print(f"   크기: {size_mb:.1f} MB")

    if size_mb > 10_000:
        print(f"   ⚠️ 경고: 10GB 제한 초과! ({size_mb:.0f} MB)")

    # Zip 내용 확인
    print("\n   📦 Zip 구조:")
    with zipfile.ZipFile(zip_path, "r") as zf:
        for info in zf.infolist():
            print(f"      {info.filename:40s}  {info.file_size / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
