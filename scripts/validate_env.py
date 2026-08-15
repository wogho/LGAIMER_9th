#!/usr/bin/env python3
"""개발·제출 검증 인프라의 필수 구성요소를 Fail-Fast로 점검한다."""
from __future__ import annotations

import importlib
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def check_python() -> bool:
    version = sys.version_info
    print(f"[CHECK] Python: {version.major}.{version.minor}.{version.micro}")
    ok = version.major == 3 and version.minor >= 11
    print("  ✅ OK" if ok else "  ❌ Python 3.11 이상 필요")
    return ok


def check_packages() -> bool:
    required = [
        "joblib",
        "numpy",
        "pandas",
        "sklearn",
    ]
    optional = [
        "catboost",
        "lightgbm",
        "xgboost",
        "scipy",
        "optuna",
        "matplotlib",
        "seaborn",
        "shap",
    ]

    all_ok = True
    print("\n[CHECK] 필수 추론 패키지:")
    for package in required:
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "?")
            print(f"  ✅ {package:15s} {version}")
        except ImportError:
            print(f"  ❌ {package:15s} — NOT INSTALLED")
            all_ok = False

    print("\n[CHECK] 선택 개발 패키지:")
    for package in optional:
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "?")
            print(f"  ✅ {package:15s} {version}")
        except ImportError:
            print(f"  ⏭️  {package:15s} — 미설치 (선택)")

    return all_ok


def check_submit_runtime() -> bool:
    """평가 서버와 맞춘 Python·패키지 버전을 별도 환경에서 확인."""
    runtime = ROOT / ".venv-submit" / "bin" / "python"
    expected = {
        "python": "3.11.15",
        "pandas": "2.0.3",
        "numpy": "1.26.4",
        "sklearn": "1.8.0",
        "joblib": "1.5.3",
        "scipy": "1.15.3",
    }

    print("\n[CHECK] Python 3.11 제출 검증 환경:")
    if not runtime.is_file():
        print(f"  ❌ {runtime} — MISSING")
        return False

    code = (
        "import sys,pandas,numpy,sklearn,joblib,scipy;"
        "print('|'.join([sys.version.split()[0],pandas.__version__,"
        "numpy.__version__,sklearn.__version__,joblib.__version__,"
        "scipy.__version__]))"
    )
    result = subprocess.run(
        [str(runtime), "-c", code],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"  ❌ 제출 환경 패키지 로드 실패: {result.stderr.strip()}")
        return False

    keys = list(expected)
    actual = dict(zip(keys, result.stdout.strip().split("|")))
    all_ok = True
    for name in keys:
        ok = actual.get(name) == expected[name]
        marker = "✅" if ok else "❌"
        print(f"  {marker} {name:10s} {actual.get(name)} (기대: {expected[name]})")
        all_ok &= ok
    return all_ok


def check_paths() -> bool:
    required_files = [
        "data/train.csv",
        "data/test.csv",
        "data/sample_submission.csv",
        "requirements_submit.txt",
        "script.py",
        "scripts/build_submission.py",
        "scripts/dry_run.py",
        "scripts/verify_features.py",
        "scripts/verify_independence.py",
        "src/features.py",
    ]
    optional_files = [
        "data/trackman_history.csv",
    ]
    required_dirs = [
        "data",
        "model",
        "output",
        "scripts",
        "src",
    ]

    all_ok = True
    print("\n[CHECK] 필수 파일:")
    for relative in required_files:
        path = ROOT / relative
        if path.is_file():
            print(f"  ✅ {relative}")
        else:
            print(f"  ❌ {relative} — MISSING")
            all_ok = False

    print("\n[CHECK] 선택 파일:")
    for relative in optional_files:
        path = ROOT / relative
        print(f"  {'✅' if path.is_file() else '⏭️ '} {relative}")

    print("\n[CHECK] 필수 디렉토리:")
    for relative in required_dirs:
        path = ROOT / relative
        if path.is_dir():
            print(f"  ✅ {relative}/")
        else:
            print(f"  ❌ {relative}/ — MISSING")
            all_ok = False

    model_files = [
        path
        for path in (ROOT / "model").glob("**/*")
        if path.is_file() and not path.name.startswith(".")
    ]
    if model_files:
        print(f"  ✅ model 파일 {len(model_files)}개")
    else:
        print("  ❌ model/에 추론용 모델 파일이 없음")
        all_ok = False

    return all_ok


def main() -> None:
    print("=" * 68)
    print("  LG Aimers — 개발·제출 검증 인프라 점검")
    print("=" * 68)

    checks = [
        check_python(),
        check_packages(),
        check_submit_runtime(),
        check_paths(),
    ]

    print("\n" + "=" * 68)
    if all(checks):
        print("  ✅ 환경 필수 구성요소 ALL PASS")
    else:
        print("  ❌ 환경 검증 실패: 위 필수 항목을 보완하세요.")
        sys.exit(1)
    print("=" * 68)


if __name__ == "__main__":
    main()
