#!/usr/bin/env python3
"""
validate_env.py — 환경 설정 검증 스크립트

실행: python scripts/validate_env.py
"""
import sys
import importlib


def check_python():
    """Python 버전 확인."""
    v = sys.version_info
    print(f"[CHECK] Python: {v.major}.{v.minor}.{v.micro}")
    if v.major != 3 or v.minor < 10:
        print("  ⚠️  Python 3.10+ 권장 (평가 서버: 3.11.15)")
    else:
        print("  ✅ OK")


def check_packages():
    """필수 패키지 설치 확인."""
    required = {
        "pandas": "2.3.3",
        "numpy": None,
        "sklearn": "1.8.0",
        "joblib": "1.5.3",
        "lightgbm": None,
        "xgboost": None,
        "scipy": None,
        "optuna": None,
        "matplotlib": None,
        "seaborn": None,
    }

    optional = {
        "catboost": None,
        "shap": None,
    }

    all_ok = True
    print("\n[CHECK] 필수 패키지:")
    for pkg, expected_ver in required.items():
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "?")
            status = "✅"
            note = ""
            if expected_ver and ver != expected_ver:
                note = f" (기대: {expected_ver})"
                status = "⚠️ "
            print(f"  {status} {pkg:15s} {ver}{note}")
        except ImportError:
            print(f"  ❌ {pkg:15s} — NOT INSTALLED")
            all_ok = False

    print("\n[CHECK] 선택 패키지:")
    for pkg, expected_ver in optional.items():
        try:
            mod = importlib.import_module(pkg)
            ver = getattr(mod, "__version__", "?")
            print(f"  ✅ {pkg:15s} {ver}")
        except ImportError:
            print(f"  ⏭️  {pkg:15s} — 미설치 (선택)")

    return all_ok


def check_data():
    """데이터 파일 존재 확인."""
    import os

    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
    files = {
        "train.csv": True,
        "test.csv": True,
        "sample_submission.csv": True,
        "trackman_history.csv": False,  # optional
    }

    print("\n[CHECK] 데이터 파일:")
    for fname, is_required in files.items():
        path = os.path.join(data_dir, fname)
        exists = os.path.isfile(path)
        if exists:
            size_mb = os.path.getsize(path) / 1e6
            print(f"  ✅ {fname:30s} ({size_mb:.1f} MB)")
        elif is_required:
            print(f"  ❌ {fname:30s} — MISSING (필수)")
        else:
            print(f"  ⏭️  {fname:30s} — 미존재 (선택)")


def check_dirs():
    """프로젝트 디렉토리 구조 확인."""
    import os

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dirs = ["data", "model", "output", "notebooks", "src", "scripts"]

    print("\n[CHECK] 디렉토리 구조:")
    for d in dirs:
        path = os.path.join(root, d)
        exists = os.path.isdir(path)
        status = "✅" if exists else "❌"
        print(f"  {status} {d}/")


def main():
    print("=" * 60)
    print("  LG Aimers Hackathon — 환경 검증")
    print("=" * 60)

    check_python()
    pkg_ok = check_packages()
    check_data()
    check_dirs()

    print("\n" + "=" * 60)
    if pkg_ok:
        print("  🎉 환경 설정 완료! 작업을 시작할 수 있습니다.")
    else:
        print("  ⚠️  일부 패키지가 누락되었습니다. pip install 후 재확인하세요.")
    print("=" * 60)


if __name__ == "__main__":
    main()
