#!/usr/bin/env python3
"""Build a deterministic isolated ZIP for the selective ensemble candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT / "scripts"))

from verify_independence import verify_submission_independence  # noqa: E402

EXPECTED_ARCHIVE_FILES = {
    "script.py",
    "requirements.txt",
    "model/lightgbm_model.txt",
    "model/catboost_model.cbm",
    "model/feature_columns.json",
    "model/ensemble_contract.json",
}
REFERENCE_SUBMISSION_PATH = (
    ROOT
    / "model"
    / "ENS-CATF-LGBMCATR5050-FINAL-2025-E2E"
    / "candidate_submission.csv"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build selective candidate ZIP")
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=ROOT / "candidate" / "selective_submission",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "output" / "candidates"
    )
    parser.add_argument(
        "--runtime-python",
        type=Path,
        default=ROOT / ".venv-submit" / "bin" / "python",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def add_deterministic(zf: zipfile.ZipFile, archive_name: str, source: Path) -> None:
    info = zipfile.ZipInfo(archive_name, date_time=(2020, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    zf.writestr(info, source.read_bytes())


def verify_runtime(python: Path) -> dict[str, str]:
    code = (
        "import json,sys,numpy,pandas,lightgbm,catboost;"
        "print(json.dumps({'python':'.'.join(map(str,sys.version_info[:3])),"
        "'numpy':numpy.__version__,'pandas':pandas.__version__,"
        "'lightgbm':lightgbm.__version__,'catboost':catboost.__version__}))"
    )
    result = subprocess.run(
        [str(python), "-c", code], capture_output=True, text=True, check=True
    )
    versions = json.loads(result.stdout)
    expected = {
        "numpy": "1.26.4",
        "pandas": "2.0.3",
        "lightgbm": "4.7.0",
        "catboost": "1.2.10",
    }
    if not versions["python"].startswith("3.11."):
        raise RuntimeError(f"Python 3.11 제출 환경이 아닙니다: {versions}")
    for name, version in expected.items():
        if versions[name] != version:
            raise RuntimeError(f"{name} 버전이 다릅니다: {versions[name]} != {version}")
    return versions


def main() -> None:
    args = parse_args()
    contract_path = args.candidate_dir / "ensemble_contract.json"
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    sources = {
        "script.py": args.candidate_dir / "script.py",
        "requirements.txt": args.candidate_dir / "requirements.txt",
        "model/lightgbm_model.txt": (
            ROOT / "model" / "LGBM-FE001-FINAL-2019-2024-R100" / "model.txt"
        ),
        "model/catboost_model.cbm": (
            ROOT / "model" / "CAT-FE001-FINAL-2019-2024-R259" / "model.cbm"
        ),
        "model/feature_columns.json": (
            ROOT
            / "model"
            / "LGBM-FE001-FINAL-2019-2024-R100"
            / "feature_columns.json"
        ),
        "model/ensemble_contract.json": contract_path,
    }
    if set(sources) != EXPECTED_ARCHIVE_FILES:
        raise RuntimeError("ZIP source 화이트리스트가 예상 구조와 다릅니다")
    hash_targets = {
        "lightgbm": sources["model/lightgbm_model.txt"],
        "catboost": sources["model/catboost_model.cbm"],
        "feature_columns": sources["model/feature_columns.json"],
    }
    for name, source in hash_targets.items():
        if sha256_file(source) != contract["model_sha256"][name]:
            raise RuntimeError(f"{name} source 해시가 앙상블 계약과 다릅니다")
    versions = verify_runtime(args.runtime_python)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    zip_path = args.output_dir / "submit_selective_candidate.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for archive_name in sorted(sources):
            add_deterministic(zf, archive_name, sources[archive_name])
    with zipfile.ZipFile(zip_path, "r") as zf:
        actual = set(zf.namelist())
        if actual != EXPECTED_ARCHIVE_FILES:
            raise RuntimeError(
                f"ZIP 화이트리스트 불일치: missing={EXPECTED_ARCHIVE_FILES-actual}, extra={actual-EXPECTED_ARCHIVE_FILES}"
            )

    with tempfile.TemporaryDirectory(prefix="selective_submit_") as temp_dir:
        sandbox = Path(temp_dir)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(sandbox)
        data_dir = sandbox / "data"
        output_dir = sandbox / "output"
        data_dir.mkdir()
        output_dir.mkdir()
        shutil.copy2(ROOT / "data" / "test.csv", data_dir / "test.csv")
        shutil.copy2(
            ROOT / "data" / "sample_submission.csv",
            data_dir / "sample_submission.csv",
        )
        result = subprocess.run(
            [str(args.runtime_python), "script.py"],
            cwd=sandbox,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip())
        submission_path = output_dir / "submission.csv"
        if not submission_path.is_file():
            raise RuntimeError("격리 E2E에서 submission.csv가 생성되지 않았습니다")
        submission = pd.read_csv(submission_path)
        sample = pd.read_csv(ROOT / "data" / "sample_submission.csv")
        if list(submission.columns) != ["row_id", "control_success"]:
            raise RuntimeError("격리 submission 컬럼이 다릅니다")
        if len(submission) != len(sample) or not submission["row_id"].equals(sample["row_id"]):
            raise RuntimeError("격리 submission 행 수 또는 row_id 순서가 다릅니다")
        prediction = pd.to_numeric(
            submission["control_success"], errors="coerce"
        ).to_numpy()
        if not np.isfinite(prediction).all() or not (
            (prediction >= 0.0) & (prediction <= 1.0)
        ).all():
            raise RuntimeError("격리 submission 확률 범위가 잘못되었습니다")
        reference = pd.read_csv(REFERENCE_SUBMISSION_PATH)
        if not submission["row_id"].equals(reference["row_id"]):
            raise RuntimeError("Python 3.11 출력 row_id가 개발 E2E와 다릅니다")
        reference_prediction = pd.to_numeric(
            reference["control_success"], errors="coerce"
        ).to_numpy()
        prediction_diff = np.abs(prediction - reference_prediction)
        maximum_prediction_difference = float(prediction_diff.max(initial=0.0))
        if not np.allclose(
            prediction, reference_prediction, rtol=0.0, atol=1e-15
        ):
            raise RuntimeError(
                "Python 3.11 확률 배열이 개발 E2E와 다릅니다: "
                f"max_abs_diff={maximum_prediction_difference:.3e}"
            )
        submission_sha256 = sha256_file(submission_path)
        independence = verify_submission_independence(
            sandbox,
            ROOT / "data" / "test.csv",
            sample_rows=5,
            timeout=120,
            atol=1e-12,
            rtol=1e-12,
            python_executable=str(args.runtime_python),
        )

    result_payload = {
        "experiment_id": "ENS-CATF-LGBMCATR5050-FINAL-SUBMIT-CANDIDATE",
        "runtime": versions,
        "archive_files": sorted(EXPECTED_ARCHIVE_FILES),
        "archive_sha256": sha256_file(zip_path),
        "archive_size_bytes": zip_path.stat().st_size,
        "development_reference_submission_sha256": sha256_file(
            REFERENCE_SUBMISSION_PATH
        ),
        "python311_submission_sha256": submission_sha256,
        "python311_vs_development_max_abs_diff": maximum_prediction_difference,
        "row_independence": independence,
        "python311_sandbox_e2e_pass": True,
        "active_submission_sync": False,
    }
    result_path = args.output_dir / "selective_candidate_build.json"
    result_path.write_text(
        json.dumps(result_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result_payload, ensure_ascii=False, indent=2))
    print(f"Saved deterministic ZIP: {zip_path}")
    print("active submission unchanged")


if __name__ == "__main__":
    main()
