#!/usr/bin/env python3
"""Verify final ZIP fail-fast behavior with deliberately invalid inputs."""

from __future__ import annotations

import json
import subprocess
import tempfile
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
FINAL_ZIP = ROOT / "output" / "submit_final_selective.zip"
RUNTIME = ROOT / ".venv-submit" / "bin" / "python"
OUTPUT_REPORT = ROOT / "output" / "final_failfast_audit.json"


def run_failure_case(
    case_name: str,
    mutate,
    base_test: pd.DataFrame,
    base_sample: pd.DataFrame,
) -> dict[str, str | bool | int]:
    with tempfile.TemporaryDirectory(prefix=f"failfast_{case_name}_") as temp_dir:
        sandbox = Path(temp_dir)
        with zipfile.ZipFile(FINAL_ZIP, "r") as archive:
            archive.extractall(sandbox)
        data_dir = sandbox / "data"
        output_dir = sandbox / "output"
        data_dir.mkdir()
        output_dir.mkdir()
        test = base_test.copy(deep=True)
        sample = base_sample.copy(deep=True)
        mutate(sandbox, test, sample)
        test.to_csv(data_dir / "test.csv", index=False, encoding="utf-8-sig")
        sample.to_csv(
            data_dir / "sample_submission.csv", index=False, encoding="utf-8-sig"
        )
        result = subprocess.run(
            [str(RUNTIME), "script.py"],
            cwd=sandbox,
            capture_output=True,
            text=True,
            timeout=120,
        )
        output_exists = (output_dir / "submission.csv").exists()
        if result.returncode == 0 or output_exists:
            raise RuntimeError(
                f"{case_name}가 Fail-Fast 되지 않았습니다: "
                f"returncode={result.returncode}, output_exists={output_exists}"
            )
        message = (result.stderr.strip() or result.stdout.strip()).splitlines()
        return {
            "pass": True,
            "returncode": result.returncode,
            "output_created": output_exists,
            "last_error_line": message[-1] if message else "",
        }


def main() -> None:
    if not FINAL_ZIP.is_file() or not RUNTIME.is_file():
        raise RuntimeError("최종 ZIP 또는 Python 3.11 제출 환경이 없습니다")
    base_test = pd.read_csv(ROOT / "data" / "test.csv", encoding="utf-8-sig")
    base_sample = pd.read_csv(
        ROOT / "data" / "sample_submission.csv", encoding="utf-8-sig"
    )

    def test_missing_id(_, test, __):
        test.loc[test.index[0], "row_id"] = None

    def test_duplicate_id(_, test, __):
        test.loc[test.index[-1], "row_id"] = test.loc[test.index[0], "row_id"]

    def sample_duplicate_id(_, __, sample):
        sample.loc[sample.index[-1], "row_id"] = sample.loc[sample.index[0], "row_id"]

    def sample_row_count(_, __, sample):
        sample.drop(index=sample.index[-1], inplace=True)

    def sample_id_set(_, __, sample):
        sample.loc[sample.index[-1], "row_id"] = "__invalid_row_id__"

    def missing_feature(_, test, __):
        test.drop(columns=["li"], inplace=True)

    def extra_feature(_, test, __):
        test["__unexpected_feature__"] = 0

    def invalid_season(_, test, __):
        test.loc[test.index[0], "season"] = 2024

    def invalid_game_type(_, test, __):
        test["game_type"] = test["game_type"].astype("object")
        test.loc[test.index[0], "game_type"] = "X"

    def missing_model(sandbox, _, __):
        (sandbox / "model" / "catboost_model.cbm").unlink()

    def tampered_model(sandbox, _, __):
        model_path = sandbox / "model" / "lightgbm_model.txt"
        content = model_path.read_bytes()
        model_path.write_bytes(content[:-1] + bytes([content[-1] ^ 1]))

    def wrong_sample_columns(_, __, sample):
        sample.rename(columns={"control_success": "prediction"}, inplace=True)

    cases = {
        "test_missing_row_id": test_missing_id,
        "test_duplicate_row_id": test_duplicate_id,
        "sample_duplicate_row_id": sample_duplicate_id,
        "sample_row_count_mismatch": sample_row_count,
        "sample_id_set_mismatch": sample_id_set,
        "missing_raw_feature": missing_feature,
        "unexpected_raw_feature": extra_feature,
        "invalid_future_season": invalid_season,
        "unsupported_game_type": invalid_game_type,
        "missing_model_file": missing_model,
        "tampered_model_hash": tampered_model,
        "wrong_sample_columns": wrong_sample_columns,
    }
    results = {
        name: run_failure_case(name, mutate, base_test, base_sample)
        for name, mutate in cases.items()
    }
    payload = {
        "schema_version": 1,
        "audit_date": "2026-08-16",
        "final_zip": "output/submit_final_selective.zip",
        "case_count": len(results),
        "cases": results,
        "all_failfast_pass": all(item["pass"] for item in results.values()),
    }
    OUTPUT_REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
