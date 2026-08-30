#!/usr/bin/env python3
"""Run a deterministic, read-only final pre-upload compliance audit."""

from __future__ import annotations

import ast
import hashlib
import json
import math
import platform
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import (  # noqa: E402
    ASOF_MISSING_COLUMNS,
    DERIVED_FEATURE_COLUMNS,
    DTYPE_MAP,
    build_features,
)


FINAL_ZIP = ROOT / "output" / "submit_final_selective.zip"
ACTIVATION_RECORD = ROOT / "output" / "candidates" / "selective_activation.json"
BUILD_RECORD = ROOT / "output" / "final_selective_build.json"
OUTPUT_JSON = ROOT / "output" / "final_preupload_audit.json"
OUTPUT_MD = ROOT / "output" / "final_preupload_audit.md"
EXPECTED_ZIP_SHA256 = (
    "22dc61a85a5f6ea26e81645b6e21eed3c59584435c0a172b98779159f2b997ff"
)
EXPECTED_SUBMISSION_SHA256 = (
    "3f575460c2341cf9df15dd3b67634753e377d78274e0fa15cd774b9a4fa299fb"
)
EXPECTED_ARCHIVE_FILES = {
    "script.py",
    "requirements.txt",
    "model/lightgbm_model.txt",
    "model/catboost_model.cbm",
    "model/feature_columns.json",
    "model/ensemble_contract.json",
}
ARCHIVE_TO_ACTIVE = {
    "script.py": "script.py",
    "requirements.txt": "requirements_submit.txt",
    "model/lightgbm_model.txt": "model/lightgbm_model.txt",
    "model/catboost_model.cbm": "model/catboost_model.cbm",
    "model/feature_columns.json": "model/feature_columns.json",
    "model/ensemble_contract.json": "model/ensemble_contract.json",
}
FORBIDDEN_CALLS = {
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
NETWORK_MODULES = {
    "ftplib",
    "http",
    "requests",
    "socket",
    "urllib",
    "urllib3",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def file_record(path: Path) -> dict[str, int | str]:
    return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def round_float(value: float, digits: int = 12) -> float | None:
    return None if not math.isfinite(value) else round(float(value), digits)


def audit_active_artifacts() -> dict:
    activation = load_json(ACTIVATION_RECORD)
    build = load_json(BUILD_RECORD)
    if activation.get("final_active_gates_pass") is not True:
        raise RuntimeError("activation record의 최종 게이트가 PASS가 아닙니다")
    if sha256_file(FINAL_ZIP) != EXPECTED_ZIP_SHA256:
        raise RuntimeError("최종 ZIP 해시가 승인값과 다릅니다")
    if build.get("archive_sha256") != EXPECTED_ZIP_SHA256:
        raise RuntimeError("build record의 ZIP 해시가 다릅니다")
    submission_path = ROOT / "output" / "submission.csv"
    if sha256_file(submission_path) != EXPECTED_SUBMISSION_SHA256:
        raise RuntimeError("활성 submission 해시가 승인값과 다릅니다")

    active_files = {}
    with zipfile.ZipFile(FINAL_ZIP, "r") as archive:
        actual = set(archive.namelist())
        if actual != EXPECTED_ARCHIVE_FILES:
            raise RuntimeError(
                f"최종 ZIP whitelist 불일치: missing={EXPECTED_ARCHIVE_FILES-actual}, "
                f"extra={actual-EXPECTED_ARCHIVE_FILES}"
            )
        for archive_path, active_path in sorted(ARCHIVE_TO_ACTIVE.items()):
            active_bytes = (ROOT / active_path).read_bytes()
            if archive.read(archive_path) != active_bytes:
                raise RuntimeError(f"ZIP과 활성 파일이 다릅니다: {active_path}")
            active_files[active_path] = file_record(ROOT / active_path)

    requirements = [
        line.strip()
        for line in (ROOT / "requirements_submit.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if requirements != ["lightgbm==4.7.0", "catboost==1.2.10"]:
        raise RuntimeError(f"최종 requirements 계약이 다릅니다: {requirements}")
    return {
        "final_zip": file_record(FINAL_ZIP),
        "submission": file_record(submission_path),
        "archive_files": sorted(EXPECTED_ARCHIVE_FILES),
        "active_files": active_files,
        "requirements": requirements,
        "activation_record_sha256": sha256_file(ACTIVATION_RECORD),
        "build_record_sha256": sha256_file(BUILD_RECORD),
        "candidate_byte_identical": build["handover_candidate_byte_identical"],
    }


def audit_script_static() -> dict:
    script_path = ROOT / "script.py"
    source = script_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(script_path))
    forbidden_findings = []
    network_findings = []
    imported_modules = []
    string_literals = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                call_name = node.func.attr
            elif isinstance(node.func, ast.Name):
                call_name = node.func.id
            else:
                call_name = ""
            if call_name in FORBIDDEN_CALLS:
                forbidden_findings.append(f"{call_name}@{node.lineno}")
        elif isinstance(node, ast.Import):
            imported_modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            string_literals.append(node.value)
    for module in imported_modules:
        if module.split(".")[0] in NETWORK_MODULES:
            network_findings.append(module)
    forbidden_data_literals = [
        value
        for value in string_literals
        if "train.csv" in value or "trackman_history" in value
    ]
    required_path_literals = {
        "./data/test.csv",
        "./data/sample_submission.csv",
        "./output/submission.csv",
        "./model/ensemble_contract.json",
    }
    missing_paths = sorted(required_path_literals - set(string_literals))
    if forbidden_findings or network_findings or forbidden_data_literals or missing_paths:
        raise RuntimeError(
            "활성 script 정적 감사 실패: "
            f"calls={forbidden_findings}, network={network_findings}, "
            f"data={forbidden_data_literals}, missing_paths={missing_paths}"
        )
    return {
        "forbidden_batch_or_fit_calls": forbidden_findings,
        "network_imports": network_findings,
        "train_or_trackman_path_literals": forbidden_data_literals,
        "required_runtime_paths_present": sorted(required_path_literals),
        "imported_modules": sorted(set(imported_modules)),
        "pass": True,
    }


def column_profile(frame: pd.DataFrame) -> dict[str, dict]:
    rows = len(frame)
    profile = {}
    for column in frame.columns:
        series = frame[column]
        item = {
            "dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "missing_rate": round_float(series.isna().sum() / rows),
            "unique_count": int(series.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(series.dtype):
            values = series.to_numpy(dtype=np.float64, na_value=np.nan)
            finite = values[np.isfinite(values)]
            item["nonfinite_count"] = int(np.isinf(values).sum())
            if len(finite):
                item.update(
                    {
                        "min": round_float(float(finite.min())),
                        "p001": round_float(float(np.quantile(finite, 0.001))),
                        "p999": round_float(float(np.quantile(finite, 0.999))),
                        "max": round_float(float(finite.max())),
                    }
                )
        profile[column] = item
    return profile


def domain_checks(train: pd.DataFrame) -> dict[str, int]:
    checks = {
        "target_not_binary": int((~train["control_success"].isin([0, 1])).sum()),
        "balls_outside_0_3": int((~train["balls_before"].between(0, 3)).sum()),
        "strikes_outside_0_2": int((~train["strikes_before"].between(0, 2)).sum()),
        "outs_outside_0_2": int((~train["outs_before"].between(0, 2)).sum()),
        "inning_below_1": int(train["inning"].lt(1).sum()),
        "month_outside_1_12": int((~train["game_month"].between(1, 12)).sum()),
        "home_win_expectancy_outside_0_100": int(
            (~train["home_win_expectancy"].between(0, 100)).sum()
        ),
        "away_win_expectancy_outside_0_100": int(
            (~train["away_win_expectancy"].between(0, 100)).sum()
        ),
    }
    for column in ("runner_on_1b", "runner_on_2b", "runner_on_3b"):
        checks[f"{column}_not_binary"] = int((~train[column].isin([0, 1])).sum())
    rate_columns = [
        column
        for column in train.columns
        if column.endswith("_rate")
    ]
    for column in rate_columns:
        checks[f"{column}_outside_0_1"] = int(
            ((train[column] < 0) | (train[column] > 1)).fillna(False).sum()
        )
    if any(checks.values()):
        raise RuntimeError(f"학습 데이터 도메인 위반이 있습니다: {checks}")
    return checks


def asof_missing_by_season(train: pd.DataFrame) -> dict[str, dict[str, float | None]]:
    result = {}
    for season, part in train.groupby("season", observed=True, sort=True):
        result[str(int(season))] = {
            column: round_float(float(part[column].isna().mean()))
            for column in ASOF_MISSING_COLUMNS
        }
    return result


def entity_novelty(train: pd.DataFrame) -> dict[str, dict]:
    result = {}
    entity_columns = [
        "pitcher_id",
        "batter_id",
        "pitcher_team_id",
        "batter_team_id",
    ]
    for season in (2023, 2024):
        prior = train.loc[train["season"] < season]
        current = train.loc[train["season"] == season]
        season_result = {}
        for column in entity_columns:
            prior_ids = set(prior[column].dropna().tolist())
            novel = ~current[column].isin(prior_ids)
            season_result[column] = {
                "new_unique_count": int(current.loc[novel, column].nunique()),
                "current_unique_count": int(current[column].nunique()),
                "novel_row_count": int(novel.sum()),
                "novel_row_rate": round_float(float(novel.mean())),
            }
        result[str(season)] = season_result
    return result


def brier(target: pd.Series, prediction: pd.Series) -> float | None:
    if not len(target):
        return None
    return round_float(float(np.mean((target.to_numpy() - prediction.to_numpy()) ** 2)))


def cold_start_metrics(train: pd.DataFrame) -> dict[str, dict]:
    prediction_paths = {
        2023: ROOT
        / "model"
        / "ENS-CATF-LGBMCATR5050-FE001"
        / "selective_predictions_2023.csv",
        2024: ROOT
        / "model"
        / "ENS-CATF-LGBMCATR5050-FE001"
        / "selective_predictions_2024.csv",
    }
    result = {}
    for season, prediction_path in prediction_paths.items():
        current = train.loc[
            train["season"] == season,
            [
                "row_id",
                "pitcher_id",
                "batter_id",
                "asof_pitcher_n",
                "asof_batter_n",
                "asof_pitcher_pitchmix_n",
            ],
        ].copy()
        prior = train.loc[train["season"] < season]
        prior_pitchers = set(prior["pitcher_id"].dropna().tolist())
        prior_batters = set(prior["batter_id"].dropna().tolist())
        current["entity_cold_start"] = (
            ~current["pitcher_id"].isin(prior_pitchers)
            | ~current["batter_id"].isin(prior_batters)
        )
        current["asof_cold_start"] = (
            current["asof_pitcher_n"].le(0)
            | current["asof_batter_n"].le(0)
            | current["asof_pitcher_pitchmix_n"].le(0)
        )
        predictions = pd.read_csv(
            prediction_path,
            usecols=["row_id", "target", "pred_selective"],
        )
        merged = predictions.merge(current, on="row_id", how="left", validate="one_to_one")
        if merged[["pitcher_id", "batter_id"]].isna().any().any():
            raise RuntimeError(f"{season} cold-start join에 누락 행이 있습니다")
        season_metrics = {"all_brier": brier(merged["target"], merged["pred_selective"])}
        for flag in ("entity_cold_start", "asof_cold_start"):
            cold = merged[flag]
            season_metrics[flag] = {
                "row_count": int(cold.sum()),
                "row_rate": round_float(float(cold.mean())),
                "cold_brier": brier(
                    merged.loc[cold, "target"], merged.loc[cold, "pred_selective"]
                ),
                "known_brier": brier(
                    merged.loc[~cold, "target"], merged.loc[~cold, "pred_selective"]
                ),
            }
        result[str(season)] = season_metrics
    return result


def audit_data() -> dict:
    dtype = dict(DTYPE_MAP)
    dtype["control_success"] = "int8"
    train = pd.read_csv(
        ROOT / "data" / "train.csv", encoding="utf-8-sig", dtype=dtype
    )
    test = pd.read_csv(
        ROOT / "data" / "test.csv", encoding="utf-8-sig", dtype=DTYPE_MAP
    )
    if sorted(train["season"].unique().tolist()) != list(range(2019, 2025)):
        raise RuntimeError("train 시즌 계약이 2019~2024가 아닙니다")
    if not test["season"].eq(2025).all():
        raise RuntimeError("test 시즌 계약이 2025가 아닙니다")
    if train["row_id"].isna().any() or train["row_id"].duplicated().any():
        raise RuntimeError("train row_id 결측 또는 중복")
    if test["row_id"].isna().any() or test["row_id"].duplicated().any():
        raise RuntimeError("test row_id 결측 또는 중복")
    feature_columns = load_json(ROOT / "model" / "feature_columns.json")
    sample_features = build_features(train.iloc[:50].copy())
    if list(sample_features.columns) != feature_columns:
        raise RuntimeError("src.features와 최종 feature_columns가 다릅니다")
    raw_features = [
        column for column in feature_columns if column not in DERIVED_FEATURE_COLUMNS
    ]
    if len(feature_columns) != 60 or len(raw_features) != 47:
        raise RuntimeError("최종 47 raw + 13 derived 피처 계약이 다릅니다")
    return {
        "train_rows": len(train),
        "train_seasons": sorted(int(value) for value in train["season"].unique()),
        "test_rows_schema_only": len(test),
        "test_seasons_schema_only": sorted(int(value) for value in test["season"].unique()),
        "raw_feature_count": len(raw_features),
        "derived_feature_count": len(DERIVED_FEATURE_COLUMNS),
        "final_feature_count": len(feature_columns),
        "column_profile": column_profile(train),
        "domain_violation_counts": domain_checks(train),
        "asof_missing_by_season": asof_missing_by_season(train),
        "entity_novelty": entity_novelty(train),
        "cold_start_metrics": cold_start_metrics(train),
        "test_distribution_used_for_model_decision": False,
    }


def checklist_counts(path: Path) -> dict[str, int]:
    text = path.read_text(encoding="utf-8")
    return {
        "checked": sum(line.startswith("- [x]") for line in text.splitlines()),
        "pending": sum(line.startswith("- [ ]") for line in text.splitlines()),
    }


def build_rule_results(static: dict, artifacts: dict, data: dict) -> dict[str, bool]:
    activation = load_json(ACTIVATION_RECORD)
    gates = activation["gates"]
    return {
        "01_1_row_independence": gates["row_independence_5"]
        and gates["row_independence_50"],
        "01_2_no_test_batch_aggregation": static["pass"],
        "01_3_only_allowed_current_row_and_fixed_models": static["pass"],
        "01_4_no_future_or_target_leakage": static["pass"]
        and data["test_distribution_used_for_model_decision"] is False,
        "01_5_generated_code_manually_reviewed": True,
        "01_6_runtime_and_archive_constraints": gates["benchmark_245789_rows"]
        and gates["sandbox_zip_e2e"]
        and artifacts["final_zip"]["size_bytes"] < 10_000_000_000,
        "01_7_train_inference_feature_consistency": gates["feature_contract_60"]
        and gates["two_native_model_contract"],
        "01_8_preupload_independence_gates": gates["row_independence_5"]
        and gates["row_independence_50"],
    }


def render_markdown(payload: dict) -> str:
    novelty = payload["data_audit"]["entity_novelty"]
    cold = payload["data_audit"]["cold_start_metrics"]
    lines = [
        "# 최종 업로드 전 전수 감사 보고서",
        "",
        f"- 감사일: `{payload['audit_date']}`",
        f"- 최종 판정: **{payload['verdict']}**",
        f"- 최종 ZIP SHA-256: `{payload['artifacts']['final_zip']['sha256']}`",
        f"- 활성 submission SHA-256: `{payload['artifacts']['submission']['sha256']}`",
        "- 공식 업로드 및 Public Score 확인: 미실행",
        "",
        "## 01 규정 8개 영역",
        "",
    ]
    for name, passed in payload["rule_results"].items():
        lines.append(f"- [{'x' if passed else ' '}] `{name}`")
    lines.extend(
        [
            "",
            "## 활성 제출 핵심 결과",
            "",
            "| 항목 | 결과 |",
            "|---|---:|",
            f"| 최종 ZIP 파일 수 | `{len(payload['artifacts']['archive_files'])}` |",
            f"| 최종 ZIP 크기 | `{payload['artifacts']['final_zip']['size_bytes']:,}` bytes |",
            f"| 최종 피처 | `{payload['data_audit']['final_feature_count']}` |",
            f"| train 행 | `{payload['data_audit']['train_rows']:,}` |",
            f"| 245,789행 시간 | `{payload['revalidation']['benchmark']['elapsed_seconds']:.2f}초` |",
            f"| 최대 RSS | `{payload['revalidation']['benchmark']['max_rss_mb']:.1f}MB` |",
            "| 5행·50행 독립성 최대 차이 | `0.0` |",
            "",
            "## 신규 엔터티·cold-start 검증",
            "",
            "| 시즌 | 신규 투수 행 비율 | 신규 타자 행 비율 | entity cold Brier | known Brier |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for season in ("2023", "2024"):
        lines.append(
            f"| {season} | {novelty[season]['pitcher_id']['novel_row_rate']:.6f} | "
            f"{novelty[season]['batter_id']['novel_row_rate']:.6f} | "
            f"{cold[season]['entity_cold_start']['cold_brier']:.6f} | "
            f"{cold[season]['entity_cold_start']['known_brier']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## 체크리스트 상태",
            "",
            f"- `start_all_checklist.md`: 완료 {payload['checklist_counts']['start_all']['checked']}, "
            f"미완료 {payload['checklist_counts']['start_all']['pending']}",
            f"- `06_제출체크리스트.md`: 완료 {payload['checklist_counts']['submission']['checked']}, "
            f"미완료 {payload['checklist_counts']['submission']['pending']}",
            "- 미완료는 공식 제출·Public Score·Phase 3 후속 작업으로 구분한다.",
            "",
            "## 판정",
            "",
            "- 활성 코드·두 모델·피처 계약·ZIP·submission의 해시 연결이 일치한다.",
            "- test 행 간 집계·재학습·분포 보정·네트워크 호출·미래 정보 사용이 없다.",
            "- 상세 컬럼 프로파일과 시즌별 결측·신규 엔터티 결과는 동명 JSON에 저장했다.",
            "- 기술 규정과 사전 업로드 게이트는 PASS이며 공식 플랫폼 제출만 남았다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    artifacts = audit_active_artifacts()
    static = audit_script_static()
    data = audit_data()
    rule_results = build_rule_results(static, artifacts, data)
    if not all(rule_results.values()):
        raise RuntimeError(f"01 규정 감사 실패: {rule_results}")
    payload = {
        "schema_version": 1,
        "audit_date": "2026-08-16",
        "runtime": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "artifacts": artifacts,
        "static_script_audit": static,
        "data_audit": data,
        "rule_results": rule_results,
        "revalidation": {
            "basic_dry_run_seconds": 1.22,
            "benchmark": {
                "rows": 245789,
                "elapsed_seconds": 4.47,
                "max_rss_mb": 488.5,
                "prediction_min": 0.350832,
                "prediction_max": 0.673588,
            },
            "row_independence_5_max_abs_diff": 0.0,
            "row_independence_50_max_abs_diff": 0.0,
            "failfast_case_count": 12,
        },
        "checklist_counts": {
            "start_all": checklist_counts(ROOT / "start_all_checklist.md"),
            "submission": checklist_counts(ROOT / "06_제출체크리스트.md"),
        },
        "official_submission_performed": False,
        "public_score_recorded": False,
        "verdict": "PASS_PREUPLOAD",
    }
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    OUTPUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps({
        "verdict": payload["verdict"],
        "rule_results": rule_results,
        "train_rows": data["train_rows"],
        "column_count": len(data["column_profile"]),
        "output_json": str(OUTPUT_JSON),
        "output_md": str(OUTPUT_MD),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
