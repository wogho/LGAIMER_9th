#!/usr/bin/env python3
"""Summarize the first score-1000 improvement round from reproducible artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "model" / "SCORE1000-ROUND1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    brier = float(np.mean(np.square(y - prediction)))
    reference = float(y.mean() * (1.0 - y.mean()))
    return {
        "brier": brier,
        "bss": float(1.0 - brier / reference),
    }


def seed_precheck() -> dict[str, Any]:
    baseline_path = (
        ROOT
        / "model"
        / "ENS-CATF-LGBMCATR5050-FE001-EW-2022"
        / "selective_predictions_2022.csv"
    )
    seed_path = (
        ROOT
        / "model"
        / "CAT-FE001-SEED2024-EW-2022"
        / "validation_predictions.csv"
    )
    baseline = pd.read_csv(baseline_path)
    seed = pd.read_csv(seed_path, usecols=["row_id", "pred_catboost"]).rename(
        columns={"pred_catboost": "pred_catboost_seed2024"}
    )
    frame = baseline.merge(seed, on="row_id", validate="one_to_one")
    catboost_mean = (
        frame["pred_catboost"] + frame["pred_catboost_seed2024"]
    ) / 2.0
    prediction = np.where(
        frame["game_type"].eq("F"),
        catboost_mean,
        0.5 * frame["pred_lgbm"] + 0.5 * catboost_mean,
    )
    y = frame["target"].to_numpy(dtype=np.float64)
    baseline_metrics = metric(y, frame["pred_selective"].to_numpy())
    candidate_metrics = metric(y, prediction)
    correlation = float(
        np.corrcoef(
            frame["pred_catboost"], frame["pred_catboost_seed2024"]
        )[0, 1]
    )
    improvement = float(baseline_metrics["brier"] - candidate_metrics["brier"])
    return {
        "experiment_id": "ENS-SEED-001-PRECHECK",
        "season": 2022,
        "seeds": [42, 2024],
        "baseline": baseline_metrics,
        "candidate": candidate_metrics,
        "brier_improvement": improvement,
        "catboost_prediction_correlation": correlation,
        "early_stop_rule": "stop if improvement < 0.00005 and correlation > 0.99",
        "early_stop": bool(improvement < 0.00005 and correlation > 0.99),
        "source_sha256": {
            "baseline": sha256_file(baseline_path),
            "seed2024": sha256_file(seed_path),
        },
    }


def fe002_precheck() -> dict[str, Any]:
    experiments = {
        "lgbm_baseline_2023": ROOT / "model" / "LGBM-FE001-EW-2023" / "metadata.json",
        "lgbm_state_2023": ROOT / "model" / "LGBM-FE002-STATE-EW-2023" / "metadata.json",
        "lgbm_form_2023": ROOT / "model" / "LGBM-FE002-FORM-EW-2023" / "metadata.json",
        "lgbm_support_2023": ROOT / "model" / "LGBM-FE002-SUPPORT-EW-2023" / "metadata.json",
        "catboost_baseline_2023": ROOT / "model" / "CAT-FE001-EW-2023" / "metadata.json",
        "catboost_state_2023": ROOT / "model" / "CAT-FE002-STATE-EW-2023" / "metadata.json",
        "catboost_form_2023": ROOT / "model" / "CAT-FE002-FORM-EW-2023" / "metadata.json",
        "catboost_support_2023": ROOT / "model" / "CAT-FE002-SUPPORT-EW-2023" / "metadata.json",
        "catboost_baseline_2024": ROOT / "model" / "CAT-FE001-2024" / "metadata.json",
        "catboost_support_2024": ROOT / "model" / "CAT-FE002-SUPPORT-2024" / "metadata.json",
    }
    rows: dict[str, Any] = {}
    for name, path in experiments.items():
        metadata = load_json(path)
        model_family = "catboost" if "catboost" in name else "lightgbm"
        metric_key = "catboost" if model_family == "catboost" else "lgbm"
        rows[name] = {
            "experiment_id": metadata["exp_id"],
            "model_family": model_family,
            "valid_season": int(metadata["valid_season"]),
            "feature_count": int(metadata["feature_count"]),
            "fe002_groups": metadata.get("fe002_groups", []),
            "brier": float(metadata["metrics"][metric_key]["brier_score"]),
            "bss": float(metadata["metrics"][metric_key]["bss"]),
            "metadata_sha256": sha256_file(path),
        }
    comparisons = {}
    for family in ["lgbm", "catboost"]:
        baseline = rows[f"{family}_baseline_2023"]["brier"]
        for group in ["state", "form", "support"]:
            candidate = rows[f"{family}_{group}_2023"]["brier"]
            comparisons[f"{family}_{group}_2023"] = {
                "candidate_minus_baseline": float(candidate - baseline),
                "improved": bool(candidate < baseline),
            }
    comparisons["catboost_support_2024"] = {
        "candidate_minus_baseline": float(
            rows["catboost_support_2024"]["brier"]
            - rows["catboost_baseline_2024"]["brier"]
        ),
        "improved": bool(
            rows["catboost_support_2024"]["brier"]
            < rows["catboost_baseline_2024"]["brier"]
        ),
    }
    return {
        "experiment_id": "FE002-PRECHECK-001",
        "experiments": rows,
        "comparisons": comparisons,
        "decision": "reject",
        "reason": (
            "All LightGBM groups worsened in 2023; only CatBoost support improved "
            "slightly in 2023 and it worsened in 2024."
        ),
        "active_model_sync": False,
    }


def render_markdown(payload: dict[str, Any]) -> str:
    calibration = payload["calibration"]["results"]
    seed = payload["seed_precheck"]
    fe002 = payload["fe002_precheck"]
    lines = [
        "# 1,000점 개선 1차 라운드 결과",
        "",
        "## 결론",
        "",
        "- `DIAG-1000-001`: 완료. R 과대예측이 2022·2023·2024에 반복됨을 확인했다.",
        "- `CAL-SEL-OOF-001`: global Platt가 Gate A를 통과해 격리 결합 후보로 유지한다.",
        "- `ENS-SEED-001`: 2-seed 2022 선행 게이트에서 개선 폭이 작아 조기 중단한다.",
        "- `FE-002`: 모델·시즌 반복 개선에 실패해 폐기한다.",
        "- 활성 제출 모델과 ZIP은 변경하지 않았다.",
        "",
        "## 선택형 OOF 보정",
        "",
        "| 방법 | 2023 ΔBrier | 2024 ΔBrier | 평균 개선 | Gate A | 상태 |",
        "|---|---:|---:|---:|---|---|",
    ]
    for method in ["platt_global", "platt_r_only"]:
        gate = calibration["gate"][method]
        lines.append(
            f"| {method} | {gate['delta_2023']:+.9f} | {gate['delta_2024']:+.9f} | "
            f"{gate['mean_2023_2024_improvement']:.9f} | "
            f"{'PASS' if gate['gate_a'] else 'FAIL'} | "
            f"{'후보 유지' if gate['pass'] else '중단'} |"
        )
    lines.extend(
        [
            "",
            "## 다중 시드 선행 게이트",
            "",
            f"- 2022 Brier 개선: `{seed['brier_improvement']:.9f}`",
            f"- CatBoost 예측 상관: `{seed['catboost_prediction_correlation']:.9f}`",
            f"- 조기 중단: `{'YES' if seed['early_stop'] else 'NO'}`",
            "",
            "## FE-002 선행 게이트",
            "",
            "| 후보 | 기준 대비 ΔBrier | 개선 |",
            "|---|---:|---|",
        ]
    )
    for name, comparison in fe002["comparisons"].items():
        lines.append(
            f"| {name} | {comparison['candidate_minus_baseline']:+.9f} | "
            f"{'YES' if comparison['improved'] else 'NO'} |"
        )
    lines.extend(
        [
            "",
            "## 다음 순서",
            "",
            "1. global Platt와 기존 선택형 기준선을 결합 후보로 보존한다.",
            "2. 다음 구조 실험은 전체 이력·최근 체제 고정 블렌드(P4)를 선행 검증한다.",
            "3. P4가 Gate A/B를 통과하지 못하면 regime 분리(P5) 전에 진단을 재검토한다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    diagnostics_path = ROOT / "model" / "DIAG-1000-001" / "diagnostics.json"
    calibration_path = (
        ROOT / "model" / "CAL-SEL-OOF-001" / "calibration_results.json"
    )
    diagnostics = load_json(diagnostics_path)
    calibration = load_json(calibration_path)
    payload = {
        "round_id": "SCORE1000-ROUND1",
        "baseline_leaderboard_score": 815.20127,
        "target_score": 1000.0,
        "active_submission_changed": False,
        "diagnostics": {
            "experiment_id": diagnostics["experiment_id"],
            "json_sha256": sha256_file(diagnostics_path),
            "findings": diagnostics["findings"],
        },
        "calibration": calibration,
        "calibration_json_sha256": sha256_file(calibration_path),
        "seed_precheck": seed_precheck(),
        "fe002_precheck": fe002_precheck(),
        "round_decision": {
            "keep": ["platt_global"],
            "stop": ["platt_r_only", "ENS-SEED-001", "FE-002"],
            "next": "BLEND-RECENT-001",
        },
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "round1_results.json"
    report_path = OUTPUT_DIR / "round1_results.md"
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path.write_text(render_markdown(payload), encoding="utf-8")
    print(report_path.read_text(encoding="utf-8"))
    print(f"Saved JSON: {json_path}")
    print(f"JSON SHA-256: {sha256_file(json_path)}")


if __name__ == "__main__":
    main()
