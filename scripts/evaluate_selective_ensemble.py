#!/usr/bin/env python3
"""Evaluate the predeclared F=CatBoost, R=50:50 FE-001 ensemble."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_fixed_ensemble import (  # noqa: E402
    DEFAULT_SOURCES,
    load_source_contract,
    segmented_metrics,
    sha256_file,
)

SELECTIVE_SOURCES = {
    2022: {
        "lightgbm": ROOT / "model" / "LGBM-FE001-EW-2022",
        "catboost": ROOT / "model" / "CAT-FE001-EW-2022",
    },
    **DEFAULT_SOURCES,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate predeclared F=CatBoost, R=50:50 FE-001 ensemble"
    )
    parser.add_argument("--train-path", type=Path, default=ROOT / "data" / "train.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: multi-season candidate directory or season-specific directory",
    )
    parser.add_argument(
        "--evaluation-seasons",
        default="2023,2024",
        help="Comma-separated Holdout seasons (available: 2022,2023,2024)",
    )
    return parser.parse_args()


def render_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# FE-001 game_type 선택형 앙상블",
        "",
        "- 사전 선언 규칙: `F=CatBoost`, `R=LightGBM 0.5 + CatBoost 0.5`",
        "- 연속 가중치 탐색과 test 예측 분포 사용 없음",
        "",
        "| 시즌 | 모델 | Brier | BSS | ECE10 | 평균 오차 | CatBoost 대비 |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for season in results["evaluation_seasons"]:
        season_result = results["seasons"][str(season)]
        catboost_brier = season_result["catboost"]["overall"]["brier"]
        for method in ["lightgbm", "catboost", "ensemble_50_50", "selective"]:
            overall = season_result[method]["overall"]
            delta = overall["brier"] - catboost_brier
            lines.append(
                f"| {season} | {method} | {overall['brier']:.9f} | "
                f"{overall['bss']:.9f} | {overall['ece_quantile_10']:.9f} | "
                f"{overall['calibration_gap']:+.9f} | {delta:+.9f} |"
            )
    lines.extend(
        [
            "",
            "## 계약 검증",
            "",
            f"- F CatBoost exact identity: `{'PASS' if results['identity_checks']['f_catboost_all'] else 'FAIL'}`",
            f"- R 고정 50:50 exact identity: `{'PASS' if results['identity_checks']['r_50_50_all'] else 'FAIL'}`",
            "",
            "## 게이트",
            "",
            "- 규칙: 선택형 앙상블 Brier가 모든 평가 시즌에서 CatBoost 단독보다 낮아야 함",
            f"- 판정: `{'PASS' if results['gate_pass'] else 'FAIL'}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    evaluation_seasons = [
        int(value.strip())
        for value in args.evaluation_seasons.split(",")
        if value.strip()
    ]
    if not evaluation_seasons or len(evaluation_seasons) != len(set(evaluation_seasons)):
        raise ValueError("평가 시즌이 비어 있거나 중복되었습니다")
    if sorted(evaluation_seasons) != evaluation_seasons:
        raise ValueError("평가 시즌은 오름차순이어야 합니다")
    unavailable = sorted(set(evaluation_seasons) - set(SELECTIVE_SOURCES))
    if unavailable:
        raise ValueError(f"원천 산출물이 없는 평가 시즌입니다: {unavailable}")
    experiment_id = "ENS-CATF-LGBMCATR5050-FE001"
    if evaluation_seasons != [2023, 2024]:
        experiment_id += "-EW-" + "-".join(str(value) for value in evaluation_seasons)
    output_dir = args.output_dir or ROOT / "model" / experiment_id
    context = pd.read_csv(args.train_path, usecols=["row_id", "game_type"])
    if context["row_id"].duplicated().any():
        raise ValueError("train 원본 row_id가 중복되었습니다")

    output_dir.mkdir(parents=True, exist_ok=True)
    seasons: dict[str, Any] = {}
    contracts: dict[str, Any] = {}
    gate_checks: list[bool] = []
    identity_checks = {"f_catboost": {}, "r_50_50": {}}

    for season in evaluation_seasons:
        source_dirs = SELECTIVE_SOURCES[season]
        frame, contract = load_source_contract(season, source_dirs)
        frame = frame.merge(context, how="left", on="row_id", validate="one_to_one")
        if frame["game_type"].isna().any():
            raise ValueError("선택형 앙상블 예측에 game_type을 결합하지 못했습니다")
        unexpected_types = sorted(set(frame["game_type"]) - {"F", "R"})
        if unexpected_types:
            raise ValueError(f"지원하지 않는 game_type이 있습니다: {unexpected_types}")
        if set(frame["game_type"]) != {"F", "R"}:
            raise ValueError("평가 시즌에 F와 R game_type이 모두 있어야 합니다")

        frame["pred_selective"] = np.where(
            frame["game_type"].eq("F"),
            frame["pred_catboost"],
            frame["pred_ensemble_50_50"],
        )
        f_mask = frame["game_type"].eq("F")
        r_mask = frame["game_type"].eq("R")
        f_identity = bool(
            np.array_equal(
                frame.loc[f_mask, "pred_selective"].to_numpy(),
                frame.loc[f_mask, "pred_catboost"].to_numpy(),
            )
        )
        r_identity = bool(
            np.array_equal(
                frame.loc[r_mask, "pred_selective"].to_numpy(),
                frame.loc[r_mask, "pred_ensemble_50_50"].to_numpy(),
            )
        )
        if not f_identity or not r_identity:
            raise AssertionError("선택형 앙상블 구간 적용 계약이 깨졌습니다")

        season_metrics = {
            "lightgbm": segmented_metrics(frame, "pred_lgbm"),
            "catboost": segmented_metrics(frame, "pred_catboost"),
            "ensemble_50_50": segmented_metrics(frame, "pred_ensemble_50_50"),
            "selective": segmented_metrics(frame, "pred_selective"),
        }
        selective_brier = season_metrics["selective"]["overall"]["brier"]
        lightgbm_brier = season_metrics["lightgbm"]["overall"]["brier"]
        catboost_brier = season_metrics["catboost"]["overall"]["brier"]
        fixed_brier = season_metrics["ensemble_50_50"]["overall"]["brier"]
        season_metrics["selective"]["overall"].update(
            {
                "delta_vs_lightgbm": float(selective_brier - lightgbm_brier),
                "delta_vs_catboost": float(selective_brier - catboost_brier),
                "delta_vs_fixed_50_50": float(selective_brier - fixed_brier),
            }
        )
        gate_check = selective_brier < catboost_brier
        season_metrics["gate_pass"] = gate_check
        gate_checks.append(gate_check)
        seasons[str(season)] = season_metrics
        contract["selection_rule"] = {
            "F": {"lightgbm": 0.0, "catboost": 1.0},
            "R": {"lightgbm": 0.5, "catboost": 0.5},
        }
        contract["selection_rule_status"] = "predeclared_not_fitted_in_this_run"
        contracts[str(season)] = contract
        identity_checks["f_catboost"][str(season)] = f_identity
        identity_checks["r_50_50"][str(season)] = r_identity
        frame.to_csv(
            output_dir / f"selective_predictions_{season}.csv", index=False
        )

    results = {
        "experiment_id": experiment_id,
        "evaluation_seasons": evaluation_seasons,
        "selection_rule": {
            "F": {"lightgbm": 0.0, "catboost": 1.0},
            "R": {"lightgbm": 0.5, "catboost": 0.5},
        },
        "selection_rule_status": "predeclared_not_fitted_in_this_run",
        "continuous_weight_search": False,
        "test_prediction_distribution_used": False,
        "gate_rule": "selective Brier lower than CatBoost in every evaluation season",
        "gate_pass": all(gate_checks),
        "identity_checks": {
            "f_catboost_by_season": identity_checks["f_catboost"],
            "r_50_50_by_season": identity_checks["r_50_50"],
            "f_catboost_all": all(identity_checks["f_catboost"].values()),
            "r_50_50_all": all(identity_checks["r_50_50"].values()),
        },
        "seasons": seasons,
        "contracts": contracts,
        "active_model_sync": False,
    }
    results_path = output_dir / "selective_results.json"
    report_path = output_dir / "selective_results.md"
    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path.write_text(render_markdown(results), encoding="utf-8")
    print(report_path.read_text(encoding="utf-8"))
    print(f"Saved results: {results_path}")
    print(f"Results SHA-256: {sha256_file(results_path)}")


if __name__ == "__main__":
    main()
