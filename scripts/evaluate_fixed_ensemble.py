#!/usr/bin/env python3
"""Evaluate a predeclared 50:50 LightGBM/CatBoost FE-001 ensemble."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import brier_score, brier_skill_score


DEFAULT_SOURCES = {
    2023: {
        "lightgbm": ROOT / "model" / "LGBM-FE001-EW-2023",
        "catboost": ROOT / "model" / "CAT-FE001-EW-2023",
    },
    2024: {
        "lightgbm": ROOT / "model" / "LGBM-FE001-2024",
        "catboost": ROOT / "model" / "CAT-FE001-2024",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate fixed FE-001 50:50 ensemble")
    parser.add_argument("--train-path", type=Path, default=ROOT / "data" / "train.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "model" / "ENS-LGBM-CAT-FE001-5050",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile_ece(y_true: np.ndarray, prediction: np.ndarray) -> float:
    frame = pd.DataFrame({"target": y_true, "prediction": prediction})
    frame["bin"] = pd.qcut(
        frame["prediction"], q=10, labels=False, duplicates="drop"
    )
    grouped = frame.groupby("bin", observed=True).agg(
        rows=("target", "size"),
        target_mean=("target", "mean"),
        prediction_mean=("prediction", "mean"),
    )
    return float(
        (
            grouped["rows"]
            / len(frame)
            * (grouped["target_mean"] - grouped["prediction_mean"]).abs()
        ).sum()
    )


def metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    return {
        "rows": int(len(y_true)),
        "brier": float(brier_score(y_true, prediction)),
        "bss": float(brier_skill_score(y_true, prediction)),
        "ece_quantile_10": quantile_ece(y_true, prediction),
        "prediction_mean": float(prediction.mean()),
        "target_mean": float(y_true.mean()),
        "calibration_gap": float(prediction.mean() - y_true.mean()),
    }


def segmented_metrics(
    frame: pd.DataFrame, prediction_column: str
) -> dict[str, dict[str, float | int]]:
    output = {
        "overall": metrics(
            frame["target"].to_numpy(dtype=np.int8),
            frame[prediction_column].to_numpy(dtype=np.float64),
        )
    }
    for game_type in ["F", "R"]:
        subset = frame.loc[frame["game_type"].eq(game_type)]
        output[f"game_type_{game_type.lower()}"] = metrics(
            subset["target"].to_numpy(dtype=np.int8),
            subset[prediction_column].to_numpy(dtype=np.float64),
        )
    return output


def load_source_contract(
    season: int, source_dirs: dict[str, Path]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    loaded: dict[str, Any] = {}
    reference_features: list[str] | None = None
    for family, directory in source_dirs.items():
        metadata_path = directory / "metadata.json"
        features_path = directory / "feature_columns.json"
        predictions_path = directory / "validation_predictions.csv"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        feature_columns = json.loads(features_path.read_text(encoding="utf-8"))
        if int(metadata["valid_season"]) != season:
            raise ValueError(f"{family} source 검증 시즌이 다릅니다: {directory}")
        if int(metadata["feature_count"]) != 60 or len(feature_columns) != 60:
            raise ValueError(f"{family} source가 FE-001 60피처가 아닙니다")
        if reference_features is None:
            reference_features = feature_columns
        elif feature_columns != reference_features:
            raise ValueError("LightGBM과 CatBoost 피처 계약이 다릅니다")
        prediction_column = "pred_lgbm" if family == "lightgbm" else "pred_catboost"
        frame = pd.read_csv(
            predictions_path,
            usecols=["row_id", "season", "target", prediction_column],
        )
        if frame["row_id"].duplicated().any() or not frame["season"].eq(season).all():
            raise ValueError(f"{family} 예측 row_id 또는 시즌 계약이 잘못되었습니다")
        loaded[family] = {
            "frame": frame,
            "prediction_column": prediction_column,
            "source": {
                "exp_id": metadata["exp_id"],
                "directory": str(directory),
                "train_seasons": metadata["train_seasons"],
                "valid_season": season,
                "prediction_file": predictions_path.name,
                "prediction_sha256": sha256_file(predictions_path),
                "model_file": metadata.get("model_file", "model.txt"),
                "metadata_sha256": sha256_file(metadata_path),
            },
        }

    lightgbm = loaded["lightgbm"]["frame"]
    catboost = loaded["catboost"]["frame"]
    merged = lightgbm.merge(
        catboost,
        how="inner",
        on=["row_id", "season", "target"],
        validate="one_to_one",
    )
    if len(merged) != len(lightgbm) or len(merged) != len(catboost):
        raise ValueError("두 모델 예측의 row_id·target 교집합이 완전하지 않습니다")
    merged["pred_ensemble_50_50"] = (
        0.5 * merged["pred_lgbm"] + 0.5 * merged["pred_catboost"]
    )
    contract = {
        "season": season,
        "weights": {"lightgbm": 0.5, "catboost": 0.5},
        "weight_selection": "predeclared_not_fitted",
        "rows": int(len(merged)),
        "feature_count": len(reference_features or []),
        "feature_columns": reference_features,
        "sources": {
            family: loaded[family]["source"] for family in ["lightgbm", "catboost"]
        },
    }
    return merged, contract


def render_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# LightGBM·CatBoost FE-001 고정 50:50 앙상블",
        "",
        "- 가중치: 사전 고정 `0.5 / 0.5`",
        "- 검증 예측이나 test 분포로 가중치를 최적화하지 않음",
        "",
        "| 시즌 | 모델 | Brier | BSS | ECE10 | 평균 오차 |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for season in results["evaluation_seasons"]:
        season_result = results["seasons"][str(season)]
        for method in ["lightgbm", "catboost", "ensemble_50_50"]:
            overall = season_result[method]["overall"]
            lines.append(
                f"| {season} | {method} | {overall['brier']:.9f} | "
                f"{overall['bss']:.9f} | {overall['ece_quantile_10']:.9f} | "
                f"{overall['calibration_gap']:+.9f} |"
            )
    lines.extend(
        [
            "",
            "## 게이트",
            "",
            "- 규칙: 모든 시즌에서 두 단독 모델보다 Brier가 낮아야 함",
            f"- 판정: `{'PASS' if results['gate_pass'] else 'FAIL'}`",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    context = pd.read_csv(args.train_path, usecols=["row_id", "game_type"])
    if context["row_id"].duplicated().any():
        raise ValueError("train 원본 row_id가 중복되었습니다")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seasons: dict[str, Any] = {}
    contracts: dict[str, Any] = {}
    gate_checks: list[bool] = []
    for season, source_dirs in DEFAULT_SOURCES.items():
        frame, contract = load_source_contract(season, source_dirs)
        frame = frame.merge(context, how="left", on="row_id", validate="one_to_one")
        if frame["game_type"].isna().any():
            raise ValueError("앙상블 예측에 game_type을 결합하지 못했습니다")
        season_metrics = {
            "lightgbm": segmented_metrics(frame, "pred_lgbm"),
            "catboost": segmented_metrics(frame, "pred_catboost"),
            "ensemble_50_50": segmented_metrics(frame, "pred_ensemble_50_50"),
            "prediction_correlation": float(
                frame[["pred_lgbm", "pred_catboost"]].corr().iloc[0, 1]
            ),
        }
        ensemble_brier = season_metrics["ensemble_50_50"]["overall"]["brier"]
        lightgbm_brier = season_metrics["lightgbm"]["overall"]["brier"]
        catboost_brier = season_metrics["catboost"]["overall"]["brier"]
        season_metrics["ensemble_50_50"]["overall"]["delta_vs_lightgbm"] = float(
            ensemble_brier - lightgbm_brier
        )
        season_metrics["ensemble_50_50"]["overall"]["delta_vs_catboost"] = float(
            ensemble_brier - catboost_brier
        )
        gate_check = ensemble_brier < min(lightgbm_brier, catboost_brier)
        season_metrics["gate_pass"] = gate_check
        gate_checks.append(gate_check)
        seasons[str(season)] = season_metrics
        contracts[str(season)] = contract
        frame.to_csv(
            args.output_dir / f"ensemble_predictions_{season}.csv", index=False
        )

    results = {
        "experiment_id": "ENS-LGBM-CAT-FE001-5050",
        "evaluation_seasons": sorted(DEFAULT_SOURCES),
        "weights": {"lightgbm": 0.5, "catboost": 0.5},
        "weight_selection": "predeclared_not_fitted",
        "gate_rule": "ensemble Brier lower than both single models in every season",
        "gate_pass": all(gate_checks),
        "seasons": seasons,
        "contracts": contracts,
        "active_model_sync": False,
    }
    results_path = args.output_dir / "ensemble_results.json"
    report_path = args.output_dir / "ensemble_results.md"
    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path.write_text(render_markdown(results), encoding="utf-8")
    print(report_path.read_text(encoding="utf-8"))
    print(f"Saved results: {results_path}")


if __name__ == "__main__":
    main()
