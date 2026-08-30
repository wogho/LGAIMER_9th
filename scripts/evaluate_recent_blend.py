#!/usr/bin/env python3
"""Evaluate fixed full-history/recent-3-season LightGBM blends with temporal Platt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "model" / "BLEND-RECENT-001"
SELECTIVE_FILES = {
    2022: ROOT / "model" / "ENS-CATF-LGBMCATR5050-FE001-EW-2022" / "selective_predictions_2022.csv",
    2023: ROOT / "model" / "ENS-CATF-LGBMCATR5050-FE001" / "selective_predictions_2023.csv",
    2024: ROOT / "model" / "ENS-CATF-LGBMCATR5050-FE001" / "selective_predictions_2024.csv",
}
RECENT_FILES = {
    2023: ROOT / "model" / "LGBM-FE001-RECENT3-EW-2023" / "validation_predictions.csv",
    2024: ROOT / "model" / "LGBM-FE001-RECENT3-2024" / "validation_predictions.csv",
}
FULL_WEIGHTS = [1.0, 0.75, 0.5, 0.0]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    score = float(np.mean(np.square(target - prediction)))
    reference = float(target.mean() * (1 - target.mean()))
    return {
        "brier": score,
        "bss": float(1 - score / reference),
        "prediction_mean": float(prediction.mean()),
        "target_mean": float(target.mean()),
        "calibration_gap": float(prediction.mean() - target.mean()),
    }


def load_frames() -> dict[int, pd.DataFrame]:
    frames: dict[int, pd.DataFrame] = {}
    for season, path in SELECTIVE_FILES.items():
        frame = pd.read_csv(path)
        if season in RECENT_FILES:
            recent = pd.read_csv(
                RECENT_FILES[season], usecols=["row_id", "pred_lgbm"]
            ).rename(columns={"pred_lgbm": "pred_lgbm_recent"})
            frame = frame.merge(recent, on="row_id", validate="one_to_one")
        else:
            frame["pred_lgbm_recent"] = frame["pred_lgbm"]
        frames[season] = frame
    return frames


def selective_prediction(frame: pd.DataFrame, full_weight: float) -> np.ndarray:
    lgbm = (
        full_weight * frame["pred_lgbm"]
        + (1 - full_weight) * frame["pred_lgbm_recent"]
    )
    return np.where(
        frame["game_type"].eq("F"),
        frame["pred_catboost"],
        0.5 * lgbm + 0.5 * frame["pred_catboost"],
    )


def evaluate() -> dict[str, Any]:
    frames = load_frames()
    baseline_brier = {
        season: metrics(
            frame["target"].to_numpy(dtype=np.float64),
            frame["pred_selective"].to_numpy(dtype=np.float64),
        )["brier"]
        for season, frame in frames.items()
    }
    candidates: dict[str, Any] = {}
    for full_weight in FULL_WEIGHTS:
        key = f"full_{full_weight:.2f}_recent_{1-full_weight:.2f}"
        predictions = {
            season: selective_prediction(frame, full_weight)
            for season, frame in frames.items()
        }
        season_results: dict[str, Any] = {}
        for season in [2022, 2023, 2024]:
            target = frames[season]["target"].to_numpy(dtype=np.float64)
            raw = metrics(target, predictions[season])
            if season == 2022:
                calibrated_prediction = predictions[season]
                coefficient = None
                intercept = None
                calibration_seasons: list[int] = []
            else:
                prior_seasons = [value for value in [2022, 2023] if value < season]
                calibration_target = np.concatenate(
                    [frames[value]["target"].to_numpy() for value in prior_seasons]
                )
                calibration_prediction = np.concatenate(
                    [predictions[value] for value in prior_seasons]
                )
                calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
                calibrator.fit(logit(calibration_prediction), calibration_target)
                calibrated_prediction = calibrator.predict_proba(
                    logit(predictions[season])
                )[:, 1]
                coefficient = float(calibrator.coef_[0, 0])
                intercept = float(calibrator.intercept_[0])
                calibration_seasons = prior_seasons
            calibrated = metrics(target, calibrated_prediction)
            raw["delta_vs_active_baseline"] = float(raw["brier"] - baseline_brier[season])
            calibrated["delta_vs_active_baseline"] = float(
                calibrated["brier"] - baseline_brier[season]
            )
            season_results[str(season)] = {
                "calibration_seasons": calibration_seasons,
                "raw": raw,
                "global_platt": calibrated,
                "calibrator": {
                    "coefficient": coefficient,
                    "intercept": intercept,
                },
            }
        deltas = {
            season: season_results[str(season)]["global_platt"][
                "delta_vs_active_baseline"
            ]
            for season in [2022, 2023, 2024]
        }
        mean_improvement = float(-(deltas[2023] + deltas[2024]) / 2)
        gate_a = bool(
            deltas[2023] < 0
            and deltas[2024] < 0
            and deltas[2022] <= 0.00010
            and mean_improvement >= 0.00010
        )
        candidates[key] = {
            "full_weight": full_weight,
            "recent_weight": 1 - full_weight,
            "seasons": season_results,
            "mean_2023_2024_improvement": mean_improvement,
            "gate_a": gate_a,
        }

    reference = candidates["full_1.00_recent_0.00"]
    best_key = max(
        candidates,
        key=lambda key: candidates[key]["mean_2023_2024_improvement"],
    )
    best = candidates[best_key]
    incremental = float(
        best["mean_2023_2024_improvement"]
        - reference["mean_2023_2024_improvement"]
    )
    return {
        "experiment_id": "BLEND-RECENT-001",
        "recent_window": 3,
        "fixed_weight_candidates": FULL_WEIGHTS,
        "candidates": candidates,
        "best_key": best_key,
        "incremental_mean_improvement_vs_global_platt_only": incremental,
        "materiality_threshold": 0.00001,
        "material_increment_pass": bool(incremental >= 0.00001),
        "decision": "keep_global_platt_only" if incremental < 0.00001 else "keep_blend",
        "test_distribution_used": False,
        "external_data_used": False,
        "active_model_sync": False,
        "source_sha256": {
            "selective": {
                str(season): sha256_file(path)
                for season, path in SELECTIVE_FILES.items()
            },
            "recent": {
                str(season): sha256_file(path) for season, path in RECENT_FILES.items()
            },
        },
    }


def render_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# BLEND-RECENT-001 고정 최근 체제 블렌드",
        "",
        "- 최근 모델: 평가 시즌 직전 3개 시즌만 학습한 FE-001 LightGBM",
        "- F는 CatBoost 유지, R의 LightGBM 구성요소만 전체/최근 고정 블렌드",
        "- 각 평가 시즌보다 과거인 OOF로 global Platt 재적합",
        "",
        "| 전체:최근 | 2023 ΔBrier | 2024 ΔBrier | 평균 개선 | Gate A |",
        "|---|---:|---:|---:|---|",
    ]
    for candidate in results["candidates"].values():
        s2023 = candidate["seasons"]["2023"]["global_platt"]
        s2024 = candidate["seasons"]["2024"]["global_platt"]
        lines.append(
            f"| {candidate['full_weight']:.2f}:{candidate['recent_weight']:.2f} | "
            f"{s2023['delta_vs_active_baseline']:+.9f} | "
            f"{s2024['delta_vs_active_baseline']:+.9f} | "
            f"{candidate['mean_2023_2024_improvement']:.9f} | "
            f"{'PASS' if candidate['gate_a'] else 'FAIL'} |"
        )
    lines.extend(
        [
            "",
            f"- 최고 후보: `{results['best_key']}`",
            "- global Platt 단독 대비 평균 추가 개선: "
            f"`{results['incremental_mean_improvement_vs_global_platt_only']:.9f}`",
            f"- 실질 개선 기준 통과: `{'YES' if results['material_increment_pass'] else 'NO'}`",
            f"- 판정: `{results['decision']}`",
            "- 활성 제출 모델은 변경하지 않았다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    results = evaluate()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "blend_results.json"
    report_path = OUTPUT_DIR / "blend_results.md"
    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path.write_text(render_markdown(results), encoding="utf-8")
    print(report_path.read_text(encoding="utf-8"))
    print(f"JSON SHA-256: {sha256_file(json_path)}")


if __name__ == "__main__":
    main()
