#!/usr/bin/env python3
"""Evaluate R-only LightGBM regime candidates with forward global Platt."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "model" / "REGIME-R-001"
SELECTIVE_FILES = {
    2022: ROOT / "model" / "ENS-CATF-LGBMCATR5050-FE001-EW-2022" / "selective_predictions_2022.csv",
    2023: ROOT / "model" / "ENS-CATF-LGBMCATR5050-FE001" / "selective_predictions_2023.csv",
    2024: ROOT / "model" / "ENS-CATF-LGBMCATR5050-FE001" / "selective_predictions_2024.csv",
}
R_MODEL_FILES = {
    2022: ROOT / "model" / "LGBM-FE001-RONLY-EW-2022" / "validation_predictions.csv",
    2023: ROOT / "model" / "LGBM-FE001-RONLY-EW-2023" / "validation_predictions.csv",
    2024: ROOT / "model" / "LGBM-FE001-RONLY-2024" / "validation_predictions.csv",
}
R_LGBM_WEIGHTS = [0.5, 0.75, 1.0]


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
    for season in [2022, 2023, 2024]:
        base = pd.read_csv(SELECTIVE_FILES[season])
        r_prediction = pd.read_csv(
            R_MODEL_FILES[season], usecols=["row_id", "pred_lgbm"]
        ).rename(columns={"pred_lgbm": "pred_lgbm_r_only"})
        frame = base.merge(r_prediction, on="row_id", how="left", validate="one_to_one")
        r_mask = frame["game_type"].eq("R")
        if frame.loc[r_mask, "pred_lgbm_r_only"].isna().any():
            raise ValueError(f"{season} R-only 예측 결합에 실패했습니다")
        if frame.loc[~r_mask, "pred_lgbm_r_only"].notna().any():
            raise ValueError(f"{season} R-only 예측에 F 행이 포함됐습니다")
        frames[season] = frame
    return frames


def candidate_prediction(frame: pd.DataFrame, r_lgbm_weight: float) -> np.ndarray:
    output = frame["pred_catboost"].to_numpy(dtype=np.float64).copy()
    r_mask = frame["game_type"].eq("R").to_numpy(dtype=bool)
    output[r_mask] = (
        r_lgbm_weight * frame.loc[r_mask, "pred_lgbm_r_only"].to_numpy()
        + (1 - r_lgbm_weight) * frame.loc[r_mask, "pred_catboost"].to_numpy()
    )
    return output


def evaluate() -> tuple[dict[str, Any], pd.DataFrame, dict[str, Any]]:
    frames = load_frames()
    baseline_brier = {
        season: metrics(
            frame["target"].to_numpy(dtype=np.float64),
            frame["pred_selective"].to_numpy(dtype=np.float64),
        )["brier"]
        for season, frame in frames.items()
    }
    candidates: dict[str, Any] = {}
    prediction_frames: dict[float, dict[int, np.ndarray]] = {}
    for weight in R_LGBM_WEIGHTS:
        key = f"r_lgbm_{weight:.2f}_catboost_{1-weight:.2f}"
        raw_predictions = {
            season: candidate_prediction(frame, weight)
            for season, frame in frames.items()
        }
        prediction_frames[weight] = raw_predictions
        season_results: dict[str, Any] = {}
        for season in [2022, 2023, 2024]:
            target = frames[season]["target"].to_numpy(dtype=np.float64)
            raw = metrics(target, raw_predictions[season])
            if season == 2022:
                calibrated_prediction = raw_predictions[season]
                calibration_seasons: list[int] = []
                coefficient = None
                intercept = None
            else:
                prior = [value for value in [2022, 2023] if value < season]
                calibration_target = np.concatenate(
                    [frames[value]["target"].to_numpy() for value in prior]
                )
                calibration_prediction = np.concatenate(
                    [raw_predictions[value] for value in prior]
                )
                calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
                calibrator.fit(logit(calibration_prediction), calibration_target)
                calibrated_prediction = calibrator.predict_proba(
                    logit(raw_predictions[season])
                )[:, 1]
                calibration_seasons = prior
                coefficient = float(calibrator.coef_[0, 0])
                intercept = float(calibrator.intercept_[0])
            calibrated = metrics(target, calibrated_prediction)
            raw["delta_vs_active_baseline"] = float(raw["brier"] - baseline_brier[season])
            calibrated["delta_vs_active_baseline"] = float(
                calibrated["brier"] - baseline_brier[season]
            )
            season_results[str(season)] = {
                "calibration_seasons": calibration_seasons,
                "raw": raw,
                "global_platt": calibrated,
                "calibrator": {"coefficient": coefficient, "intercept": intercept},
            }
        deltas = {
            season: season_results[str(season)]["global_platt"]["delta_vs_active_baseline"]
            for season in [2022, 2023, 2024]
        }
        mean_improvement = float(-(deltas[2023] + deltas[2024]) / 2)
        gate_a = bool(
            deltas[2023] < 0
            and deltas[2024] < 0
            and deltas[2022] <= 0.00010
            and mean_improvement >= 0.00010
        )
        gate_b = bool(
            deltas[2024] <= -0.00020
            and deltas[2023] <= 0.00005
            and deltas[2022] <= 0.00010
        )
        candidates[key] = {
            "r_lgbm_weight": weight,
            "catboost_weight": 1 - weight,
            "seasons": season_results,
            "mean_2023_2024_improvement": mean_improvement,
            "gate_a": gate_a,
            "gate_b": gate_b,
            "pass": bool(gate_a or gate_b),
        }

    passed = [item for item in candidates.items() if item[1]["pass"]]
    if not passed:
        raise RuntimeError("Gate A/B를 통과한 R regime 후보가 없습니다")
    selected_key, selected = max(
        passed, key=lambda item: item[1]["mean_2023_2024_improvement"]
    )
    selected_weight = float(selected["r_lgbm_weight"])
    selected_predictions = prediction_frames[selected_weight]
    all_target = np.concatenate(
        [frames[season]["target"].to_numpy() for season in [2022, 2023, 2024]]
    )
    all_prediction = np.concatenate(
        [selected_predictions[season] for season in [2022, 2023, 2024]]
    )
    future_calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    future_calibrator.fit(logit(all_prediction), all_target)
    future_contract = {
        "schema_version": 1,
        "artifact_role": "future_inference_candidate_not_active",
        "method": "platt_logit",
        "scope": "all_rows_after_regime_selection",
        "input_clip_epsilon": 1e-6,
        "coefficient": float(future_calibrator.coef_[0, 0]),
        "intercept": float(future_calibrator.intercept_[0]),
        "fit_seasons": [2022, 2023, 2024],
        "fit_rows": int(len(all_target)),
        "source_candidate": selected_key,
        "active_model_sync": False,
    }
    output_frames = []
    for season, frame in frames.items():
        output = frame[
            ["row_id", "season", "target", "game_type", "pred_catboost"]
        ].copy()
        output["pred_lgbm_r_only"] = frame["pred_lgbm_r_only"]
        output["pred_regime_raw"] = selected_predictions[season]
        if season == 2022:
            output["pred_regime_forward_platt"] = selected_predictions[season]
        else:
            params = selected["seasons"][str(season)]["calibrator"]
            z = (
                params["coefficient"] * logit(selected_predictions[season]).reshape(-1)
                + params["intercept"]
            )
            output["pred_regime_forward_platt"] = 1 / (1 + np.exp(-z))
        output_frames.append(output)
    predictions = pd.concat(output_frames, ignore_index=True)
    results = {
        "experiment_id": "REGIME-R-001",
        "fixed_weight_candidates": R_LGBM_WEIGHTS,
        "candidates": candidates,
        "selected_candidate": selected_key,
        "selection_rule": {
            "F": {"catboost": 1.0},
            "R": {"lightgbm_r_only": selected_weight, "catboost": 1 - selected_weight},
            "postprocess": "global_platt",
        },
        "future_calibrator_file": "future_global_platt.json",
        "test_distribution_used": False,
        "external_data_used": False,
        "active_model_sync": False,
        "source_sha256": {
            "selective": {str(s): sha256_file(p) for s, p in SELECTIVE_FILES.items()},
            "r_only": {str(s): sha256_file(p) for s, p in R_MODEL_FILES.items()},
        },
    }
    return results, predictions, future_contract


def render_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# REGIME-R-001 R 전용 모델",
        "",
        "- F: CatBoost",
        "- R: R 전용 LightGBM과 CatBoost의 사전 고정 후보 가중치",
        "- 후처리: 평가 시즌보다 과거 OOF로 적합한 global Platt",
        "",
        "| R LGBM : CAT | 2022 Δ | 2023 Δ | 2024 Δ | 평균 개선 | Gate A |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for candidate in results["candidates"].values():
        seasons = candidate["seasons"]
        lines.append(
            f"| {candidate['r_lgbm_weight']:.2f}:{candidate['catboost_weight']:.2f} | "
            f"{seasons['2022']['global_platt']['delta_vs_active_baseline']:+.9f} | "
            f"{seasons['2023']['global_platt']['delta_vs_active_baseline']:+.9f} | "
            f"{seasons['2024']['global_platt']['delta_vs_active_baseline']:+.9f} | "
            f"{candidate['mean_2023_2024_improvement']:.9f} | "
            f"{'PASS' if candidate['gate_a'] else 'FAIL'} |"
        )
    selected = results["candidates"][results["selected_candidate"]]
    lines.extend(
        [
            "",
            f"- 선택 후보: `{results['selected_candidate']}`",
            f"- 2023/2024 평균 Brier 개선: `{selected['mean_2023_2024_improvement']:.9f}`",
            "- 판정: 격리 최종 학습·E2E 후보",
            "- 활성 제출 모델은 변경하지 않았다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    results, predictions, future_contract = evaluate()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results_path = OUTPUT_DIR / "regime_results.json"
    report_path = OUTPUT_DIR / "regime_results.md"
    predictions_path = OUTPUT_DIR / "regime_oof_predictions.csv"
    future_path = OUTPUT_DIR / "future_global_platt.json"
    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path.write_text(render_markdown(results), encoding="utf-8")
    predictions.to_csv(predictions_path, index=False)
    future_path.write_text(
        json.dumps(future_contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(report_path.read_text(encoding="utf-8"))
    print(f"Results SHA-256: {sha256_file(results_path)}")
    print(f"Future calibrator SHA-256: {sha256_file(future_path)}")


if __name__ == "__main__":
    main()
