#!/usr/bin/env python3
"""Evaluate temporal Platt calibration on the deployed selective ensemble."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[1]
PREDICTION_FILES = {
    2022: ROOT
    / "model"
    / "ENS-CATF-LGBMCATR5050-FE001-EW-2022"
    / "selective_predictions_2022.csv",
    2023: ROOT
    / "model"
    / "ENS-CATF-LGBMCATR5050-FE001"
    / "selective_predictions_2023.csv",
    2024: ROOT
    / "model"
    / "ENS-CATF-LGBMCATR5050-FE001"
    / "selective_predictions_2024.csv",
}
REQUIRED_COLUMNS = [
    "row_id",
    "season",
    "target",
    "pred_lgbm",
    "pred_catboost",
    "pred_ensemble_50_50",
    "game_type",
    "pred_selective",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "model" / "CAL-SEL-OOF-001",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def probability_logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def quantile_ece(y_true: np.ndarray, prediction: np.ndarray, bins: int = 10) -> float:
    frame = pd.DataFrame(
        {
            "target": np.asarray(y_true, dtype=np.float64),
            "prediction": np.asarray(prediction, dtype=np.float64),
        }
    )
    frame["bin"] = pd.qcut(
        frame["prediction"].rank(method="first"),
        q=bins,
        labels=False,
        duplicates="drop",
    )
    grouped = frame.groupby("bin", observed=True).agg(
        rows=("target", "size"),
        target_mean=("target", "mean"),
        prediction_mean=("prediction", "mean"),
    )
    return float(
        np.average(
            np.abs(grouped["target_mean"] - grouped["prediction_mean"]),
            weights=grouped["rows"],
        )
    )


def metrics(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    y_true = np.asarray(y_true, dtype=np.int8)
    prediction = np.asarray(prediction, dtype=np.float64)
    score = float(np.mean(np.square(y_true - prediction)))
    reference = float(y_true.mean() * (1.0 - y_true.mean()))
    return {
        "rows": int(len(y_true)),
        "brier": score,
        "bss": float(1.0 - score / reference) if reference > 0 else 0.0,
        "ece_quantile_10": quantile_ece(y_true, prediction),
        "prediction_mean": float(prediction.mean()),
        "target_mean": float(y_true.mean()),
        "calibration_gap": float(prediction.mean() - y_true.mean()),
    }


def segmented_metrics(
    frame: pd.DataFrame, prediction: np.ndarray
) -> dict[str, dict[str, float | int]]:
    output = {"overall": metrics(frame["target"].to_numpy(), prediction)}
    for game_type in ["F", "R"]:
        mask = frame["game_type"].astype("string").eq(game_type).to_numpy(dtype=bool)
        output[f"game_type_{game_type.lower()}"] = metrics(
            frame.loc[mask, "target"].to_numpy(), prediction[mask]
        )
    return output


def load_oof() -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    sources: dict[str, Any] = {}
    seen_rows: set[str] = set()
    for season, path in PREDICTION_FILES.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path, usecols=REQUIRED_COLUMNS)
        if frame.empty or not frame["season"].eq(season).all():
            raise ValueError(f"{season} OOF season 계약이 깨졌습니다")
        if frame["row_id"].isna().any() or frame["row_id"].duplicated().any():
            raise ValueError(f"{season} OOF row_id 계약이 깨졌습니다")
        duplicate_across_seasons = seen_rows & set(frame["row_id"])
        if duplicate_across_seasons:
            raise ValueError("서로 다른 OOF 시즌 사이에 row_id가 중복되었습니다")
        seen_rows.update(frame["row_id"])
        expected = np.where(
            frame["game_type"].eq("F"),
            frame["pred_catboost"],
            frame["pred_ensemble_50_50"],
        )
        if not np.array_equal(frame["pred_selective"].to_numpy(), expected):
            raise ValueError(f"{season} 선택형 identity 계약이 깨졌습니다")
        if not set(frame["target"].unique()).issubset({0, 1}):
            raise ValueError(f"{season} target이 이진값이 아닙니다")
        if not np.isfinite(frame["pred_selective"]).all() or not frame[
            "pred_selective"
        ].between(0.0, 1.0).all():
            raise ValueError(f"{season} 선택형 확률 범위가 잘못되었습니다")
        frames.append(frame)
        sources[str(season)] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256_file(path),
            "rows": int(len(frame)),
        }
    oof = pd.concat(frames, ignore_index=True).sort_values(
        ["season", "row_id"]
    ).reset_index(drop=True)
    return oof, sources


def fit_platt(frame: pd.DataFrame, r_only: bool) -> tuple[LogisticRegression, int]:
    fit_frame = frame.loc[frame["game_type"].eq("R")] if r_only else frame
    if fit_frame.empty or fit_frame["target"].nunique() != 2:
        raise ValueError("Platt 학습 행 또는 target class가 부족합니다")
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(
        probability_logit(fit_frame["pred_selective"].to_numpy()),
        fit_frame["target"].to_numpy(dtype=np.int8),
    )
    return model, int(len(fit_frame))


def apply_platt(
    model: LogisticRegression,
    frame: pd.DataFrame,
    r_only: bool,
) -> np.ndarray:
    original = frame["pred_selective"].to_numpy(dtype=np.float64)
    output = original.copy()
    mask = (
        frame["game_type"].eq("R").to_numpy(dtype=bool)
        if r_only
        else np.ones(len(frame), dtype=bool)
    )
    output[mask] = model.predict_proba(probability_logit(original[mask]))[:, 1]
    if r_only and not np.array_equal(output[~mask], original[~mask]):
        raise AssertionError("R-only Platt가 F 예측을 변경했습니다")
    return output


def evaluate(oof: pd.DataFrame) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    results: dict[str, Any] = {}
    calibrators: dict[str, Any] = {}
    prediction_outputs: list[pd.DataFrame] = []

    # 2022 is the immutable guardrail because no earlier selective OOF exists.
    evaluation_2022 = oof.loc[oof["season"].eq(2022)].copy()
    base_2022 = evaluation_2022["pred_selective"].to_numpy(dtype=np.float64)
    results["2022"] = {
        "calibration_seasons": [],
        "policy": "identity_guardrail_no_earlier_selective_oof",
        "uncalibrated": segmented_metrics(evaluation_2022, base_2022),
        "platt_global": segmented_metrics(evaluation_2022, base_2022),
        "platt_r_only": segmented_metrics(evaluation_2022, base_2022),
    }
    output_2022 = evaluation_2022[REQUIRED_COLUMNS].copy()
    output_2022["pred_platt_global"] = base_2022
    output_2022["pred_platt_r_only"] = base_2022
    prediction_outputs.append(output_2022)

    for eval_season in [2023, 2024]:
        calibration = oof.loc[oof["season"].lt(eval_season)].copy()
        evaluation = oof.loc[oof["season"].eq(eval_season)].copy()
        if calibration.empty or evaluation.empty:
            raise ValueError(f"{eval_season} calibration/evaluation 행이 없습니다")
        calibration_seasons = sorted(
            int(value) for value in calibration["season"].unique()
        )
        global_model, global_rows = fit_platt(calibration, r_only=False)
        r_model, r_rows = fit_platt(calibration, r_only=True)
        baseline = evaluation["pred_selective"].to_numpy(dtype=np.float64)
        pred_global = apply_platt(global_model, evaluation, r_only=False)
        pred_r_only = apply_platt(r_model, evaluation, r_only=True)
        season_result = {
            "calibration_seasons": calibration_seasons,
            "calibration_rows": int(len(calibration)),
            "uncalibrated": segmented_metrics(evaluation, baseline),
            "platt_global": segmented_metrics(evaluation, pred_global),
            "platt_r_only": segmented_metrics(evaluation, pred_r_only),
        }
        baseline_brier = season_result["uncalibrated"]["overall"]["brier"]
        for method in ["platt_global", "platt_r_only"]:
            season_result[method]["overall"]["brier_delta"] = float(
                season_result[method]["overall"]["brier"] - baseline_brier
            )
        results[str(eval_season)] = season_result
        calibrators[str(eval_season)] = {
            "calibration_seasons": calibration_seasons,
            "platt_global": {
                "fit_rows": global_rows,
                "coefficient": float(global_model.coef_[0, 0]),
                "intercept": float(global_model.intercept_[0]),
            },
            "platt_r_only": {
                "fit_filter": "game_type == R",
                "apply_filter": "game_type == R",
                "other_rows": "identity",
                "fit_rows": r_rows,
                "coefficient": float(r_model.coef_[0, 0]),
                "intercept": float(r_model.intercept_[0]),
            },
        }
        output = evaluation[REQUIRED_COLUMNS].copy()
        output["pred_platt_global"] = pred_global
        output["pred_platt_r_only"] = pred_r_only
        prediction_outputs.append(output)

    gate: dict[str, Any] = {
        "definition": {
            "gate_a": (
                "2023 and 2024 Brier both improve; 2022 unchanged/worse <=0.00010; "
                "mean 2023/2024 improvement >=0.00010"
            ),
            "gate_b": (
                "2024 improvement >=0.00020; 2023 worse <=0.00005; "
                "2022 worse <=0.00010; ECE and gap not both worse"
            ),
        }
    }
    for method in ["platt_global", "platt_r_only"]:
        delta_2022 = float(
            results["2022"][method]["overall"]["brier"]
            - results["2022"]["uncalibrated"]["overall"]["brier"]
        )
        delta_2023 = float(results["2023"][method]["overall"]["brier_delta"])
        delta_2024 = float(results["2024"][method]["overall"]["brier_delta"])
        mean_improvement = float(-(delta_2023 + delta_2024) / 2.0)
        gate_a = bool(
            delta_2023 < 0.0
            and delta_2024 < 0.0
            and delta_2022 <= 0.00010
            and mean_improvement >= 0.00010
        )
        baseline_2023 = results["2023"]["uncalibrated"]["overall"]
        baseline_2024 = results["2024"]["uncalibrated"]["overall"]
        candidate_2023 = results["2023"][method]["overall"]
        candidate_2024 = results["2024"][method]["overall"]
        both_calibration_worse = bool(
            candidate_2023["ece_quantile_10"] > baseline_2023["ece_quantile_10"]
            and abs(candidate_2023["calibration_gap"])
            > abs(baseline_2023["calibration_gap"])
        ) or bool(
            candidate_2024["ece_quantile_10"] > baseline_2024["ece_quantile_10"]
            and abs(candidate_2024["calibration_gap"])
            > abs(baseline_2024["calibration_gap"])
        )
        gate_b = bool(
            delta_2024 <= -0.00020
            and delta_2023 <= 0.00005
            and delta_2022 <= 0.00010
            and not both_calibration_worse
        )
        gate[method] = {
            "delta_2022": delta_2022,
            "delta_2023": delta_2023,
            "delta_2024": delta_2024,
            "mean_2023_2024_improvement": mean_improvement,
            "gate_a": gate_a,
            "gate_b": gate_b,
            "pass": bool(gate_a or gate_b),
        }
    results["gate"] = gate
    all_predictions = pd.concat(prediction_outputs, ignore_index=True).sort_values(
        ["season", "row_id"]
    )
    return results, calibrators, all_predictions


def fit_future_calibrators(oof: pd.DataFrame) -> dict[str, Any]:
    global_model, global_rows = fit_platt(oof, r_only=False)
    r_model, r_rows = fit_platt(oof, r_only=True)
    r_frame = oof.loc[oof["game_type"].eq("R")]
    return {
        "schema_version": 1,
        "artifact_role": "future_inference_candidates_not_active",
        "recommended_by_temporal_gate": "platt_global",
        "input": {"type": "probability", "clip_epsilon": 1e-6},
        "candidates": {
            "platt_global": {
                "scope": "all_rows",
                "fit_rows": global_rows,
                "parameters": {
                    "coefficient": float(global_model.coef_[0, 0]),
                    "intercept": float(global_model.intercept_[0]),
                },
            },
            "platt_r_only": {
                "scope": {
                    "column": "game_type",
                    "apply_value": "R",
                    "other_rows": "identity",
                },
                "fit_rows": r_rows,
                "parameters": {
                    "coefficient": float(r_model.coef_[0, 0]),
                    "intercept": float(r_model.intercept_[0]),
                },
            },
        },
        "fit_contract": {
            "mode": "strict_temporal_selective_oof",
            "oof_seasons": sorted(int(value) for value in oof["season"].unique()),
            "oof_rows_total": int(len(oof)),
            "r_rows_selected": r_rows,
            "r_target_mean": float(r_frame["target"].mean()),
            "r_prediction_mean": float(r_frame["pred_selective"].mean()),
        },
        "evaluation_policy": (
            "Do not evaluate this all-OOF future fit on its source rows. "
            "Use the forward 2023/2024 gate results for selection."
        ),
        "active_model_sync": False,
    }


def render_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# CAL-SEL-OOF-001 선택형 시간 OOF 보정",
        "",
        "- 입력: 실제 제출 규칙으로 생성된 선택형 앙상블 OOF 예측",
        "- 2022: 이전 선택형 OOF 부재로 identity guardrail",
        "- 2023/2024: 평가 시즌보다 과거인 선택형 OOF만 보정기 학습에 사용",
        "- F는 identity, R-only Platt를 주 후보로 평가",
        "",
        "| 시즌 | 학습 시즌 | 방법 | Brier | 기준 대비 | BSS | ECE10 | gap |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for season in [2022, 2023, 2024]:
        season_result = results[str(season)]
        for method in ["uncalibrated", "platt_global", "platt_r_only"]:
            overall = season_result[method]["overall"]
            delta = overall.get("brier_delta", 0.0)
            lines.append(
                f"| {season} | {season_result['calibration_seasons']} | {method} | "
                f"{overall['brier']:.9f} | {delta:+.9f} | {overall['bss']:.9f} | "
                f"{overall['ece_quantile_10']:.9f} | {overall['calibration_gap']:+.9f} |"
            )
    lines.extend(["", "## Gate A/B 판정", ""])
    for method in ["platt_global", "platt_r_only"]:
        gate = results["gate"][method]
        lines.extend(
            [
                f"### {method}",
                "",
                f"- 2023 Brier delta: `{gate['delta_2023']:+.9f}`",
                f"- 2024 Brier delta: `{gate['delta_2024']:+.9f}`",
                f"- 2023/2024 평균 개선: `{gate['mean_2023_2024_improvement']:.9f}`",
                f"- Gate A: `{'PASS' if gate['gate_a'] else 'FAIL'}`",
                f"- Gate B: `{'PASS' if gate['gate_b'] else 'FAIL'}`",
                f"- 종합: `{'PASS' if gate['pass'] else 'FAIL'}`",
                "",
            ]
        )
    lines.extend(
        [
            "## 규정 및 상태",
            "",
            "- 보정 학습에는 공식 train의 과거 시즌 OOF만 사용했다.",
            "- test 예측 분포와 test 행 간 통계를 사용하지 않았다.",
            "- 미래용 보정기는 격리 후보이며 활성 제출에는 연결하지 않았다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    oof, sources = load_oof()
    results, calibrators, predictions = evaluate(oof)
    future_calibrators = fit_future_calibrators(oof)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = args.output_dir / "calibrated_predictions.csv"
    results_path = args.output_dir / "calibration_results.json"
    calibrators_path = args.output_dir / "evaluation_calibrators.json"
    future_path = args.output_dir / "future_platt_candidates.json"
    report_path = args.output_dir / "calibration_results.md"
    contract_path = args.output_dir / "oof_contract.json"
    predictions.to_csv(predictions_path, index=False)
    contract = {
        "experiment_id": "CAL-SEL-OOF-001",
        "mode": "strict_temporal_selective_oof",
        "sources": sources,
        "oof_seasons": sorted(PREDICTION_FILES),
        "oof_rows": int(len(oof)),
        "selection_rule": {
            "F": {"lightgbm": 0.0, "catboost": 1.0},
            "R": {"lightgbm": 0.5, "catboost": 0.5},
        },
        "test_distribution_used": False,
        "external_data_used": False,
        "active_model_sync": False,
    }
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    results_payload = {
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "contract_file": contract_path.name,
        "predictions_file": predictions_path.name,
        "future_calibrator_file": future_path.name,
        "results": results,
    }
    results_path.write_text(
        json.dumps(results_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    calibrators_path.write_text(
        json.dumps(calibrators, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    future_path.write_text(
        json.dumps(future_calibrators, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    report_path.write_text(render_markdown(results), encoding="utf-8")
    print(report_path.read_text(encoding="utf-8"))
    print(f"Saved results: {results_path}")
    print(f"Results SHA-256: {sha256_file(results_path)}")


if __name__ == "__main__":
    main()
