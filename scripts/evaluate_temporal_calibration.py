#!/usr/bin/env python3
"""Build strict expanding-season OOF predictions and evaluate calibrators."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import brier_score, brier_skill_score


DEFAULT_EXPERIMENT_DIRS = [
    ROOT / "model" / "LGBM-FE001-EW-2021",
    ROOT / "model" / "LGBM-FE001-EW-2022",
    ROOT / "model" / "LGBM-FE001-EW-2023",
    ROOT / "model" / "LGBM-FE001-2024",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Platt and isotonic calibration without holdout reuse"
    )
    parser.add_argument(
        "--experiment-dirs",
        nargs="+",
        type=Path,
        default=DEFAULT_EXPERIMENT_DIRS,
        help="Expanding-season experiment directories in any order",
    )
    parser.add_argument(
        "--eval-seasons",
        default="2023,2024",
        help="Comma-separated seasons evaluated using only earlier OOF seasons",
    )
    parser.add_argument("--train-path", type=Path, default=ROOT / "data" / "train.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "model" / "CAL-FE001-TEMPORAL-OOF",
    )
    return parser.parse_args()


def probability_logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile_ece(y_true: np.ndarray, prediction: np.ndarray, bins: int = 10) -> float:
    frame = pd.DataFrame(
        {
            "target": np.asarray(y_true, dtype=np.float64),
            "prediction": np.asarray(prediction, dtype=np.float64),
        }
    )
    frame["bin"] = pd.qcut(
        frame["prediction"], q=bins, labels=False, duplicates="drop"
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
    y_true = np.asarray(y_true, dtype=np.int8)
    prediction = np.asarray(prediction, dtype=np.float64)
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
    frame: pd.DataFrame, prediction: np.ndarray
) -> dict[str, dict[str, float | int]]:
    output = {"overall": metrics(frame["target"].to_numpy(), prediction)}
    game_type = frame["game_type"].astype("string")
    for value in ["F", "R"]:
        mask = game_type.eq(value).fillna(False).to_numpy(dtype=bool)
        output[f"game_type_{value.lower()}"] = metrics(
            frame.loc[mask, "target"].to_numpy(), prediction[mask]
        )
    return output


def load_oof_contract(
    experiment_dirs: list[Path], train_path: Path
) -> tuple[pd.DataFrame, dict[str, Any]]:
    frames: list[pd.DataFrame] = []
    sources: list[dict[str, Any]] = []
    reference_features: list[str] | None = None
    reference_hyperparameters: dict[str, Any] | None = None
    seen_seasons: set[int] = set()

    for experiment_dir in experiment_dirs:
        metadata_path = experiment_dir / "metadata.json"
        predictions_path = experiment_dir / "validation_predictions.csv"
        features_path = experiment_dir / "feature_columns.json"
        for path in [metadata_path, predictions_path, features_path]:
            if not path.is_file():
                raise FileNotFoundError(f"필수 실험 산출물이 없습니다: {path}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        feature_columns = json.loads(features_path.read_text(encoding="utf-8"))
        valid_season = int(metadata["valid_season"])
        if valid_season in seen_seasons:
            raise ValueError(f"검증 시즌이 중복되었습니다: {valid_season}")
        seen_seasons.add(valid_season)
        expected_train_seasons = list(range(2019, valid_season))
        if metadata["train_seasons"] != expected_train_seasons:
            raise ValueError(
                f"{metadata['exp_id']}가 expanding-season 계약과 다릅니다: "
                f"{metadata['train_seasons']} != {expected_train_seasons}"
            )
        if metadata.get("target_aggregate") is not None:
            raise ValueError(f"OOF 기준 모델에 타깃 집계가 포함되었습니다: {metadata['exp_id']}")
        if int(metadata["feature_count"]) != 60 or len(feature_columns) != 60:
            raise ValueError(f"FE-001 60피처 계약이 아닙니다: {metadata['exp_id']}")
        if reference_features is None:
            reference_features = feature_columns
            reference_hyperparameters = metadata["hyperparameters"]
        else:
            if feature_columns != reference_features:
                raise ValueError(f"피처 계약이 다릅니다: {metadata['exp_id']}")
            if metadata["hyperparameters"] != reference_hyperparameters:
                raise ValueError(f"하이퍼파라미터 계약이 다릅니다: {metadata['exp_id']}")

        frame = pd.read_csv(
            predictions_path,
            usecols=["row_id", "season", "target", "pred_lgbm"],
        )
        if frame.empty or not frame["season"].eq(valid_season).all():
            raise ValueError(f"예측 파일의 시즌 계약이 다릅니다: {metadata['exp_id']}")
        if frame["row_id"].duplicated().any():
            raise ValueError(f"예측 파일에 row_id 중복이 있습니다: {metadata['exp_id']}")
        if not set(frame["target"].unique()).issubset({0, 1}):
            raise ValueError(f"타깃이 이진값이 아닙니다: {metadata['exp_id']}")
        if not np.isfinite(frame["pred_lgbm"]).all() or not frame[
            "pred_lgbm"
        ].between(0.0, 1.0).all():
            raise ValueError(f"예측 확률 범위가 잘못되었습니다: {metadata['exp_id']}")
        rebuilt_brier = float(brier_score(frame["target"], frame["pred_lgbm"]))
        stored_brier = float(metadata["metrics"]["lgbm"]["brier_score"])
        if not np.isclose(rebuilt_brier, stored_brier, rtol=0.0, atol=1e-15):
            raise ValueError(
                f"저장 Brier와 예측 재계산값이 다릅니다: {metadata['exp_id']}"
            )
        frame["source_exp_id"] = metadata["exp_id"]
        frames.append(frame)
        sources.append(
            {
                "exp_id": metadata["exp_id"],
                "experiment_dir": str(experiment_dir),
                "train_seasons": metadata["train_seasons"],
                "valid_season": valid_season,
                "rows": int(len(frame)),
                "best_iteration": int(metadata["best_iteration"]),
                "brier": rebuilt_brier,
            }
        )

    oof = pd.concat(frames, ignore_index=True).sort_values(
        ["season", "row_id"]
    ).reset_index(drop=True)
    if oof["row_id"].duplicated().any():
        raise ValueError("서로 다른 OOF 시즌 사이에 row_id가 중복되었습니다")

    row_context = pd.read_csv(train_path, usecols=["row_id", "game_type"])
    if row_context["row_id"].duplicated().any():
        raise ValueError("원본 train 데이터의 row_id가 중복되었습니다")
    oof = oof.merge(row_context, how="left", on="row_id", validate="one_to_one")
    if oof["game_type"].isna().any():
        raise ValueError("OOF row_id와 원본 game_type의 결합에 실패했습니다")
    oof["target"] = oof["target"].astype("int8")
    oof["season"] = oof["season"].astype("int16")

    contract = {
        "mode": "strict_expanding_season_oof",
        "source_count": len(sources),
        "sources": sorted(sources, key=lambda item: item["valid_season"]),
        "oof_seasons": sorted(int(value) for value in oof["season"].unique()),
        "oof_rows": int(len(oof)),
        "feature_count": len(reference_features or []),
        "feature_columns": reference_features,
        "hyperparameters": reference_hyperparameters,
    }
    return oof, contract


def evaluate_calibration(
    oof: pd.DataFrame, eval_seasons: list[int]
) -> tuple[dict[str, Any], dict[str, Any]]:
    results: dict[str, Any] = {}
    calibrators: dict[str, Any] = {}
    for eval_season in eval_seasons:
        calibration = oof.loc[oof["season"].lt(eval_season)].copy()
        evaluation = oof.loc[oof["season"].eq(eval_season)].copy()
        calibration_seasons = sorted(
            int(value) for value in calibration["season"].unique()
        )
        if len(calibration_seasons) < 2:
            raise ValueError(
                f"{eval_season} 평가 전 calibration OOF 시즌이 2개 미만입니다"
            )
        if evaluation.empty:
            raise ValueError(f"평가 OOF 시즌이 없습니다: {eval_season}")

        cal_y = calibration["target"].to_numpy(dtype=np.int8)
        cal_p = calibration["pred_lgbm"].to_numpy(dtype=np.float64)
        eval_p = evaluation["pred_lgbm"].to_numpy(dtype=np.float64)

        platt = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        platt.fit(probability_logit(cal_p), cal_y)
        pred_platt = platt.predict_proba(probability_logit(eval_p))[:, 1]

        isotonic = IsotonicRegression(out_of_bounds="clip")
        isotonic.fit(cal_p, cal_y)
        pred_isotonic = isotonic.predict(eval_p)

        calibration_r_mask = (
            calibration["game_type"]
            .astype("string")
            .eq("R")
            .fillna(False)
            .to_numpy(dtype=bool)
        )
        evaluation_r_mask = (
            evaluation["game_type"]
            .astype("string")
            .eq("R")
            .fillna(False)
            .to_numpy(dtype=bool)
        )
        platt_r_only = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
        platt_r_only.fit(
            probability_logit(cal_p[calibration_r_mask]),
            cal_y[calibration_r_mask],
        )
        pred_platt_r_only = eval_p.copy()
        pred_platt_r_only[evaluation_r_mask] = platt_r_only.predict_proba(
            probability_logit(eval_p[evaluation_r_mask])
        )[:, 1]
        if not np.array_equal(
            pred_platt_r_only[~evaluation_r_mask],
            eval_p[~evaluation_r_mask],
        ):
            raise AssertionError("R 전용 보정이 F 예측값을 변경했습니다")

        season_result = {
            "calibration_seasons": calibration_seasons,
            "calibration_rows": int(len(calibration)),
            "evaluation_rows": int(len(evaluation)),
            "uncalibrated": segmented_metrics(evaluation, eval_p),
            "platt": segmented_metrics(evaluation, pred_platt),
            "isotonic": segmented_metrics(evaluation, pred_isotonic),
            "platt_r_only": segmented_metrics(evaluation, pred_platt_r_only),
        }
        baseline_brier = season_result["uncalibrated"]["overall"]["brier"]
        for method in ["platt", "isotonic", "platt_r_only"]:
            method_brier = season_result[method]["overall"]["brier"]
            season_result[method]["overall"]["brier_delta"] = float(
                method_brier - baseline_brier
            )
        results[str(eval_season)] = season_result
        calibrators[str(eval_season)] = {
            "calibration_seasons": calibration_seasons,
            "platt": {
                "input": "logit_clipped_1e-6",
                "coefficient": float(platt.coef_[0, 0]),
                "intercept": float(platt.intercept_[0]),
            },
            "isotonic": {
                "out_of_bounds": "clip",
                "x_thresholds": isotonic.X_thresholds_.tolist(),
                "y_thresholds": isotonic.y_thresholds_.tolist(),
            },
            "platt_r_only": {
                "input": "logit_clipped_1e-6",
                "fit_filter": "game_type == R",
                "apply_filter": "game_type == R",
                "other_rows": "identity",
                "fit_rows": int(calibration_r_mask.sum()),
                "coefficient": float(platt_r_only.coef_[0, 0]),
                "intercept": float(platt_r_only.intercept_[0]),
            },
        }

    method_gate = {
        method: all(
            results[str(season)][method]["overall"]["brier_delta"] < 0.0
            for season in eval_seasons
        )
        for method in ["platt", "isotonic", "platt_r_only"]
    }
    results["gate"] = {
        "rule": "Brier must improve on every evaluation season",
        "evaluation_seasons": eval_seasons,
        "method_pass": method_gate,
    }
    return results, calibrators


def fit_future_r_only_platt(
    oof: pd.DataFrame,
    oof_path: Path,
    contract_path: Path,
) -> dict[str, Any]:
    """Fit a deployment-shaped calibrator on all available strict temporal OOF R rows."""

    r_mask = (
        oof["game_type"]
        .astype("string")
        .eq("R")
        .fillna(False)
        .to_numpy(dtype=bool)
    )
    if not r_mask.any():
        raise ValueError("미래 보정기 학습용 game_type=R OOF 행이 없습니다")
    target = oof.loc[r_mask, "target"].to_numpy(dtype=np.int8)
    prediction = oof.loc[r_mask, "pred_lgbm"].to_numpy(dtype=np.float64)
    calibrator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    calibrator.fit(probability_logit(prediction), target)
    return {
        "schema_version": 1,
        "artifact_role": "future_inference_candidate",
        "method": "platt_logit",
        "input": {
            "type": "probability",
            "clip_epsilon": 1e-6,
        },
        "scope": {
            "column": "game_type",
            "apply_value": "R",
            "other_rows": "identity",
        },
        "parameters": {
            "coefficient": float(calibrator.coef_[0, 0]),
            "intercept": float(calibrator.intercept_[0]),
        },
        "fit_contract": {
            "mode": "strict_expanding_season_oof",
            "oof_seasons": sorted(int(value) for value in oof["season"].unique()),
            "oof_rows_total": int(len(oof)),
            "oof_rows_selected": int(r_mask.sum()),
            "target_mean_selected": float(target.mean()),
            "source_oof_file": oof_path.name,
            "source_oof_sha256": sha256_file(oof_path),
            "source_contract_file": contract_path.name,
            "source_contract_sha256": sha256_file(contract_path),
        },
        "evaluation_policy": (
            "Do not score this all-OOF fit on its source rows; use only for future "
            "inference after the temporal evaluation gate has passed."
        ),
    }


def render_markdown(contract: dict[str, Any], results: dict[str, Any]) -> str:
    lines = [
        "# FE-001 시간 OOF 확률 보정 평가",
        "",
        f"- OOF 시즌: `{contract['oof_seasons']}`",
        f"- OOF 행 수: `{contract['oof_rows']:,}`",
        "- 원칙: 평가 시즌보다 이른 OOF 시즌만 보정기 학습에 사용",
        "",
        "| 평가 시즌 | 보정 학습 시즌 | 방법 | Brier | 기준 대비 | BSS | ECE10 | 평균 오차 |",
        "|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for season in results["gate"]["evaluation_seasons"]:
        season_result = results[str(season)]
        for method in ["uncalibrated", "platt", "isotonic", "platt_r_only"]:
            overall = season_result[method]["overall"]
            delta = overall.get("brier_delta", 0.0)
            lines.append(
                f"| {season} | {season_result['calibration_seasons']} | {method} | "
                f"{overall['brier']:.9f} | {delta:+.9f} | {overall['bss']:.9f} | "
                f"{overall['ece_quantile_10']:.9f} | {overall['calibration_gap']:+.9f} |"
            )
    lines.extend(
        [
            "",
            "## 게이트 판정",
            "",
            f"- Platt: `{'PASS' if results['gate']['method_pass']['platt'] else 'FAIL'}`",
            f"- Isotonic: `{'PASS' if results['gate']['method_pass']['isotonic'] else 'FAIL'}`",
            "- R-only Platt: "
            f"`{'PASS' if results['gate']['method_pass']['platt_r_only'] else 'FAIL'}`",
            "- 모든 평가 시즌의 Brier를 개선한 방법만 채택 후보로 인정한다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    eval_seasons = [
        int(value.strip()) for value in args.eval_seasons.split(",") if value.strip()
    ]
    if len(eval_seasons) != len(set(eval_seasons)):
        raise ValueError("평가 시즌이 중복되었습니다")

    oof, contract = load_oof_contract(args.experiment_dirs, args.train_path)
    results, calibrators = evaluate_calibration(oof, eval_seasons)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    oof_path = args.output_dir / "expanding_oof_predictions.csv"
    results_path = args.output_dir / "calibration_results.json"
    calibrators_path = args.output_dir / "evaluation_calibrators.json"
    future_calibrator_path = args.output_dir / "future_r_only_platt.json"
    report_path = args.output_dir / "calibration_results.md"
    contract_path = args.output_dir / "oof_contract.json"
    oof.to_csv(oof_path, index=False)
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    future_calibrator = fit_future_r_only_platt(oof, oof_path, contract_path)
    future_calibrator_path.write_text(
        json.dumps(future_calibrator, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    results_payload = {
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
        },
        "oof_contract_file": contract_path.name,
        "oof_predictions_file": oof_path.name,
        "future_calibrator_file": future_calibrator_path.name,
        "results": results,
    }
    results_path.write_text(
        json.dumps(results_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    calibrators_path.write_text(
        json.dumps(calibrators, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path.write_text(render_markdown(contract, results), encoding="utf-8")

    print(report_path.read_text(encoding="utf-8"))
    print(f"Saved OOF: {oof_path}")
    print(f"Saved contract: {contract_path}")
    print(f"Saved results: {results_path}")
    print(f"Saved evaluation calibrators: {calibrators_path}")
    print(f"Saved future R-only Platt candidate: {future_calibrator_path}")


if __name__ == "__main__":
    main()
