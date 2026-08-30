#!/usr/bin/env python3
"""Build the DIAG-1000-001 error map from train-only temporal validation data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


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
CONTEXT_COLUMNS = [
    "row_id",
    "season",
    "game_month",
    "inning",
    "game_type",
    "balls_before",
    "strikes_before",
    "score_diff_pitcher_team",
    "num_runners_on",
    "base_state",
    "li",
    "pitcher_id",
    "batter_id",
    "asof_pitcher_n",
    "asof_batter_n",
    "asof_pitcher_pitchmix_n",
    "control_success",
]
PREDICTION_COLUMNS = [
    "row_id",
    "season",
    "target",
    "pred_lgbm",
    "pred_catboost",
    "pred_ensemble_50_50",
    "game_type",
    "pred_selective",
]
MODEL_COLUMNS = {
    "lightgbm": "pred_lgbm",
    "catboost": "pred_catboost",
    "ensemble_50_50": "pred_ensemble_50_50",
    "selective": "pred_selective",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, default=ROOT / "data" / "train.csv")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "model" / "DIAG-1000-001",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def brier(y: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.square(y - prediction)))


def ece_quantile(y: np.ndarray, prediction: np.ndarray, bins: int = 10) -> float:
    frame = pd.DataFrame({"target": y, "prediction": prediction})
    rank = frame["prediction"].rank(method="first")
    groups = pd.qcut(rank, q=min(bins, len(frame)), labels=False, duplicates="drop")
    grouped = frame.groupby(groups, observed=True).agg(
        rows=("target", "size"),
        target_mean=("target", "mean"),
        prediction_mean=("prediction", "mean"),
    )
    return float(
        np.average(
            np.abs(grouped["prediction_mean"] - grouped["target_mean"]),
            weights=grouped["rows"],
        )
    )


def metric_payload(frame: pd.DataFrame, prediction_column: str) -> dict[str, Any]:
    y = frame["target"].to_numpy(dtype=np.float64)
    prediction = frame[prediction_column].to_numpy(dtype=np.float64)
    reference = float(np.mean(y) * (1.0 - np.mean(y)))
    score = brier(y, prediction)
    return {
        "rows": int(len(frame)),
        "target_mean": float(np.mean(y)),
        "prediction_mean": float(np.mean(prediction)),
        "calibration_gap": float(np.mean(prediction) - np.mean(y)),
        "brier": score,
        "bss": float(1.0 - score / reference) if reference > 0 else 0.0,
        "ece_quantile_10": ece_quantile(y, prediction),
        "squared_error_sum": float(np.square(y - prediction).sum()),
    }


def count_state(frame: pd.DataFrame) -> pd.Series:
    balls = frame["balls_before"]
    strikes = frame["strikes_before"]
    full = balls.eq(3) & strikes.eq(2)
    pitcher_ahead = (
        (balls.eq(0) & strikes.isin([1, 2]))
        | (balls.eq(1) & strikes.eq(2))
    )
    batter_ahead = (
        (balls.eq(2) & strikes.eq(0))
        | (balls.eq(3) & strikes.isin([0, 1]))
    )
    return pd.Series(
        np.select(
            [full, pitcher_ahead, batter_ahead],
            ["full_count", "ahead_pitcher", "ahead_batter"],
            default="neutral",
        ),
        index=frame.index,
        dtype="string",
    )


def quantile_label(series: pd.Series, bins: int = 10, prefix: str = "q") -> pd.Series:
    ranks = series.rank(method="first")
    labels = pd.qcut(ranks, q=bins, labels=False, duplicates="drop")
    return labels.map(lambda value: f"{prefix}{int(value) + 1:02d}").astype("string")


def add_context_features(frame: pd.DataFrame, all_context: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame["count_state"] = count_state(frame)
    frame["exact_count"] = (
        frame["balls_before"].astype("string")
        + "-"
        + frame["strikes_before"].astype("string")
    )
    frame["inning_bucket"] = pd.cut(
        frame["inning"],
        bins=[-np.inf, 3, 6, 9, np.inf],
        labels=["01-03", "04-06", "07-09", "10+"],
    ).astype("string")
    score = frame["score_diff_pitcher_team"]
    frame["score_margin_bucket"] = pd.Series(
        np.select(
            [score.le(-4), score.between(-3, -1), score.eq(0), score.between(1, 3)],
            ["behind_4+", "behind_1_3", "tied", "ahead_1_3"],
            default="ahead_4+",
        ),
        index=frame.index,
        dtype="string",
    )
    frame["leverage_bucket"] = pd.cut(
        frame["li"],
        bins=[-np.inf, 0.5, 1.0, 2.0, np.inf],
        labels=["li<=0.5", "0.5<li<=1", "1<li<=2", "li>2"],
    ).astype("string")
    frame["asof_cold_start"] = np.where(
        frame[["asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n"]]
        .le(0)
        .any(axis=1),
        "cold",
        "known",
    )
    frame["model_disagreement"] = np.abs(frame["pred_lgbm"] - frame["pred_catboost"])
    frame["prediction_decile"] = quantile_label(frame["pred_selective"], prefix="p")
    frame["disagreement_decile"] = quantile_label(frame["model_disagreement"], prefix="d")

    valid_season = int(frame["season"].iloc[0])
    history = all_context[all_context["season"].lt(valid_season)]
    known_pitchers = set(history["pitcher_id"].dropna().tolist())
    known_batters = set(history["batter_id"].dropna().tolist())
    pitcher_new = ~frame["pitcher_id"].isin(known_pitchers)
    batter_new = ~frame["batter_id"].isin(known_batters)
    frame["entity_cold_start"] = pd.Series(
        np.select(
            [pitcher_new & batter_new, pitcher_new, batter_new],
            ["both_new", "pitcher_new", "batter_new"],
            default="both_known",
        ),
        index=frame.index,
        dtype="string",
    )
    return frame


def validate_and_load(train_path: Path) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    context = pd.read_csv(
        train_path,
        encoding="utf-8-sig",
        usecols=CONTEXT_COLUMNS,
    )
    if context["row_id"].isna().any() or context["row_id"].duplicated().any():
        raise ValueError("train row_id에 결측 또는 중복이 있습니다")

    prediction_frames: dict[int, pd.DataFrame] = {}
    for season, path in PREDICTION_FILES.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        prediction = pd.read_csv(path, usecols=PREDICTION_COLUMNS)
        if prediction["row_id"].isna().any() or prediction["row_id"].duplicated().any():
            raise ValueError(f"{season} prediction row_id 계약이 깨졌습니다")
        if not prediction["season"].eq(season).all():
            raise ValueError(f"{season} prediction season 계약이 깨졌습니다")
        if not np.allclose(
            prediction["pred_selective"],
            np.where(
                prediction["game_type"].eq("F"),
                prediction["pred_catboost"],
                prediction["pred_ensemble_50_50"],
            ),
            rtol=0.0,
            atol=0.0,
        ):
            raise ValueError(f"{season} 선택형 예측 identity가 깨졌습니다")

        season_context = context[context["season"].eq(season)]
        merged = prediction.merge(
            season_context,
            on=["row_id", "season"],
            how="left",
            suffixes=("_prediction", ""),
            validate="one_to_one",
        )
        if merged["control_success"].isna().any():
            raise ValueError(f"{season} prediction과 train context 결합에 실패했습니다")
        if not np.array_equal(
            merged["target"].to_numpy(), merged["control_success"].to_numpy()
        ):
            raise ValueError(f"{season} prediction target이 train 정답과 다릅니다")
        if not merged["game_type_prediction"].astype("string").equals(
            merged["game_type"].astype("string")
        ):
            raise ValueError(f"{season} game_type 계약이 다릅니다")
        merged = merged.drop(columns=["control_success", "game_type_prediction"])
        prediction_frames[season] = add_context_features(merged, context)
    return context, prediction_frames


def group_metrics(frame: pd.DataFrame, dimension: str) -> list[dict[str, Any]]:
    total_loss = float(
        np.square(frame["target"] - frame["pred_selective"]).sum()
    )
    results: list[dict[str, Any]] = []
    for value, group in frame.groupby(dimension, observed=True, dropna=False):
        row: dict[str, Any] = {
            "group": str(value),
            "row_share": float(len(group) / len(frame)),
        }
        for method, column in MODEL_COLUMNS.items():
            metrics = metric_payload(group, column)
            row[method] = metrics
        row["selective"]["loss_contribution"] = float(
            row["selective"]["squared_error_sum"] / total_loss
        )
        row["delta_selective_vs_lightgbm"] = float(
            row["selective"]["brier"] - row["lightgbm"]["brier"]
        )
        row["delta_selective_vs_catboost"] = float(
            row["selective"]["brier"] - row["catboost"]["brier"]
        )
        results.append(row)
    return sorted(results, key=lambda item: (-item["row_share"], item["group"]))


def disagreement_summary(frame: pd.DataFrame) -> dict[str, Any]:
    y = frame["target"].to_numpy(dtype=np.float64)
    residual_lgbm = y - frame["pred_lgbm"].to_numpy(dtype=np.float64)
    residual_cat = y - frame["pred_catboost"].to_numpy(dtype=np.float64)
    prediction_corr = float(frame[["pred_lgbm", "pred_catboost"]].corr().iloc[0, 1])
    residual_corr = float(np.corrcoef(residual_lgbm, residual_cat)[0, 1])
    lgbm_sq = np.square(residual_lgbm)
    cat_sq = np.square(residual_cat)
    tie = np.isclose(lgbm_sq, cat_sq, rtol=0.0, atol=1e-15)
    return {
        "prediction_correlation": prediction_corr,
        "residual_correlation": residual_corr,
        "mean_absolute_disagreement": float(frame["model_disagreement"].mean()),
        "p90_absolute_disagreement": float(frame["model_disagreement"].quantile(0.90)),
        "p99_absolute_disagreement": float(frame["model_disagreement"].quantile(0.99)),
        "lightgbm_row_win_rate": float(np.mean((lgbm_sq < cat_sq) & ~tie)),
        "catboost_row_win_rate": float(np.mean((cat_sq < lgbm_sq) & ~tie)),
        "tie_rate": float(np.mean(tie)),
    }


def stable_group_priorities(
    dimensions: dict[str, dict[str, list[dict[str, Any]]]],
) -> list[dict[str, Any]]:
    priorities: list[dict[str, Any]] = []
    for dimension, seasons in dimensions.items():
        grouped: dict[str, list[tuple[int, dict[str, Any]]]] = {}
        for season_text, rows in seasons.items():
            for row in rows:
                grouped.setdefault(row["group"], []).append((int(season_text), row))
        for group, season_rows in grouped.items():
            if len(season_rows) != len(PREDICTION_FILES):
                continue
            if min(row["selective"]["rows"] for _, row in season_rows) < 1000:
                continue
            gaps = [row["selective"]["calibration_gap"] for _, row in season_rows]
            signs = {int(np.sign(value)) for value in gaps if abs(value) >= 1e-6}
            total_rows = sum(row["selective"]["rows"] for _, row in season_rows)
            weighted_gap = sum(
                row["selective"]["rows"] * row["selective"]["calibration_gap"]
                for _, row in season_rows
            ) / total_rows
            priorities.append(
                {
                    "dimension": dimension,
                    "group": group,
                    "seasons": [season for season, _ in season_rows],
                    "rows": int(total_rows),
                    "weighted_calibration_gap": float(weighted_gap),
                    "same_gap_direction": len(signs) <= 1,
                    "season_gaps": {
                        str(season): float(row["selective"]["calibration_gap"])
                        for season, row in season_rows
                    },
                    "priority_mass": float(abs(weighted_gap) * total_rows),
                }
            )
    return sorted(priorities, key=lambda item: -item["priority_mass"])


def build_findings(results: dict[str, Any]) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    r_rows = []
    for season in sorted(PREDICTION_FILES):
        rows = results["dimensions"]["game_type"][str(season)]
        r_rows.append(next(row for row in rows if row["group"] == "R"))
    r_gaps = [row["selective"]["calibration_gap"] for row in r_rows]
    if all(value > 0 for value in r_gaps):
        findings.append(
            {
                "priority": "P1",
                "finding": "game_type=R에서 2022·2023·2024 모두 평균 예측이 실제보다 높다.",
                "action": "최종 선택형 R 예측 자체에 expanding-OOF Platt를 적용하는 CAL-SEL-OOF-001을 최우선 검증한다.",
            }
        )

    disagreement = results["dimensions"]["disagreement_decile"]
    high_rows = []
    for season in sorted(PREDICTION_FILES):
        row = next(
            item
            for item in disagreement[str(season)]
            if item["group"] == "d10"
        )
        high_rows.append(row)
    if all(row["selective"]["rows"] > 0 for row in high_rows):
        deltas = [row["delta_selective_vs_catboost"] for row in high_rows]
        direction = "CatBoost" if all(value > 0 for value in deltas) else "혼합"
        findings.append(
            {
                "priority": "P2",
                "finding": f"두 모델 불일치 상위 10%의 CatBoost 대비 선택형 방향은 {direction}으로 관측됐다.",
                "action": "다중 시드로 모델별 분산을 줄인 뒤 동일 disagreement 진단을 반복한다. validation disagreement를 test 선택 규칙으로 직접 사용하지 않는다.",
            }
        )

    stable = [
        item
        for item in results["stable_calibration_priorities"]
        if item["same_gap_direction"]
    ]
    if stable:
        top = stable[0]
        findings.append(
            {
                "priority": "P3",
                "finding": (
                    f"안정적 calibration 우선 구간은 {top['dimension']}={top['group']}이며 "
                    f"가중 gap은 {top['weighted_calibration_gap']:+.6f}다."
                ),
                "action": "FE-002는 이 구간의 현재 행 상호작용을 우선 설계하고 그룹별 고정값 보정은 사용하지 않는다.",
            }
        )
    return findings


def markdown_table(rows: list[list[str]], headers: list[str]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def render_markdown(results: dict[str, Any]) -> str:
    lines = [
        "# DIAG-1000-001 기준선 오차 지도",
        "",
        "- 데이터: 공식 train의 2022·2023·2024 시간 Holdout 예측만 사용",
        "- 기준선: `F=CatBoost`, `R=LightGBM 0.5 + CatBoost 0.5` 선택형 앙상블",
        "- test 행·test 분포·외부 데이터·외부 API 사용 없음",
        "",
        "## 전체 시즌 지표",
        "",
    ]
    overall_rows: list[list[str]] = []
    for season in sorted(PREDICTION_FILES):
        season_metrics = results["overall"][str(season)]["selective"]
        overall_rows.append(
            [
                str(season),
                f"{season_metrics['rows']:,}",
                f"{season_metrics['brier']:.9f}",
                f"{season_metrics['bss']:.9f}",
                f"{season_metrics['ece_quantile_10']:.9f}",
                f"{season_metrics['calibration_gap']:+.9f}",
            ]
        )
    lines.extend(
        markdown_table(
            overall_rows,
            ["시즌", "행", "Brier", "BSS", "ECE10", "Calibration gap"],
        )
    )
    lines.extend(["", "## game_type 오차", ""])
    game_rows: list[list[str]] = []
    for season in sorted(PREDICTION_FILES):
        for row in results["dimensions"]["game_type"][str(season)]:
            selective = row["selective"]
            game_rows.append(
                [
                    str(season),
                    row["group"],
                    f"{selective['rows']:,}",
                    f"{selective['brier']:.9f}",
                    f"{selective['calibration_gap']:+.9f}",
                    f"{row['delta_selective_vs_catboost']:+.9f}",
                ]
            )
    lines.extend(
        markdown_table(
            game_rows,
            ["시즌", "type", "행", "선택형 Brier", "gap", "vs CatBoost"],
        )
    )
    lines.extend(["", "## 모델 불일치", ""])
    disagreement_rows = []
    for season in sorted(PREDICTION_FILES):
        row = results["disagreement"][str(season)]
        disagreement_rows.append(
            [
                str(season),
                f"{row['prediction_correlation']:.6f}",
                f"{row['residual_correlation']:.6f}",
                f"{row['mean_absolute_disagreement']:.6f}",
                f"{row['p90_absolute_disagreement']:.6f}",
                f"{row['catboost_row_win_rate']:.4f}",
            ]
        )
    lines.extend(
        markdown_table(
            disagreement_rows,
            ["시즌", "예측 상관", "잔차 상관", "평균 |Δ|", "p90 |Δ|", "CAT 행 승률"],
        )
    )
    lines.extend(["", "## 안정적 calibration 우선 구간", ""])
    stable_rows = []
    for row in results["stable_calibration_priorities"][:15]:
        stable_rows.append(
            [
                row["dimension"],
                row["group"],
                f"{row['rows']:,}",
                f"{row['weighted_calibration_gap']:+.6f}",
                "YES" if row["same_gap_direction"] else "NO",
            ]
        )
    lines.extend(
        markdown_table(
            stable_rows,
            ["차원", "구간", "누적 행", "가중 gap", "동일 방향"],
        )
    )
    lines.extend(["", "## 다음 액션 판정", ""])
    for item in results["findings"]:
        lines.extend(
            [
                f"### {item['priority']}",
                "",
                f"- 발견: {item['finding']}",
                f"- 조치: {item['action']}",
                "",
            ]
        )
    lines.extend(
        [
            "## 규정 판정",
            "",
            "- 현재 진단은 공식 train validation만 사용했다.",
            "- 진단 그룹·분위수는 모델 선택을 위한 train 분석이며 test 추론 피처로 사용하지 않는다.",
            "- 다음 후보도 현재 행 입력과 과거 train에서 고정한 모델·보정기만 사용해야 한다.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    _, prediction_frames = validate_and_load(args.train_path)
    dimensions_to_audit = [
        "game_type",
        "count_state",
        "exact_count",
        "game_month",
        "inning_bucket",
        "score_margin_bucket",
        "num_runners_on",
        "base_state",
        "leverage_bucket",
        "asof_cold_start",
        "entity_cold_start",
        "prediction_decile",
        "disagreement_decile",
    ]

    overall: dict[str, Any] = {}
    dimensions: dict[str, dict[str, list[dict[str, Any]]]] = {
        dimension: {} for dimension in dimensions_to_audit
    }
    disagreement: dict[str, Any] = {}
    for season, frame in prediction_frames.items():
        overall[str(season)] = {
            method: metric_payload(frame, column)
            for method, column in MODEL_COLUMNS.items()
        }
        disagreement[str(season)] = disagreement_summary(frame)
        for dimension in dimensions_to_audit:
            dimensions[dimension][str(season)] = group_metrics(frame, dimension)

    results: dict[str, Any] = {
        "experiment_id": "DIAG-1000-001",
        "evaluation_seasons": sorted(PREDICTION_FILES),
        "data_scope": "official train temporal validation only",
        "test_distribution_used": False,
        "external_data_used": False,
        "selection_rule": {
            "F": {"lightgbm": 0.0, "catboost": 1.0},
            "R": {"lightgbm": 0.5, "catboost": 0.5},
        },
        "source_files": {
            str(season): {
                "path": str(path.relative_to(ROOT)),
                "sha256": sha256_file(path),
            }
            for season, path in PREDICTION_FILES.items()
        },
        "train_path": str(args.train_path.relative_to(ROOT)),
        "train_sha256": sha256_file(args.train_path),
        "overall": overall,
        "dimensions": dimensions,
        "disagreement": disagreement,
    }
    results["stable_calibration_priorities"] = stable_group_priorities(dimensions)
    results["findings"] = build_findings(results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "diagnostics.json"
    markdown_path = args.output_dir / "diagnostics.md"
    json_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    markdown_path.write_text(render_markdown(results), encoding="utf-8")
    print(markdown_path.read_text(encoding="utf-8"))
    print(f"Saved JSON: {json_path}")
    print(f"JSON SHA-256: {sha256_file(json_path)}")


if __name__ == "__main__":
    main()
