#!/usr/bin/env python3
"""Diagnose composition and calibration of the highest prediction decile."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ENTITY_COLS = ["pitcher_id", "batter_id", "pitcher_team_id", "batter_team_id"]
CONTEXT_COLS = [
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "game_type",
    "balls_before",
    "strikes_before",
    "outs_before",
    "num_runners_on",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare the highest prediction decile with the remaining holdout rows"
    )
    parser.add_argument("--train-path", default=ROOT / "data" / "train.csv", type=Path)
    parser.add_argument(
        "--predictions",
        default=ROOT / "model" / "LGBM-EW-2023" / "validation_predictions.csv",
        type=Path,
    )
    parser.add_argument(
        "--metadata",
        default=ROOT / "model" / "LGBM-EW-2023" / "metadata.json",
        type=Path,
    )
    parser.add_argument("--quantiles", default=10, type=int)
    parser.add_argument(
        "--output-json",
        default=ROOT / "model" / "LGBM-EW-2023" / "decile_diagnostics.json",
        type=Path,
    )
    parser.add_argument(
        "--output-md",
        default=ROOT / "model" / "LGBM-EW-2023" / "decile_diagnostics.md",
        type=Path,
    )
    return parser.parse_args()


def finite_float(value):
    value = float(value)
    return value if np.isfinite(value) else None


def subset_metrics(frame):
    return {
        "rows": int(len(frame)),
        "pred_mean": finite_float(frame["pred_lgbm"].mean()),
        "target_mean": finite_float(frame["target"].mean()),
        "calibration_gap": finite_float(
            frame["pred_lgbm"].mean() - frame["target"].mean()
        ),
        "brier": finite_float(np.mean((frame["target"] - frame["pred_lgbm"]) ** 2)),
    }


def concentration(frame, column):
    shares = frame[column].astype("string").fillna("<NA>").value_counts(normalize=True)
    return {
        "unique": int(shares.size),
        "top1_share": finite_float(shares.iloc[:1].sum()),
        "top5_share": finite_float(shares.iloc[:5].sum()),
        "hhi": finite_float(np.square(shares).sum()),
    }


def grouped_rows(full, top, column, limit=10):
    full_agg = full.groupby(column, dropna=False, observed=True).agg(
        full_rows=("row_id", "size"),
        full_pred=("pred_lgbm", "mean"),
        full_target=("target", "mean"),
    )
    top_agg = top.groupby(column, dropna=False, observed=True).agg(
        top_rows=("row_id", "size"),
        top_pred=("pred_lgbm", "mean"),
        top_target=("target", "mean"),
    )
    table = top_agg.join(full_agg, how="left").reset_index()
    table["top_share"] = table["top_rows"] / len(top)
    table["full_share"] = table["full_rows"] / len(full)
    table["share_lift"] = table["top_share"] / table["full_share"]
    table["top_gap"] = table["top_pred"] - table["top_target"]
    table["gap_contribution"] = table["top_rows"] * table["top_gap"] / len(top)

    columns = [
        column,
        "top_rows",
        "top_share",
        "full_share",
        "share_lift",
        "top_pred",
        "top_target",
        "top_gap",
        "gap_contribution",
    ]

    def records(sorted_table):
        values = sorted_table[columns].copy()
        values[column] = values[column].astype("string").fillna("<NA>")
        return json.loads(values.to_json(orient="records"))

    return {
        "largest_groups": records(table.sort_values("top_rows", ascending=False).head(limit)),
        "largest_overprediction_contributors": records(
            table.sort_values("gap_contribution", ascending=False).head(limit)
        ),
    }


def missingness(frame, top_mask, columns):
    rows = []
    for column in columns:
        top_rate = frame.loc[top_mask, column].isna().mean()
        rest_rate = frame.loc[~top_mask, column].isna().mean()
        rows.append(
            {
                "feature": column,
                "top_missing_rate": finite_float(top_rate),
                "rest_missing_rate": finite_float(rest_rate),
                "delta": finite_float(top_rate - rest_rate),
            }
        )
    return sorted(rows, key=lambda row: abs(row["delta"]), reverse=True)


def numeric_shift(frame, top_mask, columns):
    rows = []
    for column in columns:
        top_values = frame.loc[top_mask, column]
        rest_values = frame.loc[~top_mask, column]
        pooled_std = frame[column].std()
        delta = top_values.mean() - rest_values.mean()
        rows.append(
            {
                "feature": column,
                "top_mean": finite_float(top_values.mean()),
                "rest_mean": finite_float(rest_values.mean()),
                "mean_delta": finite_float(delta),
                "standardized_delta": finite_float(delta / pooled_std)
                if pooled_std and np.isfinite(pooled_std)
                else None,
            }
        )
    return sorted(
        rows,
        key=lambda row: abs(row["standardized_delta"] or 0.0),
        reverse=True,
    )


def markdown_table(rows, columns, headers):
    output = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        cells = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                cells.append(f"{value:.6f}")
            else:
                cells.append(str(value))
        output.append("| " + " | ".join(cells) + " |")
    return "\n".join(output)


def main():
    args = parse_args()
    if args.quantiles < 2:
        raise ValueError("--quantiles must be at least 2")

    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    valid_season = int(metadata["valid_season"])
    feature_columns = metadata["features"]
    diagnostic_columns = ENTITY_COLS + CONTEXT_COLS + [
        "asof_pitcher_prev1_game_success_rate"
    ]
    requested_columns = list(
        dict.fromkeys(
            ["row_id", "season", "control_success"]
            + feature_columns
            + diagnostic_columns
        )
    )
    raw_columns = set(
        pd.read_csv(args.train_path, encoding="utf-8-sig", nrows=0).columns
    )
    asof_columns = [
        column for column in feature_columns
        if column.startswith("asof_") and column in raw_columns
    ]
    usecols = [column for column in requested_columns if column in raw_columns]
    required_diagnostic_columns = {
        "row_id",
        "season",
        "control_success",
        "game_type",
        "pitcher_team_id",
        "batter_team_id",
    }
    missing_diagnostic_columns = required_diagnostic_columns - set(usecols)
    if missing_diagnostic_columns:
        raise ValueError(
            "train.csv에 필수 진단 컬럼이 없습니다: "
            f"{sorted(missing_diagnostic_columns)}"
        )

    predictions = pd.read_csv(args.predictions)
    if predictions["row_id"].duplicated().any():
        raise ValueError("validation_predictions.csv contains duplicate row_id values")

    raw = pd.read_csv(args.train_path, encoding="utf-8-sig", usecols=usecols)
    history = raw[raw["season"].isin(metadata["train_seasons"])]
    valid = raw[raw["season"] == valid_season].copy()
    valid = valid.merge(
        predictions[["row_id", "target", "pred_lgbm"]],
        on="row_id",
        how="inner",
        validate="one_to_one",
    )
    if len(valid) != len(predictions):
        raise ValueError(f"row merge mismatch: raw={len(valid)}, predictions={len(predictions)}")
    if not np.array_equal(valid["control_success"].to_numpy(), valid["target"].to_numpy()):
        raise ValueError("target mismatch between train.csv and validation_predictions.csv")

    valid["prediction_decile"] = pd.qcut(
        valid["pred_lgbm"], args.quantiles, labels=False, duplicates="drop"
    )
    actual_bins = int(valid["prediction_decile"].nunique())
    if actual_bins != args.quantiles:
        raise ValueError(f"expected {args.quantiles} bins, got {actual_bins}")
    top_mask = valid["prediction_decile"] == valid["prediction_decile"].max()
    top = valid[top_mask]
    rest = valid[~top_mask]

    entities = {}
    for column in ENTITY_COLS:
        seen = set(history[column].dropna().unique())
        top_unseen = ~top[column].isin(seen)
        rest_unseen = ~rest[column].isin(seen)
        entities[column] = {
            "top_concentration": concentration(top, column),
            "rest_concentration": concentration(rest, column),
            "top_unseen_row_rate": finite_float(top_unseen.mean()),
            "rest_unseen_row_rate": finite_float(rest_unseen.mean()),
            **grouped_rows(valid, top, column),
        }

    contexts = {column: grouped_rows(valid, top, column, limit=8) for column in CONTEXT_COLS}
    game_type_history = []
    for (season, game_type), group in raw.groupby(
        ["season", "game_type"], dropna=False, observed=True
    ):
        game_type_history.append(
            {
                "season": int(season),
                "game_type": str(game_type),
                "rows": int(len(group)),
                "target_mean": finite_float(group["control_success"].mean()),
            }
        )

    team_13_involved = valid["pitcher_team_id"].eq(13) | valid["batter_team_id"].eq(13)
    top_team_13_involved = team_13_involved.loc[top.index]
    top_prev_game_missing = top["asof_pitcher_prev1_game_success_rate"].isna()
    historical_batters = set(history["batter_id"].dropna().unique())
    top_unseen_batter = ~top["batter_id"].isin(historical_batters)
    key_segments = {
        "valid_game_type_f": subset_metrics(valid[valid["game_type"].eq("F")]),
        "valid_game_type_r": subset_metrics(valid[valid["game_type"].eq("R")]),
        "top_game_type_f": subset_metrics(top[top["game_type"].eq("F")]),
        "top_team_13_involved": subset_metrics(top[top_team_13_involved]),
        "top_previous_game_rates_missing": subset_metrics(top[top_prev_game_missing]),
        "top_previous_game_rates_present": subset_metrics(top[~top_prev_game_missing]),
        "top_unseen_batter": subset_metrics(top[top_unseen_batter]),
        "top_seen_batter": subset_metrics(top[~top_unseen_batter]),
    }
    result = {
        "experiment": metadata["exp_id"],
        "valid_season": valid_season,
        "quantiles": args.quantiles,
        "top_decile_min_prediction": finite_float(top["pred_lgbm"].min()),
        "overall": subset_metrics(valid),
        "top_decile": subset_metrics(top),
        "remaining_rows": subset_metrics(rest),
        "asof_missingness": missingness(valid, top_mask, asof_columns),
        "asof_numeric_shift": numeric_shift(valid, top_mask, asof_columns),
        "entities": entities,
        "contexts": contexts,
        "game_type_target_history": game_type_history,
        "key_segments": key_segments,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    top_metrics = result["top_decile"]
    missing_top = result["asof_missingness"][:8]
    shift_top = result["asof_numeric_shift"][:8]
    entity_summary = []
    for column, values in entities.items():
        entity_summary.append(
            {
                "entity": column,
                "top1": values["top_concentration"]["top1_share"],
                "top5": values["top_concentration"]["top5_share"],
                "hhi": values["top_concentration"]["hhi"],
                "unseen_top": values["top_unseen_row_rate"],
                "unseen_rest": values["rest_unseen_row_rate"],
            }
        )

    game_type_f_history = [
        row for row in game_type_history if row["game_type"] == "F"
    ]
    top_game_types = contexts["game_type"]["largest_groups"]
    segment_summary = []
    for segment, values in key_segments.items():
        segment_summary.append(
            {
                "segment": segment,
                "rows": values["rows"],
                "pred_mean": values["pred_mean"],
                "target_mean": values["target_mean"],
                "calibration_gap": values["calibration_gap"],
            }
        )

    md = [
        f"# {metadata['exp_id']} 상위 예측 {100 // args.quantiles}% 진단",
        "",
        f"- 검증 시즌: `{valid_season}`",
        f"- 상위 구간 행 수: `{top_metrics['rows']:,}`",
        f"- 상위 구간 최소 예측값: `{result['top_decile_min_prediction']:.6f}`",
        f"- 평균 예측 / 실제: `{top_metrics['pred_mean']:.6f} / {top_metrics['target_mean']:.6f}`",
        f"- Calibration gap: `{top_metrics['calibration_gap']:+.6f}`",
        "",
        "## 핵심 범주 진단",
        "",
        markdown_table(
            top_game_types,
            ["game_type", "top_rows", "top_share", "full_share", "top_gap"],
            ["game_type", "상위 행", "상위 비중", "전체 비중", "상위 오차"],
        ),
        "",
        markdown_table(
            game_type_f_history,
            ["season", "rows", "target_mean"],
            ["시즌", "F 행", "F 실제 성공률"],
        ),
        "",
        markdown_table(
            segment_summary,
            ["segment", "rows", "pred_mean", "target_mean", "calibration_gap"],
            ["구간", "행", "예측", "실제", "오차"],
        ),
        "",
        "## asof 결측률 차이 상위",
        "",
        markdown_table(
            missing_top,
            ["feature", "top_missing_rate", "rest_missing_rate", "delta"],
            ["피처", "상위 결측률", "나머지 결측률", "차이"],
        ),
        "",
        "## asof 평균 이동 상위",
        "",
        markdown_table(
            shift_top,
            ["feature", "top_mean", "rest_mean", "standardized_delta"],
            ["피처", "상위 평균", "나머지 평균", "표준화 차이"],
        ),
        "",
        "## 선수·팀 집중도와 과거 미등장률",
        "",
        markdown_table(
            entity_summary,
            ["entity", "top1", "top5", "hhi", "unseen_top", "unseen_rest"],
            ["구분", "Top1 비중", "Top5 비중", "HHI", "상위 미등장률", "나머지 미등장률"],
        ),
        "",
        "세부 그룹별 표본 수·예측·실제·격차·오차 기여도는 JSON 파일에 저장했다.",
    ]
    args.output_md.write_text("\n".join(md) + "\n", encoding="utf-8")

    print(json.dumps({
        "output_json": str(args.output_json),
        "output_md": str(args.output_md),
        "top_decile": top_metrics,
        "largest_missingness_delta": missing_top[0],
        "largest_numeric_shift": shift_top[0],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
