#!/usr/bin/env python3
"""Audit safe lookup feasibility for situational and TrackMan aggregates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
PHYSICAL_COLUMNS = [
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed",
]
CONTEXT_KEYS = [
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-path", type=Path, default=ROOT / "data" / "train.csv")
    parser.add_argument("--test-path", type=Path, default=ROOT / "data" / "test.csv")
    parser.add_argument(
        "--trackman-path",
        type=Path,
        default=ROOT / "data" / "trackman_history.csv",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=ROOT / "model" / "aggregate_feasibility.json",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=ROOT / "model" / "aggregate_feasibility.md",
    )
    parser.add_argument("--valid-season", type=int, default=2023)
    return parser.parse_args()


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


def support_summary(
    history: pd.DataFrame,
    valid: pd.DataFrame,
    keys: list[str],
) -> dict[str, object]:
    counts = history.groupby(keys, observed=True).size().rename("history_n").reset_index()
    mapped = valid[keys].merge(counts, on=keys, how="left", validate="many_to_one")
    n = mapped["history_n"].fillna(0).to_numpy(dtype=np.int64)
    positive = n[n > 0]
    return {
        "keys": keys,
        "lookup_rows": int(len(counts)),
        "valid_rows": int(len(valid)),
        "covered_row_rate": float(np.mean(n > 0)),
        "unseen_row_rate": float(np.mean(n == 0)),
        "row_rate_n_lt_30": float(np.mean(n < 30)),
        "row_rate_n_lt_50": float(np.mean(n < 50)),
        "row_rate_n_lt_100": float(np.mean(n < 100)),
        "positive_support_quantiles": {
            "p10": float(np.quantile(positive, 0.10)),
            "p25": float(np.quantile(positive, 0.25)),
            "p50": float(np.quantile(positive, 0.50)),
            "p75": float(np.quantile(positive, 0.75)),
            "p90": float(np.quantile(positive, 0.90)),
        },
    }


def json_dump(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    train_columns = [
        "season",
        "game_month",
        "game_dayofweek",
        "inning",
        "top_bottom",
        "balls_before",
        "strikes_before",
        "outs_before",
        "runner_on_2b",
        "runner_on_3b",
        "pitcher_id",
        "batter_id",
        "pitcher_team_id",
        "batter_team_id",
        "asof_pitcher_success_rate",
        "control_success",
    ]
    trackman_columns = [
        "season",
        *CONTEXT_KEYS[0:3],
        "top_bottom",
        *CONTEXT_KEYS[4:],
        "pitcher_trackman_id",
        "batter_trackman_id",
        "pitcher_team",
        "batter_team",
        "pitch_type_group",
        *PHYSICAL_COLUMNS,
    ]
    trackman_columns = list(dict.fromkeys(trackman_columns))
    test_columns = [
        "season",
        "pitcher_id",
        "batter_id",
        "pitcher_team_id",
        "batter_team_id",
    ]

    train = pd.read_csv(args.train_path, encoding="utf-8-sig", usecols=train_columns)
    test = pd.read_csv(args.test_path, encoding="utf-8-sig", usecols=test_columns)
    trackman = pd.read_csv(
        args.trackman_path,
        encoding="utf-8-sig",
        usecols=trackman_columns,
    )

    main_pitchers = set(train["pitcher_id"].dropna().astype(str))
    main_batters = set(train["batter_id"].dropna().astype(str))
    tm_pitchers = set(trackman["pitcher_trackman_id"].dropna().astype(str))
    tm_batters = set(trackman["batter_trackman_id"].dropna().astype(str))

    trackman_2024 = trackman[trackman["season"].eq(2024)].copy()
    trackman_2024["top_bottom"] = (
        trackman_2024["top_bottom"].astype("string").str[0].str.upper()
    )
    train_2024 = train[train["season"].eq(2024)].copy()
    tm_context_counts = (
        trackman_2024.groupby(CONTEXT_KEYS, observed=True)
        .size()
        .rename("candidate_rows")
        .reset_index()
    )
    context_match = train_2024[CONTEXT_KEYS].merge(
        tm_context_counts,
        on=CONTEXT_KEYS,
        how="left",
        validate="many_to_one",
    )
    candidates = context_match["candidate_rows"].fillna(0).to_numpy(dtype=np.int64)
    positive_candidates = candidates[candidates > 0]

    train["count_state"] = count_state(train)
    train["scoring_pos_runner"] = (
        train["runner_on_2b"].eq(1) | train["runner_on_3b"].eq(1)
    ).astype("int8")
    history = train[train["season"].lt(args.valid_season)].copy()
    valid = train[train["season"].eq(args.valid_season)].copy()

    situational = {
        "pitcher_count_state": support_summary(
            history,
            valid,
            ["pitcher_id", "count_state"],
        ),
        "pitcher_scoring_position": support_summary(
            history,
            valid,
            ["pitcher_id", "scoring_pos_runner"],
        ),
    }

    season_context_rates = (
        train.groupby(["season", "count_state"], observed=True)["control_success"]
        .agg(rows="size", success_rate="mean")
        .reset_index()
        .to_dict(orient="records")
    )

    payload: dict[str, object] = {
        "valid_season": args.valid_season,
        "trackman": {
            "rows": int(len(trackman)),
            "seasons": sorted(int(value) for value in trackman["season"].unique()),
            "pitcher_ids": int(len(tm_pitchers)),
            "batter_ids": int(len(tm_batters)),
            "main_pitcher_ids": int(len(main_pitchers)),
            "main_batter_ids": int(len(main_batters)),
            "pitcher_id_intersection": int(len(main_pitchers & tm_pitchers)),
            "batter_id_intersection": int(len(main_batters & tm_batters)),
            "test_sample_pitcher_direct_coverage": float(
                test["pitcher_id"].astype(str).isin(tm_pitchers).mean()
            ),
            "test_sample_batter_direct_coverage": float(
                test["batter_id"].astype(str).isin(tm_batters).mean()
            ),
            "pitcher_teams": int(trackman["pitcher_team"].nunique(dropna=True)),
            "main_pitcher_team_ids": int(train["pitcher_team_id"].nunique(dropna=True)),
            "physical_missing_rates": {
                column: float(trackman[column].isna().mean())
                for column in PHYSICAL_COLUMNS
            },
            "context_only_2024_match": {
                "shared_keys": CONTEXT_KEYS,
                "main_rows": int(len(train_2024)),
                "no_candidate_rate": float(np.mean(candidates == 0)),
                "unique_candidate_rate": float(np.mean(candidates == 1)),
                "multiple_candidate_rate": float(np.mean(candidates > 1)),
                "positive_candidate_quantiles": {
                    "p10": float(np.quantile(positive_candidates, 0.10)),
                    "p50": float(np.quantile(positive_candidates, 0.50)),
                    "p90": float(np.quantile(positive_candidates, 0.90)),
                    "max": int(positive_candidates.max()),
                },
            },
            "direct_player_profile_lookup_feasible": False,
            "reason": (
                "메인 익명 선수 ID와 TrackMan 선수 ID 교집합이 없고, "
                "공통 경기상황 키는 1:1 행 또는 선수 매핑을 만들지 못함"
            ),
        },
        "situational": situational,
        "season_count_state_rates": season_context_rates,
        "decisions": {
            "trackman_player_profile": "reject_without_official_crosswalk",
            "trackman_current_pitch_measurements": "forbidden",
            "trackman_global_pitch_type_physics": "redundant_with_pitchmix_rates",
            "pitcher_count_state_deviation": "feasible_for_controlled_experiment",
            "pitcher_scoring_position_deviation": "feasible_but_lower_priority",
            "team_target_history": "defer_due_to_asof_and_id_redundancy",
            "current_pitch_type_history": "not_feasible_without_current_pitch_type",
        },
    }
    json_dump(args.output_json, payload)

    count_support = situational["pitcher_count_state"]
    scoring_support = situational["pitcher_scoring_position"]
    track = payload["trackman"]
    context = track["context_only_2024_match"]
    markdown = f"""# 집계 피처 타당성 감사

## TrackMan

- 행 수: `{track['rows']:,}` / 시즌: `{track['seasons']}`
- 메인↔TrackMan 투수 ID 교집합: `{track['pitcher_id_intersection']}`
- 메인↔TrackMan 타자 ID 교집합: `{track['batter_id_intersection']}`
- 2024 공통 문맥 키 유일 후보 비율: `{context['unique_candidate_rate']:.6f}`
- 2024 공통 문맥 키 다중 후보 비율: `{context['multiple_candidate_rate']:.6f}`
- 판정: 공식 ID crosswalk 없이는 선수별 TrackMan 프로파일 lookup 불가

## 조건부 집계 ({args.valid_season} 검증 기준)

| 후보 | 검증 행 커버리지 | n<30 행 비율 | n<50 행 비율 | 양수 support 중앙값 | 판정 |
|---|---:|---:|---:|---:|---|
| 투수×count_state | `{count_support['covered_row_rate']:.6f}` | `{count_support['row_rate_n_lt_30']:.6f}` | `{count_support['row_rate_n_lt_50']:.6f}` | `{count_support['positive_support_quantiles']['p50']:.1f}` | 통제 실험 가능 |
| 투수×득점권 여부 | `{scoring_support['covered_row_rate']:.6f}` | `{scoring_support['row_rate_n_lt_30']:.6f}` | `{scoring_support['row_rate_n_lt_50']:.6f}` | `{scoring_support['positive_support_quantiles']['p50']:.1f}` | 후순위 |

## 최종 판정

- TrackMan 선수 물리 프로파일: 공식 crosswalk 전까지 폐기
- 현재 투구 구종별 집계: test에 현재 구종이 없어 불가
- 다음 단일 후보: 이전 시즌만 사용한 투수×count_state 성공률 편차
"""
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(markdown, encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved: {args.output_json}")
    print(f"saved: {args.output_md}")


if __name__ == "__main__":
    main()
