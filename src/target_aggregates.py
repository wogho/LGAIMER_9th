"""Target-derived aggregate features built without same-season or future leakage."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.features import TARGET_COL, derive_count_state, derive_scoring_pos_runner


PITCHER_ENTITY_COL = "pitcher_id"
PITCHER_COUNT_FEATURE = "target_hist_pitcher_n"
PITCHER_RATE_FEATURE = "target_hist_pitcher_success_rate"
PITCHER_FEATURE_COLUMNS = [PITCHER_COUNT_FEATURE, PITCHER_RATE_FEATURE]

BATTER_ENTITY_COL = "batter_id"
BATTER_COUNT_FEATURE = "target_hist_batter_n"
BATTER_RATE_FEATURE = "target_hist_batter_success_rate"
BATTER_FEATURE_COLUMNS = [BATTER_COUNT_FEATURE, BATTER_RATE_FEATURE]

PITCHER_COUNT_STATE_CONDITION_COL = "count_state"
PITCHER_COUNT_STATE_COUNT_FEATURE = "target_hist_pitcher_count_n"
PITCHER_COUNT_STATE_DELTA_FEATURE = "target_hist_pitcher_count_delta"
PITCHER_COUNT_STATE_FEATURE_COLUMNS = [
    PITCHER_COUNT_STATE_COUNT_FEATURE,
    PITCHER_COUNT_STATE_DELTA_FEATURE,
]

PITCHER_SCORING_POS_CONDITION_COL = "scoring_pos_runner"
PITCHER_SCORING_POS_COUNT_FEATURE = "target_hist_pitcher_scoring_pos_n"
PITCHER_SCORING_POS_DELTA_FEATURE = "target_hist_pitcher_scoring_pos_delta"
PITCHER_SCORING_POS_FEATURE_COLUMNS = [
    PITCHER_SCORING_POS_COUNT_FEATURE,
    PITCHER_SCORING_POS_DELTA_FEATURE,
]

# Backward-compatible aliases used by the existing pitcher verification contract.
ENTITY_COL = PITCHER_ENTITY_COL
COUNT_FEATURE = PITCHER_COUNT_FEATURE
RATE_FEATURE = PITCHER_RATE_FEATURE
AGGREGATE_FEATURE_COLUMNS = PITCHER_FEATURE_COLUMNS
INITIAL_FALLBACK = 0.5


def _validate_inputs(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    smoothing: float,
    entity_col: str,
) -> None:
    required = {"season", entity_col, TARGET_COL}
    for name, frame in (("train", train_df), ("valid", valid_df)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} 데이터에 집계 필수 컬럼이 없습니다: {sorted(missing)}")
        if frame[entity_col].isna().any():
            raise ValueError(f"{name} 데이터의 {entity_col}에 결측값이 있습니다")
    if not np.isfinite(smoothing) or smoothing <= 0:
        raise ValueError("smoothing은 0보다 큰 유한값이어야 합니다")
    if train_df.empty or valid_df.empty:
        raise ValueError("train과 valid 데이터는 비어 있을 수 없습니다")
    if int(train_df["season"].max()) >= int(valid_df["season"].min()):
        raise ValueError("valid 시즌은 모든 train 시즌보다 뒤여야 합니다")
    target_values = set(train_df[TARGET_COL].dropna().unique())
    if not target_values.issubset({0, 1}):
        raise ValueError(f"{TARGET_COL}은 이진값이어야 합니다: {sorted(target_values)}")


def _history_stats(
    history: pd.DataFrame,
    smoothing: float,
    fallback: float,
    entity_col: str,
    count_feature: str,
    rate_feature: str,
) -> pd.DataFrame:
    if history.empty:
        return pd.DataFrame(columns=["success_sum", count_feature, rate_feature])

    stats = history.groupby(entity_col, observed=True)[TARGET_COL].agg(
        success_sum="sum",
        **{count_feature: "count"},
    )
    stats[rate_feature] = (
        stats["success_sum"] + smoothing * fallback
    ) / (stats[count_feature] + smoothing)
    return stats


def _map_stats(
    rows: pd.DataFrame,
    stats: pd.DataFrame,
    fallback: float,
    entity_col: str,
    count_feature: str,
    rate_feature: str,
) -> pd.DataFrame:
    counts = rows[entity_col].map(stats[count_feature]).fillna(0).astype("int32")
    rates = rows[entity_col].map(stats[rate_feature]).fillna(fallback).astype("float32")
    return pd.DataFrame(
        {
            count_feature: counts,
            rate_feature: rates,
        },
        index=rows.index,
    )


def build_pitcher_target_history(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    smoothing: float = 100.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build strict-prior-season pitcher features and a train-only lookup."""

    return _build_target_history(
        train_df,
        valid_df,
        smoothing=smoothing,
        entity_col=PITCHER_ENTITY_COL,
        count_feature=PITCHER_COUNT_FEATURE,
        rate_feature=PITCHER_RATE_FEATURE,
    )


def build_batter_target_history(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    smoothing: float = 50.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build strict-prior-season batter features and a train-only lookup."""

    return _build_target_history(
        train_df,
        valid_df,
        smoothing=smoothing,
        entity_col=BATTER_ENTITY_COL,
        count_feature=BATTER_COUNT_FEATURE,
        rate_feature=BATTER_RATE_FEATURE,
    )


def _pitcher_count_state_stats(
    history: pd.DataFrame,
    smoothing: float,
) -> pd.DataFrame:
    """Build count-conditional rates shrunk to each pitcher's overall rate."""

    columns = [
        PITCHER_ENTITY_COL,
        PITCHER_COUNT_STATE_CONDITION_COL,
        PITCHER_COUNT_STATE_COUNT_FEATURE,
        PITCHER_COUNT_STATE_DELTA_FEATURE,
    ]
    if history.empty:
        return pd.DataFrame(columns=columns)

    working = history[[PITCHER_ENTITY_COL, TARGET_COL]].copy()
    working[PITCHER_COUNT_STATE_CONDITION_COL] = derive_count_state(history).astype(
        "string"
    )
    global_rate = float(working[TARGET_COL].mean())
    pitcher = working.groupby(PITCHER_ENTITY_COL, observed=True)[TARGET_COL].agg(
        pitcher_success_sum="sum",
        pitcher_n="count",
    )
    pitcher["pitcher_rate"] = (
        pitcher["pitcher_success_sum"] + smoothing * global_rate
    ) / (pitcher["pitcher_n"] + smoothing)

    conditional = working.groupby(
        [PITCHER_ENTITY_COL, PITCHER_COUNT_STATE_CONDITION_COL],
        observed=True,
    )[TARGET_COL].agg(
        conditional_success_sum="sum",
        **{PITCHER_COUNT_STATE_COUNT_FEATURE: "count"},
    )
    conditional = conditional.join(pitcher[["pitcher_rate"]], on=PITCHER_ENTITY_COL)
    conditional_rate = (
        conditional["conditional_success_sum"]
        + smoothing * conditional["pitcher_rate"]
    ) / (conditional[PITCHER_COUNT_STATE_COUNT_FEATURE] + smoothing)
    conditional[PITCHER_COUNT_STATE_DELTA_FEATURE] = (
        conditional_rate - conditional["pitcher_rate"]
    )
    lookup = conditional.reset_index()[columns].copy()
    lookup[PITCHER_COUNT_STATE_COUNT_FEATURE] = lookup[
        PITCHER_COUNT_STATE_COUNT_FEATURE
    ].astype("int32")
    lookup[PITCHER_COUNT_STATE_DELTA_FEATURE] = lookup[
        PITCHER_COUNT_STATE_DELTA_FEATURE
    ].astype("float32")
    return lookup.sort_values(
        [PITCHER_ENTITY_COL, PITCHER_COUNT_STATE_CONDITION_COL]
    ).reset_index(drop=True)


def apply_pitcher_count_state_target_lookup(
    rows: pd.DataFrame,
    lookup: pd.DataFrame,
    fallback: float = 0.0,
) -> pd.DataFrame:
    """Apply a train-only pitcher/count-state lookup; unseen combinations map to 0."""

    del fallback  # Kept for the common persisted-lookup verifier interface.
    required_lookup = {
        PITCHER_ENTITY_COL,
        PITCHER_COUNT_STATE_CONDITION_COL,
        *PITCHER_COUNT_STATE_FEATURE_COLUMNS,
    }
    missing_lookup = required_lookup - set(lookup.columns)
    if missing_lookup:
        raise ValueError(f"lookup 필수 컬럼이 없습니다: {sorted(missing_lookup)}")
    if PITCHER_ENTITY_COL not in rows.columns:
        raise ValueError(f"입력 데이터에 {PITCHER_ENTITY_COL}이 없습니다")
    if lookup.duplicated(
        [PITCHER_ENTITY_COL, PITCHER_COUNT_STATE_CONDITION_COL]
    ).any():
        raise ValueError("lookup의 pitcher_id×count_state 키가 중복되었습니다")

    if lookup.empty:
        return pd.DataFrame(
            {
                PITCHER_COUNT_STATE_COUNT_FEATURE: np.zeros(len(rows), dtype="int32"),
                PITCHER_COUNT_STATE_DELTA_FEATURE: np.zeros(len(rows), dtype="float32"),
            },
            index=rows.index,
        )

    keys = pd.DataFrame(
        {
            "__row_position": np.arange(len(rows), dtype="int64"),
            PITCHER_ENTITY_COL: rows[PITCHER_ENTITY_COL].to_numpy(),
            PITCHER_COUNT_STATE_CONDITION_COL: derive_count_state(rows).astype(
                "string"
            ),
        },
        index=rows.index,
    )
    mapped = keys.merge(
        lookup,
        how="left",
        on=[PITCHER_ENTITY_COL, PITCHER_COUNT_STATE_CONDITION_COL],
        sort=False,
        validate="many_to_one",
    )
    mapped = mapped.sort_values("__row_position")
    result = mapped[PITCHER_COUNT_STATE_FEATURE_COLUMNS].copy()
    result.index = rows.index
    result[PITCHER_COUNT_STATE_COUNT_FEATURE] = result[
        PITCHER_COUNT_STATE_COUNT_FEATURE
    ].fillna(0).astype("int32")
    result[PITCHER_COUNT_STATE_DELTA_FEATURE] = result[
        PITCHER_COUNT_STATE_DELTA_FEATURE
    ].fillna(0.0).astype("float32")
    result.index.name = rows.index.name
    return result


def build_pitcher_count_state_target_history(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    smoothing: float = 100.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build strict-prior-season pitcher/count-state relative tendency features."""

    smoothing = float(smoothing)
    _validate_inputs(train_df, valid_df, smoothing, PITCHER_ENTITY_COL)
    required_count_columns = {"balls_before", "strikes_before"}
    for name, frame in (("train", train_df), ("valid", valid_df)):
        missing = required_count_columns - set(frame.columns)
        if missing:
            raise ValueError(
                f"{name} 데이터에 count_state 필수 컬럼이 없습니다: {sorted(missing)}"
            )

    train_features = pd.DataFrame(index=train_df.index)
    train_features[PITCHER_COUNT_STATE_COUNT_FEATURE] = np.int32(0)
    train_features[PITCHER_COUNT_STATE_DELTA_FEATURE] = np.float32(0.0)
    season_audit: list[dict[str, Any]] = []
    for season in sorted(int(value) for value in train_df["season"].unique()):
        season_rows = train_df.loc[train_df["season"].eq(season)]
        history = train_df.loc[train_df["season"].lt(season)]
        lookup = _pitcher_count_state_stats(history, smoothing)
        mapped = apply_pitcher_count_state_target_lookup(season_rows, lookup)
        train_features.loc[
            season_rows.index, PITCHER_COUNT_STATE_FEATURE_COLUMNS
        ] = mapped
        season_audit.append(
            {
                "season": season,
                "history_seasons": sorted(
                    int(value) for value in history["season"].unique()
                ),
                "history_rows": int(len(history)),
                "fallback": 0.0,
                "unseen_row_rate": float(
                    mapped[PITCHER_COUNT_STATE_COUNT_FEATURE].eq(0).mean()
                ),
            }
        )

    train_features[PITCHER_COUNT_STATE_COUNT_FEATURE] = train_features[
        PITCHER_COUNT_STATE_COUNT_FEATURE
    ].astype("int32")
    train_features[PITCHER_COUNT_STATE_DELTA_FEATURE] = train_features[
        PITCHER_COUNT_STATE_DELTA_FEATURE
    ].astype("float32")
    inference_lookup = _pitcher_count_state_stats(train_df, smoothing)
    valid_features = apply_pitcher_count_state_target_lookup(
        valid_df, inference_lookup
    )
    metadata: dict[str, Any] = {
        "enabled": True,
        "mode": "strict_prior_season_expanding",
        "entity_column": PITCHER_ENTITY_COL,
        "condition_column": PITCHER_COUNT_STATE_CONDITION_COL,
        "target_column": TARGET_COL,
        "feature_columns": PITCHER_COUNT_STATE_FEATURE_COLUMNS,
        "smoothing": smoothing,
        "initial_fallback": 0.0,
        "inference_fallback": 0.0,
        "shrinkage_prior": "same_pitcher_overall_smoothed_rate",
        "unseen_policy": "count=0,delta=0",
        "train_seasons": sorted(int(value) for value in train_df["season"].unique()),
        "valid_seasons": sorted(int(value) for value in valid_df["season"].unique()),
        "lookup_rows": int(len(inference_lookup)),
        "valid_unseen_row_rate": float(
            valid_features[PITCHER_COUNT_STATE_COUNT_FEATURE].eq(0).mean()
        ),
        "season_audit": season_audit,
    }
    return train_features, valid_features, inference_lookup, metadata


def _pitcher_scoring_pos_stats(
    history: pd.DataFrame,
    smoothing: float,
) -> pd.DataFrame:
    """Build scoring-position rates shrunk to each pitcher's overall rate."""

    columns = [
        PITCHER_ENTITY_COL,
        PITCHER_SCORING_POS_CONDITION_COL,
        PITCHER_SCORING_POS_COUNT_FEATURE,
        PITCHER_SCORING_POS_DELTA_FEATURE,
    ]
    if history.empty:
        return pd.DataFrame(columns=columns)

    working = history[[PITCHER_ENTITY_COL, TARGET_COL]].copy()
    working[PITCHER_SCORING_POS_CONDITION_COL] = derive_scoring_pos_runner(history)
    global_rate = float(working[TARGET_COL].mean())
    pitcher = working.groupby(PITCHER_ENTITY_COL, observed=True)[TARGET_COL].agg(
        pitcher_success_sum="sum",
        pitcher_n="count",
    )
    pitcher["pitcher_rate"] = (
        pitcher["pitcher_success_sum"] + smoothing * global_rate
    ) / (pitcher["pitcher_n"] + smoothing)
    conditional = working.groupby(
        [PITCHER_ENTITY_COL, PITCHER_SCORING_POS_CONDITION_COL],
        observed=True,
    )[TARGET_COL].agg(
        conditional_success_sum="sum",
        **{PITCHER_SCORING_POS_COUNT_FEATURE: "count"},
    )
    conditional = conditional.join(pitcher[["pitcher_rate"]], on=PITCHER_ENTITY_COL)
    conditional_rate = (
        conditional["conditional_success_sum"]
        + smoothing * conditional["pitcher_rate"]
    ) / (conditional[PITCHER_SCORING_POS_COUNT_FEATURE] + smoothing)
    conditional[PITCHER_SCORING_POS_DELTA_FEATURE] = (
        conditional_rate - conditional["pitcher_rate"]
    )
    lookup = conditional.reset_index()[columns].copy()
    lookup[PITCHER_SCORING_POS_CONDITION_COL] = lookup[
        PITCHER_SCORING_POS_CONDITION_COL
    ].astype("int8")
    lookup[PITCHER_SCORING_POS_COUNT_FEATURE] = lookup[
        PITCHER_SCORING_POS_COUNT_FEATURE
    ].astype("int32")
    lookup[PITCHER_SCORING_POS_DELTA_FEATURE] = lookup[
        PITCHER_SCORING_POS_DELTA_FEATURE
    ].astype("float32")
    return lookup.sort_values(
        [PITCHER_ENTITY_COL, PITCHER_SCORING_POS_CONDITION_COL]
    ).reset_index(drop=True)


def apply_pitcher_scoring_pos_target_lookup(
    rows: pd.DataFrame,
    lookup: pd.DataFrame,
    fallback: float = 0.0,
) -> pd.DataFrame:
    """Apply a train-only pitcher/scoring-position lookup with zero fallback."""

    del fallback
    required_lookup = {
        PITCHER_ENTITY_COL,
        PITCHER_SCORING_POS_CONDITION_COL,
        *PITCHER_SCORING_POS_FEATURE_COLUMNS,
    }
    missing_lookup = required_lookup - set(lookup.columns)
    if missing_lookup:
        raise ValueError(f"lookup 필수 컬럼이 없습니다: {sorted(missing_lookup)}")
    if PITCHER_ENTITY_COL not in rows.columns:
        raise ValueError(f"입력 데이터에 {PITCHER_ENTITY_COL}이 없습니다")
    if lookup.duplicated(
        [PITCHER_ENTITY_COL, PITCHER_SCORING_POS_CONDITION_COL]
    ).any():
        raise ValueError("lookup의 pitcher_id×scoring_pos_runner 키가 중복되었습니다")
    if lookup.empty:
        return pd.DataFrame(
            {
                PITCHER_SCORING_POS_COUNT_FEATURE: np.zeros(
                    len(rows), dtype="int32"
                ),
                PITCHER_SCORING_POS_DELTA_FEATURE: np.zeros(
                    len(rows), dtype="float32"
                ),
            },
            index=rows.index,
        )

    keys = pd.DataFrame(
        {
            "__row_position": np.arange(len(rows), dtype="int64"),
            PITCHER_ENTITY_COL: rows[PITCHER_ENTITY_COL].to_numpy(),
            PITCHER_SCORING_POS_CONDITION_COL: derive_scoring_pos_runner(
                rows
            ).to_numpy(),
        },
        index=rows.index,
    )
    mapped = keys.merge(
        lookup,
        how="left",
        on=[PITCHER_ENTITY_COL, PITCHER_SCORING_POS_CONDITION_COL],
        sort=False,
        validate="many_to_one",
    ).sort_values("__row_position")
    result = mapped[PITCHER_SCORING_POS_FEATURE_COLUMNS].copy()
    result.index = rows.index
    result[PITCHER_SCORING_POS_COUNT_FEATURE] = result[
        PITCHER_SCORING_POS_COUNT_FEATURE
    ].fillna(0).astype("int32")
    result[PITCHER_SCORING_POS_DELTA_FEATURE] = result[
        PITCHER_SCORING_POS_DELTA_FEATURE
    ].fillna(0.0).astype("float32")
    result.index.name = rows.index.name
    return result


def build_pitcher_scoring_pos_target_history(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    smoothing: float = 100.0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build strict-prior-season pitcher/scoring-position relative tendencies."""

    smoothing = float(smoothing)
    _validate_inputs(train_df, valid_df, smoothing, PITCHER_ENTITY_COL)
    required_condition_columns = {"runner_on_2b", "runner_on_3b"}
    for name, frame in (("train", train_df), ("valid", valid_df)):
        missing = required_condition_columns - set(frame.columns)
        if missing:
            raise ValueError(
                f"{name} 데이터에 득점권 필수 컬럼이 없습니다: {sorted(missing)}"
            )

    train_features = pd.DataFrame(index=train_df.index)
    train_features[PITCHER_SCORING_POS_COUNT_FEATURE] = np.int32(0)
    train_features[PITCHER_SCORING_POS_DELTA_FEATURE] = np.float32(0.0)
    season_audit: list[dict[str, Any]] = []
    for season in sorted(int(value) for value in train_df["season"].unique()):
        season_rows = train_df.loc[train_df["season"].eq(season)]
        history = train_df.loc[train_df["season"].lt(season)]
        lookup = _pitcher_scoring_pos_stats(history, smoothing)
        mapped = apply_pitcher_scoring_pos_target_lookup(season_rows, lookup)
        train_features.loc[
            season_rows.index, PITCHER_SCORING_POS_FEATURE_COLUMNS
        ] = mapped
        season_audit.append(
            {
                "season": season,
                "history_seasons": sorted(
                    int(value) for value in history["season"].unique()
                ),
                "history_rows": int(len(history)),
                "fallback": 0.0,
                "unseen_row_rate": float(
                    mapped[PITCHER_SCORING_POS_COUNT_FEATURE].eq(0).mean()
                ),
            }
        )

    train_features[PITCHER_SCORING_POS_COUNT_FEATURE] = train_features[
        PITCHER_SCORING_POS_COUNT_FEATURE
    ].astype("int32")
    train_features[PITCHER_SCORING_POS_DELTA_FEATURE] = train_features[
        PITCHER_SCORING_POS_DELTA_FEATURE
    ].astype("float32")
    inference_lookup = _pitcher_scoring_pos_stats(train_df, smoothing)
    valid_features = apply_pitcher_scoring_pos_target_lookup(valid_df, inference_lookup)
    metadata: dict[str, Any] = {
        "enabled": True,
        "mode": "strict_prior_season_expanding",
        "entity_column": PITCHER_ENTITY_COL,
        "condition_column": PITCHER_SCORING_POS_CONDITION_COL,
        "target_column": TARGET_COL,
        "feature_columns": PITCHER_SCORING_POS_FEATURE_COLUMNS,
        "smoothing": smoothing,
        "initial_fallback": 0.0,
        "inference_fallback": 0.0,
        "shrinkage_prior": "same_pitcher_overall_smoothed_rate",
        "unseen_policy": "count=0,delta=0",
        "train_seasons": sorted(int(value) for value in train_df["season"].unique()),
        "valid_seasons": sorted(int(value) for value in valid_df["season"].unique()),
        "lookup_rows": int(len(inference_lookup)),
        "valid_unseen_row_rate": float(
            valid_features[PITCHER_SCORING_POS_COUNT_FEATURE].eq(0).mean()
        ),
        "season_audit": season_audit,
    }
    return train_features, valid_features, inference_lookup, metadata


def _build_target_history(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame,
    smoothing: float,
    entity_col: str,
    count_feature: str,
    rate_feature: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build strict-prior-season train features and train-only valid lookup features.

    Training rows for season ``s`` use targets only from train seasons earlier than
    ``s``. Validation rows use all supplied training seasons. The returned lookup is
    suitable for future inference and contains no validation targets.
    """

    smoothing = float(smoothing)
    _validate_inputs(train_df, valid_df, smoothing, entity_col)
    feature_columns = [count_feature, rate_feature]

    train_features = pd.DataFrame(index=train_df.index, columns=feature_columns)
    season_audit: list[dict[str, Any]] = []

    for season in sorted(int(value) for value in train_df["season"].unique()):
        season_mask = train_df["season"].eq(season)
        season_rows = train_df.loc[season_mask]
        history = train_df.loc[train_df["season"].lt(season)]
        fallback = (
            float(history[TARGET_COL].mean())
            if not history.empty
            else INITIAL_FALLBACK
        )
        stats = _history_stats(
            history,
            smoothing,
            fallback,
            entity_col,
            count_feature,
            rate_feature,
        )
        mapped = _map_stats(
            season_rows,
            stats,
            fallback,
            entity_col,
            count_feature,
            rate_feature,
        )
        train_features.loc[season_rows.index, feature_columns] = mapped
        unseen_row_rate = float(mapped[count_feature].eq(0).mean())
        season_audit.append(
            {
                "season": season,
                "history_seasons": sorted(
                    int(value) for value in history["season"].unique()
                ),
                "history_rows": int(len(history)),
                "fallback": fallback,
                "unseen_row_rate": unseen_row_rate,
            }
        )

    train_features[count_feature] = train_features[count_feature].astype("int32")
    train_features[rate_feature] = train_features[rate_feature].astype("float32")

    inference_fallback = float(train_df[TARGET_COL].mean())
    inference_stats = _history_stats(
        train_df,
        smoothing,
        inference_fallback,
        entity_col,
        count_feature,
        rate_feature,
    )
    valid_features = _map_stats(
        valid_df,
        inference_stats,
        inference_fallback,
        entity_col,
        count_feature,
        rate_feature,
    )

    lookup = inference_stats.reset_index()[
        [entity_col, count_feature, rate_feature]
    ].copy()
    lookup[entity_col] = lookup[entity_col].astype(train_df[entity_col].dtype)
    lookup[count_feature] = lookup[count_feature].astype("int32")
    lookup[rate_feature] = lookup[rate_feature].astype("float32")
    lookup = lookup.sort_values(entity_col).reset_index(drop=True)

    metadata: dict[str, Any] = {
        "enabled": True,
        "mode": "strict_prior_season_expanding",
        "entity_column": entity_col,
        "target_column": TARGET_COL,
        "feature_columns": feature_columns,
        "smoothing": smoothing,
        "initial_fallback": INITIAL_FALLBACK,
        "inference_fallback": inference_fallback,
        "train_seasons": sorted(int(value) for value in train_df["season"].unique()),
        "valid_seasons": sorted(int(value) for value in valid_df["season"].unique()),
        "lookup_rows": int(len(lookup)),
        "valid_unseen_row_rate": float(valid_features[count_feature].eq(0).mean()),
        "season_audit": season_audit,
    }
    return train_features, valid_features, lookup, metadata


def apply_pitcher_target_lookup(
    rows: pd.DataFrame,
    lookup: pd.DataFrame,
    fallback: float,
) -> pd.DataFrame:
    """Apply a persisted train-only pitcher lookup to inference rows."""

    return _apply_target_lookup(
        rows,
        lookup,
        fallback,
        entity_col=PITCHER_ENTITY_COL,
        count_feature=PITCHER_COUNT_FEATURE,
        rate_feature=PITCHER_RATE_FEATURE,
    )


def apply_batter_target_lookup(
    rows: pd.DataFrame,
    lookup: pd.DataFrame,
    fallback: float,
) -> pd.DataFrame:
    """Apply a persisted train-only batter lookup to inference rows."""

    return _apply_target_lookup(
        rows,
        lookup,
        fallback,
        entity_col=BATTER_ENTITY_COL,
        count_feature=BATTER_COUNT_FEATURE,
        rate_feature=BATTER_RATE_FEATURE,
    )


def _apply_target_lookup(
    rows: pd.DataFrame,
    lookup: pd.DataFrame,
    fallback: float,
    entity_col: str,
    count_feature: str,
    rate_feature: str,
) -> pd.DataFrame:
    """Apply a persisted train-only entity lookup to inference rows."""

    required_lookup = {entity_col, count_feature, rate_feature}
    missing_lookup = required_lookup - set(lookup.columns)
    if missing_lookup:
        raise ValueError(f"lookup 필수 컬럼이 없습니다: {sorted(missing_lookup)}")
    if entity_col not in rows.columns:
        raise ValueError(f"입력 데이터에 {entity_col}이 없습니다")
    if lookup[entity_col].duplicated().any():
        raise ValueError(f"lookup의 {entity_col}이 중복되었습니다")
    indexed = lookup.set_index(entity_col)
    return _map_stats(
        rows,
        indexed,
        float(fallback),
        entity_col,
        count_feature,
        rate_feature,
    )
