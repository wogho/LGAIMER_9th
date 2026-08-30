#!/usr/bin/env python3
"""Verify leakage barriers, fallback behavior, and lookup determinism."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.target_aggregates import (
    BATTER_COUNT_FEATURE,
    BATTER_ENTITY_COL,
    BATTER_RATE_FEATURE,
    COUNT_FEATURE,
    ENTITY_COL,
    RATE_FEATURE,
    apply_batter_target_lookup,
    apply_pitcher_count_state_target_lookup,
    apply_pitcher_scoring_pos_target_lookup,
    apply_pitcher_target_lookup,
    build_batter_target_history,
    build_pitcher_count_state_target_history,
    build_pitcher_scoring_pos_target_history,
    build_pitcher_target_history,
    PITCHER_COUNT_STATE_COUNT_FEATURE,
    PITCHER_COUNT_STATE_DELTA_FEATURE,
    PITCHER_SCORING_POS_COUNT_FEATURE,
    PITCHER_SCORING_POS_DELTA_FEATURE,
)
from src.features import build_features, load_data


def toy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2019, 2019, 2020, 2020, 2021, 2021],
            "pitcher_id": [1, 1, 1, 2, 1, 3],
            "batter_id": [10, 10, 10, 20, 10, 30],
            "balls_before": [0, 0, 0, 0, 0, 3],
            "strikes_before": [0, 1, 0, 0, 0, 2],
            "runner_on_2b": [0, 1, 0, 0, 0, 1],
            "runner_on_3b": [0, 0, 0, 0, 0, 0],
            "control_success": [1, 0, 1, 0, 0, 1],
        }
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify target aggregate leakage barriers and experiment artifacts"
    )
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--train-path", type=Path, default=ROOT / "data" / "train.csv")
    return parser.parse_args()


def verify_toy_contract() -> None:
    train = toy_frame().query("season < 2021").copy()
    valid = toy_frame().query("season == 2021").copy()

    train_features, valid_features, lookup, metadata = build_pitcher_target_history(
        train,
        valid,
        smoothing=2.0,
    )

    first_season = train["season"].eq(2019)
    assert train_features.loc[first_season, COUNT_FEATURE].eq(0).all()
    assert np.allclose(train_features.loc[first_season, RATE_FEATURE], 0.5)

    row_2020_pitcher_1 = train["season"].eq(2020) & train["pitcher_id"].eq(1)
    assert train_features.loc[row_2020_pitcher_1, COUNT_FEATURE].item() == 2
    assert np.isclose(train_features.loc[row_2020_pitcher_1, RATE_FEATURE].item(), 0.5)

    # Changing same-season targets must not alter that season's aggregate features.
    mutated = train.copy()
    mutated.loc[mutated["season"].eq(2020), "control_success"] ^= 1
    mutated_features, _, _, _ = build_pitcher_target_history(
        mutated,
        valid,
        smoothing=2.0,
    )
    pd.testing.assert_frame_equal(
        train_features.loc[train["season"].eq(2020)],
        mutated_features.loc[train["season"].eq(2020)],
    )

    assert valid_features.loc[valid["pitcher_id"].eq(1), COUNT_FEATURE].item() == 3
    assert valid_features.loc[valid["pitcher_id"].eq(3), COUNT_FEATURE].item() == 0
    assert np.isclose(
        valid_features.loc[valid["pitcher_id"].eq(3), RATE_FEATURE].item(),
        train["control_success"].mean(),
    )

    applied = apply_pitcher_target_lookup(
        valid,
        lookup,
        metadata["inference_fallback"],
    )
    pd.testing.assert_frame_equal(valid_features, applied)

    reversed_train = train.iloc[::-1].copy()
    reversed_features, reversed_valid, reversed_lookup, reversed_metadata = (
        build_pitcher_target_history(reversed_train, valid, smoothing=2.0)
    )
    pd.testing.assert_frame_equal(
        train_features.sort_index(),
        reversed_features.sort_index(),
    )
    pd.testing.assert_frame_equal(valid_features, reversed_valid)
    pd.testing.assert_frame_equal(lookup, reversed_lookup)
    assert metadata["inference_fallback"] == reversed_metadata["inference_fallback"]

    print("PASS strict prior-season leakage barrier")
    print("PASS unseen pitcher fallback")
    print("PASS persisted lookup equivalence")
    print("PASS row-order determinism")


def verify_batter_toy_contract() -> None:
    train = toy_frame().query("season < 2021").copy()
    valid = toy_frame().query("season == 2021").copy()
    train_features, valid_features, lookup, metadata = build_batter_target_history(
        train,
        valid,
        smoothing=2.0,
    )

    first_season = train["season"].eq(2019)
    assert train_features.loc[first_season, BATTER_COUNT_FEATURE].eq(0).all()
    assert np.allclose(train_features.loc[first_season, BATTER_RATE_FEATURE], 0.5)

    mutated = train.copy()
    mutated.loc[mutated["season"].eq(2020), "control_success"] ^= 1
    mutated_features, _, _, _ = build_batter_target_history(
        mutated,
        valid,
        smoothing=2.0,
    )
    pd.testing.assert_frame_equal(
        train_features.loc[train["season"].eq(2020)],
        mutated_features.loc[train["season"].eq(2020)],
    )

    assert valid_features.loc[valid["batter_id"].eq(10), BATTER_COUNT_FEATURE].item() == 3
    assert valid_features.loc[valid["batter_id"].eq(30), BATTER_COUNT_FEATURE].item() == 0
    applied = apply_batter_target_lookup(
        valid,
        lookup,
        metadata["inference_fallback"],
    )
    pd.testing.assert_frame_equal(valid_features, applied)
    print("PASS batter strict prior-season leakage barrier")
    print("PASS unseen batter fallback and lookup equivalence")


def verify_pitcher_count_state_toy_contract() -> None:
    train = toy_frame().query("season < 2021").copy()
    valid = toy_frame().query("season == 2021").copy()
    train_features, valid_features, lookup, metadata = (
        build_pitcher_count_state_target_history(train, valid, smoothing=2.0)
    )

    first_season = train["season"].eq(2019)
    assert train_features.loc[
        first_season, PITCHER_COUNT_STATE_COUNT_FEATURE
    ].eq(0).all()
    assert np.allclose(
        train_features.loc[first_season, PITCHER_COUNT_STATE_DELTA_FEATURE], 0.0
    )

    row_2020_pitcher_1 = train["season"].eq(2020) & train["pitcher_id"].eq(1)
    assert train_features.loc[
        row_2020_pitcher_1, PITCHER_COUNT_STATE_COUNT_FEATURE
    ].item() == 1
    assert np.isclose(
        train_features.loc[
            row_2020_pitcher_1, PITCHER_COUNT_STATE_DELTA_FEATURE
        ].item(),
        1.0 / 6.0,
    )

    mutated = train.copy()
    mutated.loc[mutated["season"].eq(2020), "control_success"] ^= 1
    mutated_features, _, _, _ = build_pitcher_count_state_target_history(
        mutated, valid, smoothing=2.0
    )
    pd.testing.assert_frame_equal(
        train_features.loc[train["season"].eq(2020)],
        mutated_features.loc[train["season"].eq(2020)],
    )

    valid_pitcher_1 = valid["pitcher_id"].eq(1)
    assert valid_features.loc[
        valid_pitcher_1, PITCHER_COUNT_STATE_COUNT_FEATURE
    ].item() == 2
    assert np.isclose(
        valid_features.loc[
            valid_pitcher_1, PITCHER_COUNT_STATE_DELTA_FEATURE
        ].item(),
        0.2,
    )
    assert valid_features.loc[
        valid["pitcher_id"].eq(3), PITCHER_COUNT_STATE_COUNT_FEATURE
    ].item() == 0
    assert valid_features.loc[
        valid["pitcher_id"].eq(3), PITCHER_COUNT_STATE_DELTA_FEATURE
    ].item() == 0.0

    unseen_condition = pd.DataFrame(
        {
            "pitcher_id": [1],
            "balls_before": [3],
            "strikes_before": [2],
        }
    )
    unseen_features = apply_pitcher_count_state_target_lookup(
        unseen_condition, lookup, metadata["inference_fallback"]
    )
    assert unseen_features[PITCHER_COUNT_STATE_COUNT_FEATURE].item() == 0
    assert unseen_features[PITCHER_COUNT_STATE_DELTA_FEATURE].item() == 0.0

    applied = apply_pitcher_count_state_target_lookup(
        valid, lookup, metadata["inference_fallback"]
    )
    pd.testing.assert_frame_equal(valid_features, applied)

    reversed_features, reversed_valid, reversed_lookup, _ = (
        build_pitcher_count_state_target_history(
            train.iloc[::-1].copy(), valid, smoothing=2.0
        )
    )
    pd.testing.assert_frame_equal(
        train_features.sort_index(), reversed_features.sort_index()
    )
    pd.testing.assert_frame_equal(valid_features, reversed_valid)
    pd.testing.assert_frame_equal(lookup, reversed_lookup)
    print("PASS pitcher/count-state strict prior-season leakage barrier")
    print("PASS unseen pitcher and unseen condition zero fallback")
    print("PASS pitcher/count-state smoothing formula and lookup equivalence")


def verify_pitcher_scoring_pos_toy_contract() -> None:
    train = toy_frame().query("season < 2021").copy()
    valid = toy_frame().query("season == 2021").copy()
    train_features, valid_features, lookup, metadata = (
        build_pitcher_scoring_pos_target_history(train, valid, smoothing=2.0)
    )

    first_season = train["season"].eq(2019)
    assert train_features.loc[
        first_season, PITCHER_SCORING_POS_COUNT_FEATURE
    ].eq(0).all()
    assert np.allclose(
        train_features.loc[first_season, PITCHER_SCORING_POS_DELTA_FEATURE], 0.0
    )

    row_2020_pitcher_1 = train["season"].eq(2020) & train["pitcher_id"].eq(1)
    assert train_features.loc[
        row_2020_pitcher_1, PITCHER_SCORING_POS_COUNT_FEATURE
    ].item() == 1
    assert np.isclose(
        train_features.loc[
            row_2020_pitcher_1, PITCHER_SCORING_POS_DELTA_FEATURE
        ].item(),
        1.0 / 6.0,
    )

    mutated = train.copy()
    mutated.loc[mutated["season"].eq(2020), "control_success"] ^= 1
    mutated_features, _, _, _ = build_pitcher_scoring_pos_target_history(
        mutated, valid, smoothing=2.0
    )
    pd.testing.assert_frame_equal(
        train_features.loc[train["season"].eq(2020)],
        mutated_features.loc[train["season"].eq(2020)],
    )

    valid_pitcher_1 = valid["pitcher_id"].eq(1)
    assert valid_features.loc[
        valid_pitcher_1, PITCHER_SCORING_POS_COUNT_FEATURE
    ].item() == 2
    assert np.isclose(
        valid_features.loc[
            valid_pitcher_1, PITCHER_SCORING_POS_DELTA_FEATURE
        ].item(),
        0.2,
    )
    assert valid_features.loc[
        valid["pitcher_id"].eq(3), PITCHER_SCORING_POS_COUNT_FEATURE
    ].item() == 0
    assert valid_features.loc[
        valid["pitcher_id"].eq(3), PITCHER_SCORING_POS_DELTA_FEATURE
    ].item() == 0.0

    unseen_condition = pd.DataFrame(
        {
            "pitcher_id": [2],
            "runner_on_2b": [1],
            "runner_on_3b": [0],
        }
    )
    unseen_features = apply_pitcher_scoring_pos_target_lookup(
        unseen_condition, lookup, metadata["inference_fallback"]
    )
    assert unseen_features[PITCHER_SCORING_POS_COUNT_FEATURE].item() == 0
    assert unseen_features[PITCHER_SCORING_POS_DELTA_FEATURE].item() == 0.0

    applied = apply_pitcher_scoring_pos_target_lookup(
        valid, lookup, metadata["inference_fallback"]
    )
    pd.testing.assert_frame_equal(valid_features, applied)
    reversed_features, reversed_valid, reversed_lookup, _ = (
        build_pitcher_scoring_pos_target_history(
            train.iloc[::-1].copy(), valid, smoothing=2.0
        )
    )
    pd.testing.assert_frame_equal(
        train_features.sort_index(), reversed_features.sort_index()
    )
    pd.testing.assert_frame_equal(valid_features, reversed_valid)
    pd.testing.assert_frame_equal(lookup, reversed_lookup)
    print("PASS pitcher/scoring-position strict prior-season leakage barrier")
    print("PASS unseen pitcher and unseen scoring-position zero fallback")
    print("PASS pitcher/scoring-position smoothing formula and lookup equivalence")


def verify_experiment(experiment_dir: Path, train_path: Path) -> None:
    import lightgbm as lgb

    metadata = json.loads((experiment_dir / "metadata.json").read_text(encoding="utf-8"))
    aggregate_config = json.loads(
        (experiment_dir / "target_aggregate.json").read_text(encoding="utf-8")
    )
    persisted_features = json.loads(
        (experiment_dir / "feature_columns.json").read_text(encoding="utf-8")
    )
    saved_lookup = pd.read_csv(experiment_dir / aggregate_config["lookup_file"])

    raw = load_data(str(train_path), is_train=True)
    train = raw[raw["season"].isin(metadata["train_seasons"])].copy()
    valid = raw[raw["season"].eq(metadata["valid_season"])].copy()
    entity_column = aggregate_config["entity_column"]
    condition_column = aggregate_config.get("condition_column")
    if entity_column == ENTITY_COL and condition_column == "count_state":
        aggregate_builder = build_pitcher_count_state_target_history
        lookup_applier = apply_pitcher_count_state_target_lookup
    elif entity_column == ENTITY_COL and condition_column == "scoring_pos_runner":
        aggregate_builder = build_pitcher_scoring_pos_target_history
        lookup_applier = apply_pitcher_scoring_pos_target_lookup
    elif entity_column == ENTITY_COL:
        aggregate_builder = build_pitcher_target_history
        lookup_applier = apply_pitcher_target_lookup
    elif entity_column == BATTER_ENTITY_COL:
        aggregate_builder = build_batter_target_history
        lookup_applier = apply_batter_target_lookup
    else:
        raise ValueError(f"지원하지 않는 집계 엔터티입니다: {entity_column}")

    _, rebuilt_valid, rebuilt_lookup, rebuilt_config = aggregate_builder(
        train,
        valid,
        smoothing=aggregate_config["smoothing"],
    )

    saved_lookup = saved_lookup.astype(rebuilt_lookup.dtypes.to_dict())
    pd.testing.assert_frame_equal(saved_lookup, rebuilt_lookup, check_exact=True)
    applied_valid = lookup_applier(
        valid,
        saved_lookup,
        aggregate_config["inference_fallback"],
    )
    pd.testing.assert_frame_equal(applied_valid, rebuilt_valid, check_exact=True)

    expected_config = {
        **rebuilt_config,
        "lookup_file": aggregate_config["lookup_file"],
    }
    assert aggregate_config == expected_config
    assert metadata["target_aggregate"] == aggregate_config

    model_input = pd.concat([build_features(valid), applied_valid], axis=1)
    assert list(model_input.columns) == persisted_features
    booster = lgb.Booster(model_file=str(experiment_dir / "model.txt"))
    assert booster.feature_name() == persisted_features
    assert booster.num_feature() == len(persisted_features)

    print(
        f"PASS actual lookup rebuild ({len(rebuilt_lookup):,} "
        f"{aggregate_config['entity_column']} values)"
    )
    print(f"PASS actual validation lookup ({len(valid):,} rows)")
    print(f"PASS aggregate model contract ({len(persisted_features)} features)")


def main() -> None:
    args = parse_args()
    verify_toy_contract()
    verify_batter_toy_contract()
    verify_pitcher_count_state_toy_contract()
    verify_pitcher_scoring_pos_toy_contract()
    if args.experiment_dir is not None:
        verify_experiment(args.experiment_dir, args.train_path)


if __name__ == "__main__":
    main()
