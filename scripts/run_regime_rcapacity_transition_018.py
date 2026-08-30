#!/usr/bin/env python3
"""Run the preregistered R-capacity regime split on one temporal fold.

This is a research-only train-data screen.  It intentionally does not read
test.csv, save a production model, or build a submission archive.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import catboost as cb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_combo_full_candidate_002 import COLS as BASE_COLS, extras, prep
from scripts.screen_trackman_context_003 import NEW, attach, load_tm
from src.asof_state_features import add_state_for_cutoff, add_state_walkforward
from src.target_aggregates import build_pitcher_count_state_target_history

BASELINE_OOF = ROOT / "model" / "COMBO-RESID3-OOF-007" / "oof_predictions.csv"
FEATURE_COLUMNS = BASE_COLS + NEW
BASELINE_WEIGHT = 0.25
CANDIDATE_WEIGHT = 0.75
REQUIRED_RELATIVE_IMPROVEMENT = 0.02


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric(pred: np.ndarray, target: np.ndarray) -> float:
    return float(1e5 * np.corrcoef(pred, target)[0, 1] ** 2)


def brier(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, required=True, choices=(2023, 2024))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    season = args.season
    start = time.time()
    experiment_id = f"REGIME-RCAPACITY-TRANSITION-018-{season}"
    output_dir = ROOT / "model" / experiment_id
    if output_dir.exists():
        raise RuntimeError(f"refusing to overwrite existing experiment directory: {output_dir}")

    raw = pd.read_csv(ROOT / "data" / "train.csv", encoding="utf-8-sig")
    if raw.row_id.duplicated().any():
        raise RuntimeError("train row_id is not unique")
    train = raw.loc[raw.season < season].copy()
    valid = raw.loc[raw.season == season].copy()
    if train.empty or valid.empty:
        raise RuntimeError(f"empty temporal fold: {season}")

    trackman, mapping = load_tm()
    train_state = add_state_walkforward(
        train.drop(columns=["row_id", "control_success"]), season
    )
    valid_state = add_state_for_cutoff(
        valid.drop(columns=["row_id", "control_success"]),
        train.drop(columns=["row_id", "control_success"]),
    )
    train_hist, valid_hist, lookup, _ = build_pitcher_count_state_target_history(
        train, valid.assign(control_success=0), smoothing=100.0
    )
    x_train = pd.concat(
        [extras(train_state.reset_index(drop=True)), train_hist.reset_index(drop=True)],
        axis=1,
    )
    x_valid = pd.concat(
        [extras(valid_state.reset_index(drop=True)), valid_hist.reset_index(drop=True)],
        axis=1,
    )
    x_train = attach(x_train, trackman, season)[FEATURE_COLUMNS]
    x_valid = attach(x_valid, trackman, season)[FEATURE_COLUMNS]
    x_train, categorical_columns = prep(x_train)
    x_valid, _ = prep(x_valid)
    y_train = train.control_success.to_numpy(np.int8)
    y_valid = valid.control_success.to_numpy(np.int8)

    candidate_pred = np.zeros(len(valid), dtype=float)
    regime_results: list[dict[str, object]] = []
    for regime in ("F", "R"):
        train_mask = train.game_type.astype(str).eq(regime).to_numpy()
        valid_mask = valid.game_type.astype(str).eq(regime).to_numpy()
        if regime == "F":
            train_mask &= train.season.ne(2020).to_numpy()
            model_params = {
                "iterations": 300,
                "learning_rate": 0.05,
                "depth": 6,
                "l2_leaf_reg": 10,
                "early_stopping_rounds": 50,
            }
        else:
            model_params = {
                "iterations": 600,
                "learning_rate": 0.03,
                "depth": 7,
                "l2_leaf_reg": 20,
                "early_stopping_rounds": 80,
            }
        if not train_mask.any() or not valid_mask.any():
            raise RuntimeError(f"empty regime partition: {season}/{regime}")
        model = cb.CatBoostClassifier(
            **model_params,
            loss_function="Logloss",
            thread_count=3,
            random_seed=42,
            allow_writing_files=False,
            verbose=False,
        )
        model.fit(
            cb.Pool(
                x_train.loc[train_mask],
                label=y_train[train_mask],
                cat_features=categorical_columns,
                feature_names=FEATURE_COLUMNS,
            ),
            eval_set=cb.Pool(
                x_valid.loc[valid_mask],
                label=y_valid[valid_mask],
                cat_features=categorical_columns,
                feature_names=FEATURE_COLUMNS,
            ),
            verbose=False,
        )
        regime_pred = model.predict_proba(
            cb.Pool(
                x_valid.loc[valid_mask],
                cat_features=categorical_columns,
                feature_names=FEATURE_COLUMNS,
            )
        )[:, 1]
        candidate_pred[valid_mask] = regime_pred
        regime_results.append(
            {
                "regime": regime,
                "train_rows": int(train_mask.sum()),
                "valid_rows": int(valid_mask.sum()),
                "excluded_train_seasons": [2020] if regime == "F" else [],
                "best_trees": int(model.get_best_iteration() + 1),
                "metric": metric(regime_pred, y_valid[valid_mask]),
                "brier": brier(regime_pred, y_valid[valid_mask]),
                "model_params": model_params,
            }
        )
        del model, regime_pred
        gc.collect()

    if not np.isfinite(candidate_pred).all() or not (
        (candidate_pred >= 0) & (candidate_pred <= 1)
    ).all():
        raise RuntimeError("candidate prediction finite/range contract failed")

    baseline = pd.read_csv(
        BASELINE_OOF, usecols=["row_id", "season", "target", "pred"]
    )
    baseline = baseline.loc[baseline.season.eq(season)].copy()
    candidate_frame = pd.DataFrame(
        {
            "row_id": valid.row_id.to_numpy(),
            "season": season,
            "target": y_valid,
            "candidate_pred": candidate_pred,
        }
    )
    paired = baseline.merge(
        candidate_frame,
        on=["row_id", "season", "target"],
        how="inner",
        validate="one_to_one",
    )
    if len(paired) != len(valid) or paired.row_id.nunique() != len(valid):
        raise RuntimeError("baseline/candidate paired row contract failed")
    paired["baseline_pred"] = paired.pop("pred")
    paired["blend_pred"] = (
        BASELINE_WEIGHT * paired.baseline_pred
        + CANDIDATE_WEIGHT * paired.candidate_pred
    )
    if not np.isfinite(paired[["baseline_pred", "blend_pred"]].to_numpy()).all():
        raise RuntimeError("baseline/blend finite contract failed")

    target = paired.target.to_numpy(np.int8)
    baseline_pred = paired.baseline_pred.to_numpy(float)
    aligned_candidate_pred = paired.candidate_pred.to_numpy(float)
    blend_pred = paired.blend_pred.to_numpy(float)
    metrics = {
        "baseline": {
            "metric": metric(baseline_pred, target),
            "brier": brier(baseline_pred, target),
        },
        "split_rcapacity_raw": {
            "metric": metric(aligned_candidate_pred, target),
            "brier": brier(aligned_candidate_pred, target),
        },
        "baseline25_split75": {
            "metric": metric(blend_pred, target),
            "brier": brier(blend_pred, target),
        },
    }
    for name in ("split_rcapacity_raw", "baseline25_split75"):
        metrics[name]["relative_improvement"] = (
            metrics[name]["metric"] / metrics["baseline"]["metric"] - 1.0
        )
        metrics[name]["delta_brier"] = (
            metrics[name]["brier"] - metrics["baseline"]["brier"]
        )
    metrics["baseline25_split75"]["gate_pass_2pct"] = bool(
        metrics["baseline25_split75"]["relative_improvement"]
        >= REQUIRED_RELATIVE_IMPROVEMENT
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    predictions_path = output_dir / "paired_predictions.csv"
    paired.sort_values("row_id").to_csv(predictions_path, index=False)
    report = {
        "experiment_id": experiment_id,
        "season": season,
        "hypothesis": "fixed F exclude-2020 plus R depth-7 capacity transfers across time",
        "official_train_only": True,
        "test_used": False,
        "external_data_used": False,
        "feature_count": len(FEATURE_COLUMNS),
        "train_rows": int(len(train)),
        "valid_rows": int(len(valid)),
        "row_id_unique": bool(paired.row_id.nunique() == len(paired)),
        "prediction_range": {
            "candidate": [
                float(aligned_candidate_pred.min()),
                float(aligned_candidate_pred.max()),
            ],
            "blend": [float(blend_pred.min()), float(blend_pred.max())],
        },
        "fixed_blend_weights": {
            "baseline": BASELINE_WEIGHT,
            "split_rcapacity": CANDIDATE_WEIGHT,
            "sum": BASELINE_WEIGHT + CANDIDATE_WEIGHT,
        },
        "required_relative_improvement": REQUIRED_RELATIVE_IMPROVEMENT,
        "regimes": regime_results,
        "metrics": metrics,
        "input_hashes": {
            "train_csv": sha256_file(ROOT / "data" / "train.csv"),
            "trackman_csv": sha256_file(ROOT / "data" / "trackman_history.csv"),
            "baseline_oof": sha256_file(BASELINE_OOF),
            "pitcher_map": sha256_file(
                ROOT / "model" / "TRACKMAN-MAP-004" / "pitcher_id_map.csv"
            ),
        },
        "predictions_sha256": sha256_file(predictions_path),
        "elapsed_sec": round(time.time() - start, 3),
        "status": "SCREEN_COMPLETE_AUDIT_PENDING",
        "submission_status": "HOLD_NO_ZIP",
    }
    report_path = output_dir / "screen_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
