#!/usr/bin/env python3
"""Strict-forward evaluation of reference-10 v66 deviations on the 113A path.

This script deliberately evaluates one frozen hypothesis.  It does not read
test.csv, search weights, package a submission, or mutate the 113A champion.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "model/REF4-113A-V66-NESTED-117A"
TRAIN_PATH = ROOT / "data/train.csv"
ANCHOR_PATH = ROOT / "model/REF4-110-ORIGINAL-R2/expert_oof.csv"
TARGET = "control_success"
PRIOR = 0.523766
CLIP = (0.005, 0.995)
VALIDATION_YEARS = (2022, 2023, 2024)
TIME_WEIGHTS = {2022: 0.2, 2023: 0.3, 2024: 0.5}
AXES = (
    ("platoon", "pitcher_id", "pitcher_hand_key", 300.0, 0.12),
    (
        "advantage",
        "pitcher_hand_key",
        "pitcher_hand_advantage_key",
        2000.0,
        0.495,
    ),
    ("runner", "pitcher_hand_key", "runner_key", 2000.0, 0.27),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def add_context(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["pitcher_hand_key"] = (
        output["pitcher_id"].astype(str)
        + "|"
        + output["batter_hand"].astype(str)
    )
    output["advantage"] = (
        output["strikes_before"] > output["balls_before"]
    ).astype(np.int8)
    output["pitcher_hand_advantage_key"] = (
        output["pitcher_hand_key"] + "|" + output["advantage"].astype(str)
    )
    output["runner_gate"] = output["num_runners_on"].gt(0).astype(np.int8)
    output["runner_key"] = (
        output["pitcher_hand_key"] + "|" + output["runner_gate"].astype(str)
    )
    return output


def disjoint_features(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    n_p = frame["asof_pitcher_n"].fillna(0).to_numpy(float)
    n_b = frame["asof_batter_n"].fillna(0).to_numpy(float)
    p_rate = frame["asof_pitcher_success_rate"].fillna(PRIOR).to_numpy(float)
    prev1 = frame["asof_pitcher_prev1_game_success_rate"].fillna(PRIOR).to_numpy(float)
    b_rate = frame["asof_batter_success_rate"].fillna(PRIOR).to_numpy(float)
    b_mid = frame["asof_batter_middle_rate"].fillna(PRIOR).to_numpy(float)
    gamma_p = n_p / (n_p + 20.0)
    gamma_b = n_b / (n_b + 40.0)
    p_shrunk = gamma_p * p_rate + (1.0 - gamma_p) * PRIOR
    b_shrunk = gamma_b * (1.0 - b_rate) + (1.0 - gamma_b) * PRIOR
    p_left = frame["pitcher_hand"].astype(str).eq("L").to_numpy(float)
    b_left = frame["batter_hand"].astype(str).eq("L").to_numpy(float)
    balls = frame["balls_before"].fillna(0).to_numpy(float)
    strikes = frame["strikes_before"].fillna(0).to_numpy(float)
    features = pd.DataFrame(
        {
            "eb_pitcher_shrunk": p_shrunk,
            "eb_batter_shrunk": b_shrunk,
            "eb_form_gap": prev1 - (1.0 - b_rate),
            "gamma_p": gamma_p,
            "gamma_b": gamma_b,
            "platoon": (p_left != b_left).astype(float),
            "count_pressure": (balls - strikes) * (balls + strikes + 1.0) / 7.0,
            "is_2s": (strikes == 2).astype(float),
            "is_3b": (balls == 3).astype(float),
            "asof_pitcher_success_rate": p_rate,
            "asof_pitcher_prev1_game_success_rate": prev1,
            "asof_batter_middle_rate": b_mid,
            "li": frame["li"].fillna(0.98).to_numpy(float),
        }
    )
    anchor = np.clip(0.70 * p_rate + 0.30 * prev1, 0.05, 0.95)
    return features, anchor


def fit_disjoint_fold(
    train_rows: pd.DataFrame, valid_rows: pd.DataFrame
) -> np.ndarray:
    x_train, anchor_train = disjoint_features(train_rows)
    x_valid, _ = disjoint_features(valid_rows)
    residual = train_rows[TARGET].to_numpy(float) - anchor_train
    model = CatBoostRegressor(
        iterations=220,
        depth=5,
        learning_rate=0.035,
        l2_leaf_reg=30.0,
        loss_function="RMSE",
        random_seed=42,
        thread_count=-1,
        allow_writing_files=False,
        verbose=False,
    )
    model.fit(x_train, residual)
    return model.predict(x_valid).astype(float)


def nested_table(
    history: pd.DataFrame, parent: str, child: str, shrinkage: float
) -> pd.Series:
    parent_mean = history.groupby(parent, sort=False, observed=True)[TARGET].mean()
    grouped = history.groupby(child, sort=False, observed=True).agg(
        child_sum=(TARGET, "sum"),
        child_n=(TARGET, "size"),
        parent_key=(parent, "first"),
    )
    child_rate = grouped["child_sum"] / grouped["child_n"]
    parent_rate = grouped["parent_key"].map(parent_mean)
    return (
        (child_rate - parent_rate)
        * grouped["child_n"]
        / (grouped["child_n"] + shrinkage)
    )


def v66_fold(
    history: pd.DataFrame, valid_rows: pd.DataFrame
) -> tuple[np.ndarray, dict[str, dict[str, float | int]]]:
    correction = np.zeros(len(valid_rows), dtype=float)
    audit: dict[str, dict[str, float | int]] = {}
    for name, parent, child, shrinkage, weight in AXES:
        table = nested_table(history, parent, child, shrinkage)
        values = valid_rows[child].map(table).fillna(0.0).to_numpy(float)
        correction += weight * values
        audit[name] = {
            "table_keys": int(len(table)),
            "shrinkage": shrinkage,
            "weight": weight,
            "coverage": float(np.mean(values != 0.0)),
            "raw_mean": float(values.mean()),
            "raw_std": float(values.std()),
        }
    return correction, audit


def brier(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.square(target - prediction)))


def pitcher_bootstrap(
    target: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    pitcher: np.ndarray,
    repeats: int = 10000,
    seed: int = 1172024,
) -> dict[str, float | int]:
    row_delta = np.square(candidate - target) - np.square(baseline - target)
    grouped = pd.DataFrame(
        {"pitcher": pitcher.astype(str), "delta": row_delta}
    ).groupby("pitcher", sort=False)["delta"].agg(["sum", "size"])
    sums = grouped["sum"].to_numpy(float)
    sizes = grouped["size"].to_numpy(float)
    rng = np.random.default_rng(seed)
    values = np.empty(repeats, dtype=float)
    for start in range(0, repeats, 64):
        count = min(64, repeats - start)
        sample = rng.integers(0, len(grouped), size=(count, len(grouped)))
        values[start : start + count] = (
            sums[sample].sum(axis=1) / sizes[sample].sum(axis=1)
        )
    return {
        "repeats": repeats,
        "pitcher_clusters": int(len(grouped)),
        "mean_delta": float(values.mean()),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
    }


def main() -> None:
    started = time.time()
    contract = json.loads((EXPERIMENT / "audit_contract.json").read_text())
    preflight = json.loads((EXPERIMENT / "preflight_report.json").read_text())
    if contract["status"] != "LOCKED_BEFORE_RESULTS":
        raise RuntimeError("audit contract is not locked")
    if preflight["status"] != "AUDIT_VERIFIED":
        raise RuntimeError("preflight is not audit verified")
    if sha256(TRAIN_PATH) != preflight["checks"]["train_sha256"]:
        raise RuntimeError("train.csv hash changed after preflight")
    if sha256(ANCHOR_PATH) != preflight["source_provenance"]["strict_anchor_sha256"]:
        raise RuntimeError("strict anchor hash changed after preflight")

    columns = [
        "row_id",
        "season",
        "game_type",
        "pitcher_id",
        "batter_id",
        "pitcher_hand",
        "batter_hand",
        "balls_before",
        "strikes_before",
        "num_runners_on",
        "li",
        "asof_pitcher_n",
        "asof_batter_n",
        "asof_pitcher_success_rate",
        "asof_pitcher_prev1_game_success_rate",
        "asof_batter_success_rate",
        "asof_batter_middle_rate",
        TARGET,
    ]
    raw = add_context(pd.read_csv(TRAIN_PATH, usecols=columns, low_memory=False))
    anchor = pd.read_csv(
        ANCHOR_PATH,
        usecols=["row_id", "season", "game_type", "pitcher_id", "target", "p109c"],
        low_memory=False,
    )
    validation = raw.loc[raw["season"].isin(VALIDATION_YEARS)].reset_index(drop=True)
    if not np.array_equal(validation["row_id"].astype(str), anchor["row_id"].astype(str)):
        raise RuntimeError("anchor row order mismatch")
    if not np.array_equal(validation[TARGET].to_numpy(float), anchor["target"].to_numpy(float)):
        raise RuntimeError("anchor target mismatch")

    fold_frames: list[pd.DataFrame] = []
    fold_audit: dict[str, object] = {}
    for year in VALIDATION_YEARS:
        print(f"[117A] strict fold {year}", flush=True)
        fit = raw["season"].lt(year)
        valid = raw["season"].eq(year)
        history = raw.loc[fit].reset_index(drop=True)
        rows = raw.loc[valid].reset_index(drop=True)
        local_anchor = anchor.loc[anchor["season"].eq(year)].reset_index(drop=True)
        if not np.array_equal(rows["row_id"].astype(str), local_anchor["row_id"].astype(str)):
            raise RuntimeError(f"fold {year} anchor row mismatch")

        eb_residual = fit_disjoint_fold(history, rows)
        regular = rows["game_type"].astype(str).eq("R").to_numpy()
        p113a = local_anchor["p109c"].to_numpy(float).copy()
        p113a[regular] += 0.035 * eb_residual[regular]
        p113a = np.clip(p113a, *CLIP)
        v66_delta, axes_audit = v66_fold(history, rows)
        p117a = np.clip(p113a + v66_delta, *CLIP)
        y = rows[TARGET].to_numpy(float)
        fold_frames.append(
            pd.DataFrame(
                {
                    "row_id": rows["row_id"].astype(str),
                    "season": year,
                    "game_type": rows["game_type"].astype(str),
                    "pitcher_id": rows["pitcher_id"].astype(str),
                    "target": y,
                    "p109c": local_anchor["p109c"].to_numpy(float),
                    "disjoint_eb_residual": eb_residual,
                    "p113a_strict": p113a,
                    "v66_delta": v66_delta,
                    "p117a": p117a,
                }
            )
        )
        fold_audit[str(year)] = {
            "train_seasons": sorted(map(int, history["season"].unique())),
            "train_rows": int(len(history)),
            "valid_rows": int(len(rows)),
            "validation_labels_used_in_fit": False,
            "axes": axes_audit,
            "p113a_brier": brier(y, p113a),
            "p117a_brier": brier(y, p117a),
            "delta_brier": brier(y, p117a) - brier(y, p113a),
            "mean_absolute_change": float(np.mean(np.abs(p117a - p113a))),
            "correction_range": [float(v66_delta.min()), float(v66_delta.max())],
        }

    predictions = pd.concat(fold_frames, ignore_index=True)
    deltas = {year: float(fold_audit[str(year)]["delta_brier"]) for year in VALIDATION_YEARS}
    weighted_delta = float(sum(TIME_WEIGHTS[y] * deltas[y] for y in VALIDATION_YEARS))
    mask_2024 = predictions["season"].eq(2024).to_numpy()
    bootstrap = pitcher_bootstrap(
        predictions.loc[mask_2024, "target"].to_numpy(float),
        predictions.loc[mask_2024, "p113a_strict"].to_numpy(float),
        predictions.loc[mask_2024, "p117a"].to_numpy(float),
        predictions.loc[mask_2024, "pitcher_id"].to_numpy(),
    )
    gate_results = {
        "delta_2024": deltas[2024] <= -0.0001,
        "delta_2022": deltas[2022] <= 0.00005,
        "time_weighted": weighted_delta < 0.0,
        "worst_season": max(deltas.values()) <= 0.00005,
        "bootstrap_2024_ci_high_below_zero": bootstrap["ci_high"] < 0.0,
    }
    passed = bool(all(gate_results.values()))
    output_path = EXPERIMENT / "oof_predictions.csv"
    predictions.to_csv(output_path, index=False)
    report = {
        "experiment_id": "REF4-113A-V66-NESTED-117A",
        "status": "PENDING_AUDIT",
        "candidate_status": "PERFORMANCE_GATE_PASS_PENDING_AUDIT" if passed else "REJECTED_PERFORMANCE_GATE_PENDING_AUDIT",
        "hypothesis_count": 1,
        "model_count": 3,
        "oof_rows": int(len(predictions)),
        "folds": fold_audit,
        "time_weighted_delta": weighted_delta,
        "worst_season_delta": max(deltas.values()),
        "pitcher_cluster_bootstrap_2024": bootstrap,
        "gate_results": gate_results,
        "performance_gate_pass": passed,
        "prediction_ranges": {
            "p113a_strict": [float(predictions["p113a_strict"].min()), float(predictions["p113a_strict"].max())],
            "p117a": [float(predictions["p117a"].min()), float(predictions["p117a"].max())],
        },
        "elapsed_seconds": float(time.time() - started),
        "test_read": False,
        "zip_created": False,
        "oof_predictions_sha256": sha256(output_path),
    }
    (EXPERIMENT / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
