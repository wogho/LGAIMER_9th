"""Row-independent production corrections for the REF4 119 candidates.

Every lookup/model in this module is fitted offline from the official training
and TrackMan history files.  No statistic is computed across evaluation rows.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PRIOR = 0.523766
PITCH_GROUPS = ("fastball", "breaking", "offspeed", "other")


def _hand_text(values: pd.Series) -> pd.Series:
    return values.map({1: "Left", 2: "Right", "1": "Left", "2": "Right"}).fillna(
        values.astype(str)
    )


def apply_abs_era_119a(raw: pd.DataFrame, prediction: np.ndarray, model_dir: Path) -> np.ndarray:
    meta = json.loads((model_dir / "abs_era_119a_meta.json").read_text(encoding="utf-8"))
    if not meta.get("enabled", False):
        return prediction.copy()
    table = pd.read_csv(model_dir / "abs_era_119a_table.csv")
    keys = pd.DataFrame(
        {
            "game_type": raw["game_type"].astype(str),
            "balls_before": raw["balls_before"].fillna(0).astype(int),
            "strikes_before": raw["strikes_before"].fillna(0).astype(int),
            "hand_match": (raw["pitcher_hand"].astype(str) == raw["batter_hand"].astype(str)).astype(int),
        },
        index=raw.index,
    )
    correction = keys.merge(
        table, on=["game_type", "balls_before", "strikes_before", "hand_match"], how="left", sort=False
    )["correction"].fillna(0.0).to_numpy(float)
    correction = np.clip(correction, -float(meta["delta_cap"]), float(meta["delta_cap"]))
    return np.clip(prediction + correction, 0.02, 0.98)


def _moe_context(raw: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    balls = raw["balls_before"].fillna(0).astype(int).clip(0, 3)
    strikes = raw["strikes_before"].fillna(0).astype(int).clip(0, 2)
    p_hand = raw["pitcher_hand"].astype(str)
    b_hand = raw["batter_hand"].astype(str)
    values: dict[str, np.ndarray] = {"intercept": np.ones(len(raw), dtype=float)}
    for b in range(4):
        for s in range(3):
            values[f"count_{b}_{s}"] = ((balls == b) & (strikes == s)).to_numpy(float)
    values.update(
        {
            "game_type_f": raw["game_type"].astype(str).eq("F").to_numpy(float),
            "pitcher_left": p_hand.isin(["1", "L", "Left"]).to_numpy(float),
            "batter_left": b_hand.isin(["1", "L", "Left"]).to_numpy(float),
            "hand_match": p_hand.eq(b_hand).to_numpy(float),
            "log_pitcher_n": np.log1p(raw["asof_pitcher_n"].fillna(0).clip(lower=0).to_numpy(float)) / 10.0,
            "log_batter_n": np.log1p(raw["asof_batter_n"].fillna(0).clip(lower=0).to_numpy(float)) / 10.0,
            "pitcher_rate": raw["asof_pitcher_success_rate"].fillna(PRIOR).to_numpy(float) - PRIOR,
            "pitcher_recent": raw["asof_pitcher_prev1_game_success_rate"].fillna(PRIOR).to_numpy(float) - PRIOR,
            "batter_rate": raw["asof_batter_success_rate"].fillna(PRIOR).to_numpy(float) - PRIOR,
            "li": raw["li"].fillna(0.98).clip(0, 5).to_numpy(float) / 5.0,
            "runners": raw["num_runners_on"].fillna(0).clip(0, 3).to_numpy(float) / 3.0,
            "inning": raw["inning"].fillna(1).clip(1, 15).to_numpy(float) / 15.0,
        }
    )
    columns = list(values)
    return np.column_stack([values[column] for column in columns]), columns


def _latent_probabilities(raw: pd.DataFrame, table_path: Path) -> tuple[np.ndarray, np.ndarray]:
    table = pd.read_csv(table_path, dtype={"pitcher_id": str, "batter_hand": str})
    keys = pd.DataFrame(
        {
            "pitcher_id": raw["pitcher_id"].astype(str),
            "balls_before": raw["balls_before"].fillna(0).astype(int),
            "strikes_before": raw["strikes_before"].fillna(0).astype(int),
            "batter_hand": _hand_text(raw["batter_hand"]),
        }
    )
    joined = keys.merge(
        table,
        on=["pitcher_id", "balls_before", "strikes_before", "batter_hand"],
        how="left",
        sort=False,
    )
    fallback = np.column_stack(
        [
            raw["asof_pitcher_fastball_rate"].fillna(0.50).to_numpy(float),
            raw["asof_pitcher_breaking_rate"].fillna(0.30).to_numpy(float),
            raw["asof_pitcher_offspeed_rate"].fillna(0.15).to_numpy(float),
        ]
    )
    fallback = np.column_stack([fallback, np.maximum(0.0, 1.0 - fallback.sum(axis=1))])
    q = np.column_stack([joined[f"q_{group}"].to_numpy(float) for group in PITCH_GROUPS])
    missing = ~np.isfinite(q).all(axis=1)
    q[missing] = fallback[missing]
    q = np.clip(q, 1e-6, None)
    q /= q.sum(axis=1, keepdims=True)
    reliability = joined["latent_reliability"].fillna(0.0).to_numpy(float)
    return q, reliability


def apply_latent_moe_119b(raw: pd.DataFrame, prediction: np.ndarray, model_dir: Path) -> np.ndarray:
    meta = json.loads((model_dir / "latent_moe_119b_meta.json").read_text(encoding="utf-8"))
    if not meta.get("enabled", False):
        return prediction.copy()
    bundle = np.load(model_dir / "latent_moe_119b_models.npz", allow_pickle=False)
    x, columns = _moe_context(raw.reset_index(drop=True))
    if columns != bundle["columns"].tolist():
        raise RuntimeError("119B feature contract mismatch")
    q, reliability = _latent_probabilities(raw.reset_index(drop=True), model_dir / "latent_pitch_119b.csv")
    raw_delta = x @ bundle["coef"].T
    cap = float(meta["delta_cap"])
    scale = float(meta["correction_scale"])
    expert_predictions = np.clip(prediction[:, None] + np.clip(scale * raw_delta, -cap, cap), 0.02, 0.98)
    mixed = np.sum(q * expert_predictions, axis=1)
    gate = np.clip(0.25 + 0.75 * reliability, 0.25, 1.0)
    return np.clip(prediction + gate * (mixed - prediction), 0.02, 0.98)


def apply_quantile_drift_119c(raw: pd.DataFrame, prediction: np.ndarray, model_dir: Path) -> np.ndarray:
    meta = json.loads((model_dir / "quantile_drift_119c_meta.json").read_text(encoding="utf-8"))
    if not meta.get("enabled", False):
        return prediction.copy()
    table = pd.read_csv(model_dir / "trackman_quantile_119c.csv", dtype={"pitcher_id": str})
    bundle = np.load(model_dir / "quantile_drift_119c_model.npz", allow_pickle=False)
    columns = bundle["columns"].tolist()
    features = pd.DataFrame({"pitcher_id": raw["pitcher_id"].astype(str)}).merge(
        table, on="pitcher_id", how="left", sort=False
    )
    x = features.reindex(columns=columns).astype(float)
    values = x.to_numpy(float)
    mean = bundle["mean"]
    std = bundle["std"]
    values = np.where(np.isfinite(values), values, mean)
    z = (values - mean) / std
    delta = z @ bundle["coef"]
    delta = np.clip(float(meta["correction_scale"]) * delta, -float(meta["delta_cap"]), float(meta["delta_cap"]))
    reliability = features["tm_reliability"].fillna(0.0).clip(0, 1).to_numpy(float)
    return np.clip(prediction + reliability * delta, 0.02, 0.98)
