#!/usr/bin/env python3
"""Strict-forward evaluation of reference-10 v64 dynamic pitcher state."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "model/REF4-113A-DYNAMIC-STATE-117B"
BASELINE = ROOT / "model/REF4-113A-V66-NESTED-117A/oof_predictions.csv"
TRAIN = ROOT / "data/train.csv"
TARGET = "control_success"
STATE_SMOOTHING = 200.0
TRANSITION_RIDGE = 1.0
CURRENT_STRENGTH = 100.0
WEIGHT = 0.25
CLIP = (0.005, 0.995)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def logit(value: np.ndarray | float) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=float), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def expit(value: np.ndarray | float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(value, dtype=float), -30.0, 30.0)))


@dataclass(frozen=True)
class CareerState:
    n: pd.Series
    successes: pd.Series


def season_states(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, float]]:
    stats = frame.groupby(["season", "pitcher_id"], sort=False)[TARGET].agg(["sum", "count"])
    league = frame.groupby("season", sort=False)[TARGET].mean()
    league_rates = {int(year): float(rate) for year, rate in league.items()}
    years = stats.index.get_level_values("season")
    priors = np.asarray([league_rates[int(year)] for year in years], dtype=float)
    counts = stats["count"].to_numpy(float)
    posterior = (stats["sum"].to_numpy(float) + STATE_SMOOTHING * priors) / (
        counts + STATE_SMOOTHING
    )
    states = stats.copy()
    states["latent"] = logit(posterior) - logit(priors)
    states["reliability"] = counts / (counts + STATE_SMOOTHING)
    return states, league_rates


def fit_ar1(states: pd.DataFrame, prediction_year: int) -> tuple[float, int]:
    history = states.loc[states.index.get_level_values("season") < prediction_year].reset_index()
    previous = history.rename(
        columns={"season": "previous_year", "latent": "previous_latent", "reliability": "previous_reliability"}
    )[["pitcher_id", "previous_year", "previous_latent", "previous_reliability"]]
    current = history.rename(
        columns={"season": "current_year", "latent": "current_latent", "reliability": "current_reliability"}
    )[["pitcher_id", "current_year", "current_latent", "current_reliability"]]
    pairs = current.merge(previous, on="pitcher_id", how="inner")
    pairs = pairs.loc[pairs["current_year"].eq(pairs["previous_year"] + 1)]
    weight = np.sqrt(
        pairs["current_reliability"].to_numpy(float)
        * pairs["previous_reliability"].to_numpy(float)
    )
    x = pairs["previous_latent"].to_numpy(float)
    y = pairs["current_latent"].to_numpy(float)
    rho = np.sum(weight * x * y) / (np.sum(weight * x * x) + TRANSITION_RIDGE)
    return float(np.clip(rho, 0.0, 1.0)), int(len(pairs))


def career_before(frame: pd.DataFrame, prediction_year: int) -> CareerState:
    history = frame.loc[frame["season"].lt(prediction_year)]
    index = history.groupby("pitcher_id", sort=False)["asof_pitcher_n"].idxmax()
    last = history.loc[index]
    before_n = last["asof_pitcher_n"].to_numpy(float)
    before_success = np.rint(before_n * last["asof_pitcher_success_rate"].to_numpy(float))
    ids = last["pitcher_id"].to_numpy()
    return CareerState(
        n=pd.Series(before_n + 1.0, index=ids),
        successes=pd.Series(before_success + last[TARGET].to_numpy(float), index=ids),
    )


def dynamic_delta(
    frame: pd.DataFrame,
    rows: pd.DataFrame,
    states: pd.DataFrame,
    league_rates: dict[int, float],
    prediction_year: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    rho, pair_count = fit_ar1(states, prediction_year)
    history = states.loc[states.index.get_level_values("season") < prediction_year].reset_index()
    last_index = history.groupby("pitcher_id", sort=False)["season"].idxmax()
    latest = history.loc[last_index, ["pitcher_id", "season", "latent"]].set_index("pitcher_id")
    career = career_before(frame, prediction_year)
    ids = rows["pitcher_id"]
    prior_n = ids.map(career.n).fillna(0.0).to_numpy(float)
    prior_success = ids.map(career.successes).fillna(0.0).to_numpy(float)
    career_n = rows["asof_pitcher_n"].to_numpy(float)
    career_success = np.rint(career_n * rows["asof_pitcher_success_rate"].to_numpy(float))
    current_n = np.maximum(career_n - prior_n, 0.0)
    current_success = np.clip(career_success - prior_success, 0.0, current_n)
    last_latent = ids.map(latest["latent"]).fillna(0.0).to_numpy(float)
    last_year = ids.map(latest["season"]).to_numpy(float)
    known = np.isfinite(last_year)
    gap = np.where(known, prediction_year - last_year, 0.0)
    ar_latent = np.where(known, last_latent * np.power(rho, gap), 0.0)
    league_prior = float(league_rates[prediction_year - 1])
    state_prior = expit(logit(league_prior) + ar_latent)
    dynamic = (current_success + CURRENT_STRENGTH * state_prior) / (current_n + CURRENT_STRENGTH)
    neutral = (current_success + CURRENT_STRENGTH * league_prior) / (current_n + CURRENT_STRENGTH)
    regular = rows["game_type"].astype(str).eq("R").to_numpy()
    delta = WEIGHT * (dynamic - neutral) * regular.astype(float)
    return delta, {
        "rho": rho,
        "transition_pairs": pair_count,
        "known_row_fraction": float(known.mean()),
        "current_n_mean": float(current_n.mean()),
        "correction_mean": float(delta.mean()),
        "correction_std": float(delta.std()),
    }


def bootstrap(target: np.ndarray, base: np.ndarray, candidate: np.ndarray, pitcher: np.ndarray) -> dict[str, float | int]:
    row_delta = np.square(candidate - target) - np.square(base - target)
    grouped = pd.DataFrame({"pitcher": pitcher.astype(str), "delta": row_delta}).groupby("pitcher")["delta"].agg(["sum", "size"])
    sums, sizes = grouped["sum"].to_numpy(float), grouped["size"].to_numpy(float)
    rng = np.random.default_rng(1172025)
    values = np.empty(10000, dtype=float)
    for start in range(0, 10000, 64):
        count = min(64, 10000 - start)
        sample = rng.integers(0, len(grouped), size=(count, len(grouped)))
        values[start:start + count] = sums[sample].sum(axis=1) / sizes[sample].sum(axis=1)
    return {"repeats": 10000, "pitcher_clusters": int(len(grouped)), "mean_delta": float(values.mean()), "ci_low": float(np.quantile(values, 0.025)), "ci_high": float(np.quantile(values, 0.975))}


def main() -> None:
    contract = json.loads((EXP / "audit_contract.json").read_text())
    preflight = json.loads((EXP / "preflight_report.json").read_text())
    if contract["status"] != "LOCKED_BEFORE_RESULTS" or preflight["status"] != "AUDIT_VERIFIED":
        raise RuntimeError("117B preflight is not locked and verified")
    if sha256(BASELINE) != preflight["checks"]["baseline_oof_sha256"]:
        raise RuntimeError("strict 113A baseline hash mismatch")
    raw = pd.read_csv(
        TRAIN,
        usecols=["row_id", "season", "game_type", "pitcher_id", "asof_pitcher_n", "asof_pitcher_success_rate", TARGET],
        low_memory=False,
    )
    missing = raw["asof_pitcher_success_rate"].isna()
    if not raw.loc[missing, "asof_pitcher_n"].eq(0).all():
        raise RuntimeError("unexpected positive-count missing pitcher rate")
    raw["asof_pitcher_success_rate"] = raw["asof_pitcher_success_rate"].fillna(0.0)
    baseline = pd.read_csv(BASELINE, usecols=["row_id", "season", "game_type", "pitcher_id", "target", "p113a_strict"], low_memory=False)
    rows = raw.loc[raw["season"].isin((2022, 2023, 2024))].reset_index(drop=True)
    if not np.array_equal(rows["row_id"].astype(str), baseline["row_id"].astype(str)):
        raise RuntimeError("strict baseline row mismatch")
    states, league_rates = season_states(raw)
    parts = []
    folds: dict[str, object] = {}
    for year in (2022, 2023, 2024):
        local_rows = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        local = baseline.loc[baseline["season"].eq(year)].reset_index(drop=True)
        delta, state_audit = dynamic_delta(raw, local_rows, states, league_rates, year)
        base = local["p113a_strict"].to_numpy(float)
        prediction = np.clip(base + delta, *CLIP)
        target = local["target"].to_numpy(float)
        base_brier = float(np.mean(np.square(base - target)))
        candidate_brier = float(np.mean(np.square(prediction - target)))
        folds[str(year)] = {
            "train_seasons": list(range(2019, year)),
            "valid_rows": int(len(local)),
            "validation_labels_used_in_fit": False,
            "state": state_audit,
            "p113a_brier": base_brier,
            "p117b_brier": candidate_brier,
            "delta_brier": candidate_brier - base_brier,
            "mean_absolute_change": float(np.mean(np.abs(prediction - base))),
            "correction_range": [float(delta.min()), float(delta.max())],
        }
        part = local.copy()
        part["dynamic_state_delta"] = delta
        part["p117b"] = prediction
        parts.append(part)
    output = pd.concat(parts, ignore_index=True)
    deltas = {year: float(folds[str(year)]["delta_brier"]) for year in (2022, 2023, 2024)}
    weighted = 0.2 * deltas[2022] + 0.3 * deltas[2023] + 0.5 * deltas[2024]
    active = output["season"].eq(2024)
    boot = bootstrap(output.loc[active, "target"].to_numpy(float), output.loc[active, "p113a_strict"].to_numpy(float), output.loc[active, "p117b"].to_numpy(float), output.loc[active, "pitcher_id"].to_numpy())
    gates = {
        "delta_2024": deltas[2024] <= -0.0001,
        "delta_2022": deltas[2022] <= 0.00005,
        "time_weighted": weighted < 0.0,
        "worst_season": max(deltas.values()) <= 0.00005,
        "bootstrap_2024_ci_high_below_zero": boot["ci_high"] < 0.0,
    }
    path = EXP / "oof_predictions.csv"
    output.to_csv(path, index=False)
    passed = bool(all(gates.values()))
    report = {
        "experiment_id": "REF4-113A-DYNAMIC-STATE-117B",
        "status": "PENDING_AUDIT",
        "candidate_status": "PERFORMANCE_GATE_PASS_PENDING_AUDIT" if passed else "REJECTED_PERFORMANCE_GATE_PENDING_AUDIT",
        "hypothesis_count": 1,
        "oof_rows": int(len(output)),
        "folds": folds,
        "time_weighted_delta": weighted,
        "worst_season_delta": max(deltas.values()),
        "pitcher_cluster_bootstrap_2024": boot,
        "gate_results": gates,
        "performance_gate_pass": passed,
        "test_read": False,
        "zip_created": False,
        "oof_predictions_sha256": sha256(path),
    }
    (EXP / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
