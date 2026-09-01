#!/usr/bin/env python3
"""Strict evaluation of the exact reference-10 v59 batter exposure axis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "model/REF4-113A-BATTER-EXPOSURE-117D"
BASELINE = ROOT / "model/REF4-113A-V66-NESTED-117A/oof_predictions.csv"
TRAIN = ROOT / "data/train.csv"
SLOPE = 2.0907659421884613e-6
CLIP = (0.005, 0.995)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def exposure_delta(raw: pd.DataFrame, rows: pd.DataFrame, year: int) -> tuple[np.ndarray, dict[str, object]]:
    source = raw.loc[raw["season"].between(year - 2, year - 1)]
    counts = source.groupby("batter_id", sort=True).size().astype(float)
    center = float(counts.mean())
    table = SLOPE * (counts - center)
    values = rows["batter_id"].map(table)
    return values.fillna(0.0).to_numpy(float), {
        "source_seasons": [year - 2, year - 1],
        "known_batters": int(len(table)),
        "entity_mean_count": center,
        "unknown_row_fraction": float(values.isna().mean()),
        "entity_delta_range": [float(table.min()), float(table.max())],
    }


def bootstrap(target: np.ndarray, base: np.ndarray, candidate: np.ndarray, pitcher: np.ndarray) -> dict[str, float | int]:
    row_delta = np.square(candidate - target) - np.square(base - target)
    grouped = pd.DataFrame({"pitcher": pitcher.astype(str), "delta": row_delta}).groupby("pitcher")["delta"].agg(["sum", "size"])
    sums, sizes = grouped["sum"].to_numpy(float), grouped["size"].to_numpy(float)
    rng = np.random.default_rng(1172027)
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
        raise RuntimeError("117D preflight is not locked and verified")
    if sha256(BASELINE) != preflight["checks"]["baseline_oof_sha256"]:
        raise RuntimeError("strict 113A baseline hash mismatch")
    raw = pd.read_csv(TRAIN, usecols=["row_id", "season", "game_type", "pitcher_id", "batter_id", "control_success"], low_memory=False)
    baseline = pd.read_csv(BASELINE, usecols=["row_id", "season", "game_type", "pitcher_id", "target", "p113a_strict"], low_memory=False)
    rows = raw.loc[raw["season"].isin((2022, 2023, 2024))].reset_index(drop=True)
    if not np.array_equal(rows["row_id"].astype(str), baseline["row_id"].astype(str)):
        raise RuntimeError("strict baseline row mismatch")
    parts = []
    folds: dict[str, object] = {}
    for year in (2022, 2023, 2024):
        local_rows = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        local = baseline.loc[baseline["season"].eq(year)].reset_index(drop=True)
        delta, table_audit = exposure_delta(raw, local_rows, year)
        base = local["p113a_strict"].to_numpy(float)
        prediction = np.clip(base + delta, *CLIP)
        target = local["target"].to_numpy(float)
        base_brier = float(np.mean(np.square(base - target)))
        candidate_brier = float(np.mean(np.square(prediction - target)))
        folds[str(year)] = {
            "valid_rows": int(len(local)),
            "labels_used_to_build_direction": False,
            "validation_labels_used_in_fit": False,
            "table": table_audit,
            "p113a_brier": base_brier,
            "p117d_brier": candidate_brier,
            "delta_brier": candidate_brier - base_brier,
            "mean_absolute_change": float(np.mean(np.abs(prediction - base))),
            "correction_range": [float(delta.min()), float(delta.max())],
        }
        part = local.copy()
        part["batter_exposure_delta"] = delta
        part["p117d"] = prediction
        parts.append(part)
    output = pd.concat(parts, ignore_index=True)
    deltas = {year: float(folds[str(year)]["delta_brier"]) for year in (2022, 2023, 2024)}
    weighted = 0.2 * deltas[2022] + 0.3 * deltas[2023] + 0.5 * deltas[2024]
    active = output["season"].eq(2024)
    boot = bootstrap(output.loc[active, "target"].to_numpy(float), output.loc[active, "p113a_strict"].to_numpy(float), output.loc[active, "p117d"].to_numpy(float), output.loc[active, "pitcher_id"].to_numpy())
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
        "experiment_id": "REF4-113A-BATTER-EXPOSURE-117D",
        "status": "PENDING_AUDIT",
        "candidate_status": "PERFORMANCE_GATE_PASS_PENDING_AUDIT" if passed else "REJECTED_PERFORMANCE_GATE_PENDING_AUDIT",
        "hypothesis_count": 1,
        "model_count": 0,
        "oof_rows": int(len(output)),
        "folds": folds,
        "time_weighted_delta": weighted,
        "worst_season_delta": max(deltas.values()),
        "pitcher_cluster_bootstrap_2024": boot,
        "gate_results": gates,
        "performance_gate_pass": passed,
        "test_read": False,
        "production_table_created": False,
        "zip_created": False,
        "oof_predictions_sha256": sha256(path),
    }
    (EXP / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
