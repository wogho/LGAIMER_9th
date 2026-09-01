#!/usr/bin/env python3
"""Independent raw-data audit for REF4-113A-V66-NESTED-117A."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "model/REF4-113A-V66-NESTED-117A"
TARGET = "control_success"
AXES = (
    ("pitcher_id", "pitcher_hand_key", 300.0, 0.12),
    ("pitcher_hand_key", "pitcher_hand_advantage_key", 2000.0, 0.495),
    ("pitcher_hand_key", "runner_key", 2000.0, 0.27),
)


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def contexts(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["pitcher_hand_key"] = out["pitcher_id"].astype(str).str.cat(
        out["batter_hand"].astype(str), sep="|"
    )
    advantage = out["strikes_before"].gt(out["balls_before"]).astype(np.int8)
    runner = out["num_runners_on"].gt(0).astype(np.int8)
    out["pitcher_hand_advantage_key"] = out["pitcher_hand_key"].str.cat(
        advantage.astype(str), sep="|"
    )
    out["runner_key"] = out["pitcher_hand_key"].str.cat(
        runner.astype(str), sep="|"
    )
    return out


def correction(history: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    result = np.zeros(len(rows), dtype=float)
    for parent, child, shrinkage, weight in AXES:
        parent_rate = history.groupby(parent, sort=False)[TARGET].mean()
        grouped = history.groupby(child, sort=False).agg(
            total=(TARGET, "sum"), count=(TARGET, "size"), parent=(parent, "first")
        )
        delta = (
            (grouped["total"] / grouped["count"] - grouped["parent"].map(parent_rate))
            * grouped["count"]
            / (grouped["count"] + shrinkage)
        )
        result += weight * rows[child].map(delta).fillna(0.0).to_numpy(float)
    return result


def main() -> None:
    result = json.loads((EXP / "result.json").read_text())
    prediction_path = EXP / "oof_predictions.csv"
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, actual: object) -> None:
        checks.append({"name": name, "passed": bool(passed), "actual": actual})

    check(
        "oof_sha256",
        digest(prediction_path) == result["oof_predictions_sha256"],
        digest(prediction_path),
    )
    prediction = pd.read_csv(prediction_path, low_memory=False)
    raw = contexts(
        pd.read_csv(
            ROOT / "data/train.csv",
            usecols=[
                "row_id", "season", "pitcher_id", "batter_hand",
                "balls_before", "strikes_before", "num_runners_on", TARGET,
            ],
            low_memory=False,
        )
    )
    rows = raw.loc[raw["season"].isin((2022, 2023, 2024))].reset_index(drop=True)
    check("row_count", len(prediction) == 746504, len(prediction))
    check("row_id_unique", prediction["row_id"].nunique() == len(prediction), int(prediction["row_id"].nunique()))
    check("row_order", np.array_equal(prediction["row_id"].astype(str), rows["row_id"].astype(str)), None)
    check("target", np.array_equal(prediction["target"].to_numpy(float), rows[TARGET].to_numpy(float)), None)
    check("finite", np.isfinite(prediction[["p113a_strict", "v66_delta", "p117a"]]).all().all(), None)
    check("probability_range", prediction[["p113a_strict", "p117a"]].min().min() >= 0.0 and prediction[["p113a_strict", "p117a"]].max().max() <= 1.0, None)

    deltas: dict[int, float] = {}
    for year in (2022, 2023, 2024):
        historical = raw.loc[raw["season"].lt(year)]
        valid = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        expected_delta = correction(historical, valid)
        local = prediction.loc[prediction["season"].eq(year)].reset_index(drop=True)
        max_error = float(np.max(np.abs(expected_delta - local["v66_delta"].to_numpy(float))))
        check(f"v66_formula_{year}", max_error <= 1e-12, max_error)
        expected_candidate = np.clip(local["p113a_strict"].to_numpy(float) + expected_delta, 0.005, 0.995)
        candidate_error = float(np.max(np.abs(expected_candidate - local["p117a"].to_numpy(float))))
        check(f"candidate_formula_{year}", candidate_error <= 1e-12, candidate_error)
        y = local["target"].to_numpy(float)
        delta_brier = float(np.mean((expected_candidate - y) ** 2) - np.mean((local["p113a_strict"].to_numpy(float) - y) ** 2))
        deltas[year] = delta_brier
        metric_error = abs(delta_brier - float(result["folds"][str(year)]["delta_brier"]))
        check(f"metric_{year}", metric_error <= 1e-15, metric_error)

    gates = {
        "delta_2024": deltas[2024] <= -0.0001,
        "delta_2022": deltas[2022] <= 0.00005,
        "time_weighted": 0.2 * deltas[2022] + 0.3 * deltas[2023] + 0.5 * deltas[2024] < 0.0,
        "worst_season": max(deltas.values()) <= 0.00005,
        "bootstrap_2024_ci_high_below_zero": float(result["pitcher_cluster_bootstrap_2024"]["ci_high"]) < 0.0,
    }
    check("gate_reproduction", gates == result["gate_results"], gates)
    failures = [entry for entry in checks if not entry["passed"]]
    report = {
        "experiment_id": "REF4-113A-V66-NESTED-117A",
        "status": "AUDIT_VERIFIED" if not failures else "AUDIT_FAILED",
        "diagnostic_outcome": "REJECTED_PERFORMANCE_GATE",
        "checked_count": len(checks),
        "passed_count": len(checks) - len(failures),
        "mismatch_count": len(failures),
        "checks": checks,
        "performance_gate_pass": False,
        "test_read": False,
        "zip_created": False,
    }
    (EXP / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
