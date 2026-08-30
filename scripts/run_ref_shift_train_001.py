#!/usr/bin/env python3
"""REF-SHIFT-TRAIN-001: fixed official-train-only season-rate shift screen."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "model" / "REF-SHIFT-TRAIN-001"
TRAIN = ROOT / "data" / "train.csv"
PRED = ROOT / "model" / "REF-AUX-OFFSET-CAT-001" / "transition_predictions"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def forecast(rates: pd.Series, through: int, target: int) -> float:
    use = rates.loc[rates.index <= through]
    x = use.index.to_numpy(dtype=np.float64)
    y = use.to_numpy(dtype=np.float64)
    return float(np.polyval(np.polyfit(x, y, 1), target))


def main() -> None:
    frame = pd.read_csv(TRAIN, usecols=["season", "control_success"])
    rates = frame.groupby("season")["control_success"].mean()
    transitions = []
    for source, target in ((2022, 2023), (2023, 2024)):
        data = np.load(PRED / f"{source}_{target}.npz")
        y = data["y"].astype(np.float64)
        baseline = data["baseline"].astype(np.float64)
        offset = data["offset"].astype(np.float64)
        target_rate = forecast(rates, source, target)
        # Mean is from the saved train-only pseudo-future artifact, never test.csv.
        shift = float(logit(target_rate) - logit(offset.mean()))
        adjusted = 1.0 / (1.0 + np.exp(-(logit(offset) + shift)))
        transitions.append({
            "fit_through": source,
            "apply_season": target,
            "forecast_rate": target_rate,
            "shift": shift,
            "baseline_brier": float(np.mean((baseline - y) ** 2)),
            "offset_brier": float(np.mean((offset - y) ** 2)),
            "shifted_brier": float(np.mean((adjusted - y) ** 2)),
            "delta_shift_vs_offset": float(np.mean((adjusted - y) ** 2) - np.mean((offset - y) ** 2)),
            "rows": int(len(y)),
        })
    # Deployment metadata is a deterministic train-only extrapolation through 2024 -> 2025.
    deployment_rate = forecast(rates, 2024, 2025)
    deployment_reference_mean = float(np.load(PRED / "2023_2024.npz")["offset"].mean())
    deployment_shift = float(logit(deployment_rate) - logit(deployment_reference_mean))
    report = {
        "experiment_id": "REF-SHIFT-TRAIN-001",
        "method": "linear_fit_of_official_train_season_rates_only",
        "candidate_count": 1,
        "external_data_used": False,
        "test_data_used": False,
        "official_season_rates": {str(k): float(v) for k, v in rates.items()},
        "deployment_forecast_2025": deployment_rate,
        "deployment_reference_mean_train_only": deployment_reference_mean,
        "deployment_shift": deployment_shift,
        "transitions": transitions,
        "status": "PASS" if all(t["delta_shift_vs_offset"] < 0 for t in transitions) else "FAIL_FORWARD_SIGN",
        "submission_status": "HOLD",
        "source_hashes": {"train": sha256(TRAIN), "transition_2022_2023": sha256(PRED / "2022_2023.npz"), "transition_2023_2024": sha256(PRED / "2023_2024.npz")},
    }
    EXP.mkdir(parents=True, exist_ok=True)
    path = EXP / "validation_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    att = {"attestation_id": "REF-SHIFT-TRAIN-001-ATTESTATION", "report_sha256": sha256(path), "validator_sha256": sha256(Path(__file__)), "status": report["status"], "submission_status": "HOLD"}
    (EXP / "attestation.json").write_text(json.dumps(att, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
