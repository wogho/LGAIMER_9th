#!/usr/bin/env python3
"""Quantify why REF-AUX-CAT-002 diverged from the public 998-point recipe."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "github_reference" / "1번 레포" / "test"
OUT = ROOT / "model" / "REF-GAP-001"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def main():
    train = pd.read_csv(ROOT / "data" / "train.csv", usecols=["season", "control_success"])
    rates = train.groupby("season")["control_success"].mean()
    rows = []
    for start in (2019, 2020, 2021, 2022, 2023):
        years = rates.loc[start:2023].index.to_numpy(dtype=float)
        values = rates.loc[start:2023].to_numpy(dtype=float)
        forecast = float(np.polyval(np.polyfit(years, values, 1), 2025))
        rows.append({"fit_start": start, "fit_end": 2023, "forecast_2025": forecast})

    base_path = ROOT / "model" / "ENS-CATF-LGBMCATR5050-FE001" / "selective_predictions_2024.csv"
    pred = pd.read_csv(base_path)["pred_selective"].to_numpy(dtype=np.float64)
    y = train.loc[train["season"].eq(2024), "control_success"].to_numpy(dtype=np.float64)
    base_brier = float(np.mean((pred - y) ** 2))
    shift_screens = []
    for row in rows:
        target = row["forecast_2025"]
        shift = float(logit(target) - logit(pred.mean()))
        adjusted = 1.0 / (1.0 + np.exp(-(logit(pred) + shift)))
        shift_screens.append({**row, "shift": shift, "delta_brier_on_2024": float(np.mean((adjusted - y) ** 2) - base_brier), "mean_after": float(adjusted.mean())})

    ref_meta = json.loads((REF / "runs" / "012_shift_full" / "model" / "meta.json").read_text(encoding="utf-8"))
    ref_result = json.loads((REF / "runs" / "012_shift_full" / "result.json").read_text(encoding="utf-8"))
    report = {
        "experiment_id": "REF-GAP-001",
        "official_score_observed": 869.5916957228,
        "previous_official_score": 886.2488171351,
        "score_delta": -16.6571214123,
        "reference_claim": ref_result.get("lb_2025"),
        "reference_claim_is_self_report": True,
        "reference_components": {
            "success_seeds": len(ref_meta["seeds"]),
            "aux_seeds": len(ref_meta["offset"]["seeds"]),
            "feature_count": len(ref_meta["feature_cols"]),
            "offset_b": ref_meta["offset"]["b"],
            "offset_c": ref_meta["offset"]["c"],
            "logit_shift": ref_meta["logit_shift"],
            "target_rate": 0.477,
        },
        "ours_candidate_gap": {
            "success_models": 1,
            "aux_models": 2,
            "offset_fit": "2022/2023 forward diagnostic, not reference 2024 OOF ensemble",
            "global_shift": "omitted",
        },
        "official_train_season_rates": {str(k): float(v) for k, v in rates.items()},
        "official_only_shift_screens": shift_screens,
        "rules": {
            "external_data_allowed": False,
            "reference_kbo_shift_adoptable": False,
            "test_row_aggregation": False,
        },
        "source_hashes": {
            "train": sha256(ROOT / "data" / "train.csv"),
            "reference_meta": sha256(REF / "runs" / "012_shift_full" / "model" / "meta.json"),
            "reference_result": sha256(REF / "runs" / "012_shift_full" / "result.json"),
            "ours_pred_2024": sha256(base_path),
        },
        "status": "DIAGNOSIS_COMPLETE",
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
