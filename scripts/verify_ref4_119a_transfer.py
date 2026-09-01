#!/usr/bin/env python3
"""Exact season-to-next-season transfer audit for the frozen 119A formula."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model/REF4-119-RESEARCH/119a_transfer_report.json"
KEYS = ["game_type", "balls_before", "strikes_before", "hand_match"]


def clustered_ci(y, baseline, candidate, pitcher, seed, repeats=2000):
    gain = np.square(baseline - y) - np.square(candidate - y)
    codes, unique = pd.factorize(pd.Series(pitcher).astype(str), sort=True)
    sums = np.bincount(codes, weights=gain, minlength=len(unique))
    sizes = np.bincount(codes, minlength=len(unique))
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats)
    for pos in range(repeats):
        sample = rng.integers(0, len(unique), len(unique))
        draws[pos] = sums[sample].sum() / sizes[sample].sum()
    return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]


def main():
    anchor = pd.read_csv(
        ROOT / "model/REF4-113A-V66-NESTED-117A/oof_predictions.csv",
        usecols=["row_id", "season", "target", "p113a_strict"],
    )
    raw = pd.read_csv(
        ROOT / "data/train.csv",
        usecols=["row_id", "season", "game_type", "balls_before", "strikes_before", "pitcher_hand", "batter_hand", "pitcher_id", "control_success"],
    )
    data = raw.loc[raw["season"].isin([2022, 2023, 2024])].merge(anchor, on=["row_id", "season"], validate="one_to_one")
    if not np.array_equal(data["control_success"].to_numpy(float), data["target"].to_numpy(float)):
        raise RuntimeError("target mismatch")
    data["hand_match"] = (data["pitcher_hand"].astype(str) == data["batter_hand"].astype(str)).astype(int)
    results = []
    for fit_year, valid_year in ((2022, 2023), (2023, 2024)):
        fit = data.loc[data["season"].eq(fit_year)].copy()
        valid = data.loc[data["season"].eq(valid_year)].copy()
        fit["residual"] = fit["target"] - fit["p113a_strict"]
        table = fit.groupby(KEYS, observed=True)["residual"].agg(["mean", "size"]).reset_index()
        table["correction"] = (0.25 * table["mean"] * table["size"] / (table["size"] + 1000.0)).clip(-0.012, 0.012)
        joined = valid.merge(table[KEYS + ["correction"]], on=KEYS, how="left", sort=False)
        correction = joined["correction"].fillna(0).to_numpy(float)
        y = joined["target"].to_numpy(float)
        baseline = joined["p113a_strict"].to_numpy(float)
        candidate = np.clip(baseline + correction, 0.02, 0.98)
        gain = float(np.mean(np.square(baseline - y) - np.square(candidate - y)))
        results.append(
            {
                "fit_season": fit_year,
                "validation_season": valid_year,
                "rows": int(len(valid)),
                "brier_gain": gain,
                "mean_absolute_change": float(np.mean(np.abs(candidate - baseline))),
                "max_absolute_change": float(np.max(np.abs(candidate - baseline))),
                "pitcher_cluster_ci": clustered_ci(y, baseline, candidate, joined["pitcher_id"], 119300 + valid_year),
            }
        )
    passed = all(item["brier_gain"] > 0 and item["max_absolute_change"] <= 0.012 + 1e-12 for item in results)
    output = {"candidate": "ABS-ERA-RESIDUAL-119A", "status": "AUDIT_VERIFIED", "formula_frozen": True, "results": results, "season_transfer_pass": passed}
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
