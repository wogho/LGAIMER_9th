#!/usr/bin/env python3
"""Strict-forward training and comparison for REF4 candidates 119A/B/C.

The official 113A leaderboard result is not treated as a validation metric.
The evaluation anchor is the audited 746,504-row strict reconstruction stored
by experiment 117A.  No test file is read by this script.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data/train.csv"
TM_PATH = ROOT / "data/trackman_history.csv"
ANCHOR_PATH = ROOT / "model/REF4-113A-V66-NESTED-117A/oof_predictions.csv"
OUT = ROOT / "model/REF4-119-RESEARCH"
PRIOR = 0.523766
YEARS = (2022, 2023, 2024)
PITCH_GROUPS = ("fastball", "breaking", "offspeed", "other")
METRICS = ("rel_speed", "spin_rate", "induced_vert_break", "horz_break", "extension", "rel_height", "rel_side", "zone_speed")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.square(y - p)))


def pitcher_bootstrap(y, baseline, candidate, pitcher, seed, repeats=2000):
    gain = np.square(baseline - y) - np.square(candidate - y)
    codes, unique = pd.factorize(pd.Series(pitcher).astype(str), sort=True)
    sums = np.bincount(codes, weights=gain, minlength=len(unique))
    sizes = np.bincount(codes, minlength=len(unique))
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats, dtype=float)
    for start in range(0, repeats, 64):
        count = min(64, repeats - start)
        sample = rng.integers(0, len(unique), size=(count, len(unique)))
        draws[start : start + count] = sums[sample].sum(1) / sizes[sample].sum(1)
    return {
        "clusters": int(len(unique)),
        "repeats": repeats,
        "mean_gain": float(draws.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def hand_text(values: pd.Series) -> pd.Series:
    return values.map({1: "Left", 2: "Right", "1": "Left", "2": "Right"}).fillna(values.astype(str))


def build_mapping(raw: pd.DataFrame, tm_keys: pd.DataFrame, target_year: int) -> pd.DataFrame:
    train = raw.loc[raw["season"].lt(target_year), ["pitcher_id", "season", "game_month", "game_dayofweek", "pitcher_hand"]].copy()
    track = tm_keys.loc[tm_keys["season"].lt(target_year)].copy()
    train["key"] = train["season"].astype(str) + "_" + train["game_month"].astype(str) + "_" + train["game_dayofweek"].astype(str)
    track["key"] = track["season"].astype(str) + "_" + track["game_month"].astype(str) + "_" + track["game_dayofweek"].astype(str)
    a = train.groupby(["pitcher_id", "key"]).size().unstack(fill_value=0)
    b = track.groupby(["pitcher_trackman_id", "key"]).size().unstack(fill_value=0)
    columns = a.columns.union(b.columns)
    a = a.reindex(columns=columns, fill_value=0)
    b = b.reindex(columns=columns, fill_value=0)
    similarity = (normalize(csr_matrix(a.values)) @ normalize(csr_matrix(b.values)).T).toarray()
    train_hand = train.groupby("pitcher_id")["pitcher_hand"].first().reindex(a.index).to_numpy()
    tm_hand = track.groupby("pitcher_trackman_id")["pitcher_hand"].first().reindex(b.index).map({"Left": 1, "Right": 2}).to_numpy()
    similarity[train_hand[:, None] != tm_hand[None, :]] = -1
    best = similarity.argmax(1)
    reverse = similarity.argmax(0)
    ordered = np.sort(similarity, axis=1)
    confidence = similarity[np.arange(len(a)), best]
    margin = ordered[:, -1] - ordered[:, -2]
    accepted = (reverse[best] == np.arange(len(a))) & (confidence >= 0.90) & (margin >= 0.03)
    return pd.DataFrame(
        {
            "pitcher_id": a.index[accepted].astype(str),
            "pitcher_trackman_id": b.index[best[accepted]],
            "mapping_similarity": confidence[accepted],
            "mapping_margin": margin[accepted],
        }
    )


def latent_table(tm: pd.DataFrame, mapping: pd.DataFrame, target_year: int) -> pd.DataFrame:
    history = tm.loc[tm["season"].lt(target_year), ["pitcher_trackman_id", "balls_before", "strikes_before", "batter_hand", "pitch_type_group"]].merge(
        mapping[["pitcher_id", "pitcher_trackman_id", "mapping_similarity"]], on="pitcher_trackman_id", how="inner", validate="many_to_one"
    )
    history["batter_hand"] = hand_text(history["batter_hand"])
    overall = history.groupby(["pitcher_id", "pitch_type_group"]).size().unstack(fill_value=0).reindex(columns=PITCH_GROUPS, fill_value=0)
    parent = (overall + 50.0 * np.array([0.50, 0.30, 0.15, 0.05])) / (overall.sum(1).to_numpy()[:, None] + 50.0)
    keys = ["pitcher_id", "balls_before", "strikes_before", "batter_hand"]
    context = history.groupby(keys + ["pitch_type_group"]).size().unstack(fill_value=0).reindex(columns=PITCH_GROUPS, fill_value=0)
    index = context.index
    n = context.sum(1).to_numpy(float)
    parent_values = parent.reindex(index.get_level_values("pitcher_id")).to_numpy(float)
    probabilities = (context.to_numpy(float) + 100.0 * parent_values) / (n[:, None] + 100.0)
    result = index.to_frame(index=False)
    result["latent_n"] = n
    result["mapping_similarity"] = mapping.set_index("pitcher_id")["mapping_similarity"].reindex(result["pitcher_id"]).to_numpy(float)
    result["latent_reliability"] = (n / (n + 100.0)) * result["mapping_similarity"].fillna(0.0)
    for pos, group in enumerate(PITCH_GROUPS):
        result[f"q_{group}"] = probabilities[:, pos]
    return result


def latent_probabilities(rows: pd.DataFrame, table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    keys = pd.DataFrame(
        {
            "pitcher_id": rows["pitcher_id"].astype(str),
            "balls_before": rows["balls_before"].fillna(0).astype(int),
            "strikes_before": rows["strikes_before"].fillna(0).astype(int),
            "batter_hand": hand_text(rows["batter_hand"]),
        }
    )
    joined = keys.merge(table, on=["pitcher_id", "balls_before", "strikes_before", "batter_hand"], how="left", sort=False)
    q = np.column_stack([joined[f"q_{group}"].to_numpy(float) for group in PITCH_GROUPS])
    fallback = np.column_stack(
        [
            rows["asof_pitcher_fastball_rate"].fillna(0.50).to_numpy(float),
            rows["asof_pitcher_breaking_rate"].fillna(0.30).to_numpy(float),
            rows["asof_pitcher_offspeed_rate"].fillna(0.15).to_numpy(float),
        ]
    )
    fallback = np.column_stack([fallback, np.maximum(0.0, 1.0 - fallback.sum(1))])
    missing = ~np.isfinite(q).all(1)
    q[missing] = fallback[missing]
    q = np.clip(q, 1e-6, None)
    q /= q.sum(1, keepdims=True)
    return q, joined["latent_reliability"].fillna(0.0).to_numpy(float)


def moe_context(rows: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    balls = rows["balls_before"].fillna(0).astype(int).clip(0, 3)
    strikes = rows["strikes_before"].fillna(0).astype(int).clip(0, 2)
    p_hand = rows["pitcher_hand"].astype(str)
    b_hand = rows["batter_hand"].astype(str)
    data: dict[str, np.ndarray] = {"intercept": np.ones(len(rows))}
    for b in range(4):
        for s in range(3):
            data[f"count_{b}_{s}"] = ((balls == b) & (strikes == s)).to_numpy(float)
    data.update(
        {
            "game_type_f": rows["game_type"].astype(str).eq("F").to_numpy(float),
            "pitcher_left": p_hand.isin(["1", "L", "Left"]).to_numpy(float),
            "batter_left": b_hand.isin(["1", "L", "Left"]).to_numpy(float),
            "hand_match": p_hand.eq(b_hand).to_numpy(float),
            "log_pitcher_n": np.log1p(rows["asof_pitcher_n"].fillna(0).clip(lower=0).to_numpy(float)) / 10.0,
            "log_batter_n": np.log1p(rows["asof_batter_n"].fillna(0).clip(lower=0).to_numpy(float)) / 10.0,
            "pitcher_rate": rows["asof_pitcher_success_rate"].fillna(PRIOR).to_numpy(float) - PRIOR,
            "pitcher_recent": rows["asof_pitcher_prev1_game_success_rate"].fillna(PRIOR).to_numpy(float) - PRIOR,
            "batter_rate": rows["asof_batter_success_rate"].fillna(PRIOR).to_numpy(float) - PRIOR,
            "li": rows["li"].fillna(0.98).clip(0, 5).to_numpy(float) / 5.0,
            "runners": rows["num_runners_on"].fillna(0).clip(0, 3).to_numpy(float) / 3.0,
            "inning": rows["inning"].fillna(1).clip(1, 15).to_numpy(float) / 15.0,
        }
    )
    columns = list(data)
    return np.column_stack([data[column] for column in columns]), columns


def fit_moe(x, q, reliability, residual) -> np.ndarray:
    coefficients = []
    for pos in range(len(PITCH_GROUPS)):
        weights = 0.05 + q[:, pos] * (0.25 + 0.75 * reliability)
        model = Ridge(alpha=20000.0, fit_intercept=False).fit(x, residual, sample_weight=weights)
        coefficients.append(model.coef_)
    return np.vstack(coefficients)


def predict_moe(anchor, x, q, reliability, coefficients, scale=0.30, cap=0.012):
    delta = x @ coefficients.T
    experts = np.clip(anchor[:, None] + np.clip(scale * delta, -cap, cap), 0.02, 0.98)
    mixed = np.sum(q * experts, axis=1)
    gate = np.clip(0.25 + 0.75 * reliability, 0.25, 1.0)
    return np.clip(anchor + gate * (mixed - anchor), 0.02, 0.98)


def quantile_table(tm: pd.DataFrame, mapping: pd.DataFrame, target_year: int) -> pd.DataFrame:
    history = tm.loc[tm["season"].lt(target_year)].merge(
        mapping[["pitcher_id", "pitcher_trackman_id", "mapping_similarity"]], on="pitcher_trackman_id", how="inner", validate="many_to_one"
    )
    last = history.loc[history["season"].eq(target_year - 1)]
    result = mapping[["pitcher_id", "mapping_similarity"]].drop_duplicates("pitcher_id").set_index("pitcher_id")
    result["tm_n"] = history.groupby("pitcher_id").size()
    result["tm_last_n"] = last.groupby("pitcher_id").size()
    result["tm_reliability"] = (result["tm_last_n"].fillna(0) / (result["tm_last_n"].fillna(0) + 200.0)) * result["mapping_similarity"]
    for metric in METRICS:
        career_q = history.groupby("pitcher_id")[metric].quantile([0.1, 0.5, 0.9]).unstack()
        last_q = last.groupby("pitcher_id")[metric].quantile([0.1, 0.5, 0.9]).unstack()
        for quantile, suffix in ((0.1, "q10"), (0.5, "q50"), (0.9, "q90")):
            result[f"career_{metric}_{suffix}"] = career_q.get(quantile)
            result[f"last_{metric}_{suffix}"] = last_q.get(quantile)
        result[f"drift_{metric}_median"] = result[f"last_{metric}_q50"] - result[f"career_{metric}_q50"]
        result[f"drift_{metric}_spread"] = (result[f"last_{metric}_q90"] - result[f"last_{metric}_q10"]) - (result[f"career_{metric}_q90"] - result[f"career_{metric}_q10"])
    for group in PITCH_GROUPS:
        subset = history.loc[history["pitch_type_group"].eq(group)]
        for metric in ("rel_speed", "spin_rate", "induced_vert_break", "horz_break"):
            values = subset.groupby("pitcher_id")[metric].quantile([0.1, 0.5, 0.9]).unstack()
            result[f"{group}_{metric}_q50"] = values.get(0.5)
            result[f"{group}_{metric}_spread"] = values.get(0.9) - values.get(0.1)
    return result.reset_index()


def quantile_features(rows: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"pitcher_id": rows["pitcher_id"].astype(str)}).merge(table, on="pitcher_id", how="left", sort=False).drop(columns="pitcher_id")


def fit_quantile(x: pd.DataFrame, residual: np.ndarray):
    values = x.to_numpy(float)
    mean = np.nanmean(values, axis=0)
    mean = np.where(np.isfinite(mean), mean, 0.0)
    values = np.where(np.isfinite(values), values, mean)
    std = np.nanstd(values, axis=0)
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)
    z = (values - mean) / std
    model = Ridge(alpha=50000.0, fit_intercept=False).fit(z, residual)
    return model.coef_, mean, std


def predict_quantile(anchor, x, coef, mean, std, scale=0.25, cap=0.010):
    values = x.to_numpy(float)
    values = np.where(np.isfinite(values), values, mean)
    delta = np.clip(scale * (((values - mean) / std) @ coef), -cap, cap)
    reliability = x["tm_reliability"].fillna(0.0).clip(0, 1).to_numpy(float)
    return np.clip(anchor + reliability * delta, 0.02, 0.98)


def evaluate(name, rows, baseline, candidate, year, seed):
    y = rows["control_success"].to_numpy(float)
    gain = brier(y, baseline) - brier(y, candidate)
    return {
        "candidate": name,
        "season": int(year),
        "rows": int(len(rows)),
        "baseline_brier": brier(y, baseline),
        "candidate_brier": brier(y, candidate),
        "brier_gain": gain,
        "mean_absolute_change": float(np.mean(np.abs(candidate - baseline))),
        "max_absolute_change": float(np.max(np.abs(candidate - baseline))),
        "pitcher_bootstrap": pitcher_bootstrap(y, baseline, candidate, rows["pitcher_id"].to_numpy(), seed),
    }


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    columns = [
        "row_id", "season", "game_month", "game_dayofweek", "game_type", "inning", "balls_before", "strikes_before",
        "num_runners_on", "li", "pitcher_id", "batter_id", "pitcher_hand", "batter_hand", "asof_pitcher_n",
        "asof_batter_n", "asof_pitcher_success_rate", "asof_pitcher_prev1_game_success_rate", "asof_batter_success_rate",
        "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate", "control_success",
    ]
    raw = pd.read_csv(TRAIN_PATH, usecols=columns, low_memory=False)
    anchor = pd.read_csv(ANCHOR_PATH, usecols=["row_id", "season", "target", "p113a_strict"], low_memory=False)
    validation = raw.loc[raw["season"].isin(YEARS)].reset_index(drop=True)
    if not np.array_equal(validation["row_id"].astype(str), anchor["row_id"].astype(str)):
        raise RuntimeError("strict 113A anchor row order mismatch")
    if not np.array_equal(validation["control_success"].to_numpy(float), anchor["target"].to_numpy(float)):
        raise RuntimeError("strict 113A anchor target mismatch")
    validation["p113a_strict"] = anchor["p113a_strict"].to_numpy(float)

    tm_columns = ["pitcher_trackman_id", "season", "game_month", "game_dayofweek", "pitcher_hand", "batter_hand", "balls_before", "strikes_before", "pitch_type_group", *METRICS]
    tm = pd.read_csv(TM_PATH, usecols=tm_columns, low_memory=False)
    tm_keys = tm[["pitcher_trackman_id", "season", "game_month", "game_dayofweek", "pitcher_hand"]]
    mappings = {year: build_mapping(raw, tm_keys, year) for year in (*YEARS, 2025)}
    for year, mapping in mappings.items():
        mapping.to_csv(OUT / f"mapping_{year}.csv", index=False)

    report: dict[str, object] = {
        "experiment": "REF4-119-ABC",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "champion": {"version": "113A", "official_score": 1121.9039933605},
        "provenance": {"train_sha256": sha256(TRAIN_PATH), "trackman_sha256": sha256(TM_PATH), "strict_anchor_sha256": sha256(ANCHOR_PATH)},
        "test_read": False,
        "mapping_rows": {str(year): int(len(mapping)) for year, mapping in mappings.items()},
    }

    # 119A: fixed 2024-era partial-pooling correction.  Month is used only for
    # the within-season train/validation boundary, never as an evaluation-row aggregate.
    abs_results = []
    abs_predictions = []
    abs_keys = ["game_type", "balls_before", "strikes_before", "hand_match"]
    for year in YEARS:
        rows = validation.loc[validation["season"].eq(year)].copy()
        rows["hand_match"] = (rows["pitcher_hand"].astype(str) == rows["batter_hand"].astype(str)).astype(int)
        fit = rows.loc[rows["game_month"].le(6)].copy()
        valid = rows.loc[rows["game_month"].ge(7)].copy()
        fit["residual"] = fit["control_success"] - fit["p113a_strict"]
        table = fit.groupby(abs_keys, observed=True)["residual"].agg(["mean", "size"]).reset_index()
        table["correction"] = 0.25 * table["mean"] * table["size"] / (table["size"] + 1000.0)
        table["correction"] = table["correction"].clip(-0.012, 0.012)
        joined = valid.merge(table[abs_keys + ["correction"]], on=abs_keys, how="left", sort=False)
        baseline = joined["p113a_strict"].to_numpy(float)
        candidate = np.clip(baseline + joined["correction"].fillna(0.0).to_numpy(float), 0.02, 0.98)
        result = evaluate("119A", joined, baseline, candidate, year, 119000 + year)
        result["protocol"] = f"{year} months<=6 -> months>=7"
        abs_results.append(result)
        abs_predictions.append(pd.DataFrame({"row_id": joined["row_id"], "season": year, "target": joined["control_success"], "baseline": baseline, "candidate": candidate}))
    gain_by_year = {item["season"]: item["brier_gain"] for item in abs_results}
    abs_gate = bool(gain_by_year[2024] > 0.0 and min(gain_by_year.values()) >= -0.00005 and max(item["max_absolute_change"] for item in abs_results) <= 0.012 + 1e-12)
    full_2024 = validation.loc[validation["season"].eq(2024)].copy()
    full_2024["hand_match"] = (full_2024["pitcher_hand"].astype(str) == full_2024["batter_hand"].astype(str)).astype(int)
    full_2024["residual"] = full_2024["control_success"] - full_2024["p113a_strict"]
    abs_table = full_2024.groupby(abs_keys, observed=True)["residual"].agg(["mean", "size"]).reset_index()
    abs_table["correction"] = (0.25 * abs_table["mean"] * abs_table["size"] / (abs_table["size"] + 1000.0)).clip(-0.012, 0.012)
    abs_table[abs_keys + ["correction", "size"]].to_csv(OUT / "abs_era_119a_table.csv", index=False)
    (OUT / "abs_era_119a_meta.json").write_text(json.dumps({"candidate": "ABS-ERA-RESIDUAL-119A", "enabled": abs_gate, "correction_scale": 0.25, "shrinkage": 1000.0, "delta_cap": 0.012, "gate_pass": abs_gate}, indent=2) + "\n")
    pd.concat(abs_predictions, ignore_index=True).to_csv(OUT / "oof_119a.csv", index=False)
    report["119A"] = {"name": "ABS-ERA-RESIDUAL-119A", "results": abs_results, "gate_pass": abs_gate}

    # Shared strictly historical latent/quantile tables.
    latent = {year: latent_table(tm, mappings[year], year) for year in (*YEARS, 2025)}
    quantiles = {year: quantile_table(tm, mappings[year], year) for year in (*YEARS, 2025)}
    frames = {year: validation.loc[validation["season"].eq(year)].reset_index(drop=True) for year in YEARS}

    # 119B: four weighted experts and a convex marginalization.
    x_moe = {}
    q_moe = {}
    rel_moe = {}
    moe_columns = None
    for year in YEARS:
        x_moe[year], columns_now = moe_context(frames[year])
        q_moe[year], rel_moe[year] = latent_probabilities(frames[year], latent[year])
        if moe_columns is None:
            moe_columns = columns_now
        elif moe_columns != columns_now:
            raise RuntimeError("119B feature column drift")
    moe_results = []
    moe_predictions = []
    for valid_year, fit_years in ((2023, (2022,)), (2024, (2022, 2023))):
        x_fit = np.vstack([x_moe[year] for year in fit_years])
        q_fit = np.vstack([q_moe[year] for year in fit_years])
        rel_fit = np.concatenate([rel_moe[year] for year in fit_years])
        residual = np.concatenate([frames[year]["control_success"].to_numpy(float) - frames[year]["p113a_strict"].to_numpy(float) for year in fit_years])
        coefficients = fit_moe(x_fit, q_fit, rel_fit, residual)
        baseline = frames[valid_year]["p113a_strict"].to_numpy(float)
        candidate = predict_moe(baseline, x_moe[valid_year], q_moe[valid_year], rel_moe[valid_year], coefficients)
        result = evaluate("119B", frames[valid_year], baseline, candidate, valid_year, 119100 + valid_year)
        result["fit_seasons"] = list(fit_years)
        result["latent_coverage"] = float(np.mean(rel_moe[valid_year] > 0))
        moe_results.append(result)
        moe_predictions.append(pd.DataFrame({"row_id": frames[valid_year]["row_id"], "season": valid_year, "target": frames[valid_year]["control_success"], "baseline": baseline, "candidate": candidate}))
    moe_gains = [item["brier_gain"] for item in moe_results]
    moe_gate = bool(moe_results[-1]["brier_gain"] > 0 and min(moe_gains) >= -0.00003 and max(item["max_absolute_change"] for item in moe_results) <= 0.012 + 1e-12)
    x_all = np.vstack([x_moe[year] for year in YEARS])
    q_all = np.vstack([q_moe[year] for year in YEARS])
    rel_all = np.concatenate([rel_moe[year] for year in YEARS])
    residual_all = np.concatenate([frames[year]["control_success"].to_numpy(float) - frames[year]["p113a_strict"].to_numpy(float) for year in YEARS])
    final_moe = fit_moe(x_all, q_all, rel_all, residual_all)
    np.savez_compressed(OUT / "latent_moe_119b_models.npz", columns=np.array(moe_columns), coef=final_moe)
    latent[2025].to_csv(OUT / "latent_pitch_119b.csv", index=False)
    (OUT / "latent_moe_119b_meta.json").write_text(json.dumps({"candidate": "LATENT-PITCH-MARGINAL-MOE-119B", "enabled": moe_gate, "correction_scale": 0.30, "delta_cap": 0.012, "ridge_alpha": 20000.0, "gate_pass": moe_gate}, indent=2) + "\n")
    pd.concat(moe_predictions, ignore_index=True).to_csv(OUT / "oof_119b.csv", index=False)
    report["119B"] = {"name": "LATENT-PITCH-MARGINAL-MOE-119B", "results": moe_results, "gate_pass": moe_gate}

    # 119C: past-only distributional mechanics and year-to-year quantile drift.
    x_quantile = {year: quantile_features(frames[year], quantiles[year]) for year in YEARS}
    quantile_columns = x_quantile[2022].columns.tolist()
    quantile_results = []
    quantile_predictions = []
    for valid_year, fit_years in ((2023, (2022,)), (2024, (2022, 2023))):
        x_fit = pd.concat([x_quantile[year] for year in fit_years], ignore_index=True)
        residual = np.concatenate([frames[year]["control_success"].to_numpy(float) - frames[year]["p113a_strict"].to_numpy(float) for year in fit_years])
        coef, mean, std = fit_quantile(x_fit, residual)
        baseline = frames[valid_year]["p113a_strict"].to_numpy(float)
        candidate = predict_quantile(baseline, x_quantile[valid_year], coef, mean, std)
        result = evaluate("119C", frames[valid_year], baseline, candidate, valid_year, 119200 + valid_year)
        result["fit_seasons"] = list(fit_years)
        result["mapped_row_coverage"] = float(np.mean(x_quantile[valid_year]["tm_reliability"].fillna(0).to_numpy() > 0))
        quantile_results.append(result)
        quantile_predictions.append(pd.DataFrame({"row_id": frames[valid_year]["row_id"], "season": valid_year, "target": frames[valid_year]["control_success"], "baseline": baseline, "candidate": candidate}))
    quantile_gains = [item["brier_gain"] for item in quantile_results]
    quantile_gate = bool(quantile_results[-1]["brier_gain"] > 0 and min(quantile_gains) >= -0.00003 and max(item["max_absolute_change"] for item in quantile_results) <= 0.010 + 1e-12)
    x_quantile_all = pd.concat([x_quantile[year] for year in YEARS], ignore_index=True)
    residual_quantile_all = np.concatenate([frames[year]["control_success"].to_numpy(float) - frames[year]["p113a_strict"].to_numpy(float) for year in YEARS])
    coef, mean, std = fit_quantile(x_quantile_all, residual_quantile_all)
    np.savez_compressed(OUT / "quantile_drift_119c_model.npz", columns=np.array(quantile_columns), coef=coef, mean=mean, std=std)
    quantiles[2025].to_csv(OUT / "trackman_quantile_119c.csv", index=False)
    (OUT / "quantile_drift_119c_meta.json").write_text(json.dumps({"candidate": "TRACKMAN-QUANTILE-DRIFT-119C", "enabled": quantile_gate, "correction_scale": 0.25, "delta_cap": 0.010, "ridge_alpha": 50000.0, "gate_pass": quantile_gate}, indent=2) + "\n")
    pd.concat(quantile_predictions, ignore_index=True).to_csv(OUT / "oof_119c.csv", index=False)
    report["119C"] = {"name": "TRACKMAN-QUANTILE-DRIFT-119C", "results": quantile_results, "gate_pass": quantile_gate}

    report["elapsed_seconds"] = float(time.time() - started)
    report["status"] = "RESEARCH_COMPLETE"
    (OUT / "comparison_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: {"gate_pass": report[key]["gate_pass"], "gains": [item["brier_gain"] for item in report[key]["results"]]} for key in ("119A", "119B", "119C")}, indent=2))


if __name__ == "__main__":
    main()
