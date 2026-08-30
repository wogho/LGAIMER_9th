#!/usr/bin/env python3
"""Nested-forward actual Transition Gate OOF ablation for current REF4 baseline."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")
try: os.nice(10)
except OSError: pass

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "REF4-TRANSITION-GATE-033A"
OUT = ROOT / "model" / EXPERIMENT_ID
BASE = ROOT / "model" / "REF4-EXACT-OOF-031A"
GATE_BASE = ROOT / "model" / "REF4-ADAPTIVE-GATE-031B"
TRAIN = ROOT / "data" / "train.csv"
CAT = ["game_type", "prior_type", "transition", "count", "hand", "team_type"]
SCALE = 0.15
REPS = 2000
BOOTSTRAP_SEED = 330200


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""): digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); tmp.replace(path)


def prior_type_table(history: pd.DataFrame, target_year: int) -> pd.Series:
    past = history.loc[history["season"].lt(target_year)]
    counts = past.groupby(["pitcher_id", "season", "game_type"], observed=True).size().rename("n").reset_index()
    dominant = counts.sort_values("n").groupby(["pitcher_id", "season"], observed=True).tail(1)
    latest = dominant.sort_values("season").groupby("pitcher_id", observed=True).tail(1)
    return latest.set_index(latest["pitcher_id"].astype(str))["game_type"].astype(str)


def features(rows: pd.DataFrame, base: np.ndarray, prior: pd.Series) -> pd.DataFrame:
    pid = rows["pitcher_id"].astype(str); previous = pid.map(prior).fillna("NEW").astype(str); current = rows["game_type"].astype(str)
    x = pd.DataFrame({
        "game_type": current, "prior_type": previous, "transition": previous + ">" + current,
        "count": rows["balls_before"].astype(str) + "-" + rows["strikes_before"].astype(str),
        "hand": rows["pitcher_hand"].astype(str) + "-" + rows["batter_hand"].astype(str),
        "team_type": rows["pitcher_team_id"].astype(str) + "|" + current,
        "base_prediction": base,
        "log_pitcher_n": np.log1p(pd.to_numeric(rows["asof_pitcher_n"], errors="coerce").fillna(0)),
        "career": pd.to_numeric(rows["asof_pitcher_success_rate"], errors="coerce"),
        "recent1": pd.to_numeric(rows["asof_pitcher_prev1_game_success_rate"], errors="coerce"),
        "recent3": pd.to_numeric(rows["asof_pitcher_prev3_game_success_rate"], errors="coerce"),
        "recent5": pd.to_numeric(rows["asof_pitcher_prev5_game_success_rate"], errors="coerce"),
        "middle": pd.to_numeric(rows["asof_pitcher_middle_rate"], errors="coerce"),
        "reverse": pd.to_numeric(rows["asof_pitcher_reverse_rate"], errors="coerce"),
        "li": pd.to_numeric(rows["li"], errors="coerce"), "inning": pd.to_numeric(rows["inning"], errors="coerce"),
        "runners": pd.to_numeric(rows["num_runners_on"], errors="coerce"),
    })
    x[CAT] = x[CAT].astype("string").fillna("__MISSING__").astype(str)
    return x.replace([np.inf, -np.inf], np.nan)


def fit_gate(path: Path, frames: dict[int, dict[str, object]], years: tuple[int, ...]) -> CatBoostRegressor:
    x = pd.concat([frames[year]["features"] for year in years], ignore_index=True)  # type: ignore[arg-type]
    residual = np.concatenate([frames[year]["target"] - frames[year]["base"] for year in years])  # type: ignore[operator]
    weights = np.concatenate([np.full(len(frames[year]["target"]), 0.30 if len(years) > 1 and year == min(years) else 1.0) for year in years])  # type: ignore[arg-type]
    model = CatBoostRegressor()
    if path.exists(): model.load_model(path)
    else:
        model = CatBoostRegressor(iterations=250, depth=6, learning_rate=.025, loss_function="RMSE", l2_leaf_reg=100, random_strength=.2, bootstrap_type="Bernoulli", subsample=.8, random_seed=968500, thread_count=3, allow_writing_files=False, verbose=False)
        model.fit(x, residual, sample_weight=weights, cat_features=CAT); model.save_model(path)
    del x, residual, weights; gc.collect(); return model


def metric(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    rate = float(y.mean()); brier = float(np.mean((y - p) ** 2)); bss = float(1 - brier / (rate * (1 - rate)))
    return {"rows": int(len(y)), "target_rate": rate, "brier": brier, "bss": bss, "local_score": 100000 * bss}


def cluster_ci(y: np.ndarray, base: np.ndarray, candidate: np.ndarray, pitcher: np.ndarray, seed: int) -> dict[str, float | int]:
    gain = (y - base) ** 2 - (y - candidate) ** 2
    groups = pd.DataFrame({"pitcher": pitcher.astype(str), "gain": gain}).groupby("pitcher", sort=True)["gain"].agg(["sum", "count"])
    sums = groups["sum"].to_numpy(float); counts = groups["count"].to_numpy(float); draws = np.random.default_rng(seed).integers(0, len(groups), size=(REPS, len(groups)))
    values = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {"clusters": int(len(groups)), "repetitions": REPS, "seed": seed, "brier_gain": float(gain.mean()), "ci_low": float(np.quantile(values, .025)), "ci_high": float(np.quantile(values, .975))}


def main() -> None:
    started = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    preflight = json.loads((OUT / "preflight_report.json").read_text(encoding="utf-8"))
    if preflight["status"] != "AUDIT_VERIFIED" or preflight["mismatch_count"] != 0: raise RuntimeError("033A preflight not verified")
    raw = pd.read_csv(TRAIN, low_memory=False)
    oofs = {
        2022: pd.read_csv(GATE_BASE / "oof_2022.csv", dtype={"row_id": str, "game_type": str, "pitcher_id": str}),
        2023: pd.read_csv(BASE / "oof_2023.csv", dtype={"row_id": str, "game_type": str, "pitcher_id": str}),
        2024: pd.read_csv(BASE / "oof_2024.csv", dtype={"row_id": str, "game_type": str, "pitcher_id": str}),
    }
    needed = ["row_id", "season", "pitcher_id", "game_type", "pitcher_hand", "batter_hand", "pitcher_team_id", "balls_before", "strikes_before", "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev5_game_success_rate", "asof_pitcher_middle_rate", "asof_pitcher_reverse_rate", "li", "inning", "num_runners_on", "control_success"]
    frames: dict[int, dict[str, object]] = {}; lookup_hashes: dict[str, str] = {}
    lookup_dir = OUT / "prior_type"; lookup_dir.mkdir(exist_ok=True)
    for year in (2022, 2023, 2024):
        rows = raw.loc[raw["season"].eq(year), needed].reset_index(drop=True); oof = oofs[year]
        if not np.array_equal(rows["row_id"].astype(str).to_numpy(), oof["row_id"].astype(str).to_numpy()): raise RuntimeError(f"row mismatch {year}")
        if not np.array_equal(rows["control_success"].to_numpy(float), oof["target"].to_numpy(float)): raise RuntimeError(f"target mismatch {year}")
        prior = prior_type_table(raw, year); lookup_path = lookup_dir / f"prior_type_{year}.csv"
        pd.DataFrame({"pitcher_id": prior.index.astype(str), "prior_type": prior.to_numpy(str)}).sort_values("pitcher_id").to_csv(lookup_path, index=False)
        lookup_hashes[str(year)] = sha256_path(lookup_path)
        base = oof["prediction"].to_numpy(float)
        frames[year] = {"rows": rows, "oof": oof, "target": rows["control_success"].to_numpy(float), "base": base, "features": features(rows, base, prior)}

    contract = {"gate_for_2023": {"training_years": [2022], "validation_year": 2023}, "gate_for_2024": {"training_years": [2022, 2023], "training_weights": [0.30, 1.00], "validation_year": 2024}, "iterations": 250, "depth": 6, "learning_rate": .025, "l2_leaf_reg": 100, "random_seed": 968500, "thread_count": 3, "transition_scale": SCALE, "bootstrap_repetitions": REPS, "bootstrap_seed": BOOTSTRAP_SEED, "prior_lookup_hashes": lookup_hashes}
    contract_path = OUT / "gate_contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text(encoding="utf-8")) != contract: raise RuntimeError("existing transition contract mismatch")
    write_json(contract_path, contract)
    model_dir = OUT / "gate_models"; model_dir.mkdir(exist_ok=True)
    gates = {2023: fit_gate(model_dir / "gate_for_2023.cbm", frames, (2022,)), 2024: fit_gate(model_dir / "gate_for_2024.cbm", frames, (2022, 2023))}

    parts = []; base_metrics: dict[str, object] = {}; candidate_metrics: dict[str, object] = {}; cis: dict[str, object] = {}
    for year in (2023, 2024):
        frame = frames[year]; correction = gates[year].predict(frame["features"])  # type: ignore[arg-type]
        y = frame["target"]; base = frame["base"]  # type: ignore[assignment]
        candidate = np.clip(base + SCALE * correction, 1e-5, 1 - 1e-5)
        oof = frame["oof"]  # type: ignore[assignment]
        parts.append(pd.DataFrame({"row_id": oof["row_id"], "season": year, "pitcher_id": oof["pitcher_id"], "target": y, "base_prediction": base, "transition_residual": correction, "transition_015_prediction": candidate}))
        base_metrics[str(year)] = metric(y, base); candidate_metrics[str(year)] = metric(y, candidate)
        cis[str(year)] = cluster_ci(y, base, candidate, oof["pitcher_id"].astype(str).to_numpy(), BOOTSTRAP_SEED + year)
    output = pd.concat(parts, ignore_index=True); output.to_csv(OUT / "oof_predictions.csv", index=False)
    y_all = output["target"].to_numpy(float); base_all = output["base_prediction"].to_numpy(float); candidate_all = output["transition_015_prediction"].to_numpy(float)
    base_metrics["pooled"] = metric(y_all, base_all); candidate_metrics["pooled"] = metric(y_all, candidate_all)
    base_metrics["worst_season_bss"] = min(float(base_metrics[str(y)]["bss"]) for y in (2023, 2024))  # type: ignore[index]
    candidate_metrics["worst_season_bss"] = min(float(candidate_metrics[str(y)]["bss"]) for y in (2023, 2024))  # type: ignore[index]
    season_gain = {str(year): float(base_metrics[str(year)]["brier"] - candidate_metrics[str(year)]["brier"]) for year in (2023, 2024)}  # type: ignore[index]
    pooled_gain = float(base_metrics["pooled"]["brier"] - candidate_metrics["pooled"]["brier"])  # type: ignore[index]
    worst_gain = float(candidate_metrics["worst_season_bss"] - base_metrics["worst_season_bss"])
    gate_checks = {"2023_brier_gain_positive": season_gain["2023"] > 0, "2024_brier_gain_positive": season_gain["2024"] > 0, "pooled_brier_gain_positive": pooled_gain > 0, "worst_season_bss_gain_positive": worst_gain > 0, "2023_cluster_ci_low_positive": float(cis["2023"]["ci_low"]) > 0, "2024_cluster_ci_low_positive": float(cis["2024"]["ci_low"]) > 0}  # type: ignore[index]
    candidate = {"name": "transition_015", "metrics": candidate_metrics, "season_brier_gain": season_gain, "pooled_brier_gain": pooled_gain, "worst_season_bss_gain": worst_gain, "cluster_ci": cis, "mean_abs_change": float(np.mean(np.abs(candidate_all - base_all))), "mean_change": float(np.mean(candidate_all - base_all)), "gate_checks": gate_checks, "promotion_pass": all(gate_checks.values())}
    result = {"experiment_id": EXPERIMENT_ID, "candidate_status": "PENDING_AUDIT", "source_official_score": 1068.25021, "base_experiment": "REF4-EXACT-OOF-031A", "base_metrics": base_metrics, "candidate": candidate, "candidate_count": 1, "actual_leaf_count": 1, "gate_checks_count": 6, "model_count": len(list(model_dir.glob("*.cbm"))), "oof_rows": len(output), "test_inference_performed": False, "full_train_performed": False, "zip_created": False, "elapsed_seconds": time.time() - started}
    write_json(OUT / "result.json", result)
    embedded = json.dumps({"candidate_status": result["candidate_status"], "base_metrics": base_metrics, "candidate": candidate, "candidate_count": 1, "model_count": result["model_count"], "oof_rows": len(output)}, sort_keys=True)
    lines = [f"# {EXPERIMENT_ID}", "", "- status: `PENDING_AUDIT`", "- candidate count: `1`", f"- models: `{result['model_count']}`", "- test/full-train/ZIP: `false/false/false`", "", "| candidate | 2023 Brier gain | 2024 Brier gain | pooled Brier gain | worst BSS gain | pre-audit pass |", "|---|---:|---:|---:|---:|---|", f"| transition_015 | {season_gain['2023']:.12g} | {season_gain['2024']:.12g} | {pooled_gain:.12g} | {worst_gain:.12g} | {str(candidate['promotion_pass']).lower()} |", "", "<!-- RESULT_JSON_BEGIN", embedded, "RESULT_JSON_END -->"]
    (OUT / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    artifacts = [ROOT / "scripts" / "preflight_ref4_transition_gate_033a.py", ROOT / "scripts" / "run_ref4_transition_gate_033a.py", ROOT / "scripts" / "verify_ref4_transition_gate_033a.py", OUT / "preflight_report.json", OUT / "preflight_report.md", contract_path, OUT / "oof_predictions.csv", OUT / "result.json", OUT / "result.md", BASE / "audit_manifest.json", BASE / "audit_attestation.json", BASE / "oof_2023.csv", BASE / "oof_2024.csv", GATE_BASE / "audit_manifest.json", GATE_BASE / "audit_attestation.json", GATE_BASE / "oof_2022.csv", TRAIN, ROOT / "start03_reference.md", ROOT / "01_제약과금지사항.md", ROOT / "output" / "submit_ref4_champion_030.zip"]
    artifacts.extend(sorted(lookup_dir.glob("*.csv"))); artifacts.extend(sorted(model_dir.glob("*.cbm")))
    records = {str(path.relative_to(ROOT)): {"sha256": sha256_path(path), "size": path.stat().st_size} for path in artifacts}
    audit = {"experiment_id": EXPERIMENT_ID, "status": "PENDING_VALIDATION", "artifact_count": len(records), "artifacts": records, "model_count": result["model_count"], "oof_rows": len(output), "candidate_count": 1, "leaf_count": 1, "gate_checks_count": 6}
    write_json(OUT / "audit_manifest.json", audit)
    print(json.dumps({"status": "PENDING_AUDIT", "candidate_count": 1, "promotion_pass": candidate["promotion_pass"], "model_count": result["model_count"], "oof_rows": len(output), "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__": main()
