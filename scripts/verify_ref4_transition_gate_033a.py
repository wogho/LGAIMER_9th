#!/usr/bin/env python3
"""Independent refit and evidence validator for Transition Gate 033A."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "REF4-TRANSITION-GATE-033A"
OUT = ROOT / "model" / EXPERIMENT_ID
BASE = ROOT / "model" / "REF4-EXACT-OOF-031A"
GATE_BASE = ROOT / "model" / "REF4-ADAPTIVE-GATE-031B"
CAT = ["game_type", "prior_type", "transition", "count", "hand", "team_type"]
SCALE = .15
REPS = 2000
BOOTSTRAP_SEED = 330200


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""): digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def independent_prior(raw: pd.DataFrame, target: int) -> pd.Series:
    counts = raw.loc[raw["season"].lt(target)].groupby(["pitcher_id", "season", "game_type"], observed=True).size().rename("n").reset_index()
    dominant = counts.sort_values("n").groupby(["pitcher_id", "season"], observed=True).tail(1)
    latest = dominant.sort_values("season").groupby("pitcher_id", observed=True).tail(1)
    return latest.set_index(latest["pitcher_id"].astype(str))["game_type"].astype(str)


def independent_features(rows: pd.DataFrame, base: np.ndarray, lookup: pd.Series) -> pd.DataFrame:
    pitcher = rows["pitcher_id"].astype(str); previous = pitcher.map(lookup).fillna("NEW").astype(str); current = rows["game_type"].astype(str)
    frame = pd.DataFrame({
        "game_type": current, "prior_type": previous, "transition": previous + ">" + current,
        "count": rows["balls_before"].astype(str) + "-" + rows["strikes_before"].astype(str),
        "hand": rows["pitcher_hand"].astype(str) + "-" + rows["batter_hand"].astype(str),
        "team_type": rows["pitcher_team_id"].astype(str) + "|" + current,
        "base_prediction": base, "log_pitcher_n": np.log1p(pd.to_numeric(rows["asof_pitcher_n"], errors="coerce").fillna(0)),
        "career": pd.to_numeric(rows["asof_pitcher_success_rate"], errors="coerce"),
        "recent1": pd.to_numeric(rows["asof_pitcher_prev1_game_success_rate"], errors="coerce"),
        "recent3": pd.to_numeric(rows["asof_pitcher_prev3_game_success_rate"], errors="coerce"),
        "recent5": pd.to_numeric(rows["asof_pitcher_prev5_game_success_rate"], errors="coerce"),
        "middle": pd.to_numeric(rows["asof_pitcher_middle_rate"], errors="coerce"),
        "reverse": pd.to_numeric(rows["asof_pitcher_reverse_rate"], errors="coerce"),
        "li": pd.to_numeric(rows["li"], errors="coerce"), "inning": pd.to_numeric(rows["inning"], errors="coerce"),
        "runners": pd.to_numeric(rows["num_runners_on"], errors="coerce"),
    })
    frame[CAT] = frame[CAT].astype("string").fillna("__MISSING__").astype(str)
    return frame.replace([np.inf, -np.inf], np.nan)


def independent_fit(frames: dict[int, dict[str, object]], years: tuple[int, ...]) -> CatBoostRegressor:
    x = pd.concat([frames[y]["features"] for y in years], ignore_index=True)  # type: ignore[arg-type]
    target = np.concatenate([frames[y]["target"] - frames[y]["base"] for y in years])  # type: ignore[operator]
    weights = np.concatenate([np.full(len(frames[y]["target"]), .30 if len(years) > 1 and y == min(years) else 1.) for y in years])  # type: ignore[arg-type]
    model = CatBoostRegressor(iterations=250, depth=6, learning_rate=.025, loss_function="RMSE", l2_leaf_reg=100, random_strength=.2, bootstrap_type="Bernoulli", subsample=.8, random_seed=968500, thread_count=3, allow_writing_files=False, verbose=False)
    model.fit(x, target, sample_weight=weights, cat_features=CAT); return model


def metric(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    rate = float(y.mean()); brier = float(np.mean((y - p) ** 2)); bss = float(1 - brier / (rate * (1 - rate)))
    return {"rows": int(len(y)), "target_rate": rate, "brier": brier, "bss": bss, "local_score": 100000 * bss}


def cluster_ci(y: np.ndarray, base: np.ndarray, candidate: np.ndarray, pitcher: np.ndarray, seed: int) -> dict[str, float | int]:
    gain = (y - base) ** 2 - (y - candidate) ** 2
    groups = pd.DataFrame({"pitcher": pitcher.astype(str), "gain": gain}).groupby("pitcher", sort=True)["gain"].agg(["sum", "count"])
    sums = groups["sum"].to_numpy(float); counts = groups["count"].to_numpy(float); draws = np.random.default_rng(seed).integers(0, len(groups), size=(REPS, len(groups)))
    sample = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {"clusters": int(len(groups)), "repetitions": REPS, "seed": seed, "brier_gain": float(gain.mean()), "ci_low": float(np.quantile(sample, .025)), "ci_high": float(np.quantile(sample, .975))}


def compare(a: object, b: object, path: str = "") -> tuple[float, list[str]]:
    maximum = 0.; failures: list[str] = []
    if isinstance(a, dict) and isinstance(b, dict):
        if set(a) != set(b): return 0., [f"{path}:keys"]
        for key in a:
            diff, child = compare(a[key], b[key], f"{path}.{key}" if path else str(key)); maximum = max(maximum, diff); failures.extend(child)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b): failures.append(f"{path}:length")
        else:
            for i, (left, right) in enumerate(zip(a, b)):
                diff, child = compare(left, right, f"{path}[{i}]"); maximum = max(maximum, diff); failures.extend(child)
    elif isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)) and not isinstance(b, bool):
        diff = abs(float(a) - float(b)); maximum = max(maximum, diff)
        if diff > 1e-10: failures.append(f"{path}:{diff}")
    elif a != b: failures.append(f"{path}:{a!r}!={b!r}")
    return maximum, failures


def main() -> None:
    checks: list[dict[str, object]] = []; mismatches: list[str] = []
    def check(name: str, passed: bool, actual: object) -> None:
        checks.append({"name": name, "checked": True, "passed": bool(passed), "actual": actual})
        if not passed: mismatches.append(name)

    required = [OUT / "audit_manifest.json", OUT / "result.json", OUT / "result.md", OUT / "oof_predictions.csv", OUT / "gate_contract.json"]
    for path in required: check(f"exists:{path.name}", path.is_file(), path.is_file())
    audit = json.loads((OUT / "audit_manifest.json").read_text(encoding="utf-8")); result = json.loads((OUT / "result.json").read_text(encoding="utf-8")); contract = json.loads((OUT / "gate_contract.json").read_text(encoding="utf-8"))
    artifact_failures = []
    for relative, record in audit["artifacts"].items():
        path = ROOT / relative
        if not path.is_file(): artifact_failures.append(f"missing:{relative}"); continue
        if sha256_path(path) != record["sha256"]: artifact_failures.append(f"sha256:{relative}")
        if path.stat().st_size != record["size"]: artifact_failures.append(f"size:{relative}")
    check("artifact_hashes", not artifact_failures, artifact_failures)
    check("artifact_count", len(audit["artifacts"]) == audit["artifact_count"], {"actual": len(audit["artifacts"]), "recorded": audit["artifact_count"]})
    model_files = sorted((OUT / "gate_models").glob("*.cbm")); check("model_count", len(model_files) == audit["model_count"] == result["model_count"] == 2, {"files": len(model_files), "audit": audit["model_count"], "result": result["model_count"]})
    check("contract_structure", contract["gate_for_2023"] == {"training_years": [2022], "validation_year": 2023} and contract["gate_for_2024"] == {"training_years": [2022, 2023], "training_weights": [.3, 1.], "validation_year": 2024} and contract["transition_scale"] == SCALE, contract)

    oofs = {2022: pd.read_csv(GATE_BASE / "oof_2022.csv", dtype={"row_id": str, "game_type": str, "pitcher_id": str}), 2023: pd.read_csv(BASE / "oof_2023.csv", dtype={"row_id": str, "game_type": str, "pitcher_id": str}), 2024: pd.read_csv(BASE / "oof_2024.csv", dtype={"row_id": str, "game_type": str, "pitcher_id": str})}
    for year, directory in ((2022, GATE_BASE), (2023, BASE), (2024, BASE)):
        source_audit = json.loads((directory / "audit_manifest.json").read_text(encoding="utf-8")); source_path = directory / f"oof_{year}.csv"; key = str(source_path.relative_to(ROOT))
        check(f"source_oof_hash_{year}", sha256_path(source_path) == source_audit["artifacts"][key]["sha256"], {"actual": sha256_path(source_path), "recorded": source_audit["artifacts"][key]["sha256"]})

    needed = ["row_id", "season", "pitcher_id", "game_type", "pitcher_hand", "batter_hand", "pitcher_team_id", "balls_before", "strikes_before", "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev5_game_success_rate", "asof_pitcher_middle_rate", "asof_pitcher_reverse_rate", "li", "inning", "num_runners_on", "control_success"]
    raw = pd.read_csv(ROOT / "data" / "train.csv", usecols=needed, low_memory=False); frames: dict[int, dict[str, object]] = {}
    for year in (2022, 2023, 2024):
        rows = raw.loc[raw["season"].eq(year)].reset_index(drop=True); oof = oofs[year]
        check(f"row_order_{year}", np.array_equal(rows["row_id"].astype(str).to_numpy(), oof["row_id"].astype(str).to_numpy()), {"raw": len(rows), "oof": len(oof)})
        check(f"target_{year}", np.array_equal(rows["control_success"].to_numpy(float), oof["target"].to_numpy(float)), int(np.sum(rows["control_success"].to_numpy(float) != oof["target"].to_numpy(float))))
        lookup = independent_prior(raw, year); expected_lookup = pd.DataFrame({"pitcher_id": lookup.index.astype(str), "prior_type": lookup.to_numpy(str)}).sort_values("pitcher_id").reset_index(drop=True)
        stored_lookup = pd.read_csv(OUT / "prior_type" / f"prior_type_{year}.csv", dtype=str).sort_values("pitcher_id").reset_index(drop=True)
        check(f"prior_lookup_{year}", expected_lookup.equals(stored_lookup) and sha256_path(OUT / "prior_type" / f"prior_type_{year}.csv") == contract["prior_lookup_hashes"][str(year)], {"rows": len(stored_lookup), "hash": sha256_path(OUT / "prior_type" / f"prior_type_{year}.csv")})
        base = oof["prediction"].to_numpy(float); frames[year] = {"rows": rows, "oof": oof, "target": rows["control_success"].to_numpy(float), "base": base, "features": independent_features(rows, base, lookup)}

    output = pd.read_csv(OUT / "oof_predictions.csv", dtype={"row_id": str, "pitcher_id": str}); expected_raw = raw.loc[raw["season"].isin([2023, 2024])].reset_index(drop=True)
    check("output_rows", len(output) == len(expected_raw) == audit["oof_rows"] == result["oof_rows"], {"output": len(output), "expected": len(expected_raw)})
    check("output_order", np.array_equal(output["row_id"].astype(str).to_numpy(), expected_raw["row_id"].astype(str).to_numpy()), int(np.sum(output["row_id"].astype(str).to_numpy() != expected_raw["row_id"].astype(str).to_numpy())))
    check("output_target", np.array_equal(output["target"].to_numpy(float), expected_raw["control_success"].to_numpy(float)), int(np.sum(output["target"].to_numpy(float) != expected_raw["control_success"].to_numpy(float))))
    check("finite_range", np.isfinite(output[["base_prediction", "transition_residual", "transition_015_prediction"]].to_numpy(float)).all() and ((output["transition_015_prediction"] >= 0) & (output["transition_015_prediction"] <= 1)).all(), {"min": float(output["transition_015_prediction"].min()), "max": float(output["transition_015_prediction"].max())})

    independent_corrections = {}; prediction_diffs = {}
    for year, train_years in ((2023, (2022,)), (2024, (2022, 2023))):
        independent = independent_fit(frames, train_years); correction = independent.predict(frames[year]["features"])  # type: ignore[arg-type]
        saved = CatBoostRegressor(); saved.load_model(OUT / "gate_models" / f"gate_for_{year}.cbm"); saved_correction = saved.predict(frames[year]["features"])  # type: ignore[arg-type]
        stored = output.loc[output["season"].eq(year), "transition_residual"].to_numpy(float)
        prediction_diffs[f"independent_{year}"] = float(np.max(np.abs(correction - stored))); prediction_diffs[f"saved_{year}"] = float(np.max(np.abs(saved_correction - stored)))
        independent_corrections[year] = correction
    check("independent_refit_predictions", max(prediction_diffs.values()) <= 1e-10, prediction_diffs)

    base_metrics: dict[str, object] = {}; candidate_metrics: dict[str, object] = {}; cis: dict[str, object] = {}; formula_diffs = {}
    for year in (2023, 2024):
        mask = output["season"].eq(year).to_numpy(); y = output.loc[mask, "target"].to_numpy(float); base = frames[year]["base"]  # type: ignore[assignment]
        candidate = np.clip(base + SCALE * independent_corrections[year], 1e-5, 1 - 1e-5); stored = output.loc[mask, "transition_015_prediction"].to_numpy(float)
        formula_diffs[str(year)] = float(np.max(np.abs(candidate - stored))); base_metrics[str(year)] = metric(y, base); candidate_metrics[str(year)] = metric(y, candidate)
        cis[str(year)] = cluster_ci(y, base, candidate, output.loc[mask, "pitcher_id"].astype(str).to_numpy(), BOOTSTRAP_SEED + year)
    check("candidate_formula", max(formula_diffs.values()) <= 1e-10, formula_diffs)
    y_all = output["target"].to_numpy(float); base_all = output["base_prediction"].to_numpy(float); candidate_all = output["transition_015_prediction"].to_numpy(float)
    base_metrics["pooled"] = metric(y_all, base_all); candidate_metrics["pooled"] = metric(y_all, candidate_all)
    base_metrics["worst_season_bss"] = min(float(base_metrics[str(y)]["bss"]) for y in (2023, 2024))  # type: ignore[index]
    candidate_metrics["worst_season_bss"] = min(float(candidate_metrics[str(y)]["bss"]) for y in (2023, 2024))  # type: ignore[index]
    season_gain = {str(year): float(base_metrics[str(year)]["brier"] - candidate_metrics[str(year)]["brier"]) for year in (2023, 2024)}  # type: ignore[index]
    pooled_gain = float(base_metrics["pooled"]["brier"] - candidate_metrics["pooled"]["brier"])  # type: ignore[index]
    worst_gain = float(candidate_metrics["worst_season_bss"] - base_metrics["worst_season_bss"])
    gate_checks = {"2023_brier_gain_positive": season_gain["2023"] > 0, "2024_brier_gain_positive": season_gain["2024"] > 0, "pooled_brier_gain_positive": pooled_gain > 0, "worst_season_bss_gain_positive": worst_gain > 0, "2023_cluster_ci_low_positive": float(cis["2023"]["ci_low"]) > 0, "2024_cluster_ci_low_positive": float(cis["2024"]["ci_low"]) > 0}  # type: ignore[index]
    expected_candidate = {"name": "transition_015", "metrics": candidate_metrics, "season_brier_gain": season_gain, "pooled_brier_gain": pooled_gain, "worst_season_bss_gain": worst_gain, "cluster_ci": cis, "mean_abs_change": float(np.mean(np.abs(candidate_all - base_all))), "mean_change": float(np.mean(candidate_all - base_all)), "gate_checks": gate_checks, "promotion_pass": all(gate_checks.values())}
    diff, failures = compare({"base_metrics": result["base_metrics"], "candidate": result["candidate"]}, {"base_metrics": base_metrics, "candidate": expected_candidate}); check("metrics_ci_gate_recomputed", not failures, {"max_abs_diff": diff, "failures": failures})
    check("leaf_gate_counts", result["candidate_count"] == result["actual_leaf_count"] == audit["candidate_count"] == audit["leaf_count"] == 1 and result["gate_checks_count"] == audit["gate_checks_count"] == 6, {"leaf": result["actual_leaf_count"], "gates": result["gate_checks_count"]})
    check("no_test_fulltrain_zip", result["test_inference_performed"] is False and result["full_train_performed"] is False and result["zip_created"] is False, {"test": result["test_inference_performed"], "fulltrain": result["full_train_performed"], "zip": result["zip_created"]})
    markdown = (OUT / "result.md").read_text(encoding="utf-8"); match = re.search(r"<!-- RESULT_JSON_BEGIN\n(.*?)\nRESULT_JSON_END -->", markdown, re.DOTALL); embedded = json.loads(match.group(1)) if match else None
    expected_embedded = {"candidate_status": result["candidate_status"], "base_metrics": result["base_metrics"], "candidate": result["candidate"], "candidate_count": 1, "model_count": result["model_count"], "oof_rows": result["oof_rows"]}; check("json_markdown_embedded", embedded == expected_embedded, embedded == expected_embedded)

    status = "AUDIT_VERIFIED" if not mismatches else "FAIL"; promotion = "PASS" if status == "AUDIT_VERIFIED" and all(gate_checks.values()) else "FAIL/HOLD"
    report = {"experiment_id": EXPERIMENT_ID, "status": status, "promotion_status": promotion, "checked_count": len(checks), "passed_count": sum(bool(x["passed"]) for x in checks), "mismatch_count": len(mismatches), "mismatches": mismatches, "actual_leaf_count": 1, "gate_checks_count": 6, "model_count": len(model_files), "oof_rows": len(output), "base_metrics": base_metrics, "candidate": expected_candidate, "independent_refit_max_abs_diff": max(prediction_diffs.values()), "formula_max_abs_diff": max(formula_diffs.values()), "checks": checks}
    report_path = OUT / "validation_report.json"; write_json(report_path, report)
    attestation = {"experiment_id": EXPERIMENT_ID, "status": status, "promotion_status": promotion, "manifest_sha256": sha256_path(OUT / "audit_manifest.json"), "validation_report_sha256": sha256_path(report_path), "validator_sha256": sha256_path(Path(__file__).resolve()), "checked_count": report["checked_count"], "passed_count": report["passed_count"], "mismatch_count": report["mismatch_count"], "actual_leaf_count": 1, "gate_checks_count": 6, "model_count": len(model_files), "oof_rows": len(output)}
    write_json(OUT / "audit_attestation.json", attestation); print(json.dumps(attestation, indent=2))
    if mismatches: raise SystemExit(1)


if __name__ == "__main__": main()
