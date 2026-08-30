#!/usr/bin/env python3
"""Independent validator for REF4-ADAPTIVE-GATE-031B."""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "REF4-ADAPTIVE-GATE-031B"
OUT = ROOT / "model" / EXPERIMENT_ID
BASE = ROOT / "model" / "REF4-EXACT-OOF-031A"
SOURCE_MODEL = ROOT / "model" / "REF4-CHAMPION-STACK-030"
RAW_PATH = ROOT / "data" / "train.csv"
SCALES = {"scale_050": 0.50, "scale_075": 0.75, "scale_100": 1.00}
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 310200


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def adjusted_components(oof: pd.DataFrame, regime: dict[str, float]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    futures = oof["game_type"].eq("F").to_numpy()
    p0g = oof["p_v2_global"].to_numpy(float); p0f = oof["p_v2_f"].to_numpy(float)
    p1g = oof["p_v3_55_global"].to_numpy(float); p1f = oof["p_v3_55_f"].to_numpy(float)
    p2g = oof["p_v3_30_global"].to_numpy(float)
    p2a = oof["p_v3_30_f_all"].to_numpy(float); p2r = oof["p_v3_30_f_recent"].to_numpy(float)
    p0 = np.where(futures, p0g + regime["v2_scale"] * (p0f - p0g), p0g)
    p1 = np.where(futures, p1g + regime["v355_scale"] * (p1f - p1g), p1g)
    recent_inner = p2g + regime["v330_recent_inner_scale"] * (p2r - p2g)
    f30 = regime["v330_all_weight"] * p2a + (1.0 - regime["v330_all_weight"]) * recent_inner
    p2 = np.where(futures, p2g + regime["v330_scale"] * (f30 - p2g), p2g)
    risks: list[np.ndarray] = []
    for name in ("middle", "wild", "reverse"):
        global_risk = oof[f"risk_{name}_global"].to_numpy(float)
        f_risk = oof[f"risk_{name}_f"].to_numpy(float)
        risks.append(np.where(futures, global_risk + regime["subtype_scale"] * (f_risk - global_risk), global_risk))
    return [p0, p1, p2], risks


def build_gate_features(raw: pd.DataFrame, oof: pd.DataFrame, regime: dict[str, float]) -> pd.DataFrame:
    if not np.array_equal(raw["row_id"].astype(str).to_numpy(), oof["row_id"].astype(str).to_numpy()):
        raise RuntimeError("raw/OOF row_id mismatch")
    predictions, risks = adjusted_components(oof, regime)
    values = np.column_stack(predictions + risks)
    pred3 = values[:, :3]
    x = pd.DataFrame(values, columns=["p_v2", "p_v3_55", "p_v3_30", "risk_middle", "risk_wild", "risk_reverse"])
    x["ensemble_std"] = pred3.std(axis=1)
    x["ensemble_range"] = pred3.max(axis=1) - pred3.min(axis=1)
    x["old_prediction"] = oof["prediction_no_shift"].to_numpy(float)
    x["log_pitcher_n"] = np.log1p(pd.to_numeric(raw["asof_pitcher_n"], errors="coerce").fillna(0).clip(lower=0)).to_numpy()
    x["log_batter_n"] = np.log1p(pd.to_numeric(raw["asof_batter_n"], errors="coerce").fillna(0).clip(lower=0)).to_numpy()
    for output, source in (("li", "li"), ("inning", "inning"), ("balls", "balls_before"), ("strikes", "strikes_before"), ("runners", "num_runners_on")):
        x[output] = pd.to_numeric(raw[source], errors="coerce").fillna(0 if output == "li" else np.nan).to_numpy()
    recent = raw[["asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev5_game_success_rate"]].apply(pd.to_numeric, errors="coerce")
    x["recent_std"] = recent.std(axis=1).fillna(0.15).to_numpy()
    x["recent_gap"] = (recent.mean(axis=1) - pd.to_numeric(raw["asof_pitcher_success_rate"], errors="coerce")).fillna(0).to_numpy()
    return x.replace([np.inf, -np.inf], np.nan)


def metric(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    rate = float(y.mean())
    brier = float(np.mean((y - prediction) ** 2))
    bss = float(1.0 - brier / (rate * (1.0 - rate)))
    return {"rows": int(len(y)), "target_rate": rate, "brier": brier, "bss": bss, "local_score": 100000.0 * bss}


def cluster_ci(y: np.ndarray, base: np.ndarray, candidate: np.ndarray, pitchers: np.ndarray, seed: int) -> dict[str, float | int]:
    gain = (y - base) ** 2 - (y - candidate) ** 2
    grouped = pd.DataFrame({"pitcher": pitchers.astype(str), "gain": gain}).groupby("pitcher", sort=True)["gain"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float); counts = grouped["count"].to_numpy(float)
    draws = np.random.default_rng(seed).integers(0, len(grouped), size=(BOOTSTRAP_REPS, len(grouped)))
    sampled = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {
        "clusters": int(len(grouped)), "repetitions": BOOTSTRAP_REPS, "seed": int(seed),
        "brier_gain": float(gain.mean()), "ci_low": float(np.quantile(sampled, 0.025)),
        "ci_high": float(np.quantile(sampled, 0.975)),
    }


def fit_independent(frames: dict[int, dict[str, object]], years: tuple[int, ...]) -> CatBoostRegressor:
    latest = max(years)
    x = pd.concat([frames[year]["features"] for year in years], ignore_index=True)  # type: ignore[arg-type]
    target = np.concatenate([
        frames[year]["target"] - frames[year]["oof"]["prediction_no_shift"].to_numpy(float)  # type: ignore[index,operator]
        for year in years
    ])
    weights = np.concatenate([np.full(len(frames[year]["target"]), 0.55 ** (latest - year)) for year in years])  # type: ignore[arg-type]
    model = CatBoostRegressor(
        iterations=73, depth=3, learning_rate=0.025, loss_function="RMSE", l2_leaf_reg=30,
        random_strength=0.2, bootstrap_type="Bernoulli", subsample=0.8,
        random_seed=280033, thread_count=3, allow_writing_files=False, verbose=False,
    )
    model.fit(x, target, sample_weight=weights)
    return model


def max_numeric_difference(actual: object, expected: object, path: str = "") -> tuple[float, list[str]]:
    maximum = 0.0
    mismatches: list[str] = []
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected):
            mismatches.append(f"{path}:keys")
            return maximum, mismatches
        for key in actual:
            diff, child = max_numeric_difference(actual[key], expected[key], f"{path}.{key}" if path else str(key))
            maximum = max(maximum, diff); mismatches.extend(child)
    elif isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            mismatches.append(f"{path}:length")
        else:
            for index, (left, right) in enumerate(zip(actual, expected)):
                diff, child = max_numeric_difference(left, right, f"{path}[{index}]")
                maximum = max(maximum, diff); mismatches.extend(child)
    elif isinstance(actual, (int, float)) and not isinstance(actual, bool) and isinstance(expected, (int, float)) and not isinstance(expected, bool):
        difference = abs(float(actual) - float(expected))
        maximum = max(maximum, difference)
        if difference > 1e-10:
            mismatches.append(f"{path}:{difference}")
    elif actual != expected:
        mismatches.append(f"{path}:{actual!r}!={expected!r}")
    return maximum, mismatches


def main() -> None:
    checks: list[dict[str, object]] = []
    mismatches: list[str] = []

    def check(name: str, passed: bool, actual: object) -> None:
        checks.append({"name": name, "checked": True, "passed": bool(passed), "actual": actual})
        if not passed:
            mismatches.append(name)

    paths = {
        "manifest": OUT / "audit_manifest.json", "result": OUT / "result.json",
        "markdown": OUT / "result.md", "predictions": OUT / "gate_oof_predictions.csv",
        "oof_2022": OUT / "oof_2022.csv", "contract": OUT / "gate_contract.json",
    }
    for name, path in paths.items():
        check(f"exists:{name}", path.is_file(), path.is_file())
    audit = json.loads(paths["manifest"].read_text(encoding="utf-8"))
    execution = json.loads(paths["result"].read_text(encoding="utf-8"))
    contract = json.loads(paths["contract"].read_text(encoding="utf-8"))
    source_manifest = json.loads((SOURCE_MODEL / "manifest.json").read_text(encoding="utf-8"))
    regime = {key: float(value) for key, value in json.loads((SOURCE_MODEL / "f_regime_meta.json").read_text(encoding="utf-8")).items()}

    artifact_failures: list[str] = []
    for relative, record in audit["artifacts"].items():
        path = ROOT / relative
        if not path.is_file():
            artifact_failures.append(f"missing:{relative}"); continue
        if sha256_path(path) != record["sha256"]:
            artifact_failures.append(f"sha256:{relative}")
        if path.stat().st_size != record["size"]:
            artifact_failures.append(f"size:{relative}")
    check("manifest_artifact_hashes", not artifact_failures, artifact_failures)
    check("manifest_artifact_count", len(audit["artifacts"]) == audit["artifact_count"], {"actual": len(audit["artifacts"]), "recorded": audit["artifact_count"]})

    base_models = sorted((OUT / "fold_2022" / "models").glob("*.cbm"))
    gate_models = sorted((OUT / "gate_models").glob("*.cbm"))
    check("base_2022_model_count", len(base_models) == 55, len(base_models))
    check("nested_gate_model_count", len(gate_models) == 2, len(gate_models))
    check("total_model_count", len(base_models) + len(gate_models) == audit["model_count"] == execution["model_count"] == 57, {"filesystem": len(base_models) + len(gate_models), "audit": audit["model_count"], "execution": execution["model_count"]})
    fold_meta = json.loads((OUT / "fold_2022" / "fold_metadata.json").read_text(encoding="utf-8"))
    check("fold_2022_time_separation", fold_meta["train_max_season"] < 2022 and fold_meta["trackman"]["mapping_source_max_season"] < 2022 and fold_meta["trackman"]["trackman_source_max_season"] < 2022, fold_meta)
    check("fold_2022_trackman_unique", fold_meta["trackman"]["duplicate_keys"] == 0, fold_meta["trackman"]["duplicate_keys"])
    check("fold_2022_subtype_nonempty", fold_meta["subtype"]["recovered_rows"] > 0, fold_meta["subtype"])
    check("contract_training_years", contract["gate_for_2023"] == {"training_years": [2022], "validation_year": 2023} and contract["gate_for_2024"] == {"training_years": [2022, 2023], "validation_year": 2024}, contract)
    check("contract_scales", contract["scales"] == SCALES, contract["scales"])
    check("contract_bootstrap", contract["bootstrap_repetitions"] == BOOTSTRAP_REPS and contract["bootstrap_seed"] == BOOTSTRAP_SEED, {"repetitions": contract["bootstrap_repetitions"], "seed": contract["bootstrap_seed"]})

    base_audit = json.loads((BASE / "audit_manifest.json").read_text(encoding="utf-8"))
    oof_by_year: dict[int, pd.DataFrame] = {2022: pd.read_csv(paths["oof_2022"], dtype={"row_id": str, "game_type": str, "pitcher_id": str})}
    for year in (2023, 2024):
        path = BASE / f"oof_{year}.csv"; key = str(path.relative_to(ROOT))
        actual_hash = sha256_path(path); recorded_hash = base_audit["artifacts"][key]["sha256"]
        check(f"frozen_base_hash_{year}", actual_hash == recorded_hash == contract["base_oof_hashes"][str(year)], {"actual": actual_hash, "recorded": recorded_hash, "contract": contract["base_oof_hashes"][str(year)]})
        oof_by_year[year] = pd.read_csv(path, dtype={"row_id": str, "game_type": str, "pitcher_id": str})

    needed = [
        "row_id", "season", "pitcher_id", "control_success", "asof_pitcher_n", "asof_batter_n", "li",
        "inning", "balls_before", "strikes_before", "num_runners_on", "asof_pitcher_success_rate",
        "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev5_game_success_rate",
    ]
    raw = pd.read_csv(RAW_PATH, usecols=needed, low_memory=False)
    frames: dict[int, dict[str, object]] = {}
    for year in (2022, 2023, 2024):
        raw_year = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        oof = oof_by_year[year].reset_index(drop=True)
        ids_match = np.array_equal(raw_year["row_id"].astype(str).to_numpy(), oof["row_id"].astype(str).to_numpy())
        target_match = np.array_equal(raw_year["control_success"].to_numpy(float), oof["target"].to_numpy(float))
        check(f"oof_{year}_row_order", ids_match, {"raw": len(raw_year), "oof": len(oof)})
        check(f"oof_{year}_target", target_match, int(np.sum(raw_year["control_success"].to_numpy(float) != oof["target"].to_numpy(float))) if len(raw_year) == len(oof) else "length")
        predictions, risks = adjusted_components(oof, regime)
        main = np.average(np.vstack(predictions), axis=0, weights=np.asarray(source_manifest["main_weights"], float))
        no_shift = float(source_manifest["stack_intercept"]) + np.column_stack([main, *risks]) @ np.asarray(source_manifest["stack_coefficients"], float)
        final = np.clip(no_shift + float(source_manifest["global_shift"]), 1e-5, 1 - 1e-5)
        check(f"oof_{year}_formula_no_shift", float(np.max(np.abs(no_shift - oof["prediction_no_shift"].to_numpy(float)))) <= 1e-12, float(np.max(np.abs(no_shift - oof["prediction_no_shift"].to_numpy(float)))))
        check(f"oof_{year}_formula_final", float(np.max(np.abs(final - oof["prediction"].to_numpy(float)))) <= 1e-12, float(np.max(np.abs(final - oof["prediction"].to_numpy(float)))))
        frames[year] = {"raw": raw_year, "oof": oof, "target": raw_year["control_success"].to_numpy(float), "features": build_gate_features(raw_year, oof, regime)}

    predictions = pd.read_csv(paths["predictions"], dtype={"row_id": str, "pitcher_id": str})
    expected_raw = raw.loc[raw["season"].isin([2023, 2024])].reset_index(drop=True)
    check("gate_oof_row_count", len(predictions) == len(expected_raw) == audit["oof_rows"] == execution["oof_rows"], {"predictions": len(predictions), "expected": len(expected_raw), "audit": audit["oof_rows"]})
    check("gate_oof_row_order", np.array_equal(predictions["row_id"].astype(str).to_numpy(), expected_raw["row_id"].astype(str).to_numpy()), int(np.sum(predictions["row_id"].astype(str).to_numpy() != expected_raw["row_id"].astype(str).to_numpy())))
    check("gate_oof_target", np.array_equal(predictions["target"].to_numpy(float), expected_raw["control_success"].to_numpy(float)), int(np.sum(predictions["target"].to_numpy(float) != expected_raw["control_success"].to_numpy(float))))
    check("gate_oof_pitcher", np.array_equal(predictions["pitcher_id"].astype(str).to_numpy(), expected_raw["pitcher_id"].astype(str).to_numpy()), int(np.sum(predictions["pitcher_id"].astype(str).to_numpy() != expected_raw["pitcher_id"].astype(str).to_numpy())))
    numeric = ["base_prediction", "prediction_no_shift", "gate_residual", *SCALES]
    check("gate_predictions_finite", all(np.isfinite(predictions[column].to_numpy(float)).all() for column in numeric), [column for column in numeric if not np.isfinite(predictions[column].to_numpy(float)).all()])
    check("gate_candidates_range", all(((predictions[column] >= 0) & (predictions[column] <= 1)).all() for column in SCALES), [column for column in SCALES if not ((predictions[column] >= 0) & (predictions[column] <= 1)).all()])

    saved_residuals: dict[int, np.ndarray] = {}
    independent_residuals: dict[int, np.ndarray] = {}
    for year, train_years in ((2023, (2022,)), (2024, (2022, 2023))):
        saved = CatBoostRegressor(); saved.load_model(OUT / "gate_models" / f"gate_for_{year}.cbm")
        saved_residuals[year] = saved.predict(frames[year]["features"])  # type: ignore[arg-type]
        independent = fit_independent(frames, train_years)
        independent_residuals[year] = independent.predict(frames[year]["features"])  # type: ignore[arg-type]
        stored = predictions.loc[predictions["season"].eq(year), "gate_residual"].to_numpy(float)
        check(f"gate_{year}_saved_prediction", float(np.max(np.abs(saved_residuals[year] - stored))) <= 1e-12, float(np.max(np.abs(saved_residuals[year] - stored))))
        check(f"gate_{year}_independent_refit", float(np.max(np.abs(independent_residuals[year] - stored))) <= 1e-10, float(np.max(np.abs(independent_residuals[year] - stored))))

    base_metrics: dict[str, object] = {}
    candidate_metrics: dict[str, dict[str, object]] = {name: {} for name in SCALES}
    cis: dict[str, dict[str, object]] = {name: {} for name in SCALES}
    formula_diffs: dict[str, float] = {}
    for year in (2023, 2024):
        mask = predictions["season"].eq(year).to_numpy()
        y = predictions.loc[mask, "target"].to_numpy(float)
        base_prediction = oof_by_year[year]["prediction"].to_numpy(float)
        no_shift = oof_by_year[year]["prediction_no_shift"].to_numpy(float)
        check(f"gate_{year}_base_source", float(np.max(np.abs(base_prediction - predictions.loc[mask, "base_prediction"].to_numpy(float)))) <= 1e-12, float(np.max(np.abs(base_prediction - predictions.loc[mask, "base_prediction"].to_numpy(float)))))
        base_metrics[str(year)] = metric(y, base_prediction)
        for index, (name, scale) in enumerate(SCALES.items()):
            recomputed = np.clip(no_shift + scale * independent_residuals[year] + float(source_manifest["global_shift"]), 1e-5, 1 - 1e-5)
            stored = predictions.loc[mask, name].to_numpy(float)
            formula_diffs[f"{year}.{name}"] = float(np.max(np.abs(recomputed - stored)))
            candidate_metrics[name][str(year)] = metric(y, recomputed)
            cis[name][str(year)] = cluster_ci(y, base_prediction, recomputed, predictions.loc[mask, "pitcher_id"].astype(str).to_numpy(), BOOTSTRAP_SEED + year * 10 + index)
    check("candidate_formulas", max(formula_diffs.values()) <= 1e-10, {"max_abs_diff": max(formula_diffs.values()), "differences": formula_diffs})

    y_all = predictions["target"].to_numpy(float); base_all = predictions["base_prediction"].to_numpy(float)
    base_metrics["pooled"] = metric(y_all, base_all)
    base_metrics["worst_season_bss"] = min(float(base_metrics[str(year)]["bss"]) for year in (2023, 2024))  # type: ignore[index]
    evaluated: dict[str, object] = {}; passed: list[str] = []
    for name, scale in SCALES.items():
        candidate_all = predictions[name].to_numpy(float)
        candidate_metrics[name]["pooled"] = metric(y_all, candidate_all)
        candidate_metrics[name]["worst_season_bss"] = min(float(candidate_metrics[name][str(year)]["bss"]) for year in (2023, 2024))  # type: ignore[index]
        season_gain = {str(year): float(base_metrics[str(year)]["brier"] - candidate_metrics[name][str(year)]["brier"]) for year in (2023, 2024)}  # type: ignore[index]
        pooled_gain = float(base_metrics["pooled"]["brier"] - candidate_metrics[name]["pooled"]["brier"])  # type: ignore[index]
        worst_gain = float(candidate_metrics[name]["worst_season_bss"] - base_metrics["worst_season_bss"])
        gate_checks = {
            "2023_brier_gain_positive": season_gain["2023"] > 0,
            "2024_brier_gain_positive": season_gain["2024"] > 0,
            "pooled_brier_gain_positive": pooled_gain > 0,
            "worst_season_bss_gain_positive": worst_gain > 0,
            "2023_cluster_ci_low_positive": float(cis[name]["2023"]["ci_low"]) > 0,  # type: ignore[index]
            "2024_cluster_ci_low_positive": float(cis[name]["2024"]["ci_low"]) > 0,  # type: ignore[index]
        }
        promotion_pass = all(gate_checks.values())
        if promotion_pass:
            passed.append(name)
        evaluated[name] = {
            "scale": scale, "metrics": candidate_metrics[name], "season_brier_gain": season_gain,
            "pooled_brier_gain": pooled_gain, "worst_season_bss_gain": worst_gain,
            "cluster_ci": cis[name], "mean_abs_change": float(np.mean(np.abs(candidate_all - base_all))),
            "mean_change": float(np.mean(candidate_all - base_all)), "gate_checks": gate_checks,
            "promotion_pass": promotion_pass,
        }

    diff, result_mismatches = max_numeric_difference(
        {"base_metrics": execution["base_metrics"], "candidates": execution["candidates"], "preaudit_passed_candidates": execution["preaudit_passed_candidates"]},
        {"base_metrics": base_metrics, "candidates": evaluated, "preaudit_passed_candidates": passed},
    )
    check("metrics_ci_gate_recomputed", not result_mismatches, {"max_abs_diff": diff, "mismatches": result_mismatches})
    check("candidate_leaf_counts", execution["candidate_count"] == execution["actual_leaf_count"] == audit["candidate_count"] == audit["leaf_count"] == len(SCALES), {"execution": execution["candidate_count"], "leaf": execution["actual_leaf_count"], "audit": audit["candidate_count"]})
    check("gate_checks_count", execution["gate_checks_count"] == audit["gate_checks_count"] == len(SCALES) * 6, {"execution": execution["gate_checks_count"], "audit": audit["gate_checks_count"]})
    check("no_test_fulltrain_zip", execution["test_inference_performed"] is False and execution["full_train_performed"] is False and execution["zip_created"] is False, {"test": execution["test_inference_performed"], "full_train": execution["full_train_performed"], "zip": execution["zip_created"]})

    markdown = paths["markdown"].read_text(encoding="utf-8")
    match = re.search(r"<!-- RESULT_JSON_BEGIN\n(.*?)\nRESULT_JSON_END -->", markdown, flags=re.DOTALL)
    embedded = json.loads(match.group(1)) if match else None
    expected_embedded = {
        "candidate_status": execution["candidate_status"], "base_metrics": execution["base_metrics"],
        "candidates": execution["candidates"], "preaudit_passed_candidates": execution["preaudit_passed_candidates"],
        "candidate_count": execution["candidate_count"], "model_count": execution["model_count"], "oof_rows": execution["oof_rows"],
    }
    check("json_markdown_embedded", embedded == expected_embedded, embedded == expected_embedded)

    audit_status = "AUDIT_VERIFIED" if not mismatches else "FAIL"
    promotion_status = "PASS" if audit_status == "AUDIT_VERIFIED" and passed else "FAIL/HOLD"
    report = {
        "experiment_id": EXPERIMENT_ID, "status": audit_status, "promotion_status": promotion_status,
        "passed_candidates": passed, "checked_count": len(checks),
        "passed_count": sum(bool(item["passed"]) for item in checks), "mismatch_count": len(mismatches),
        "mismatches": mismatches, "actual_leaf_count": len(SCALES),
        "gate_checks_count": len(SCALES) * 6, "model_count": len(base_models) + len(gate_models),
        "oof_rows": len(predictions), "base_metrics": base_metrics, "candidates": evaluated,
        "max_formula_abs_diff": max(formula_diffs.values()), "checks": checks,
    }
    report_path = OUT / "validation_report.json"
    write_json(report_path, report)
    validator = Path(__file__).resolve()
    attestation = {
        "experiment_id": EXPERIMENT_ID, "status": audit_status, "promotion_status": promotion_status,
        "manifest_sha256": sha256_path(paths["manifest"]), "validation_report_sha256": sha256_path(report_path),
        "validator_sha256": sha256_path(validator), "checked_count": report["checked_count"],
        "passed_count": report["passed_count"], "mismatch_count": report["mismatch_count"],
        "actual_leaf_count": len(SCALES), "gate_checks_count": len(SCALES) * 6,
        "model_count": report["model_count"], "oof_rows": report["oof_rows"],
        "passed_candidates": passed,
    }
    write_json(OUT / "audit_attestation.json", attestation)
    print(json.dumps(attestation, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
