#!/usr/bin/env python3
"""Nested-forward Adaptive Gate ablation on the exact REF4-030 OOF structure."""
from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")
try:
    os.nice(10)
except OSError:
    pass

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "REF4-ADAPTIVE-GATE-031B"
OUT = ROOT / "model" / EXPERIMENT_ID
BASE = ROOT / "model" / "REF4-EXACT-OOF-031A"
SOURCE_MODEL = ROOT / "model" / "REF4-CHAMPION-STACK-030"
TRAIN = ROOT / "data" / "train.csv"
BASE_RUNNER = ROOT / "scripts" / "run_ref4_exact_oof_031a.py"
SCALES = {"scale_050": 0.50, "scale_075": 0.75, "scale_100": 1.00}
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 310200
THREAD_COUNT = 3


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def load_base_runner():
    spec = importlib.util.spec_from_file_location("ref4_exact_oof_031a_frozen", BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BASE_RUNNER}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.OUT = OUT
    return module


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
    expected_ids = raw["row_id"].astype(str).to_numpy()
    actual_ids = oof["row_id"].astype(str).to_numpy()
    if not np.array_equal(expected_ids, actual_ids):
        raise RuntimeError("raw/OOF row_id order mismatch while building gate features")
    predictions, risks = adjusted_components(oof, regime)
    base = np.column_stack(predictions + risks)
    pred3 = base[:, :3]
    x = pd.DataFrame(base, columns=["p_v2", "p_v3_55", "p_v3_30", "risk_middle", "risk_wild", "risk_reverse"])
    x["ensemble_std"] = pred3.std(axis=1)
    x["ensemble_range"] = pred3.max(axis=1) - pred3.min(axis=1)
    x["old_prediction"] = oof["prediction_no_shift"].to_numpy(float)
    x["log_pitcher_n"] = np.log1p(pd.to_numeric(raw["asof_pitcher_n"], errors="coerce").fillna(0).clip(lower=0)).to_numpy()
    x["log_batter_n"] = np.log1p(pd.to_numeric(raw["asof_batter_n"], errors="coerce").fillna(0).clip(lower=0)).to_numpy()
    x["li"] = pd.to_numeric(raw["li"], errors="coerce").fillna(0).to_numpy()
    x["inning"] = pd.to_numeric(raw["inning"], errors="coerce").to_numpy()
    x["balls"] = pd.to_numeric(raw["balls_before"], errors="coerce").to_numpy()
    x["strikes"] = pd.to_numeric(raw["strikes_before"], errors="coerce").to_numpy()
    x["runners"] = pd.to_numeric(raw["num_runners_on"], errors="coerce").to_numpy()
    recent = raw[[
        "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
    ]].apply(pd.to_numeric, errors="coerce")
    x["recent_std"] = recent.std(axis=1).fillna(0.15).to_numpy()
    x["recent_gap"] = (
        recent.mean(axis=1) - pd.to_numeric(raw["asof_pitcher_success_rate"], errors="coerce")
    ).fillna(0).to_numpy()
    return x.replace([np.inf, -np.inf], np.nan)


def metric(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    rate = float(y.mean())
    brier = float(np.mean((y - prediction) ** 2))
    bss = float(1.0 - brier / (rate * (1.0 - rate)))
    return {"rows": int(len(y)), "target_rate": rate, "brier": brier, "bss": bss, "local_score": 100000.0 * bss}


def cluster_ci(y: np.ndarray, base: np.ndarray, candidate: np.ndarray, pitchers: np.ndarray, seed: int) -> dict[str, float | int]:
    loss_gain = (y - base) ** 2 - (y - candidate) ** 2
    grouped = pd.DataFrame({"pitcher": pitchers.astype(str), "gain": loss_gain}).groupby("pitcher", sort=True)["gain"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(grouped), size=(BOOTSTRAP_REPS, len(grouped)))
    values = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {
        "clusters": int(len(grouped)), "repetitions": BOOTSTRAP_REPS, "seed": int(seed),
        "brier_gain": float(loss_gain.mean()), "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
    }


def fit_gate(path: Path, frames: dict[int, dict[str, object]], years: tuple[int, ...]) -> CatBoostRegressor:
    max_year = max(years)
    x = pd.concat([frames[year]["features"] for year in years], ignore_index=True)  # type: ignore[arg-type]
    residual = np.concatenate([
        frames[year]["target"] - frames[year]["oof"]["prediction_no_shift"].to_numpy(float)  # type: ignore[index,operator]
        for year in years
    ])
    weights = np.concatenate([
        np.full(len(frames[year]["target"]), 0.55 ** (max_year - year), dtype=float)  # type: ignore[arg-type]
        for year in years
    ])
    model = CatBoostRegressor()
    if path.exists():
        model.load_model(path)
    else:
        model = CatBoostRegressor(
            iterations=73, depth=3, learning_rate=0.025, loss_function="RMSE", l2_leaf_reg=30,
            random_strength=0.2, bootstrap_type="Bernoulli", subsample=0.8,
            random_seed=280033, thread_count=THREAD_COUNT, allow_writing_files=False, verbose=False,
        )
        model.fit(x, residual, sample_weight=weights)
        model.save_model(path)
    del x, residual, weights
    gc.collect()
    return model


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    preflight = json.loads((OUT / "preflight_report.json").read_text(encoding="utf-8"))
    if preflight.get("status") != "AUDIT_VERIFIED" or preflight.get("mismatch_count") != 0:
        raise RuntimeError("031B preflight must be AUDIT_VERIFIED")

    source_manifest = json.loads((SOURCE_MODEL / "manifest.json").read_text(encoding="utf-8"))
    regime = {key: float(value) for key, value in json.loads((SOURCE_MODEL / "f_regime_meta.json").read_text(encoding="utf-8")).items()}
    if regime["transition_scale"] != 0.0:
        raise RuntimeError("031B requires transition_scale == 0")
    raw = pd.read_csv(TRAIN, low_memory=False)

    base_module = load_base_runner()
    print("[2022] exact current-structure base OOF", flush=True)
    oof_2022, fold_meta = base_module.fold_predictions(raw, 2022, source_manifest, regime)
    oof_2022.to_csv(OUT / "oof_2022.csv", index=False)

    base_audit = json.loads((BASE / "audit_manifest.json").read_text(encoding="utf-8"))
    oof_by_year: dict[int, pd.DataFrame] = {2022: oof_2022}
    for year in (2023, 2024):
        path = BASE / f"oof_{year}.csv"
        key = str(path.relative_to(ROOT))
        expected_hash = base_audit["artifacts"][key]["sha256"]
        if sha256_path(path) != expected_hash:
            raise RuntimeError(f"frozen base OOF changed: {path}")
        oof_by_year[year] = pd.read_csv(path, dtype={"row_id": str, "game_type": str, "pitcher_id": str})

    needed_raw_columns = [
        "row_id", "season", "pitcher_id", "control_success", "asof_pitcher_n", "asof_batter_n", "li",
        "inning", "balls_before", "strikes_before", "num_runners_on", "asof_pitcher_success_rate",
        "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate",
        "asof_pitcher_prev5_game_success_rate",
    ]
    frames: dict[int, dict[str, object]] = {}
    for year in (2022, 2023, 2024):
        raw_year = raw.loc[raw["season"].eq(year), needed_raw_columns].reset_index(drop=True)
        oof = oof_by_year[year].reset_index(drop=True)
        target = raw_year["control_success"].to_numpy(float)
        if not np.array_equal(target, oof["target"].to_numpy(float)):
            raise RuntimeError(f"target mismatch for {year}")
        frames[year] = {"raw": raw_year, "oof": oof, "target": target, "features": build_gate_features(raw_year, oof, regime)}

    gate_dir = OUT / "gate_models"
    gate_dir.mkdir(exist_ok=True)
    gate_contract = {
        "gate_for_2023": {"training_years": [2022], "validation_year": 2023},
        "gate_for_2024": {"training_years": [2022, 2023], "validation_year": 2024},
        "decay": 0.55, "scales": SCALES, "bootstrap_repetitions": BOOTSTRAP_REPS,
        "bootstrap_seed": BOOTSTRAP_SEED, "thread_count": THREAD_COUNT,
        "target": "control_success - prediction_no_shift", "global_shift": float(source_manifest["global_shift"]),
        "base_oof_hashes": {str(year): sha256_path(BASE / f"oof_{year}.csv") for year in (2023, 2024)},
    }
    contract_path = OUT / "gate_contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text(encoding="utf-8")) != gate_contract:
        raise RuntimeError("existing gate contract mismatch")
    write_json(contract_path, gate_contract)

    gates: dict[int, CatBoostRegressor] = {}
    for valid_year, train_years in ((2023, (2022,)), (2024, (2022, 2023))):
        print(f"[{valid_year}] fit nested gate from {train_years}", flush=True)
        gates[valid_year] = fit_gate(gate_dir / f"gate_for_{valid_year}.cbm", frames, train_years)

    prediction_parts: list[pd.DataFrame] = []
    base_metrics: dict[str, object] = {}
    candidate_metrics: dict[str, dict[str, object]] = {name: {} for name in SCALES}
    ci_by_candidate: dict[str, dict[str, object]] = {name: {} for name in SCALES}
    for valid_year in (2023, 2024):
        frame = frames[valid_year]
        oof = frame["oof"]  # type: ignore[assignment]
        y = frame["target"]  # type: ignore[assignment]
        correction = gates[valid_year].predict(frame["features"])  # type: ignore[arg-type]
        base_prediction = oof["prediction"].to_numpy(float)
        no_shift = oof["prediction_no_shift"].to_numpy(float)
        part = pd.DataFrame({
            "row_id": oof["row_id"].astype(str), "season": valid_year,
            "pitcher_id": oof["pitcher_id"].astype(str), "target": y,
            "base_prediction": base_prediction, "prediction_no_shift": no_shift,
            "gate_residual": correction,
        })
        base_metrics[str(valid_year)] = metric(y, base_prediction)
        for index, (name, scale) in enumerate(SCALES.items()):
            prediction = np.clip(no_shift + scale * correction + float(source_manifest["global_shift"]), 1e-5, 1 - 1e-5)
            part[name] = prediction
            candidate_metrics[name][str(valid_year)] = metric(y, prediction)
            ci_by_candidate[name][str(valid_year)] = cluster_ci(
                y, base_prediction, prediction, oof["pitcher_id"].astype(str).to_numpy(),
                BOOTSTRAP_SEED + valid_year * 10 + index,
            )
        prediction_parts.append(part)
    predictions = pd.concat(prediction_parts, ignore_index=True)
    predictions.to_csv(OUT / "gate_oof_predictions.csv", index=False)

    y_all = predictions["target"].to_numpy(float)
    base_all = predictions["base_prediction"].to_numpy(float)
    base_metrics["pooled"] = metric(y_all, base_all)
    base_metrics["worst_season_bss"] = min(float(base_metrics[str(year)]["bss"]) for year in (2023, 2024))  # type: ignore[index]

    evaluated: dict[str, object] = {}
    passed_names: list[str] = []
    for name, scale in SCALES.items():
        candidate_all = predictions[name].to_numpy(float)
        candidate_metrics[name]["pooled"] = metric(y_all, candidate_all)
        candidate_metrics[name]["worst_season_bss"] = min(float(candidate_metrics[name][str(year)]["bss"]) for year in (2023, 2024))  # type: ignore[index]
        season_brier_gain = {
            str(year): float(base_metrics[str(year)]["brier"] - candidate_metrics[name][str(year)]["brier"])  # type: ignore[index]
            for year in (2023, 2024)
        }
        pooled_gain = float(base_metrics["pooled"]["brier"] - candidate_metrics[name]["pooled"]["brier"])  # type: ignore[index]
        worst_gain = float(candidate_metrics[name]["worst_season_bss"] - base_metrics["worst_season_bss"])
        gates_checked = {
            "2023_brier_gain_positive": season_brier_gain["2023"] > 0.0,
            "2024_brier_gain_positive": season_brier_gain["2024"] > 0.0,
            "pooled_brier_gain_positive": pooled_gain > 0.0,
            "worst_season_bss_gain_positive": worst_gain > 0.0,
            "2023_cluster_ci_low_positive": float(ci_by_candidate[name]["2023"]["ci_low"]) > 0.0,  # type: ignore[index]
            "2024_cluster_ci_low_positive": float(ci_by_candidate[name]["2024"]["ci_low"]) > 0.0,  # type: ignore[index]
        }
        promotion_pass = all(gates_checked.values())
        if promotion_pass:
            passed_names.append(name)
        evaluated[name] = {
            "scale": scale, "metrics": candidate_metrics[name], "season_brier_gain": season_brier_gain,
            "pooled_brier_gain": pooled_gain, "worst_season_bss_gain": worst_gain,
            "cluster_ci": ci_by_candidate[name], "mean_abs_change": float(np.mean(np.abs(candidate_all - base_all))),
            "mean_change": float(np.mean(candidate_all - base_all)), "gate_checks": gates_checked,
            "promotion_pass": promotion_pass,
        }

    result = {
        "experiment_id": EXPERIMENT_ID, "candidate_status": "PENDING_AUDIT",
        "source_official_score": 1068.25021, "base_experiment": "REF4-EXACT-OOF-031A",
        "base_metrics": base_metrics, "candidates": evaluated, "preaudit_passed_candidates": passed_names,
        "candidate_count": len(SCALES), "actual_leaf_count": len(SCALES),
        "gate_checks_count": len(SCALES) * 6, "base_2022_fold": fold_meta,
        "model_count": len(list((OUT / "fold_2022" / "models").glob("*.cbm"))) + len(list(gate_dir.glob("*.cbm"))),
        "oof_rows": int(len(predictions)), "test_inference_performed": False,
        "full_train_performed": False, "zip_created": False, "elapsed_seconds": time.time() - started,
    }
    result_path = OUT / "result.json"
    write_json(result_path, result)
    embedded = json.dumps({
        "candidate_status": result["candidate_status"], "base_metrics": base_metrics,
        "candidates": evaluated, "preaudit_passed_candidates": passed_names,
        "candidate_count": len(SCALES), "model_count": result["model_count"], "oof_rows": result["oof_rows"],
    }, sort_keys=True)
    lines = [
        f"# {EXPERIMENT_ID}", "", "- status: `PENDING_AUDIT`",
        f"- candidate count: `{len(SCALES)}`", f"- models: `{result['model_count']}`",
        "- test inference: `false`", "- full train: `false`", "- ZIP created: `false`", "",
        "| candidate | scale | 2023 Brier gain | 2024 Brier gain | pooled Brier gain | worst-season BSS gain | pre-audit pass |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for name, scale in SCALES.items():
        item = evaluated[name]  # type: ignore[index]
        lines.append(
            f"| {name} | {scale:.2f} | {item['season_brier_gain']['2023']:.12g} | "
            f"{item['season_brier_gain']['2024']:.12g} | {item['pooled_brier_gain']:.12g} | "
            f"{item['worst_season_bss_gain']:.12g} | {str(item['promotion_pass']).lower()} |"
        )
    lines.extend(["", "<!-- RESULT_JSON_BEGIN", embedded, "RESULT_JSON_END -->"])
    result_md = OUT / "result.md"
    result_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    artifacts = [
        ROOT / "scripts" / "preflight_ref4_adaptive_gate_031b.py",
        ROOT / "scripts" / "run_ref4_adaptive_gate_031b.py",
        ROOT / "scripts" / "verify_ref4_adaptive_gate_031b.py",
        BASE_RUNNER, OUT / "preflight_report.json", OUT / "preflight_report.md",
        OUT / "oof_2022.csv", OUT / "gate_contract.json", OUT / "gate_oof_predictions.csv",
        result_path, result_md, BASE / "audit_manifest.json", BASE / "validation_report.json",
        BASE / "audit_attestation.json", BASE / "oof_2023.csv", BASE / "oof_2024.csv",
        SOURCE_MODEL / "manifest.json", SOURCE_MODEL / "f_regime_meta.json", TRAIN,
        ROOT / "data" / "trackman_history.csv", ROOT / "start03_reference.md",
        ROOT / "01_제약과금지사항.md", ROOT / "output" / "submit_ref4_champion_030.zip",
    ]
    artifacts.extend(sorted(path for path in (OUT / "fold_2022").glob("*") if path.is_file()))
    artifacts.extend(sorted((OUT / "fold_2022" / "models").glob("*.cbm")))
    artifacts.extend(sorted(gate_dir.glob("*.cbm")))
    artifact_records = {
        str(path.relative_to(ROOT)): {"sha256": sha256_path(path), "size": path.stat().st_size}
        for path in artifacts
    }
    audit_manifest = {
        "experiment_id": EXPERIMENT_ID, "status": "PENDING_VALIDATION", "created_after_outputs": True,
        "artifact_count": len(artifact_records), "artifacts": artifact_records,
        "model_count": result["model_count"], "oof_rows": result["oof_rows"],
        "candidate_count": len(SCALES), "leaf_count": len(SCALES), "gate_checks_count": len(SCALES) * 6,
    }
    write_json(OUT / "audit_manifest.json", audit_manifest)
    print(json.dumps({
        "status": "PENDING_AUDIT", "model_count": result["model_count"], "oof_rows": result["oof_rows"],
        "candidate_count": len(SCALES), "preaudit_passed_candidates": passed_names,
        "elapsed_seconds": result["elapsed_seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
