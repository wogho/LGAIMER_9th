#!/usr/bin/env python3
"""Run one fixed, model-free, three-year OOF confirmation."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "model" / "REF4-OOF-DIAG-034A"
REPS = 2000
SEED = 340200
CONFIG = {
    "shift": {"id": "REF4-SHIFT-034B", "name": "shift_000", "column": "prediction_no_shift", "gates": 8},
    "f-regime": {"id": "REF4-F-REGIME-032B", "name": "f_regime_075", "column": "prediction_f075", "gates": 14},
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def metric(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    rate = float(y.mean())
    brier = float(np.mean((y - prediction) ** 2))
    bss = float(1.0 - brier / (rate * (1.0 - rate)))
    return {"rows": int(len(y)), "target_rate": rate, "prediction_mean": float(prediction.mean()), "brier": brier, "bss": bss, "local_score": 100000.0 * bss}


def cluster_ci(y: np.ndarray, base: np.ndarray, candidate: np.ndarray, pitcher: np.ndarray, seed: int) -> dict[str, float | int]:
    gain = (y - base) ** 2 - (y - candidate) ** 2
    groups = pd.DataFrame({"pitcher": pitcher.astype(str), "gain": gain}).groupby("pitcher", sort=True)["gain"].agg(["sum", "count"])
    sums = groups["sum"].to_numpy(float)
    counts = groups["count"].to_numpy(float)
    draws = np.random.default_rng(seed).integers(0, len(groups), size=(REPS, len(groups)))
    samples = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {"clusters": int(len(groups)), "repetitions": REPS, "seed": seed, "brier_gain": float(gain.mean()), "ci_low": float(np.quantile(samples, .025)), "ci_high": float(np.quantile(samples, .975))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=CONFIG)
    args = parser.parse_args()
    config = CONFIG[args.experiment]
    experiment_id = str(config["id"])
    out = ROOT / "model" / experiment_id
    started = time.time()
    preflight = json.loads((out / "preflight_report.json").read_text(encoding="utf-8"))
    if preflight["status"] != "AUDIT_VERIFIED" or preflight["mismatch_count"] != 0:
        raise RuntimeError("preflight must be AUDIT_VERIFIED")

    source = pd.read_csv(DIAG / "diagnostic_rows.csv", dtype={"row_id": str, "game_type": str, "pitcher_id": str})
    output = source[["row_id", "season", "game_type", "pitcher_id", "target"]].copy()
    output["base_prediction"] = source["prediction_current"].to_numpy(float)
    output["candidate_prediction"] = source[str(config["column"])].to_numpy(float)
    output.to_csv(out / "oof_predictions.csv", index=False)

    years = (2022, 2023, 2024)
    base_metrics: dict[str, object] = {}
    candidate_metrics: dict[str, object] = {}
    cis: dict[str, object] = {}
    for year in years:
        part = output.loc[output["season"].eq(year)]
        y = part["target"].to_numpy(float)
        base = part["base_prediction"].to_numpy(float)
        candidate = part["candidate_prediction"].to_numpy(float)
        base_metrics[str(year)] = metric(y, base)
        candidate_metrics[str(year)] = metric(y, candidate)
        cis[str(year)] = cluster_ci(y, base, candidate, part["pitcher_id"].astype(str).to_numpy(), SEED + year)

    y_all = output["target"].to_numpy(float)
    base_all = output["base_prediction"].to_numpy(float)
    candidate_all = output["candidate_prediction"].to_numpy(float)
    base_metrics["pooled"] = metric(y_all, base_all)
    candidate_metrics["pooled"] = metric(y_all, candidate_all)
    base_metrics["worst_season_bss"] = min(float(base_metrics[str(year)]["bss"]) for year in years)  # type: ignore[index]
    candidate_metrics["worst_season_bss"] = min(float(candidate_metrics[str(year)]["bss"]) for year in years)  # type: ignore[index]
    season_gain = {str(year): float(base_metrics[str(year)]["brier"] - candidate_metrics[str(year)]["brier"]) for year in years}  # type: ignore[index]
    pooled_gain = float(base_metrics["pooled"]["brier"] - candidate_metrics["pooled"]["brier"])  # type: ignore[index]
    worst_gain = float(candidate_metrics["worst_season_bss"] - base_metrics["worst_season_bss"])
    gates = {f"{year}_brier_gain_positive": season_gain[str(year)] > 0 for year in years}
    gates["pooled_brier_gain_positive"] = pooled_gain > 0
    gates["worst_season_bss_gain_positive"] = worst_gain > 0
    gates.update({f"{year}_cluster_ci_low_positive": float(cis[str(year)]["ci_low"]) > 0 for year in years})  # type: ignore[index]

    f_slice: dict[str, object] | None = None
    if args.experiment == "f-regime":
        f_slice = {}
        for year in years:
            part = output.loc[output["season"].eq(year) & output["game_type"].eq("F")]
            y = part["target"].to_numpy(float)
            base = part["base_prediction"].to_numpy(float)
            candidate = part["candidate_prediction"].to_numpy(float)
            base_metric = metric(y, base)
            candidate_metric = metric(y, candidate)
            ci = cluster_ci(y, base, candidate, part["pitcher_id"].astype(str).to_numpy(), SEED + 10000 + year)
            gain = float(base_metric["brier"] - candidate_metric["brier"])
            f_slice[str(year)] = {"base": base_metric, "candidate": candidate_metric, "brier_gain": gain, "cluster_ci": ci}
            gates[f"f_slice_{year}_brier_gain_positive"] = gain > 0
            gates[f"f_slice_{year}_cluster_ci_low_positive"] = float(ci["ci_low"]) > 0

    if len(gates) != int(config["gates"]):
        raise RuntimeError(f"gate count mismatch: {len(gates)}")
    candidate_result = {
        "name": config["name"], "metrics": candidate_metrics,
        "season_brier_gain": season_gain, "pooled_brier_gain": pooled_gain,
        "worst_season_bss_gain": worst_gain, "cluster_ci": cis,
        "f_slice": f_slice, "mean_abs_change": float(np.mean(np.abs(candidate_all - base_all))),
        "mean_change": float(np.mean(candidate_all - base_all)),
        "gate_checks": gates, "promotion_pass": all(gates.values()),
    }
    result = {
        "experiment_id": experiment_id, "candidate_status": "PENDING_AUDIT",
        "source_official_score": 1068.25021, "base_experiment": "REF4-OOF-DIAG-034A",
        "base_metrics": base_metrics, "candidate": candidate_result,
        "candidate_count": 1, "actual_leaf_count": 1,
        "gate_checks_count": len(gates), "model_count": 0, "oof_rows": len(output),
        "test_inference_performed": False, "full_train_performed": False,
        "zip_created": False, "elapsed_seconds": time.time() - started,
    }
    write_json(out / "result.json", result)
    embedded = json.dumps({"candidate_status": result["candidate_status"], "base_metrics": base_metrics, "candidate": candidate_result, "candidate_count": 1, "model_count": 0, "oof_rows": len(output)}, sort_keys=True)
    lines = [
        f"# {experiment_id}", "", "- status: `PENDING_AUDIT`", "- candidate/leaf: `1/1`",
        f"- performance gates: `{len(gates)}`", "- models: `0`", "- test/full-train/ZIP: `false/false/false`", "",
        "| year | Brier gain | cluster CI low | cluster CI high |", "|---|---:|---:|---:|",
    ]
    for year in years:
        lines.append(f"| {year} | {season_gain[str(year)]:+.12g} | {float(cis[str(year)]['ci_low']):+.12g} | {float(cis[str(year)]['ci_high']):+.12g} |")  # type: ignore[index]
    lines.extend(["", f"- pooled Brier gain: `{pooled_gain:+.12g}`", f"- worst-season BSS gain: `{worst_gain:+.12g}`", f"- pre-audit promotion: `{str(candidate_result['promotion_pass']).lower()}`", "", "<!-- RESULT_JSON_BEGIN", embedded, "RESULT_JSON_END -->"])
    (out / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    artifact_paths = [
        ROOT / "scripts" / "preflight_ref4_oof_confirmation.py",
        ROOT / "scripts" / "run_ref4_oof_confirmation.py",
        ROOT / "scripts" / "verify_ref4_oof_confirmation.py",
        out / "preflight_report.json", out / "preflight_report.md",
        out / "oof_predictions.csv", out / "result.json", out / "result.md",
        DIAG / "audit_manifest.json", DIAG / "validation_report.json",
        DIAG / "audit_attestation.json", DIAG / "diagnostic_rows.csv",
        ROOT / "model" / "REF4-CHAMPION-STACK-030" / "manifest.json",
        ROOT / "model" / "REF4-CHAMPION-STACK-030" / "f_regime_meta.json",
        ROOT / "data" / "train.csv", ROOT / "start03_reference.md",
        ROOT / "01_제약과금지사항.md", ROOT / "output" / "submit_ref4_champion_030.zip",
    ]
    records = {str(path.relative_to(ROOT)): {"sha256": sha256_path(path), "size": path.stat().st_size} for path in artifact_paths}
    audit = {
        "experiment_id": experiment_id, "status": "PENDING_VALIDATION",
        "artifact_count": len(records), "artifacts": records, "model_count": 0,
        "oof_rows": len(output), "candidate_count": 1, "leaf_count": 1,
        "gate_checks_count": len(gates),
    }
    write_json(out / "audit_manifest.json", audit)
    print(json.dumps({"experiment_id": experiment_id, "status": "PENDING_AUDIT", "candidate_count": 1, "gate_checks_count": len(gates), "promotion_pass": candidate_result["promotion_pass"], "passed_gates": sum(gates.values()), "model_count": 0, "oof_rows": len(output), "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
