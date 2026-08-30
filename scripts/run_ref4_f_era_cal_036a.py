#!/usr/bin/env python3
"""Fit one 2023-F affine Brier calibrator and evaluate it on 2024."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "REF4-F-ERA-CAL-036A"
OUT = ROOT / "model" / EXPERIMENT_ID
BASE = ROOT / "model" / "REF4-EXACT-OOF-031A"
OOF22 = ROOT / "model" / "REF4-ADAPTIVE-GATE-031B"
YEARS = (2022, 2023, 2024)


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
    return {"rows": int(len(y)), "target_rate": rate, "prediction_mean": float(prediction.mean()), "mean_bias": float(prediction.mean() - rate), "absolute_mean_bias": abs(float(prediction.mean() - rate)), "brier": brier, "bss": bss, "local_score": 100000.0 * bss}


def ece(y: np.ndarray, prediction: np.ndarray) -> float:
    bins = np.minimum((prediction * 10).astype(int), 9)
    total = len(y)
    value = 0.0
    for bin_id in range(10):
        mask = bins == bin_id
        if mask.any():
            value += float(mask.sum() / total) * abs(float(prediction[mask].mean() - y[mask].mean()))
    return float(value)


def cluster_ci(y: np.ndarray, base: np.ndarray, candidate: np.ndarray, pitcher: np.ndarray, seed: int) -> dict[str, float | int]:
    gain = (y - base) ** 2 - (y - candidate) ** 2
    groups = pd.DataFrame({"pitcher": pitcher.astype(str), "gain": gain}).groupby("pitcher", sort=True)["gain"].agg(["sum", "count"])
    sums = groups["sum"].to_numpy(float)
    counts = groups["count"].to_numpy(float)
    draws = np.random.default_rng(seed).integers(0, len(groups), size=(2000, len(groups)))
    samples = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {"clusters": int(len(groups)), "repetitions": 2000, "seed": seed, "brier_gain": float(gain.mean()), "ci_low": float(np.quantile(samples, .025)), "ci_high": float(np.quantile(samples, .975))}


def main() -> None:
    started = time.time()
    preflight = json.loads((OUT / "preflight_report.json").read_text(encoding="utf-8"))
    contract = json.loads((OUT / "gate_contract.json").read_text(encoding="utf-8"))
    if preflight["status"] != "AUDIT_VERIFIED" or preflight["mismatch_count"] != 0:
        raise RuntimeError("036A preflight must be AUDIT_VERIFIED")
    paths = {2022: OOF22 / "oof_2022.csv", 2023: BASE / "oof_2023.csv", 2024: BASE / "oof_2024.csv"}
    frames = {year: pd.read_csv(path, dtype={"row_id": str, "game_type": str, "pitcher_id": str}) for year, path in paths.items()}
    fit = frames[2023].loc[frames[2023]["game_type"].eq("F")]
    design = np.column_stack([np.ones(len(fit)), fit["prediction"].to_numpy(float)])
    coefficients, residuals, rank, singular_values = np.linalg.lstsq(design, fit["target"].to_numpy(float), rcond=None)
    calibrator = {
        "experiment_id": EXPERIMENT_ID, "candidate_name": contract["candidate_names"][0],
        "fit_year": 2023, "fit_game_type": "F", "fit_rows": int(len(fit)),
        "intercept": float(coefficients[0]), "slope": float(coefficients[1]),
        "rank": int(rank), "singular_values": [float(value) for value in singular_values],
        "residual_sum_squares": [float(value) for value in residuals],
        "clip_low": float(contract["clip"][0]), "clip_high": float(contract["clip"][1]),
        "source_oof_sha256": sha256_path(paths[2023]),
    }
    write_json(OUT / "calibrator.json", calibrator)

    parts: list[pd.DataFrame] = []
    for year in YEARS:
        source = frames[year]
        base = source["prediction"].to_numpy(float)
        candidate = base.copy()
        mask = source["game_type"].eq("F").to_numpy()
        candidate[mask] = np.clip(calibrator["intercept"] + calibrator["slope"] * base[mask], calibrator["clip_low"], calibrator["clip_high"])
        role = {2022: "HISTORICAL_COUNTERFACTUAL_NOT_GATE", 2023: "IN_SAMPLE_FIT_DIAGNOSTIC_NOT_GATE", 2024: "NESTED_VALIDATION_GATE"}[year]
        parts.append(pd.DataFrame({"row_id": source["row_id"].astype(str), "season": year, "game_type": source["game_type"].astype(str), "pitcher_id": source["pitcher_id"].astype(str), "target": source["target"].to_numpy(float), "base_prediction": base, "candidate_prediction": candidate, "evaluation_role": role}))
    output = pd.concat(parts, ignore_index=True)
    output.to_csv(OUT / "oof_predictions.csv", index=False)

    overall: dict[str, object] = {}
    f_slice: dict[str, object] = {}
    for year in YEARS:
        year_part = output.loc[output["season"].eq(year)]
        for destination, part in ((overall, year_part), (f_slice, year_part.loc[year_part["game_type"].eq("F")])):
            y = part["target"].to_numpy(float)
            base = part["base_prediction"].to_numpy(float)
            candidate = part["candidate_prediction"].to_numpy(float)
            destination[str(year)] = {"base": metric(y, base), "candidate": metric(y, candidate), "brier_gain": float(np.mean((y - base) ** 2) - np.mean((y - candidate) ** 2)), "base_ece": ece(y, base), "candidate_ece": ece(y, candidate), "ece_gain": float(ece(y, base) - ece(y, candidate))}
    valid = output.loc[output["season"].eq(2024)]
    valid_f = valid.loc[valid["game_type"].eq("F")]
    ci_all = cluster_ci(valid["target"].to_numpy(float), valid["base_prediction"].to_numpy(float), valid["candidate_prediction"].to_numpy(float), valid["pitcher_id"].astype(str).to_numpy(), 360200)
    ci_f = cluster_ci(valid_f["target"].to_numpy(float), valid_f["base_prediction"].to_numpy(float), valid_f["candidate_prediction"].to_numpy(float), valid_f["pitcher_id"].astype(str).to_numpy(), 360201)
    subchecks = {
        "2024_all_brier_gain_positive": float(overall["2024"]["brier_gain"]) > 0,
        "2024_f_brier_gain_positive": float(f_slice["2024"]["brier_gain"]) > 0,
        "2024_all_cluster_ci_low_positive": float(ci_all["ci_low"]) > 0,
        "2024_f_cluster_ci_low_positive": float(ci_f["ci_low"]) > 0,
        "2024_f_absolute_mean_bias_reduced": float(f_slice["2024"]["candidate"]["absolute_mean_bias"]) < float(f_slice["2024"]["base"]["absolute_mean_bias"]),
        "2024_f_ece_reduced": float(f_slice["2024"]["candidate_ece"]) < float(f_slice["2024"]["base_ece"]),
    }
    if len(subchecks) != contract["promotion_subcheck_count"]:
        raise RuntimeError("promotion subcheck count mismatch")
    composite_gate = {"all_six_subchecks": all(subchecks.values())}
    candidate = {
        "candidate_name": contract["candidate_names"][0], "calibrator": calibrator,
        "overall": overall, "f_slice": f_slice, "cluster_ci": {"2024_all": ci_all, "2024_f": ci_f},
        "promotion_subchecks": subchecks, "promotion_subcheck_count": len(subchecks),
        "passed_subcheck_count": sum(subchecks.values()), "composite_gate": composite_gate,
        "promotion_pass": all(composite_gate.values()),
    }
    result = {
        "experiment_id": EXPERIMENT_ID, "candidate_status": "PENDING_AUDIT",
        "promotion_scope": "EVAL_PASS/HOLD_FOR_FULLTRAIN_APPROVAL" if candidate["promotion_pass"] else "FAIL/HOLD",
        "source_official_score": 1068.25021, "candidate": candidate,
        "candidate_count": len(contract["candidate_names"]), "actual_leaf_count": len(contract["candidate_names"]),
        "gate_checks_count": len(composite_gate), "promotion_subcheck_count": len(subchecks),
        "model_count": 1, "oof_rows": len(output), "test_inference_performed": False,
        "full_train_performed": False, "zip_created": False, "elapsed_seconds": time.time() - started,
    }
    write_json(OUT / "result.json", result)
    embedded_value = {key: value for key, value in result.items() if key != "elapsed_seconds"}
    embedded = json.dumps(embedded_value, sort_keys=True)
    lines = [f"# {EXPERIMENT_ID}", "", "- status: `PENDING_AUDIT`", f"- promotion scope: `{result['promotion_scope']}`", "- candidate/leaf/composite-gate/subcheck/model: `1/1/1/6/1`", "- test/full-train/ZIP: `false/false/false`", "", "| scope | base Brier | candidate Brier | gain |", "|---|---:|---:|---:|", f"| 2024 all | {overall['2024']['base']['brier']:.12g} | {overall['2024']['candidate']['brier']:.12g} | {overall['2024']['brier_gain']:+.12g} |", f"| 2024 F | {f_slice['2024']['base']['brier']:.12g} | {f_slice['2024']['candidate']['brier']:.12g} | {f_slice['2024']['brier_gain']:+.12g} |", "", "<!-- RESULT_JSON_BEGIN", embedded, "RESULT_JSON_END -->"]
    (OUT / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    artifact_paths = [ROOT / "scripts" / "preflight_ref4_f_era_cal_036a.py", ROOT / "scripts" / "run_ref4_f_era_cal_036a.py", ROOT / "scripts" / "verify_ref4_f_era_cal_036a.py", OUT / "gate_contract.json", OUT / "preflight_report.json", OUT / "preflight_report.md", OUT / "calibrator.json", OUT / "oof_predictions.csv", OUT / "result.json", OUT / "result.md", BASE / "audit_manifest.json", BASE / "audit_attestation.json", BASE / "oof_2023.csv", BASE / "oof_2024.csv", OOF22 / "audit_manifest.json", OOF22 / "audit_attestation.json", OOF22 / "oof_2022.csv", ROOT / "model" / "REF4-F-DRIFT-DIAG-035A" / "audit_manifest.json", ROOT / "model" / "REF4-F-DRIFT-DIAG-035A" / "audit_attestation.json", ROOT / "data" / "train.csv", ROOT / "start03_reference.md", ROOT / "01_제약과금지사항.md", ROOT / "output" / "submit_ref4_champion_030.zip"]
    records = {str(path.relative_to(ROOT)): {"sha256": sha256_path(path), "size": path.stat().st_size} for path in artifact_paths}
    audit = {"experiment_id": EXPERIMENT_ID, "status": "PENDING_VALIDATION", "artifact_count": len(records), "artifacts": records, "candidate_count": result["candidate_count"], "leaf_count": result["actual_leaf_count"], "gate_checks_count": result["gate_checks_count"], "promotion_subcheck_count": result["promotion_subcheck_count"], "model_count": result["model_count"], "oof_rows": result["oof_rows"]}
    write_json(OUT / "audit_manifest.json", audit)
    print(json.dumps({"experiment_id": EXPERIMENT_ID, "status": "PENDING_AUDIT", "promotion_scope": result["promotion_scope"], "candidate_count": result["candidate_count"], "gate_checks_count": result["gate_checks_count"], "promotion_subcheck_count": result["promotion_subcheck_count"], "passed_subcheck_count": candidate["passed_subcheck_count"], "model_count": 1, "oof_rows": len(output), "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
