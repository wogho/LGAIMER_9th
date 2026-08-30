#!/usr/bin/env python3
"""Evaluate the single predeclared F-Regime 0.75 recombination candidate."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "REF4-F-REGIME-032A"
OUT = ROOT / "model" / EXPERIMENT_ID
BASE = ROOT / "model" / "REF4-EXACT-OOF-031A"
SOURCE = ROOT / "model" / "REF4-CHAMPION-STACK-030"
REPS = 2000
SEED = 320200


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""): digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp"); tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"); tmp.replace(path)


def prediction(oof: pd.DataFrame, regime: dict[str, float], manifest: dict[str, object]) -> np.ndarray:
    futures = oof["game_type"].eq("F").to_numpy()
    p0g = oof["p_v2_global"].to_numpy(float); p0f = oof["p_v2_f"].to_numpy(float)
    p1g = oof["p_v3_55_global"].to_numpy(float); p1f = oof["p_v3_55_f"].to_numpy(float)
    p2g = oof["p_v3_30_global"].to_numpy(float); p2a = oof["p_v3_30_f_all"].to_numpy(float); p2r = oof["p_v3_30_f_recent"].to_numpy(float)
    p0 = np.where(futures, p0g + regime["v2_scale"] * (p0f - p0g), p0g)
    p1 = np.where(futures, p1g + regime["v355_scale"] * (p1f - p1g), p1g)
    inner = p2g + regime["v330_recent_inner_scale"] * (p2r - p2g)
    f30 = regime["v330_all_weight"] * p2a + (1 - regime["v330_all_weight"]) * inner
    p2 = np.where(futures, p2g + regime["v330_scale"] * (f30 - p2g), p2g)
    risks = []
    for name in ("middle", "wild", "reverse"):
        rg = oof[f"risk_{name}_global"].to_numpy(float); rf = oof[f"risk_{name}_f"].to_numpy(float)
        risks.append(np.where(futures, rg + regime["subtype_scale"] * (rf - rg), rg))
    main = np.average(np.vstack([p0, p1, p2]), axis=0, weights=np.asarray(manifest["main_weights"], float))
    no_shift = float(manifest["stack_intercept"]) + np.column_stack([main, *risks]) @ np.asarray(manifest["stack_coefficients"], float)
    return np.clip(no_shift + float(manifest["global_shift"]), 1e-5, 1 - 1e-5)


def metric(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    rate = float(y.mean()); brier = float(np.mean((y - p) ** 2)); bss = float(1 - brier / (rate * (1 - rate)))
    return {"rows": int(len(y)), "target_rate": rate, "brier": brier, "bss": bss, "local_score": 100000 * bss}


def cluster_ci(y: np.ndarray, base: np.ndarray, candidate: np.ndarray, pitchers: np.ndarray, seed: int) -> dict[str, float | int]:
    gain = (y - base) ** 2 - (y - candidate) ** 2
    grouped = pd.DataFrame({"pitcher": pitchers.astype(str), "gain": gain}).groupby("pitcher", sort=True)["gain"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float); counts = grouped["count"].to_numpy(float)
    draws = np.random.default_rng(seed).integers(0, len(grouped), size=(REPS, len(grouped)))
    sample = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {"clusters": int(len(grouped)), "repetitions": REPS, "seed": seed, "brier_gain": float(gain.mean()), "ci_low": float(np.quantile(sample, .025)), "ci_high": float(np.quantile(sample, .975))}


def main() -> None:
    started = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    preflight = json.loads((OUT / "preflight_report.json").read_text(encoding="utf-8"))
    if preflight["status"] != "AUDIT_VERIFIED" or preflight["mismatch_count"] != 0: raise RuntimeError("032A preflight not verified")
    current = {key: float(value) for key, value in preflight["current_regime"].items()}
    candidate_regime = {key: float(value) for key, value in preflight["candidate_regime"].items()}
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    oof = pd.read_csv(BASE / "oof_predictions.csv", dtype={"row_id": str, "game_type": str, "pitcher_id": str})
    base_prediction = prediction(oof, current, manifest); stored_base = oof["prediction"].to_numpy(float)
    if float(np.max(np.abs(base_prediction - stored_base))) > 1e-12: raise RuntimeError("current regime does not reproduce frozen base")
    candidate_prediction = prediction(oof, candidate_regime, manifest)
    output = pd.DataFrame({"row_id": oof["row_id"], "season": oof["season"], "game_type": oof["game_type"], "pitcher_id": oof["pitcher_id"], "target": oof["target"], "base_prediction": base_prediction, "f_regime_075_prediction": candidate_prediction})
    output.to_csv(OUT / "oof_predictions.csv", index=False)

    base_metrics: dict[str, object] = {}; candidate_metrics: dict[str, object] = {}; cis: dict[str, object] = {}
    for year in (2023, 2024):
        mask = oof["season"].eq(year).to_numpy(); y = oof.loc[mask, "target"].to_numpy(float)
        base_metrics[str(year)] = metric(y, base_prediction[mask]); candidate_metrics[str(year)] = metric(y, candidate_prediction[mask])
        cis[str(year)] = cluster_ci(y, base_prediction[mask], candidate_prediction[mask], oof.loc[mask, "pitcher_id"].astype(str).to_numpy(), SEED + year)
    y_all = oof["target"].to_numpy(float)
    base_metrics["pooled"] = metric(y_all, base_prediction); candidate_metrics["pooled"] = metric(y_all, candidate_prediction)
    base_metrics["worst_season_bss"] = min(float(base_metrics[str(y)]["bss"]) for y in (2023, 2024))  # type: ignore[index]
    candidate_metrics["worst_season_bss"] = min(float(candidate_metrics[str(y)]["bss"]) for y in (2023, 2024))  # type: ignore[index]
    season_gain = {str(year): float(base_metrics[str(year)]["brier"] - candidate_metrics[str(year)]["brier"]) for year in (2023, 2024)}  # type: ignore[index]
    pooled_gain = float(base_metrics["pooled"]["brier"] - candidate_metrics["pooled"]["brier"])  # type: ignore[index]
    worst_gain = float(candidate_metrics["worst_season_bss"] - base_metrics["worst_season_bss"])
    gates = {"2023_brier_gain_positive": season_gain["2023"] > 0, "2024_brier_gain_positive": season_gain["2024"] > 0, "pooled_brier_gain_positive": pooled_gain > 0, "worst_season_bss_gain_positive": worst_gain > 0, "2023_cluster_ci_low_positive": float(cis["2023"]["ci_low"]) > 0, "2024_cluster_ci_low_positive": float(cis["2024"]["ci_low"]) > 0}  # type: ignore[index]
    candidate = {"name": "f_regime_075", "regime": candidate_regime, "metrics": candidate_metrics, "season_brier_gain": season_gain, "pooled_brier_gain": pooled_gain, "worst_season_bss_gain": worst_gain, "cluster_ci": cis, "mean_abs_change": float(np.mean(np.abs(candidate_prediction - base_prediction))), "mean_change": float(np.mean(candidate_prediction - base_prediction)), "gate_checks": gates, "promotion_pass": all(gates.values())}
    result = {"experiment_id": EXPERIMENT_ID, "candidate_status": "PENDING_AUDIT", "source_official_score": 1068.25021, "base_experiment": "REF4-EXACT-OOF-031A", "base_regime": current, "base_metrics": base_metrics, "candidate": candidate, "candidate_count": 1, "actual_leaf_count": 1, "gate_checks_count": 6, "model_count": 0, "oof_rows": len(output), "test_inference_performed": False, "full_train_performed": False, "zip_created": False, "elapsed_seconds": time.time() - started}
    write_json(OUT / "result.json", result)
    embedded = json.dumps({"candidate_status": result["candidate_status"], "base_metrics": base_metrics, "candidate": candidate, "candidate_count": 1, "model_count": 0, "oof_rows": len(output)}, sort_keys=True)
    lines = [f"# {EXPERIMENT_ID}", "", "- status: `PENDING_AUDIT`", "- candidate count: `1`", "- models trained: `0`", "- test/full-train/ZIP: `false/false/false`", "", "| candidate | 2023 Brier gain | 2024 Brier gain | pooled Brier gain | worst BSS gain | pre-audit pass |", "|---|---:|---:|---:|---:|---|", f"| f_regime_075 | {season_gain['2023']:.12g} | {season_gain['2024']:.12g} | {pooled_gain:.12g} | {worst_gain:.12g} | {str(candidate['promotion_pass']).lower()} |", "", "<!-- RESULT_JSON_BEGIN", embedded, "RESULT_JSON_END -->"]
    (OUT / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    artifact_paths = [ROOT / "scripts" / "preflight_ref4_f_regime_032a.py", ROOT / "scripts" / "run_ref4_f_regime_032a.py", ROOT / "scripts" / "verify_ref4_f_regime_032a.py", OUT / "preflight_report.json", OUT / "preflight_report.md", OUT / "oof_predictions.csv", OUT / "result.json", OUT / "result.md", BASE / "audit_manifest.json", BASE / "validation_report.json", BASE / "audit_attestation.json", BASE / "oof_predictions.csv", SOURCE / "manifest.json", SOURCE / "f_regime_meta.json", ROOT / "data" / "train.csv", ROOT / "start03_reference.md", ROOT / "01_제약과금지사항.md", ROOT / "output" / "submit_ref4_champion_030.zip"]
    records = {str(path.relative_to(ROOT)): {"sha256": sha256_path(path), "size": path.stat().st_size} for path in artifact_paths}
    audit = {"experiment_id": EXPERIMENT_ID, "status": "PENDING_VALIDATION", "artifact_count": len(records), "artifacts": records, "model_count": 0, "oof_rows": len(output), "candidate_count": 1, "leaf_count": 1, "gate_checks_count": 6}
    write_json(OUT / "audit_manifest.json", audit)
    print(json.dumps({"status": "PENDING_AUDIT", "candidate_count": 1, "promotion_pass": candidate["promotion_pass"], "model_count": 0, "oof_rows": len(output), "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__": main()
