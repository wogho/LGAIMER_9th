#!/usr/bin/env python3
"""Strict-forward evaluation of one frozen R-only 113A reliability curve."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model/REF4-R-CROSSFIT-RELIABILITY-CURVE-121A"
TRAIN = ROOT / "data/train.csv"
ANCHOR = ROOT / "model/REF4-113A-V66-NESTED-117A/oof_predictions.csv"
CHAMPION = ROOT / "output/submit_ref4_super_ensemble_113A.zip"
CONTRACT = OUT / "audit_contract.json"
PREFLIGHT = OUT / "preflight_report.json"
TARGET = "control_success"
N_BINS = 8
SHRINKAGE = 300.0
BLEND_WEIGHT = 0.25
RAW_DELTA_CAP = 0.12
CALIBRATOR_CLIP = (0.005, 0.995)
MAX_FINAL_CHANGE = 0.03
BOOTSTRAP_REPEATS = 10_000
BOOTSTRAP_SEED = 121_2024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.square(p - y)))


def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    index = np.minimum((np.clip(p, 0.0, 1.0) * bins).astype(int), bins - 1)
    total = 0.0
    for group in range(bins):
        mask = index == group
        if mask.any():
            total += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return total


def fit_curve(source: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, float]:
    if not source["game_type"].eq("R").all():
        raise RuntimeError("curve source contains non-Regular rows")
    p = source["p113a_strict"].to_numpy(float)
    y = source["target"].to_numpy(float)
    inner = np.quantile(p, np.linspace(0.0, 1.0, N_BINS + 1)[1:-1])
    if len(np.unique(inner)) != N_BINS - 1:
        raise RuntimeError("non-unique probability quantile boundaries")
    bins = np.searchsorted(inner, p, side="right")
    residual = y - p
    center = float(residual.mean())
    centered = residual - center
    frame = pd.DataFrame({"bin": bins, "centered_residual": centered})
    curve = frame.groupby("bin", sort=True)["centered_residual"].agg(["sum", "size", "mean"]).reindex(range(N_BINS))
    if curve.isna().any().any():
        raise RuntimeError("empty reliability bin")
    curve["raw_delta"] = curve["sum"] / (curve["size"] + SHRINKAGE)
    curve["raw_delta"] = curve["raw_delta"].clip(-RAW_DELTA_CAP, RAW_DELTA_CAP)
    curve["lower"] = np.r_[-np.inf, inner]
    curve["upper"] = np.r_[inner, np.inf]
    curve.index.name = "bin"
    return inner, curve.reset_index(), center


def apply_curve(p: np.ndarray, inner: np.ndarray, curve: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bins = np.searchsorted(inner, p, side="right")
    delta_by_bin = curve.sort_values("bin")["raw_delta"].to_numpy(float)
    raw_delta = delta_by_bin[bins]
    calibrated = np.clip(p + raw_delta, *CALIBRATOR_CLIP)
    candidate = (1.0 - BLEND_WEIGHT) * p + BLEND_WEIGHT * calibrated
    return candidate, raw_delta, calibrated


def gain_metrics(y: np.ndarray, baseline: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    base = brier(y, baseline)
    cand = brier(y, candidate)
    return {
        "rows": int(len(y)),
        "baseline_brier": base,
        "candidate_brier": cand,
        "brier_gain": base - cand,
        "baseline_ece15": ece(y, baseline),
        "candidate_ece15": ece(y, candidate),
        "ece15_gain": ece(y, baseline) - ece(y, candidate),
        "mean_absolute_change": float(np.mean(np.abs(candidate - baseline))),
        "max_absolute_change": float(np.max(np.abs(candidate - baseline))),
    }


def grouped_metrics(rows: pd.DataFrame, baseline: np.ndarray, candidate: np.ndarray) -> dict[str, object]:
    y = rows["target"].to_numpy(float)
    result: dict[str, object] = {"overall": gain_metrics(y, baseline, candidate), "game_type": {}, "halves": {}, "quarters": {}}
    for kind in ("R", "F"):
        mask = rows["game_type"].eq(kind).to_numpy()
        result["game_type"][kind] = gain_metrics(y[mask], baseline[mask], candidate[mask])
    half_masks = {"H1": rows["game_month"].le(6).to_numpy(), "H2": rows["game_month"].ge(7).to_numpy()}
    quarter_masks = {
        "Q1_m_le_4": rows["game_month"].le(4).to_numpy(),
        "Q2_m_5_6": rows["game_month"].between(5, 6).to_numpy(),
        "Q3_m_7_8": rows["game_month"].between(7, 8).to_numpy(),
        "Q4_m_ge_9": rows["game_month"].ge(9).to_numpy(),
    }
    for name, mask in half_masks.items():
        result["halves"][name] = gain_metrics(y[mask], baseline[mask], candidate[mask])
    for name, mask in quarter_masks.items():
        if not mask.any():
            raise RuntimeError(f"empty quarter: {name}")
        result["quarters"][name] = gain_metrics(y[mask], baseline[mask], candidate[mask])
    return result


def pitcher_bootstrap(rows: pd.DataFrame, baseline: np.ndarray, candidate: np.ndarray) -> dict[str, float | int]:
    y = rows["target"].to_numpy(float)
    gain = np.square(baseline - y) - np.square(candidate - y)
    grouped = pd.DataFrame({"pitcher": rows["pitcher_id"].astype(str), "gain": gain}).groupby("pitcher", sort=False)["gain"].agg(["sum", "size"])
    sums = grouped["sum"].to_numpy(float)
    sizes = grouped["size"].to_numpy(float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    values = np.empty(BOOTSTRAP_REPEATS, dtype=float)
    for start in range(0, BOOTSTRAP_REPEATS, 64):
        count = min(64, BOOTSTRAP_REPEATS - start)
        sample = rng.integers(0, len(grouped), size=(count, len(grouped)))
        values[start : start + count] = sums[sample].sum(axis=1) / sizes[sample].sum(axis=1)
    return {
        "repeats": BOOTSTRAP_REPEATS,
        "seed": BOOTSTRAP_SEED,
        "pitcher_clusters": int(len(grouped)),
        "mean_gain": float(values.mean()),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "positive_fraction": float(np.mean(values > 0.0)),
    }


def report_payload(result: dict[str, object]) -> dict[str, object]:
    return {
        "experiment_id": result["experiment_id"],
        "performance_gate_pass": result["performance_gate_pass"],
        "decision": result["decision"],
        "gates": result["gates"],
        "fold_summary": {
            year: {
                "overall_gain": fold["metrics"]["overall"]["brier_gain"],
                "R_gain": fold["metrics"]["game_type"]["R"]["brier_gain"],
                "F_max_abs_diff": fold["F_max_abs_difference"],
            }
            for year, fold in result["folds"].items()
        },
    }


def make_markdown(result: dict[str, object]) -> str:
    payload = report_payload(result)
    rows = []
    for year, fold in result["folds"].items():
        rows.append(
            f"| {year} | {fold['metrics']['overall']['brier_gain']:.12f} | "
            f"{fold['metrics']['game_type']['R']['brier_gain']:.12f} | "
            f"{fold['F_max_abs_difference']:.3e} |"
        )
    return "\n".join(
        [
            "# REF4-R-CROSSFIT-RELIABILITY-CURVE-121A",
            "",
            f"- Performance gate: `{result['performance_gate_pass']}`",
            f"- Decision: `{result['decision']}`",
            "- ZIP created: `False`",
            "",
            "| Season | Overall Brier gain | R Brier gain | F max abs diff |",
            "|---:|---:|---:|---:|",
            *rows,
            "",
            "<!-- REPORT_PAYLOAD_BEGIN",
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            "REPORT_PAYLOAD_END -->",
            "",
        ]
    )


def main() -> None:
    started = time.time()
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    preflight = json.loads(PREFLIGHT.read_text(encoding="utf-8"))
    if contract.get("status") != "LOCKED_BEFORE_FORMAL_RUN":
        raise RuntimeError("121A contract is not locked")
    if preflight.get("status") != "AUDIT_VERIFIED" or preflight.get("mismatch_count") != 0:
        raise RuntimeError("121A preflight is not audit verified")
    for path, key in ((TRAIN, "train_sha256"), (ANCHOR, "strict_anchor_sha256"), (CHAMPION, "champion_zip_sha256")):
        if sha256(path) != preflight["source_provenance"][key]:
            raise RuntimeError(f"source changed after preflight: {path}")

    raw = pd.read_csv(TRAIN, usecols=["row_id", "season", "game_month", "game_type", "pitcher_id", TARGET], low_memory=False)
    anchor = pd.read_csv(ANCHOR, usecols=["row_id", "season", "game_type", "pitcher_id", "target", "p113a_strict"], low_memory=False)
    validation = raw.loc[raw["season"].isin((2022, 2023, 2024))].reset_index(drop=True)
    if not np.array_equal(validation["row_id"].astype(str), anchor["row_id"].astype(str)):
        raise RuntimeError("strict anchor row order mismatch")
    if not np.array_equal(validation[TARGET].to_numpy(float), anchor["target"].to_numpy(float)):
        raise RuntimeError("strict anchor target mismatch")
    if not np.array_equal(validation["game_type"].astype(str), anchor["game_type"].astype(str)):
        raise RuntimeError("strict anchor game_type mismatch")
    validation["target"] = anchor["target"].to_numpy(float)
    validation["p113a_strict"] = anchor["p113a_strict"].to_numpy(float)

    folds: dict[str, object] = {}
    predictions: list[pd.DataFrame] = []
    curves: list[pd.DataFrame] = []
    for year, fit_years in ((2023, (2022,)), (2024, (2022, 2023))):
        source = validation.loc[validation["season"].isin(fit_years) & validation["game_type"].eq("R")].reset_index(drop=True)
        rows = validation.loc[validation["season"].eq(year)].reset_index(drop=True)
        inner, curve, center = fit_curve(source)
        regular = rows["game_type"].eq("R").to_numpy()
        baseline = rows["p113a_strict"].to_numpy(float)
        candidate = baseline.copy()
        raw_delta = np.zeros(len(rows), dtype=float)
        calibrated = baseline.copy()
        candidate_R, delta_R, calibrated_R = apply_curve(baseline[regular], inner, curve)
        candidate[regular] = candidate_R
        raw_delta[regular] = delta_R
        calibrated[regular] = calibrated_R
        if np.max(np.abs(candidate[~regular] - baseline[~regular])) != 0.0:
            raise RuntimeError("F prediction changed")
        if np.max(np.abs(candidate - baseline)) > MAX_FINAL_CHANGE + 1e-15:
            raise RuntimeError("final change exceeds frozen cap")
        metrics = grouped_metrics(rows, baseline, candidate)
        bootstrap = pitcher_bootstrap(rows, baseline, candidate) if year == 2024 else None
        folds[str(year)] = {
            "fit_seasons": list(fit_years),
            "fit_rows_R": int(len(source)),
            "validation_rows": int(len(rows)),
            "validation_labels_used_in_fit": False,
            "source_R_residual_mean_removed": center,
            "inner_edges": inner.tolist(),
            "metrics": metrics,
            "F_max_abs_difference": float(np.max(np.abs(candidate[~regular] - baseline[~regular]))),
            "pitcher_bootstrap": bootstrap,
        }
        curve.insert(0, "validation_season", year)
        curve.insert(1, "fit_seasons", "+".join(map(str, fit_years)))
        curves.append(curve)
        predictions.append(
            pd.DataFrame(
                {
                    "row_id": rows["row_id"].astype(str),
                    "season": year,
                    "game_month": rows["game_month"].to_numpy(int),
                    "game_type": rows["game_type"].astype(str),
                    "pitcher_id": rows["pitcher_id"].astype(str),
                    "target": rows["target"].to_numpy(float),
                    "p113a_strict": baseline,
                    "raw_curve_delta": raw_delta,
                    "p_calibrator": calibrated,
                    "p121a": candidate,
                }
            )
        )

    prediction_frame = pd.concat(predictions, ignore_index=True)
    prediction_frame.to_csv(OUT / "strict_predictions.csv.gz", index=False, compression="gzip")
    curve_frame = pd.concat(curves, ignore_index=True)
    curve_frame.to_csv(OUT / "strict_curves.csv", index=False)

    final_source = validation.loc[validation["game_type"].eq("R")].reset_index(drop=True)
    final_inner, final_curve, final_center = fit_curve(final_source)
    final_curve.to_csv(OUT / "production_curve.csv", index=False)
    final_calibrator = {
        "experiment_id": "REF4-R-CROSSFIT-RELIABILITY-CURVE-121A",
        "apply_game_type": "R",
        "fit_seasons": [2022, 2023, 2024],
        "fit_rows_R": int(len(final_source)),
        "n_bins": N_BINS,
        "inner_edges": final_inner.tolist(),
        "raw_delta_by_bin": final_curve.sort_values("bin")["raw_delta"].tolist(),
        "source_R_residual_mean_removed": final_center,
        "shrinkage_k": SHRINKAGE,
        "raw_delta_cap": RAW_DELTA_CAP,
        "calibrator_clip": list(CALIBRATOR_CLIP),
        "convex_weight": BLEND_WEIGHT,
        "final_max_absolute_change": MAX_FINAL_CHANGE,
        "F_action": "identity",
    }
    (OUT / "production_calibrator.json").write_text(json.dumps(final_calibrator, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    f23 = folds["2023"]
    f24 = folds["2024"]
    gates = {
        "2023_overall_gain_at_least_0_00030": f23["metrics"]["overall"]["brier_gain"] >= 0.00030,
        "2024_overall_gain_at_least_0_00030": f24["metrics"]["overall"]["brier_gain"] >= 0.00030,
        "2024_R_gain_at_least_0_00050": f24["metrics"]["game_type"]["R"]["brier_gain"] >= 0.00050,
        "2023_all_halves_positive": min(item["brier_gain"] for item in f23["metrics"]["halves"].values()) > 0.0,
        "2024_all_halves_positive": min(item["brier_gain"] for item in f24["metrics"]["halves"].values()) > 0.0,
        "2024_all_quarters_positive": min(item["brier_gain"] for item in f24["metrics"]["quarters"].values()) > 0.0,
        "2024_bootstrap_ci_low_positive": f24["pitcher_bootstrap"]["ci_low"] > 0.0,
        "2024_ece15_not_worse": f24["metrics"]["overall"]["ece15_gain"] >= 0.0,
        "F_exact_identity": f23["F_max_abs_difference"] == 0.0 and f24["F_max_abs_difference"] == 0.0,
        "max_change_at_most_0_03": max(f23["metrics"]["overall"]["max_absolute_change"], f24["metrics"]["overall"]["max_absolute_change"]) <= MAX_FINAL_CHANGE + 1e-15,
        "finite_unit_interval": bool(np.isfinite(prediction_frame[["p113a_strict", "p_calibrator", "p121a"]]).all().all() and prediction_frame[["p113a_strict", "p_calibrator", "p121a"]].min().min() >= 0.0 and prediction_frame[["p113a_strict", "p_calibrator", "p121a"]].max().max() <= 1.0),
    }
    performance_pass = all(gates.values())
    result: dict[str, object] = {
        "experiment_id": "REF4-R-CROSSFIT-RELIABILITY-CURVE-121A",
        "status": "FORMAL_RESEARCH_COMPLETE",
        "candidate_status": "RESEARCH_GATE_PASS_PENDING_PRODUCTION_COMPATIBILITY" if performance_pass else "REJECTED_PERFORMANCE_GATE",
        "single_hypothesis_count": 1,
        "candidate_count": 1,
        "configuration": final_calibrator,
        "folds": folds,
        "gates": gates,
        "gate_count": len(gates),
        "performance_gate_pass": performance_pass,
        "decision": "RUN_PRODUCTION_COMPATIBILITY_AUDIT" if performance_pass else "REJECT_KEEP_113A",
        "test_read": False,
        "zip_created": False,
        "elapsed_seconds": float(time.time() - started),
        "output_hashes": {
            "strict_predictions_sha256": sha256(OUT / "strict_predictions.csv.gz"),
            "strict_curves_sha256": sha256(OUT / "strict_curves.csv"),
            "production_curve_sha256": sha256(OUT / "production_curve.csv"),
            "production_calibrator_sha256": sha256(OUT / "production_calibrator.json"),
        },
    }
    (OUT / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "research_report.md").write_text(make_markdown(result), encoding="utf-8")

    manifest_paths = [
        ROOT / "scripts/preflight_ref4_121a.py",
        ROOT / "scripts/run_ref4_r_crossfit_reliability_curve_121a.py",
        ROOT / "scripts/verify_ref4_r_crossfit_reliability_curve_121a.py",
        TRAIN,
        ANCHOR,
        CHAMPION,
        CONTRACT,
        PREFLIGHT,
        OUT / "strict_predictions.csv.gz",
        OUT / "strict_curves.csv",
        OUT / "production_curve.csv",
        OUT / "production_calibrator.json",
        OUT / "result.json",
        OUT / "research_report.md",
    ]
    manifest = {
        "experiment_id": result["experiment_id"],
        "created_after_result": True,
        "file_count": len(manifest_paths),
        "files": [
            {"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in manifest_paths
        ],
        "commands": [
            ".venv/bin/python scripts/preflight_ref4_121a.py",
            ".venv/bin/python scripts/run_ref4_r_crossfit_reliability_curve_121a.py",
            ".venv/bin/python scripts/verify_ref4_r_crossfit_reliability_curve_121a.py",
        ],
        "test_read": False,
        "zip_created": False,
    }
    (OUT / "audit_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"gates": gates, "performance_gate_pass": performance_pass, "folds": report_payload(result)["fold_summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
