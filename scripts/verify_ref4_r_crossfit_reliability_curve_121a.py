#!/usr/bin/env python3
"""Independent source-level audit for REF4-R-CROSSFIT-RELIABILITY-CURVE-121A."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model/REF4-R-CROSSFIT-RELIABILITY-CURVE-121A"
TRAIN = ROOT / "data/train.csv"
ANCHOR = ROOT / "model/REF4-113A-V66-NESTED-117A/oof_predictions.csv"
N_BINS = 8
K = 300.0
W = 0.25
RAW_CAP = 0.12
BOOT_N = 10_000
BOOT_SEED = 121_2024


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fit_apply(source: pd.DataFrame, rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    p_fit = source["p113a_strict"].to_numpy(float)
    residual = source["target"].to_numpy(float) - p_fit
    centered = residual - residual.mean()
    inner = np.quantile(p_fit, np.linspace(0.0, 1.0, N_BINS + 1)[1:-1])
    source_bin = np.searchsorted(inner, p_fit, side="right")
    grouped = pd.DataFrame({"bin": source_bin, "value": centered}).groupby("bin", sort=True)["value"].agg(["sum", "size", "mean"]).reindex(range(N_BINS))
    grouped["raw_delta"] = (grouped["sum"] / (grouped["size"] + K)).clip(-RAW_CAP, RAW_CAP)
    grouped["lower"] = np.r_[-np.inf, inner]
    grouped["upper"] = np.r_[inner, np.inf]
    p = rows["p113a_strict"].to_numpy(float)
    bins = np.searchsorted(inner, p, side="right")
    delta = grouped["raw_delta"].to_numpy(float)[bins]
    calibrated = np.clip(p + delta, 0.005, 0.995)
    candidate = (1.0 - W) * p + W * calibrated
    return candidate, delta, grouped.reset_index()


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.square(p - y)))


def ece(y: np.ndarray, p: np.ndarray) -> float:
    index = np.minimum((np.clip(p, 0.0, 1.0) * 15).astype(int), 14)
    value = 0.0
    for group in range(15):
        mask = index == group
        if mask.any():
            value += float(mask.mean()) * abs(float(p[mask].mean()) - float(y[mask].mean()))
    return value


def gain(y: np.ndarray, base: np.ndarray, cand: np.ndarray) -> float:
    return brier(y, base) - brier(y, cand)


def bootstrap(rows: pd.DataFrame, base: np.ndarray, cand: np.ndarray) -> dict[str, float | int]:
    y = rows["target"].to_numpy(float)
    row_gain = np.square(base - y) - np.square(cand - y)
    grouped = pd.DataFrame({"pitcher": rows["pitcher_id"].astype(str), "gain": row_gain}).groupby("pitcher", sort=False)["gain"].agg(["sum", "size"])
    sums, sizes = grouped["sum"].to_numpy(float), grouped["size"].to_numpy(float)
    rng = np.random.default_rng(BOOT_SEED)
    values = np.empty(BOOT_N)
    for start in range(0, BOOT_N, 64):
        count = min(64, BOOT_N - start)
        sample = rng.integers(0, len(grouped), size=(count, len(grouped)))
        values[start : start + count] = sums[sample].sum(axis=1) / sizes[sample].sum(axis=1)
    return {
        "repeats": BOOT_N,
        "seed": BOOT_SEED,
        "pitcher_clusters": int(len(grouped)),
        "mean_gain": float(values.mean()),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "positive_fraction": float(np.mean(values > 0.0)),
    }


def main() -> None:
    result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    manifest = json.loads((OUT / "audit_manifest.json").read_text(encoding="utf-8"))
    preflight = json.loads((OUT / "preflight_report.json").read_text(encoding="utf-8"))
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, actual: object) -> None:
        checks.append({"name": name, "passed": bool(passed), "actual": actual})

    check("preflight_status", preflight.get("status") == "AUDIT_VERIFIED", preflight.get("status"))
    check("preflight_mismatch_count", preflight.get("mismatch_count") == 0, preflight.get("mismatch_count"))
    manifest_mismatch = []
    for item in manifest["files"]:
        path = ROOT / item["path"]
        actual = sha256(path) if path.is_file() else None
        if actual != item["sha256"] or (path.stat().st_size if path.is_file() else None) != item["bytes"]:
            manifest_mismatch.append(item["path"])
    check("manifest_file_count", manifest["file_count"] == len(manifest["files"]), {"recorded": manifest["file_count"], "actual": len(manifest["files"])})
    check("manifest_hashes_and_sizes", not manifest_mismatch, manifest_mismatch)

    raw = pd.read_csv(TRAIN, usecols=["row_id", "season", "game_month", "game_type", "pitcher_id", "control_success"], low_memory=False)
    anchor = pd.read_csv(ANCHOR, usecols=["row_id", "season", "game_type", "pitcher_id", "target", "p113a_strict"], low_memory=False)
    validation = raw.loc[raw["season"].isin((2022, 2023, 2024))].reset_index(drop=True)
    check("raw_train_rows", len(raw) == 1_475_092, len(raw))
    check("raw_train_row_id_unique", raw["row_id"].nunique() == len(raw), int(raw["row_id"].nunique()))
    check("raw_target_binary_finite", bool(np.isfinite(raw["control_success"]).all() and set(raw["control_success"].unique()) == {0, 1}), sorted(raw["control_success"].unique().tolist()))
    check("anchor_rows", len(anchor) == 746_504, len(anchor))
    check("anchor_row_id_unique", anchor["row_id"].nunique() == len(anchor), int(anchor["row_id"].nunique()))
    check("anchor_row_order", np.array_equal(anchor["row_id"].astype(str), validation["row_id"].astype(str)), None)
    check("anchor_target", np.array_equal(anchor["target"].to_numpy(float), validation["control_success"].to_numpy(float)), None)
    check("anchor_game_type", np.array_equal(anchor["game_type"].astype(str), validation["game_type"].astype(str)), None)
    check("anchor_finite_unit", bool(np.isfinite(anchor[["target", "p113a_strict"]]).all().all() and anchor["p113a_strict"].between(0.0, 1.0).all()), None)
    validation["target"] = anchor["target"].to_numpy(float)
    validation["p113a_strict"] = anchor["p113a_strict"].to_numpy(float)

    stored = pd.read_csv(OUT / "strict_predictions.csv.gz", low_memory=False)
    expected_order = validation.loc[validation["season"].isin((2023, 2024)), "row_id"].astype(str).reset_index(drop=True)
    check("prediction_rows", len(stored) == len(expected_order), len(stored))
    check("prediction_row_id_unique", stored["row_id"].nunique() == len(stored), int(stored["row_id"].nunique()))
    check("prediction_row_order", np.array_equal(stored["row_id"].astype(str), expected_order), None)
    check("prediction_finite_unit", bool(np.isfinite(stored[["target", "p113a_strict", "raw_curve_delta", "p_calibrator", "p121a"]]).all().all() and stored[["p113a_strict", "p_calibrator", "p121a"]].min().min() >= 0.0 and stored[["p113a_strict", "p_calibrator", "p121a"]].max().max() <= 1.0), None)

    recomputed: dict[str, object] = {}
    formula_errors = []
    curve_errors = []
    for year, fit_years in ((2023, (2022,)), (2024, (2022, 2023))):
        source = validation.loc[validation["season"].isin(fit_years) & validation["game_type"].eq("R")].reset_index(drop=True)
        rows = validation.loc[validation["season"].eq(year)].reset_index(drop=True)
        local = stored.loc[stored["season"].eq(year)].reset_index(drop=True)
        base = rows["p113a_strict"].to_numpy(float)
        regular = rows["game_type"].eq("R").to_numpy()
        candidate = base.copy()
        raw_delta = np.zeros(len(rows))
        candidate_R, delta_R, curve = fit_apply(source, rows.loc[regular].reset_index(drop=True))
        candidate[regular] = candidate_R
        raw_delta[regular] = delta_R
        formula_errors.append(float(np.max(np.abs(candidate - local["p121a"].to_numpy(float)))))
        formula_errors.append(float(np.max(np.abs(raw_delta - local["raw_curve_delta"].to_numpy(float)))))
        stored_curve = pd.read_csv(OUT / "strict_curves.csv").loc[lambda x: x["validation_season"].eq(year)].reset_index(drop=True)
        curve_errors.append(float(np.max(np.abs(curve[["sum", "size", "mean", "raw_delta"]].to_numpy(float) - stored_curve[["sum", "size", "mean", "raw_delta"]].to_numpy(float)))))
        y = rows["target"].to_numpy(float)
        overall_gain = gain(y, base, candidate)
        r_gain = gain(y[regular], base[regular], candidate[regular])
        f_diff = float(np.max(np.abs(candidate[~regular] - base[~regular])))
        half = {
            "H1": gain(y[rows["game_month"].le(6)], base[rows["game_month"].le(6)], candidate[rows["game_month"].le(6)]),
            "H2": gain(y[rows["game_month"].ge(7)], base[rows["game_month"].ge(7)], candidate[rows["game_month"].ge(7)]),
        }
        quarter_masks = {
            "Q1_m_le_4": rows["game_month"].le(4).to_numpy(),
            "Q2_m_5_6": rows["game_month"].between(5, 6).to_numpy(),
            "Q3_m_7_8": rows["game_month"].between(7, 8).to_numpy(),
            "Q4_m_ge_9": rows["game_month"].ge(9).to_numpy(),
        }
        quarters = {name: gain(y[mask], base[mask], candidate[mask]) for name, mask in quarter_masks.items()}
        recomputed[str(year)] = {
            "overall_gain": overall_gain,
            "R_gain": r_gain,
            "F_diff": f_diff,
            "halves": half,
            "quarters": quarters,
            "ece_gain": ece(y, base) - ece(y, candidate),
            "max_change": float(np.max(np.abs(candidate - base))),
            "bootstrap": bootstrap(rows, base, candidate) if year == 2024 else None,
        }
        recorded = result["folds"][str(year)]
        check(f"metric_overall_{year}", abs(overall_gain - recorded["metrics"]["overall"]["brier_gain"]) <= 1e-15, abs(overall_gain - recorded["metrics"]["overall"]["brier_gain"]))
        check(f"metric_R_{year}", abs(r_gain - recorded["metrics"]["game_type"]["R"]["brier_gain"]) <= 1e-15, abs(r_gain - recorded["metrics"]["game_type"]["R"]["brier_gain"]))
    check("formula_reproduction", max(formula_errors) <= 1e-12, max(formula_errors))
    check("curve_reproduction", max(curve_errors) <= 1e-12, max(curve_errors))

    r23, r24 = recomputed["2023"], recomputed["2024"]
    expected_gates = {
        "2023_overall_gain_at_least_0_00030": r23["overall_gain"] >= 0.00030,
        "2024_overall_gain_at_least_0_00030": r24["overall_gain"] >= 0.00030,
        "2024_R_gain_at_least_0_00050": r24["R_gain"] >= 0.00050,
        "2023_all_halves_positive": min(r23["halves"].values()) > 0.0,
        "2024_all_halves_positive": min(r24["halves"].values()) > 0.0,
        "2024_all_quarters_positive": min(r24["quarters"].values()) > 0.0,
        "2024_bootstrap_ci_low_positive": r24["bootstrap"]["ci_low"] > 0.0,
        "2024_ece15_not_worse": r24["ece_gain"] >= 0.0,
        "F_exact_identity": r23["F_diff"] == 0.0 and r24["F_diff"] == 0.0,
        "max_change_at_most_0_03": max(r23["max_change"], r24["max_change"]) <= 0.03 + 1e-15,
        "finite_unit_interval": bool(np.isfinite(stored[["p113a_strict", "p_calibrator", "p121a"]]).all().all() and stored[["p113a_strict", "p_calibrator", "p121a"]].min().min() >= 0.0 and stored[["p113a_strict", "p_calibrator", "p121a"]].max().max() <= 1.0),
    }
    check("gate_count", result["gate_count"] == len(expected_gates) == len(result["gates"]), {"recorded": result["gate_count"], "actual": len(expected_gates)})
    check("gate_reproduction", expected_gates == result["gates"], {"expected": expected_gates, "recorded": result["gates"]})
    check("performance_decision", result["performance_gate_pass"] == all(expected_gates.values()), {"expected": all(expected_gates.values()), "recorded": result["performance_gate_pass"]})

    markdown = (OUT / "research_report.md").read_text(encoding="utf-8")
    match = re.search(r"<!-- REPORT_PAYLOAD_BEGIN\n(.*?)\nREPORT_PAYLOAD_END -->", markdown, flags=re.S)
    markdown_payload = json.loads(match.group(1)) if match else None
    expected_payload = {
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
    check("json_markdown_alignment", markdown_payload == expected_payload, markdown_payload)
    check("candidate_count", result["candidate_count"] == 1 and result["single_hypothesis_count"] == 1, {"candidate_count": result["candidate_count"], "hypothesis_count": result["single_hypothesis_count"]})
    check("test_not_read", result["test_read"] is False and manifest["test_read"] is False, None)
    check("zip_not_created", result["zip_created"] is False and manifest["zip_created"] is False, None)

    failures = [entry for entry in checks if not entry["passed"]]
    status = "AUDIT_VERIFIED" if not failures else "AUDIT_FAIL"
    report = {
        "experiment_id": result["experiment_id"],
        "status": status,
        "checked_count": len(checks),
        "passed_count": len(checks) - len(failures),
        "mismatch_count": len(failures),
        "candidate_count": 1,
        "gate_checks_count": len(expected_gates),
        "checks": checks,
        "recomputed": recomputed,
        "performance_gate_pass": all(expected_gates.values()),
        "decision": "RUN_PRODUCTION_COMPATIBILITY_AUDIT" if all(expected_gates.values()) and not failures else "REJECT_OR_HOLD_KEEP_113A",
        "test_read": False,
        "zip_created": False,
    }
    report_path = OUT / "validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    attestation = {
        "experiment_id": result["experiment_id"],
        "status": status,
        "manifest_sha256": sha256(OUT / "audit_manifest.json"),
        "validation_report_sha256": sha256(report_path),
        "validator_sha256": sha256(Path(__file__)),
        "manifest_file_count": manifest["file_count"],
        "candidate_count": 1,
        "checked_count": len(checks),
        "passed_count": len(checks) - len(failures),
        "mismatch_count": len(failures),
        "gate_checks_count": len(expected_gates),
        "performance_gate_pass": all(expected_gates.values()),
        "remaining_unverified": ["production-vs-strict 113A prediction-distribution compatibility"] if all(expected_gates.values()) else [],
        "zip_created": False,
    }
    (OUT / "audit_attestation.json").write_text(json.dumps(attestation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    decision = "PROCEED_TO_PRODUCTION_COMPATIBILITY" if status == "AUDIT_VERIFIED" and all(expected_gates.values()) else "REJECT_OR_HOLD_KEEP_113A"
    (OUT / "FINAL_DECISION.md").write_text(
        "# 121A strict research decision\n\n"
        f"- Audit status: `{status}`\n"
        f"- Performance gate: `{all(expected_gates.values())}`\n"
        f"- Decision: `{decision}`\n"
        f"- Checked: `{len(checks)}`; mismatches: `{len(failures)}`; gates: `{len(expected_gates)}`\n"
        "- Test read: `False`; ZIP created: `False`\n",
        encoding="utf-8",
    )
    print(json.dumps(attestation, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
