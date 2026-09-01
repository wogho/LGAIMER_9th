#!/usr/bin/env python3
"""Independent arithmetic/provenance audit of the 121A production compatibility run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wasserstein_distance


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model/REF4-R-CROSSFIT-RELIABILITY-CURVE-121A"
ZIP = ROOT / "output/submit_ref4_super_ensemble_113A.zip"
ANCHOR = ROOT / "model/REF4-113A-V66-NESTED-117A/oof_predictions.csv"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def psi(a: np.ndarray, b: np.ndarray) -> float:
    x = np.clip(np.bincount(a, minlength=8) / len(a), 1e-9, None)
    y = np.clip(np.bincount(b, minlength=8) / len(b), 1e-9, None)
    return float(np.sum((y - x) * np.log(y / x)))


def main() -> None:
    contract = json.loads((OUT / "production_compatibility_contract.json").read_text())
    result = json.loads((OUT / "production_compatibility_result.json").read_text())
    strict_attestation = json.loads((OUT / "audit_attestation.json").read_text())
    calibrator = json.loads((OUT / "production_calibrator.json").read_text())
    stored = pd.read_csv(OUT / "production_compatibility_predictions.csv.gz", low_memory=False)
    anchor = pd.read_csv(ANCHOR, usecols=["row_id", "season", "game_type", "p113a_strict"], low_memory=False)
    expected = anchor.loc[anchor["season"].eq(2024)].reset_index(drop=True)
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, actual: object) -> None:
        checks.append({"name": name, "passed": bool(passed), "actual": actual})

    check("strict_audit_verified", strict_attestation.get("status") == "AUDIT_VERIFIED" and strict_attestation.get("performance_gate_pass") is True, strict_attestation.get("status"))
    check("contract_locked", contract.get("status") == "LOCKED_BEFORE_PRODUCTION_INFERENCE", contract.get("status"))
    check("champion_zip_hash", sha256(ZIP) == result["champion_zip_sha256"], sha256(ZIP))
    check("calibrator_hash", sha256(OUT / "production_calibrator.json") == result["calibrator_sha256"], sha256(OUT / "production_calibrator.json"))
    check("prediction_hash", sha256(OUT / "production_compatibility_predictions.csv.gz") == result["prediction_sha256"], sha256(OUT / "production_compatibility_predictions.csv.gz"))
    check("inference_log_hash", sha256(OUT / "production_compatibility_inference.log") == result["inference_log_sha256"], sha256(OUT / "production_compatibility_inference.log"))
    check("row_count", len(stored) == 253_507 == len(expected), len(stored))
    check("row_id_unique", stored["row_id"].nunique() == len(stored), int(stored["row_id"].nunique()))
    check("row_order", np.array_equal(stored["row_id"].astype(str), expected["row_id"].astype(str)), None)
    check("game_type", np.array_equal(stored["game_type"].astype(str), expected["game_type"].astype(str)), None)
    check("strict_prediction", float(np.max(np.abs(stored["p113a_strict"].to_numpy(float) - expected["p113a_strict"].to_numpy(float)))) <= 1e-15, None)
    check("finite_unit", bool(np.isfinite(stored[["p113a_strict", "p113a_production_full", "p121a_production_geometry"]]).all().all() and stored[["p113a_strict", "p113a_production_full", "p121a_production_geometry"]].min().min() >= 0.0 and stored[["p113a_strict", "p113a_production_full", "p121a_production_geometry"]].max().max() <= 1.0), None)

    regular = stored["game_type"].eq("R").to_numpy()
    p_strict = stored["p113a_strict"].to_numpy(float)
    p_prod = stored["p113a_production_full"].to_numpy(float)
    p_candidate = stored["p121a_production_geometry"].to_numpy(float)
    edges = np.asarray(calibrator["inner_edges"], dtype=float)
    deltas = np.asarray(calibrator["raw_delta_by_bin"], dtype=float)
    bins_strict = np.searchsorted(edges, p_strict[regular], side="right")
    bins_prod = np.searchsorted(edges, p_prod[regular], side="right")
    raw = deltas[bins_prod]
    expected_candidate_R = 0.75 * p_prod[regular] + 0.25 * np.clip(p_prod[regular] + raw, 0.005, 0.995)
    formula_error = float(np.max(np.abs(expected_candidate_R - p_candidate[regular])))
    f_diff = float(np.max(np.abs(p_candidate[~regular] - p_prod[~regular])))
    strict_change = np.abs(0.75 * p_strict[regular] + 0.25 * np.clip(p_strict[regular] + deltas[bins_strict], 0.005, 0.995) - p_strict[regular])
    prod_change = np.abs(p_candidate[regular] - p_prod[regular])
    metrics = {
        "regular_spearman": float(spearmanr(p_strict[regular], p_prod[regular]).statistic),
        "regular_wasserstein": float(wasserstein_distance(p_strict[regular], p_prod[regular])),
        "regular_abs_mean_difference": float(abs(p_prod[regular].mean() - p_strict[regular].mean())),
        "regular_std_ratio": float(p_prod[regular].std() / p_strict[regular].std()),
        "regular_probability_bin_PSI": psi(bins_strict, bins_prod),
        "regular_same_bin_fraction": float(np.mean(bins_strict == bins_prod)),
        "regular_mean_abs_correction_ratio": float(prod_change.mean() / strict_change.mean()),
        "candidate_max_abs_change": float(np.max(np.abs(p_candidate - p_prod))),
        "F_max_abs_difference": f_diff,
    }
    expected_gates = {
        "regular_spearman_at_least_0_85": metrics["regular_spearman"] >= 0.85,
        "regular_wasserstein_at_most_0_03": metrics["regular_wasserstein"] <= 0.03,
        "regular_abs_mean_difference_at_most_0_02": metrics["regular_abs_mean_difference"] <= 0.02,
        "regular_std_ratio_in_0_70_1_40": 0.70 <= metrics["regular_std_ratio"] <= 1.40,
        "regular_probability_bin_PSI_at_most_0_10": metrics["regular_probability_bin_PSI"] <= 0.10,
        "regular_same_bin_fraction_at_least_0_55": metrics["regular_same_bin_fraction"] >= 0.55,
        "regular_mean_abs_correction_ratio_in_0_50_2_00": 0.50 <= metrics["regular_mean_abs_correction_ratio"] <= 2.00,
        "candidate_max_abs_change_at_most_0_03": metrics["candidate_max_abs_change"] <= 0.03 + 1e-15,
        "F_exact_identity": f_diff == 0.0,
        "finite_unit_interval": bool(np.isfinite(p_prod).all() and np.isfinite(p_candidate).all() and p_prod.min() >= 0.0 and p_prod.max() <= 1.0 and p_candidate.min() >= 0.0 and p_candidate.max() <= 1.0),
    }
    check("candidate_formula", formula_error <= 1e-12, formula_error)
    check("F_exact_identity", f_diff == 0.0, f_diff)
    metric_errors = {key: abs(metrics[key] - result["metrics"][key]) for key in metrics}
    check("metric_reproduction", max(metric_errors.values()) <= 1e-12, metric_errors)
    check("gate_count", result["gate_count"] == len(expected_gates) == len(result["gates"]), {"recorded": result["gate_count"], "actual": len(expected_gates)})
    check("gate_reproduction", expected_gates == result["gates"], {"expected": expected_gates, "recorded": result["gates"]})
    check("compatibility_decision", result["compatibility_gate_pass"] == all(expected_gates.values()), {"expected": all(expected_gates.values()), "recorded": result["compatibility_gate_pass"]})
    check("target_not_read_or_scored", result["target_column_read"] is False and result["target_metric_computed"] is False, None)
    check("test_not_read", result["test_read"] is False, None)
    check("zip_not_created", result["zip_created"] is False, None)

    failures = [entry for entry in checks if not entry["passed"]]
    status = "AUDIT_VERIFIED" if not failures else "AUDIT_FAIL"
    verified = status == "AUDIT_VERIFIED" and all(expected_gates.values())
    report = {
        "experiment_id": result["experiment_id"],
        "status": status,
        "checked_count": len(checks),
        "passed_count": len(checks) - len(failures),
        "mismatch_count": len(failures),
        "gate_checks_count": len(expected_gates),
        "metrics": metrics,
        "gates": expected_gates,
        "compatibility_gate_pass": all(expected_gates.values()),
        "production_build_eligible": verified,
        "decision": "BUILD_121A_PRODUCTION_CANDIDATE" if verified else "REJECT_KEEP_113A",
        "target_column_read": False,
        "test_read": False,
        "zip_created": False,
        "checks": checks,
    }
    path = OUT / "production_compatibility_validation_report.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    attestation = {
        "experiment_id": result["experiment_id"],
        "status": status,
        "validator_sha256": sha256(Path(__file__)),
        "result_sha256": sha256(OUT / "production_compatibility_result.json"),
        "validation_report_sha256": sha256(path),
        "checked_count": len(checks),
        "mismatch_count": len(failures),
        "gate_checks_count": len(expected_gates),
        "compatibility_gate_pass": all(expected_gates.values()),
        "production_build_eligible": verified,
        "zip_created": False,
    }
    (OUT / "production_compatibility_attestation.json").write_text(json.dumps(attestation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failed_gates = [name for name, passed in expected_gates.items() if not passed]
    final_status = {
        "experiment_id": result["experiment_id"],
        "status": "AUDIT_VERIFIED_PRODUCTION_INCOMPATIBLE" if status == "AUDIT_VERIFIED" and not all(expected_gates.values()) else ("AUDIT_VERIFIED_BUILD_ELIGIBLE" if verified else "AUDIT_FAIL"),
        "strict_research_performance_gate_pass": bool(strict_attestation["performance_gate_pass"]),
        "production_compatibility_gate_pass": bool(all(expected_gates.values())),
        "failed_production_gates": failed_gates,
        "production_build_eligible": bool(verified),
        "decision": "BUILD_121A_PRODUCTION_CANDIDATE" if verified else "REJECT_121A_KEEP_113A",
        "champion_preserved": "113A",
        "champion_zip_sha256": sha256(ZIP),
        "submission_zip_created": False,
        "test_read": False,
    }
    (OUT / "final_status.json").write_text(json.dumps(final_status, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "FINAL_DECISION.md").write_text(
        "# 121A final decision\n\n"
        f"- Status: `{final_status['status']}`\n"
        f"- Strict research performance gate: `{final_status['strict_research_performance_gate_pass']}`\n"
        f"- Production compatibility gate: `{final_status['production_compatibility_gate_pass']}`\n"
        f"- Failed production gates: `{', '.join(failed_gates)}`\n"
        f"- Decision: `{final_status['decision']}`\n"
        "- Champion preserved: `113A`\n"
        "- Test read: `False`; submission ZIP created: `False`\n",
        encoding="utf-8",
    )
    print(json.dumps(attestation, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
