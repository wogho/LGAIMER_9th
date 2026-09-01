#!/usr/bin/env python3
"""Lock and verify the pre-result contract for REF4-R-CROSSFIT-RELIABILITY-CURVE-121A."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model/REF4-R-CROSSFIT-RELIABILITY-CURVE-121A"
TRAIN = ROOT / "data/train.csv"
ANCHOR = ROOT / "model/REF4-113A-V66-NESTED-117A/oof_predictions.csv"
ANCHOR_AUDIT = ROOT / "model/REF4-113A-V66-NESTED-117A/validation_report.json"
CHAMPION = ROOT / "output/submit_ref4_super_ensemble_113A.zip"

EXPECTED = {
    "train_sha256": "d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff",
    "anchor_sha256": "560e1ca40a21f0b9b296f612e6764e50eaa2a6f62b08561b86dd9d1803c23aa6",
    "champion_sha256": "40149ebe251191be26b01728faa456112edd81e52f3356bb3d65319070165df1",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    contract = {
        "experiment_id": "REF4-R-CROSSFIT-RELIABILITY-CURVE-121A",
        "status": "LOCKED_BEFORE_FORMAL_RUN",
        "created_date": "2026-08-31",
        "official_champion": {
            "version": "113A",
            "official_score": 1121.9039933605,
            "score_evidence": "USER_REPORTED_OFFICIAL_LEADERBOARD_RESULT",
            "zip": str(CHAMPION.relative_to(ROOT)),
            "zip_sha256": EXPECTED["champion_sha256"],
        },
        "prior_observation_disclosure": {
            "status": "HYPOTHESIS_WAS_SCREENED_BEFORE_THIS_FORMAL_CONTRACT",
            "description": (
                "A read-only headroom screen found that probability-only bins transferred "
                "from 2022 to 2023 and from 2022+2023 to 2024, while pitcher/batter sample-count "
                "axes reduced the gain. The formal run therefore freezes one p-only candidate; "
                "it is confirmatory, not a blind discovery test."
            ),
        },
        "single_hypothesis": (
            "For Regular rows only, a past-season reliability curve indexed solely by the strict "
            "113A probability removes a stable nonlinear calibration-shape error that transfers "
            "to the next season."
        ),
        "frozen_candidate": {
            "apply_game_type": "R",
            "futures_action": "identity_exact_113A",
            "probability_quantile_bins": 8,
            "bin_edges": "source_R_prediction_quantiles_with_infinite_outer_edges",
            "residual": "target_minus_p113a_strict",
            "centering": "subtract_source_R_global_residual_mean_before_bin_aggregation",
            "shrinkage_k": 300.0,
            "raw_calibrator_clip": [-0.12, 0.12],
            "calibrator_probability_clip": [0.005, 0.995],
            "combination": "p121a=(1-0.25)*p113a+0.25*p_calibrator",
            "convex_weight": 0.25,
            "final_max_absolute_change": 0.03,
            "unknown_bin_delta": 0.0,
        },
        "strict_protocol": {
            "2023": {"fit_seasons": [2022], "validation_seasons": [2023]},
            "2024": {"fit_seasons": [2022, 2023], "validation_seasons": [2024]},
            "validation_labels_used_in_fit": False,
            "row_order": "official_train_order_with_explicit_row_id_assertion",
            "test_read": False,
        },
        "promotion_gates": {
            "2023_overall_brier_gain_min": 0.00030,
            "2024_overall_brier_gain_min": 0.00030,
            "2024_R_brier_gain_min": 0.00050,
            "2023_all_halves_gain_strictly_positive": True,
            "2024_all_halves_gain_strictly_positive": True,
            "2024_all_quarters_gain_strictly_positive": True,
            "2024_pitcher_bootstrap_gain_ci_low_strictly_positive": True,
            "2024_ece15_not_worse": True,
            "F_prediction_max_abs_difference": 0.0,
            "all_predictions_finite_and_in_unit_interval": True,
            "max_absolute_change_max": 0.03,
        },
        "excluded_changes": [
            "pitcher sample-count axis",
            "batter sample-count axis",
            "handedness or count-context axis",
            "weight, bin-count, shrinkage, or cap search",
            "test distribution or test-row aggregation",
            "changes to any 113A model or existing weight",
            "submission packaging before all research and compatibility gates pass",
        ],
        "output_rule": (
            "Run strict research and independent audit first. Do not read test.csv, mutate 113A, "
            "or create a ZIP unless every performance gate and the later production-distribution "
            "compatibility gate pass."
        ),
    }
    (OUT / "audit_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, actual: object) -> None:
        checks.append({"name": name, "passed": bool(passed), "actual": actual})

    for path in (TRAIN, ANCHOR, ANCHOR_AUDIT, CHAMPION):
        check(f"exists:{path.relative_to(ROOT)}", path.is_file(), path.is_file())
    train_hash = sha256(TRAIN)
    anchor_hash = sha256(ANCHOR)
    champion_hash = sha256(CHAMPION)
    check("train_sha256", train_hash == EXPECTED["train_sha256"], train_hash)
    check("anchor_sha256", anchor_hash == EXPECTED["anchor_sha256"], anchor_hash)
    check("champion_sha256", champion_hash == EXPECTED["champion_sha256"], champion_hash)
    zip_test = subprocess.run(
        ["unzip", "-t", str(CHAMPION)], capture_output=True, text=True, check=False
    )
    check("champion_zip_integrity", zip_test.returncode == 0, zip_test.returncode)
    anchor_audit = json.loads(ANCHOR_AUDIT.read_text(encoding="utf-8"))
    check("anchor_audit_verified", anchor_audit.get("status") == "AUDIT_VERIFIED", anchor_audit.get("status"))
    check("anchor_audit_mismatch_count", anchor_audit.get("mismatch_count") == 0, anchor_audit.get("mismatch_count"))

    train = pd.read_csv(TRAIN, usecols=["row_id", "season", "control_success"], low_memory=False)
    anchor = pd.read_csv(
        ANCHOR,
        usecols=["row_id", "season", "game_type", "pitcher_id", "target", "p113a_strict"],
        low_memory=False,
    )
    expected_rows = train.loc[train["season"].isin((2022, 2023, 2024))].reset_index(drop=True)
    check("train_rows", len(train) == 1_475_092, len(train))
    check("train_row_id_unique", train["row_id"].nunique() == len(train), int(train["row_id"].nunique()))
    check("train_target_binary_finite", bool(np.isfinite(train["control_success"]).all() and set(train["control_success"].unique()) == {0, 1}), sorted(train["control_success"].unique().tolist()))
    check("anchor_rows", len(anchor) == 746_504, len(anchor))
    check("anchor_row_id_unique", anchor["row_id"].nunique() == len(anchor), int(anchor["row_id"].nunique()))
    check("anchor_row_order", np.array_equal(anchor["row_id"].astype(str), expected_rows["row_id"].astype(str)), None)
    check("anchor_target", np.array_equal(anchor["target"].to_numpy(float), expected_rows["control_success"].to_numpy(float)), None)
    check("anchor_finite", bool(np.isfinite(anchor[["target", "p113a_strict"]]).all().all()), None)
    check("anchor_probability_range", bool(anchor["p113a_strict"].between(0.0, 1.0).all()), [float(anchor["p113a_strict"].min()), float(anchor["p113a_strict"].max())])
    check("anchor_seasons", set(anchor["season"].unique()) == {2022, 2023, 2024}, sorted(anchor["season"].unique().tolist()))
    check("test_not_read", True, False)

    failures = [entry for entry in checks if not entry["passed"]]
    report = {
        "experiment_id": contract["experiment_id"],
        "status": "AUDIT_VERIFIED" if not failures else "AUDIT_FAIL_DATA",
        "checked_count": len(checks),
        "passed_count": len(checks) - len(failures),
        "mismatch_count": len(failures),
        "checks": checks,
        "source_provenance": {
            "train_sha256": train_hash,
            "strict_anchor_sha256": anchor_hash,
            "champion_zip_sha256": champion_hash,
            "anchor_validation_report_sha256": sha256(ANCHOR_AUDIT),
        },
        "test_read": False,
        "zip_created": False,
    }
    (OUT / "preflight_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
