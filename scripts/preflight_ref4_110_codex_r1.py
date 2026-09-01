#!/usr/bin/env python3
"""Independent preflight for the clean Codex 110A/110B/110C rebuild."""
from __future__ import annotations

import hashlib
import json
import math
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "REF4-110-CODEX-R1"
OUT = ROOT / "model" / EXPERIMENT_ID
TRAIN = ROOT / "data/train.csv"
BASE = ROOT / "model/REF4-OOF-DIAG-034A"
ZIP_109C = ROOT / "output/submit_ref4_super_ensemble_109C.zip"
LB_109 = ROOT / "output/lb_record_109.json"
EXPECTED_TRAIN_SHA = "d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff"
EXPECTED_109C_SHA = "03e874949ae7172af0dab16d9c3f52de94d5ac9256e571ddedd377b785d9634f"

CANDIDATES = {
    "REF4-TEMPORAL-CROSSFIT-MOE-110A-CODEX-R1": {
        "hypothesis": "A shallow temporal residual router can select among three strictly-forward residual experts row by row.",
        "single_change": "router over fixed CB-only, CB+LGB, and CB+LGB+XGB expert predictions",
    },
    "REF4-CROSSFIT-SUPER-110B-CODEX-R1": {
        "hypothesis": "Replacing in-sample residual targets with strict forward OOF targets improves temporal transfer.",
        "single_change": "strict forward residual target generation",
    },
    "REF4-HIERARCHICAL-EB-SUPER-110C-CODEX-R1": {
        "hypothesis": "Fold-local pitcher-by-batter-hand empirical Bayes features add signal beyond 110B.",
        "single_change": "fixed Beta-Binomial EB features added to 110B",
    },
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, object]] = []
    failures: list[str] = []

    def check(name: str, passed: bool, actual: object) -> None:
        checks.append({"name": name, "checked": True, "passed": bool(passed), "actual": actual})
        if not passed:
            failures.append(name)

    required = [
        ROOT / "start02_dev_log.md",
        ROOT / "start03_reference.md",
        ROOT / "start04_uptostage.md",
        ROOT / "01_제약과금지사항.md",
        ROOT / "08_Gemini_작업위임서.md",
        TRAIN,
        BASE / "diagnostic_rows.csv",
        BASE / "audit_manifest.json",
        BASE / "validation_report.json",
        BASE / "audit_attestation.json",
        ZIP_109C,
        LB_109,
    ]
    for path in required:
        check(f"exists:{path.relative_to(ROOT)}", path.is_file(), path.is_file())

    if failures:
        result = {"experiment_id": EXPERIMENT_ID, "status": "BLOCKED", "checked_count": len(checks), "passed_count": sum(c["passed"] for c in checks), "mismatch_count": len(failures), "failures": failures, "checks": checks}
        write_json(OUT / "preflight_report.json", result)
        raise SystemExit(json.dumps(result, ensure_ascii=False))

    train_sha = sha256_path(TRAIN)
    check("train_sha256", train_sha == EXPECTED_TRAIN_SHA, train_sha)
    train = pd.read_csv(TRAIN, low_memory=False)
    check("train_rows", len(train) == 1_475_092, len(train))
    check("train_row_id_unique", train["row_id"].is_unique, int(train["row_id"].nunique()))
    target = pd.to_numeric(train["control_success"], errors="coerce").to_numpy(float)
    check("train_target_finite", np.isfinite(target).all(), int(np.isfinite(target).sum()))
    check("train_target_binary", set(np.unique(target)) == {0.0, 1.0}, np.unique(target).tolist())
    season_rows = train.groupby("season", sort=True).size().astype(int).to_dict()
    expected_rows = {2019: 237413, 2020: 244087, 2021: 247088, 2022: 247472, 2023: 245525, 2024: 253507}
    check("train_season_rows", season_rows == expected_rows, season_rows)

    zip_sha = sha256_path(ZIP_109C)
    check("109c_zip_sha256", zip_sha == EXPECTED_109C_SHA, zip_sha)
    with zipfile.ZipFile(ZIP_109C) as archive:
        bad = archive.testzip()
        names = archive.namelist()
    check("109c_zip_integrity", bad is None, bad)
    check("109c_zip_root_contract", {"script.py", "requirements.txt"}.issubset(names) and any(n.startswith("model/") for n in names), {"members": len(names)})
    lb = json.loads(LB_109.read_text(encoding="utf-8"))
    check("109c_lb_binding", lb.get("version") == "109C" and lb.get("zip_name") == ZIP_109C.name and math.isclose(float(lb.get("score")), 1120.8914462094, abs_tol=1e-12), lb)

    upstream_att = json.loads((BASE / "audit_attestation.json").read_text(encoding="utf-8"))
    upstream_manifest = BASE / "audit_manifest.json"
    upstream_report = BASE / "validation_report.json"
    check("upstream_attestation_status", upstream_att.get("status") == "AUDIT_VERIFIED" and upstream_att.get("mismatch_count") == 0, upstream_att)
    check("upstream_manifest_attested", sha256_path(upstream_manifest) == upstream_att.get("manifest_sha256"), sha256_path(upstream_manifest))
    check("upstream_report_attested", sha256_path(upstream_report) == upstream_att.get("validation_report_sha256"), sha256_path(upstream_report))
    validator = ROOT / "scripts/verify_ref4_oof_diag_034a.py"
    check("upstream_validator_attested", sha256_path(validator) == upstream_att.get("validator_sha256"), sha256_path(validator))

    manifest = json.loads(upstream_manifest.read_text(encoding="utf-8"))
    artifact_mismatches = []
    for relative, record in manifest["artifacts"].items():
        path = ROOT / relative
        if not path.is_file() or sha256_path(path) != record["sha256"] or path.stat().st_size != record["size"]:
            artifact_mismatches.append(relative)
    # The historical 030 submission ZIP was removed after the audited OOF was
    # produced.  It is not an input to this experiment and the active rollback
    # is the independently hashed 109C ZIP above.  Every computational OOF
    # artifact must still match; no other exception is accepted.
    allowed_missing_legacy = ["output/submit_ref4_champion_030.zip"]
    critical_mismatches = [name for name in artifact_mismatches if name not in allowed_missing_legacy]
    check("upstream_critical_artifact_hashes", not critical_mismatches, critical_mismatches)
    check("upstream_legacy_archive_exception_exact", artifact_mismatches == allowed_missing_legacy, artifact_mismatches)
    check("upstream_artifact_count", len(manifest["artifacts"]) == manifest["artifact_count"], {"actual": len(manifest["artifacts"]), "recorded": manifest["artifact_count"]})

    diag = pd.read_csv(BASE / "diagnostic_rows.csv", dtype={"row_id": str, "game_type": str, "pitcher_id": str})
    check("source_oof_rows", len(diag) == 746_504, len(diag))
    check("source_oof_row_id_unique", diag["row_id"].is_unique, int(diag["row_id"].nunique()))
    check("source_oof_years", sorted(diag["season"].unique().tolist()) == [2022, 2023, 2024], sorted(diag["season"].unique().tolist()))
    pred_cols = [c for c in diag.columns if c.startswith("prediction_") or c.startswith("p_") or c.startswith("risk_")]
    pred_values = diag[pred_cols].to_numpy(float)
    check("source_predictions_finite", np.isfinite(pred_values).all(), {"columns": len(pred_cols), "finite": int(np.isfinite(pred_values).sum()), "total": int(pred_values.size)})
    check("source_predictions_range", bool(((pred_values >= 0.0) & (pred_values <= 1.0)).all()), {"min": float(pred_values.min()), "max": float(pred_values.max())})
    source_rows = train.loc[train["season"].isin([2022, 2023, 2024]), ["row_id", "season", "control_success"]].copy()
    source_rows["row_id"] = source_rows["row_id"].astype(str)
    paired = diag[["row_id", "season", "target"]].merge(source_rows, on="row_id", how="outer", validate="one_to_one", indicator=True, suffixes=("_oof", "_train"))
    pair_ok = len(paired) == len(diag) and paired["_merge"].eq("both").all() and paired["season_oof"].eq(paired["season_train"]).all() and np.array_equal(paired["target"].to_numpy(float), paired["control_success"].to_numpy(float))
    check("source_oof_train_pairing", pair_ok, {"rows": len(paired), "merge_counts": paired["_merge"].value_counts().to_dict()})

    candidate_ids = sorted(CANDIDATES)
    check("candidate_count", len(candidate_ids) == 3, {"count": len(candidate_ids), "ids": candidate_ids})
    check("disk_free_minimum", (free_bytes := __import__("shutil").disk_usage(ROOT).free) >= 8 * 1024**3, {"free_bytes": free_bytes, "minimum_bytes": 8 * 1024**3})

    status = "AUDIT_VERIFIED" if not failures else "AUDIT_INCOMPLETE"
    contract = {
        "experiment_id": EXPERIMENT_ID,
        "status": "LOCKED_BEFORE_RESULTS",
        "official_champion": {"version": "109C", "score_evidence": "USER_REPORTED_OFFICIAL_LEADERBOARD_RESULT", "score": float(lb["score"]), "zip": str(ZIP_109C.relative_to(ROOT)), "zip_sha256": zip_sha},
        "rollback": {"zip": str(ZIP_109C.relative_to(ROOT)), "zip_sha256": zip_sha},
        "candidate_count": len(candidate_ids),
        "candidates": CANDIDATES,
        "strict_oof_source": str((BASE / "diagnostic_rows.csv").relative_to(ROOT)),
        "valid_years": [2022, 2023, 2024],
        "temporal_protocol": {"2022": "identity warm-up", "2023": "fit only 2022 OOF", "2024": "fit only 2022-2023 OOF", "2025": "fit 2022-2024 OOF"},
        "fixed_weights": {"regular_residual": 0.085, "futures_residual": 0.035, "recent_season": {"2022": 0.20, "2023": 0.30, "2024": 0.50}},
        "fixed_eb": {"global_prior_weight": 100.0, "hand_prior_weight": 50.0, "reliability_k": 25.0},
        "promotion_gate": {"delta_brier_2024_max": -0.0001, "worst_season_delta_max": 0.00005, "pitcher_cluster_bootstrap_2024_ci_high_max": 0.0, "row_independence_atol": 1e-12, "full_runtime_seconds_max": 600.0},
        "excluded_untrusted_inputs": ["candidate/p_103_full_train.npy", "scripts/build_ref4_super_ensemble_110a.py", "scripts/build_ref4_super_ensemble_110b.py", "scripts/build_ref4_super_ensemble_110c.py", "output/submit_ref4_super_ensemble_111A.zip"],
        "documented_non_input_gap": {"path": "output/submit_ref4_champion_030.zip", "reason": "historical archive absent; all computational OOF artifacts match and active rollback is 109C"},
        "submission_rule": "Package all three only after audited comparison; approve only the best candidate that passes every gate. Otherwise retain 109C.",
    }
    write_json(OUT / "audit_contract.json", contract)
    report = {"experiment_id": EXPERIMENT_ID, "status": status, "checked_count": len(checks), "passed_count": sum(bool(c["passed"]) for c in checks), "mismatch_count": len(failures), "failures": failures, "checks": checks, "candidate_count": len(candidate_ids), "candidate_ids": candidate_ids, "train_sha256": train_sha, "champion_zip_sha256": zip_sha}
    write_json(OUT / "preflight_report.json", report)
    (OUT / "preflight_report.md").write_text(
        f"# {EXPERIMENT_ID} preflight\n\n- status: `{status}`\n- checked: `{len(checks)}`\n- mismatches: `{len(failures)}`\n- candidates: `{len(candidate_ids)}`\n- 109C rollback SHA-256: `{zip_sha}`\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "checked_count": len(checks), "passed_count": report["passed_count"], "mismatch_count": len(failures), "candidate_count": len(candidate_ids), "free_bytes": free_bytes}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
