#!/usr/bin/env python3
"""Independent validator for the 040A provenance dossier."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import catboost as cb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "REF4-CHAMPION-PROVENANCE-DOSSIER-040A"
OUT = ROOT / "model" / EXPERIMENT_ID
ORIGINAL = ROOT / "model" / "REF4-CHAMPION-STACK-030"
REPRO = ROOT / "model" / "REF4-CHAMPION-REPRO-038A"
ZIP = ROOT / "output" / "submit_ref4_champion_030.zip"
EXPECTED_ZIP_SHA256 = "ca6af4bdf26d4bc79d503cdc6a09b0057a7ea9bdd77e29012f9e459a50cd66b8"
STAGES = {
    "037A": ROOT / "model" / "REF4-CHAMPION-ERA-PROVENANCE-037A",
    "038A": REPRO,
    "039A": ROOT / "model" / "REF4-CHAMPION-REPRO-STRUCTURE-039A",
}
VALIDATORS = {
    "037A": ROOT / "scripts" / "verify_ref4_champion_era_provenance_037a.py",
    "038A": ROOT / "scripts" / "verify_ref4_champion_repro_038a.py",
    "039A": ROOT / "scripts" / "verify_ref4_structure_039a.py",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def embedded_thread(model: cb.CatBoost) -> int:
    metadata = dict(model.get_metadata())
    params = json.loads(metadata["params"])
    return int(params.get("flat_params", {}).get("thread_count", params.get("system_options", {}).get("thread_count", -1)))


def add_check(checks: list[dict[str, Any]], check_id: str, passed: bool, actual: Any) -> None:
    checks.append({"check_id": check_id, "checked": True, "status": "PASS" if passed else "FAIL", "actual": actual})


def manifest_mismatches(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    mismatches = []
    for relative, expected in manifest.get("artifacts", {}).items():
        path = ROOT / relative
        actual_hash = sha256_file(path) if path.is_file() else None
        actual_size = path.stat().st_size if path.is_file() else None
        if actual_hash != expected["sha256"] or actual_size != expected["size"]:
            mismatches.append({"path": relative, "expected": expected["sha256"], "actual": actual_hash})
    return mismatches


def main() -> None:
    manifest = json.loads((OUT / "audit_manifest.json").read_text())
    contract = json.loads((OUT / "audit_contract.json").read_text())
    preflight = json.loads((OUT / "preflight_report.json").read_text())
    index = json.loads((OUT / "lineage_index.json").read_text())
    limitations = json.loads((OUT / "known_limitations.json").read_text())
    result = json.loads((OUT / "result.json").read_text())
    checks: list[dict[str, Any]] = []

    current_manifest_mismatches = manifest_mismatches(manifest)
    add_check(checks, "audit_manifest_artifacts", not current_manifest_mismatches, current_manifest_mismatches)
    add_check(checks, "audit_manifest_count", manifest.get("artifact_count") == len(manifest.get("artifacts", {})), {
        "declared": manifest.get("artifact_count"), "actual": len(manifest.get("artifacts", {}))
    })
    add_check(checks, "preflight_verified", preflight.get("status") == "AUDIT_VERIFIED" and preflight.get("fail_count") == 0, {
        "status": preflight.get("status"), "fail_count": preflight.get("fail_count")
    })
    add_check(checks, "contract_counts", contract.get("expected_stage_count") == 3 and contract.get("expected_model_count") == 56 and contract.get("candidate_count") == contract.get("leaf_count") == contract.get("gate_count") == 0, {
        "stages": contract.get("expected_stage_count"), "models": contract.get("expected_model_count")
    })

    stage_failures = []
    legacy_gap_count = 0
    for stage, directory in STAGES.items():
        stage_manifest_path = directory / "audit_manifest.json"
        stage_report_path = directory / "validation_report.json"
        stage_attestation_path = directory / "audit_attestation.json"
        stage_manifest = json.loads(stage_manifest_path.read_text())
        stage_report = json.loads(stage_report_path.read_text())
        stage_attestation = json.loads(stage_attestation_path.read_text())
        artifact_failures = manifest_mismatches(stage_manifest)
        report_link = stage_attestation.get("validation_report_sha256") == sha256_file(stage_report_path)
        validator_link = stage_attestation.get("validator_sha256") == sha256_file(VALIDATORS[stage])
        manifest_field = "manifest_sha256" if stage == "037A" else "audit_manifest_sha256"
        if manifest_field in stage_attestation:
            manifest_link = stage_attestation[manifest_field] == sha256_file(stage_manifest_path)
        else:
            manifest_link = stage == "038A"
            legacy_gap_count += int(stage == "038A")
        declared_count = stage_manifest.get("artifact_count")
        actual_count = len(stage_manifest.get("artifacts", {}))
        if artifact_failures or not report_link or not validator_link or not manifest_link or declared_count != actual_count:
            stage_failures.append({
                "stage": stage, "artifact_failures": artifact_failures, "report_link": report_link,
                "validator_link": validator_link, "manifest_link_or_disclosed_gap": manifest_link,
                "declared_artifacts": declared_count, "actual_artifacts": actual_count,
            })
    add_check(checks, "upstream_stage_hashes_and_links", not stage_failures and legacy_gap_count == 1, {
        "failures": stage_failures, "legacy_gap_count": legacy_gap_count
    })

    att037 = json.loads((STAGES["037A"] / "audit_attestation.json").read_text())
    build038 = json.loads((REPRO / "build_manifest.json").read_text())
    report038 = json.loads((REPRO / "validation_report.json").read_text())
    att039 = json.loads((STAGES["039A"] / "audit_attestation.json").read_text())
    stage_semantics = {
        "037A": att037.get("status") == "AUDIT_VERIFIED" and att037.get("provenance_status") == "AUDIT_FAIL_PROVENANCE" and att037.get("parameter_mismatch_count") == 10 and att037.get("binding_evidence_count") == 0,
        "038A": build038.get("model_count") == 56 and build038.get("parameter_mismatch_count") == 0 and report038.get("status") == "AUDIT_VERIFIED" and report038.get("prediction_models_within_tolerance") == 56 and report038.get("mismatch_count") == 0,
        "039A": att039.get("status") == "AUDIT_VERIFIED" and att039.get("structure_status") == "STRUCTURE_EQUIVALENT" and att039.get("model_pair_count") == 56 and att039.get("structure_match_count") == 56 and att039.get("structure_mismatch_count") == 0,
    }
    add_check(checks, "upstream_stage_semantics", all(stage_semantics.values()), stage_semantics)

    train_path, trackman_path = ROOT / "data" / "train.csv", ROOT / "data" / "trackman_history.csv"
    train = pd.read_csv(train_path, usecols=["row_id", "season", "control_success"], low_memory=False)
    trackman = pd.read_csv(trackman_path, usecols=["season"], low_memory=False)
    target = pd.to_numeric(train["control_success"], errors="coerce").to_numpy(float)
    source_actual = {
        "train_sha256": sha256_file(train_path), "train_rows": len(train),
        "train_unique_row_ids": int(train["row_id"].nunique(dropna=False)), "train_row_id_unique": bool(train["row_id"].is_unique),
        "train_season_min": int(train["season"].min()), "train_season_max": int(train["season"].max()),
        "target_finite_binary": bool(np.isfinite(target).all() and np.isin(target, [0.0, 1.0]).all()),
        "trackman_sha256": sha256_file(trackman_path), "trackman_rows": len(trackman),
        "trackman_season_min": int(trackman["season"].min()), "trackman_season_max": int(trackman["season"].max()),
    }
    source_index = index["source_data"]
    source_matches_index = (
        source_index["train"]["sha256"] == source_actual["train_sha256"]
        and source_index["train"]["row_count"] == source_actual["train_rows"]
        and source_index["train"]["row_id_unique_count"] == source_actual["train_unique_row_ids"]
        and source_index["train"]["row_id_is_unique"] == source_actual["train_row_id_unique"]
        and source_index["train"]["target_is_finite_binary"] == source_actual["target_finite_binary"]
        and source_index["trackman"]["sha256"] == source_actual["trackman_sha256"]
        and source_index["trackman"]["row_count"] == source_actual["trackman_rows"]
        and source_index["trackman"]["season_max"] == source_actual["trackman_season_max"]
    )
    add_check(checks, "source_data_recomputed", source_matches_index and source_actual["train_row_id_unique"] and source_actual["target_finite_binary"], source_actual)
    del train, trackman, target

    prebuild038 = json.loads((REPRO / "prebuild_manifest.json").read_text())
    source_contract = {row["path"]: row["sha256"] for row in prebuild038["source_inputs"]}
    add_check(checks, "source_to_prebuild_binding", source_contract.get("data/train.csv") == source_actual["train_sha256"] and source_contract.get("data/trackman_history.csv") == source_actual["trackman_sha256"], source_contract)

    stored_lineage = pd.read_csv(OUT / "model_lineage.csv", keep_default_na=False).set_index("model")
    parity = pd.read_csv(REPRO / "prediction_parity.csv").set_index("file")
    structure = pd.read_csv(STAGES["039A"] / "structure_inventory.csv").set_index("model")
    build_models = {row["file"]: row for row in build038["models"]}
    model_failures = []
    names = {path.name for path in ORIGINAL.glob("*.cbm")}
    contract_by_name = {row["file"]: row for row in prebuild038["model_contract"]}
    for name in sorted(names):
        if name not in stored_lineage.index or name not in contract_by_name or name not in build_models or name not in parity.index or name not in structure.index:
            model_failures.append({"model": name, "reason": "missing_binding_row"})
            continue
        specification = contract_by_name[name]
        model = cb.CatBoost()
        model.load_model(REPRO / name)
        params = model.get_all_params()
        row = stored_lineage.loc[name]
        actual = {
            "thread": embedded_thread(model), "seed": int(params.get("random_seed", -1)),
            "tree_count": int(model.tree_count_), "feature_count": len(model.feature_names_),
            "original_hash": sha256_file(ORIGINAL / name), "repro_hash": sha256_file(REPRO / name),
        }
        passed = (
            actual["thread"] == int(specification["thread_count"]) == int(row["embedded_thread_count"])
            and actual["seed"] == int(specification["random_seed"]) == int(row["embedded_random_seed"])
            and actual["tree_count"] == int(row["tree_count"])
            and actual["feature_count"] == int(row["feature_count"])
            and actual["original_hash"] == str(row["original_cbm_sha256"])
            and actual["repro_hash"] == str(row["repro_cbm_sha256"]) == build_models[name]["sha256"]
            and float(parity.loc[name, "max_abs_difference"]) == float(row["prediction_max_abs_difference"])
            and str(parity.loc[name, "within_tolerance"]).lower() == str(row["prediction_within_tolerance"]).lower()
            and str(structure.loc[name, "semantic_match"]).lower() == str(row["semantic_structure_equal"]).lower()
            and str(structure.loc[name, "original_semantic_sha256"]) == str(row["original_semantic_sha256"])
            and str(structure.loc[name, "repro_semantic_sha256"]) == str(row["repro_semantic_sha256"])
        )
        if not passed:
            model_failures.append({"model": name, "actual": actual})
    add_check(checks, "model_lineage_recomputed", not model_failures and len(names) == len(stored_lineage) == 56, {
        "source_count": len(names), "lineage_count": len(stored_lineage), "failures": model_failures
    })

    model_summary = index["model_summary"]
    model_summary_expected = (
        model_summary["model_count"] == 56
        and model_summary["thread_4_count"] == sum(int(row["thread_count"]) == 4 for row in prebuild038["model_contract"])
        and model_summary["thread_3_count"] == sum(int(row["thread_count"]) == 3 for row in prebuild038["model_contract"])
        and model_summary["parameter_contract_match_count"] == 56
        and model_summary["prediction_parity_count"] == 56
        and model_summary["structure_equivalent_count"] == 56
        and model_summary["cbm_byte_equal_count"] == 0
    )
    add_check(checks, "model_summary_recomputed", model_summary_expected, model_summary)

    trackman_prior = pd.read_csv(REPRO / "trackman_prior_features.csv", usecols=["season"])
    derived = index["derived_artifacts"]
    derived_expected = (
        derived["trackman_prior_rows"] == len(trackman_prior)
        and derived["trackman_prior_target_season_max"] == int(trackman_prior["season"].max()) == 2025
        and derived["trackman_source_max_season"] == source_actual["trackman_season_max"] < 2025
        and derived["trackman_2025_cutoff_valid"] is True
        and derived["pitcher_snapshot_rows"] == len(pd.read_pickle(REPRO / "pitcher_snapshots.pkl"))
        and derived["batter_snapshot_rows"] == len(pd.read_pickle(REPRO / "batter_snapshots.pkl"))
        and derived["pitchmix_snapshot_rows"] == len(pd.read_pickle(REPRO / "pitchmix_snapshots.pkl"))
    )
    add_check(checks, "derived_cutoff_and_snapshots", derived_expected, derived)

    required_limitations = {
        "USER_REPORTED_SCORE_NOT_LOCALLY_RECOMPUTED", "ORIGINAL_030_BUILD_BINDING_ABSENT",
        "LEGACY_ATTESTATION_MANIFEST_LINK_ABSENT", "CBM_BYTES_DIFFER", "REPRO_PACKAGE_NOT_BUILT_OR_SCORED",
    }
    actual_limitations = {item["code"] for item in limitations.get("limitations", [])}
    add_check(checks, "known_limitations_complete", limitations.get("limitation_count") == len(actual_limitations) == 5 and actual_limitations == required_limitations, {
        "declared": limitations.get("limitation_count"), "codes": sorted(actual_limitations)
    })

    dossier = (OUT / "reproducibility_dossier.md").read_text()
    begin, end = "<!-- LINEAGE_INDEX_JSON_BEGIN\n", "\nLINEAGE_INDEX_JSON_END -->"
    embedded_index = None
    if dossier.count(begin) == 1 and dossier.count(end) == 1:
        embedded_index = json.loads(dossier.split(begin, 1)[1].split(end, 1)[0])
    add_check(checks, "dossier_index_parity", embedded_index == index, "exact" if embedded_index == index else "mismatch")

    result_md = (OUT / "result.md").read_text()
    result_begin, result_end = "<!-- RESULT_JSON_BEGIN\n", "\nRESULT_JSON_END -->"
    embedded_result = None
    if result_md.count(result_begin) == 1 and result_md.count(result_end) == 1:
        embedded_result = json.loads(result_md.split(result_begin, 1)[1].split(result_end, 1)[0])
    add_check(checks, "result_markdown_parity", embedded_result == result, "exact" if embedded_result == result else "mismatch")
    result_expected = (
        result.get("lineage_status") == "PROVENANCE_DOSSIER_VERIFIED"
        and result.get("stage_count") == 3 and result.get("model_count") == 56
        and result.get("stage_artifact_mismatch_count") == 0
        and result.get("stage_link_mismatch_count") == 0
        and result.get("model_mismatch_count") == 0
        and result.get("known_limitation_count") == 5
        and result.get("legacy_link_gap_count") == 1
        and result.get("executor_failure_count") == 0
    )
    add_check(checks, "result_counts_recomputed", result_expected, result)

    official = index["official_artifact"]
    champion_hash = sha256_file(ZIP)
    official_expected = (
        champion_hash == EXPECTED_ZIP_SHA256 == official["sha256"]
        and official["score_evidence_class"] == "user_reported_leaderboard_result_not_locally_recomputed"
        and float(official["leaderboard_score"]) == 1068.25021
    )
    add_check(checks, "official_artifact_and_score_class", official_expected, official)
    operation_keys = [
        "test_read_performed", "test_inference_performed", "training_performed", "model_modified",
        "candidate_created", "zip_created", "submission_performed",
    ]
    zero_counts = index.get("candidate_count") == index.get("leaf_count") == index.get("gate_count") == 0
    operations_false = all(index.get(key) is False for key in operation_keys)
    candidate_exists = (ROOT / "candidate" / EXPERIMENT_ID).exists()
    zip_matches = sorted(str(path.relative_to(ROOT)) for path in (ROOT / "output").glob("*040A*.zip"))
    add_check(checks, "zero_candidates_and_forbidden_operations", zero_counts and operations_false and not candidate_exists and not zip_matches, {
        "zero_counts": zero_counts, "operations_false": operations_false,
        "candidate_exists": candidate_exists, "zip_matches": zip_matches,
    })
    add_check(checks, "lineage_executor_checks", index.get("lineage_status") == "PROVENANCE_DOSSIER_VERIFIED" and index.get("executor_failure_count") == 0 and all(index.get("executor_checks", {}).values()), {
        "lineage_status": index.get("lineage_status"), "executor_failures": index.get("executor_failures")
    })
    add_check(checks, "audit_manifest_counts", manifest.get("stage_count") == 3 and manifest.get("model_count") == 56 and manifest.get("stage_artifact_mismatch_count") == 0 and manifest.get("model_mismatch_count") == 0, {
        "stage_count": manifest.get("stage_count"), "model_count": manifest.get("model_count"),
        "stage_artifact_mismatch_count": manifest.get("stage_artifact_mismatch_count"),
        "model_mismatch_count": manifest.get("model_mismatch_count"),
    })

    failed = [row["check_id"] for row in checks if row["status"] != "PASS"]
    final_status = "AUDIT_VERIFIED" if not failed else "AUDIT_FAIL_PROVENANCE"
    lineage_status = "PROVENANCE_DOSSIER_VERIFIED" if not failed else "AUDIT_FAIL_PROVENANCE"
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": final_status,
        "lineage_status": lineage_status,
        "check_count": len(checks),
        "pass_count": len(checks) - len(failed),
        "fail_count": len(failed),
        "mismatch_count": len(failed),
        "failed_checks": failed,
        "stage_count": 3,
        "model_count": 56,
        "known_limitation_count": 5,
        "legacy_link_gap_count": legacy_gap_count,
        "candidate_count": 0, "leaf_count": 0, "gate_count": 0,
        "test_read_performed": False, "test_inference_performed": False,
        "training_performed": False, "zip_created": False,
        "checks": checks,
        "artifacts": {
            "audit_manifest_sha256": sha256_file(OUT / "audit_manifest.json"),
            "lineage_index_sha256": sha256_file(OUT / "lineage_index.json"),
            "model_lineage_sha256": sha256_file(OUT / "model_lineage.csv"),
            "known_limitations_sha256": sha256_file(OUT / "known_limitations.json"),
            "reproducibility_dossier_sha256": sha256_file(OUT / "reproducibility_dossier.md"),
            "champion_zip_sha256": champion_hash,
        },
    }
    report_path = OUT / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    validator_path = Path(__file__).resolve()
    attestation = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": final_status,
        "lineage_status": lineage_status,
        "audit_manifest_sha256": sha256_file(OUT / "audit_manifest.json"),
        "validation_report_sha256": sha256_file(report_path),
        "validator_sha256": sha256_file(validator_path),
        "lineage_index_sha256": sha256_file(OUT / "lineage_index.json"),
        "model_lineage_sha256": sha256_file(OUT / "model_lineage.csv"),
        "known_limitations_sha256": sha256_file(OUT / "known_limitations.json"),
        "reproducibility_dossier_sha256": sha256_file(OUT / "reproducibility_dossier.md"),
        "champion_zip_sha256": champion_hash,
        "score_evidence_class": "user_reported_leaderboard_result_not_locally_recomputed",
        "check_count": len(checks), "pass_count": len(checks) - len(failed),
        "fail_count": len(failed), "mismatch_count": len(failed),
        "stage_count": 3, "model_count": 56,
        "known_limitation_count": 5, "legacy_link_gap_count": legacy_gap_count,
        "candidate_count": 0, "leaf_count": 0, "gate_count": 0,
        "test_read_performed": False, "test_inference_performed": False,
        "training_performed": False, "zip_created": False,
    }
    (OUT / "audit_attestation.json").write_text(json.dumps(attestation, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({
        "status": final_status, "lineage_status": lineage_status,
        "checks": len(checks), "passed": len(checks) - len(failed), "mismatches": len(failed),
        "stages": 3, "models": 56, "known_limitations": 5,
        "legacy_link_gaps": legacy_gap_count,
    }, indent=2))
    if final_status != "AUDIT_VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
