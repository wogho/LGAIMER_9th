#!/usr/bin/env python3
"""Recompute and consolidate the 037A-039A REF4 provenance chain."""
from __future__ import annotations

import hashlib
import json
import pickle
import time
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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def record(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "size": path.stat().st_size}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def verify_bound_records(binding: dict[str, Any]) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    keys = ["stage_critical_artifacts", "source_inputs", "model_inputs", "champion_zip", "governing_documents"]
    for key in keys:
        values = binding[key] if isinstance(binding[key], list) else [binding[key]]
        for value in values:
            path = ROOT / value["path"]
            actual = sha256_file(path) if path.is_file() else None
            size = path.stat().st_size if path.is_file() else None
            if actual != value["sha256"] or size != value["size"]:
                mismatches.append({"path": value["path"], "expected": value["sha256"], "actual": actual})
    return mismatches


def verify_manifest_artifacts(stage: str, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    for relative, expected in manifest.get("artifacts", {}).items():
        path = ROOT / relative
        actual_hash = sha256_file(path) if path.is_file() else None
        actual_size = path.stat().st_size if path.is_file() else None
        if actual_hash != expected["sha256"] or actual_size != expected["size"]:
            mismatches.append({
                "stage": stage, "path": relative, "expected_hash": expected["sha256"],
                "actual_hash": actual_hash, "expected_size": expected["size"], "actual_size": actual_size,
            })
    return mismatches


def stage_linkage(stage: str) -> dict[str, Any]:
    directory = STAGES[stage]
    manifest_path = directory / "audit_manifest.json"
    report_path = directory / "validation_report.json"
    attestation_path = directory / "audit_attestation.json"
    manifest = json.loads(manifest_path.read_text())
    report = json.loads(report_path.read_text())
    attestation = json.loads(attestation_path.read_text())
    manifest_hash = sha256_file(manifest_path)
    report_hash = sha256_file(report_path)
    validator_hash = sha256_file(VALIDATORS[stage])
    manifest_field = "manifest_sha256" if stage == "037A" else "audit_manifest_sha256"
    direct_manifest_link_present = manifest_field in attestation
    return {
        "stage": stage,
        "directory": str(directory.relative_to(ROOT)),
        "manifest_sha256": manifest_hash,
        "validation_report_sha256": report_hash,
        "attestation_sha256": sha256_file(attestation_path),
        "validator_path": str(VALIDATORS[stage].relative_to(ROOT)),
        "validator_sha256": validator_hash,
        "manifest_artifact_count": len(manifest.get("artifacts", {})),
        "manifest_declared_artifact_count": manifest.get("artifact_count"),
        "manifest_artifact_mismatches": verify_manifest_artifacts(stage, manifest),
        "attestation_status": attestation.get("status"),
        "report_status": report.get("status") or report.get("validation_status"),
        "provenance_status": attestation.get("provenance_status") or report.get("provenance_status"),
        "structure_status": attestation.get("structure_status") or report.get("structure_status"),
        "attestation_mismatch_count": attestation.get("mismatch_count"),
        "report_mismatch_count": report.get("mismatch_count"),
        "direct_manifest_link_present": direct_manifest_link_present,
        "direct_manifest_link_match": attestation.get(manifest_field) == manifest_hash if direct_manifest_link_present else None,
        "report_link_match": attestation.get("validation_report_sha256") == report_hash,
        "validator_link_match": attestation.get("validator_sha256") == validator_hash,
        "legacy_link_gap_code": "LEGACY_ATTESTATION_MANIFEST_LINK_ABSENT" if stage == "038A" and not direct_manifest_link_present else None,
    }


def embedded_thread(model: cb.CatBoost) -> int:
    metadata = dict(model.get_metadata())
    params = json.loads(metadata["params"])
    return int(params.get("flat_params", {}).get("thread_count", params.get("system_options", {}).get("thread_count", -1)))


def build_model_lineage() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    prebuild = json.loads((REPRO / "prebuild_manifest.json").read_text())
    build = json.loads((REPRO / "build_manifest.json").read_text())
    parity = pd.read_csv(REPRO / "prediction_parity.csv").set_index("file")
    structure = pd.read_csv(STAGES["039A"] / "structure_inventory.csv").set_index("model")
    build_models = {row["file"]: row for row in build["models"]}
    rows: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    for specification in prebuild["model_contract"]:
        name = specification["file"]
        original_path, repro_path = ORIGINAL / name, REPRO / name
        model = cb.CatBoost()
        model.load_model(repro_path)
        params = model.get_all_params()
        build_row = build_models[name]
        parity_row = parity.loc[name]
        structure_row = structure.loc[name]
        actual_thread = embedded_thread(model)
        actual_seed = int(params.get("random_seed", -1))
        row = {
            "model": name,
            "family": specification["family"],
            "model_kind": specification["model_kind"],
            "training_scope": specification["scope"],
            "expected_thread_count": int(specification["thread_count"]),
            "embedded_thread_count": actual_thread,
            "expected_random_seed": int(specification["random_seed"]),
            "embedded_random_seed": actual_seed,
            "tree_count": int(model.tree_count_),
            "feature_count": len(model.feature_names_),
            "feature_names_sha256": sha256_bytes("\n".join(model.feature_names_).encode()),
            "original_cbm_sha256": sha256_file(original_path),
            "repro_cbm_sha256": sha256_file(repro_path),
            "cbm_byte_equal": sha256_file(original_path) == sha256_file(repro_path),
            "prediction_max_abs_difference": float(parity_row["max_abs_difference"]),
            "prediction_within_tolerance": bool(parity_row["within_tolerance"]),
            "semantic_structure_equal": bool(structure_row["semantic_match"]),
            "original_semantic_sha256": str(structure_row["original_semantic_sha256"]),
            "repro_semantic_sha256": str(structure_row["repro_semantic_sha256"]),
            "build_record_sha256_match": build_row["sha256"] == sha256_file(repro_path),
            "parameter_contract_match": actual_thread == int(specification["thread_count"]) and actual_seed == int(specification["random_seed"]),
        }
        rows.append(row)
        if not all((row["prediction_within_tolerance"], row["semantic_structure_equal"], row["build_record_sha256_match"], row["parameter_contract_match"])):
            mismatches.append({"model": name, "row": row})
    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True), mismatches


def main() -> None:
    started = time.time()
    preflight = json.loads((OUT / "preflight_report.json").read_text())
    contract = json.loads((OUT / "audit_contract.json").read_text())
    binding = json.loads((OUT / "input_binding.json").read_text())
    if preflight.get("status") != "AUDIT_VERIFIED" or preflight.get("fail_count") != 0:
        raise RuntimeError("Preflight is not AUDIT_VERIFIED")
    binding_mismatches = verify_bound_records(binding)
    if binding_mismatches:
        raise RuntimeError(f"Input binding changed after preflight: {binding_mismatches}")

    stage_records = [stage_linkage(stage) for stage in STAGES]
    stage_artifact_mismatches = [item for stage in stage_records for item in stage["manifest_artifact_mismatches"]]

    train_path = ROOT / "data" / "train.csv"
    trackman_path = ROOT / "data" / "trackman_history.csv"
    train = pd.read_csv(train_path, usecols=["row_id", "season", "control_success"], low_memory=False)
    trackman = pd.read_csv(trackman_path, usecols=["season"], low_memory=False)
    target = pd.to_numeric(train["control_success"], errors="coerce").to_numpy(float)
    source_summary = {
        "train": {
            "path": str(train_path.relative_to(ROOT)), "sha256": sha256_file(train_path), "size": train_path.stat().st_size,
            "row_count": len(train), "row_id_unique_count": int(train["row_id"].nunique(dropna=False)),
            "row_id_is_unique": bool(train["row_id"].is_unique), "season_min": int(train["season"].min()),
            "season_max": int(train["season"].max()), "seasons": sorted(int(value) for value in train["season"].unique()),
            "target_finite_count": int(np.isfinite(target).sum()),
            "target_binary_count": int(np.isin(target, [0.0, 1.0]).sum()),
            "target_is_finite_binary": bool(np.isfinite(target).all() and np.isin(target, [0.0, 1.0]).all()),
        },
        "trackman": {
            "path": str(trackman_path.relative_to(ROOT)), "sha256": sha256_file(trackman_path), "size": trackman_path.stat().st_size,
            "row_count": len(trackman), "season_min": int(trackman["season"].min()), "season_max": int(trackman["season"].max()),
            "seasons": sorted(int(value) for value in trackman["season"].unique()),
        },
    }
    del train, trackman, target

    prebuild = json.loads((REPRO / "prebuild_manifest.json").read_text())
    build = json.loads((REPRO / "build_manifest.json").read_text())
    prebuild_sources = {row["path"]: row["sha256"] for row in prebuild["source_inputs"]}
    source_binding_match = all(prebuild_sources.get(summary["path"]) == summary["sha256"] for summary in source_summary.values())

    lineage, model_mismatches = build_model_lineage()
    lineage.to_csv(OUT / "model_lineage.csv", index=False)
    scope_counts = [
        {"training_scope": str(scope), "model_count": int(count)}
        for scope, count in lineage.groupby("training_scope").size().sort_index().items()
    ]
    family_counts = [
        {"family": str(family), "model_count": int(count)}
        for family, count in lineage.groupby("family").size().sort_index().items()
    ]

    prior_table = pd.read_csv(REPRO / "trackman_prior_features.csv", usecols=["season"])
    with (REPRO / "prior_type.pkl").open("rb") as handle:
        prior_lookup = pickle.load(handle)
    derived_summary = {
        "trackman_prior_rows": len(prior_table),
        "trackman_prior_target_season_min": int(prior_table["season"].min()),
        "trackman_prior_target_season_max": int(prior_table["season"].max()),
        "trackman_source_max_season": source_summary["trackman"]["season_max"],
        "trackman_2025_cutoff_valid": bool(source_summary["trackman"]["season_max"] < 2025 and int(prior_table["season"].max()) == 2025),
        "prior_type_entries": len(prior_lookup),
        "pitcher_snapshot_rows": len(pd.read_pickle(REPRO / "pitcher_snapshots.pkl")),
        "batter_snapshot_rows": len(pd.read_pickle(REPRO / "batter_snapshots.pkl")),
        "pitchmix_snapshot_rows": len(pd.read_pickle(REPRO / "pitchmix_snapshots.pkl")),
    }

    audit037 = json.loads((STAGES["037A"] / "audit_attestation.json").read_text())
    result037 = json.loads((STAGES["037A"] / "result.json").read_text())
    report038 = json.loads((REPRO / "validation_report.json").read_text())
    attestation039 = json.loads((STAGES["039A"] / "audit_attestation.json").read_text())
    diagnostic037_expected = (
        audit037.get("status") == "AUDIT_VERIFIED"
        and audit037.get("provenance_status") == "AUDIT_FAIL_PROVENANCE"
        and audit037.get("parameter_mismatch_count") == 10
        and audit037.get("binding_evidence_count") == 0
        and result037.get("provenance_status") == "AUDIT_FAIL_PROVENANCE"
    )
    reproduction038_expected = (
        build.get("model_count") == 56 and build.get("parameter_mismatch_count") == 0
        and report038.get("status") == "AUDIT_VERIFIED" and report038.get("mismatch_count") == 0
        and report038.get("prediction_models_within_tolerance") == 56
    )
    structure039_expected = (
        attestation039.get("status") == "AUDIT_VERIFIED"
        and attestation039.get("structure_status") == "STRUCTURE_EQUIVALENT"
        and attestation039.get("model_pair_count") == 56
        and attestation039.get("structure_match_count") == 56
        and attestation039.get("structure_mismatch_count") == 0
    )

    limitations = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "limitation_count": 5,
        "limitations": [
            {"code": "USER_REPORTED_SCORE_NOT_LOCALLY_RECOMPUTED", "material": True, "description": "1068.25021 is a user-provided leaderboard record tied to the preserved champion ZIP, not a locally reproducible metric."},
            {"code": "ORIGINAL_030_BUILD_BINDING_ABSENT", "material": True, "description": "The historical 030 files lacked a creation-time train/script/model binding; 038A provides a separate source-bound reproduction."},
            {"code": "LEGACY_ATTESTATION_MANIFEST_LINK_ABSENT", "material": False, "description": "038A attestation binds its report and validator but not the later seal audit manifest; 040A binds both artifacts without rewriting history."},
            {"code": "CBM_BYTES_DIFFER", "material": False, "description": "Original and reproduced CBM byte hashes differ even though fixed-probe predictions and inference-semantic structures match."},
            {"code": "REPRO_PACKAGE_NOT_BUILT_OR_SCORED", "material": True, "description": "038A was not assembled into a candidate/ZIP, run on test, submitted, or assigned the historical leaderboard score."},
        ],
    }
    write_json(OUT / "known_limitations.json", limitations)

    direct_links_ok = all(stage["report_link_match"] and stage["validator_link_match"] for stage in stage_records)
    manifest_links_ok = all(
        stage["direct_manifest_link_match"] is True
        for stage in stage_records if stage["stage"] != "038A"
    ) and next(stage for stage in stage_records if stage["stage"] == "038A")["legacy_link_gap_code"] == "LEGACY_ATTESTATION_MANIFEST_LINK_ABSENT"
    data_valid = (
        source_summary["train"]["row_id_is_unique"]
        and source_summary["train"]["target_is_finite_binary"]
        and source_summary["train"]["row_count"] == source_summary["train"]["row_id_unique_count"]
        and source_binding_match
    )
    model_valid = (
        len(lineage) == 56 and not model_mismatches
        and int(lineage["parameter_contract_match"].sum()) == 56
        and int(lineage["prediction_within_tolerance"].sum()) == 56
        and int(lineage["semantic_structure_equal"].sum()) == 56
    )
    executor_failures = []
    checks = {
        "preflight_binding": not binding_mismatches,
        "stage_artifact_hashes": not stage_artifact_mismatches,
        "stage_report_validator_links": direct_links_ok,
        "stage_manifest_links_or_disclosed_gap": manifest_links_ok,
        "source_data": data_valid,
        "diagnostic_037A": diagnostic037_expected,
        "reproduction_038A": reproduction038_expected,
        "structure_039A": structure039_expected,
        "model_lineage": model_valid,
        "derived_cutoff": derived_summary["trackman_2025_cutoff_valid"],
        "champion_zip": sha256_file(ZIP) == contract["expected_champion_zip_sha256"],
        "zero_candidate_leaf_gate": contract["candidate_count"] == contract["leaf_count"] == contract["gate_count"] == 0,
    }
    executor_failures = [name for name, passed in checks.items() if not passed]
    lineage_status = "PROVENANCE_DOSSIER_VERIFIED" if not executor_failures else "AUDIT_FAIL_PROVENANCE"
    lineage_index = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_status": "PENDING_VALIDATION",
        "lineage_status": lineage_status,
        "stages": stage_records,
        "source_data": source_summary,
        "training_environment": prebuild["environment"],
        "model_summary": {
            "model_count": len(lineage),
            "thread_4_count": int(lineage["expected_thread_count"].eq(4).sum()),
            "thread_3_count": int(lineage["expected_thread_count"].eq(3).sum()),
            "parameter_contract_match_count": int(lineage["parameter_contract_match"].sum()),
            "prediction_parity_count": int(lineage["prediction_within_tolerance"].sum()),
            "structure_equivalent_count": int(lineage["semantic_structure_equal"].sum()),
            "cbm_byte_equal_count": int(lineage["cbm_byte_equal"].sum()),
            "scope_counts": scope_counts,
            "family_counts": family_counts,
        },
        "derived_artifacts": derived_summary,
        "official_artifact": {
            "path": str(ZIP.relative_to(ROOT)), "sha256": sha256_file(ZIP), "size": ZIP.stat().st_size,
            "leaderboard_score": contract["official_score_value"],
            "score_evidence_class": contract["official_score_evidence_class"],
        },
        "known_limitations_path": str((OUT / "known_limitations.json").relative_to(ROOT)),
        "known_limitation_count": limitations["limitation_count"],
        "executor_checks": checks,
        "executor_failure_count": len(executor_failures),
        "executor_failures": executor_failures,
        "candidate_count": 0,
        "leaf_count": 0,
        "gate_count": 0,
        "test_read_performed": False,
        "test_inference_performed": False,
        "training_performed": False,
        "model_modified": False,
        "candidate_created": False,
        "zip_created": False,
        "submission_performed": False,
    }
    write_json(OUT / "lineage_index.json", lineage_index)

    embedded = json.dumps(lineage_index, ensure_ascii=False, sort_keys=True)
    dossier_lines = [
        f"# {EXPERIMENT_ID}", "",
        f"- execution status: `{lineage_index['execution_status']}`",
        f"- lineage status: `{lineage_status}`",
        f"- stages: `{len(stage_records)}` (`037A → 038A → 039A`)",
        f"- models: `{len(lineage)}` (thread 4: `{lineage_index['model_summary']['thread_4_count']}`, thread 3: `{lineage_index['model_summary']['thread_3_count']}`)",
        f"- parameter/prediction/structure matches: `{lineage_index['model_summary']['parameter_contract_match_count']}/{lineage_index['model_summary']['prediction_parity_count']}/{lineage_index['model_summary']['structure_equivalent_count']}`",
        f"- known limitations: `{limitations['limitation_count']}`",
        "- candidate/leaf/gate: `0/0/0`",
        "- test/training/model-write/ZIP/submission: `false/false/false/false/false`",
        "", "## Lineage", "",
        "1. `037A`: calculation verified; historical artifact provenance failed because creation-time binding was absent and 10 embedded thread values differed from the then-current script.",
        "2. `038A`: source-bound 56-model full-train reproduction; parameters and fixed-source probe predictions verified.",
        "3. `039A`: 56/56 inference-semantic CatBoost structures verified exact.",
        "", "## Score ownership", "",
        f"The value `{contract['official_score_value']}` is user-reported and belongs only to `{ZIP.relative_to(ROOT)}` with SHA-256 `{sha256_file(ZIP)}`. It is not a locally recomputed score and is not transferred to 038A.",
        "", "## Known limitations", "",
    ]
    dossier_lines.extend(f"- `{item['code']}`: {item['description']}" for item in limitations["limitations"])
    dossier_lines.extend(["", "<!-- LINEAGE_INDEX_JSON_BEGIN", embedded, "LINEAGE_INDEX_JSON_END -->"])
    (OUT / "reproducibility_dossier.md").write_text("\n".join(dossier_lines) + "\n")

    result = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "execution_status": "PENDING_VALIDATION",
        "lineage_status": lineage_status,
        "stage_count": len(stage_records),
        "stage_artifact_mismatch_count": len(stage_artifact_mismatches),
        "stage_link_mismatch_count": int(not direct_links_ok) + int(not manifest_links_ok),
        "model_count": len(lineage),
        "model_mismatch_count": len(model_mismatches),
        "source_binding_match": source_binding_match,
        "known_limitation_count": limitations["limitation_count"],
        "legacy_link_gap_count": sum(stage["legacy_link_gap_code"] is not None for stage in stage_records),
        "executor_check_count": len(checks),
        "executor_failure_count": len(executor_failures),
        "candidate_count": 0, "leaf_count": 0, "gate_count": 0,
        "test_read_performed": False, "test_inference_performed": False,
        "training_performed": False, "model_modified": False,
        "candidate_created": False, "zip_created": False, "submission_performed": False,
        "elapsed_seconds": time.time() - started,
    }
    write_json(OUT / "result.json", result)
    result_embedded = json.dumps(result, ensure_ascii=False, sort_keys=True)
    (OUT / "result.md").write_text(
        f"# {EXPERIMENT_ID} result\n\n- execution status: `{result['execution_status']}`\n"
        f"- lineage status: `{lineage_status}`\n- stages/models: `{len(stage_records)}/{len(lineage)}`\n"
        f"- artifact/link/model mismatch: `{len(stage_artifact_mismatches)}/{result['stage_link_mismatch_count']}/{len(model_mismatches)}`\n"
        "- candidate/leaf/gate: `0/0/0`\n- test/training/ZIP: `false/false/false`\n\n"
        f"<!-- RESULT_JSON_BEGIN\n{result_embedded}\nRESULT_JSON_END -->\n"
    )

    upstream_critical = []
    stage_names = {
        "037A": ["audit_manifest.json", "validation_report.json", "audit_attestation.json", "result.json", "component_inventory.csv", "package_inventory.csv", "lookup_snapshot_report.json"],
        "038A": ["audit_manifest.json", "validation_report.json", "audit_attestation.json", "result.json", "prebuild_manifest.json", "build_manifest.json", "prediction_parity.csv"],
        "039A": ["audit_manifest.json", "validation_report.json", "audit_attestation.json", "result.json", "structure_inventory.csv", "verification_inventory.csv", "structure_mismatches.json"],
    }
    for stage, names in stage_names.items():
        upstream_critical.extend(STAGES[stage] / name for name in names)
    artifact_paths = [
        ROOT / "scripts" / "preflight_ref4_provenance_dossier_040a.py",
        ROOT / "scripts" / "run_ref4_provenance_dossier_040a.py",
        ROOT / "scripts" / "verify_ref4_provenance_dossier_040a.py",
        OUT / "audit_contract.json", OUT / "contract_snapshot.md", OUT / "input_binding.json",
        OUT / "preflight_report.json", OUT / "preflight_report.md", OUT / "lineage_index.json",
        OUT / "model_lineage.csv", OUT / "known_limitations.json", OUT / "reproducibility_dossier.md",
        OUT / "result.json", OUT / "result.md", train_path, trackman_path, ZIP,
    ] + upstream_critical + sorted(ORIGINAL.glob("*.cbm")) + sorted(REPRO.glob("*.cbm"))
    records = {str(path.relative_to(ROOT)): record(path) for path in artifact_paths}
    audit = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "PENDING_VALIDATION",
        "artifact_count": len(records),
        "artifacts": records,
        "stage_count": len(stage_records),
        "model_count": len(lineage),
        "stage_artifact_mismatch_count": len(stage_artifact_mismatches),
        "model_mismatch_count": len(model_mismatches),
        "candidate_count": 0, "leaf_count": 0, "gate_count": 0,
        "test_read_performed": False, "test_inference_performed": False,
        "training_performed": False, "zip_created": False,
    }
    write_json(OUT / "audit_manifest.json", audit)
    print(json.dumps({
        "execution_status": result["execution_status"], "lineage_status": lineage_status,
        "stages": len(stage_records), "models": len(lineage),
        "stage_artifact_mismatches": len(stage_artifact_mismatches),
        "model_mismatches": len(model_mismatches), "executor_checks": len(checks),
        "executor_failures": len(executor_failures), "elapsed_seconds": result["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
