#!/usr/bin/env python3
"""Preflight contract and immutable input binding for 039A."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "REF4-CHAMPION-REPRO-STRUCTURE-039A"
ORIGINAL = ROOT / "model" / "REF4-CHAMPION-STACK-030"
REPRO = ROOT / "model" / "REF4-CHAMPION-REPRO-038A"
OUT = ROOT / "model" / EXPERIMENT_ID
ZIP = ROOT / "output" / "submit_ref4_champion_030.zip"
EXPECTED_ZIP_SHA256 = "ca6af4bdf26d4bc79d503cdc6a09b0057a7ea9bdd77e29012f9e459a50cd66b8"
SEMANTIC_FIELDS = ["features_info", "ctr_data", "oblivious_trees", "scale_and_bias", "model_info.class_params"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path.relative_to(ROOT)), "size": path.stat().st_size, "sha256": sha256_file(path)}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Refusing to reuse non-empty audit directory: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)

    original = sorted(ORIGINAL.glob("*.cbm")) if ORIGINAL.is_dir() else []
    repro = sorted(REPRO.glob("*.cbm")) if REPRO.is_dir() else []
    original_names = {path.name for path in original}
    repro_names = {path.name for path in repro}
    upstream_audit_path = REPRO / "audit_manifest.json"
    upstream_validation_path = REPRO / "validation_report.json"
    upstream_audit = json.loads(upstream_audit_path.read_text()) if upstream_audit_path.is_file() else {}
    upstream_validation = json.loads(upstream_validation_path.read_text()) if upstream_validation_path.is_file() else {}

    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, actual: Any) -> None:
        checks.append({"check_id": check_id, "checked": True, "status": "PASS" if passed else "FAIL", "actual": actual})

    check("workspace_root", Path.cwd().resolve() == ROOT, str(Path.cwd().resolve()))
    check("constraints_present", (ROOT / "01_제약과금지사항.md").is_file(), "01_제약과금지사항.md")
    check("current_reference_present", (ROOT / "start03_reference.md").is_file(), "start03_reference.md")
    check("current_worklog_present", (ROOT / "start04_uptostage.md").is_file(), "start04_uptostage.md")
    check("deprecated_handover_not_authoritative", True, "08_Gemini_작업위임서.md explicitly superseded by user")
    check("original_directory", ORIGINAL.is_dir(), str(ORIGINAL.relative_to(ROOT)))
    check("repro_directory", REPRO.is_dir(), str(REPRO.relative_to(ROOT)))
    check("original_model_count", len(original) == 56, len(original))
    check("repro_model_count", len(repro) == 56, len(repro))
    check("model_name_set", original_names == repro_names and len(original_names) == 56, {
        "missing_in_repro": sorted(original_names - repro_names), "extra_in_repro": sorted(repro_names - original_names)
    })
    check("upstream_audit", upstream_audit.get("status") == "AUDIT_VERIFIED" and upstream_audit.get("model_count") == 56 and upstream_audit.get("validation_mismatch_count") == 0, {
        "status": upstream_audit.get("status"), "model_count": upstream_audit.get("model_count"),
        "validation_mismatch_count": upstream_audit.get("validation_mismatch_count")
    })
    check("upstream_validation", upstream_validation.get("status") == "AUDIT_VERIFIED" and upstream_validation.get("mismatch_count") == 0, {
        "status": upstream_validation.get("status"), "mismatch_count": upstream_validation.get("mismatch_count")
    })
    zip_hash = sha256_file(ZIP) if ZIP.is_file() else None
    check("champion_zip_preserved", zip_hash == EXPECTED_ZIP_SHA256, zip_hash)
    check("no_039_candidate", not (ROOT / "candidate" / EXPERIMENT_ID).exists(), (ROOT / "candidate" / EXPERIMENT_ID).exists())
    zip_matches = sorted(str(path.relative_to(ROOT)) for path in (ROOT / "output").glob("*039A*.zip"))
    check("no_039_zip", not zip_matches, zip_matches)
    check("semantic_field_count", len(SEMANTIC_FIELDS) == 5, SEMANTIC_FIELDS)
    check("expected_pair_count", len(original_names & repro_names) == 56, len(original_names & repro_names))
    check("expected_json_export_count", 2 * len(original_names & repro_names) == 112, 2 * len(original_names & repro_names))

    start04 = (ROOT / "start04_uptostage.md").read_text()
    marker = "## 28. 실행 계약: Champion Reproduction Structure Audit"
    if start04.count(marker) != 1:
        contract_snapshot = ""
    else:
        contract_snapshot = marker + start04.split(marker, 1)[1]
    (OUT / "contract_snapshot.md").write_text(contract_snapshot)
    checks.append({
        "check_id": "contract_section_unique", "checked": True,
        "status": "PASS" if start04.count(marker) == 1 else "FAIL",
        "actual": start04.count(marker),
    })
    failed = [row["check_id"] for row in checks if row["status"] != "PASS"]

    contract = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "declared_at_utc": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "all 56 original/reproduction pairs have exactly equal inference-semantic CatBoost JSON payloads",
        "semantic_fields": SEMANTIC_FIELDS,
        "excluded_nonsemantic_model_info": ["catboost_version_info", "model_guid", "output_options", "params", "train_finish_time", "training"],
        "numeric_tolerance": 0.0,
        "expected_model_pair_count": 56,
        "expected_json_export_count": 112,
        "candidate_count": 0,
        "leaf_count": 0,
        "gate_count": 0,
        "success_status": "STRUCTURE_EQUIVALENT",
        "failure_status": "AUDIT_FAIL_STRUCTURE",
        "training_allowed": False,
        "test_read_allowed": False,
        "test_inference_allowed": False,
        "candidate_creation_allowed": False,
        "zip_creation_allowed": False,
        "submission_allowed": False,
        "official_score_reference": 1068.25021,
        "official_score_owner": "output/submit_ref4_champion_030.zip",
        "commands": {
            "preflight": ".venv/bin/python scripts/preflight_ref4_structure_039a.py",
            "executor": ".venv/bin/python scripts/run_ref4_structure_039a.py",
            "validator": ".venv/bin/python scripts/verify_ref4_structure_039a.py",
        },
        "planned_change_files": [
            "scripts/preflight_ref4_structure_039a.py", "scripts/run_ref4_structure_039a.py",
            "scripts/verify_ref4_structure_039a.py", "start04_uptostage.md",
            f"model/{EXPERIMENT_ID}/*",
        ],
    }
    write_json(OUT / "audit_contract.json", contract)

    inputs = {
        "original_models": [file_record(path) for path in original],
        "repro_models": [file_record(path) for path in repro],
        "upstream_artifacts": [file_record(upstream_audit_path), file_record(upstream_validation_path)],
        "champion_zip": file_record(ZIP),
        "governing_documents": [
            file_record(ROOT / "01_제약과금지사항.md"),
            file_record(ROOT / "start03_reference.md"),
            file_record(ROOT / "start04_uptostage.md"),
        ],
    }
    write_json(OUT / "input_binding.json", inputs)
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "AUDIT_VERIFIED" if not failed else "BLOCKED",
        "check_count": len(checks),
        "pass_count": sum(row["status"] == "PASS" for row in checks),
        "fail_count": len(failed),
        "failed_checks": failed,
        "model_pair_count": len(original_names & repro_names),
        "json_export_count_expected": 2 * len(original_names & repro_names),
        "candidate_count": 0,
        "leaf_count": 0,
        "gate_count": 0,
        "test_read_performed": False,
        "test_inference_performed": False,
        "training_performed": False,
        "zip_created": False,
        "checks": checks,
    }
    write_json(OUT / "preflight_report.json", report)
    (OUT / "preflight_report.md").write_text(
        f"# {EXPERIMENT_ID} preflight\n\n"
        f"- status: `{report['status']}`\n"
        f"- checks: `{report['pass_count']}/{report['check_count']}`\n"
        f"- model pairs: `{report['model_pair_count']}`\n"
        f"- expected JSON exports: `{report['json_export_count_expected']}`\n"
        "- candidate/leaf/gate: `0/0/0`\n"
        "- test/training/ZIP: `false/false/false`\n"
    )
    print(json.dumps({
        "status": report["status"], "checks": report["check_count"], "passed": report["pass_count"],
        "failed": report["fail_count"], "model_pairs": report["model_pair_count"],
        "expected_json_exports": report["json_export_count_expected"],
    }, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
