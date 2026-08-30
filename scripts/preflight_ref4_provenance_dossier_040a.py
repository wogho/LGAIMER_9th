#!/usr/bin/env python3
"""Preflight and immutable input binding for the 040A provenance dossier."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    return {"path": str(path.relative_to(ROOT)), "size": path.stat().st_size, "sha256": sha256_file(path)}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    if OUT.exists() and any(OUT.iterdir()):
        raise RuntimeError(f"Refusing to reuse non-empty output directory: {OUT}")
    OUT.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, actual: Any) -> None:
        checks.append({"check_id": check_id, "checked": True, "status": "PASS" if passed else "FAIL", "actual": actual})

    check("workspace_root", Path.cwd().resolve() == ROOT, str(Path.cwd().resolve()))
    check("constraints_present", (ROOT / "01_제약과금지사항.md").is_file(), True)
    check("current_reference_present", (ROOT / "start03_reference.md").is_file(), True)
    check("current_worklog_present", (ROOT / "start04_uptostage.md").is_file(), True)
    check("deprecated_handover_not_authoritative", True, "08_Gemini_작업위임서.md superseded by explicit user instruction")
    check("stage_count", len(STAGES) == 3, len(STAGES))
    for stage, directory in STAGES.items():
        check(f"stage_{stage}_directory", directory.is_dir(), str(directory.relative_to(ROOT)))
        required = [directory / name for name in ("audit_manifest.json", "validation_report.json", "audit_attestation.json")]
        check(f"stage_{stage}_critical_files", all(path.is_file() for path in required), [path.name for path in required if not path.is_file()])
    original_models = sorted(ORIGINAL.glob("*.cbm")) if ORIGINAL.is_dir() else []
    repro_models = sorted(REPRO.glob("*.cbm")) if REPRO.is_dir() else []
    check("original_model_count", len(original_models) == 56, len(original_models))
    check("repro_model_count", len(repro_models) == 56, len(repro_models))
    check("model_name_set", {path.name for path in original_models} == {path.name for path in repro_models}, {
        "original": len(original_models), "repro": len(repro_models)
    })
    check("official_train_present", (ROOT / "data" / "train.csv").is_file(), True)
    check("trackman_present", (ROOT / "data" / "trackman_history.csv").is_file(), True)
    zip_hash = sha256_file(ZIP) if ZIP.is_file() else None
    check("champion_zip_preserved", zip_hash == EXPECTED_ZIP_SHA256, zip_hash)
    check("no_040_candidate", not (ROOT / "candidate" / EXPERIMENT_ID).exists(), (ROOT / "candidate" / EXPERIMENT_ID).exists())
    zip_matches = sorted(str(path.relative_to(ROOT)) for path in (ROOT / "output").glob("*040A*.zip"))
    check("no_040_zip", not zip_matches, zip_matches)
    start04 = (ROOT / "start04_uptostage.md").read_text()
    marker = "## 30. 실행 계약: Champion Provenance Dossier"
    check("contract_section_unique", start04.count(marker) == 1, start04.count(marker))
    snapshot = marker + start04.split(marker, 1)[1] if start04.count(marker) == 1 else ""
    (OUT / "contract_snapshot.md").write_text(snapshot)

    contract = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "declared_at_utc": datetime.now(timezone.utc).isoformat(),
        "objective": "independently bind and consolidate the 037A-038A-039A provenance chain",
        "stage_ids": list(STAGES),
        "expected_stage_count": 3,
        "expected_model_count": 56,
        "candidate_count": 0,
        "leaf_count": 0,
        "gate_count": 0,
        "official_score_value": 1068.25021,
        "official_score_evidence_class": "user_reported_leaderboard_result_not_locally_recomputed",
        "official_score_artifact": "output/submit_ref4_champion_030.zip",
        "expected_champion_zip_sha256": EXPECTED_ZIP_SHA256,
        "known_legacy_link_gap": "038A attestation lacks a direct audit_manifest_sha256 field because the seal manifest was created later",
        "success_status": "PROVENANCE_DOSSIER_VERIFIED",
        "training_allowed": False,
        "model_modification_allowed": False,
        "test_read_allowed": False,
        "test_inference_allowed": False,
        "candidate_creation_allowed": False,
        "zip_creation_allowed": False,
        "submission_allowed": False,
        "planned_outputs": [
            "lineage_index.json", "model_lineage.csv", "known_limitations.json",
            "reproducibility_dossier.md", "result.json", "result.md",
            "audit_manifest.json", "validation_report.json", "audit_attestation.json",
        ],
        "commands": {
            "preflight": ".venv/bin/python scripts/preflight_ref4_provenance_dossier_040a.py",
            "executor": ".venv/bin/python scripts/run_ref4_provenance_dossier_040a.py",
            "validator": ".venv/bin/python scripts/verify_ref4_provenance_dossier_040a.py",
        },
    }
    write_json(OUT / "audit_contract.json", contract)

    stage_critical: list[dict[str, Any]] = []
    for stage, directory in STAGES.items():
        for name in ("audit_manifest.json", "validation_report.json", "audit_attestation.json", "result.json"):
            path = directory / name
            if path.is_file():
                stage_critical.append({"stage": stage, **record(path)})
    input_binding = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "stage_critical_artifacts": stage_critical,
        "source_inputs": [record(ROOT / "data" / "train.csv"), record(ROOT / "data" / "trackman_history.csv")],
        "model_inputs": [record(path) for path in original_models + repro_models],
        "champion_zip": record(ZIP),
        "governing_documents": [
            record(ROOT / "01_제약과금지사항.md"), record(ROOT / "start03_reference.md"),
            record(ROOT / "start04_uptostage.md"),
        ],
    }
    write_json(OUT / "input_binding.json", input_binding)
    failed = [row["check_id"] for row in checks if row["status"] != "PASS"]
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "AUDIT_VERIFIED" if not failed else "BLOCKED",
        "check_count": len(checks),
        "pass_count": len(checks) - len(failed),
        "fail_count": len(failed),
        "failed_checks": failed,
        "stage_count": len(STAGES),
        "model_count_original": len(original_models),
        "model_count_repro": len(repro_models),
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
        f"- status: `{report['status']}`\n- checks: `{report['pass_count']}/{report['check_count']}`\n"
        f"- stages: `{report['stage_count']}`\n- models original/repro: `{len(original_models)}/{len(repro_models)}`\n"
        "- candidate/leaf/gate: `0/0/0`\n- test/training/ZIP: `false/false/false`\n"
    )
    print(json.dumps({
        "status": report["status"], "checks": report["check_count"], "passed": report["pass_count"],
        "failed": report["fail_count"], "stages": report["stage_count"],
        "models_original": len(original_models), "models_repro": len(repro_models),
    }, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
