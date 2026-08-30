#!/usr/bin/env python3
"""Preflight for the read-only REF4 champion provenance audit."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "REF4-CHAMPION-ERA-PROVENANCE-037A"
OUT = ROOT / "model" / EXPERIMENT_ID
SOURCE = ROOT / "model" / "REF4-CHAMPION-STACK-030"
CANDIDATE = ROOT / "candidate" / "REF4-CHAMPION-STACK-030"
ZIP = ROOT / "output" / "submit_ref4_champion_030.zip"
PRIOR = ROOT / "model" / "REF4-F-ERA-CAL-036A"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, actual: object) -> None:
        checks.append({"name": name, "checked": True, "passed": bool(passed), "actual": actual})

    required = [SOURCE, CANDIDATE, ZIP, ROOT / "data" / "train.csv", ROOT / "data" / "trackman_history.csv", ROOT / "scripts" / "train_and_package_ref4_champion_030.py", ROOT / "scripts" / "build_ref4_trackman_030.py", ROOT / "start03_reference.md", ROOT / "start04_uptostage.md", ROOT / "01_제약과금지사항.md", PRIOR / "audit_manifest.json", PRIOR / "validation_report.json", PRIOR / "audit_attestation.json"]
    for path in required:
        check(f"exists:{path.relative_to(ROOT)}", path.exists(), path.exists())
    prior_attestation = json.loads((PRIOR / "audit_attestation.json").read_text(encoding="utf-8"))
    check("prior_validation_verified", prior_attestation["status"] == "AUDIT_VERIFIED" and prior_attestation["mismatch_count"] == 0, {"status": prior_attestation["status"], "mismatch": prior_attestation["mismatch_count"]})
    check("prior_manifest_hash", sha256_path(PRIOR / "audit_manifest.json") == prior_attestation["manifest_sha256"], sha256_path(PRIOR / "audit_manifest.json"))
    check("prior_report_hash", sha256_path(PRIOR / "validation_report.json") == prior_attestation["validation_report_sha256"], sha256_path(PRIOR / "validation_report.json"))
    train_seasons = sorted(pd.read_csv(ROOT / "data" / "train.csv", usecols=["season"])["season"].astype(int).unique().tolist())
    trackman_seasons = sorted(pd.read_csv(ROOT / "data" / "trackman_history.csv", usecols=["season"])["season"].astype(int).unique().tolist())
    check("train_seasons", train_seasons == list(range(2019, 2025)), train_seasons)
    check("trackman_seasons", trackman_seasons == list(range(2019, 2025)), trackman_seasons)
    source_files = sorted(path for path in SOURCE.iterdir() if path.is_file())
    candidate_files = sorted(path for path in CANDIDATE.rglob("*") if path.is_file())
    cbm_files = sorted(SOURCE.glob("*.cbm"))
    with zipfile.ZipFile(ZIP) as archive:
        zip_members = archive.namelist()
    check("source_files_nonempty", len(source_files) > 0, len(source_files))
    check("candidate_files_nonempty", len(candidate_files) > 0, len(candidate_files))
    check("cbm_files_nonempty", len(cbm_files) > 0, len(cbm_files))
    check("zip_members_nonempty_unique", len(zip_members) > 0 and len(zip_members) == len(set(zip_members)), {"members": len(zip_members), "unique": len(set(zip_members))})
    check("champion_zip_hash", sha256_path(ZIP) == "ca6af4bdf26d4bc79d503cdc6a09b0057a7ea9bdd77e29012f9e459a50cd66b8", sha256_path(ZIP))
    contract = {"experiment_id": EXPERIMENT_ID, "audit_target": "REF4-CHAMPION-STACK-030", "candidate_count": 0, "actual_leaf_count": 0, "gate_checks_count": 0, "model_count_created": 0, "audited_cbm_count": len(cbm_files), "source_file_count": len(source_files), "candidate_file_count": len(candidate_files), "zip_member_count": len(zip_members), "expected_train_seasons": train_seasons, "expected_trackman_seasons": trackman_seasons, "test_read_performed": False, "test_inference_performed": False, "training_performed": False, "zip_created": False}
    write_json(OUT / "audit_contract.json", contract)
    mismatches = [str(row["name"]) for row in checks if not row["passed"]]
    status = "AUDIT_VERIFIED" if not mismatches else "FAIL"
    report = {"experiment_id": EXPERIMENT_ID, "status": status, "checked_count": len(checks), "passed_count": sum(bool(row["passed"]) for row in checks), "mismatch_count": len(mismatches), "mismatches": mismatches, **{key: contract[key] for key in ("candidate_count", "actual_leaf_count", "gate_checks_count", "model_count_created", "audited_cbm_count", "source_file_count", "candidate_file_count", "zip_member_count")}, "start04_sha256_at_contract": sha256_path(ROOT / "start04_uptostage.md"), "checks": checks}
    write_json(OUT / "preflight_report.json", report)
    (OUT / "preflight_report.md").write_text("\n".join([f"# {EXPERIMENT_ID} preflight", "", f"- status: `{status}`", f"- checked: `{report['passed_count']}/{report['checked_count']}`", f"- mismatch: `{report['mismatch_count']}`", f"- audited CBM: `{report['audited_cbm_count']}`", "- candidate/leaf/gate/new-model: `0/0/0/0`", "- test-read/test-inference/training/ZIP: `false/false/false/false`"]) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("experiment_id", "status", "checked_count", "passed_count", "mismatch_count", "audited_cbm_count", "source_file_count", "candidate_file_count", "zip_member_count")}, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
