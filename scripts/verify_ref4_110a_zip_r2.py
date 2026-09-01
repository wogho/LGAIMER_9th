#!/usr/bin/env python3
"""Final independent ZIP audit for REF4 110A R2."""
from __future__ import annotations

import ast
import hashlib
import json
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "model/REF4-TEMPORAL-CROSSFIT-MOE-110A-R2"
PROD = DEST / "production_package"
ZIP = ROOT / "output/submit_ref4_temporal_crossfit_moe_110A_R2.zip"
PYTHON = ROOT / ".venv/bin/python"
DUMMY = ROOT / "data/test.csv"
ROLLBACK = ROOT / "output/submit_ref4_super_ensemble_109C.zip"
ROLLBACK_SHA = "03e874949ae7172af0dab16d9c3f52de94d5ac9256e571ddedd377b785d9634f"
FORBIDDEN = {"socket", "requests", "urllib", "http", "ftplib", "paramiko"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    checks = []

    def check(name: str, passed: bool, actual=None) -> None:
        checks.append({"name": name, "checked": True, "passed": bool(passed), "actual": actual})

    package = json.loads((DEST / "package_record.json").read_text())
    technical = json.loads((DEST / "technical_validation_report.json").read_text())
    technical_manifest = json.loads((DEST / "technical_audit_manifest.json").read_text())
    performance = json.loads((ROOT / "model/REF4-110-ORIGINAL-R2/validation_report.json").read_text())
    check("performance_winner", performance.get("status") == "AUDIT_VERIFIED" and performance.get("provisional_winner") == "110A", performance.get("provisional_winner"))
    check("technical_verified", technical.get("status") == "AUDIT_VERIFIED" and technical.get("mismatch_count") == 0, technical.get("status"))
    check("zip_exists", ZIP.exists(), ZIP.exists())
    zip_hash = sha256(ZIP)
    check("zip_record_hash", zip_hash == package.get("zip_sha256"), zip_hash)
    check("zip_size", ZIP.stat().st_size <= 500 * 1024 * 1024, ZIP.stat().st_size)
    check("rollback_integrity", sha256(ROLLBACK) == ROLLBACK_SHA, sha256(ROLLBACK))

    with zipfile.ZipFile(ZIP) as archive:
        bad = archive.testzip()
        members = archive.namelist()
        check("zip_crc", bad is None, bad)
        check("member_count", len(members) == package.get("member_count") == 170, len(members))
        check("members_unique", len(members) == len(set(members)), len(set(members)))
        unsafe = [name for name in members if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts]
        check("safe_member_paths", not unsafe, unsafe)
        roots = {PurePosixPath(name).parts[0] for name in members}
        check("root_layout", {"script.py", "requirements.txt", "model", "src"}.issubset(roots), sorted(roots))
        check("no_runtime_data", not ({"data", "output"} & roots), sorted({"data", "output"} & roots))
        mismatches = []
        for name in members:
            source = PROD / PurePosixPath(name)
            if not source.is_file():
                mismatches.append(f"missing:{name}")
                continue
            digest = hashlib.sha256(archive.read(name)).hexdigest()
            if digest != sha256(source):
                mismatches.append(f"hash:{name}")
        check("zip_source_binding", not mismatches, mismatches[:20])
        script_hash = hashlib.sha256(archive.read("script.py")).hexdigest()
        check("technical_script_binding", script_hash == technical_manifest["artifacts"]["script.py"], script_hash)

    with tempfile.TemporaryDirectory(prefix="ref4_110a_zip_audit_") as temporary:
        extracted = Path(temporary)
        with zipfile.ZipFile(ZIP) as archive:
            archive.extractall(extracted)
        forbidden_found = []
        for source in extracted.rglob("*.py"):
            tree = ast.parse(source.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                module = None
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] in FORBIDDEN:
                            forbidden_found.append(f"{source.relative_to(extracted)}:{alias.name}")
                elif isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] in FORBIDDEN:
                    forbidden_found.append(f"{source.relative_to(extracted)}:{node.module}")
        check("static_network", not forbidden_found, forbidden_found)
        (extracted / "data").mkdir()
        (extracted / "output").mkdir()
        shutil.copy2(DUMMY, extracted / "data/test.csv")
        started = time.monotonic()
        completed = subprocess.run([str(PYTHON), "script.py"], cwd=extracted, text=True, capture_output=True, timeout=600)
        elapsed = time.monotonic() - started
        check("extracted_dummy_returncode", completed.returncode == 0, completed.stderr[-2000:])
        submission = extracted / "output/submission.csv"
        check("extracted_dummy_output", submission.is_file(), submission.exists())
        if submission.is_file():
            result = pd.read_csv(submission, dtype={"row_id": str})
            values = result["control_success"].to_numpy(float)
            check("extracted_dummy_rows", len(result) == 5, len(result))
            check("extracted_dummy_values", np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), [float(values.min()), float(values.max())])
        check("extracted_dummy_runtime", elapsed <= 600.0, elapsed)

    failed = [item for item in checks if not item["passed"]]
    status = "AUDIT_VERIFIED" if not failed else "MISMATCH"
    report = {
        "experiment_id": "REF4-TEMPORAL-CROSSFIT-MOE-110A-R2",
        "status": status,
        "checked_count": len(checks),
        "passed_count": len(checks) - len(failed),
        "mismatch_count": len(failed),
        "failures": failed,
        "zip": str(ZIP.relative_to(ROOT)),
        "zip_sha256": zip_hash,
        "zip_bytes": ZIP.stat().st_size,
        "zip_mib": ZIP.stat().st_size / (1024 * 1024),
        "member_count": len(members),
        "extracted_dummy_runtime_seconds": elapsed,
        "technical_stress_253507_runtime_seconds": technical["stress_253507_runtime_seconds"],
        "row_independence_max_diff": technical["row_independence_max_diff"],
        "column_permutation_max_diff": technical["column_permutation_max_diff"],
        "checks": checks,
    }
    report_path = DEST / "zip_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    manifest = {
        "experiment_id": report["experiment_id"],
        "status": status,
        "zip_sha256": zip_hash,
        "package_record_sha256": sha256(DEST / "package_record.json"),
        "technical_audit_manifest_sha256": sha256(DEST / "technical_audit_manifest.json"),
        "zip_validation_report_sha256": sha256(report_path),
        "packager_sha256": sha256(ROOT / "scripts/package_ref4_110a_r2.py"),
        "validator_sha256": sha256(Path(__file__)),
    }
    manifest_path = DEST / "final_audit_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    attestation = {
        "experiment_id": report["experiment_id"],
        "status": status,
        "manifest_sha256": sha256(manifest_path),
        "validation_report_sha256": sha256(report_path),
        "validator_sha256": sha256(Path(__file__)),
        "checked_count": report["checked_count"],
        "passed_count": report["passed_count"],
        "mismatch_count": report["mismatch_count"],
        "zip_sha256": zip_hash,
    }
    (DEST / "final_audit_attestation.json").write_text(json.dumps(attestation, indent=2) + "\n")
    print(json.dumps(attestation, indent=2), flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
