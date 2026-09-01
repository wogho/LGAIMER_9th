#!/usr/bin/env python3
"""Package 110A only after its production technical audit is verified."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "model/REF4-TEMPORAL-CROSSFIT-MOE-110A-R2"
PROD = DEST / "production_package"
ZIP = ROOT / "output/submit_ref4_temporal_crossfit_moe_110A_R2.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    audit = json.loads((DEST / "technical_validation_report.json").read_text())
    if audit.get("status") != "AUDIT_VERIFIED" or audit.get("mismatch_count") != 0:
        raise RuntimeError("production technical audit is not verified")
    if ZIP.exists():
        raise RuntimeError(f"refusing to overwrite existing ZIP: {ZIP}")
    files = sorted(path for path in PROD.rglob("*") if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc")
    roots = {path.relative_to(PROD).parts[0] for path in files}
    if not {"script.py", "requirements.txt", "model", "src"}.issubset(roots):
        raise RuntimeError(f"invalid roots: {roots}")
    if any(path.relative_to(PROD).parts[0] in {"data", "output"} for path in files):
        raise RuntimeError("runtime data/output leaked into production directory")
    with zipfile.ZipFile(ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, path.relative_to(PROD).as_posix())
    with zipfile.ZipFile(ZIP) as archive:
        bad = archive.testzip()
        members = archive.namelist()
    if bad is not None:
        raise RuntimeError(f"corrupt member: {bad}")
    record = {
        "experiment_id": "REF4-TEMPORAL-CROSSFIT-MOE-110A-R2",
        "status": "PENDING_ZIP_AUDIT",
        "zip": str(ZIP.relative_to(ROOT)),
        "zip_sha256": sha256(ZIP),
        "zip_bytes": ZIP.stat().st_size,
        "zip_mib": ZIP.stat().st_size / (1024 * 1024),
        "member_count": len(members),
        "source_file_count": len(files),
        "technical_validation_report_sha256": sha256(DEST / "technical_validation_report.json"),
        "technical_audit_manifest_sha256": sha256(DEST / "technical_audit_manifest.json"),
    }
    (DEST / "package_record.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
