#!/usr/bin/env python3
"""Build the immutable validation-audit manifest for REF4-126.

This script does not train, predict on test data, or create a submission.  It
records the locally retrieved Colab artifacts, the frozen inputs/code, and a
fresh read-only rclone comparison against the shared Google Drive directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "REF4-JM-R-RESIDUAL-STRICT-GPU-126"
EXPERIMENT_DIR = ROOT / "model" / EXPERIMENT_ID
RAW_DIR = EXPERIMENT_DIR / "remote_raw"
MANIFEST_PATH = EXPERIMENT_DIR / "audit_manifest.json"
REMOTE_CHECK_PATH = EXPERIMENT_DIR / "remote_transfer_verification.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run_remote_checks() -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for name in ("results", "logs", "checkpoints"):
        command = [
            "rclone",
            "check",
            f"lgaimer126_gdrive:REF4_126/{name}",
            str(RAW_DIR / name),
            "--checksum",
        ]
        completed = subprocess.run(command, text=True, capture_output=True, check=False)
        checks.append(
            {
                "name": name,
                "command": command,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
                "pass": completed.returncode == 0,
            }
        )
    return {
        "experiment_id": EXPERIMENT_ID,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "read_only": True,
        "checks": checks,
        "all_pass": all(item["pass"] for item in checks),
    }


def role(path: Path) -> str:
    relative = path.relative_to(ROOT).as_posix()
    if relative.startswith(f"model/{EXPERIMENT_ID}/remote_raw/results/"):
        return "colab_result"
    if relative.startswith(f"model/{EXPERIMENT_ID}/remote_raw/logs/"):
        return "colab_log"
    if relative.startswith(f"model/{EXPERIMENT_ID}/remote_raw/checkpoints/"):
        return "colab_checkpoint"
    if relative.startswith("data/") or "strict_113A" in relative:
        return "frozen_input"
    if relative.endswith(".zip"):
        return "frozen_package"
    if relative.startswith("scripts/"):
        return "audit_or_runner_code"
    return "contract_or_provenance"


def main() -> int:
    required_direct = [
        ROOT / "data" / "train.csv",
        ROOT / "model" / "REF4-113A-V66-NESTED-117A" / "oof_predictions.csv",
        ROOT / "model" / "REF4-113A-V66-NESTED-117A" / "audit_contract.json",
        ROOT / "model" / "REF4-113A-V66-NESTED-117A" / "preflight_report.json",
        ROOT / "model" / "REF4-113A-V66-NESTED-117A" / "result.json",
        ROOT / "model" / "REF4-113A-V66-NESTED-117A" / "validation_report.json",
        ROOT / "scripts" / "run_ref4_jm_r_residual_strict_gpu_126.py",
        ROOT / "scripts" / "build_ref4_126_audit_manifest.py",
        ROOT / "scripts" / "verify_ref4_126_audit.py",
        ROOT / "colab" / "126" / "REF4_126_CODE.zip",
        ROOT / "colab" / "126" / "SHA256SUMS",
        ROOT / "colab" / "REF4_126_T4.ipynb",
        ROOT / "colab.md",
        ROOT / "output" / "submit_ref4_super_ensemble_113A.zip",
    ]
    remote_files = sorted(path for path in RAW_DIR.rglob("*") if path.is_file())
    if not remote_files:
        raise FileNotFoundError(f"No retrieved Colab artifacts under {RAW_DIR}")
    missing = [str(path) for path in required_direct if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Required audit files are missing: {missing}")

    remote_verification = run_remote_checks()
    atomic_json(REMOTE_CHECK_PATH, remote_verification)
    if not remote_verification["all_pass"]:
        raise RuntimeError("Remote/local rclone verification failed")

    paths = sorted(set(required_direct + remote_files + [REMOTE_CHECK_PATH]))
    entries = [
        {
            "path": path.relative_to(ROOT).as_posix(),
            "role": role(path),
            "bytes": path.stat().st_size,
            "sha256": sha256(path),
        }
        for path in paths
    ]
    manifest = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "scope": "strict_forward_validation_only",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "entry_count": len(entries),
        "entries": entries,
        "remote_transfer_verification": {
            "path": REMOTE_CHECK_PATH.relative_to(ROOT).as_posix(),
            "sha256": sha256(REMOTE_CHECK_PATH),
            "checks": len(remote_verification["checks"]),
            "all_pass": remote_verification["all_pass"],
        },
        "candidate_count": 1,
        "candidate_ids": [EXPERIMENT_ID],
        "production_fit_expected": False,
        "submission_zip_expected": False,
    }
    atomic_json(MANIFEST_PATH, manifest)
    print(
        json.dumps(
            {
                "status": "MANIFEST_CREATED",
                "path": str(MANIFEST_PATH),
                "entries": len(entries),
                "sha256": sha256(MANIFEST_PATH),
                "remote_checks": len(remote_verification["checks"]),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
