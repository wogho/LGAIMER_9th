#!/usr/bin/env python3
"""Seal verified 038A artifacts into a deterministic audit manifest."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model" / "REF4-CHAMPION-REPRO-038A"
CHAMPION_ZIP = ROOT / "output" / "submit_ref4_champion_030.zip"
EXPECTED_CHAMPION_SHA256 = "ca6af4bdf26d4bc79d503cdc6a09b0057a7ea9bdd77e29012f9e459a50cd66b8"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def record(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "size": path.stat().st_size}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    validation = json.loads((OUT / "validation_report.json").read_text())
    build = json.loads((OUT / "build_manifest.json").read_text())
    if validation["status"] != "AUDIT_VERIFIED" or validation["mismatch_count"] != 0:
        raise RuntimeError("Cannot seal a failed validation")
    if build["model_count"] != 56 or build["parameter_mismatch_count"] != 0:
        raise RuntimeError("Cannot seal an incomplete build")
    champion_hash = sha256_file(CHAMPION_ZIP)
    if champion_hash != EXPECTED_CHAMPION_SHA256:
        raise RuntimeError("Official champion ZIP changed")

    byte_match_count = 0
    original_by_name = {
        path.name: sha256_file(path)
        for path in (ROOT / "model" / "REF4-CHAMPION-STACK-030").glob("*.cbm")
    }
    for row in build["models"]:
        byte_match_count += int(original_by_name.get(row["file"]) == row["sha256"])

    result = {
        "schema_version": 1,
        "experiment_id": "REF4-CHAMPION-REPRO-038A",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_status": "COMPLETE",
        "audit_status": "AUDIT_VERIFIED",
        "reproduction_status": "PREDICTION_EQUIVALENT_ON_FIXED_SOURCE_PROBES",
        "model_count": 56,
        "model_parameter_mismatch_count": 0,
        "prediction_models_within_tolerance": validation["prediction_models_within_tolerance"],
        "prediction_tolerance": validation["tolerance"],
        "max_abs_prediction_difference": validation["max_abs_prediction_difference"],
        "model_byte_match_count": byte_match_count,
        "derived_artifact_equal_count": sum(int(row["content_equal"]) for row in build["derived_artifact_comparison"].values()),
        "derived_artifact_count": len(build["derived_artifact_comparison"]),
        "postprocess_recovery": build["recovery_event"],
        "official_score_preserved": 1068.25021,
        "official_champion_zip_sha256": champion_hash,
        "candidate_count": 0,
        "actual_leaf_count": 0,
        "gate_count": 0,
        "test_read_performed": False,
        "test_inference_performed": False,
        "candidate_created": False,
        "zip_created": False,
        "submission_performed": False,
    }
    write_json(OUT / "result.json", result)
    lines = [
        "# REF4-CHAMPION-REPRO-038A",
        "",
        "- execution status: `COMPLETE`",
        "- audit status: `AUDIT_VERIFIED`",
        "- models: `56/56`",
        "- parameter mismatch: `0`",
        f"- prediction parity: `{validation['prediction_models_within_tolerance']}/56` within `{validation['tolerance']}`",
        f"- maximum absolute prediction difference: `{validation['max_abs_prediction_difference']}`",
        f"- byte-identical CBM: `{byte_match_count}/56` (not a success requirement)",
        "- candidate/leaf/gate: `0/0/0`",
        "- test-read/test-inference/new-ZIP/submission: `false/false/false/false`",
        "- official score preserved: `1068.25021`",
        "",
        "The training completed before a non-mutating postprocess reader error. The finalizer read",
        "thread counts from embedded CBM metadata, performed no retraining or overwrite, and retained",
        "the failed status snapshot in `build_manifest.json`.",
    ]
    (OUT / "result.md").write_text("\n".join(lines) + "\n")

    artifact_paths = [
        ROOT / "scripts" / "run_ref4_champion_repro_038a.py",
        ROOT / "scripts" / "finalize_ref4_champion_repro_038a.py",
        ROOT / "scripts" / "verify_ref4_champion_repro_038a.py",
        Path(__file__).resolve(),
        OUT / "prebuild_manifest.json", OUT / "build_manifest.json",
        OUT / "prediction_parity.csv", OUT / "validation_report.json",
        OUT / "audit_attestation.json", OUT / "result.json", OUT / "result.md",
        CHAMPION_ZIP,
    ]
    artifact_paths.extend(sorted(OUT.glob("*.cbm")))
    records = {str(path.relative_to(ROOT)): record(path) for path in artifact_paths}
    audit = {
        "schema_version": 1,
        "experiment_id": "REF4-CHAMPION-REPRO-038A",
        "status": "AUDIT_VERIFIED",
        "artifact_count": len(records),
        "artifacts": records,
        "model_count": 56,
        "parameter_mismatch_count": 0,
        "validation_check_count": validation["check_count"],
        "validation_mismatch_count": validation["mismatch_count"],
        "candidate_count": 0,
        "leaf_count": 0,
        "gate_count": 0,
        "test_read_performed": False,
        "test_inference_performed": False,
        "zip_created": False,
    }
    write_json(OUT / "audit_manifest.json", audit)
    write_json(OUT / "execution_status.json", {
        "experiment_id": "REF4-CHAMPION-REPRO-038A",
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "sealed", "state": "AUDIT_VERIFIED",
        "model_count": 56, "validation_mismatch_count": 0,
        "retraining_after_postprocess_error": False,
    })
    print(json.dumps({
        "status": "AUDIT_VERIFIED", "artifact_count": len(records),
        "model_byte_match_count": byte_match_count,
        "champion_zip_sha256": champion_hash,
    }, indent=2))


if __name__ == "__main__":
    main()
