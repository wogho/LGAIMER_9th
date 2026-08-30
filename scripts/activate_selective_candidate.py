#!/usr/bin/env python3
"""Atomically activate or roll back the validated selective candidate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "output" / "candidates" / "selective_candidate_handover.json"
ACTIVATION_RECORD = ROOT / "output" / "candidates" / "selective_activation.json"
EXPECTED_MANIFEST_SHA256 = (
    "cdbbca6d2cbe92b580d21714c3ca9ddb907d646538af319391030dd8608cf23a"
)
EXPECTED_ACTIVATION_MAP = {
    "script.py": "script.py",
    "requirements.txt": "requirements_submit.txt",
    "model/lightgbm_model.txt": "model/lightgbm_model.txt",
    "model/catboost_model.cbm": "model/catboost_model.cbm",
    "model/feature_columns.json": "model/feature_columns.json",
    "model/ensemble_contract.json": "model/ensemble_contract.json",
}
EXPECTED_RESTORE_PATHS = {
    "script.py",
    "requirements_submit.txt",
    "model/model.txt",
    "model/feature_columns.json",
    "output/submission.csv",
}
EXPECTED_REMOVE_PATHS = {
    "model/lightgbm_model.txt",
    "model/catboost_model.cbm",
    "model/ensemble_contract.json",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def file_record(path: Path) -> dict[str, int | str]:
    return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def atomic_write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", dir=path.parent, delete=False
        ) as temporary:
            temporary.write(content)
            temporary.flush()
            os.fsync(temporary.fileno())
            temporary_name = temporary.name
        os.chmod(temporary_name, 0o644)
        os.replace(temporary_name, path)
        temporary_name = None
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def load_and_verify_manifest() -> dict:
    if not MANIFEST_PATH.is_file():
        raise RuntimeError(f"handover manifest가 없습니다: {MANIFEST_PATH}")
    if sha256_file(MANIFEST_PATH) != EXPECTED_MANIFEST_SHA256:
        raise RuntimeError("handover manifest 해시가 승인된 값과 다릅니다")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if manifest.get("audit_pass") is not True:
        raise RuntimeError("handover 감사 결과가 PASS가 아닙니다")
    if manifest.get("active_submission_sync") is not False:
        raise RuntimeError("이미 활성화된 handover manifest입니다")
    activation_map = {
        item["candidate_archive_path"]: item["active_path"]
        for item in manifest.get("activation_map", [])
    }
    if activation_map != EXPECTED_ACTIVATION_MAP:
        raise RuntimeError("activation map이 승인된 6경로와 다릅니다")
    if set(manifest["rollback"]["restore_paths"]) != EXPECTED_RESTORE_PATHS:
        raise RuntimeError("rollback restore 경로가 승인된 5경로와 다릅니다")
    if set(manifest["rollback"]["remove_paths"]) != EXPECTED_REMOVE_PATHS:
        raise RuntimeError("rollback remove 경로가 승인된 3경로와 다릅니다")
    return manifest


def resolve_artifact(relative_path: str) -> Path:
    path = (ROOT / relative_path).resolve()
    if ROOT not in path.parents:
        raise RuntimeError(f"프로젝트 밖 artifact 경로입니다: {relative_path}")
    return path


def verify_pre_activation_state(manifest: dict) -> None:
    for relative_path, expected in manifest["pre_activation_active_snapshot"].items():
        path = resolve_artifact(relative_path)
        if not path.is_file() or file_record(path) != expected:
            raise RuntimeError(f"활성 사전 상태가 handover snapshot과 다릅니다: {relative_path}")
    for relative_path in manifest["pre_activation_absent_paths"]:
        if resolve_artifact(relative_path).exists():
            raise RuntimeError(f"후보 전용 활성 경로가 이미 존재합니다: {relative_path}")


def verify_archives(manifest: dict) -> tuple[Path, Path]:
    candidate = resolve_artifact(manifest["candidate"]["archive_path"])
    rollback = resolve_artifact(manifest["rollback"]["archive_path"])
    if not candidate.is_file() or sha256_file(candidate) != manifest["candidate"]["archive_sha256"]:
        raise RuntimeError("후보 ZIP이 handover manifest와 다릅니다")
    if not rollback.is_file() or sha256_file(rollback) != manifest["rollback"]["archive_sha256"]:
        raise RuntimeError("rollback ZIP이 handover manifest와 다릅니다")
    return candidate, rollback


def restore_snapshot(manifest: dict, rollback_zip: Path) -> None:
    with zipfile.ZipFile(rollback_zip, "r") as archive:
        if set(archive.namelist()) != EXPECTED_RESTORE_PATHS:
            raise RuntimeError("rollback ZIP 내부 파일 목록이 다릅니다")
        for relative_path in sorted(EXPECTED_RESTORE_PATHS):
            atomic_write(resolve_artifact(relative_path), archive.read(relative_path))
    for relative_path in sorted(EXPECTED_REMOVE_PATHS):
        path = resolve_artifact(relative_path)
        if path.exists():
            path.unlink()
    for relative_path, expected in manifest["pre_activation_active_snapshot"].items():
        if file_record(resolve_artifact(relative_path)) != expected:
            raise RuntimeError(f"rollback 복원 해시가 다릅니다: {relative_path}")


def activate(manifest: dict, candidate_zip: Path, rollback_zip: Path) -> None:
    verify_pre_activation_state(manifest)
    activation_map = {
        item["candidate_archive_path"]: item["active_path"]
        for item in manifest["activation_map"]
    }
    try:
        with zipfile.ZipFile(candidate_zip, "r") as archive:
            if set(archive.namelist()) != set(EXPECTED_ACTIVATION_MAP):
                raise RuntimeError("후보 ZIP 내부 파일 목록이 다릅니다")
            for source, destination in sorted(activation_map.items()):
                content = archive.read(source)
                expected = manifest["candidate"]["archive_files"][source]
                if sha256_bytes(content) != expected["sha256"]:
                    raise RuntimeError(f"후보 파일 해시가 manifest와 다릅니다: {source}")
                atomic_write(resolve_artifact(destination), content)

        active_files = {
            destination: file_record(resolve_artifact(destination))
            for destination in sorted(activation_map.values())
        }
        for source, destination in activation_map.items():
            if active_files[destination] != manifest["candidate"]["archive_files"][source]:
                raise RuntimeError(f"활성 파일 해시가 후보와 다릅니다: {destination}")
    except Exception:
        restore_snapshot(manifest, rollback_zip)
        raise

    record = {
        "schema_version": 1,
        "experiment_id": manifest["experiment_id"],
        "activation_date": "2026-08-16",
        "handover_manifest_sha256": sha256_file(MANIFEST_PATH),
        "candidate_archive_sha256": sha256_file(candidate_zip),
        "rollback_archive_sha256": sha256_file(rollback_zip),
        "active_files": active_files,
        "candidate_contract_active_model_sync": False,
        "candidate_contract_flag_semantics": "immutable candidate provenance",
        "active_submission_sync": True,
        "final_active_gates_pass": False,
    }
    atomic_write(
        ACTIVATION_RECORD,
        (json.dumps(record, ensure_ascii=False, indent=2) + "\n").encode("utf-8"),
    )
    print(json.dumps(record, ensure_ascii=False, indent=2))
    print("selective candidate activated; final active gates are still pending")


def rollback(manifest: dict, rollback_zip: Path) -> None:
    restore_snapshot(manifest, rollback_zip)
    ACTIVATION_RECORD.unlink(missing_ok=True)
    print("pre-activation active snapshot restored exactly")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--activate", action="store_true")
    action.add_argument("--rollback", action="store_true")
    args = parser.parse_args()

    manifest = load_and_verify_manifest()
    candidate_zip, rollback_zip = verify_archives(manifest)
    if args.activate:
        activate(manifest, candidate_zip, rollback_zip)
    else:
        rollback(manifest, rollback_zip)


if __name__ == "__main__":
    main()
