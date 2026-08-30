#!/usr/bin/env python3
"""Create and verify a deterministic pre-activation handover/rollback contract.

This script never writes to active submission paths. It stages the current active
state and the validated selective candidate in a temporary directory, performs a
dry-run activation, then restores the pre-activation snapshot and verifies exact
byte identity.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output" / "candidates"
CANDIDATE_ZIP = OUTPUT_DIR / "submit_selective_candidate.zip"
CANDIDATE_BUILD_REPORT = OUTPUT_DIR / "selective_candidate_build.json"
ROLLBACK_ZIP = OUTPUT_DIR / "rollback_active_before_selective.zip"
HANDOVER_MANIFEST = OUTPUT_DIR / "selective_candidate_handover.json"
RULES_PATH = ROOT / "01_제약과금지사항.md"

EXPECTED_CANDIDATE_FILES = {
    "script.py",
    "requirements.txt",
    "model/lightgbm_model.txt",
    "model/catboost_model.cbm",
    "model/feature_columns.json",
    "model/ensemble_contract.json",
}

# Candidate archive path -> repository active path after an explicitly approved sync.
ACTIVATION_MAP = {
    "script.py": "script.py",
    "requirements.txt": "requirements_submit.txt",
    "model/lightgbm_model.txt": "model/lightgbm_model.txt",
    "model/catboost_model.cbm": "model/catboost_model.cbm",
    "model/feature_columns.json": "model/feature_columns.json",
    "model/ensemble_contract.json": "model/ensemble_contract.json",
}

ACTIVE_SNAPSHOT_FILES = {
    "script.py",
    "requirements_submit.txt",
    "model/model.txt",
    "model/feature_columns.json",
    "output/submission.csv",
}

# These paths do not exist in the pre-activation state and must be removed on rollback.
ROLLBACK_REMOVE_PATHS = {
    "model/lightgbm_model.txt",
    "model/catboost_model.cbm",
    "model/ensemble_contract.json",
}

EXPECTED_CANDIDATE_ZIP_SHA256 = (
    "22dc61a85a5f6ea26e81645b6e21eed3c59584435c0a172b98779159f2b997ff"
)
EXPECTED_BUILD_REPORT_SHA256 = (
    "ad4b97077c5a50d8c4c246a8709dd158fd189cf87433657741a719c9eca2b4e0"
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, int | str]:
    return {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}


def add_deterministic(
    archive: zipfile.ZipFile, archive_name: str, content: bytes
) -> None:
    info = zipfile.ZipInfo(archive_name, date_time=(2020, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.create_system = 3
    info.external_attr = 0o100644 << 16
    archive.writestr(info, content)


def require_inputs() -> None:
    required = [CANDIDATE_ZIP, CANDIDATE_BUILD_REPORT, RULES_PATH]
    required.extend(ROOT / path for path in sorted(ACTIVE_SNAPSHOT_FILES))
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"필수 handover 입력이 없습니다: {missing}")

    candidate_hash = sha256_file(CANDIDATE_ZIP)
    if candidate_hash != EXPECTED_CANDIDATE_ZIP_SHA256:
        raise RuntimeError(
            f"검증 후보 ZIP 해시가 다릅니다: {candidate_hash}"
        )
    report_hash = sha256_file(CANDIDATE_BUILD_REPORT)
    if report_hash != EXPECTED_BUILD_REPORT_SHA256:
        raise RuntimeError(
            f"후보 build 보고서 해시가 다릅니다: {report_hash}"
        )

    report = json.loads(CANDIDATE_BUILD_REPORT.read_text(encoding="utf-8"))
    if report.get("python311_sandbox_e2e_pass") is not True:
        raise RuntimeError("후보 build 보고서가 Python 3.11 E2E PASS가 아닙니다")
    if report.get("active_submission_sync") is not False:
        raise RuntimeError("후보 build 보고서의 active sync 상태가 false가 아닙니다")
    if report.get("archive_sha256") != candidate_hash:
        raise RuntimeError("후보 build 보고서와 ZIP 해시가 일치하지 않습니다")
    if set(report.get("archive_files", [])) != EXPECTED_CANDIDATE_FILES:
        raise RuntimeError("후보 build 보고서의 ZIP 파일 목록이 다릅니다")

    existing_candidate_only = [
        path for path in sorted(ROLLBACK_REMOVE_PATHS) if (ROOT / path).exists()
    ]
    if existing_candidate_only:
        raise RuntimeError(
            "활성 전용 경로에 후보 전환 파일이 이미 존재합니다: "
            f"{existing_candidate_only}"
        )


def build_rollback_archive() -> None:
    with zipfile.ZipFile(ROLLBACK_ZIP, "w") as archive:
        for relative_path in sorted(ACTIVE_SNAPSHOT_FILES):
            add_deterministic(
                archive, relative_path, (ROOT / relative_path).read_bytes()
            )
    with zipfile.ZipFile(ROLLBACK_ZIP, "r") as archive:
        actual = set(archive.namelist())
    if actual != ACTIVE_SNAPSHOT_FILES:
        raise RuntimeError(
            f"rollback ZIP 목록이 다릅니다: missing={ACTIVE_SNAPSHOT_FILES-actual}, "
            f"extra={actual-ACTIVE_SNAPSHOT_FILES}"
        )


def verify_candidate_archive() -> dict[str, dict[str, int | str]]:
    with zipfile.ZipFile(CANDIDATE_ZIP, "r") as archive:
        actual = set(archive.namelist())
        if actual != EXPECTED_CANDIDATE_FILES:
            raise RuntimeError(
                f"후보 ZIP 목록이 다릅니다: missing={EXPECTED_CANDIDATE_FILES-actual}, "
                f"extra={actual-EXPECTED_CANDIDATE_FILES}"
            )
        return {
            name: {
                "sha256": sha256_bytes(archive.read(name)),
                "size_bytes": archive.getinfo(name).file_size,
            }
            for name in sorted(actual)
        }


def stage_activation_and_rollback(
    active_before: dict[str, dict[str, int | str]],
) -> dict[str, bool]:
    with tempfile.TemporaryDirectory(prefix="selective_handover_") as temp_dir:
        staged_root = Path(temp_dir)

        # Recreate the current active state from the rollback artifact.
        with zipfile.ZipFile(ROLLBACK_ZIP, "r") as rollback_archive:
            rollback_archive.extractall(staged_root)

        # Apply candidate bytes only inside the temporary activation root.
        with zipfile.ZipFile(CANDIDATE_ZIP, "r") as candidate_archive:
            for source, destination in sorted(ACTIVATION_MAP.items()):
                target = staged_root / destination
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(candidate_archive.read(source))
                if sha256_file(target) != sha256_bytes(candidate_archive.read(source)):
                    raise RuntimeError(f"임시 activation 복사가 다릅니다: {destination}")

        activation_pass = all((staged_root / path).is_file() for path in ACTIVATION_MAP.values())
        if not activation_pass:
            raise RuntimeError("임시 activation 대상 파일이 모두 생성되지 않았습니다")

        # Roll back overwritten files, then remove files introduced by activation.
        with zipfile.ZipFile(ROLLBACK_ZIP, "r") as rollback_archive:
            rollback_archive.extractall(staged_root)
        for relative_path in sorted(ROLLBACK_REMOVE_PATHS):
            target = staged_root / relative_path
            if target.exists():
                target.unlink()

        restored_pass = True
        for relative_path, expected in active_before.items():
            restored = staged_root / relative_path
            if not restored.is_file() or file_record(restored) != expected:
                restored_pass = False
                break
        removal_pass = all(
            not (staged_root / path).exists() for path in ROLLBACK_REMOVE_PATHS
        )
        if not restored_pass or not removal_pass:
            raise RuntimeError("임시 rollback이 활성 사전 상태를 정확히 복원하지 못했습니다")

    return {
        "staged_activation_pass": activation_pass,
        "rollback_restore_exact_pass": restored_pass,
        "rollback_candidate_only_removal_pass": removal_pass,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    require_inputs()

    active_before = {
        path: file_record(ROOT / path) for path in sorted(ACTIVE_SNAPSHOT_FILES)
    }
    candidate_files = verify_candidate_archive()
    build_rollback_archive()
    dry_run = stage_activation_and_rollback(active_before)
    active_after = {
        path: file_record(ROOT / path) for path in sorted(ACTIVE_SNAPSHOT_FILES)
    }
    active_unchanged = active_after == active_before
    if not active_unchanged:
        raise RuntimeError("handover 감사 중 실제 활성 파일이 변경되었습니다")

    payload = {
        "schema_version": 1,
        "experiment_id": "ENS-CATF-LGBMCATR5050-FINAL-SUBMIT-CANDIDATE",
        "audit_date": "2026-08-16",
        "authorization_required_for_activation": True,
        "active_submission_sync": False,
        "governance": {
            "rules_path": "01_제약과금지사항.md",
            "rules_sha256": sha256_file(RULES_PATH),
            "audit_script_path": "scripts/audit_selective_handover.py",
            "audit_script_sha256": sha256_file(Path(__file__)),
            "writes_to_active_paths": False,
        },
        "candidate": {
            "archive_path": "output/candidates/submit_selective_candidate.zip",
            "archive_sha256": sha256_file(CANDIDATE_ZIP),
            "archive_size_bytes": CANDIDATE_ZIP.stat().st_size,
            "build_report_path": "output/candidates/selective_candidate_build.json",
            "build_report_sha256": sha256_file(CANDIDATE_BUILD_REPORT),
            "archive_files": candidate_files,
        },
        "activation_map": [
            {"candidate_archive_path": source, "active_path": destination}
            for source, destination in sorted(ACTIVATION_MAP.items())
        ],
        "pre_activation_active_snapshot": active_before,
        "pre_activation_absent_paths": sorted(ROLLBACK_REMOVE_PATHS),
        "rollback": {
            "archive_path": "output/candidates/rollback_active_before_selective.zip",
            "archive_sha256": sha256_file(ROLLBACK_ZIP),
            "archive_size_bytes": ROLLBACK_ZIP.stat().st_size,
            "restore_paths": sorted(ACTIVE_SNAPSHOT_FILES),
            "remove_paths": sorted(ROLLBACK_REMOVE_PATHS),
        },
        "dry_run": dry_run,
        "active_state_unchanged_after_audit": active_unchanged,
        "audit_pass": all(dry_run.values()) and active_unchanged,
        "next_action": "explicit activation approval and final active-package gates",
    }
    HANDOVER_MANIFEST.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"Saved rollback archive: {ROLLBACK_ZIP}")
    print(f"Saved handover manifest: {HANDOVER_MANIFEST}")
    print("active submission unchanged")


if __name__ == "__main__":
    main()
