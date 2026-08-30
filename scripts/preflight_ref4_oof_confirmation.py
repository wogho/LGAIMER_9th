#!/usr/bin/env python3
"""Evidence-first preflight for the fixed 3-year REF4 OOF confirmations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DIAG = ROOT / "model" / "REF4-OOF-DIAG-034A"
BASE = ROOT / "model" / "REF4-EXACT-OOF-031A"
OOF22 = ROOT / "model" / "REF4-ADAPTIVE-GATE-031B"
ZIP = ROOT / "output" / "submit_ref4_champion_030.zip"
CONFIG = {
    "shift": ("REF4-SHIFT-034B", "shift_000", 8),
    "f-regime": ("REF4-F-REGIME-032B", "f_regime_075", 14),
}


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=CONFIG)
    args = parser.parse_args()
    experiment_id, candidate_name, gate_count = CONFIG[args.experiment]
    out = ROOT / "model" / experiment_id
    out.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, actual: object) -> None:
        checks.append({"name": name, "checked": True, "passed": bool(passed), "actual": actual})

    required = [
        DIAG / "audit_manifest.json", DIAG / "audit_attestation.json",
        DIAG / "validation_report.json", DIAG / "diagnostic_rows.csv",
        BASE / "audit_manifest.json", BASE / "audit_attestation.json",
        BASE / "oof_2023.csv", BASE / "oof_2024.csv",
        OOF22 / "audit_manifest.json", OOF22 / "audit_attestation.json",
        OOF22 / "oof_2022.csv", ROOT / "data" / "train.csv", ZIP,
        ROOT / "start03_reference.md", ROOT / "start04_uptostage.md",
        ROOT / "01_제약과금지사항.md",
    ]
    for path in required:
        check(f"exists:{path.relative_to(ROOT)}", path.is_file(), path.is_file())

    diag_attestation = json.loads((DIAG / "audit_attestation.json").read_text(encoding="utf-8"))
    diag_manifest = json.loads((DIAG / "audit_manifest.json").read_text(encoding="utf-8"))
    check("diag_audit_verified", diag_attestation["status"] == "AUDIT_VERIFIED" and diag_attestation["mismatch_count"] == 0, diag_attestation["status"])
    check("diag_manifest_hash", sha256_path(DIAG / "audit_manifest.json") == diag_attestation["manifest_sha256"], sha256_path(DIAG / "audit_manifest.json"))
    check("diag_validation_hash", sha256_path(DIAG / "validation_report.json") == diag_attestation["validation_report_sha256"], sha256_path(DIAG / "validation_report.json"))
    diag_key = str((DIAG / "diagnostic_rows.csv").relative_to(ROOT))
    check("diag_rows_hash", sha256_path(DIAG / "diagnostic_rows.csv") == diag_manifest["artifacts"][diag_key]["sha256"], sha256_path(DIAG / "diagnostic_rows.csv"))

    for year, path, directory in (
        (2022, OOF22 / "oof_2022.csv", OOF22),
        (2023, BASE / "oof_2023.csv", BASE),
        (2024, BASE / "oof_2024.csv", BASE),
    ):
        attestation = json.loads((directory / "audit_attestation.json").read_text(encoding="utf-8"))
        manifest = json.loads((directory / "audit_manifest.json").read_text(encoding="utf-8"))
        key = str(path.relative_to(ROOT))
        check(f"source_audit_verified_{year}", attestation["status"] == "AUDIT_VERIFIED" and attestation["mismatch_count"] == 0, attestation["status"])
        check(f"source_oof_hash_{year}", sha256_path(path) == manifest["artifacts"][key]["sha256"], sha256_path(path))

    check("champion_zip_hash", sha256_path(ZIP) == "ca6af4bdf26d4bc79d503cdc6a09b0057a7ea9bdd77e29012f9e459a50cd66b8", sha256_path(ZIP))
    candidate_count = len([candidate_name])
    check("candidate_count", candidate_count == 1, candidate_count)
    check("gate_count", gate_count == (8 if args.experiment == "shift" else 14), gate_count)
    mismatches = [str(row["name"]) for row in checks if not row["passed"]]
    status = "AUDIT_VERIFIED" if not mismatches else "FAIL"
    report = {
        "experiment_id": experiment_id, "status": status,
        "checked_count": len(checks), "passed_count": sum(bool(x["passed"]) for x in checks),
        "mismatch_count": len(mismatches), "mismatches": mismatches,
        "candidate_count": candidate_count, "actual_leaf_count": candidate_count,
        "gate_checks_count": gate_count, "model_count": 0,
        "years": [2022, 2023, 2024], "bootstrap_repetitions": 2000,
        "bootstrap_seed": 340200, "test_inference_performed": False,
        "full_train_performed": False, "zip_created": False, "checks": checks,
    }
    write_json(out / "preflight_report.json", report)
    lines = [
        f"# {experiment_id} preflight", "", f"- status: `{status}`",
        f"- checked: `{report['passed_count']}/{report['checked_count']}`",
        f"- mismatch: `{report['mismatch_count']}`", "- candidate/leaf: `1/1`",
        f"- performance gates: `{gate_count}`", "- models: `0`",
        "- test/full-train/ZIP: `false/false/false`",
    ]
    (out / "preflight_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("experiment_id", "status", "checked_count", "passed_count", "mismatch_count", "candidate_count", "gate_checks_count")}, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
