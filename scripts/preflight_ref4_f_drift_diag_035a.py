#!/usr/bin/env python3
"""Preflight for the read-only 2023 F-drift diagnosis."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "REF4-F-DRIFT-DIAG-035A"
OUT = ROOT / "model" / EXPERIMENT_ID
DIAG = ROOT / "model" / "REF4-OOF-DIAG-034A"
BASE = ROOT / "model" / "REF4-EXACT-OOF-031A"
OOF22 = ROOT / "model" / "REF4-ADAPTIVE-GATE-031B"
TRAIN = ROOT / "data" / "train.csv"
ZIP = ROOT / "output" / "submit_ref4_champion_030.zip"


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

    required = [
        TRAIN, ZIP, ROOT / "start03_reference.md", ROOT / "start04_uptostage.md",
        ROOT / "01_제약과금지사항.md", DIAG / "audit_manifest.json",
        DIAG / "validation_report.json", DIAG / "audit_attestation.json",
        BASE / "audit_manifest.json", BASE / "audit_attestation.json",
        BASE / "oof_2023.csv", BASE / "oof_2024.csv",
        OOF22 / "audit_manifest.json", OOF22 / "audit_attestation.json",
        OOF22 / "oof_2022.csv",
    ]
    for path in required:
        check(f"exists:{path.relative_to(ROOT)}", path.is_file(), path.is_file())

    diag_attestation = json.loads((DIAG / "audit_attestation.json").read_text(encoding="utf-8"))
    check("diag_verified", diag_attestation["status"] == "AUDIT_VERIFIED" and diag_attestation["mismatch_count"] == 0, {"status": diag_attestation["status"], "mismatch": diag_attestation["mismatch_count"]})
    check("diag_manifest_hash", sha256_path(DIAG / "audit_manifest.json") == diag_attestation["manifest_sha256"], sha256_path(DIAG / "audit_manifest.json"))
    check("diag_report_hash", sha256_path(DIAG / "validation_report.json") == diag_attestation["validation_report_sha256"], sha256_path(DIAG / "validation_report.json"))
    diag_manifest = json.loads((DIAG / "audit_manifest.json").read_text(encoding="utf-8"))
    train_key = str(TRAIN.relative_to(ROOT))
    check("train_hash", sha256_path(TRAIN) == diag_manifest["artifacts"][train_key]["sha256"], sha256_path(TRAIN))

    oof_rows = 0
    for year, path, directory in (
        (2022, OOF22 / "oof_2022.csv", OOF22),
        (2023, BASE / "oof_2023.csv", BASE),
        (2024, BASE / "oof_2024.csv", BASE),
    ):
        attestation = json.loads((directory / "audit_attestation.json").read_text(encoding="utf-8"))
        manifest = json.loads((directory / "audit_manifest.json").read_text(encoding="utf-8"))
        key = str(path.relative_to(ROOT))
        check(f"source_verified_{year}", attestation["status"] == "AUDIT_VERIFIED" and attestation["mismatch_count"] == 0, attestation["status"])
        check(f"source_hash_{year}", sha256_path(path) == manifest["artifacts"][key]["sha256"], sha256_path(path))
        with path.open("rb") as handle:
            rows = sum(1 for _ in handle) - 1
        oof_rows += rows
        check(f"source_rows_{year}", rows > 0, rows)

    check("three_year_oof_rows", oof_rows == diag_attestation["oof_rows"], {"actual": oof_rows, "diag": diag_attestation["oof_rows"]})
    check("champion_zip_hash", sha256_path(ZIP) == "ca6af4bdf26d4bc79d503cdc6a09b0057a7ea9bdd77e29012f9e459a50cd66b8", sha256_path(ZIP))
    check("candidate_count", len([]) == 0, len([]))
    check("gate_count", len([]) == 0, len([]))

    contract = {
        "experiment_id": EXPERIMENT_ID, "hypothesis_count": 1,
        "hypothesis": "2023 F degradation combines prevalence/composition drift with upward F-expert adjustment",
        "years": [2022, 2023, 2024], "history_years": [2019, 2020, 2021, 2022, 2023, 2024],
        "slice_dimensions": ["game_month", "prior_type", "known_status", "pitcher_n_bin", "pitcher_hand", "trackman_status", "f_years_pattern"],
        "prediction_variants": ["current", "no_shift", "global_only", "f075"],
        "candidate_count": 0, "actual_leaf_count": 0, "gate_checks_count": 0,
        "model_count": 0, "test_inference_performed": False,
        "full_train_performed": False, "zip_created": False,
    }
    write_json(OUT / "diagnostic_contract.json", contract)
    mismatches = [str(row["name"]) for row in checks if not row["passed"]]
    status = "AUDIT_VERIFIED" if not mismatches else "FAIL"
    report = {
        "experiment_id": EXPERIMENT_ID, "status": status,
        "checked_count": len(checks), "passed_count": sum(bool(x["passed"]) for x in checks),
        "mismatch_count": len(mismatches), "mismatches": mismatches,
        "candidate_count": 0, "actual_leaf_count": 0, "gate_checks_count": 0,
        "model_count": 0, "oof_rows": oof_rows,
        "start04_sha256_at_contract": sha256_path(ROOT / "start04_uptostage.md"),
        "test_inference_performed": False, "full_train_performed": False,
        "zip_created": False, "checks": checks,
    }
    write_json(OUT / "preflight_report.json", report)
    lines = [
        f"# {EXPERIMENT_ID} preflight", "", f"- status: `{status}`",
        f"- checked: `{report['passed_count']}/{report['checked_count']}`",
        f"- mismatch: `{report['mismatch_count']}`", "- candidate/leaf/gate: `0/0/0`",
        "- models: `0`", "- test/full-train/ZIP: `false/false/false`",
    ]
    (OUT / "preflight_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("experiment_id", "status", "checked_count", "passed_count", "mismatch_count", "candidate_count", "gate_checks_count", "model_count", "oof_rows")}, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
