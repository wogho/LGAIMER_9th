#!/usr/bin/env python3
"""Evidence-first preflight for the fixed era-aware F calibrator."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "REF4-F-ERA-CAL-036A"
OUT = ROOT / "model" / EXPERIMENT_ID
DIAG = ROOT / "model" / "REF4-F-DRIFT-DIAG-035A"
BASE = ROOT / "model" / "REF4-EXACT-OOF-031A"
OOF22 = ROOT / "model" / "REF4-ADAPTIVE-GATE-031B"
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
        DIAG / "audit_manifest.json", DIAG / "validation_report.json", DIAG / "audit_attestation.json", DIAG / "f_diagnostic_rows.csv",
        BASE / "audit_manifest.json", BASE / "audit_attestation.json", BASE / "oof_2023.csv", BASE / "oof_2024.csv",
        OOF22 / "audit_manifest.json", OOF22 / "audit_attestation.json", OOF22 / "oof_2022.csv",
        ROOT / "data" / "train.csv", ROOT / "start03_reference.md", ROOT / "start04_uptostage.md",
        ROOT / "01_제약과금지사항.md", ZIP,
    ]
    for path in required:
        check(f"exists:{path.relative_to(ROOT)}", path.is_file(), path.is_file())

    diag_attestation = json.loads((DIAG / "audit_attestation.json").read_text(encoding="utf-8"))
    diag_manifest = json.loads((DIAG / "audit_manifest.json").read_text(encoding="utf-8"))
    check("diag_verified", diag_attestation["status"] == "AUDIT_VERIFIED" and diag_attestation["mismatch_count"] == 0, {"status": diag_attestation["status"], "mismatch": diag_attestation["mismatch_count"]})
    check("diag_manifest_hash", sha256_path(DIAG / "audit_manifest.json") == diag_attestation["manifest_sha256"], sha256_path(DIAG / "audit_manifest.json"))
    check("diag_report_hash", sha256_path(DIAG / "validation_report.json") == diag_attestation["validation_report_sha256"], sha256_path(DIAG / "validation_report.json"))
    f_key = str((DIAG / "f_diagnostic_rows.csv").relative_to(ROOT))
    check("diag_f_rows_hash", sha256_path(DIAG / "f_diagnostic_rows.csv") == diag_manifest["artifacts"][f_key]["sha256"], sha256_path(DIAG / "f_diagnostic_rows.csv"))

    total_rows = 0
    for year, path, directory in ((2022, OOF22 / "oof_2022.csv", OOF22), (2023, BASE / "oof_2023.csv", BASE), (2024, BASE / "oof_2024.csv", BASE)):
        attestation = json.loads((directory / "audit_attestation.json").read_text(encoding="utf-8"))
        manifest = json.loads((directory / "audit_manifest.json").read_text(encoding="utf-8"))
        key = str(path.relative_to(ROOT))
        check(f"source_verified_{year}", attestation["status"] == "AUDIT_VERIFIED" and attestation["mismatch_count"] == 0, attestation["status"])
        check(f"source_hash_{year}", sha256_path(path) == manifest["artifacts"][key]["sha256"], sha256_path(path))
        with path.open("rb") as handle:
            rows = sum(1 for _ in handle) - 1
        total_rows += rows
        check(f"source_rows_{year}", rows > 0, rows)
    check("oof_total_rows", total_rows == diag_attestation["oof_rows"], {"actual": total_rows, "diag": diag_attestation["oof_rows"]})
    check("champion_zip_hash", sha256_path(ZIP) == "ca6af4bdf26d4bc79d503cdc6a09b0057a7ea9bdd77e29012f9e459a50cd66b8", sha256_path(ZIP))

    candidate_names = ["ref4_exact_current/f_era_affine_brier_2023"]
    composite_gates = ["all_six_subchecks"]
    subchecks = ["2024_all_gain", "2024_f_gain", "2024_all_ci", "2024_f_ci", "2024_f_bias", "2024_f_ece"]
    check("candidate_leaf_count", len(candidate_names) == 1, len(candidate_names))
    check("composite_gate_count", len(composite_gates) == len(candidate_names) == 1, len(composite_gates))
    check("subcheck_count", len(subchecks) == 6, len(subchecks))
    contract = {
        "experiment_id": EXPERIMENT_ID, "candidate_names": candidate_names,
        "fit_year": 2023, "validation_year": 2024, "historical_counterfactual_year": 2022,
        "fit_slice": {"game_type": "F"}, "application_slice": {"game_type": "F"},
        "method": "affine_least_squares_brier", "solver": "numpy.linalg.lstsq", "rcond": None,
        "clip": [1e-5, 1 - 1e-5], "ece_edges": [index / 10 for index in range(11)],
        "bootstrap_repetitions": 2000, "bootstrap_seed": 360200,
        "candidate_count": len(candidate_names), "actual_leaf_count": len(candidate_names),
        "gate_checks_count": len(composite_gates), "promotion_subcheck_count": len(subchecks),
        "model_count": 1, "test_inference_performed": False, "full_train_performed": False,
        "zip_created": False,
    }
    write_json(OUT / "gate_contract.json", contract)
    mismatches = [str(row["name"]) for row in checks if not row["passed"]]
    status = "AUDIT_VERIFIED" if not mismatches else "FAIL"
    report = {
        "experiment_id": EXPERIMENT_ID, "status": status,
        "checked_count": len(checks), "passed_count": sum(bool(row["passed"]) for row in checks),
        "mismatch_count": len(mismatches), "mismatches": mismatches,
        "candidate_count": len(candidate_names), "actual_leaf_count": len(candidate_names),
        "gate_checks_count": len(composite_gates), "promotion_subcheck_count": len(subchecks),
        "model_count": 1, "oof_rows": total_rows,
        "start04_sha256_at_contract": sha256_path(ROOT / "start04_uptostage.md"),
        "test_inference_performed": False, "full_train_performed": False, "zip_created": False,
        "checks": checks,
    }
    write_json(OUT / "preflight_report.json", report)
    lines = [f"# {EXPERIMENT_ID} preflight", "", f"- status: `{status}`", f"- checked: `{report['passed_count']}/{report['checked_count']}`", f"- mismatch: `{report['mismatch_count']}`", "- candidate/leaf/composite-gate/subcheck/model: `1/1/1/6/1`", "- test/full-train/ZIP: `false/false/false`"]
    (OUT / "preflight_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("experiment_id", "status", "checked_count", "passed_count", "mismatch_count", "candidate_count", "gate_checks_count", "promotion_subcheck_count", "model_count", "oof_rows")}, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
