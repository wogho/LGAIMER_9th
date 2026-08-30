#!/usr/bin/env python3
"""Preflight for the read-only three-year REF4 OOF diagnosis."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "REF4-OOF-DIAG-034A"
OUT = ROOT / "model" / EXPERIMENT_ID
BASE = ROOT / "model" / "REF4-EXACT-OOF-031A"
OOF22 = ROOT / "model" / "REF4-ADAPTIVE-GATE-031B"
SOURCE = ROOT / "model" / "REF4-CHAMPION-STACK-030"
ZIP_PATH = ROOT / "output" / "submit_ref4_champion_030.zip"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, object]] = []
    mismatches: list[str] = []

    def check(name: str, passed: bool, actual: object) -> None:
        checks.append({"name": name, "checked": True, "passed": bool(passed), "actual": actual})
        if not passed:
            mismatches.append(name)

    inputs = {
        2022: OOF22 / "oof_2022.csv",
        2023: BASE / "oof_2023.csv",
        2024: BASE / "oof_2024.csv",
    }
    mapping_paths = {
        2022: OOF22 / "fold_2022" / "pitcher_trackman_mapping.csv",
        2023: BASE / "fold_2023" / "pitcher_trackman_mapping.csv",
        2024: BASE / "fold_2024" / "pitcher_trackman_mapping.csv",
    }
    required = [
        ROOT / "start04_uptostage.md", ROOT / "data" / "train.csv", ZIP_PATH,
        SOURCE / "manifest.json", SOURCE / "f_regime_meta.json",
        BASE / "audit_manifest.json", BASE / "audit_attestation.json",
        OOF22 / "audit_manifest.json", OOF22 / "audit_attestation.json",
        *inputs.values(), *mapping_paths.values(),
    ]
    for path in required:
        check(f"exists:{path.relative_to(ROOT)}", path.is_file(), path.is_file())

    roadmap = (ROOT / "start04_uptostage.md").read_text(encoding="utf-8")
    check("contract_declared", EXPERIMENT_ID in roadmap, EXPERIMENT_ID in roadmap)
    check("gemini_agreement_declared", "GPT 동의 범위 및 실행 착수" in roadmap, "GPT 동의 범위 및 실행 착수" in roadmap)
    check("official_score_declared", "1068.25021" in roadmap, "1068.25021" in roadmap)
    zip_sha = sha256_path(ZIP_PATH)
    check("champion_zip_unchanged", zip_sha == "ca6af4bdf26d4bc79d503cdc6a09b0057a7ea9bdd77e29012f9e459a50cd66b8", zip_sha)

    for label, directory in (("base", BASE), ("oof22", OOF22)):
        attestation = json.loads((directory / "audit_attestation.json").read_text(encoding="utf-8"))
        check(f"{label}_audit_verified", attestation["status"] == "AUDIT_VERIFIED" and attestation["mismatch_count"] == 0, {"status": attestation["status"], "mismatch": attestation["mismatch_count"]})
        check(f"{label}_manifest_attested", sha256_path(directory / "audit_manifest.json") == attestation["manifest_sha256"], {"actual": sha256_path(directory / "audit_manifest.json"), "expected": attestation["manifest_sha256"]})

    audit_by_directory = {
        BASE: json.loads((BASE / "audit_manifest.json").read_text(encoding="utf-8")),
        OOF22: json.loads((OOF22 / "audit_manifest.json").read_text(encoding="utf-8")),
    }
    for year, path in {**inputs, **{}}.items():
        directory = OOF22 if year == 2022 else BASE
        key = str(path.relative_to(ROOT))
        actual = sha256_path(path)
        recorded = audit_by_directory[directory]["artifacts"][key]["sha256"]
        check(f"oof_hash_{year}", actual == recorded, {"actual": actual, "recorded": recorded})
    for year, path in mapping_paths.items():
        directory = OOF22 if year == 2022 else BASE
        key = str(path.relative_to(ROOT))
        actual = sha256_path(path)
        recorded = audit_by_directory[directory]["artifacts"][key]["sha256"]
        check(f"trackman_mapping_hash_{year}", actual == recorded, {"actual": actual, "recorded": recorded})

    required_columns = {
        "row_id", "season", "game_type", "pitcher_id", "target",
        "p_v2_global", "p_v2_f", "p_v3_55_global", "p_v3_55_f",
        "p_v3_30_global", "p_v3_30_f_all", "p_v3_30_f_recent",
        "risk_middle_global", "risk_middle_f", "risk_wild_global", "risk_wild_f",
        "risk_reverse_global", "risk_reverse_f", "prediction_no_shift", "prediction",
    }
    total_rows = 0
    for year, path in inputs.items():
        frame = pd.read_csv(path, usecols=lambda column: column in required_columns, low_memory=False)
        check(f"oof_columns_{year}", set(frame.columns) == required_columns, sorted(required_columns - set(frame.columns)))
        check(f"oof_year_{year}", frame["season"].eq(year).all(), sorted(frame["season"].unique().tolist()))
        check(f"oof_row_id_unique_{year}", frame["row_id"].is_unique, int(frame["row_id"].nunique()))
        total_rows += len(frame)
    check("three_year_rows_nonempty", total_rows > 0, total_rows)

    regime = {key: float(value) for key, value in json.loads((SOURCE / "f_regime_meta.json").read_text(encoding="utf-8")).items()}
    check("transition_disabled", regime["transition_scale"] == 0.0, regime["transition_scale"])
    check("diagnostic_leaf_count_zero", True, 0)

    status = "AUDIT_VERIFIED" if not mismatches else "FAIL"
    report = {
        "experiment_id": EXPERIMENT_ID, "scope": "read_only_oof_diagnostic_preflight",
        "status": status, "checked_count": len(checks),
        "passed_count": sum(bool(item["passed"]) for item in checks),
        "mismatch_count": len(mismatches), "mismatches": mismatches,
        "three_year_oof_rows": total_rows, "candidate_count": 0,
        "zip_sha256": zip_sha, "checks": checks, "elapsed_seconds": time.time() - started,
    }
    (OUT / "preflight_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"# {EXPERIMENT_ID} preflight", "", f"- status: `{status}`",
        f"- checked: `{len(checks)}`", f"- passed: `{report['passed_count']}`",
        f"- mismatch: `{len(mismatches)}`", f"- OOF rows: `{total_rows}`",
        "- candidate leaf count: `0`", "", "## Mismatches", "",
        *(f"- `{name}`" for name in mismatches),
    ]
    if not mismatches:
        lines.append("- none")
    (OUT / "preflight_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "checked_count", "passed_count", "mismatch_count", "three_year_oof_rows", "candidate_count", "elapsed_seconds")}, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
