#!/usr/bin/env python3
"""Evidence-first preflight for nested forward Adaptive Gate experiment 031B."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "REF4-ADAPTIVE-GATE-031B"
OUT = ROOT / "model" / EXPERIMENT_ID
BASE = ROOT / "model" / "REF4-EXACT-OOF-031A"
TRAIN = ROOT / "data" / "train.csv"
ROADMAP = ROOT / "start04_uptostage.md"
BASE_VALIDATOR = ROOT / "scripts" / "verify_ref4_exact_oof_031a.py"


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

    required = [
        TRAIN, ROADMAP, BASE / "audit_manifest.json", BASE / "validation_report.json",
        BASE / "audit_attestation.json", BASE / "oof_2023.csv", BASE / "oof_2024.csv",
        BASE / "oof_predictions.csv", BASE_VALIDATOR,
        ROOT / "scripts" / "run_ref4_exact_oof_031a.py",
        ROOT / "model" / "REF4-CHAMPION-STACK-030" / "manifest.json",
        ROOT / "model" / "REF4-CHAMPION-STACK-030" / "f_regime_meta.json",
        ROOT / "output" / "submit_ref4_champion_030.zip",
    ]
    for path in required:
        check(f"exists:{path.relative_to(ROOT)}", path.is_file(), path.is_file())

    roadmap = ROADMAP.read_text(encoding="utf-8")
    check("experiment_contract_declared", EXPERIMENT_ID in roadmap, EXPERIMENT_ID in roadmap)
    check("official_score_declared", "1068.25021" in roadmap, "1068.25021" in roadmap)

    attestation = json.loads((BASE / "audit_attestation.json").read_text(encoding="utf-8"))
    check("base_audit_verified", attestation.get("status") == "AUDIT_VERIFIED", attestation.get("status"))
    check("base_mismatch_zero", attestation.get("mismatch_count") == 0, attestation.get("mismatch_count"))
    check("base_model_count", attestation.get("model_count") == 110, attestation.get("model_count"))
    check("base_oof_rows", attestation.get("oof_rows") == 499032, attestation.get("oof_rows"))
    attested_hashes = {
        "manifest": (BASE / "audit_manifest.json", attestation["manifest_sha256"]),
        "report": (BASE / "validation_report.json", attestation["validation_report_sha256"]),
        "validator": (BASE_VALIDATOR, attestation["validator_sha256"]),
    }
    for label, (path, expected) in attested_hashes.items():
        actual = sha256_path(path)
        check(f"base_attested_hash:{label}", actual == expected, {"actual": actual, "expected": expected})

    base_manifest = json.loads((BASE / "audit_manifest.json").read_text(encoding="utf-8"))
    for name in ("oof_2023.csv", "oof_2024.csv", "oof_predictions.csv"):
        path = BASE / name
        key = str(path.relative_to(ROOT))
        recorded = base_manifest["artifacts"].get(key, {}).get("sha256")
        actual = sha256_path(path)
        check(f"base_oof_hash:{name}", recorded == actual, {"actual": actual, "recorded": recorded})

    base_report = json.loads((BASE / "validation_report.json").read_text(encoding="utf-8"))
    check("base_registration_gate", base_report.get("registration_gate") is True, base_report.get("registration_gate"))
    no_test = next((item for item in base_report["checks"] if item["name"] == "no_test_or_zip"), None)
    check("base_no_test_or_zip", bool(no_test and no_test["passed"]), no_test)

    raw = pd.read_csv(TRAIN, usecols=["row_id", "season", "control_success", "game_type"])
    counts = {str(int(year)): int(count) for year, count in raw.groupby("season").size().items()}
    check("raw_row_id_unique", raw["row_id"].is_unique, int(raw["row_id"].nunique()))
    check("raw_2022_nonempty", counts.get("2022", 0) > 0, counts)
    check("raw_2022_has_history", set(raw.loc[raw["season"].lt(2022), "season"].unique()) == {2019, 2020, 2021}, counts)
    check("raw_2021_f_nonempty", int((raw["season"].eq(2021) & raw["game_type"].eq("F")).sum()) > 0, int((raw["season"].eq(2021) & raw["game_type"].eq("F")).sum()))
    check("target_binary", set(raw["control_success"].unique()) == {0, 1}, sorted(raw["control_success"].unique().tolist()))

    status = "AUDIT_VERIFIED" if not mismatches else "FAIL"
    report = {
        "experiment_id": EXPERIMENT_ID, "scope": "nested_gate_preflight", "status": status,
        "checked_count": len(checks), "passed_count": sum(bool(item["passed"]) for item in checks),
        "mismatch_count": len(mismatches), "mismatches": mismatches, "checks": checks,
        "elapsed_seconds": time.time() - started,
    }
    (OUT / "preflight_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"# {EXPERIMENT_ID} preflight", "", f"- status: `{status}`",
        f"- checked: `{len(checks)}`", f"- passed: `{report['passed_count']}`",
        f"- mismatch: `{len(mismatches)}`", "", "## Mismatches", "",
        *(f"- `{name}`" for name in mismatches),
    ]
    if not mismatches:
        lines.append("- none")
    (OUT / "preflight_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "checked_count", "passed_count", "mismatch_count", "elapsed_seconds")}, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
