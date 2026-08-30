#!/usr/bin/env python3
"""Preflight for nested actual Transition Gate OOF experiment 033A."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "REF4-TRANSITION-GATE-033A"
OUT = ROOT / "model" / EXPERIMENT_ID
BASE = ROOT / "model" / "REF4-EXACT-OOF-031A"
GATE_BASE = ROOT / "model" / "REF4-ADAPTIVE-GATE-031B"
SOURCE = ROOT / "model" / "REF4-CHAMPION-STACK-030"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""): digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    started = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, object]] = []; mismatches: list[str] = []
    def check(name: str, passed: bool, actual: object) -> None:
        checks.append({"name": name, "checked": True, "passed": bool(passed), "actual": actual})
        if not passed: mismatches.append(name)

    required = [ROOT / "start04_uptostage.md", BASE / "audit_manifest.json", BASE / "audit_attestation.json", BASE / "oof_2023.csv", BASE / "oof_2024.csv", GATE_BASE / "audit_manifest.json", GATE_BASE / "audit_attestation.json", GATE_BASE / "oof_2022.csv", SOURCE / "manifest.json", SOURCE / "f_regime_meta.json", ROOT / "data" / "train.csv"]
    for path in required: check(f"exists:{path.relative_to(ROOT)}", path.is_file(), path.is_file())
    roadmap = (ROOT / "start04_uptostage.md").read_text(encoding="utf-8")
    check("contract_declared", EXPERIMENT_ID in roadmap, EXPERIMENT_ID in roadmap)
    check("official_score_declared", "1068.25021" in roadmap, "1068.25021" in roadmap)

    for label, directory, oof_names in (("031A", BASE, ("oof_2023.csv", "oof_2024.csv")), ("031B", GATE_BASE, ("oof_2022.csv",))):
        att = json.loads((directory / "audit_attestation.json").read_text(encoding="utf-8")); audit = json.loads((directory / "audit_manifest.json").read_text(encoding="utf-8"))
        check(f"{label}_audit_verified", att["status"] == "AUDIT_VERIFIED" and att["mismatch_count"] == 0, {"status": att["status"], "mismatch": att["mismatch_count"]})
        check(f"{label}_manifest_attested", sha256_path(directory / "audit_manifest.json") == att["manifest_sha256"], {"actual": sha256_path(directory / "audit_manifest.json"), "expected": att["manifest_sha256"]})
        for name in oof_names:
            path = directory / name; key = str(path.relative_to(ROOT)); actual = sha256_path(path); recorded = audit["artifacts"][key]["sha256"]
            check(f"{label}_oof_hash:{name}", actual == recorded, {"actual": actual, "recorded": recorded})

    regime = json.loads((SOURCE / "f_regime_meta.json").read_text(encoding="utf-8"))
    check("transition_currently_disabled", float(regime["transition_scale"]) == 0.0, regime["transition_scale"])
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    check("base_manifest_shape", len(manifest["main_weights"]) == 3 and len(manifest["stack_coefficients"]) == 4, {"main": len(manifest["main_weights"]), "stack": len(manifest["stack_coefficients"])})
    needed = ["row_id", "season", "pitcher_id", "game_type", "pitcher_hand", "batter_hand", "pitcher_team_id", "balls_before", "strikes_before", "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev5_game_success_rate", "asof_pitcher_middle_rate", "asof_pitcher_reverse_rate", "li", "inning", "num_runners_on", "control_success"]
    raw = pd.read_csv(ROOT / "data" / "train.csv", usecols=needed, low_memory=False)
    check("raw_row_id_unique", raw["row_id"].is_unique, int(raw["row_id"].nunique()))
    check("required_years_nonempty", all(int(raw["season"].eq(year).sum()) > 0 for year in (2022, 2023, 2024)), {str(year): int(raw["season"].eq(year).sum()) for year in (2022, 2023, 2024)})
    check("transition_categories_nonempty", all(raw[column].notna().any() for column in ("game_type", "pitcher_id", "pitcher_team_id")), {column: int(raw[column].notna().sum()) for column in ("game_type", "pitcher_id", "pitcher_team_id")})

    status = "AUDIT_VERIFIED" if not mismatches else "FAIL"
    report = {"experiment_id": EXPERIMENT_ID, "status": status, "checked_count": len(checks), "passed_count": sum(bool(x["passed"]) for x in checks), "mismatch_count": len(mismatches), "mismatches": mismatches, "checks": checks, "elapsed_seconds": time.time() - started}
    (OUT / "preflight_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [f"# {EXPERIMENT_ID} preflight", "", f"- status: `{status}`", f"- checked: `{len(checks)}`", f"- mismatch: `{len(mismatches)}`", "", "## Mismatches", "", *(f"- `{x}`" for x in mismatches)]
    if not mismatches: lines.append("- none")
    (OUT / "preflight_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "checked_count", "passed_count", "mismatch_count", "elapsed_seconds")}, indent=2))
    if mismatches: raise SystemExit(1)


if __name__ == "__main__": main()
