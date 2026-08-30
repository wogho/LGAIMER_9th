#!/usr/bin/env python3
"""Preflight for the single fixed F-Regime 0.75 OOF ablation."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "REF4-F-REGIME-032A"
OUT = ROOT / "model" / EXPERIMENT_ID
BASE = ROOT / "model" / "REF4-EXACT-OOF-031A"
SOURCE = ROOT / "model" / "REF4-CHAMPION-STACK-030"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    started = time.time(); OUT.mkdir(parents=True, exist_ok=True)
    checks: list[dict[str, object]] = []; mismatches: list[str] = []

    def check(name: str, passed: bool, actual: object) -> None:
        checks.append({"name": name, "checked": True, "passed": bool(passed), "actual": actual})
        if not passed: mismatches.append(name)

    required = [
        ROOT / "start04_uptostage.md", BASE / "audit_manifest.json", BASE / "validation_report.json",
        BASE / "audit_attestation.json", BASE / "oof_predictions.csv", SOURCE / "manifest.json",
        SOURCE / "f_regime_meta.json", ROOT / "data" / "train.csv",
    ]
    for path in required: check(f"exists:{path.relative_to(ROOT)}", path.is_file(), path.is_file())
    roadmap = (ROOT / "start04_uptostage.md").read_text(encoding="utf-8")
    check("contract_declared", EXPERIMENT_ID in roadmap, EXPERIMENT_ID in roadmap)
    check("official_score_declared", "1068.25021" in roadmap, "1068.25021" in roadmap)

    attestation = json.loads((BASE / "audit_attestation.json").read_text(encoding="utf-8"))
    check("base_audit_verified", attestation["status"] == "AUDIT_VERIFIED", attestation["status"])
    check("base_mismatch_zero", attestation["mismatch_count"] == 0, attestation["mismatch_count"])
    attested = {
        BASE / "audit_manifest.json": attestation["manifest_sha256"],
        BASE / "validation_report.json": attestation["validation_report_sha256"],
        ROOT / "scripts" / "verify_ref4_exact_oof_031a.py": attestation["validator_sha256"],
    }
    for path, expected in attested.items():
        actual = sha256_path(path); check(f"attested_hash:{path.name}", actual == expected, {"actual": actual, "expected": expected})
    audit = json.loads((BASE / "audit_manifest.json").read_text(encoding="utf-8"))
    oof_key = str((BASE / "oof_predictions.csv").relative_to(ROOT)); oof_hash = sha256_path(BASE / "oof_predictions.csv")
    check("base_oof_hash", audit["artifacts"][oof_key]["sha256"] == oof_hash, {"actual": oof_hash, "recorded": audit["artifacts"][oof_key]["sha256"]})

    regime = {key: float(value) for key, value in json.loads((SOURCE / "f_regime_meta.json").read_text(encoding="utf-8")).items()}
    expected_current = {"v2_scale": 2.0, "v355_scale": 0.5, "v330_scale": 0.5, "subtype_scale": 0.75, "v330_all_weight": 0.25, "v330_recent_inner_scale": 0.25, "transition_scale": 0.0}
    check("current_regime_exact", regime == expected_current, regime)
    candidate = dict(regime)
    for key in ("v2_scale", "v355_scale", "v330_scale", "subtype_scale"): candidate[key] *= 0.75
    expected_candidate = {"v2_scale": 1.5, "v355_scale": 0.375, "v330_scale": 0.375, "subtype_scale": 0.5625, "v330_all_weight": 0.25, "v330_recent_inner_scale": 0.25, "transition_scale": 0.0}
    check("candidate_regime_exact", candidate == expected_candidate, candidate)
    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    check("manifest_shape", len(manifest["main_weights"]) == 3 and len(manifest["stack_coefficients"]) == 4, {"weights": len(manifest["main_weights"]), "coefficients": len(manifest["stack_coefficients"])})
    check("global_shift", float(manifest["global_shift"]) == 0.0052, manifest["global_shift"])

    status = "AUDIT_VERIFIED" if not mismatches else "FAIL"
    report = {"experiment_id": EXPERIMENT_ID, "status": status, "checked_count": len(checks), "passed_count": sum(bool(x["passed"]) for x in checks), "mismatch_count": len(mismatches), "mismatches": mismatches, "current_regime": regime, "candidate_regime": candidate, "checks": checks, "elapsed_seconds": time.time() - started}
    (OUT / "preflight_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [f"# {EXPERIMENT_ID} preflight", "", f"- status: `{status}`", f"- checked: `{len(checks)}`", f"- mismatch: `{len(mismatches)}`", "", "## Mismatches", "", *(f"- `{x}`" for x in mismatches)]
    if not mismatches: lines.append("- none")
    (OUT / "preflight_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "checked_count", "passed_count", "mismatch_count", "elapsed_seconds")}, indent=2))
    if mismatches: raise SystemExit(1)


if __name__ == "__main__": main()
