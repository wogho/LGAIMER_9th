#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model/REF4-TRAINONLY-SPLIT-RESIDUAL-046A"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    contract = json.loads((OUT / "audit_contract.json").read_text())
    checks = []

    def check(name, passed, actual, expected=None):
        checks.append({"name": name, "checked": True, "pass": bool(passed), "actual": actual, "expected": expected})

    required = [
        ROOT / contract["official_train"],
        ROOT / contract["preserve_zip"],
        ROOT / "01_제약과금지사항.md",
        ROOT / "start04_uptostage.md",
        ROOT / "src/entity_context_split.py",
    ] + [ROOT / p for p in contract["base_oof"].values()] + [ROOT / p for p in contract["base_attestations"]]
    for path in required:
        check(f"exists:{path.relative_to(ROOT)}", path.is_file(), path.stat().st_size if path.is_file() else None)

    attestation_paths = [ROOT / p for p in contract["base_attestations"]]
    attestations = [json.loads(path.read_text()) for path in attestation_paths]
    check(
        "upstream_audits",
        all((a.get("overall_status") or a.get("status")) == "AUDIT_VERIFIED" and a.get("mismatch_count") == 0 for a in attestations),
        [{"status": a.get("overall_status") or a.get("status"), "mismatch_count": a.get("mismatch_count")} for a in attestations],
    )
    upstream_manifests = {}
    for attestation_path, attestation in zip(attestation_paths, attestations):
        manifest_path = attestation_path.parent / "audit_manifest.json"
        report_path = attestation_path.parent / "validation_report.json"
        manifest = json.loads(manifest_path.read_text())
        upstream_manifests[attestation_path.parent.name] = manifest
        expected_manifest_hash = attestation.get("audit_manifest_sha256") or attestation.get("manifest_sha256")
        check(
            f"attestation_binding:{attestation_path.parent.name}",
            sha256(manifest_path) == expected_manifest_hash
            and sha256(report_path) == attestation.get("validation_report_sha256"),
            {"manifest": sha256(manifest_path), "report": sha256(report_path)},
            {"manifest": expected_manifest_hash, "report": attestation.get("validation_report_sha256")},
        )
    for year, rel in contract["base_oof"].items():
        owner = "REF4-ADAPTIVE-GATE-031B" if year == "2022" else "REF4-EXACT-OOF-031A"
        entry = upstream_manifests[owner]["artifacts"].get(rel)
        check(
            f"oof_manifest_binding:{year}",
            entry is not None and entry.get("sha256") == sha256(ROOT / rel) and entry.get("size") == (ROOT / rel).stat().st_size,
            {"entry": entry, "actual_sha256": sha256(ROOT / rel), "actual_size": (ROOT / rel).stat().st_size},
        )
    check(
        "fixed_candidate_contract",
        contract["candidate_count"] == 1
        and contract["ridge_contract"]["parameter_sweep"] is False
        and contract["production_fit"] is False
        and contract["test_read"] is False
        and contract["zip_creation"] is False,
        {k: contract[k] for k in ("candidate_count", "production_fit", "test_read", "zip_creation")},
    )
    check(
        "fixed_feature_count",
        len(contract["feature_contract"]["entities"])
        * len(contract["feature_contract"]["contexts"])
        * len(contract["feature_contract"]["shrinkage_grid"]) == 40,
        40,
    )
    check("rollback_hash", sha256(ROOT / contract["preserve_zip"]) == "fe5e7eb7731a7b16942a82b5ed144825a87accc05ecd508d69e578c68d288e2a", sha256(ROOT / contract["preserve_zip"]))

    train = pd.read_csv(ROOT / contract["official_train"], usecols=["row_id", "season", "control_success"])
    check("train_rows_unique", len(train) == 1_475_092 and train.row_id.is_unique, {"rows": len(train), "unique": bool(train.row_id.is_unique)})
    check("train_target", train.control_success.notna().all() and set(train.control_success.unique()) <= {0, 1}, sorted(train.control_success.unique().tolist()))
    for year, rel in contract["base_oof"].items():
        pred = pd.read_csv(ROOT / rel, usecols=["row_id", "season", "target", "prediction", "pitcher_id"])
        merged = pred.merge(train, on="row_id", how="left", validate="one_to_one", suffixes=("_oof", "_raw"))
        valid = (
            pred.row_id.is_unique
            and set(pred.season.unique()) == {int(year)}
            and len(merged) == len(pred)
            and merged.season_raw.notna().all()
            and np.array_equal(merged.target.to_numpy(int), merged.control_success.to_numpy(int))
            and np.isfinite(pred.prediction.to_numpy(float)).all()
            and pred.prediction.between(0, 1).all()
        )
        check(f"oof_{year}", valid, {"rows": len(pred), "season": sorted(pred.season.unique().tolist())})

    failures = [item["name"] for item in checks if not item["pass"]]
    report = {
        "experiment_id": contract["experiment_id"],
        "status": "AUDIT_VERIFIED" if not failures else "BLOCKED",
        "checked_count": len(checks),
        "pass_count": len(checks) - len(failures),
        "fail_count": len(failures),
        "failures": failures,
        "checks": checks,
        "training_performed": False,
        "test_read": False,
        "zip_created": False,
    }
    (OUT / "preflight_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (OUT / "preflight_report.md").write_text(
        f"# {contract['experiment_id']} preflight\n\n- status: `{report['status']}`\n- checked: `{len(checks)}`\n- failures: `{len(failures)}`\n"
    )
    print(json.dumps({k: report[k] for k in ("status", "checked_count", "pass_count", "fail_count", "failures")}, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
