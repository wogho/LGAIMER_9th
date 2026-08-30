#!/usr/bin/env python3
"""Fail-fast preflight for REF4-TRAINONLY-RECENT-STACK-041A."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model/REF4-TRAINONLY-RECENT-STACK-041A"
CONTRACT = OUT / "audit_contract.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    checks: list[dict[str, object]] = []

    def check(name: str, passed: bool, actual: object) -> None:
        checks.append({"name": name, "checked": True, "pass": bool(passed), "actual": actual})

    required = [
        ROOT / "start02_dev_log.md", ROOT / "start03_reference.md",
        ROOT / "start04_uptostage.md", ROOT / "01_제약과금지사항.md",
        ROOT / "output/submit_ref4_champion_030.zip",
        ROOT / "model/REF4-CHAMPION-PROVENANCE-DOSSIER-040A/audit_attestation.json",
        ROOT / "model/REF4-CHAMPION-STACK-030/manifest.json",
        ROOT / "model/REF4-CHAMPION-STACK-030/f_regime_meta.json",
    ] + [ROOT / p for p in contract["source_folds"].values()]
    for path in required:
        check(f"exists:{path.relative_to(ROOT)}", path.is_file(), path.stat().st_size if path.is_file() else None)

    check("experiment_id", contract.get("experiment_id") == "REF4-TRAINONLY-RECENT-STACK-041A", contract.get("experiment_id"))
    check("single_hypothesis_declared", isinstance(contract.get("single_hypothesis"), str) and len(contract["single_hypothesis"]) > 30, contract.get("single_hypothesis"))
    check("one_fixed_estimator", contract.get("fixed_estimator") == {"type": "StandardScaler_then_Ridge", "alpha": 1000.0, "fit_intercept": True, "training_objective": "squared_error"}, contract.get("fixed_estimator"))
    check("temporal_protocol", contract.get("temporal_protocol") == [{"fit_season": 2022, "validation_season": 2023}, {"fit_season": 2023, "validation_season": 2024}], contract.get("temporal_protocol"))
    check("test_blocked", contract.get("test_access_allowed_before_audit_verified_gate_pass") is False, contract.get("test_access_allowed_before_audit_verified_gate_pass"))
    check("zip_blocked", contract.get("zip_allowed_before_audit_verified_gate_pass") is False, contract.get("zip_allowed_before_audit_verified_gate_pass"))

    dossier = json.loads(required[5].read_text(encoding="utf-8"))
    check("upstream_040a_verified", dossier.get("status") == "AUDIT_VERIFIED" and dossier.get("lineage_status") == "PROVENANCE_DOSSIER_VERIFIED" and dossier.get("fail_count") == 0, {"status": dossier.get("status"), "lineage_status": dossier.get("lineage_status"), "fail_count": dossier.get("fail_count")})
    champion_hash = sha256(ROOT / contract["preserve_zip"])
    check("rollback_zip_hash", champion_hash == "ca6af4bdf26d4bc79d503cdc6a09b0057a7ea9bdd77e29012f9e459a50cd66b8", champion_hash)

    expected_cols = {
        "row_id", "season", "game_type", "pitcher_id", "target",
        "p_v2_global", "p_v2_f", "p_v3_55_global", "p_v3_55_f",
        "p_v3_30_global", "p_v3_30_f_all", "p_v3_30_f_recent",
        "risk_middle_global", "risk_middle_f", "risk_wild_global", "risk_wild_f",
        "risk_reverse_global", "risk_reverse_f", "prediction_no_shift", "prediction",
    }
    source_summary = {}
    for year, rel in contract["source_folds"].items():
        path = ROOT / rel
        header = pd.read_csv(path, nrows=0)
        source_summary[year] = {"columns": len(header.columns), "sha256": sha256(path), "size": path.stat().st_size}
        check(f"source_columns_{year}", expected_cols.issubset(header.columns), sorted(expected_cols - set(header.columns)))
    check("source_paths_distinct", len(set(contract["source_folds"].values())) == len(contract["source_folds"]), contract["source_folds"])
    check("out_is_new_isolated_dir", OUT != ROOT / "model/REF4-CHAMPION-STACK-030", str(OUT.relative_to(ROOT)))

    failures = [c["name"] for c in checks if not c["pass"]]
    report = {
        "experiment_id": contract["experiment_id"], "status": "AUDIT_VERIFIED" if not failures else "BLOCKED",
        "checked_count": len(checks), "pass_count": len(checks) - len(failures), "fail_count": len(failures),
        "failures": failures, "checks": checks, "source_summary": source_summary,
        "test_read": False, "training_performed": False, "zip_created": False,
    }
    (OUT / "preflight_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUT / "preflight_report.md").write_text(
        f"# {contract['experiment_id']} preflight\n\n- status: `{report['status']}`\n- checked: `{len(checks)}`\n- failures: `{len(failures)}`\n",
        encoding="utf-8",
    )
    print(json.dumps({k: report[k] for k in ("status", "checked_count", "pass_count", "fail_count", "failures")}, ensure_ascii=False, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
