#!/usr/bin/env python3
"""Independent evidence audit for REF-AUX-OFFSET-CAT-001."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "model" / "REF-AUX-OFFSET-CAT-001"
REPORT = EXP / "validation_report.json"
ATTEST = EXP / "attestation.json"
TRAIN = ROOT / "data" / "train.csv"
LABELS = ROOT / "model" / "REF-AUX-LABEL-001" / "recovered_labels.csv.gz"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fail(checks, name, detail):
    checks.append({"name": name, "checked": True, "pass": False, "detail": detail})


def main() -> None:
    checks = []
    report = json.loads(REPORT.read_text(encoding="utf-8"))
    att = json.loads(ATTEST.read_text(encoding="utf-8"))
    if report["experiment_id"] != "REF-AUX-OFFSET-CAT-001":
        fail(checks, "experiment_id", report.get("experiment_id"))
    else:
        checks.append({"name": "experiment_id", "checked": True, "pass": True})

    train = pd.read_csv(TRAIN, usecols=["row_id", "season", "control_success"])
    labels = pd.read_csv(LABELS, usecols=["row_id", "success", "middle", "reverse"])
    if len(train) != len(labels) or not train["row_id"].equals(labels["row_id"]):
        fail(checks, "train_label_alignment", {"train_rows": len(train), "label_rows": len(labels)})
    else:
        checks.append({"name": "train_label_alignment", "checked": True, "pass": True, "rows": int(len(train))})
    valid = labels[["success", "middle", "reverse"]].notna().all(axis=1)
    binary = labels.loc[valid, ["success", "middle", "reverse"]].isin([0, 1]).all().all()
    checks.append({"name": "recovered_labels_binary", "checked": True, "pass": bool(binary), "valid_rows": int(valid.sum())})

    expected_sources = {"train": TRAIN, "labels": LABELS}
    pred_paths = {
        "pred_2022": ROOT / "model/ENS-CATF-LGBMCATR5050-FE001-EW-2022/selective_predictions_2022.csv",
        "pred_2023": ROOT / "model/ENS-CATF-LGBMCATR5050-FE001/selective_predictions_2023.csv",
        "pred_2024": ROOT / "model/ENS-CATF-LGBMCATR5050-FE001/selective_predictions_2024.csv",
    }
    expected_sources.update(pred_paths)
    source_ok = True
    for key, path in expected_sources.items():
        actual = sha256(path)
        expected = report["source_hashes"][key]
        source_ok &= actual == expected
    checks.append({"name": "source_hashes", "checked": True, "pass": bool(source_ok), "count": len(expected_sources)})

    artifacts_ok = True
    for item in report["aux_artifacts"] + report["transition_artifacts"]:
        path = ROOT / item["path"]
        ok = path.is_file() and sha256(path) == item["sha256"]
        artifacts_ok &= ok
    checks.append({"name": "artifact_hashes", "checked": True, "pass": bool(artifacts_ok), "count": len(report["aux_artifacts"]) + len(report["transition_artifacts"])})

    transitions_ok = True
    recomputed = []
    for item, recorded in zip(report["transition_artifacts"], report["transitions"]):
        data = np.load(ROOT / item["path"])
        y = np.asarray(data["y"], dtype=np.float64)
        baseline = np.asarray(data["baseline"], dtype=np.float64)
        offset = np.asarray(data["offset"], dtype=np.float64)
        finite = np.isfinite(y).all() and np.isfinite(baseline).all() and np.isfinite(offset).all()
        bounded = ((baseline >= 0) & (baseline <= 1)).all() and ((offset >= 0) & (offset <= 1)).all()
        delta = float(np.mean((y - offset) ** 2) - np.mean((y - baseline) ** 2))
        expected_rows = int((train["season"] == item["apply_season"]).sum())
        ok = finite and bounded and len(y) == item["n_rows"] == recorded["n_rows"] == expected_rows and np.isclose(delta, recorded["delta_brier"], rtol=0, atol=1e-15)
        transitions_ok &= ok
        recomputed.append({"fit_season": item["fit_season"], "apply_season": item["apply_season"], "rows": len(y), "delta_brier": delta, "pass": bool(ok)})
    checks.append({"name": "transition_recompute", "checked": True, "pass": bool(transitions_ok), "count": len(recomputed), "results": recomputed})

    report_hash_ok = sha256(REPORT) == att["report_sha256"]
    checks.append({"name": "source_attestation_report_hash", "checked": True, "pass": bool(report_hash_ok)})
    all_pass = all(c["pass"] for c in checks)
    result = {"audit_id": "REF-AUX-OFFSET-CAT-001-INDEPENDENT-AUDIT", "status": "AUDIT_VERIFIED" if all_pass else "AUDIT_FAIL", "checks": checks, "checked_count": len(checks), "failed_count": sum(not c["pass"] for c in checks)}
    out = EXP / "independent_audit_report.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    independent_att = {"attestation_id": "REF-AUX-OFFSET-CAT-001-INDEPENDENT-AUDIT-ATTESTATION", "report_sha256": sha256(out), "validator_sha256": sha256(Path(__file__)), "checked_count": len(checks), "failed_count": sum(not c["pass"] for c in checks), "status": result["status"]}
    (EXP / "independent_audit_attestation.json").write_text(json.dumps(independent_att, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
