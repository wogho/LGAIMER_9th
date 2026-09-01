#!/usr/bin/env python3
"""End-to-end technical audit for the audited 110A production directory."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT = ROOT / "model/REF4-110-ORIGINAL-R2"
DEST = ROOT / "model/REF4-TEMPORAL-CROSSFIT-MOE-110A-R2"
PROD = DEST / "production_package"
PYTHON = ROOT / ".venv/bin/python"
TEST = ROOT / "data/test.csv"
STRESS = ROOT / "model/REF4-SUPER-BLEND-CHAMPION-077A/test_sandbox/repo3/data/test.csv"
ROLLBACK = ROOT / "output/submit_ref4_super_ensemble_109C.zip"
ROLLBACK_SHA = "03e874949ae7172af0dab16d9c3f52de94d5ac9256e571ddedd377b785d9634f"
FORBIDDEN = {"socket", "requests", "urllib", "http", "ftplib", "paramiko"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    checks: list[dict] = []

    def check(name: str, passed: bool, actual=None) -> None:
        checks.append({"name": name, "checked": True, "passed": bool(passed), "actual": actual})
        if not passed:
            raise RuntimeError(f"technical check failed: {name}: {actual}")

    performance = json.loads((EXPERIMENT / "validation_report.json").read_text())
    build = json.loads((DEST / "build_provenance.json").read_text())
    check("performance_audit", performance.get("status") == "AUDIT_VERIFIED" and performance.get("provisional_winner") == "110A", performance.get("provisional_winner"))
    check("build_pending", build.get("status") == "PENDING_TECHNICAL_AUDIT" and build.get("test_read") is False and build.get("zip_created") is False, build.get("status"))
    for relative in ("script.py", "requirements.txt", "model", "src", "model/manifest.json"):
        check(f"layout:{relative}", (PROD / relative).exists(), relative)
    manifest = json.loads((PROD / "model/manifest.json").read_text())
    check("manifest_version", manifest.get("version") == "REF4-TEMPORAL-CROSSFIT-MOE-110A-R2", manifest.get("version"))
    check("router_rows", manifest.get("moe_router_train_rows") == 746_504, manifest.get("moe_router_train_rows"))
    model_names = {path.name for path in (PROD / "model").iterdir() if path.is_file()}
    expected_108 = {f"moe108_super_resid_{family}_seed{seed}.{suffix}" for family, suffix in (("cb", "cbm"), ("lgb", "txt")) for seed in (42, 1, 2, 3, 4)}
    expected_108 |= {f"moe108_fut_resid_cb_seed{seed}.cbm" for seed in (42, 1, 2, 3)}
    expected_router = {f"moe_router_seed{seed}.cbm" for seed in (42, 1, 2)}
    check("108_model_set", expected_108.issubset(model_names) and len(expected_108) == 14, sorted(expected_108 - model_names))
    check("router_model_set", expected_router.issubset(model_names), sorted(expected_router - model_names))
    tree = ast.parse((PROD / "script.py").read_text())
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module.split(".")[0])
    check("static_network", not (imports & FORBIDDEN), sorted(imports & FORBIDDEN))
    check("rollback_integrity", sha256(ROLLBACK) == ROLLBACK_SHA, sha256(ROLLBACK))

    print("[technical] reading official dummy and audited 2024 stress source only after performance audit", flush=True)
    dummy = pd.read_csv(TEST, low_memory=False)
    check("official_dummy_rows", len(dummy) == 5, len(dummy))
    stress = pd.read_csv(STRESS, low_memory=False)
    if "control_success" in stress.columns:
        stress = stress.drop(columns=["control_success"])
    check("stress_rows", len(stress) == 253_507, len(stress))
    check("stress_schema", set(stress.columns) == set(dummy.columns), {"stress": len(stress.columns), "dummy": len(dummy.columns)})
    data_dir, output_dir = PROD / "data", PROD / "output"
    data_dir.mkdir(exist_ok=False)
    output_dir.mkdir(exist_ok=False)

    def run_frame(frame: pd.DataFrame, label: str, timeout: float = 600.0) -> tuple[pd.DataFrame, float]:
        frame.to_csv(data_dir / "test.csv", index=False)
        submission = output_dir / "submission.csv"
        if submission.exists():
            submission.unlink()
        started = time.monotonic()
        completed = subprocess.run([str(PYTHON), "script.py"], cwd=PROD, text=True, capture_output=True, timeout=timeout)
        elapsed = time.monotonic() - started
        if completed.returncode != 0:
            raise RuntimeError(f"{label} inference failed: {completed.stderr[-4000:]}")
        if not submission.exists():
            raise RuntimeError(f"{label} did not create submission.csv")
        result = pd.read_csv(submission, dtype={"row_id": str})
        print(f"[technical] {label}: rows={len(result)} elapsed={elapsed:.2f}s", flush=True)
        return result, elapsed

    try:
        official, official_elapsed = run_frame(dummy, "official_dummy5")
        check("official_dummy_output", len(official) == 5, len(official))
        sample = stress.iloc[[0, 17, 101, 999, 5001, 20001, 70001, 140001]].reset_index(drop=True)
        batch, _ = run_frame(sample, "batch8")
        check("batch_rows", len(batch) == 8, len(batch))
        check("batch_ids", batch["row_id"].astype(str).tolist() == sample["row_id"].astype(str).tolist(), None)
        batch_p = batch["control_success"].to_numpy(float)
        check("batch_values", np.isfinite(batch_p).all() and ((batch_p >= 0) & (batch_p <= 1)).all(), [float(batch_p.min()), float(batch_p.max())])

        shuffled, _ = run_frame(sample.sample(frac=1, axis=1, random_state=110), "column_permutation")
        shuffled_p = shuffled.set_index("row_id").loc[batch["row_id"], "control_success"].to_numpy(float)
        column_diff = float(np.max(np.abs(batch_p - shuffled_p)))
        check("column_permutation_invariance", column_diff <= 1e-12, column_diff)

        singles = []
        for pos in range(len(sample)):
            one, _ = run_frame(sample.iloc[[pos]], f"single_{pos}")
            singles.append(float(one["control_success"].iloc[0]))
        independence_diff = float(np.max(np.abs(batch_p - np.asarray(singles))))
        check("row_independence", independence_diff <= 1e-12, independence_diff)

        full_result, full_elapsed = run_frame(stress, "stress_253507", timeout=600.0)
        full_p = full_result["control_success"].to_numpy(float)
        check("stress_output_rows", len(full_result) == len(stress), len(full_result))
        check("stress_output_ids", full_result["row_id"].astype(str).tolist() == stress["row_id"].astype(str).tolist(), None)
        check("stress_output_values", np.isfinite(full_p).all() and ((full_p >= 0) & (full_p <= 1)).all(), [float(full_p.min()), float(full_p.max())])
        check("runtime_seconds_max_600", full_elapsed <= 600.0, full_elapsed)
        prediction_sha = sha256(output_dir / "submission.csv")
        prediction_summary = {"kind": "2024_holdout_stress_not_hidden_test", "source_sha256": sha256(STRESS), "rows": len(full_result), "min": float(full_p.min()), "max": float(full_p.max()), "mean": float(full_p.mean()), "std": float(full_p.std()), "sha256": prediction_sha}
    finally:
        for path in (data_dir / "test.csv", output_dir / "submission.csv"):
            if path.exists():
                path.unlink()
        for directory in (data_dir, output_dir):
            if directory.exists():
                directory.rmdir()

    failed = [item for item in checks if not item["passed"]]
    report = {
        "experiment_id": "REF4-TEMPORAL-CROSSFIT-MOE-110A-R2",
        "status": "AUDIT_VERIFIED" if not failed else "MISMATCH",
        "checked_count": len(checks),
        "passed_count": len(checks) - len(failed),
        "mismatch_count": len(failed),
        "failures": failed,
        "row_independence_max_diff": independence_diff,
        "column_permutation_max_diff": column_diff,
        "stress_253507_runtime_seconds": full_elapsed,
        "prediction_summary": prediction_summary,
        "zip_created": False,
        "checks": checks,
    }
    report_path = DEST / "technical_validation_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    manifest_out = {
        "experiment_id": report["experiment_id"],
        "status": report["status"],
        "artifacts": {
            "script.py": sha256(PROD / "script.py"),
            "requirements.txt": sha256(PROD / "requirements.txt"),
            "model/manifest.json": sha256(PROD / "model/manifest.json"),
            "build_provenance.json": sha256(DEST / "build_provenance.json"),
            "technical_validation_report.json": sha256(report_path),
            "builder": sha256(ROOT / "scripts/build_ref4_110a_production_r2.py"),
            "validator": sha256(Path(__file__)),
        },
        "production_model_file_count": sum(path.is_file() for path in (PROD / "model").iterdir()),
    }
    manifest_path = DEST / "technical_audit_manifest.json"
    manifest_path.write_text(json.dumps(manifest_out, indent=2) + "\n")
    attestation = {
        "experiment_id": report["experiment_id"],
        "status": report["status"],
        "manifest_sha256": sha256(manifest_path),
        "validation_report_sha256": sha256(report_path),
        "validator_sha256": sha256(Path(__file__)),
        "checked_count": report["checked_count"],
        "passed_count": report["passed_count"],
        "mismatch_count": report["mismatch_count"],
        "row_independence_max_diff": independence_diff,
        "stress_253507_runtime_seconds": full_elapsed,
    }
    (DEST / "technical_audit_attestation.json").write_text(json.dumps(attestation, indent=2) + "\n")
    print(json.dumps(attestation, indent=2), flush=True)


if __name__ == "__main__":
    main()
