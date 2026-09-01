#!/usr/bin/env python3
"""Independent metric, package, and row-independence audit for 119A/B/C."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESEARCH = ROOT / "model/REF4-119-RESEARCH"
BASE = ROOT / "model/REF4-DISJOINT-EB-113A/production_package"
CANDIDATES = {
    "119A": (ROOT / "model/REF4-ABS-ERA-RESIDUAL-119A/production_package", "apply_abs_era_119a", 0.012),
    "119B": (ROOT / "model/REF4-LATENT-PITCH-MARGINAL-MOE-119B/production_package", "apply_latent_moe_119b", 0.012),
    "119C": (ROOT / "model/REF4-TRACKMAN-QUANTILE-DRIFT-119C/production_package", "apply_quantile_drift_119c", 0.010),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def run_package(package: Path, test: pd.DataFrame) -> tuple[np.ndarray, float]:
    with tempfile.TemporaryDirectory(prefix="ref4_119_verify_") as directory:
        sandbox = Path(directory) / "package"
        shutil.copytree(package, sandbox, copy_function=os.link)
        (sandbox / "data").mkdir()
        test.to_csv(sandbox / "data/test.csv", index=False)
        started = time.time()
        process = subprocess.run(
            [str(ROOT / ".venv/bin/python"), "script.py"], cwd=sandbox, capture_output=True, text=True, timeout=180
        )
        elapsed = time.time() - started
        if process.returncode:
            raise RuntimeError(f"{package}: {process.stderr[-4000:]}")
        submission = pd.read_csv(sandbox / "output/submission.csv")
        if not np.array_equal(submission["row_id"].astype(str), test["row_id"].astype(str)):
            raise RuntimeError(f"row_id mismatch: {package}")
        return submission["control_success"].to_numpy(float), elapsed


def main() -> None:
    checks = []

    def check(name: str, passed: bool, actual=None):
        checks.append({"name": name, "passed": bool(passed), "actual": actual})
        if not passed:
            raise RuntimeError(f"audit failed: {name}: {actual}")

    report = json.loads((RESEARCH / "comparison_report.json").read_text(encoding="utf-8"))
    build = json.loads((RESEARCH / "build_report.json").read_text(encoding="utf-8"))
    check("train_hash", sha256(ROOT / "data/train.csv") == report["provenance"]["train_sha256"])
    check("trackman_hash", sha256(ROOT / "data/trackman_history.csv") == report["provenance"]["trackman_sha256"])
    check("anchor_hash", sha256(ROOT / "model/REF4-113A-V66-NESTED-117A/oof_predictions.csv") == report["provenance"]["strict_anchor_sha256"])
    check("test_not_read_during_training", report["test_read"] is False)

    for version in CANDIDATES:
        prediction = pd.read_csv(RESEARCH / f"oof_{version.lower()}.csv")
        gain = float(np.mean(np.square(prediction["baseline"] - prediction["target"]) - np.square(prediction["candidate"] - prediction["target"])))
        expected = float(np.average([item["brier_gain"] for item in report[version]["results"]], weights=[item["rows"] for item in report[version]["results"]]))
        check(f"{version}_oof_unique", prediction["row_id"].is_unique, int(prediction["row_id"].nunique()))
        check(f"{version}_oof_finite", np.isfinite(prediction[["baseline", "candidate"]].to_numpy()).all())
        check(f"{version}_metric_reproduction", abs(gain - expected) <= 1e-15, {"actual": gain, "expected": expected})

    sys_path_added = False
    import sys
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
        sys_path_added = True
    from src import ref4_119

    sample = pd.read_csv(ROOT / "data/train.csv", nrows=8, low_memory=False).drop(columns=["control_success"])
    anchor = np.linspace(0.20, 0.80, len(sample))
    for version, (package, function_name, cap) in CANDIDATES.items():
        function = getattr(ref4_119, function_name)
        full = function(sample.reset_index(drop=True), anchor.copy(), package / "model")
        singles = np.array([function(sample.iloc[[i]].reset_index(drop=True), anchor[[i]].copy(), package / "model")[0] for i in range(len(sample))])
        order = np.array([5, 1, 7, 0, 4, 2, 6, 3])
        permuted = function(sample.iloc[order].reset_index(drop=True), anchor[order].copy(), package / "model")
        restored = np.empty_like(permuted)
        restored[order] = permuted
        check(f"{version}_single_row_independence", np.max(np.abs(full - singles)) <= 1e-12, float(np.max(np.abs(full - singles))))
        check(f"{version}_permutation_independence", np.max(np.abs(full - restored)) <= 1e-12, float(np.max(np.abs(full - restored))))
        check(f"{version}_direct_delta_cap", np.max(np.abs(full - anchor)) <= cap + 1e-12, float(np.max(np.abs(full - anchor))))
    if sys_path_added:
        sys.path.remove(str(ROOT))

    test = pd.read_csv(ROOT / "data/test.csv", low_memory=False)
    base_prediction, base_time = run_package(BASE, test)
    runtime = {"113A": base_time}
    for version, (package, _, cap) in CANDIDATES.items():
        prediction, elapsed = run_package(package, test)
        runtime[version] = elapsed
        delta = float(np.max(np.abs(prediction - base_prediction)))
        check(f"{version}_production_finite_range", np.isfinite(prediction).all() and prediction.min() >= 0 and prediction.max() <= 1, [float(prediction.min()), float(prediction.max())])
        check(f"{version}_production_delta_cap", delta <= cap + 1e-12, delta)
        if version == "119C":
            check("119C_exact_113A_rollback", np.array_equal(prediction, base_prediction), delta)
        zip_path = ROOT / f"output/submit_ref4_super_ensemble_{version}.zip"
        check(f"{version}_zip_hash", sha256(zip_path) == build["packages"][version]["zip_sha256"])
        with zipfile.ZipFile(zip_path) as archive:
            bad = archive.testzip()
            names = set(archive.namelist())
            check(f"{version}_zip_integrity", bad is None, bad)
            check(f"{version}_zip_root", {"script.py", "requirements.txt", "model/manifest.json"}.issubset(names))
        tree = ast.parse((package / "script.py").read_text(encoding="utf-8"))
        forbidden = {"socket", "requests", "urllib", "httpx"}
        imports = {node.names[0].name.split(".")[0] for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) and getattr(node, "names", None)}
        check(f"{version}_network_imports", not (imports & forbidden), sorted(imports & forbidden))

    output = {
        "experiment": "REF4-119-ABC",
        "status": "AUDIT_VERIFIED",
        "checks": checks,
        "checked_count": len(checks),
        "passed_count": sum(item["passed"] for item in checks),
        "runtime_seconds_on_local_test": runtime,
        "candidate_gate_status": {version: bool(report[version]["gate_pass"]) for version in CANDIDATES},
    }
    (RESEARCH / "verification_report.json").write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": output["status"], "checked": output["checked_count"], "runtime": runtime}, indent=2))


if __name__ == "__main__":
    main()
