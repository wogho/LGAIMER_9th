#!/usr/bin/env python3
"""Run the immutable 113A ZIP on 2024 features and audit calibrator compatibility.

This stage deliberately never reads the target column and never scores the
full-trained production model on its training rows.  It compares prediction
geometry only.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wasserstein_distance


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model/REF4-R-CROSSFIT-RELIABILITY-CURVE-121A"
TRAIN = ROOT / "data/train.csv"
ANCHOR = ROOT / "model/REF4-113A-V66-NESTED-117A/oof_predictions.csv"
ZIP = ROOT / "output/submit_ref4_super_ensemble_113A.zip"
CALIBRATOR = OUT / "production_calibrator.json"
STRICT_AUDIT = OUT / "audit_attestation.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bin_index(p: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.searchsorted(edges, p, side="right")


def population_stability_index(reference_bins: np.ndarray, candidate_bins: np.ndarray, n_bins: int) -> float:
    ref = np.bincount(reference_bins, minlength=n_bins).astype(float)
    cand = np.bincount(candidate_bins, minlength=n_bins).astype(float)
    ref = np.clip(ref / ref.sum(), 1e-9, None)
    cand = np.clip(cand / cand.sum(), 1e-9, None)
    return float(np.sum((cand - ref) * np.log(cand / ref)))


def correction(p: np.ndarray, edges: np.ndarray, deltas: np.ndarray) -> np.ndarray:
    raw = deltas[bin_index(p, edges)]
    calibrated = np.clip(p + raw, 0.005, 0.995)
    return 0.75 * p + 0.25 * calibrated


def main() -> None:
    started = time.time()
    strict_attestation = json.loads(STRICT_AUDIT.read_text(encoding="utf-8"))
    if strict_attestation.get("status") != "AUDIT_VERIFIED" or not strict_attestation.get("performance_gate_pass"):
        raise RuntimeError("strict 121A audit has not passed")
    contract = {
        "experiment_id": "REF4-R-CROSSFIT-RELIABILITY-CURVE-121A",
        "stage": "PRODUCTION_DISTRIBUTION_COMPATIBILITY",
        "status": "LOCKED_BEFORE_PRODUCTION_INFERENCE",
        "created_date": "2026-08-31",
        "purpose": "Compare prediction geometry only; do not evaluate the full-trained 113A against 2024 labels.",
        "input": "official train 2024 feature columns excluding control_success",
        "immutable_inference_source": "output/submit_ref4_super_ensemble_113A.zip",
        "fixed_gates": {
            "regular_spearman_min": 0.85,
            "regular_wasserstein_max": 0.03,
            "regular_abs_mean_difference_max": 0.02,
            "regular_std_ratio_range": [0.70, 1.40],
            "regular_probability_bin_PSI_max": 0.10,
            "regular_same_bin_fraction_min": 0.55,
            "regular_mean_abs_correction_ratio_range": [0.50, 2.00],
            "candidate_max_abs_change_max": 0.03,
            "F_exact_identity": True,
            "finite_unit_interval": True,
        },
        "forbidden": [
            "reading control_success in this stage",
            "using 2024 target to tune thresholds or the curve",
            "reading test.csv",
            "creating a submission ZIP before independent compatibility audit",
        ],
    }
    (OUT / "production_compatibility_contract.json").write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # The target is excluded at CSV parse time, rather than loaded and dropped.
    features = pd.read_csv(TRAIN, usecols=lambda name: name != "control_success", low_memory=False)
    features = features.loc[features["season"].eq(2024)].reset_index(drop=True)
    anchor = pd.read_csv(
        ANCHOR,
        usecols=["row_id", "season", "game_type", "p113a_strict"],
        low_memory=False,
    )
    strict = anchor.loc[anchor["season"].eq(2024)].reset_index(drop=True)
    if not np.array_equal(features["row_id"].astype(str), strict["row_id"].astype(str)):
        raise RuntimeError("2024 feature and strict prediction row order mismatch")

    with tempfile.TemporaryDirectory(prefix="ref4_121a_prod_") as temporary:
        sandbox = Path(temporary)
        with zipfile.ZipFile(ZIP) as archive:
            archive.extractall(sandbox)
        (sandbox / "data").mkdir(exist_ok=True)
        features.to_csv(sandbox / "data/test.csv", index=False)
        process = subprocess.run(
            [str(ROOT / ".venv/bin/python"), "script.py"],
            cwd=sandbox,
            text=True,
            capture_output=True,
            timeout=900,
            check=False,
        )
        (OUT / "production_compatibility_inference.log").write_text(
            f"returncode={process.returncode}\nSTDOUT\n{process.stdout}\nSTDERR\n{process.stderr}",
            encoding="utf-8",
        )
        if process.returncode != 0:
            raise RuntimeError(f"113A production inference failed: {process.stderr[-2000:]}")
        production = pd.read_csv(sandbox / "output/submission.csv", low_memory=False)

    if not np.array_equal(production["row_id"].astype(str), strict["row_id"].astype(str)):
        raise RuntimeError("production prediction row order mismatch")
    p_strict = strict["p113a_strict"].to_numpy(float)
    p_prod = production["control_success"].to_numpy(float)
    regular = strict["game_type"].eq("R").to_numpy()
    meta = json.loads(CALIBRATOR.read_text(encoding="utf-8"))
    edges = np.asarray(meta["inner_edges"], dtype=float)
    deltas = np.asarray(meta["raw_delta_by_bin"], dtype=float)
    strict_bins = bin_index(p_strict[regular], edges)
    prod_bins = bin_index(p_prod[regular], edges)
    p121_prod = p_prod.copy()
    p121_prod[regular] = correction(p_prod[regular], edges, deltas)
    strict_candidate = correction(p_strict[regular], edges, deltas)
    strict_change = np.abs(strict_candidate - p_strict[regular])
    prod_change = np.abs(p121_prod[regular] - p_prod[regular])
    strict_std = float(np.std(p_strict[regular]))
    prod_std = float(np.std(p_prod[regular]))
    metrics = {
        "rows": int(len(strict)),
        "regular_rows": int(regular.sum()),
        "futures_rows": int((~regular).sum()),
        "regular_spearman": float(spearmanr(p_strict[regular], p_prod[regular]).statistic),
        "regular_wasserstein": float(wasserstein_distance(p_strict[regular], p_prod[regular])),
        "regular_strict_mean": float(np.mean(p_strict[regular])),
        "regular_production_mean": float(np.mean(p_prod[regular])),
        "regular_abs_mean_difference": float(abs(np.mean(p_prod[regular]) - np.mean(p_strict[regular]))),
        "regular_strict_std": strict_std,
        "regular_production_std": prod_std,
        "regular_std_ratio": prod_std / strict_std,
        "regular_probability_bin_PSI": population_stability_index(strict_bins, prod_bins, 8),
        "regular_same_bin_fraction": float(np.mean(strict_bins == prod_bins)),
        "strict_bin_fractions": (np.bincount(strict_bins, minlength=8) / len(strict_bins)).tolist(),
        "production_bin_fractions": (np.bincount(prod_bins, minlength=8) / len(prod_bins)).tolist(),
        "strict_mean_abs_correction": float(strict_change.mean()),
        "production_mean_abs_correction": float(prod_change.mean()),
        "regular_mean_abs_correction_ratio": float(prod_change.mean() / strict_change.mean()),
        "candidate_max_abs_change": float(np.max(np.abs(p121_prod - p_prod))),
        "F_max_abs_difference": float(np.max(np.abs(p121_prod[~regular] - p_prod[~regular]))),
        "production_prediction_range": [float(p_prod.min()), float(p_prod.max())],
        "candidate_prediction_range": [float(p121_prod.min()), float(p121_prod.max())],
    }
    gates = {
        "regular_spearman_at_least_0_85": metrics["regular_spearman"] >= 0.85,
        "regular_wasserstein_at_most_0_03": metrics["regular_wasserstein"] <= 0.03,
        "regular_abs_mean_difference_at_most_0_02": metrics["regular_abs_mean_difference"] <= 0.02,
        "regular_std_ratio_in_0_70_1_40": 0.70 <= metrics["regular_std_ratio"] <= 1.40,
        "regular_probability_bin_PSI_at_most_0_10": metrics["regular_probability_bin_PSI"] <= 0.10,
        "regular_same_bin_fraction_at_least_0_55": metrics["regular_same_bin_fraction"] >= 0.55,
        "regular_mean_abs_correction_ratio_in_0_50_2_00": 0.50 <= metrics["regular_mean_abs_correction_ratio"] <= 2.00,
        "candidate_max_abs_change_at_most_0_03": metrics["candidate_max_abs_change"] <= 0.03 + 1e-15,
        "F_exact_identity": metrics["F_max_abs_difference"] == 0.0,
        "finite_unit_interval": bool(np.isfinite(p_prod).all() and np.isfinite(p121_prod).all() and p_prod.min() >= 0.0 and p_prod.max() <= 1.0 and p121_prod.min() >= 0.0 and p121_prod.max() <= 1.0),
    }
    compatible = all(gates.values())
    output = pd.DataFrame(
        {
            "row_id": strict["row_id"].astype(str),
            "game_type": strict["game_type"].astype(str),
            "p113a_strict": p_strict,
            "p113a_production_full": p_prod,
            "p121a_production_geometry": p121_prod,
        }
    )
    output.to_csv(OUT / "production_compatibility_predictions.csv.gz", index=False, compression="gzip")
    report = {
        "experiment_id": contract["experiment_id"],
        "status": "PRODUCTION_COMPATIBILITY_RUN_COMPLETE",
        "target_column_read": False,
        "target_metric_computed": False,
        "test_read": False,
        "champion_zip_sha256": sha256(ZIP),
        "calibrator_sha256": sha256(CALIBRATOR),
        "metrics": metrics,
        "gates": gates,
        "gate_count": len(gates),
        "compatibility_gate_pass": compatible,
        "decision": "ELIGIBLE_FOR_121A_PRODUCTION_BUILD" if compatible else "REJECT_KEEP_113A",
        "zip_created": False,
        "elapsed_seconds": float(time.time() - started),
        "prediction_sha256": sha256(OUT / "production_compatibility_predictions.csv.gz"),
        "inference_log_sha256": sha256(OUT / "production_compatibility_inference.log"),
    }
    (OUT / "production_compatibility_result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
