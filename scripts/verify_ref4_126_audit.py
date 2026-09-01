#!/usr/bin/env python3
"""Independent, source-array audit for REF4-126 strict validation artifacts."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import platform
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import catboost
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "REF4-JM-R-RESIDUAL-STRICT-GPU-126"
EXPERIMENT_DIR = ROOT / "model" / EXPERIMENT_ID
RAW_DIR = EXPERIMENT_DIR / "remote_raw"
MANIFEST_PATH = EXPERIMENT_DIR / "audit_manifest.json"
REPORT_PATH = EXPERIMENT_DIR / "validation_report.json"
ATTESTATION_PATH = EXPERIMENT_DIR / "audit_attestation.json"
RUNNER_PATH = ROOT / "scripts" / "run_ref4_jm_r_residual_strict_gpu_126.py"
TRAIN_PATH = ROOT / "data" / "train.csv"
ANCHOR_PATH = ROOT / "model" / "REF4-113A-V66-NESTED-117A" / "oof_predictions.csv"
CODE_ZIP_PATH = ROOT / "colab" / "126" / "REF4_126_CODE.zip"
SHA_MANIFEST_PATH = ROOT / "colab" / "126" / "SHA256SUMS"

EXPECTED = {
    "train_sha256": "d2081186b458b49f60b082be480c273135833e15ba59a76d033af28bcf8763ff",
    "anchor_sha256": "560e1ca40a21f0b9b296f612e6764e50eaa2a6f62b08561b86dd9d1803c23aa6",
    "runner_sha256": "a08d76b6ba55d757010c86d167eeda65e0d76aa93d06303b81db212f3a5b7def",
    "code_zip_sha256": "39d2176a1eb75ee5c14fe93562023b6c8dd19dfcdf5c089f59af2a480d8af891",
    "rows": {2022: 247_472, 2023: 245_525, 2024: 253_507},
    "seeds": [17, 42, 777],
    "scales": [0.0, 0.025, 0.05, 0.075],
    "time_weights": {2022: 0.2, 2023: 0.3, 2024: 0.5},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    encoded = (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    with temporary.open("wb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def row_fingerprint(values: pd.Series) -> str:
    hashes = pd.util.hash_pandas_object(values.astype(str), index=False).to_numpy(np.uint64)
    return hashlib.sha256(hashes.tobytes()).hexdigest()


def brier(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.square(np.asarray(prediction, float) - np.asarray(target, float))))


def bss(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, float)
    reference = float(target.mean() * (1.0 - target.mean()))
    return float(100_000.0 * (1.0 - brier(target, prediction) / reference))


def ece_equal_width(target: np.ndarray, prediction: np.ndarray, bins: int = 10) -> float:
    target = np.asarray(target, float)
    prediction = np.asarray(prediction, float)
    indices = np.minimum((prediction * bins).astype(int), bins - 1)
    total = len(target)
    value = 0.0
    for index in range(bins):
        mask = indices == index
        if mask.any():
            value += float(mask.sum() / total) * abs(float(target[mask].mean() - prediction[mask].mean()))
    return value


def close(left: Any, right: Any, tolerance: float = 5e-12) -> bool:
    try:
        return bool(math.isclose(float(left), float(right), rel_tol=0.0, abs_tol=tolerance))
    except (TypeError, ValueError):
        return left == right


class Audit:
    def __init__(self) -> None:
        self.checks: list[dict[str, Any]] = []

    def add(self, name: str, passed: bool, actual: Any = None, expected: Any = None) -> None:
        self.checks.append(
            {
                "name": name,
                "status": "PASS" if bool(passed) else "FAIL",
                "actual": actual,
                "expected": expected,
            }
        )

    @property
    def failures(self) -> list[dict[str, Any]]:
        return [item for item in self.checks if item["status"] != "PASS"]


def parse_sha_manifest(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        digest, relative = raw.split(maxsplit=1)
        values[relative.lstrip("*")] = digest
    return values


def bootstrap_pitcher_2024(frame: pd.DataFrame, repeats: int) -> dict[str, Any]:
    row_delta = np.square(frame["p126"].to_numpy(float) - frame["target"].to_numpy(float))
    row_delta -= np.square(
        frame["p113a_strict"].to_numpy(float) - frame["target"].to_numpy(float)
    )
    grouped = pd.DataFrame(
        {"pitcher": frame["pitcher_id"].astype(str), "delta": row_delta}
    ).groupby("pitcher", sort=False)["delta"].agg(["sum", "size"])
    sums = grouped["sum"].to_numpy(float)
    sizes = grouped["size"].to_numpy(float)
    rng = np.random.default_rng(1262024)
    values = np.empty(repeats, dtype=float)
    for start in range(0, repeats, 64):
        count = min(64, repeats - start)
        sample = rng.integers(0, len(grouped), size=(count, len(grouped)))
        values[start : start + count] = sums[sample].sum(axis=1) / sizes[sample].sum(axis=1)
    return {
        "repeats": repeats,
        "pitcher_clusters": int(len(grouped)),
        "delta_brier": float(row_delta.mean()),
        "bootstrap_mean_delta_brier": float(values.mean()),
        "ci_low": float(np.quantile(values, 0.025)),
        "ci_high": float(np.quantile(values, 0.975)),
        "improvement_probability": float(np.mean(values < 0.0)),
    }


def load_runner() -> Any:
    specification = importlib.util.spec_from_file_location("ref4_126_runner_audit", RUNNER_PATH)
    if specification is None or specification.loader is None:
        raise ImportError(RUNNER_PATH)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def main() -> int:
    audit = Audit()
    manifest = load_json(MANIFEST_PATH)
    result = load_json(RAW_DIR / "results" / "result.json")
    config = load_json(RAW_DIR / "results" / "run_config.json")
    launch = load_json(RAW_DIR / "logs" / "last_launch.json")
    remote_verification = load_json(EXPERIMENT_DIR / "remote_transfer_verification.json")

    audit.add("manifest_experiment", manifest.get("experiment_id") == EXPERIMENT_ID, manifest.get("experiment_id"), EXPERIMENT_ID)
    audit.add("manifest_entry_count", manifest.get("entry_count") == len(manifest.get("entries", [])), manifest.get("entry_count"), len(manifest.get("entries", [])))
    for entry in manifest["entries"]:
        path = ROOT / entry["path"]
        audit.add(f"manifest_exists:{entry['path']}", path.is_file(), path.is_file(), True)
        if path.is_file():
            audit.add(f"manifest_bytes:{entry['path']}", path.stat().st_size == entry["bytes"], path.stat().st_size, entry["bytes"])
            actual_hash = sha256(path)
            audit.add(f"manifest_sha256:{entry['path']}", actual_hash == entry["sha256"], actual_hash, entry["sha256"])
    audit.add("remote_checks_count", len(remote_verification["checks"]) == 3, len(remote_verification["checks"]), 3)
    audit.add("remote_checks_all_pass", remote_verification["all_pass"] is True, remote_verification["all_pass"], True)
    for item in remote_verification["checks"]:
        audit.add(f"remote_check:{item['name']}", item["returncode"] == 0 and item["pass"] is True, item["returncode"], 0)

    fixed_hashes = {
        "train": (TRAIN_PATH, EXPECTED["train_sha256"]),
        "anchor": (ANCHOR_PATH, EXPECTED["anchor_sha256"]),
        "runner": (RUNNER_PATH, EXPECTED["runner_sha256"]),
        "code_zip": (CODE_ZIP_PATH, EXPECTED["code_zip_sha256"]),
    }
    for name, (path, expected_hash) in fixed_hashes.items():
        actual_hash = sha256(path)
        audit.add(f"fixed_hash:{name}", actual_hash == expected_hash, actual_hash, expected_hash)

    staged_manifest = parse_sha_manifest(SHA_MANIFEST_PATH)
    audit.add("staged_manifest_entries", len(staged_manifest) == 8, len(staged_manifest), 8)
    audit.add("staged_train_hash", staged_manifest.get("data/train.csv") == EXPECTED["train_sha256"], staged_manifest.get("data/train.csv"), EXPECTED["train_sha256"])
    audit.add("staged_anchor_hash", staged_manifest.get("anchor/strict_113A/oof_predictions.csv") == EXPECTED["anchor_sha256"], staged_manifest.get("anchor/strict_113A/oof_predictions.csv"), EXPECTED["anchor_sha256"])
    audit.add("staged_code_hash", staged_manifest.get("code/REF4_126_CODE.zip") == EXPECTED["code_zip_sha256"], staged_manifest.get("code/REF4_126_CODE.zip"), EXPECTED["code_zip_sha256"])
    with zipfile.ZipFile(CODE_ZIP_PATH) as archive:
        bad_members = archive.testzip()
        runner_bytes = archive.read("scripts/run_ref4_jm_r_residual_strict_gpu_126.py")
        requirements = archive.read("requirements-colab.txt").decode("utf-8").strip()
    audit.add("code_zip_integrity", bad_members is None, bad_members, None)
    audit.add("zipped_runner_matches", hashlib.sha256(runner_bytes).hexdigest() == EXPECTED["runner_sha256"], hashlib.sha256(runner_bytes).hexdigest(), EXPECTED["runner_sha256"])
    audit.add("pinned_catboost", requirements == "catboost==1.2.10", requirements, "catboost==1.2.10")

    audit.add("result_complete", result.get("status") == "COMPLETE", result.get("status"), "COMPLETE")
    audit.add("result_candidate_status", result.get("candidate_status") == "PERFORMANCE_GATE_PASS", result.get("candidate_status"), "PERFORMANCE_GATE_PASS")
    audit.add("launch_complete", launch.get("status") == "COMPLETE" and launch.get("returncode") == 0, {"status": launch.get("status"), "returncode": launch.get("returncode")}, {"status": "COMPLETE", "returncode": 0})
    audit.add("t4_gpu", "T4" in str(launch.get("gpu", "")).upper(), launch.get("gpu"), "single T4")
    audit.add("verified_input_count", len(launch.get("verified_inputs", [])) == 8, len(launch.get("verified_inputs", [])), 8)
    command = launch.get("command", [])
    required_command_tokens = ["--targets", "2022,2023,2024", "--device", "gpu", "--no-cpu-fallback", "--resume"]
    audit.add("launch_command_contract", all(token in command for token in required_command_tokens), command, required_command_tokens)
    audit.add("no_smoke_command", "--smoke" not in command and "--max-rows-per-year" not in command, command, "no smoke/sample flags")

    expected_config = {
        "experiment_id": EXPERIMENT_ID,
        "requested_device": "gpu",
        "resolved_initial_device": "GPU",
        "cpu_fallback": False,
        "smoke": False,
        "max_rows_per_year": None,
        "iterations": 256,
        "depth": 6,
        "learning_rate": 0.025,
        "bootstrap_repeats": 10_000,
        "seeds": EXPECTED["seeds"],
        "scales": EXPECTED["scales"],
        "correction_clip": 0.12,
        "train_sha256": EXPECTED["train_sha256"],
        "runner_sha256": EXPECTED["runner_sha256"],
        "test_read": False,
    }
    for key, expected_value in expected_config.items():
        audit.add(f"config:{key}", config.get(key) == expected_value, config.get(key), expected_value)
    fingerprint_payload = {
        "experiment": EXPERIMENT_ID,
        "seeds": EXPECTED["seeds"],
        "scales": EXPECTED["scales"],
        "iterations": config["iterations"],
        "depth": config["depth"],
        "learning_rate": config["learning_rate"],
        "max_rows_per_year": config["max_rows_per_year"],
        "requested_device": config["requested_device"],
        "resolved_initial_device": config["resolved_initial_device"],
        "devices": config["devices"],
        "smoke": config["smoke"],
        "train_sha256": config["train_sha256"],
        "runner_sha256": config["runner_sha256"],
        "anchor": config["anchor_provenance"],
    }
    expected_fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True).encode()
    ).hexdigest()
    audit.add("config_fingerprint_recomputed", config["config_fingerprint"] == expected_fingerprint, config["config_fingerprint"], expected_fingerprint)
    audit.add("result_config_fingerprint", result["config_fingerprint"] == expected_fingerprint, result["config_fingerprint"], expected_fingerprint)
    audit.add("anchor_provenance_method", config["anchor_provenance"]["method"] == "load_audited_117a_oof", config["anchor_provenance"]["method"], "load_audited_117a_oof")
    audit.add("anchor_provenance_hash", config["anchor_provenance"]["sha256"] == EXPECTED["anchor_sha256"], config["anchor_provenance"]["sha256"], EXPECTED["anchor_sha256"])

    log_text = (RAW_DIR / "logs" / "ref4_126_20260831T170036Z.log").read_text(encoding="utf-8")
    audit.add("log_no_gpu_fallback", "GPU CatBoost failed" not in log_text and "CPU_FALLBACK" not in log_text, "fallback absent" if "CPU_FALLBACK" not in log_text else "fallback present", "fallback absent")
    audit.add("log_seed_fit_count", log_text.count("device=GPU completed") == 6, log_text.count("device=GPU completed"), 6)
    audit.add("log_completion", "status=PERFORMANCE_GATE_PASS" in log_text, "status=PERFORMANCE_GATE_PASS" in log_text, True)

    prediction_path = RAW_DIR / "results" / "oof_predictions.csv"
    metric_path = RAW_DIR / "results" / "strict_metrics.csv"
    scale_path = RAW_DIR / "results" / "strict_scale_search.csv"
    pred = pd.read_csv(prediction_path, low_memory=False)
    metrics = pd.read_csv(metric_path)
    scales = pd.read_csv(scale_path)
    anchor = pd.read_csv(
        ANCHOR_PATH,
        usecols=["row_id", "season", "game_type", "pitcher_id", "target", "p113a_strict"],
        low_memory=False,
    )
    raw_minimal = pd.read_csv(
        TRAIN_PATH,
        usecols=["row_id", "season", "game_type", "pitcher_id", "control_success"],
        low_memory=False,
    )
    raw_minimal = raw_minimal.loc[raw_minimal["season"].isin(EXPECTED["rows"])].reset_index(drop=True)

    expected_total = sum(EXPECTED["rows"].values())
    audit.add("prediction_rows", len(pred) == expected_total, len(pred), expected_total)
    audit.add("prediction_row_id_unique", pred["row_id"].astype(str).is_unique, int(pred["row_id"].astype(str).nunique()), expected_total)
    audit.add("anchor_rows", len(anchor) == expected_total, len(anchor), expected_total)
    audit.add("anchor_row_id_unique", anchor["row_id"].astype(str).is_unique, int(anchor["row_id"].astype(str).nunique()), expected_total)
    audit.add("train_row_id_unique", raw_minimal["row_id"].astype(str).is_unique, int(raw_minimal["row_id"].astype(str).nunique()), expected_total)
    actual_counts = {int(key): int(value) for key, value in pred.groupby("season").size().to_dict().items()}
    audit.add("season_row_counts", actual_counts == EXPECTED["rows"], actual_counts, EXPECTED["rows"])
    audit.add("prediction_anchor_row_order", np.array_equal(pred["row_id"].astype(str), anchor["row_id"].astype(str)), "exact" if np.array_equal(pred["row_id"].astype(str), anchor["row_id"].astype(str)) else "mismatch", "exact")
    audit.add("anchor_train_row_order", np.array_equal(anchor["row_id"].astype(str), raw_minimal["row_id"].astype(str)), "exact" if np.array_equal(anchor["row_id"].astype(str), raw_minimal["row_id"].astype(str)) else "mismatch", "exact")
    audit.add("target_anchor_exact", np.array_equal(pred["target"].to_numpy(float), anchor["target"].to_numpy(float)), "exact" if np.array_equal(pred["target"].to_numpy(float), anchor["target"].to_numpy(float)) else "mismatch", "exact")
    audit.add("target_train_exact", np.array_equal(anchor["target"].to_numpy(float), raw_minimal["control_success"].to_numpy(float)), "exact" if np.array_equal(anchor["target"].to_numpy(float), raw_minimal["control_success"].to_numpy(float)) else "mismatch", "exact")
    target = pred["target"].to_numpy(float)
    audit.add("target_finite_binary", np.isfinite(target).all() and set(np.unique(target)).issubset({0.0, 1.0}), sorted(np.unique(target).tolist()), [0.0, 1.0])
    for column in ("p113a_strict", "raw_r_correction", "selected_scale", "p126"):
        values = pred[column].to_numpy(float)
        audit.add(f"prediction_finite:{column}", np.isfinite(values).all(), int(np.isfinite(values).sum()), len(values))
    for column in ("p113a_strict", "p126"):
        values = pred[column].to_numpy(float)
        audit.add(f"prediction_range:{column}", bool(np.all((values >= 0.0) & (values <= 1.0))), [float(values.min()), float(values.max())], [0.0, 1.0])
    correction = pred["raw_r_correction"].to_numpy(float)
    audit.add("correction_clip", bool(np.max(np.abs(correction)) <= 0.12 + 1e-15), float(np.max(np.abs(correction))), 0.12)
    audit.add("anchor_prediction_exact", np.array_equal(pred["p113a_strict"].to_numpy(float), anchor["p113a_strict"].to_numpy(float)), "exact" if np.array_equal(pred["p113a_strict"].to_numpy(float), anchor["p113a_strict"].to_numpy(float)) else "mismatch", "exact")
    f_mask = pred["game_type"].astype(str).eq("F").to_numpy()
    audit.add("f_exact_identity", np.array_equal(pred.loc[f_mask, "p126"].to_numpy(float), pred.loc[f_mask, "p113a_strict"].to_numpy(float)), "exact" if np.array_equal(pred.loc[f_mask, "p126"].to_numpy(float), pred.loc[f_mask, "p113a_strict"].to_numpy(float)) else "mismatch", "exact")
    audit.add("f_raw_correction_zero", np.array_equal(pred.loc[f_mask, "raw_r_correction"].to_numpy(float), np.zeros(int(f_mask.sum()))), float(np.max(np.abs(pred.loc[f_mask, "raw_r_correction"].to_numpy(float)))), 0.0)

    metric_recalculated: list[dict[str, Any]] = []
    deltas: dict[int, float] = {}
    for year in EXPECTED["rows"]:
        year_mask = pred["season"].eq(year).to_numpy()
        for group in ("ALL", "R", "F"):
            mask = year_mask if group == "ALL" else year_mask & pred["game_type"].astype(str).eq(group).to_numpy()
            y = pred.loc[mask, "target"].to_numpy(float)
            base = pred.loc[mask, "p113a_strict"].to_numpy(float)
            candidate = pred.loc[mask, "p126"].to_numpy(float)
            row = {
                "year": year,
                "group": group,
                "rows": int(mask.sum()),
                "target_rate": float(y.mean()),
                "anchor_brier": brier(y, base),
                "candidate_brier": brier(y, candidate),
                "delta_brier": brier(y, candidate) - brier(y, base),
                "anchor_bss": bss(y, base),
                "candidate_bss": bss(y, candidate),
            }
            row["delta_bss"] = row["candidate_bss"] - row["anchor_bss"]
            metric_recalculated.append(row)
            reported_rows = metrics.loc[metrics["year"].eq(year) & metrics["group"].eq(group)]
            audit.add(f"metric_row_unique:{year}:{group}", len(reported_rows) == 1, len(reported_rows), 1)
            if len(reported_rows) == 1:
                reported = reported_rows.iloc[0]
                for key, value in row.items():
                    if key in ("year", "group"):
                        continue
                    tolerance = 1e-9 if "bss" in key else 5e-12
                    audit.add(
                        f"metric_match:{year}:{group}:{key}",
                        close(reported[key], value, tolerance=tolerance),
                        float(reported[key]),
                        value,
                    )
            if group == "ALL":
                deltas[year] = row["delta_brier"]

    audit.add("metrics_leaf_count", len(metric_recalculated) == 9, len(metric_recalculated), 9)
    weighted_delta = float(sum(EXPECTED["time_weights"][year] * deltas[year] for year in EXPECTED["rows"]))
    for year in EXPECTED["rows"]:
        audit.add(f"result_delta:{year}", close(result["delta_brier_by_year"][str(year)], deltas[year]), result["delta_brier_by_year"][str(year)], deltas[year])
    audit.add("result_time_weighted_delta", close(result["time_weighted_delta_brier"], weighted_delta), result["time_weighted_delta_brier"], weighted_delta)

    # Recalculate all strict scale-search rows from the OOF arrays.
    selection_map: dict[str, list[int]] = {
        "2022": [],
        "2023": [],
        "2024": [2023],
        "FUTURE_DEPLOYMENT": [2023, 2024],
    }
    expected_scale_rows: list[dict[str, Any]] = []
    chosen_scales: dict[str, float] = {}
    for applied_to, selection_years in selection_map.items():
        candidates: list[dict[str, Any]] = []
        for scale in EXPECTED["scales"]:
            row: dict[str, Any] = {
                "applied_to": applied_to,
                "selection_years": ",".join(map(str, selection_years)),
                "scale": scale,
            }
            if not selection_years:
                row.update(mean_delta_brier=0.0, worst_delta_brier=0.0, eligible=scale == 0.0)
            else:
                values: list[float] = []
                for selection_year in selection_years:
                    mask = pred["season"].eq(selection_year) & pred["game_type"].eq("R")
                    y = pred.loc[mask, "target"].to_numpy(float)
                    base = pred.loc[mask, "p113a_strict"].to_numpy(float)
                    raw_correction = pred.loc[mask, "raw_r_correction"].to_numpy(float)
                    candidate = np.clip(base + scale * raw_correction, 0.005, 0.995)
                    delta = brier(y, candidate) - brier(y, base)
                    row[f"delta_brier_{selection_year}"] = delta
                    values.append(delta)
                row["mean_delta_brier"] = float(np.mean(values))
                row["worst_delta_brier"] = float(np.max(values))
                row["objective"] = float(np.mean(values) + 0.35 * max(0.0, np.max(values)))
                row["eligible"] = True
            candidates.append(row)
            expected_scale_rows.append(row)
        chosen_scales[applied_to] = float(min(candidates, key=lambda item: (item.get("objective", 0.0), item["scale"]))["scale"])
    audit.add("scale_search_rows", len(scales) == len(expected_scale_rows), len(scales), len(expected_scale_rows))
    for expected_row in expected_scale_rows:
        matched = scales.loc[
            scales["applied_to"].astype(str).eq(expected_row["applied_to"])
            & np.isclose(scales["scale"].astype(float), expected_row["scale"], rtol=0, atol=1e-15)
        ]
        audit.add(f"scale_row_unique:{expected_row['applied_to']}:{expected_row['scale']}", len(matched) == 1, len(matched), 1)
        if len(matched) == 1:
            reported = matched.iloc[0]
            for key, value in expected_row.items():
                if key in ("applied_to", "selection_years", "eligible"):
                    continue
                audit.add(f"scale_match:{expected_row['applied_to']}:{expected_row['scale']}:{key}", close(reported[key], value), None if pd.isna(reported[key]) else float(reported[key]), value)
    audit.add("selected_scale_2022", close(result["applied_scales"]["2022"], chosen_scales["2022"]), result["applied_scales"]["2022"], chosen_scales["2022"])
    audit.add("selected_scale_2023", close(result["applied_scales"]["2023"], chosen_scales["2023"]), result["applied_scales"]["2023"], chosen_scales["2023"])
    audit.add("selected_scale_2024", close(result["applied_scales"]["2024"], chosen_scales["2024"]), result["applied_scales"]["2024"], chosen_scales["2024"])
    audit.add("future_scale_recommendation", close(result["future_deployment_scale_recommendation"], chosen_scales["FUTURE_DEPLOYMENT"]), result["future_deployment_scale_recommendation"], chosen_scales["FUTURE_DEPLOYMENT"])

    bootstrap = bootstrap_pitcher_2024(pred.loc[pred["season"].eq(2024)].reset_index(drop=True), 10_000)
    for key, value in bootstrap.items():
        audit.add(f"bootstrap:{key}", close(result["pitcher_cluster_bootstrap_2024"][key], value), result["pitcher_cluster_bootstrap_2024"][key], value)

    checkpoint_summary: dict[str, Any] = {}
    for year in EXPECTED["rows"]:
        fold_dir = RAW_DIR / "checkpoints" / str(year)
        metadata = load_json(fold_dir / "metadata.json")
        npz_path = fold_dir / "predictions.npz"
        audit.add(f"checkpoint_hash:{year}", sha256(npz_path) == metadata["prediction_sha256"], sha256(npz_path), metadata["prediction_sha256"])
        audit.add(f"checkpoint_config:{year}", metadata["config_fingerprint"] == expected_fingerprint, metadata["config_fingerprint"], expected_fingerprint)
        year_frame = pred.loc[pred["season"].eq(year)].reset_index(drop=True)
        audit.add(f"checkpoint_row_fingerprint:{year}", metadata["row_fingerprint"] == row_fingerprint(year_frame["row_id"]), metadata["row_fingerprint"], row_fingerprint(year_frame["row_id"]))
        with np.load(npz_path) as archive:
            checkpoint_prediction = archive["raw_correction"].astype(float)
        audit.add(f"checkpoint_rows:{year}", len(checkpoint_prediction) == EXPECTED["rows"][year], len(checkpoint_prediction), EXPECTED["rows"][year])
        audit.add(f"checkpoint_finite:{year}", np.isfinite(checkpoint_prediction).all(), int(np.isfinite(checkpoint_prediction).sum()), len(checkpoint_prediction))
        max_npz_difference = float(np.max(np.abs(checkpoint_prediction - year_frame["raw_r_correction"].to_numpy(float))))
        audit.add(f"checkpoint_oof_match:{year}", max_npz_difference <= 1e-7, max_npz_difference, "<=1e-7 (float32 checkpoint)")
        expected_fit_years = [] if year == 2022 else ([2022] if year == 2023 else [2022, 2023])
        audit.add(f"fit_years:{year}", metadata["fit_years"] == expected_fit_years, metadata["fit_years"], expected_fit_years)
        audit.add(f"validation_labels_unused:{year}", metadata["validation_labels_used_in_fit"] is False, metadata["validation_labels_used_in_fit"], False)
        expected_device = "NONE" if year == 2022 else "GPU"
        audit.add(f"checkpoint_device:{year}", metadata["device_used"] == expected_device, metadata["device_used"], expected_device)
        model_records: list[dict[str, Any]] = []
        if year > 2022:
            for seed in EXPECTED["seeds"]:
                model_path = fold_dir / "models" / f"r_residual_seed{seed}.cbm"
                model = CatBoostRegressor()
                load_error: str | None = None
                try:
                    model.load_model(str(model_path))
                except Exception as error:  # pragma: no cover - captured in report
                    load_error = repr(error)
                audit.add(f"model_load:{year}:{seed}", load_error is None, load_error, None)
                parameters = model.get_all_params() if load_error is None else {}
                audit.add(f"model_seed:{year}:{seed}", int(parameters.get("random_seed", -1)) == seed, parameters.get("random_seed"), seed)
                audit.add(f"model_depth:{year}:{seed}", int(parameters.get("depth", -1)) == 6, parameters.get("depth"), 6)
                audit.add(f"model_iterations:{year}:{seed}", int(model.tree_count_) == 256 if load_error is None else False, int(model.tree_count_) if load_error is None else None, 256)
                model_records.append({"seed": seed, "path": model_path.relative_to(ROOT).as_posix(), "bytes": model_path.stat().st_size, "sha256": sha256(model_path), "load_error": load_error})
        checkpoint_summary[str(year)] = {
            "fit_years": metadata["fit_years"],
            "device": metadata["device_used"],
            "rows": len(checkpoint_prediction),
            "npz_oof_max_abs_difference": max_npz_difference,
            "models": model_records,
        }

    # Independently reproduce saved-model predictions using the audited feature builder.
    runner = load_runner()
    raw_full = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig", low_memory=False)
    model_reproduction: dict[str, Any] = {}
    for year in (2023, 2024):
        rows = raw_full.loc[raw_full["season"].eq(year)].reset_index(drop=True)
        anchor_year = anchor.loc[anchor["season"].eq(year)].reset_index(drop=True)
        rows["p113a_strict"] = anchor_year["p113a_strict"].to_numpy(float)
        features = runner.build_features(rows, anchor_year["p113a_strict"].to_numpy(float), raw_full, year)
        audit.add(f"feature_columns:{year}", list(features.columns) == config["feature_columns"], list(features.columns), config["feature_columns"])
        regular = rows["game_type"].astype(str).eq("R").to_numpy()
        members: list[np.ndarray] = []
        validation_pool = Pool(
            features.loc[regular].reset_index(drop=True),
            cat_features=list(runner.CATEGORICAL),
        )
        for seed in EXPECTED["seeds"]:
            model = CatBoostRegressor()
            model.load_model(str(RAW_DIR / "checkpoints" / str(year) / "models" / f"r_residual_seed{seed}.cbm"))
            members.append(
                np.clip(
                    model.predict(validation_pool),
                    -0.12,
                    0.12,
                )
            )
        reproduced = np.mean(np.stack(members), axis=0)
        recorded = pred.loc[pred["season"].eq(year) & pred["game_type"].eq("R"), "raw_r_correction"].to_numpy(float)
        maximum = float(np.max(np.abs(reproduced - recorded)))
        audit.add(f"saved_model_oof_reproduction:{year}", maximum <= 1e-12, maximum, "<=1e-12")
        model_reproduction[str(year)] = {"rows": int(len(recorded)), "max_abs_difference": maximum}
        del rows, anchor_year, features, validation_pool, members, reproduced, recorded

    # The residual specialist is a valid probability expert on all corrected OOF rows.
    convex_checks: dict[str, Any] = {}
    for year in (2023, 2024):
        mask = pred["season"].eq(year) & pred["game_type"].eq("R")
        base = pred.loc[mask, "p113a_strict"].to_numpy(float)
        correction = pred.loc[mask, "raw_r_correction"].to_numpy(float)
        specialist = base + correction
        scale = float(result["applied_scales"][str(year)])
        convex = (1.0 - scale) * base + scale * specialist
        recorded = pred.loc[mask, "p126"].to_numpy(float)
        valid = bool(np.all((specialist >= 0.0) & (specialist <= 1.0)))
        maximum = float(np.max(np.abs(convex - recorded)))
        audit.add(f"specialist_probability_range:{year}", valid, [float(specialist.min()), float(specialist.max())], [0.0, 1.0])
        audit.add(f"convex_equivalence:{year}", maximum <= 2e-15, maximum, "<=2e-15")
        convex_checks[str(year)] = {
            "specialist_min": float(specialist.min()),
            "specialist_max": float(specialist.max()),
            "scale": scale,
            "max_abs_difference": maximum,
        }

    latest = pred.loc[pred["season"].eq(2024)].reset_index(drop=True)
    calibration = {
        "anchor_mean_prediction": float(latest["p113a_strict"].mean()),
        "candidate_mean_prediction": float(latest["p126"].mean()),
        "target_rate": float(latest["target"].mean()),
        "anchor_abs_calibration_gap": abs(float(latest["p113a_strict"].mean() - latest["target"].mean())),
        "candidate_abs_calibration_gap": abs(float(latest["p126"].mean() - latest["target"].mean())),
        "anchor_ece10": ece_equal_width(latest["target"].to_numpy(float), latest["p113a_strict"].to_numpy(float)),
        "candidate_ece10": ece_equal_width(latest["target"].to_numpy(float), latest["p126"].to_numpy(float)),
    }
    simultaneous_calibration_worsening = (
        calibration["candidate_abs_calibration_gap"] > calibration["anchor_abs_calibration_gap"]
        and calibration["candidate_ece10"] > calibration["anchor_ece10"]
    )
    audit.add("latest_calibration_not_simultaneously_worse", not simultaneous_calibration_worsening, simultaneous_calibration_worsening, False)

    internal_gates = {
        "f_exact_identity": bool(np.array_equal(latest.loc[latest["game_type"].eq("F"), "p126"].to_numpy(float), latest.loc[latest["game_type"].eq("F"), "p113a_strict"].to_numpy(float))),
        "latest_delta_brier_negative": deltas[2024] < 0.0,
        "time_weighted_delta_negative": weighted_delta < 0.0,
        "latest_bootstrap_ci_high_below_zero": bootstrap["ci_high"] < 0.0,
        "nonzero_strict_scale_selected": float(result["applied_scales"]["2024"]) > 0.0,
    }
    audit.add("internal_gate_record", internal_gates == result["gate_results"], internal_gates, result["gate_results"])
    audit.add("internal_gate_boolean", all(internal_gates.values()) == bool(result["performance_gate_pass"]), all(internal_gates.values()), result["performance_gate_pass"])
    worst_season_delta = max(deltas.values())
    external_gate = {
        "delta_2024_at_most_minus_0_0001": deltas[2024] <= -0.0001,
        "worst_season_at_most_plus_0_00005": worst_season_delta <= 0.00005,
        "time_weighted_improvement": weighted_delta < 0.0,
        "pitcher_bootstrap_ci_high_below_zero": bootstrap["ci_high"] < 0.0,
        "latest_calibration_not_simultaneously_worse": not simultaneous_calibration_worsening,
        "f_exact_identity": internal_gates["f_exact_identity"],
        "gpu_only": all(result["devices"][str(year)]["device_used"] == "GPU" for year in (2023, 2024)),
        "convex_equivalent_on_oof": all(item["max_abs_difference"] <= 2e-15 for item in convex_checks.values()),
    }
    audit.add("candidate_gate", all(external_gate.values()), external_gate, "all true")
    audit.add("candidate_count", manifest["candidate_count"] == 1, manifest["candidate_count"], 1)
    gate_checks_count = 1
    audit.add("gate_checks_count", gate_checks_count == manifest["candidate_count"], gate_checks_count, manifest["candidate_count"])
    audit.add("validation_only_no_production_fit", result["production_fit"] is False, result["production_fit"], False)
    audit.add("validation_only_no_zip", result["zip_created"] is False, result["zip_created"], False)
    audit.add("retrieved_no_zip", not any(path.suffix.lower() == ".zip" for path in RAW_DIR.rglob("*")), [path.relative_to(RAW_DIR).as_posix() for path in RAW_DIR.rglob("*.zip")], [])

    failures = audit.failures
    status = "AUDIT_VERIFIED_VALIDATION_ONLY" if not failures and all(external_gate.values()) else "AUDIT_FAIL"
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "scope": "strict_forward_validation_only",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "candidate_status": "VALIDATION_GATE_PASS" if all(external_gate.values()) else "NO_PROMOTION",
        "checked_count": len(audit.checks),
        "passed_count": len(audit.checks) - len(failures),
        "failed_count": len(failures),
        "failures": failures,
        "checks": audit.checks,
        "candidate_count": 1,
        "gate_checks_count": gate_checks_count,
        "metrics_recalculated": metric_recalculated,
        "delta_brier_by_year": {str(year): deltas[year] for year in EXPECTED["rows"]},
        "time_weighted_delta_brier": weighted_delta,
        "worst_season_delta_brier": worst_season_delta,
        "pitcher_cluster_bootstrap_2024": bootstrap,
        "calibration_2024": calibration,
        "internal_gates": internal_gates,
        "external_gate": external_gate,
        "selected_scales_recalculated": chosen_scales,
        "checkpoint_summary": checkpoint_summary,
        "saved_model_reproduction": model_reproduction,
        "convex_equivalence": convex_checks,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "catboost": catboost.__version__,
        },
        "unverified_in_scope": [],
        "out_of_scope": [
            "2025 production full-fit residual model",
            "113A production package integration",
            "test-row independence and permutation invariance",
            "submission ZIP runtime/size/security gates",
            "official leaderboard score",
        ],
    }
    atomic_json(REPORT_PATH, report)
    report_hash = sha256(REPORT_PATH)
    validator_hash = sha256(Path(__file__).resolve())
    attestation = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "scope": report["scope"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "manifest_path": MANIFEST_PATH.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256(MANIFEST_PATH),
        "validation_report_path": REPORT_PATH.relative_to(ROOT).as_posix(),
        "validation_report_sha256": report_hash,
        "validator_path": Path(__file__).resolve().relative_to(ROOT).as_posix(),
        "validator_sha256": validator_hash,
        "checked_count": report["checked_count"],
        "passed_count": report["passed_count"],
        "failed_count": report["failed_count"],
        "candidate_count": report["candidate_count"],
        "gate_checks_count": report["gate_checks_count"],
        "unverified_in_scope_count": len(report["unverified_in_scope"]),
        "candidate_status": report["candidate_status"],
        "production_approved": False,
        "submission_approved": False,
    }
    atomic_json(ATTESTATION_PATH, attestation)
    print(json.dumps(attestation, ensure_ascii=False, indent=2))
    return 0 if status == "AUDIT_VERIFIED_VALIDATION_ONLY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
