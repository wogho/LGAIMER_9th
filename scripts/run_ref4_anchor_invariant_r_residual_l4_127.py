#!/usr/bin/env python3
"""L4 strict-forward anchor-invariant R-only residual audit for REF4 127.

The residual target remains y - p113a_strict, but the anchor probability is
excluded from the model input. This isolates one change intended to remove
the strict-anchor versus production-anchor geometry dependency found after
126. F predictions remain bit-exact. Candidate probabilities are a convex
blend of the frozen anchor and a bounded auxiliary probability.

This is validation-only: it never reads test.csv, fits production models, or
creates a submission ZIP. Full runs require an NVIDIA L4 and forbid CPU
fallback.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from catboost import CatBoostError, CatBoostRegressor


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT / "data/train.csv"
DEFAULT_ANCHOR = ROOT / "model/REF4-113A-V66-NESTED-117A/oof_predictions.csv"
DEFAULT_P109C_ANCHOR = ROOT / "model/REF4-110-ORIGINAL-R2/expert_oof.csv"
DEFAULT_OUTPUT = ROOT / "model/REF4-ANCHOR-INVARIANT-R-RESIDUAL-L4-127"

EXPERIMENT_ID = "REF4-ANCHOR-INVARIANT-R-RESIDUAL-L4-127"
TARGET = "control_success"
YEARS = (2022, 2023, 2024)
SEEDS = (17, 42, 777)
SCALES = (0.0, 0.025, 0.05, 0.075)
CLIP = (0.005, 0.995)
TIME_WEIGHTS = {2022: 0.2, 2023: 0.3, 2024: 0.5}
CORRECTION_CLIP = 0.12

CATEGORICAL = (
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "game_dayofweek",
    "count_state",
    "hand_matchup",
    "prior_game_type",
    "league_transition",
    "team_type",
)
DROP_COLUMNS = ("row_id", TARGET, "pitcher_id", "batter_id")


@dataclass(frozen=True)
class Config:
    train: Path
    anchor: Path
    p109c_anchor: Path
    output: Path
    checkpoint_dir: Path | None
    requested_device: str
    devices: str
    threads: int
    iterations: int
    depth: int
    learning_rate: float
    bootstrap_repeats: int
    resume: bool
    cpu_fallback: bool
    max_rows_per_year: int | None
    smoke: bool


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def atomic_write_json(path: Path, value: Any) -> None:
    """Durably publish JSON only after its complete temporary file is synced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    payload = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def row_fingerprint(values: pd.Series) -> str:
    hashes = pd.util.hash_pandas_object(values.astype(str), index=False).to_numpy(np.uint64)
    return hashlib.sha256(hashes.tobytes()).hexdigest()


def array_fingerprint(values: np.ndarray) -> str:
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode())
    digest.update(json.dumps(array.shape).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


def frame_fingerprint(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(list(frame.columns), ensure_ascii=False).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy(np.uint64).tobytes())
    return digest.hexdigest()


def brier(target: np.ndarray, prediction: np.ndarray) -> float:
    return float(np.mean(np.square(np.asarray(prediction, float) - np.asarray(target, float))))


def bss(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, float)
    reference = float(target.mean() * (1.0 - target.mean()))
    if reference <= 0.0:
        return float("nan")
    return float(100_000.0 * (1.0 - brier(target, prediction) / reference))


def gpu_name() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def gpu_visible() -> bool:
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def resolve_device(requested: str) -> str:
    if requested == "cpu":
        return "CPU"
    if requested == "gpu":
        return "GPU"
    return "GPU" if gpu_visible() else "CPU"


def strict_sample(frame: pd.DataFrame, max_rows_per_year: int | None) -> pd.DataFrame:
    """Deterministic R/F-stratified sampling used only for smoke tests."""
    if max_rows_per_year is None:
        return frame.reset_index(drop=True)
    parts: list[pd.DataFrame] = []
    for year in sorted(frame["season"].unique()):
        season = frame.loc[frame["season"].eq(year)]
        groups = tuple(sorted(season["game_type"].astype(str).unique()))
        quota = max(1, max_rows_per_year // max(1, len(groups)))
        chosen = pd.concat(
            [season.loc[season["game_type"].astype(str).eq(group)].head(quota) for group in groups]
        ).sort_index()
        if len(chosen) < max_rows_per_year:
            remainder = season.loc[~season.index.isin(chosen.index)].head(max_rows_per_year - len(chosen))
            chosen = pd.concat([chosen, remainder]).sort_index()
        parts.append(chosen.head(max_rows_per_year))
    return pd.concat(parts, ignore_index=True)


def validate_anchor(raw: pd.DataFrame, anchor: pd.DataFrame) -> pd.DataFrame:
    required = {"row_id", "season", "game_type", "pitcher_id", "target", "p113a_strict"}
    missing = sorted(required - set(anchor.columns))
    if missing:
        raise ValueError(f"strict anchor missing columns: {missing}")
    expected = raw.loc[raw["season"].isin(YEARS), ["row_id", "season", TARGET]].reset_index(drop=True)
    anchor = anchor.loc[anchor["season"].isin(YEARS)].reset_index(drop=True)
    if len(expected) != len(anchor):
        raise ValueError(f"strict anchor length mismatch: train={len(expected)}, anchor={len(anchor)}")
    if not np.array_equal(expected["row_id"].astype(str), anchor["row_id"].astype(str)):
        raise ValueError("strict anchor row_id order does not match train.csv")
    if not np.array_equal(expected["season"].to_numpy(int), anchor["season"].to_numpy(int)):
        raise ValueError("strict anchor season order does not match train.csv")
    if not np.array_equal(expected[TARGET].to_numpy(float), anchor["target"].to_numpy(float)):
        raise ValueError("strict anchor target does not match train.csv")
    prediction = anchor["p113a_strict"].to_numpy(float)
    if not np.isfinite(prediction).all() or np.any((prediction <= 0.0) | (prediction >= 1.0)):
        raise ValueError("strict anchor predictions are not finite probabilities")
    return anchor


def rebuild_strict_anchor(raw: pd.DataFrame, p109c_path: Path) -> pd.DataFrame:
    """Reuse the audited 117A fold builder to reconstruct strict 113A."""
    if not p109c_path.exists():
        raise FileNotFoundError(
            f"neither strict 113A anchor nor p109c OOF exists: {p109c_path}"
        )
    scripts_dir = ROOT / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from run_ref4_113a_v66_nested_117a import add_context, fit_disjoint_fold

    source = pd.read_csv(
        p109c_path,
        usecols=["row_id", "season", "game_type", "pitcher_id", "target", "p109c"],
        low_memory=False,
    )
    local_raw = add_context(raw.copy())
    expected = local_raw.loc[local_raw["season"].isin(YEARS)].reset_index(drop=True)
    if not np.array_equal(expected["row_id"].astype(str), source["row_id"].astype(str)):
        raise ValueError("p109c OOF row order does not match train.csv")
    if not np.array_equal(expected[TARGET].to_numpy(float), source["target"].to_numpy(float)):
        raise ValueError("p109c OOF target does not match train.csv")

    parts: list[pd.DataFrame] = []
    for year in YEARS:
        history = local_raw.loc[local_raw["season"].lt(year)].reset_index(drop=True)
        rows = local_raw.loc[local_raw["season"].eq(year)].reset_index(drop=True)
        base = source.loc[source["season"].eq(year)].reset_index(drop=True)
        correction = fit_disjoint_fold(history, rows)
        regular = rows["game_type"].astype(str).eq("R").to_numpy()
        prediction = base["p109c"].to_numpy(float).copy()
        prediction[regular] += 0.035 * correction[regular]
        prediction = np.clip(prediction, *CLIP)
        parts.append(
            pd.DataFrame(
                {
                    "row_id": rows["row_id"].astype(str),
                    "season": year,
                    "game_type": rows["game_type"].astype(str),
                    "pitcher_id": rows["pitcher_id"].astype(str),
                    "target": rows[TARGET].to_numpy(float),
                    "p113a_strict": prediction,
                }
            )
        )
        del history, rows, base, correction
        gc.collect()
    return pd.concat(parts, ignore_index=True)


def load_or_rebuild_anchor(raw: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    if config.anchor.exists():
        log(f"loading audited strict 113A anchor: {config.anchor}")
        anchor = pd.read_csv(
            config.anchor,
            usecols=["row_id", "season", "game_type", "pitcher_id", "target", "p113a_strict"],
            low_memory=False,
        )
        provenance = {"method": "load_audited_117a_oof", "path": str(config.anchor), "sha256": sha256(config.anchor)}
    else:
        log("audited p113a_strict file absent; rebuilding from strict p109c OOF")
        anchor = rebuild_strict_anchor(raw, config.p109c_anchor)
        provenance = {
            "method": "rebuild_p109c_plus_fold_disjoint_eb",
            "path": str(config.p109c_anchor),
            "sha256": sha256(config.p109c_anchor),
        }
    return validate_anchor(raw, anchor), provenance


def prior_game_type_table(raw: pd.DataFrame, year: int) -> pd.Series:
    earlier = raw.loc[raw["season"].lt(year)]
    counts = (
        earlier.groupby(["pitcher_id", "season", "game_type"], sort=False, observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    if counts.empty:
        return pd.Series(dtype=object)
    dominant = counts.sort_values("n").groupby(
        ["pitcher_id", "season"], sort=False, observed=True
    ).tail(1)
    latest = dominant.sort_values("season").groupby("pitcher_id", sort=False, observed=True).tail(1)
    return latest.set_index(latest["pitcher_id"].astype(str))["game_type"]


def build_features(
    rows: pd.DataFrame,
    base_prediction: np.ndarray,
    raw: pd.DataFrame,
    year: int,
) -> pd.DataFrame:
    """Build a row-local deployable feature surface using only pre-year history."""
    features = rows.drop(columns=list(DROP_COLUMNS), errors="ignore").copy()
    pitcher = rows["pitcher_id"].astype(str)
    prior = pitcher.map(prior_game_type_table(raw, year)).astype("object").fillna("NEW").astype(str)
    current = rows["game_type"].astype(str)
    features["prior_game_type"] = prior
    features["league_transition"] = prior + ">" + current
    features["team_type"] = rows["pitcher_team_id"].astype(str) + "|" + current
    features["count_state"] = rows["balls_before"].astype(str) + "-" + rows["strikes_before"].astype(str)
    features["hand_matchup"] = rows["pitcher_hand"].astype(str) + "-" + rows["batter_hand"].astype(str)
    # 127 single-variable isolation: the anchor probability is not a feature.
    # The argument remains only to keep the fold builder interface stable.
    del base_prediction
    features["log_pitcher_n"] = np.log1p(
        pd.to_numeric(rows["asof_pitcher_n"], errors="coerce").fillna(0).clip(lower=0)
    ).astype(np.float32)
    features["log_batter_n"] = np.log1p(
        pd.to_numeric(rows["asof_batter_n"], errors="coerce").fillna(0).clip(lower=0)
    ).astype(np.float32)
    features["recent_1_minus_5"] = (
        pd.to_numeric(rows["asof_pitcher_prev1_game_success_rate"], errors="coerce")
        - pd.to_numeric(rows["asof_pitcher_prev5_game_success_rate"], errors="coerce")
    ).astype(np.float32)
    features["middle_1_minus_5"] = (
        pd.to_numeric(rows["asof_pitcher_prev1_game_middle_rate"], errors="coerce")
        - pd.to_numeric(rows["asof_pitcher_prev5_game_middle_rate"], errors="coerce")
    ).astype(np.float32)
    for column in CATEGORICAL:
        if column not in features:
            raise ValueError(f"required categorical feature is missing: {column}")
        features[column] = features[column].astype("string").fillna("__MISSING__").astype(str)
    return features


def catboost_parameters(config: Config, seed: int, device: str) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "iterations": config.iterations,
        "depth": config.depth,
        "learning_rate": config.learning_rate,
        "loss_function": "RMSE",
        "l2_leaf_reg": 100.0,
        "random_strength": 0.30,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.80,
        "random_seed": seed,
        "task_type": device,
        "allow_writing_files": False,
        "verbose": False,
    }
    if device == "GPU":
        parameters.update(devices=config.devices, border_count=32, gpu_ram_part=0.85)
    else:
        parameters["thread_count"] = config.threads
    return parameters


def train_members(
    train_x: pd.DataFrame,
    train_target: np.ndarray,
    valid_x: pd.DataFrame,
    model_dir: Path,
    config: Config,
    requested_device: str,
) -> tuple[np.ndarray, str, list[str]]:
    def run(device: str) -> tuple[np.ndarray, list[str]]:
        members: list[np.ndarray] = []
        paths: list[str] = []
        model_dir.mkdir(parents=True, exist_ok=True)
        for seed in SEEDS:
            started = time.perf_counter()
            model = CatBoostRegressor(**catboost_parameters(config, seed, device))
            model.fit(train_x, train_target, cat_features=list(CATEGORICAL))
            prediction = np.clip(model.predict(valid_x), -CORRECTION_CLIP, CORRECTION_CLIP)
            members.append(np.asarray(prediction, float))
            path = model_dir / f"r_residual_seed{seed}.cbm"
            model.save_model(str(path))
            paths.append(str(path))
            log(f"seed={seed} device={device} completed in {time.perf_counter() - started:.1f}s")
        return np.mean(np.stack(members), axis=0), paths

    try:
        prediction, paths = run(requested_device)
        return prediction, requested_device, paths
    except CatBoostError as error:
        if requested_device != "GPU" or not config.cpu_fallback:
            raise
        log(f"GPU CatBoost failed; restarting fold on CPU: {error}")
        prediction, paths = run("CPU")
        return prediction, "CPU_FALLBACK", paths


def config_fingerprint(
    config: Config,
    anchor_provenance: dict[str, Any],
    resolved_device: str,
    train_sha256: str,
    runner_sha256: str,
) -> str:
    payload = {
        "experiment": EXPERIMENT_ID,
        "seeds": SEEDS,
        "scales": SCALES,
        "iterations": config.iterations,
        "depth": config.depth,
        "learning_rate": config.learning_rate,
        "max_rows_per_year": config.max_rows_per_year,
        "requested_device": config.requested_device,
        "resolved_initial_device": resolved_device,
        "devices": config.devices,
        "threads": config.threads,
        "smoke": config.smoke,
        "train_sha256": train_sha256,
        "runner_sha256": runner_sha256,
        "anchor": anchor_provenance,
        "feature_contract": "anchor_probability_excluded",
        "combination": "convex_blend_with_bounded_auxiliary",
        "categorical": CATEGORICAL,
        "correction_clip": CORRECTION_CLIP,
        "probability_clip": CLIP,
        "bootstrap_repeats": config.bootstrap_repeats,
        "cpu_fallback": config.cpu_fallback,
        "runtime": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def checkpoint_paths(config: Config, year: int) -> tuple[Path, Path, Path]:
    checkpoint_root = config.checkpoint_dir or (config.output / "checkpoints")
    root = checkpoint_root / str(year)
    return root / "predictions.npz", root / "metadata.json", root / "models"


def load_checkpoint(
    config: Config,
    year: int,
    fingerprint: str,
    expected_rows: pd.Series,
) -> tuple[np.ndarray, dict[str, Any]] | None:
    prediction_path, metadata_path, _ = checkpoint_paths(config, year)
    if not config.resume or not prediction_path.exists() or not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("config_fingerprint") != fingerprint:
        log(f"checkpoint {year} ignored: config fingerprint changed")
        return None
    if metadata.get("row_fingerprint") != row_fingerprint(expected_rows):
        log(f"checkpoint {year} ignored: row order changed")
        return None
    if metadata.get("prediction_sha256") != sha256(prediction_path):
        log(f"checkpoint {year} ignored: prediction SHA changed")
        return None
    for record in metadata.get("model_files", []):
        model_path = Path(record["path"])
        if (
            not model_path.exists()
            or model_path.stat().st_size != record.get("size")
            or sha256(model_path) != record.get("sha256")
        ):
            log(f"checkpoint {year} ignored: model file changed: {model_path}")
            return None
    with np.load(prediction_path) as archive:
        correction = archive["raw_correction"].astype(float)
    if len(correction) != len(expected_rows) or not np.isfinite(correction).all():
        log(f"checkpoint {year} ignored: invalid correction array")
        return None
    log(f"resumed raw correction fold {year}: {prediction_path}")
    return correction, metadata


def save_checkpoint(
    config: Config,
    year: int,
    correction: np.ndarray,
    rows: pd.DataFrame,
    fingerprint: str,
    metadata: dict[str, Any],
) -> None:
    prediction_path, metadata_path, _ = checkpoint_paths(config, year)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_prediction = prediction_path.with_name(prediction_path.name + ".tmp")
    with temporary_prediction.open("wb") as stream:
        np.savez_compressed(stream, raw_correction=np.asarray(correction, np.float32))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary_prediction, prediction_path)
    atomic_write_json(
        metadata_path,
        {
            **metadata,
            "year": year,
            "rows": int(len(rows)),
            "row_fingerprint": row_fingerprint(rows["row_id"]),
            "config_fingerprint": fingerprint,
            "prediction_sha256": sha256(prediction_path),
        },
    )


def scale_grid(
    folds: dict[int, dict[str, Any]],
    selection_years: Iterable[int],
    applied_to: int | str,
) -> tuple[float, list[dict[str, Any]]]:
    years = tuple(year for year in selection_years if year in folds and folds[year]["has_correction"])
    if not years:
        return 0.0, [
            {
                "applied_to": applied_to,
                "selection_years": "",
                "scale": scale,
                "mean_delta_brier": 0.0,
                "worst_delta_brier": 0.0,
                "eligible": scale == 0.0,
            }
            for scale in SCALES
        ]
    records: list[dict[str, Any]] = []
    for scale in SCALES:
        deltas: list[float] = []
        record: dict[str, Any] = {
            "applied_to": applied_to,
            "selection_years": ",".join(map(str, years)),
            "scale": scale,
        }
        for year in years:
            fold = folds[year]
            r = fold["r_mask"]
            auxiliary = np.clip(
                fold["base"][r] + fold["raw_correction"][r], *CLIP
            )
            candidate = (1.0 - scale) * fold["base"][r] + scale * auxiliary
            delta = brier(fold["target"][r], candidate) - brier(
                fold["target"][r], fold["base"][r]
            )
            record[f"delta_brier_{year}"] = delta
            deltas.append(delta)
        record["mean_delta_brier"] = float(np.mean(deltas))
        record["worst_delta_brier"] = float(np.max(deltas))
        # Mean performance is primary; worst-year damage breaks ties.  A lower
        # scale wins exact ties, which makes a zero-effect search conservative.
        record["objective"] = float(np.mean(deltas) + 0.35 * max(0.0, np.max(deltas)))
        record["eligible"] = True
        records.append(record)
    best = min(records, key=lambda row: (row["objective"], row["scale"]))
    return float(best["scale"]), records


def metric_rows(
    year: int,
    rows: pd.DataFrame,
    target: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
) -> list[dict[str, Any]]:
    game_type = rows["game_type"].astype(str).to_numpy()
    output: list[dict[str, Any]] = []
    for group, mask in (
        ("ALL", np.ones(len(rows), dtype=bool)),
        ("R", game_type == "R"),
        ("F", game_type == "F"),
    ):
        if not mask.any():
            continue
        base_brier = brier(target[mask], base[mask])
        candidate_brier = brier(target[mask], candidate[mask])
        output.append(
            {
                "year": year,
                "group": group,
                "rows": int(mask.sum()),
                "target_rate": float(target[mask].mean()),
                "anchor_brier": base_brier,
                "candidate_brier": candidate_brier,
                "delta_brier": candidate_brier - base_brier,
                "anchor_bss": bss(target[mask], base[mask]),
                "candidate_bss": bss(target[mask], candidate[mask]),
                "delta_bss": bss(target[mask], candidate[mask]) - bss(target[mask], base[mask]),
            }
        )
    return output


def pitcher_bootstrap(
    target: np.ndarray,
    base: np.ndarray,
    candidate: np.ndarray,
    pitcher: np.ndarray,
    repeats: int,
) -> dict[str, Any]:
    row_delta = np.square(candidate - target) - np.square(base - target)
    grouped = pd.DataFrame({"pitcher": pitcher.astype(str), "delta": row_delta}).groupby(
        "pitcher", sort=False
    )["delta"].agg(["sum", "size"])
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


def run(config: Config) -> dict[str, Any]:
    started = time.perf_counter()
    config.output.mkdir(parents=True, exist_ok=True)
    log(f"loading train: {config.train}")
    raw = pd.read_csv(config.train, encoding="utf-8-sig", low_memory=False)
    required = {
        "row_id", "season", "game_type", "pitcher_id", "batter_id", TARGET,
        "pitcher_team_id", "batter_team_id", "pitcher_hand", "batter_hand",
        "balls_before", "strikes_before", "asof_pitcher_n", "asof_batter_n",
        "asof_pitcher_prev1_game_success_rate", "asof_pitcher_prev5_game_success_rate",
        "asof_pitcher_prev1_game_middle_rate", "asof_pitcher_prev5_game_middle_rate",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"train.csv missing required columns: {missing}")
    if raw["row_id"].astype(str).duplicated().any():
        raise ValueError("train.csv row_id is not unique")

    anchor, anchor_provenance = load_or_rebuild_anchor(raw, config)
    train_sha256 = sha256(config.train)
    runner_sha256 = sha256(Path(__file__).resolve())
    requested_device = resolve_device(config.requested_device)
    fingerprint = config_fingerprint(
        config,
        anchor_provenance,
        requested_device,
        train_sha256,
        runner_sha256,
    )
    run_config = {
        **{
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "experiment_id": EXPERIMENT_ID,
        "seeds": list(SEEDS),
        "scales": list(SCALES),
        "correction_clip": CORRECTION_CLIP,
        "resolved_initial_device": requested_device,
        "anchor_provenance": anchor_provenance,
        "train_sha256": train_sha256,
        "runner_sha256": runner_sha256,
        "config_fingerprint": fingerprint,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "test_read": False,
    }
    write_json(config.output / "run_config.json", run_config)

    validation = raw.loc[raw["season"].isin(YEARS)].reset_index(drop=True)
    validation["p113a_strict"] = anchor["p113a_strict"].to_numpy(float)
    validation = strict_sample(validation, config.max_rows_per_year)
    anchor_index = anchor.set_index(anchor["row_id"].astype(str))["p113a_strict"]
    validation["p113a_strict"] = validation["row_id"].astype(str).map(anchor_index).to_numpy(float)

    fold_rows = {
        year: validation.loc[validation["season"].eq(year)].reset_index(drop=True)
        for year in YEARS
    }
    features: dict[int, pd.DataFrame] = {}
    for year in YEARS:
        rows = fold_rows[year]
        features[year] = build_features(rows, rows["p113a_strict"].to_numpy(float), raw, year)
        log(f"features {year}: rows={len(rows):,}, columns={features[year].shape[1]}")
    run_config["feature_columns"] = list(features[min(YEARS)].columns)
    run_config["categorical_columns"] = list(CATEGORICAL)
    write_json(config.output / "run_config.json", run_config)

    folds: dict[int, dict[str, Any]] = {}
    device_records: dict[str, Any] = {}
    for year in YEARS:
        rows = fold_rows[year]
        base = rows["p113a_strict"].to_numpy(float)
        target = rows[TARGET].to_numpy(float)
        r_mask = rows["game_type"].astype(str).eq("R").to_numpy()
        if year == min(YEARS):
            correction = np.zeros(len(rows), dtype=float)
            metadata = {
                "warmup": True,
                "fit_years": [],
                "device_used": "NONE",
                "validation_labels_used_in_fit": False,
            }
            save_checkpoint(config, year, correction, rows, fingerprint, metadata)
        else:
            resumed = load_checkpoint(config, year, fingerprint, rows["row_id"])
            if resumed is not None:
                correction, metadata = resumed
            else:
                fit_years = tuple(previous for previous in YEARS if previous < year)
                fit_x_parts: list[pd.DataFrame] = []
                fit_target_parts: list[np.ndarray] = []
                fit_row_parts: list[pd.Series] = []
                for previous in fit_years:
                    previous_rows = fold_rows[previous]
                    previous_r = previous_rows["game_type"].astype(str).eq("R").to_numpy()
                    fit_x_parts.append(features[previous].loc[previous_r].reset_index(drop=True))
                    fit_target_parts.append(
                        previous_rows.loc[previous_r, TARGET].to_numpy(float)
                        - previous_rows.loc[previous_r, "p113a_strict"].to_numpy(float)
                    )
                    fit_row_parts.append(
                        previous_rows.loc[previous_r, "row_id"].astype(str).reset_index(drop=True)
                    )
                fit_x = pd.concat(fit_x_parts, ignore_index=True)
                fit_target = np.concatenate(fit_target_parts)
                fit_rows = pd.concat(fit_row_parts, ignore_index=True)
                correction = np.zeros(len(rows), dtype=float)
                prediction_path, _, model_dir = checkpoint_paths(config, year)
                del prediction_path
                r_prediction, device_used, model_paths = train_members(
                    fit_x,
                    fit_target,
                    features[year].loc[r_mask].reset_index(drop=True),
                    model_dir,
                    config,
                    requested_device,
                )
                correction[r_mask] = r_prediction
                metadata = {
                    "warmup": False,
                    "fit_years": list(fit_years),
                    "fit_rows_r": int(len(fit_x)),
                    "valid_rows_r": int(r_mask.sum()),
                    "device_used": device_used,
                    "gpu_name": gpu_name(),
                    "model_paths": model_paths,
                    "model_files": [
                        {
                            "path": path,
                            "size": Path(path).stat().st_size,
                            "sha256": sha256(Path(path)),
                        }
                        for path in model_paths
                    ],
                    "fit_row_fingerprint": row_fingerprint(fit_rows),
                    "fit_target_fingerprint": array_fingerprint(fit_target),
                    "fit_feature_fingerprint": frame_fingerprint(fit_x),
                    "feature_columns": list(fit_x.columns),
                    "validation_labels_used_in_fit": False,
                    "target_definition": "control_success - p113a_strict on prior R OOF rows",
                }
                save_checkpoint(config, year, correction, rows, fingerprint, metadata)
                del fit_x, fit_target, fit_rows
                gc.collect()
            device_records[str(year)] = metadata
        folds[year] = {
            "rows": rows,
            "base": base,
            "target": target,
            "r_mask": r_mask,
            "raw_correction": correction,
            "has_correction": year > min(YEARS),
        }

    scale_records: list[dict[str, Any]] = []
    applied_scales: dict[int, float] = {}
    for year in YEARS:
        scale, records = scale_grid(folds, range(min(YEARS), year), year)
        applied_scales[year] = scale
        scale_records.extend(records)
        log(f"scale for {year}: {scale:.3f} selected on {[y for y in YEARS if y < year and folds[y]['has_correction']]}")
    final_scale, final_records = scale_grid(folds, YEARS, "FUTURE_DEPLOYMENT")
    scale_records.extend(final_records)

    prediction_parts: list[pd.DataFrame] = []
    metrics: list[dict[str, Any]] = []
    for year in YEARS:
        fold = folds[year]
        scale = applied_scales[year]
        candidate = fold["base"].copy()
        r_mask = fold["r_mask"]
        auxiliary = np.clip(
            fold["base"][r_mask] + fold["raw_correction"][r_mask], *CLIP
        )
        candidate[r_mask] = (1.0 - scale) * fold["base"][r_mask] + scale * auxiliary
        f_mask = ~r_mask
        if not np.array_equal(candidate[f_mask], fold["base"][f_mask]):
            raise AssertionError(f"F exact identity failed in {year}")
        metrics.extend(
            metric_rows(year, fold["rows"], fold["target"], fold["base"], candidate)
        )
        prediction_parts.append(
            pd.DataFrame(
                {
                    "row_id": fold["rows"]["row_id"].astype(str),
                    "season": year,
                    "game_type": fold["rows"]["game_type"].astype(str),
                    "pitcher_id": fold["rows"]["pitcher_id"].astype(str),
                    "target": fold["target"],
                    "p113a_strict": fold["base"],
                    "raw_r_correction": fold["raw_correction"],
                    "selected_scale": scale,
                    "p127": candidate,
                }
            )
        )
        fold["candidate"] = candidate

    predictions = pd.concat(prediction_parts, ignore_index=True)
    metrics_frame = pd.DataFrame(metrics)
    scales_frame = pd.DataFrame(scale_records)
    predictions.to_csv(config.output / "oof_predictions.csv", index=False)
    metrics_frame.to_csv(config.output / "strict_metrics.csv", index=False)
    scales_frame.to_csv(config.output / "strict_scale_search.csv", index=False)

    latest = folds[max(YEARS)]
    bootstrap = pitcher_bootstrap(
        latest["target"],
        latest["base"],
        latest["candidate"],
        latest["rows"]["pitcher_id"].astype(str).to_numpy(),
        config.bootstrap_repeats,
    )
    all_deltas = {
        year: float(
            metrics_frame.loc[
                metrics_frame["year"].eq(year) & metrics_frame["group"].eq("ALL"),
                "delta_brier",
            ].iloc[0]
        )
        for year in YEARS
    }
    weighted_delta = float(sum(TIME_WEIGHTS[year] * all_deltas[year] for year in YEARS))
    f_exact = bool(
        np.array_equal(
            predictions.loc[predictions["game_type"].eq("F"), "p127"].to_numpy(float),
            predictions.loc[predictions["game_type"].eq("F"), "p113a_strict"].to_numpy(float),
        )
    )
    gates = {
        "f_exact_identity": f_exact,
        "latest_delta_brier_at_most_minus_0_00010": all_deltas[max(YEARS)] <= -0.00010,
        "time_weighted_delta_negative": weighted_delta < 0.0,
        "latest_bootstrap_ci_high_below_zero": bootstrap["ci_high"] < 0.0,
        "nonzero_strict_scale_selected": applied_scales[max(YEARS)] > 0.0,
        "base_prediction_feature_absent": all(
            "base_prediction" not in frame.columns for frame in features.values()
        ),
        "all_trained_folds_used_gpu": all(
            record.get("warmup") or record.get("device_used") == "GPU"
            for record in device_records.values()
        ),
    }
    performance_pass = bool(all(gates.values()))
    result = {
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE",
        "candidate_status": "PERFORMANCE_GATE_PASS" if performance_pass else "NO_PROMOTION",
        "strict_rules": {
            "target_year_labels_used_in_fit": False,
            "scale_selected_from_strictly_prior_correction_folds": True,
            "r_only_correction": True,
            "base_prediction_used_as_feature": False,
            "convex_blend": True,
            "l4_required": True,
            "f_exact_identity": f_exact,
            "test_read": False,
        },
        "rows": int(len(predictions)),
        "applied_scales": {str(year): applied_scales[year] for year in YEARS},
        "future_deployment_scale_recommendation": final_scale,
        "delta_brier_by_year": {str(year): all_deltas[year] for year in YEARS},
        "time_weighted_delta_brier": weighted_delta,
        "pitcher_cluster_bootstrap_2024": bootstrap,
        "gate_results": gates,
        "performance_gate_pass": performance_pass,
        "devices": device_records,
        "anchor_provenance": anchor_provenance,
        "config_fingerprint": fingerprint,
        "runtime_seconds": time.perf_counter() - started,
        "artifacts": {
            "oof_predictions": str(config.output / "oof_predictions.csv"),
            "strict_metrics": str(config.output / "strict_metrics.csv"),
            "strict_scale_search": str(config.output / "strict_scale_search.csv"),
            "checkpoints": str(config.checkpoint_dir or (config.output / "checkpoints")),
        },
        "production_fit": False,
        "zip_created": False,
    }
    write_json(config.output / "result.json", result)
    log(f"complete in {result['runtime_seconds'] / 60.0:.1f} min; status={result['candidate_status']}")
    return result


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_TRAIN.parent)
    parser.add_argument("--train", type=Path)
    parser.add_argument("--anchor", "--anchor-oof", dest="anchor", type=Path, default=DEFAULT_ANCHOR)
    parser.add_argument("--p109c-anchor", type=Path, default=DEFAULT_P109C_ANCHOR)
    parser.add_argument("--output", "--output-dir", dest="output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--targets", default="2022,2023,2024")
    parser.add_argument("--device", choices=("gpu",), default="gpu")
    parser.add_argument("--devices", default="0")
    parser.add_argument("--threads", type=int, default=-1)
    parser.add_argument("--iterations", type=int, default=256)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--learning-rate", type=float, default=0.025)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu-fallback", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--max-rows-per-year", type=int)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = arguments()
    train_path = args.train or (args.data_dir / "train.csv")
    targets = tuple(int(value.strip()) for value in args.targets.split(",") if value.strip())
    if targets != YEARS:
        raise ValueError(f"--targets must be exactly 2022,2023,2024, received {targets}")
    if not train_path.exists():
        raise FileNotFoundError(train_path)
    detected_gpu = gpu_name()
    if "L4" not in detected_gpu.upper():
        raise RuntimeError(f"NVIDIA L4 is required, detected={detected_gpu!r}")
    if args.cpu_fallback:
        raise ValueError("CPU fallback is forbidden for REF4 127")
    if args.iterations <= 0 or args.depth <= 0 or args.learning_rate <= 0.0:
        raise ValueError("iterations, depth, and learning-rate must be positive")
    if args.bootstrap < 50:
        raise ValueError("bootstrap must be at least 50")
    iterations = min(args.iterations, 3) if args.smoke else args.iterations
    max_rows = args.max_rows_per_year
    if args.smoke and max_rows is None:
        max_rows = 2_000
    config = Config(
        train=train_path,
        anchor=args.anchor,
        p109c_anchor=args.p109c_anchor,
        output=args.output,
        checkpoint_dir=args.checkpoint_dir,
        requested_device=args.device,
        devices=args.devices,
        threads=args.threads,
        iterations=iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        bootstrap_repeats=args.bootstrap,
        resume=args.resume,
        cpu_fallback=args.cpu_fallback,
        max_rows_per_year=max_rows,
        smoke=args.smoke,
    )
    result = run(config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
