#!/usr/bin/env python3
"""Train and compare three fixed strict-forward 110 candidates."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
try:
    os.nice(10)
except OSError:
    pass

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "REF4-110-CODEX-R1"
OUT = ROOT / "model" / EXPERIMENT_ID
TRAIN_PATH = ROOT / "data/train.csv"
SOURCE_PATH = ROOT / "model/REF4-OOF-DIAG-034A/diagnostic_rows.csv"
THREADS = 4
R_SEEDS = [42, 1, 2, 3, 4]
F_SEEDS = [42, 1, 2, 3]
ROUTER_SEEDS = [42, 1, 2]
WEIGHT_R = 0.085
WEIGHT_F = 0.035
YEARS = [2022, 2023, 2024]
BOOTSTRAP_REPS = 2000
BOOTSTRAP_SEED = 110_2026

import sys
sys.path.insert(0, str(ROOT))
from src.ref4_110_codex import (  # noqa: E402
    apply_eb_tables,
    build_eb_tables,
    build_residual_features,
    router_features,
)


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def cb_params(seed: int, futures: bool = False) -> dict[str, object]:
    return {
        "iterations": 150 if futures else 220,
        "depth": 4 if futures else 5,
        "learning_rate": 0.03,
        "l2_leaf_reg": 20.0 if futures else 15.0,
        "loss_function": "RMSE",
        "random_seed": seed,
        "thread_count": THREADS,
        "allow_writing_files": False,
        "verbose": False,
    }


def train_family(
    prefix: str,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_valid: pd.DataFrame,
    model_dir: Path,
    futures: bool,
) -> tuple[dict[str, np.ndarray], list[Path]]:
    model_dir.mkdir(parents=True, exist_ok=True)
    seeds = F_SEEDS if futures else R_SEEDS
    predictions: dict[str, list[np.ndarray]] = {"cb": [], "lgb": [], "xgb": []}
    paths: list[Path] = []
    for seed in seeds:
        cb_path = model_dir / f"{prefix}_cb_seed{seed}.cbm"
        model = CatBoostRegressor(**cb_params(seed, futures=futures))
        model.fit(x_train, y_train)
        model.save_model(cb_path)
        predictions["cb"].append(model.predict(x_valid).astype(np.float64))
        paths.append(cb_path)
        del model
        gc.collect()

        if futures:
            continue

        lgb_path = model_dir / f"{prefix}_lgb_seed{seed}.txt"
        params = {
            "objective": "regression", "metric": "rmse", "learning_rate": 0.03,
            "num_leaves": 31, "max_depth": 5, "min_data_in_leaf": 50,
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
            "random_seed": seed, "num_threads": THREADS, "verbosity": -1,
        }
        booster = lgb.train(params, lgb.Dataset(x_train, label=y_train), num_boost_round=160)
        booster.save_model(str(lgb_path))
        predictions["lgb"].append(booster.predict(x_valid).astype(np.float64))
        paths.append(lgb_path)
        del booster
        gc.collect()

        xgb_path = model_dir / f"{prefix}_xgb_seed{seed}.json"
        xgb_model = XGBRegressor(
            n_estimators=170, learning_rate=0.03, max_depth=4, subsample=0.8,
            colsample_bytree=0.8, reg_lambda=3.0, random_state=seed,
            n_jobs=THREADS, tree_method="hist",
        )
        xgb_model.fit(x_train, y_train)
        xgb_model.save_model(xgb_path)
        predictions["xgb"].append(xgb_model.predict(x_valid).astype(np.float64))
        paths.append(xgb_path)
        del xgb_model
        gc.collect()
    return {name: np.mean(members, axis=0) if members else np.zeros(len(x_valid)) for name, members in predictions.items()}, paths


def fit_candidate_fold(
    name: str,
    train_df: pd.DataFrame,
    train_anchor: np.ndarray,
    train_target: np.ndarray,
    valid_df: pd.DataFrame,
    valid_anchor: np.ndarray,
    train_eb: pd.DataFrame | None,
    valid_eb: pd.DataFrame | None,
    model_dir: Path,
) -> tuple[np.ndarray, np.ndarray, list[Path]]:
    x_train = build_residual_features(train_df, train_anchor, train_eb)
    x_valid = build_residual_features(valid_df, valid_anchor, valid_eb)
    residual = train_target - train_anchor
    is_f_train = train_df["game_type"].astype(str).eq("F").to_numpy()
    is_f_valid = valid_df["game_type"].astype(str).eq("F").to_numpy()
    correction = np.zeros(len(valid_df), dtype=np.float64)
    experts = np.column_stack([valid_anchor, valid_anchor, valid_anchor]).astype(np.float64)
    paths: list[Path] = []

    reg_train = ~is_f_train
    reg_valid = ~is_f_valid
    if reg_train.any() and reg_valid.any():
        family, saved = train_family(f"{name}_reg", x_train.loc[reg_train], residual[reg_train], x_valid.loc[reg_valid], model_dir, futures=False)
        cb = family["cb"]
        cb_lgb = np.mean(np.vstack([family["cb"], family["lgb"]]), axis=0)
        tri = np.mean(np.vstack([family["cb"], family["lgb"], family["xgb"]]), axis=0)
        correction[reg_valid] = tri
        experts[reg_valid, 0] = np.clip(valid_anchor[reg_valid] + WEIGHT_R * cb, 1e-5, 1 - 1e-5)
        experts[reg_valid, 1] = np.clip(valid_anchor[reg_valid] + WEIGHT_R * cb_lgb, 1e-5, 1 - 1e-5)
        experts[reg_valid, 2] = np.clip(valid_anchor[reg_valid] + WEIGHT_R * tri, 1e-5, 1 - 1e-5)
        paths.extend(saved)

    if is_f_train.any() and is_f_valid.any():
        family, saved = train_family(f"{name}_fut", x_train.loc[is_f_train], residual[is_f_train], x_valid.loc[is_f_valid], model_dir, futures=True)
        corr = family["cb"]
        correction[is_f_valid] = corr
        experts[is_f_valid, 0] = np.clip(valid_anchor[is_f_valid] + 0.5 * WEIGHT_F * corr, 1e-5, 1 - 1e-5)
        experts[is_f_valid, 1] = np.clip(valid_anchor[is_f_valid] + 0.75 * WEIGHT_F * corr, 1e-5, 1 - 1e-5)
        experts[is_f_valid, 2] = np.clip(valid_anchor[is_f_valid] + WEIGHT_F * corr, 1e-5, 1 - 1e-5)
        paths.extend(saved)

    prediction = np.where(is_f_valid, valid_anchor + WEIGHT_F * correction, valid_anchor + WEIGHT_R * correction)
    return np.clip(prediction, 1e-5, 1 - 1e-5), experts, paths


def fit_router(
    train_df: pd.DataFrame,
    train_experts: np.ndarray,
    train_target: np.ndarray,
    valid_df: pd.DataFrame,
    valid_experts: np.ndarray,
    model_dir: Path,
) -> tuple[np.ndarray, list[Path]]:
    model_dir.mkdir(parents=True, exist_ok=True)
    x_train = router_features(train_df, train_experts)
    x_valid = router_features(valid_df, valid_experts)
    target = train_target - train_experts[:, 2]
    members = []
    paths = []
    for seed in ROUTER_SEEDS:
        path = model_dir / f"router_seed{seed}.cbm"
        model = CatBoostRegressor(
            iterations=160, depth=4, learning_rate=0.025, l2_leaf_reg=25.0,
            loss_function="RMSE", random_seed=seed, thread_count=THREADS,
            allow_writing_files=False, verbose=False,
        )
        model.fit(x_train, target)
        model.save_model(path)
        members.append(model.predict(x_valid).astype(np.float64))
        paths.append(path)
        del model
    correction = np.mean(members, axis=0)
    return np.clip(valid_experts[:, 2] + correction, 1e-5, 1 - 1e-5), paths


def metric(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    rate = float(y.mean())
    brier = float(np.mean((y - prediction) ** 2))
    bss = float(1.0 - brier / (rate * (1.0 - rate)))
    return {"rows": int(len(y)), "target_rate": rate, "prediction_mean": float(prediction.mean()), "brier": brier, "bss": bss, "local_score": 100000.0 * bss}


def cluster_delta_ci(frame: pd.DataFrame, base: np.ndarray, candidate: np.ndarray, seed: int) -> dict[str, float | int]:
    y = frame["target"].to_numpy(float)
    delta = (y - candidate) ** 2 - (y - base) ** 2
    grouped = pd.DataFrame({"pitcher": frame["pitcher_id"].astype(str), "delta": delta}).groupby("pitcher", sort=True)["delta"].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(seed)
    samples = np.empty(BOOTSTRAP_REPS, dtype=float)
    for start in range(0, BOOTSTRAP_REPS, 100):
        stop = min(start + 100, BOOTSTRAP_REPS)
        draws = rng.integers(0, len(grouped), size=(stop - start, len(grouped)))
        samples[start:stop] = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return {"clusters": int(len(grouped)), "repetitions": BOOTSTRAP_REPS, "seed": seed, "delta_brier": float(delta.mean()), "ci_low": float(np.quantile(samples, 0.025)), "ci_high": float(np.quantile(samples, 0.975))}


def main() -> None:
    started = time.time()
    preflight = json.loads((OUT / "preflight_report.json").read_text(encoding="utf-8"))
    if preflight.get("status") != "AUDIT_VERIFIED" or preflight.get("mismatch_count") != 0:
        raise RuntimeError("preflight is not AUDIT_VERIFIED")

    source = pd.read_csv(SOURCE_PATH, dtype={"row_id": str, "game_type": str, "pitcher_id": str})
    raw = pd.read_csv(TRAIN_PATH, low_memory=False)
    raw["row_id"] = raw["row_id"].astype(str)
    raw_indexed = raw.set_index("row_id", drop=False)
    frame = raw_indexed.loc[source["row_id"]].reset_index(drop=True)
    if not np.array_equal(frame["control_success"].to_numpy(float), source["target"].to_numpy(float)):
        raise RuntimeError("source/train target mismatch")
    frame["target"] = source["target"].to_numpy(float)
    frame["base_prediction"] = source["prediction_current"].to_numpy(float)
    anchor = frame["base_prediction"].to_numpy(float)
    target = frame["target"].to_numpy(float)

    eb_all = pd.DataFrame(index=frame.index, columns=["eb_pitcher", "eb_pitcher_hand", "eb_reliability", "eb_shrunk", "eb_vs_asof", "eb_hand_vs_pitcher"], dtype=float)
    eb_provenance = {}
    for year in YEARS:
        history = raw.loc[raw["season"].lt(year)]
        tables = build_eb_tables(history, 100.0, 50.0)
        mask = frame["season"].eq(year)
        eb_all.loc[mask] = apply_eb_tables(frame.loc[mask], tables, 25.0).to_numpy()
        eb_provenance[str(year)] = {"history_max_season": int(history["season"].max()), "history_rows": int(len(history)), "pitchers": len(tables["pitcher"]), "pairs": len(tables["pitcher_hand"])}

    predictions = {"110A": anchor.copy(), "110B": anchor.copy(), "110C": anchor.copy()}
    b_experts_by_year: dict[int, np.ndarray] = {}
    model_paths: list[Path] = []
    fold_provenance = {}
    for year in [2023, 2024]:
        train_mask = frame["season"].lt(year).to_numpy()
        valid_mask = frame["season"].eq(year).to_numpy()
        fold_dir = OUT / f"fold_{year}"
        print(f"[{year}] 110B strict forward: {train_mask.sum():,} -> {valid_mask.sum():,}", flush=True)
        p_b, experts_b, paths_b = fit_candidate_fold(
            "110b", frame.loc[train_mask].reset_index(drop=True), anchor[train_mask], target[train_mask],
            frame.loc[valid_mask].reset_index(drop=True), anchor[valid_mask], None, None, fold_dir / "110B",
        )
        predictions["110B"][valid_mask] = p_b
        b_experts_by_year[year] = experts_b
        model_paths.extend(paths_b)

        print(f"[{year}] 110C strict forward EB", flush=True)
        p_c, _, paths_c = fit_candidate_fold(
            "110c", frame.loc[train_mask].reset_index(drop=True), anchor[train_mask], target[train_mask],
            frame.loc[valid_mask].reset_index(drop=True), anchor[valid_mask],
            eb_all.loc[train_mask].reset_index(drop=True), eb_all.loc[valid_mask].reset_index(drop=True), fold_dir / "110C",
        )
        predictions["110C"][valid_mask] = p_c
        model_paths.extend(paths_c)
        fold_provenance[str(year)] = {"train_seasons": sorted(frame.loc[train_mask, "season"].unique().astype(int).tolist()), "train_rows": int(train_mask.sum()), "valid_season": year, "valid_rows": int(valid_mask.sum()), "validation_labels_used_in_fit": False}

    # 110A: 2023 uses the strongest forward residual expert; the learned router
    # starts in 2024 and is fitted only on 2023 expert predictions/labels.
    mask_2023 = frame["season"].eq(2023).to_numpy()
    predictions["110A"][mask_2023] = b_experts_by_year[2023][:, 2]
    mask_2024 = frame["season"].eq(2024).to_numpy()
    p_a_2024, router_paths = fit_router(
        frame.loc[mask_2023].reset_index(drop=True), b_experts_by_year[2023], target[mask_2023],
        frame.loc[mask_2024].reset_index(drop=True), b_experts_by_year[2024], OUT / "fold_2024/110A",
    )
    predictions["110A"][mask_2024] = p_a_2024
    model_paths.extend(router_paths)

    output = frame[["row_id", "season", "game_type", "pitcher_id", "target", "base_prediction"]].copy()
    for name in ["110A", "110B", "110C"]:
        output[name] = predictions[name]
    output.to_csv(OUT / "oof_predictions.csv", index=False)

    base_metrics = {}
    candidates = []
    recent_weights = {2022: 0.20, 2023: 0.30, 2024: 0.50}
    for year in YEARS:
        mask = frame["season"].eq(year).to_numpy()
        base_metrics[str(year)] = metric(target[mask], anchor[mask])
    for name in ["110A", "110B", "110C"]:
        season_metrics = {}
        deltas = {}
        cis = {}
        for year in YEARS:
            mask = frame["season"].eq(year).to_numpy()
            season_metrics[str(year)] = metric(target[mask], predictions[name][mask])
            deltas[str(year)] = float(season_metrics[str(year)]["brier"] - base_metrics[str(year)]["brier"])
            part = output.loc[mask].reset_index(drop=True)
            cis[str(year)] = cluster_delta_ci(part, anchor[mask], predictions[name][mask], BOOTSTRAP_SEED + year + ["110A", "110B", "110C"].index(name) * 10000)
        weighted = float(sum(recent_weights[year] * deltas[str(year)] for year in YEARS))
        worst = float(max(deltas.values()))
        gates = {
            "delta_2024_at_most_minus_0_0001": deltas["2024"] <= -0.0001,
            "worst_season_delta_at_most_plus_0_00005": worst <= 0.00005,
            "time_weighted_delta_negative": weighted < 0.0,
            "bootstrap_2024_ci_high_below_zero": float(cis["2024"]["ci_high"]) < 0.0,
        }
        candidates.append({
            "candidate_name": name, "candidate_status": "PENDING_AUDIT",
            "metrics": season_metrics, "delta_brier_vs_strict_base": deltas,
            "time_weighted_delta": weighted, "worst_season_delta": worst,
            "cluster_bootstrap": cis, "gate_results": gates, "performance_gate_pass": all(gates.values()),
        })
    winner = min(candidates, key=lambda item: item["time_weighted_delta"])["candidate_name"]
    result = {
        "experiment_id": EXPERIMENT_ID, "status": "PENDING_AUDIT",
        "comparison_scope": "strict-forward residual mechanism comparison on audited REF4 predecessor OOF",
        "anchor_transfer_to_109c_verified": False,
        "official_score_estimated": False,
        "base_metrics": base_metrics,
        "candidate_count": len(candidates), "actual_leaf_count": len(candidates),
        "candidates": candidates, "provisional_winner": winner,
        "gate_checks_count": sum(len(item["gate_results"]) for item in candidates),
        "fold_provenance": fold_provenance, "eb_provenance": eb_provenance,
        "model_count": len(model_paths), "oof_rows": len(output),
        "test_read": False, "test_inference_performed": False,
        "full_train_performed": False, "zip_created": False,
        "elapsed_seconds": time.time() - started,
    }
    write_json(OUT / "result.json", result)
    lines = [
        f"# {EXPERIMENT_ID}", "", "- status: `PENDING_AUDIT`",
        f"- leaf candidates: `{len(candidates)}`", f"- OOF rows: `{len(output)}`",
        "- anchor-transfer-to-109C: `false`", "",
        "| candidate | 2022 delta | 2023 delta | 2024 delta | weighted delta | worst delta | gates |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in candidates:
        d = item["delta_brier_vs_strict_base"]
        lines.append(f"| {item['candidate_name']} | {d['2022']:+.12g} | {d['2023']:+.12g} | {d['2024']:+.12g} | {item['time_weighted_delta']:+.12g} | {item['worst_season_delta']:+.12g} | {sum(item['gate_results'].values())}/{len(item['gate_results'])} |")
    lines.extend(["", f"- provisional winner: `{winner}`", "- No official score is predicted or copied into this report."])
    (OUT / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    artifact_paths = [
        ROOT / "scripts/preflight_ref4_110_codex_r1.py",
        ROOT / "scripts/run_ref4_110_codex_r1.py",
        ROOT / "src/ref4_110_codex.py",
        OUT / "audit_contract.json", OUT / "preflight_report.json", OUT / "preflight_report.md",
        OUT / "oof_predictions.csv", OUT / "result.json", OUT / "result.md",
        TRAIN_PATH, SOURCE_PATH, ROOT / "output/submit_ref4_super_ensemble_109C.zip",
        ROOT / "output/lb_record_109.json", *model_paths,
    ]
    records = {str(path.relative_to(ROOT)): {"sha256": sha256_path(path), "size": path.stat().st_size} for path in artifact_paths}
    write_json(OUT / "audit_manifest.json", {
        "experiment_id": EXPERIMENT_ID, "status": "PENDING_VALIDATION",
        "artifact_count": len(records), "artifacts": records,
        "candidate_count": len(candidates), "leaf_count": len(candidates),
        "gate_checks_count": result["gate_checks_count"], "model_count": len(model_paths), "oof_rows": len(output),
    })
    print(json.dumps({"status": "PENDING_AUDIT", "winner": winner, "candidate_count": len(candidates), "model_count": len(model_paths), "elapsed_seconds": result["elapsed_seconds"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
