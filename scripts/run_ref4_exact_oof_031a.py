#!/usr/bin/env python3
"""Build the exact current REF4-030 structure as strict 2023/2024 forward OOF."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("MKL_NUM_THREADS", "3")
try:
    os.nice(10)
except OSError:
    pass

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "REF4-EXACT-OOF-031A"
OUT = ROOT / "model" / EXPERIMENT_ID
SOURCE_MODEL = ROOT / "model" / "REF4-CHAMPION-STACK-030"
CANDIDATE = ROOT / "candidate" / "REF4-CHAMPION-STACK-030"
TRAIN_PATH = ROOT / "data" / "train.csv"
TRACKMAN_PATH = ROOT / "data" / "trackman_history.csv"
THREAD_COUNT = 3
VALID_YEARS = (2023, 2024)

sys.path.insert(0, str(CANDIDATE))
from src.preprocessing_v2 import CAT_V2, build_v2_features, build_v3_features  # noqa: E402
from src.season_delta_features import build_snapshots  # noqa: E402
from src.season_history_v3 import build_entity_snapshots  # noqa: E402


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def strict_failure_labels(history: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Recover labels using training rows only; validation rows cannot reveal a label."""
    group = history.groupby("pitcher_id", sort=False)
    n = pd.to_numeric(history["asof_pitcher_n"], errors="coerce")
    next_n = pd.to_numeric(group["asof_pitcher_n"].shift(-1), errors="coerce")
    valid = next_n.eq(n + 1)
    recovered: list[np.ndarray] = []
    for col in ("asof_pitcher_middle_rate", "asof_pitcher_ball_rate", "asof_pitcher_reverse_rate"):
        rate = pd.to_numeric(history[col], errors="coerce")
        next_rate = pd.to_numeric(group[col].shift(-1), errors="coerce")
        delta = next_n * next_rate - n * rate
        valid &= delta.notna()
        recovered.append((delta > 0.5).to_numpy(np.float32))
    middle, ball_result, reverse = recovered
    failure = 1.0 - history["control_success"].to_numpy(np.float32)
    wild = failure * (1.0 - middle) * (1.0 - reverse)
    labels = np.stack([middle, wild, reverse], axis=1).astype(np.float32)
    mask = valid.to_numpy(bool)
    audit = {
        "rows": int(len(history)), "recovered_rows": int(mask.sum()), "coverage": float(mask.mean()),
        "ball_result_rate": float(ball_result[mask].mean()),
        "middle_rate": float(middle[mask].mean()),
        "wild_rate": float(wild[mask].mean()), "reverse_rate": float(reverse[mask].mean()),
    }
    return labels, mask, audit


def build_fold_trackman(train_history: pd.DataFrame, valid_year: int, fold_dir: Path) -> tuple[Path, dict[str, object]]:
    mapping_path = fold_dir / "pitcher_trackman_mapping.csv"
    table_path = fold_dir / "trackman_prior_features.csv"
    audit_path = fold_dir / "trackman_audit.json"
    if mapping_path.exists() and table_path.exists() and audit_path.exists():
        return table_path, json.loads(audit_path.read_text(encoding="utf-8"))

    metrics = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break", "extension", "rel_height", "rel_side", "zone_speed"]
    tm_columns = ["pitcher_trackman_id", "season", "game_month", "game_dayofweek", "pitcher_hand", "pitch_type_group", *metrics]
    tm = pd.read_csv(TRACKMAN_PATH, usecols=tm_columns, low_memory=False)
    tm = tm.loc[pd.to_numeric(tm["season"], errors="coerce").lt(valid_year)].reset_index(drop=True)
    key_train = train_history[["pitcher_id", "season", "game_month", "game_dayofweek", "pitcher_hand"]].copy()
    key_tm = tm[["pitcher_trackman_id", "season", "game_month", "game_dayofweek", "pitcher_hand"]].copy()
    key_train["key"] = key_train["season"].astype(str) + "_" + key_train["game_month"].astype(str) + "_" + key_train["game_dayofweek"].astype(str)
    key_tm["key"] = key_tm["season"].astype(str) + "_" + key_tm["game_month"].astype(str) + "_" + key_tm["game_dayofweek"].astype(str)
    a = key_train.groupby(["pitcher_id", "key"]).size().unstack(fill_value=0)
    b = key_tm.groupby(["pitcher_trackman_id", "key"]).size().unstack(fill_value=0)
    all_columns = a.columns.union(b.columns)
    a = a.reindex(columns=all_columns, fill_value=0)
    b = b.reindex(columns=all_columns, fill_value=0)
    similarity = (normalize(csr_matrix(a.values)) @ normalize(csr_matrix(b.values)).T).toarray()
    train_hand = key_train.groupby("pitcher_id")["pitcher_hand"].first().reindex(a.index).to_numpy()
    tm_hand = key_tm.groupby("pitcher_trackman_id")["pitcher_hand"].first().reindex(b.index).map({"Left": 1, "Right": 2}).to_numpy()
    similarity[train_hand[:, None] != tm_hand[None, :]] = -1
    best = similarity.argmax(axis=1)
    reverse_best = similarity.argmax(axis=0)
    ordered = np.sort(similarity, axis=1)
    confidence = similarity[np.arange(len(a)), best]
    margin = ordered[:, -1] - ordered[:, -2]
    accepted = (reverse_best[best] == np.arange(len(a))) & (confidence >= 0.90) & (margin >= 0.03)
    mapping = pd.DataFrame({
        "pitcher_id": a.index[accepted], "pitcher_trackman_id": b.index[best[accepted]],
        "mapping_similarity": confidence[accepted], "mapping_margin": margin[accepted],
    })
    mapping.to_csv(mapping_path, index=False)

    groups = {key: value for key, value in tm.groupby("pitcher_trackman_id", sort=False)}
    rows: list[dict[str, object]] = []
    min_season = int(train_history["season"].min())
    for rec in mapping.itertuples(index=False):
        player = groups[rec.pitcher_trackman_id]
        for target_season in range(min_season, valid_year + 1):
            hist = player.loc[player["season"].lt(target_season)]
            row: dict[str, object] = {
                "pitcher_id": str(rec.pitcher_id), "season": target_season,
                "tm_mapping_similarity": float(rec.mapping_similarity), "tm_n": int(len(hist)),
            }
            for column in metrics:
                row[f"tm_{column}_mean"] = hist[column].mean()
                row[f"tm_{column}_std"] = hist[column].std()
                last = hist.loc[hist["season"].eq(target_season - 1), column]
                earlier = hist.loc[hist["season"].lt(target_season - 1), column]
                row[f"tm_last_{column}_mean"] = last.mean()
                row[f"tm_last_{column}_std"] = last.std()
                row[f"tm_{column}_trend"] = last.mean() - earlier.mean()
            rates = hist["pitch_type_group"].value_counts(normalize=True)
            for pitch_type in ("fastball", "breaking", "offspeed", "other"):
                row[f"tm_{pitch_type}_rate"] = rates.get(pitch_type, np.nan)
            last_hist = hist.loc[hist["season"].eq(target_season - 1)]
            row["tm_last_n"] = int(len(last_hist))
            last_rates = last_hist["pitch_type_group"].value_counts(normalize=True)
            for pitch_type in ("fastball", "breaking", "offspeed", "other"):
                row[f"tm_last_{pitch_type}_rate"] = last_rates.get(pitch_type, np.nan)
                subset = hist.loc[hist["pitch_type_group"].eq(pitch_type)]
                recent_subset = last_hist.loc[last_hist["pitch_type_group"].eq(pitch_type)]
                for column in ("rel_speed", "spin_rate", "induced_vert_break", "horz_break"):
                    row[f"tm_{pitch_type}_{column}_mean"] = subset[column].mean()
                    row[f"tm_last_{pitch_type}_{column}_mean"] = recent_subset[column].mean()
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(table_path, index=False)
    audit = {
        "valid_year": valid_year, "mapping_source_max_season": int(key_train["season"].max()),
        "trackman_source_max_season": int(tm["season"].max()), "mapping_count": int(len(mapping)),
        "mapping_train_row_coverage": float(key_train["pitcher_id"].isin(mapping["pitcher_id"]).mean()),
        "table_rows": int(len(table)), "duplicate_keys": int(table.duplicated(["pitcher_id", "season"]).sum()),
    }
    atomic_json(audit_path, audit)
    del tm, key_tm, key_train, a, b, similarity, groups, table
    gc.collect()
    return table_path, audit


def fit_or_load_regressor(path: Path, x: pd.DataFrame, target: np.ndarray, mask: np.ndarray,
                          weights: np.ndarray | None, params: dict[str, object]) -> CatBoostRegressor:
    model = CatBoostRegressor()
    if path.exists():
        model.load_model(path)
        return model
    model = CatBoostRegressor(**params)
    model.fit(x.loc[mask], target[mask], sample_weight=weights, cat_features=CAT_V2)
    model.save_model(path)
    return model


def fit_or_load_classifier(path: Path, x: pd.DataFrame, labels: np.ndarray, indices: np.ndarray,
                           weights: np.ndarray, params: dict[str, object]) -> CatBoostClassifier:
    model = CatBoostClassifier()
    if path.exists():
        model.load_model(path)
        return model
    model = CatBoostClassifier(**params)
    model.fit(x.iloc[indices], labels, sample_weight=weights, cat_features=CAT_V2)
    model.save_model(path)
    return model


def mean_regression_predictions(stem: str, count_or_seeds: list[int], x: pd.DataFrame, base: np.ndarray,
                                train_mask: np.ndarray, valid_mask: np.ndarray, target: np.ndarray,
                                weights: np.ndarray | None, model_dir: Path, params: dict[str, object],
                                seed_key: str = "seed") -> np.ndarray:
    total = np.zeros(int(valid_mask.sum()), dtype=np.float64)
    for position, seed in enumerate(count_or_seeds):
        path = model_dir / (f"{stem}_seed{seed}.cbm" if seed_key == "seed" else f"{stem}_{position}.cbm")
        model_params = dict(params)
        model_params["random_seed"] = int(seed)
        started = time.time()
        model = fit_or_load_regressor(path, x, target, train_mask, weights, model_params)
        member = np.clip(base[valid_mask] + model.predict(x.loc[valid_mask]), 1e-6, 1 - 1e-6)
        total += member
        print(f"  {path.name}: {time.time() - started:.1f}s", flush=True)
        del model, member
        gc.collect()
    return total / len(count_or_seeds)


def fold_predictions(raw: pd.DataFrame, valid_year: int, manifest: dict[str, object], regime: dict[str, float]) -> tuple[pd.DataFrame, dict[str, object]]:
    fold_started = time.time()
    fold_dir = OUT / f"fold_{valid_year}"
    model_dir = fold_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    frame = raw.loc[raw["season"].le(valid_year)].reset_index(drop=True)
    seasons = frame["season"].to_numpy(int)
    y = frame["control_success"].to_numpy(np.float32)
    train_mask = seasons < valid_year
    valid_mask = seasons == valid_year
    train_history = frame.loc[train_mask].reset_index(drop=True)
    prior = float(y[train_mask].mean())
    fold_contract = {
        "valid_year": valid_year, "train_min_season": int(seasons[train_mask].min()),
        "train_max_season": int(seasons[train_mask].max()), "train_rows": int(train_mask.sum()),
        "valid_rows": int(valid_mask.sum()), "prior": prior, "thread_count": THREAD_COUNT,
        "expected_contributing_model_count": 55, "seeds": manifest["seeds"],
        "transition_scale": regime["transition_scale"], "source_train_sha256": sha256_path(TRAIN_PATH),
    }
    contract_path = fold_dir / "fold_contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text(encoding="utf-8")) != fold_contract:
            raise RuntimeError(f"existing fold contract mismatch: {contract_path}")
    else:
        atomic_json(contract_path, fold_contract)

    print(f"[{valid_year}] TrackMan fold-local mapping", flush=True)
    tm_path, tm_audit = build_fold_trackman(train_history, valid_year, fold_dir)
    print(f"[{valid_year}] snapshots", flush=True)
    pitcher_snap = build_snapshots(train_history)
    batter_snap = build_entity_snapshots(
        train_history, "batter_id", "asof_batter_n",
        ["asof_batter_success_rate", "asof_batter_middle_rate"], "control_success",
    )
    mix_snap = build_entity_snapshots(
        train_history, "pitcher_id", "asof_pitcher_pitchmix_n",
        ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"],
    )

    common_reg = {
        "depth": 8, "learning_rate": 0.035, "loss_function": "RMSE", "random_strength": 0.35,
        "bootstrap_type": "Bernoulli", "subsample": 0.85, "one_hot_max_size": 16,
        "thread_count": THREAD_COUNT, "allow_writing_files": False, "verbose": False,
    }
    seeds = [int(value) for value in manifest["seeds"]]
    weights55 = np.power(0.55, (valid_year - 1) - seasons[train_mask])
    weights30 = np.power(0.30, (valid_year - 1) - seasons[train_mask])
    target_indices = np.flatnonzero(train_mask)

    print(f"[{valid_year}] v2 features", flush=True)
    x2, base2 = build_v2_features(frame, prior, pitcher_snap, str(tm_path))
    target2 = y - base2
    p_v2_global = mean_regression_predictions(
        "v2_decay55", seeds, x2, base2, train_mask, valid_mask, target2, weights55,
        model_dir, {**common_reg, "iterations": 140, "l2_leaf_reg": 12},
    )
    f_train = train_mask & frame["game_type"].eq("F").to_numpy()
    f_train_indices = np.flatnonzero(f_train)
    f_weights55 = np.power(0.55, (valid_year - 1) - seasons[f_train])
    p_v2_f = mean_regression_predictions(
        "f_v2_all", [968000 + j for j in range(4)], x2, base2, f_train, valid_mask,
        target2, f_weights55, model_dir, {**common_reg, "iterations": 140, "l2_leaf_reg": 20}, seed_key="index",
    )
    del x2, base2, target2
    gc.collect()

    print(f"[{valid_year}] v3 features", flush=True)
    x3, base3 = build_v3_features(frame, prior, pitcher_snap, batter_snap, mix_snap, str(tm_path))
    target3 = y - base3
    p_v3_55_global = mean_regression_predictions(
        "v3_decay55", seeds, x3, base3, train_mask, valid_mask, target3, weights55,
        model_dir, {**common_reg, "iterations": 220, "l2_leaf_reg": 12},
    )
    p_v3_30_global = mean_regression_predictions(
        "v3_decay30", seeds, x3, base3, train_mask, valid_mask, target3, weights30,
        model_dir, {**common_reg, "iterations": 199, "l2_leaf_reg": 12},
    )
    recent_f_train = f_train & seasons.__eq__(valid_year - 1)
    p_v3_55_f = mean_regression_predictions(
        "f_v355_recent", [968100 + j for j in range(6)], x3, base3, recent_f_train, valid_mask,
        target3, None, model_dir, {**common_reg, "iterations": 220, "l2_leaf_reg": 20}, seed_key="index",
    )
    f_weights30 = np.power(0.30, (valid_year - 1) - seasons[f_train])
    p_v3_30_f_all = mean_regression_predictions(
        "f_v330_all", [968200 + j for j in range(4)], x3, base3, f_train, valid_mask,
        target3, f_weights30, model_dir, {**common_reg, "iterations": 199, "l2_leaf_reg": 20}, seed_key="index",
    )
    p_v3_30_f_recent = mean_regression_predictions(
        "f_v330_recent", [968300 + j for j in range(2)], x3, base3, recent_f_train, valid_mask,
        target3, None, model_dir, {**common_reg, "iterations": 199, "l2_leaf_reg": 20}, seed_key="index",
    )

    labels, recovered, subtype_audit = strict_failure_labels(train_history)
    recovered_train_indices = target_indices[recovered]
    recovered_seasons = seasons[recovered_train_indices]
    subtype_weights = np.power(0.30, (valid_year - 1) - recovered_seasons)
    classifier_common = {
        "depth": 7, "learning_rate": 0.04, "loss_function": "Logloss", "random_strength": 0.4,
        "bootstrap_type": "Bernoulli", "subsample": 0.85, "one_hot_max_size": 16,
        "thread_count": THREAD_COUNT, "allow_writing_files": False, "verbose": False,
    }
    subtype_specs = [("middle", 0, 100), ("wild", 1, 190), ("reverse", 2, 230)]
    risk_global: dict[str, np.ndarray] = {}
    risk_f: dict[str, np.ndarray] = {}
    valid_x3 = x3.loc[valid_mask]
    for subtype_position, (name, label_index, iterations) in enumerate(subtype_specs):
        total = np.zeros(int(valid_mask.sum()), dtype=np.float64)
        for seed in seeds:
            path = model_dir / f"subtype_{name}_seed{seed}.cbm"
            params = {**classifier_common, "iterations": iterations, "l2_leaf_reg": 12, "random_seed": seed + subtype_position * 100}
            model = fit_or_load_classifier(
                path, x3, labels[recovered, label_index], recovered_train_indices, subtype_weights, params,
            )
            total += model.predict_proba(valid_x3)[:, 1]
            print(f"  {path.name}", flush=True)
            del model
            gc.collect()
        risk_global[name] = total / len(seeds)

        f_recovered = recovered & train_history["game_type"].eq("F").to_numpy()
        f_recovered_indices = target_indices[f_recovered]
        f_subtype_weights = np.power(0.30, (valid_year - 1) - seasons[f_recovered_indices])
        path = model_dir / f"f_subtype_{name}.cbm"
        params = {**classifier_common, "iterations": iterations, "l2_leaf_reg": 20, "random_seed": 968400 + subtype_position}
        model = fit_or_load_classifier(
            path, x3, labels[f_recovered, label_index], f_recovered_indices, f_subtype_weights, params,
        )
        risk_f[name] = model.predict_proba(valid_x3)[:, 1]
        print(f"  {path.name}", flush=True)
        del model
        gc.collect()

    valid_rows = frame.loc[valid_mask].reset_index(drop=True)
    futures = valid_rows["game_type"].eq("F").to_numpy()
    p0 = np.where(futures, p_v2_global + regime["v2_scale"] * (p_v2_f - p_v2_global), p_v2_global)
    p1 = np.where(futures, p_v3_55_global + regime["v355_scale"] * (p_v3_55_f - p_v3_55_global), p_v3_55_global)
    recent_inner = p_v3_30_global + regime["v330_recent_inner_scale"] * (p_v3_30_f_recent - p_v3_30_global)
    f30 = regime["v330_all_weight"] * p_v3_30_f_all + (1.0 - regime["v330_all_weight"]) * recent_inner
    p2 = np.where(futures, p_v3_30_global + regime["v330_scale"] * (f30 - p_v3_30_global), p_v3_30_global)
    adjusted_risks = [
        np.where(futures, risk_global[name] + regime["subtype_scale"] * (risk_f[name] - risk_global[name]), risk_global[name])
        for name in ("middle", "wild", "reverse")
    ]
    main_prediction = np.average(np.vstack([p0, p1, p2]), axis=0, weights=np.asarray(manifest["main_weights"], float))
    z = np.column_stack([main_prediction, *adjusted_risks])
    prediction_no_shift = float(manifest["stack_intercept"]) + z @ np.asarray(manifest["stack_coefficients"], float)
    prediction = np.clip(prediction_no_shift + float(manifest["global_shift"]), 1e-5, 1 - 1e-5)

    output = pd.DataFrame({
        "row_id": valid_rows["row_id"].to_numpy(), "season": valid_year,
        "game_type": valid_rows["game_type"].astype(str).to_numpy(),
        "pitcher_id": valid_rows["pitcher_id"].astype(str).to_numpy(),
        "target": valid_rows["control_success"].to_numpy(np.int8),
        "p_v2_global": p_v2_global, "p_v2_f": p_v2_f,
        "p_v3_55_global": p_v3_55_global, "p_v3_55_f": p_v3_55_f,
        "p_v3_30_global": p_v3_30_global, "p_v3_30_f_all": p_v3_30_f_all,
        "p_v3_30_f_recent": p_v3_30_f_recent,
        "risk_middle_global": risk_global["middle"], "risk_middle_f": risk_f["middle"],
        "risk_wild_global": risk_global["wild"], "risk_wild_f": risk_f["wild"],
        "risk_reverse_global": risk_global["reverse"], "risk_reverse_f": risk_f["reverse"],
        "prediction_no_shift": prediction_no_shift, "prediction": prediction,
    })
    fold_meta = {
        **fold_contract, "actual_model_count": len(list(model_dir.glob("*.cbm"))),
        "trackman": tm_audit, "subtype": subtype_audit,
        "prediction_min": float(prediction.min()), "prediction_max": float(prediction.max()),
        "elapsed_seconds": time.time() - fold_started,
    }
    atomic_json(fold_dir / "fold_metadata.json", fold_meta)
    del x3, base3, target3, frame, train_history, pitcher_snap, batter_snap, mix_snap
    gc.collect()
    return output, fold_meta


def metrics_for(frame: pd.DataFrame) -> dict[str, object]:
    result: dict[str, object] = {}
    for label, part in [(str(year), frame.loc[frame["season"].eq(year)]) for year in VALID_YEARS] + [("pooled", frame)]:
        y = part["target"].to_numpy(float)
        p = part["prediction"].to_numpy(float)
        rate = float(y.mean())
        brier = float(np.mean((y - p) ** 2))
        bss = float(1.0 - brier / (rate * (1.0 - rate)))
        result[label] = {"rows": int(len(part)), "target_rate": rate, "brier": brier, "bss": bss, "local_score": 100000.0 * bss}
    result["worst_season_bss"] = min(float(result[str(year)]["bss"]) for year in VALID_YEARS)  # type: ignore[index]
    result["worst_season_local_score"] = 100000.0 * float(result["worst_season_bss"])
    return result


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    preflight = json.loads((OUT / "preflight_report.json").read_text(encoding="utf-8"))
    if preflight.get("status") != "AUDIT_VERIFIED" or preflight.get("mismatch_count") != 0:
        raise RuntimeError("source preflight is not AUDIT_VERIFIED")
    manifest = json.loads((SOURCE_MODEL / "manifest.json").read_text(encoding="utf-8"))
    regime = {k: float(v) for k, v in json.loads((SOURCE_MODEL / "f_regime_meta.json").read_text(encoding="utf-8")).items()}
    if regime["transition_scale"] != 0.0:
        raise RuntimeError("this baseline contract requires transition_scale == 0.0")
    raw = pd.read_csv(TRAIN_PATH, low_memory=False)
    folds: list[pd.DataFrame] = []
    fold_metadata: dict[str, object] = {}
    for valid_year in VALID_YEARS:
        fold_output, metadata = fold_predictions(raw, valid_year, manifest, regime)
        fold_output.to_csv(OUT / f"oof_{valid_year}.csv", index=False)
        folds.append(fold_output)
        fold_metadata[str(valid_year)] = metadata
    oof = pd.concat(folds, ignore_index=True)
    oof_path = OUT / "oof_predictions.csv"
    oof.to_csv(oof_path, index=False)
    metrics = metrics_for(oof)
    result = {
        "experiment_id": EXPERIMENT_ID, "candidate_name": "exp030_exact_forward_oof_baseline",
        "candidate_status": "PENDING_AUDIT", "valid_years": list(VALID_YEARS),
        "source_official_score": 1068.25021, "metrics": metrics,
        "folds": fold_metadata, "actual_leaf_count": 1, "gate_checks_count": 0,
        "test_inference_performed": False, "zip_created": False,
        "elapsed_seconds": time.time() - started,
    }
    result_path = OUT / "result.json"
    atomic_json(result_path, result)
    embedded = json.dumps({"candidate_name": result["candidate_name"], "candidate_status": result["candidate_status"], "metrics": metrics}, sort_keys=True)
    lines = [
        f"# {EXPERIMENT_ID}", "", f"- candidate: `{result['candidate_name']}`",
        f"- status: `{result['candidate_status']}`", "- test inference: `false`", "- ZIP created: `false`", "",
        "| split | rows | Brier | BSS | local score |", "|---|---:|---:|---:|---:|",
    ]
    for split in ("2023", "2024", "pooled"):
        item = metrics[split]  # type: ignore[index]
        lines.append(f"| {split} | {item['rows']} | {item['brier']:.12f} | {item['bss']:.12f} | {item['local_score']:.6f} |")
    lines.extend(["", "<!-- RESULT_JSON_BEGIN", embedded, "RESULT_JSON_END -->"])
    result_md = OUT / "result.md"
    result_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    artifact_paths = [
        ROOT / "scripts" / "preflight_ref4_exact_oof_031a.py",
        ROOT / "scripts" / "run_ref4_exact_oof_031a.py",
        ROOT / "scripts" / "verify_ref4_exact_oof_031a.py",
        OUT / "preflight_report.json", OUT / "preflight_report.md",
        OUT / "oof_2023.csv", OUT / "oof_2024.csv", oof_path, result_path, result_md,
        SOURCE_MODEL / "manifest.json", SOURCE_MODEL / "f_regime_meta.json",
        ROOT / "start03_reference.md", ROOT / "01_제약과금지사항.md",
        ROOT / "output" / "submit_ref4_champion_030.zip", TRAIN_PATH, TRACKMAN_PATH,
    ]
    artifact_paths.extend(sorted(path for path in OUT.glob("fold_*/*") if path.is_file()))
    artifact_paths.extend(sorted(OUT.glob("fold_*/models/*.cbm")))
    artifacts = {}
    for path in artifact_paths:
        relative = str(path.relative_to(ROOT))
        artifacts[relative] = {"sha256": sha256_path(path), "size": path.stat().st_size}
    audit_manifest = {
        "experiment_id": EXPERIMENT_ID, "status": "PENDING_VALIDATION",
        "created_after_outputs": True, "artifact_count": len(artifacts), "artifacts": artifacts,
        "model_count": len(list(OUT.glob("fold_*/models/*.cbm"))),
        "oof_rows": int(len(oof)), "leaf_count": 1,
    }
    atomic_json(OUT / "audit_manifest.json", audit_manifest)
    print(json.dumps({
        "status": "PENDING_AUDIT", "oof_rows": len(oof),
        "model_count": audit_manifest["model_count"], "metrics": metrics,
        "elapsed_seconds": result["elapsed_seconds"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
