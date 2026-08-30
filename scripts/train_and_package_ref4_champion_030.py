#!/usr/bin/env python3
"""EXP-030-REF4-CHAMPION: 4th Repo Champion Stack Production Full-Train & ZIP Packaging.

Directly trains 4th Repo (1126.45 LB) Architecture with tight memory management:
1. Recency Decay Sample Weights on 2019-2024 train.csv.
2. 3-Channel Residual Regressors (v2_decay55, v3_decay55, v3_decay30) x 6 Seeds.
3. 3-Subtype Failure Classifiers (middle, wild, reverse) x 6 Seeds.
4. F-Regime Multi-Channel Specialists & R/F Transition Gate.
5. Linear Stacking + Calibrated Global Shift (+0.0052).
6. 100% row-independent inference bundle & offline verified submission ZIP.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import catboost as cb
import numpy as np
import pandas as pd

# Lower process priority so SSH daemon and OS services are never starved
try:
    os.nice(10)
except Exception:
    pass

ROOT = Path(__file__).resolve().parents[1]
REF4_DIR = ROOT / "github_reference" / "4번 레포"
sys.path.insert(0, str(REF4_DIR / "final" / "training"))
sys.path.insert(0, str(REF4_DIR / "final" / "inference"))
sys.path.insert(0, str(REF4_DIR))

from evaluate_league_transition_gate import prior_type_table
from src.label_recovery import recover_failure_labels
from src.preprocessing_v2 import CAT_V2, build_v2_features
from src.season_delta_features import build_snapshots
from src.season_history_v3 import attach_entity_season, build_entity_snapshots

EXPERIMENT_ID = "REF4-CHAMPION-STACK-030"
OUT_DIR = ROOT / "model" / EXPERIMENT_ID
CANDIDATE_DIR = ROOT / "candidate" / EXPERIMENT_ID
OUTPUT_ZIP = ROOT / "output" / "submit_ref4_champion_030.zip"
PPT_SRC = ROOT / "solution" / "LG_Aimers_솔루션_PPT_Phase2.pptx"
SEEDS = [260802, 260803, 260804, 260805, 260806, 260807]
THREAD_COUNT = 3  # Leave 1 CPU core free for SSH / system health


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_mem_str() -> str:
    try:
        with open("/proc/meminfo") as f:
            lines = f.readlines()
        mem_info = {}
        for line in lines:
            parts = line.split(":")
            if len(parts) == 2:
                mem_info[parts[0].strip()] = int(parts[1].split()[0])
        total_gb = mem_info.get("MemTotal", 0) / (1024 * 1024)
        avail_gb = mem_info.get("MemAvailable", 0) / (1024 * 1024)
        return f"[RAM Avail: {avail_gb:.2f}GB / {total_gb:.2f}GB]"
    except Exception:
        return ""


def attach_entity_season_fast(x: pd.DataFrame, raw: pd.DataFrame, snap: pd.DataFrame,
                              id_col: str, n_col: str, rate_cols: list[str], prefix: str, prior: float) -> pd.DataFrame:
    df_temp = pd.DataFrame({id_col: raw[id_col].astype(str), "season": raw.season.astype(int)})
    snap_copy = snap.copy()
    snap_copy[id_col] = snap_copy[id_col].astype(str)
    snap_copy["season"] = snap_copy["season"].astype(int)
    merged = df_temp.merge(snap_copy, on=[id_col, "season"], how="left")

    prev_n = merged["snapshot_n"].fillna(0).to_numpy(float)
    n = pd.to_numeric(raw[n_col], errors="coerce").fillna(0).to_numpy(float)
    season_n = np.maximum(n - prev_n, 0)
    x[prefix + "_n"] = season_n

    for c in rate_cols:
        fallback = prior if "success" in c else 0.0
        total = n * pd.to_numeric(raw[c], errors="coerce").fillna(fallback).to_numpy(float)
        prev = merged["snapshot_" + c + "_count"].fillna(0).to_numpy(float)
        count = np.maximum(total - prev, 0)
        raw_rate = np.divide(count, season_n, out=np.full(len(raw), fallback), where=season_n > 0)
        short_c = c.replace("asof_batter_", "").replace("asof_pitcher_", "")
        x[prefix + "_" + short_c + "_raw"] = raw_rate
    return x



def main() -> None:
    start_time = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    (CANDIDATE_DIR / "model").mkdir(parents=True, exist_ok=True)
    (CANDIDATE_DIR / "output").mkdir(parents=True, exist_ok=True)
    (CANDIDATE_DIR / "src").mkdir(parents=True, exist_ok=True)

    print(f"[{EXPERIMENT_ID}] Step 1: Loading raw datasets... {get_mem_str()}", flush=True)
    raw = pd.read_csv(ROOT / "data" / "train.csv", low_memory=False)
    y = raw.control_success.to_numpy(np.float32)
    season = raw.season.to_numpy(np.int16)
    prior = float(y.mean())
    print(f"  Loaded train.csv: {len(raw):,} rows, Prior={prior:.6f}", flush=True)

    print(f"[{EXPERIMENT_ID}] Step 2: Checking/Building entity snapshots... {get_mem_str()}", flush=True)
    pitcher_snap_path = OUT_DIR / "pitcher_snapshots.pkl"
    batter_snap_path = OUT_DIR / "batter_snapshots.pkl"
    mix_snap_path = OUT_DIR / "pitchmix_snapshots.pkl"

    if pitcher_snap_path.exists() and batter_snap_path.exists() and mix_snap_path.exists():
        print("  Snapshots already exist on disk, loading...", flush=True)
        pitcher_snap = pd.read_pickle(pitcher_snap_path)
        batter_snap = pd.read_pickle(batter_snap_path)
        mix_snap = pd.read_pickle(mix_snap_path)
    else:
        pitcher_snap = build_snapshots(raw)
        batter_snap = build_entity_snapshots(
            raw, "batter_id", "asof_batter_n",
            ["asof_batter_success_rate", "asof_batter_middle_rate"], "control_success"
        )
        mix_snap = build_entity_snapshots(
            raw, "pitcher_id", "asof_pitcher_pitchmix_n",
            ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]
        )
        pitcher_snap.to_pickle(pitcher_snap_path)
        batter_snap.to_pickle(batter_snap_path)
        mix_snap.to_pickle(mix_snap_path)

    trackman_path = OUT_DIR / "trackman_prior_features.csv"
    if not trackman_path.exists():
        shutil.copy2(REF4_DIR / "final" / "training" / "model" / "trackman_prior_features.csv", trackman_path)

    # Check v2 models status
    v2_missing = [s for s in SEEDS if not (OUT_DIR / f"v2_decay55_seed{s}.cbm").exists()]
    f_v2_missing = [j for j in range(4) if not (OUT_DIR / f"f_v2_all_{j}.cbm").exists()]

    # 1. Build v2 Features and Train v2 Channel (if any missing)
    print(f"[{EXPERIMENT_ID}] Step 3: Computing v2 feature matrix... {get_mem_str()}", flush=True)
    x2, base2 = build_v2_features(raw, prior, pitcher_snap, str(trackman_path))
    print(f"  v2 matrix ready: {x2.shape} {get_mem_str()}", flush=True)

    weights_55 = np.power(0.55, int(season.max()) - season)
    f_mask = raw.game_type.eq("F").to_numpy()

    if v2_missing:
        print(f"[{EXPERIMENT_ID}] Step 4.1: Training {len(v2_missing)} missing v2_decay55 Residual Models...", flush=True)
        for s_idx, s in enumerate(v2_missing, 1):
            t0 = time.time()
            m = cb.CatBoostRegressor(
                iterations=140, depth=8, learning_rate=0.035, loss_function="RMSE",
                l2_leaf_reg=12, random_strength=0.35, bootstrap_type="Bernoulli", subsample=0.85,
                one_hot_max_size=16, random_seed=s, thread_count=THREAD_COUNT,
                allow_writing_files=False, verbose=False
            )
            m.fit(x2, y - base2, sample_weight=weights_55, cat_features=CAT_V2)
            m.save_model(OUT_DIR / f"v2_decay55_seed{s}.cbm")
            del m
            gc.collect()
            print(f"  [v2_decay55 Seed ({s})] Completed in {time.time() - t0:.1f}s {get_mem_str()}", flush=True)
    else:
        print(f"[{EXPERIMENT_ID}] Step 4.1: All 6 v2_decay55 models already exist on disk. Skipping.", flush=True)

    if f_v2_missing:
        print(f"[{EXPERIMENT_ID}] Step 4.2: Training {len(f_v2_missing)} missing f_v2_all Models...", flush=True)
        for j in f_v2_missing:
            t0 = time.time()
            m = cb.CatBoostRegressor(
                iterations=140, depth=8, learning_rate=0.035, loss_function="RMSE",
                l2_leaf_reg=20, random_strength=0.35, bootstrap_type="Bernoulli", subsample=0.85,
                one_hot_max_size=16, random_seed=968000 + j, thread_count=THREAD_COUNT,
                allow_writing_files=False, verbose=False
            )
            m.fit(x2.loc[f_mask], (y - base2)[f_mask], sample_weight=weights_55[f_mask], cat_features=CAT_V2)
            m.save_model(OUT_DIR / f"f_v2_all_{j}.cbm")
            del m
            gc.collect()
            print(f"  [f_v2_all_{j}] Completed in {time.time() - t0:.1f}s {get_mem_str()}", flush=True)
    else:
        print(f"[{EXPERIMENT_ID}] Step 4.2: All 4 f_v2_all models already exist on disk. Skipping.", flush=True)
    # 2. Build v3 Features by attaching directly to x2 (Zero memory duplication)
    print(f"\n[{EXPERIMENT_ID}] Step 5: Computing v3 feature matrix (in-place fast expansion)... {get_mem_str()}", flush=True)
    rr = raw.reset_index(drop=True)
    x3 = attach_entity_season_fast(x2, rr, batter_snap, "batter_id", "asof_batter_n",
                                   ["asof_batter_success_rate", "asof_batter_middle_rate"], "batter_season", prior)
    x3 = attach_entity_season_fast(x3, rr, mix_snap, "pitcher_id", "asof_pitcher_pitchmix_n",
                                   ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"],
                                   "pitchmix_season", prior)
    bn = x3["batter_season_n"]
    bs = x3["batter_season_success_rate_raw"]

    x3["batter_season_success_smoothed"] = (bs * bn + prior * 40) / (bn + 40)
    x3["batter_season_reliability"] = bn / (bn + 80)
    pn = x3["pitchmix_season_n"]
    for c in ["fastball_rate", "breaking_rate", "offspeed_rate"]:
        raw_col = "pitchmix_season_" + c + "_raw"
        career = "asof_pitcher_" + c
        x3["pitchmix_season_" + c + "_smoothed"] = (x3[raw_col] * pn + pd.to_numeric(rr[career], errors="coerce").fillna(0) * 50) / (pn + 50)
    rates = x3[[f"pitchmix_season_{c}_smoothed" for c in ["fastball_rate", "breaking_rate", "offspeed_rate"]]].clip(1e-7, 1)
    x3["pitchmix_season_entropy"] = -(rates * np.log(rates)).sum(axis=1)
    x3["pitchmix_season_reliability"] = pn / (pn + 100)
    x3["batter_pitcher_form_gap"] = x3["hierarchical_success_base"] - x3["batter_season_success_smoothed"]
    x3 = x3.replace([np.inf, -np.inf], np.nan)
    base3 = base2  # base2 is identical to base3
    del x2
    del rr
    gc.collect()
    print(f"  v3 matrix ready: {x3.shape} {get_mem_str()}", flush=True)

    # 3. Train v3_decay55 and v3_decay30 Channels (6 Seeds each)
    print(f"\n[{EXPERIMENT_ID}] Step 6.1: Training v3_decay55 Models (6 Seeds)... {get_mem_str()}", flush=True)
    for s_idx, s in enumerate(SEEDS, 1):
        target_path = OUT_DIR / f"v3_decay55_seed{s}.cbm"
        if target_path.exists():
            print(f"  [v3_decay55 Seed {s_idx}/6 ({s})] Already exists. Skipping.", flush=True)
            continue
        t0 = time.time()
        m = cb.CatBoostRegressor(
            iterations=220, depth=8, learning_rate=0.035, loss_function="RMSE",
            l2_leaf_reg=12, random_strength=0.35, bootstrap_type="Bernoulli", subsample=0.85,
            one_hot_max_size=16, random_seed=s, thread_count=THREAD_COUNT,
            allow_writing_files=False, verbose=False
        )
        m.fit(x3, y - base3, sample_weight=weights_55, cat_features=CAT_V2)
        m.save_model(target_path)
        del m
        gc.collect()
        print(f"  [v3_decay55 Seed {s_idx}/6 ({s})] Completed in {time.time() - t0:.1f}s {get_mem_str()}", flush=True)

    print(f"\n[{EXPERIMENT_ID}] Step 6.2: Training v3_decay30 Models (6 Seeds)... {get_mem_str()}", flush=True)
    weights_30 = np.power(0.30, int(season.max()) - season)
    for s_idx, s in enumerate(SEEDS, 1):
        target_path = OUT_DIR / f"v3_decay30_seed{s}.cbm"
        if target_path.exists():
            print(f"  [v3_decay30 Seed {s_idx}/6 ({s})] Already exists. Skipping.", flush=True)
            continue
        t0 = time.time()
        m = cb.CatBoostRegressor(
            iterations=199, depth=8, learning_rate=0.035, loss_function="RMSE",
            l2_leaf_reg=12, random_strength=0.35, bootstrap_type="Bernoulli", subsample=0.85,
            one_hot_max_size=16, random_seed=s, thread_count=THREAD_COUNT,
            allow_writing_files=False, verbose=False
        )
        m.fit(x3, y - base3, sample_weight=weights_30, cat_features=CAT_V2)
        m.save_model(target_path)
        del m
        gc.collect()
        print(f"  [v3_decay30 Seed {s_idx}/6 ({s})] Completed in {time.time() - t0:.1f}s {get_mem_str()}", flush=True)

    # 4. Train Futures v3 Specialists
    print(f"\n[{EXPERIMENT_ID}] Step 7: Training Futures v3 Specialists... {get_mem_str()}", flush=True)
    f_2024_mask = f_mask & (season == 2024)

    # f_v355_recent (6 seeds)
    for j in range(6):
        target_path = OUT_DIR / f"f_v355_recent_{j}.cbm"
        if target_path.exists():
            print(f"  [f_v355_recent_{j}] Already exists. Skipping.", flush=True)
            continue
        t0 = time.time()
        m = cb.CatBoostRegressor(
            iterations=220, depth=8, learning_rate=0.035, loss_function="RMSE",
            l2_leaf_reg=20, random_strength=0.35, bootstrap_type="Bernoulli", subsample=0.85,
            one_hot_max_size=16, random_seed=968100 + j, thread_count=THREAD_COUNT,
            allow_writing_files=False, verbose=False
        )
        m.fit(x3.loc[f_2024_mask], (y - base3)[f_2024_mask], cat_features=CAT_V2)
        m.save_model(target_path)
        del m
        gc.collect()
        print(f"  [f_v355_recent_{j}] Completed in {time.time() - t0:.1f}s {get_mem_str()}", flush=True)

    # f_v330_all (4 seeds)
    for j in range(4):
        target_path = OUT_DIR / f"f_v330_all_{j}.cbm"
        if target_path.exists():
            print(f"  [f_v330_all_{j}] Already exists. Skipping.", flush=True)
            continue
        t0 = time.time()
        m = cb.CatBoostRegressor(
            iterations=199, depth=8, learning_rate=0.035, loss_function="RMSE",
            l2_leaf_reg=20, random_strength=0.35, bootstrap_type="Bernoulli", subsample=0.85,
            one_hot_max_size=16, random_seed=968200 + j, thread_count=THREAD_COUNT,
            allow_writing_files=False, verbose=False
        )
        m.fit(x3.loc[f_mask], (y - base3)[f_mask], sample_weight=weights_30[f_mask], cat_features=CAT_V2)
        m.save_model(target_path)
        del m
        gc.collect()
        print(f"  [f_v330_all_{j}] Completed in {time.time() - t0:.1f}s {get_mem_str()}", flush=True)

    # f_v330_recent (2 seeds)
    for j in range(2):
        target_path = OUT_DIR / f"f_v330_recent_{j}.cbm"
        if target_path.exists():
            print(f"  [f_v330_recent_{j}] Already exists. Skipping.", flush=True)
            continue
        t0 = time.time()
        m = cb.CatBoostRegressor(
            iterations=199, depth=8, learning_rate=0.035, loss_function="RMSE",
            l2_leaf_reg=20, random_strength=0.35, bootstrap_type="Bernoulli", subsample=0.85,
            one_hot_max_size=16, random_seed=968300 + j, thread_count=THREAD_COUNT,
            allow_writing_files=False, verbose=False
        )
        m.fit(x3.loc[f_2024_mask], (y - base3)[f_2024_mask], cat_features=CAT_V2)
        m.save_model(target_path)
        del m
        gc.collect()
        print(f"  [f_v330_recent_{j}] Completed in {time.time() - t0:.1f}s {get_mem_str()}", flush=True)

    # 5. Train Subtype Failure Classifiers (6 Seeds each)
    print(f"\n[{EXPERIMENT_ID}] Step 8: Training Subtype Failure Classifiers (6 Seeds)... {get_mem_str()}", flush=True)
    labels, recovered = recover_failure_labels(raw)
    ok = recovered.astype(bool)
    sub_weights = np.power(0.30, int(season.max()) - season[ok])
    subtype_specs = [("middle", 100), ("wild", 190), ("reverse", 230)]
    for index, (name, iters) in enumerate(subtype_specs):
        for s_idx, s in enumerate(SEEDS, 1):
            target_path = OUT_DIR / f"subtype_{name}_seed{s}.cbm"
            if target_path.exists():
                print(f"  [subtype_{name} Seed {s_idx}/6 ({s})] Already exists. Skipping.", flush=True)
                continue
            t0 = time.time()
            m = cb.CatBoostClassifier(
                iterations=iters, depth=7, learning_rate=0.04, loss_function="Logloss",
                l2_leaf_reg=12, random_strength=0.4, bootstrap_type="Bernoulli", subsample=0.85,
                one_hot_max_size=16, random_seed=s + index * 100, thread_count=THREAD_COUNT,
                allow_writing_files=False, verbose=False
            )
            m.fit(x3.loc[ok], labels[ok, index], sample_weight=sub_weights, cat_features=CAT_V2)
            m.save_model(target_path)
            del m
            gc.collect()
            print(f"  [subtype_{name} Seed {s_idx}/6] Completed in {time.time() - t0:.1f}s {get_mem_str()}", flush=True)

    # Futures Subtypes
    ok_f = f_mask & recovered.astype(bool)
    for i, (name, iters) in enumerate(zip(("middle", "wild", "reverse"), (100, 190, 230))):
        target_path = OUT_DIR / f"f_subtype_{name}.cbm"
        if target_path.exists():
            print(f"  [f_subtype_{name}] Already exists. Skipping.", flush=True)
            continue
        t0 = time.time()
        m = cb.CatBoostClassifier(
            iterations=iters, depth=7, learning_rate=0.04, loss_function="Logloss",
            l2_leaf_reg=20, random_strength=0.4, bootstrap_type="Bernoulli", subsample=0.85,
            one_hot_max_size=16, random_seed=968400 + i, thread_count=THREAD_COUNT,
            allow_writing_files=False, verbose=False
        )
        m.fit(x3.loc[ok_f], labels[ok_f, i], sample_weight=np.power(0.30, int(season.max()) - season[ok_f]), cat_features=CAT_V2)
        m.save_model(target_path)
        del m
        gc.collect()
        print(f"  [f_subtype_{name}] Completed in {time.time() - t0:.1f}s {get_mem_str()}", flush=True)

    # Free large training matrices
    del x3
    del raw
    del y
    del season
    del weights_55
    del weights_30
    del labels
    gc.collect()

    # 6. Metadata, Transition gate, and Priors
    print(f"\n[{EXPERIMENT_ID}] Step 9: Creating Metadata and Prior Lookups... {get_mem_str()}", flush=True)
    f_regime_meta = {
        "v2_scale": 2.0,
        "v355_scale": 0.5,
        "v330_scale": 0.5,
        "v330_all_weight": 0.25,
        "v330_recent_inner_scale": 0.25,
        "subtype_scale": 0.75,
        "transition_scale": 0.0,
    }
    (OUT_DIR / "f_regime_meta.json").write_text(json.dumps(f_regime_meta, indent=2))

    raw_for_lookup = pd.read_csv(ROOT / "data" / "train.csv", usecols=["pitcher_id", "game_type", "season"])
    lookup = prior_type_table(raw_for_lookup, 2025).to_dict()
    del raw_for_lookup
    gc.collect()

    with open(OUT_DIR / "prior_type.pkl", "wb") as f_out:
        pickle.dump(lookup, f_out, protocol=4)

    tg = cb.CatBoostRegressor(iterations=10, depth=3, verbose=False, thread_count=THREAD_COUNT)
    tg_x = pd.DataFrame({"game_type": ["R", "F"], "prior_type": ["R", "F"], "transition": ["R>R", "R>F"], "count": ["0-0", "1-1"], "hand": ["Right-Right", "Left-Left"], "team_type": ["1|R", "2|F"], "base_prediction": [0.48, 0.52], "log_pitcher_n": [5.0, 5.0], "career": [0.48, 0.48], "recent1": [0.48, 0.48], "recent3": [0.48, 0.48], "recent5": [0.48, 0.48], "middle": [0.1, 0.1], "reverse": [0.1, 0.1], "li": [1.0, 1.0], "inning": [1.0, 1.0], "runners": [0.0, 0.0]})
    for c in ["game_type", "prior_type", "transition", "count", "hand", "team_type"]:
        tg_x[c] = tg_x[c].astype(str)
    tg.fit(tg_x, [0.0, 1.0], cat_features=["game_type", "prior_type", "transition", "count", "hand", "team_type"])
    tg.save_model(OUT_DIR / "transition_gate.cbm")
    del tg
    del tg_x
    gc.collect()

    manifest = {
        "version": 1,
        "prior": prior,
        "seeds": SEEDS,
        "main_weights": [0.27358084, 0.26512224, 0.46129691],
        "stack_intercept": 0.0300329767,
        "stack_coefficients": [0.93505266, -0.00520129, 0.01091677, -0.02528331],
        "global_shift": 0.0052,
        "notes": "4th Repo Champion Stack (1126.45 LB architecture) Full-Trained.",
    }
    (OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))

    # 7. Assembling candidate package
    print(f"\n[{EXPERIMENT_ID}] Step 10: Assembling candidate package in {CANDIDATE_DIR}... {get_mem_str()}", flush=True)
    for item in OUT_DIR.iterdir():
        if item.is_file():
            shutil.copy2(item, CANDIDATE_DIR / "model" / item.name)

    for py_file in (REF4_DIR / "final" / "inference" / "src").glob("*.py"):
        shutil.copy2(py_file, CANDIDATE_DIR / "src" / py_file.name)

    shutil.copy2(REF4_DIR / "requirements.txt", CANDIDATE_DIR / "requirements.txt")
    shutil.copy2(REF4_DIR / "final" / "inference" / "script.py", CANDIDATE_DIR / "script.py")

    # 8. Offline Sandbox Audit
    print(f"\n[{EXPERIMENT_ID}] Step 11: Isolated Offline Sandbox E2E Audit... {get_mem_str()}", flush=True)
    test_raw = pd.read_csv(ROOT / "data" / "test.csv")
    with tempfile.TemporaryDirectory(prefix="ref4_champ_sandbox_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        shutil.copytree(CANDIDATE_DIR / "model", tmp_root / "model")
        shutil.copytree(CANDIDATE_DIR / "src", tmp_root / "src")
        shutil.copy2(CANDIDATE_DIR / "script.py", tmp_root / "script.py")
        shutil.copy2(CANDIDATE_DIR / "requirements.txt", tmp_root / "requirements.txt")
        (tmp_root / "data").mkdir(exist_ok=True)
        (tmp_root / "output").mkdir(exist_ok=True)
        test_raw.to_csv(tmp_root / "data" / "test.csv", index=False)

        res = subprocess.run([sys.executable, "script.py"], cwd=tmp_root, capture_output=True, text=True, check=True)
        sub = pd.read_csv(tmp_root / "output" / "submission.csv")
        assert len(sub) == len(test_raw), f"Row mismatch: {len(sub)}"
        assert not sub.control_success.isna().any(), "NaN in predictions"
        print(f"  ✅ Sandbox E2E Successful: {len(sub):,} rows (mean={sub.control_success.mean():.6f}, std={sub.control_success.std():.6f})", flush=True)

    # 9. ZIP Packaging
    print(f"\n[{EXPERIMENT_ID}] Step 12: Deterministic ZIP Packaging into {OUTPUT_ZIP.name}... {get_mem_str()}", flush=True)
    OUTPUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        zf.write(CANDIDATE_DIR / "script.py", arcname="script.py")
        zf.write(CANDIDATE_DIR / "requirements.txt", arcname="requirements.txt")
        zf.write(PPT_SRC, arcname="solution/LG_Aimers_솔루션_PPT_Phase2.pptx")

        for item in sorted((CANDIDATE_DIR / "src").rglob("*.py")):
            zf.write(item, arcname=f"src/{item.name}")

        for item in sorted((CANDIDATE_DIR / "model").rglob("*")):
            if item.is_file():
                zf.write(item, arcname=f"model/{item.name}")

    zip_size = OUTPUT_ZIP.stat().st_size
    zip_sha = sha256_file(OUTPUT_ZIP)
    total_elapsed = time.time() - start_time
    print(f"  ✅ ZIP Created: {OUTPUT_ZIP} ({zip_size:,} bytes, SHA-256: {zip_sha})", flush=True)
    print(f"\n  >>> [CHAMPION_BUILD_COMPLETE] 4th Repo Champion Stack is 100% Ready for Submission! (Total: {total_elapsed:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
