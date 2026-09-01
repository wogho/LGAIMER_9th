#!/usr/bin/env python3
"""Rebuild the 103A backbone as 2022-2024 forward folds for original 110 R2.

Each validation season is unseen by every fitted model and every lookup table.
The script also stores the fold's in-sample 103A anchor because 107A/108C/109C
production training used an in-sample 103A anchor for its residual target.
"""
from __future__ import annotations

import gc
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model/REF4-110-ORIGINAL-R2"
RAW_PATH = ROOT / "data/train.csv"
TM_PATH = ROOT / "data/trackman_history.csv"
BASE_OOF = ROOT / "model/REF4-OOF-DIAG-034A/diagnostic_rows.csv"
SOURCE = ROOT / "model/REF4-DEEP-HIERARCHICAL-103A/production_package"
THREADS = 4
YEARS = (2022, 2023, 2024)
DEEP_SEEDS = [42, 1, 2, 3, 4, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(SOURCE))
from src.adaptive_gate import build_gate_features  # noqa: E402
from src.entity_context_split import build_split_features  # noqa: E402
from src.preprocessing_v2 import CAT_V2, build_v2_features, build_v3_features  # noqa: E402
from src.season_delta_features import build_snapshots  # noqa: E402
from src.season_history_v3 import build_entity_snapshots  # noqa: E402
from src.v54_per_season_asof_75_features import (  # noqa: E402
    build_per_season_priors,
    build_v54_per_season_asof_75_features,
)


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PSYCH = load_module("ref4_psych_r2", ROOT / "scripts/run_ref4_trainonly_psych_latent_042a.py")


def leverage(df: pd.DataFrame) -> pd.DataFrame:
    def numeric(name: str, default: float) -> np.ndarray:
        source = df[name] if name in df.columns else pd.Series(default, index=df.index)
        return pd.to_numeric(source, errors="coerce").fillna(default).to_numpy(float)

    li = numeric("li", 0.98)
    balls = numeric("balls_before", 0.0)
    strikes = numeric("strikes_before", 0.0)
    inning = numeric("inning", 1.0)
    score_column = "score_diff" if "score_diff" in df.columns else "score_diff_pitcher_team"
    score = np.abs(numeric(score_column, 0.0))
    late_close = ((inning >= 7) & (score <= 2)).astype(float)
    return pd.DataFrame({"is_high_leverage": (li >= 1.5).astype(float), "li_count_diff": li * (balls - strikes), "li_late_close": li * late_close}, index=df.index)


def fold_source(year: int) -> tuple[Path, Path]:
    parent = ROOT / ("model/REF4-ADAPTIVE-GATE-031B" if year == 2022 else "model/REF4-EXACT-OOF-031A") / f"fold_{year}"
    return parent / "models", parent / "trackman_prior_features.csv"


def mean_reg(models: Path, stem: str, seeds: list[int], x: pd.DataFrame, base: np.ndarray, indexed: bool = False) -> np.ndarray:
    members = []
    for pos, seed in enumerate(seeds):
        name = f"{stem}_{pos}.cbm" if indexed else f"{stem}_seed{seed}.cbm"
        model = CatBoostRegressor().load_model(str(models / name))
        members.append(np.clip(base + model.predict(x), 1e-6, 1 - 1e-6))
    return np.mean(members, axis=0)


def base030_predictions(rows: pd.DataFrame, history: pd.DataFrame, year: int, manifest: dict, regime: dict) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, np.ndarray]:
    model_dir, tm_path = fold_source(year)
    prior = float(history["control_success"].mean())
    ps = build_snapshots(history)
    bs = build_entity_snapshots(history, "batter_id", "asof_batter_n", ["asof_batter_success_rate", "asof_batter_middle_rate"], "control_success")
    ms = build_entity_snapshots(history, "pitcher_id", "asof_pitcher_pitchmix_n", ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"])
    x2, base2 = build_v2_features(rows, prior, ps, str(tm_path))
    x3, base3 = build_v3_features(rows, prior, ps, bs, ms, str(tm_path))
    seeds = [int(x) for x in manifest["seeds"]]
    preds = [
        mean_reg(model_dir, "v2_decay55", seeds, x2, base2),
        mean_reg(model_dir, "v3_decay55", seeds, x3, base3),
        mean_reg(model_dir, "v3_decay30", seeds, x3, base3),
    ]
    futures = rows["game_type"].eq("F").to_numpy()
    if futures.any():
        f2 = mean_reg(model_dir, "f_v2_all", list(range(4)), x2, base2, indexed=True)
        f55 = mean_reg(model_dir, "f_v355_recent", list(range(6)), x3, base3, indexed=True)
        f30a = mean_reg(model_dir, "f_v330_all", list(range(4)), x3, base3, indexed=True)
        f30r = mean_reg(model_dir, "f_v330_recent", list(range(2)), x3, base3, indexed=True)
        preds[0] = np.where(futures, preds[0] + regime["v2_scale"] * (f2 - preds[0]), preds[0])
        preds[1] = np.where(futures, preds[1] + regime["v355_scale"] * (f55 - preds[1]), preds[1])
        inner = preds[2] + regime["v330_recent_inner_scale"] * (f30r - preds[2])
        f30 = regime["v330_all_weight"] * f30a + (1 - regime["v330_all_weight"]) * inner
        preds[2] = np.where(futures, preds[2] + regime["v330_scale"] * (f30 - preds[2]), preds[2])
    risks = []
    for name in ("middle", "wild", "reverse"):
        members = []
        for seed in seeds:
            model = CatBoostClassifier().load_model(str(model_dir / f"subtype_{name}_seed{seed}.cbm"))
            members.append(model.predict_proba(x3)[:, 1])
        risk = np.mean(members, axis=0)
        if futures.any():
            fm = CatBoostClassifier().load_model(str(model_dir / f"f_subtype_{name}.cbm"))
            fr = fm.predict_proba(x3)[:, 1]
            risk = np.where(futures, risk + regime["subtype_scale"] * (fr - risk), risk)
        risks.append(risk)
    main = np.average(np.vstack(preds), axis=0, weights=np.asarray(manifest["main_weights"], float))
    no_shift = float(manifest["stack_intercept"]) + np.column_stack([main, *risks]) @ np.asarray(manifest["stack_coefficients"], float)
    gate_x = build_gate_features(rows, preds, risks, np.clip(no_shift, 1e-6, 1 - 1e-6))
    return no_shift, np.clip(no_shift + float(manifest["global_shift"]), 1e-5, 1 - 1e-5), gate_x, base3


def standardize(train: pd.DataFrame, valid: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    mean = train.mean()
    std = train.std().replace(0, 1).fillna(1)
    return np.nan_to_num(((train - mean) / std).to_numpy(float)), np.nan_to_num(((valid - mean) / std).to_numpy(float))


def fit_backbone_fold(raw: pd.DataFrame, base_lookup: pd.DataFrame, year: int, manifest: dict, regime: dict) -> tuple[pd.DataFrame, list[Path], dict]:
    started = time.time()
    fold_dir = OUT / f"backbone_fold_{year}"
    model_dir = fold_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    history = raw.loc[raw["season"].lt(year)].reset_index(drop=True)
    valid = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
    both = pd.concat([history, valid], ignore_index=True)
    no_shift, base_all, gate_x, base3 = base030_predictions(both, history, year, manifest, regime)
    nh = len(history)
    y_hist = history["control_success"].to_numpy(float)

    # Refit the inherited adaptive gate on this fold only.
    gate = CatBoostRegressor(iterations=73, depth=3, learning_rate=0.025, loss_function="RMSE", l2_leaf_reg=30, random_strength=0.2, bootstrap_type="Bernoulli", subsample=0.8, random_seed=280033, thread_count=THREADS, allow_writing_files=False, verbose=False)
    gate.fit(gate_x.iloc[:nh], y_hist - no_shift[:nh])
    gate_path = model_dir / "adaptive_gate.cbm"
    gate.save_model(gate_path)
    p = np.clip(no_shift + 0.08 * gate.predict(gate_x) + float(manifest["global_shift"]), 1e-5, 1 - 1e-5)
    saved = [gate_path]

    # Refit F psych/latent correction using only fold history.
    mapping_path = fold_source(year)[1].parent / "pitcher_trackman_mapping.csv"
    mapping = pd.read_csv(mapping_path)
    tm = pd.read_csv(TM_PATH, usecols=["pitcher_trackman_id", "season", "balls_before", "strikes_before", "batter_hand", "pitch_type_group"])
    latent_table = PSYCH.latent_table(year, mapping, tm, 50.0, 100.0)
    psych_hist = pd.concat([PSYCH.psych(history, history, 500.0), PSYCH.latent(history, latent_table)], axis=1)
    psych_valid = pd.concat([PSYCH.psych(valid, history, 500.0), PSYCH.latent(valid, latent_table)], axis=1)
    z_hist, z_valid = standardize(psych_hist, psych_valid)
    ridge_psych = Ridge(alpha=10000.0, fit_intercept=False).fit(z_hist, y_hist - p[:nh])
    corr_hist = 0.4 * ridge_psych.predict(z_hist)
    corr_valid = 0.4 * ridge_psych.predict(z_valid)
    futures = both["game_type"].eq("F").to_numpy()
    correction = np.concatenate([corr_hist, corr_valid])
    p = np.where(futures, p + correction, p)

    # Refit the R entity-context split correction.
    split_hist = build_split_features(history, history, (50.0, 200.0, 800.0, 3200.0))
    split_valid = build_split_features(history, valid, (50.0, 200.0, 800.0, 3200.0))
    zs_hist, zs_valid = standardize(split_hist, split_valid)
    r_hist = history["game_type"].ne("F").to_numpy()
    ridge_split = Ridge(alpha=10000.0, fit_intercept=False).fit(zs_hist[r_hist], (y_hist - p[:nh])[r_hist])
    split_corr = np.concatenate([ridge_split.predict(zs_hist), ridge_split.predict(zs_valid)])
    regular = ~futures
    p_split = p + split_corr

    # Fold-local R LightGBM expert, keeping the production 0.05 blend.
    x3_hist = build_v3_features(history, float(y_hist.mean()), build_snapshots(history), build_entity_snapshots(history, "batter_id", "asof_batter_n", ["asof_batter_success_rate", "asof_batter_middle_rate"], "control_success"), build_entity_snapshots(history, "pitcher_id", "asof_pitcher_pitchmix_n", ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]), str(fold_source(year)[1]))[0]
    x3_valid = build_v3_features(valid, float(y_hist.mean()), build_snapshots(history), build_entity_snapshots(history, "batter_id", "asof_batter_n", ["asof_batter_success_rate", "asof_batter_middle_rate"], "control_success"), build_entity_snapshots(history, "pitcher_id", "asof_pitcher_pitchmix_n", ["asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"]), str(fold_source(year)[1]))[0]
    for col in CAT_V2:
        if col in x3_hist:
            x3_hist[col] = x3_hist[col].astype("category")
            x3_valid[col] = x3_valid[col].astype("category")
    lgb_r = lgb.train({"objective": "regression", "learning_rate": 0.03, "max_depth": 7, "num_leaves": 45, "seed": 42, "verbosity": -1, "num_threads": THREADS}, lgb.Dataset(x3_hist.loc[r_hist], label=(y_hist - base3[:nh] - 0.0052)[r_hist]), num_boost_round=450)
    lgb_path = model_dir / "r_expert_lgbm.txt"
    lgb_r.save_model(str(lgb_path)); saved.append(lgb_path)
    r_expert = np.concatenate([np.clip(base3[:nh] + lgb_r.predict(x3_hist) + 0.0052, 1e-6, 1 - 1e-6), np.clip(base3[nh:] + lgb_r.predict(x3_valid) + 0.0052, 1e-6, 1 - 1e-6)])
    p = np.where(regular, 0.95 * p_split + 0.05 * r_expert, p)

    # 102A 40-model deep hierarchical correction.
    priors = build_per_season_priors(history)
    team_profile = {"p_team": {str(k): float(v) for k, v in history.groupby("pitcher_team_id")["control_success"].mean().items()}, "b_team": {str(k): float(v) for k, v in history.groupby("batter_team_id")["control_success"].mean().items()}, "prior": float(y_hist.mean())}
    profile_path = fold_dir / "team_asof_profile.json"
    profile_path.write_text(json.dumps(team_profile, ensure_ascii=False), encoding="utf-8")
    hist_r = history.loc[r_hist].reset_index(drop=True)
    valid_r_mask = valid["game_type"].ne("F").to_numpy()
    valid_r = valid.loc[valid_r_mask].reset_index(drop=True)
    xdeep_h, _ = build_v54_per_season_asof_75_features(hist_r, profile_path=profile_path, priors=priors, prior=float(y_hist.mean()))
    xdeep_v, _ = build_v54_per_season_asof_75_features(valid_r, profile_path=profile_path, priors=priors, prior=float(y_hist.mean()))
    xdeep_h = pd.concat([xdeep_h, leverage(hist_r)], axis=1)
    xdeep_v = pd.concat([xdeep_v, leverage(valid_r)], axis=1)
    n = pd.to_numeric(hist_r["asof_pitcher_n"], errors="coerce").fillna(0).to_numpy(float)
    rate = pd.to_numeric(hist_r["asof_pitcher_success_rate"], errors="coerce").fillna(float(y_hist.mean())).to_numpy(float)
    prev = pd.to_numeric(hist_r["asof_pitcher_prev1_game_success_rate"], errors="coerce").to_numpy(float)
    prev = np.where(np.isfinite(prev), prev, rate)
    hb_h = np.clip(0.70 * (float(y_hist.mean()) + n / (n + 25.0) * (rate - float(y_hist.mean()))) + 0.30 * prev, 0.05, 0.95)
    target_deep = hist_r["control_success"].to_numpy(float) - hb_h
    deep_hist_members, deep_valid_members = [], []
    for seed in DEEP_SEEDS:
        cb = CatBoostRegressor(iterations=350, depth=7, learning_rate=0.04, l2_leaf_reg=5, random_seed=seed, thread_count=THREADS, allow_writing_files=False, verbose=False)
        cb.fit(xdeep_h, target_deep)
        cb_path = model_dir / f"deep_cb_l2_seed{seed}.cbm"; cb.save_model(cb_path); saved.append(cb_path)
        deep_hist_members.append(cb.predict(xdeep_h)); deep_valid_members.append(cb.predict(xdeep_v))
        booster = lgb.train({"objective": "regression", "learning_rate": 0.04, "max_depth": 7, "num_leaves": 45, "seed": seed, "verbosity": -1, "num_threads": THREADS}, lgb.Dataset(xdeep_h, label=target_deep), num_boost_round=280)
        lpath = model_dir / f"deep_lgb_l2_seed{seed}.txt"; booster.save_model(str(lpath)); saved.append(lpath)
        deep_hist_members.append(booster.predict(xdeep_h)); deep_valid_members.append(booster.predict(xdeep_v))
        del cb, booster; gc.collect()
    deep_h = np.mean(deep_hist_members, axis=0)
    deep_v = np.mean(deep_valid_members, axis=0)
    offset = float(np.mean(np.clip(hb_h + deep_h, 1e-5, 1 - 1e-5)))
    deep_corr = np.zeros(len(both), float)
    deep_corr[np.flatnonzero(r_hist)] = np.clip(hb_h + deep_h, 1e-5, 1 - 1e-5) - offset
    n_v = pd.to_numeric(valid_r["asof_pitcher_n"], errors="coerce").fillna(0).to_numpy(float)
    rate_v = pd.to_numeric(valid_r["asof_pitcher_success_rate"], errors="coerce").fillna(float(y_hist.mean())).to_numpy(float)
    prev_v = pd.to_numeric(valid_r["asof_pitcher_prev1_game_success_rate"], errors="coerce").to_numpy(float)
    prev_v = np.where(np.isfinite(prev_v), prev_v, rate_v)
    hb_v = np.clip(0.70 * (float(y_hist.mean()) + n_v / (n_v + 25.0) * (rate_v - float(y_hist.mean()))) + 0.30 * prev_v, 0.05, 0.95)
    deep_corr[nh + np.flatnonzero(valid_r_mask)] = np.clip(hb_v + deep_v, 1e-5, 1 - 1e-5) - offset
    p = np.where(regular, p + 0.08 * deep_corr, p)

    # 103A refined-call classifier.
    xcall_h = x3_hist.copy(); xcall_v = x3_valid.copy()
    cat_cols = []
    for col in CAT_V2:
        if col in xcall_h:
            xcall_h[col] = xcall_h[col].astype(str); xcall_v[col] = xcall_v[col].astype(str); cat_cols.append(col)
    succ = history["control_success"].to_numpy(int)
    rev = history.get("subtype_reverse", pd.Series(np.zeros(nh))).to_numpy(int)
    mid = history.get("subtype_middle", pd.Series(np.zeros(nh))).to_numpy(int)
    wild = history.get("subtype_wild", pd.Series(np.zeros(nh))).to_numpy(int)
    call = np.zeros(nh, int)
    call = np.where((succ == 0) & (rev == 1), 3, call); call = np.where((succ == 0) & (mid == 1), 4, call); call = np.where((succ == 0) & (wild == 1), 5, call); call = np.where((succ == 0) & (rev == 0) & (mid == 0) & (wild == 0), 1, call)
    call_model = CatBoostClassifier(iterations=300, depth=6, learning_rate=0.04, loss_function="MultiClass", cat_features=cat_cols, random_seed=42, thread_count=THREADS, allow_writing_files=False, verbose=False)
    call_model.fit(xcall_h.loc[r_hist], call[r_hist])
    cpath = model_dir / "refined_call_expert.cbm"; call_model.save_model(cpath); saved.append(cpath)
    call_h = call_model.predict_proba(xcall_h)[:, 0]; call_v = call_model.predict_proba(xcall_v)[:, 0]
    call_offset = float(call_h[r_hist].mean())
    p += np.where(regular, 0.04 * (np.concatenate([call_h, call_v]) - call_offset), 0.0)

    n_all = pd.to_numeric(both["asof_pitcher_n"], errors="coerce").fillna(0).to_numpy(float)
    rate_all = pd.to_numeric(both["asof_pitcher_success_rate"], errors="coerce").fillna(float(y_hist.mean())).to_numpy(float)
    p = np.where(regular & (n_all < 15) & (rate_all > 0.65), p - 0.012, p)
    p = np.where(regular & (n_all < 15) & (rate_all < 0.40), p + 0.008, p)
    p = np.clip(p, 1e-5, 1 - 1e-5)
    out = pd.DataFrame({"row_id": both["row_id"].astype(str), "season": both["season"].astype(int), "is_validation": np.arange(len(both)) >= nh, "target": both["control_success"].to_numpy(float), "p103a": p})
    out.to_csv(fold_dir / "anchor_predictions.csv", index=False)
    meta = {"validation_year": year, "train_seasons": sorted(history["season"].unique().astype(int).tolist()), "train_rows": nh, "valid_rows": len(valid), "validation_labels_used_in_fit": False, "model_count": len(saved), "deep_mean_offset": offset, "call_mean_offset": call_offset, "elapsed_seconds": time.time() - started}
    (fold_dir / "provenance.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return out, saved, meta


def main() -> None:
    pre = json.loads((OUT / "preflight_report.json").read_text(encoding="utf-8"))
    if pre.get("status") != "AUDIT_VERIFIED" or pre.get("mismatch_count") != 0:
        raise RuntimeError("R2 preflight is not AUDIT_VERIFIED")
    raw = pd.read_csv(RAW_PATH, low_memory=False)
    base = pd.read_csv(BASE_OOF, dtype={"row_id": str})
    manifest = json.loads((ROOT / "model/REF4-CHAMPION-STACK-030/manifest.json").read_text())
    regime = json.loads((ROOT / "model/REF4-CHAMPION-STACK-030/f_regime_meta.json").read_text())
    parts, all_models, provenance = [], [], {}
    for year in YEARS:
        print(f"[{year}] rebuilding strict 103A backbone", flush=True)
        fold, models, meta = fit_backbone_fold(raw, base, year, manifest, regime)
        parts.append(fold.loc[fold["is_validation"]].copy())
        all_models.extend(models); provenance[str(year)] = meta
    oof = pd.concat(parts, ignore_index=True)
    oof.to_csv(OUT / "p103a_oof.csv", index=False)
    summary = {"status": "PENDING_VALIDATION", "oof_rows": len(oof), "model_count": len(all_models), "folds": provenance, "prediction_min": float(oof["p103a"].min()), "prediction_max": float(oof["p103a"].max())}
    (OUT / "p103a_build_result.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
