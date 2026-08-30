#!/usr/bin/env python3
"""Train-only temporal OOF screen for the reference repo's residual-3 idea.

No leaderboard values, test rows, or external labels are read.  The output is
research-only until the independent audit is complete.
"""
from __future__ import annotations

import gc, hashlib, json, sys, time
from pathlib import Path
import catboost as cb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_combo_full_candidate_002 import COLS as BASE_COLS, extras, prep
from scripts.screen_trackman_context_003 import load_tm, context_tables, attach, NEW
from src.asof_state_features import add_state_for_cutoff, add_state_walkforward
from src.target_aggregates import build_pitcher_count_state_target_history

OUT = ROOT / "model" / "COMBO-RESID3-OOF-007"
COLS = BASE_COLS + NEW

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def compact(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce research-only working-set size without changing values."""
    for col in df.select_dtypes(include=["integer"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in df.select_dtypes(include=["floating"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["object"]).columns:
        # Low-cardinality strings only; IDs remain numeric in the source.
        if df[col].nunique(dropna=False) < min(512, max(32, len(df) // 1000)):
            df[col] = df[col].astype("category")
    return df

def make_features(frame: pd.DataFrame, history: pd.DataFrame, tm, season: int, train_mode: bool) -> pd.DataFrame:
    x = pd.concat([extras(frame.reset_index(drop=True)), history.reset_index(drop=True)], axis=1)
    if train_mode:
        parts = []
        for s in sorted(frame.season.unique()):
            ix = frame.season.eq(s)
            parts.append(attach(x.loc[ix.to_numpy()].copy(), tm, int(s)))
        x = pd.concat(parts, axis=0).sort_index()
    else:
        count, hand = context_tables(tm, season)
        x = x.copy()
        x["__hand"] = x["batter_hand"].map({1: "Left", 2: "Right"}).fillna(x["batter_hand"].astype(str))
        x = x.join(count, on=["pitcher_id", "balls_before", "strikes_before"])
        x = x.join(hand, on=["pitcher_id", "__hand"]).drop(columns="__hand")
    x = x[COLS]
    x, cats = prep(x)
    return x, cats

def main() -> None:
    start = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    raw = compact(pd.read_csv(ROOT / "data" / "train.csv", encoding="utf-8-sig"))
    tm, mapping = load_tm()
    tm = compact(tm)
    seasons = [2022, 2023, 2024]
    frames = []
    fold_meta = []
    for season in seasons:
        train = raw.loc[raw.season < season].copy()
        valid = raw.loc[raw.season == season].copy()
        train = compact(train)
        valid = compact(valid)
        if train.empty or valid.empty:
            raise RuntimeError(f"empty temporal fold: {season}")
        train_base = add_state_walkforward(train.drop(columns=["row_id", "control_success"]), season)
        valid_base = add_state_for_cutoff(valid.drop(columns=["row_id", "control_success"]), train.drop(columns=["row_id", "control_success"]))
        train_hist, valid_hist, lookup, _ = build_pitcher_count_state_target_history(
            train, valid.assign(control_success=0), smoothing=100.0
        )
        xtr, cats = make_features(train_base, train_hist, tm, season, True)
        xva, _ = make_features(valid_base, valid_hist, tm, season, False)
        model = cb.CatBoostClassifier(
            iterations=300, learning_rate=0.05, depth=6, l2_leaf_reg=5,
            loss_function="Logloss", thread_count=-1, random_seed=42,
            allow_writing_files=False, verbose=False,
        )
        model.fit(cb.Pool(xtr, label=train.control_success.to_numpy(np.int8), cat_features=cats, feature_names=COLS))
        pred = model.predict_proba(cb.Pool(xva, cat_features=cats, feature_names=COLS))[:, 1]
        if len(pred) != len(valid) or not np.isfinite(pred).all() or not ((pred >= 0) & (pred <= 1)).all():
            raise RuntimeError(f"prediction contract failed: {season}")
        frames.append(pd.DataFrame({"row_id": valid.row_id.to_numpy(), "season": season, "pitcher_id": valid.pitcher_id.to_numpy(), "pitcher_hand": valid.pitcher_hand.to_numpy(), "batter_hand": valid.batter_hand.to_numpy(), "strikes_before": valid.strikes_before.to_numpy(), "num_runners_on": valid.num_runners_on.to_numpy(), "target": valid.control_success.to_numpy(np.int8), "pred": pred}))
        fold_meta.append({"season": season, "train_rows": len(train), "valid_rows": len(valid), "feature_count": len(COLS), "tree_count": model.tree_count_, "lookup_rows": len(lookup)})
        print(json.dumps(fold_meta[-1], ensure_ascii=False), flush=True)
        del train, valid, train_base, valid_base, train_hist, valid_hist, xtr, xva, model, pred
        gc.collect()
    oof = pd.concat(frames, ignore_index=True).sort_values("row_id")
    if oof.row_id.duplicated().any() or len(oof) != int(raw[raw.season.isin(seasons)].shape[0]):
        raise RuntimeError("OOF row coverage/uniqueness failed")
    oof.to_csv(OUT / "oof_predictions.csv", index=False)
    manifest = {"experiment_id": "COMBO-RESID3-OOF-007", "source_train": str(ROOT / "data" / "train.csv"), "source_train_sha256": sha256(ROOT / "data" / "train.csv"), "trackman_source": str(ROOT / "data" / "trackman_history.csv"), "trackman_sha256": sha256(ROOT / "data" / "trackman_history.csv"), "official_train_only": True, "eval_seasons": seasons, "feature_count": len(COLS), "oof_rows": len(oof), "row_id_unique": True, "prediction_range": [float(oof.pred.min()), float(oof.pred.max())], "folds": fold_meta, "status": "OOF_READY_RESIDUAL_SCREEN_PENDING", "elapsed_sec": round(time.time() - start, 1)}
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
