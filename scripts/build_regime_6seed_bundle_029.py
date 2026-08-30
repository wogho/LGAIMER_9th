#!/usr/bin/env python3
"""Build candidate inference package for REGIME-6SEED-FULL-029."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC_019_DIR = ROOT / "model" / "REGIME-RCAPACITY-FULL-019"
SRC_029_DIR = ROOT / "model" / "REGIME-6SEED-FULL-029"
CANDIDATE_DIR = ROOT / "candidate" / "REGIME-6SEED-FULL-029"
MODEL_DEST = CANDIDATE_DIR / "model"
SEEDS = [42, 7, 2024, 99, 1, 123]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    print(f"[BUILD-029] Building candidate package in: {CANDIDATE_DIR}")
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DEST.mkdir(parents=True, exist_ok=True)
    (CANDIDATE_DIR / "data").mkdir(exist_ok=True)
    (CANDIDATE_DIR / "output").mkdir(exist_ok=True)

    # 1. Copy priors and lookups from 019
    priors_to_copy = [
        "feature_columns.json",
        "asof_pitcher_id_prior.csv",
        "asof_batter_id_prior.csv",
        "asof_pitchmix_prior.csv",
        "pitcher_count_lookup.csv",
        "trackman_count_lookup.csv",
        "trackman_hand_lookup.csv",
        "pitcher_id_map_audit.csv",
    ]
    for filename in priors_to_copy:
        shutil.copy2(SRC_019_DIR / filename, MODEL_DEST / filename)

    # 2. Copy 18 CatBoost models from 029
    for s in SEEDS:
        shutil.copy2(SRC_029_DIR / f"baseline_combo_seed_{s}.cbm", MODEL_DEST / f"baseline_combo_seed_{s}.cbm")
        shutil.copy2(SRC_029_DIR / f"f_regime_seed_{s}.cbm", MODEL_DEST / f"f_regime_seed_{s}.cbm")
        shutil.copy2(SRC_029_DIR / f"r_regime_seed_{s}.cbm", MODEL_DEST / f"r_regime_seed_{s}.cbm")

    # 3. Write requirements.txt
    req_path = CANDIDATE_DIR / "requirements.txt"
    req_path.write_text("catboost==1.2.10\nnumpy>=1.26.0\npandas>=2.2.0\n", encoding="utf-8")

    # 4. Write script.py
    script_content = '''#!/usr/bin/env python3
"""Production Inference Script for REGIME-6SEED-FULL-029.

Executes 6-Seed Multi-Ensemble with 0.25 * Baseline + 0.75 * Split(F/R) blending.
Row-independent, offline, official-data-only contract.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import catboost as cb
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "model").exists() else HERE.parent
MODEL_DIR = ROOT / "model"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"

SEEDS = [42, 7, 2024, 99, 1, 123]
BASELINE_WEIGHT = 0.25
CANDIDATE_WEIGHT = 0.75

BASE_COLS = [
    "season", "game_month", "game_dayofweek", "inning", "top_bottom", "game_type",
    "balls_before", "strikes_before", "outs_before", "run_top_before", "run_bot_before",
    "run_total_before", "score_diff_home", "score_diff_pitcher_team", "runner_on_1b",
    "runner_on_2b", "runner_on_3b", "num_runners_on", "base_state", "home_win_expectancy",
    "away_win_expectancy", "li", "pitcher_id", "batter_id", "pitcher_hand", "batter_hand",
    "pitcher_team_id", "batter_team_id", "asof_pitcher_n", "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate", "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate", "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate", "asof_batter_n", "asof_batter_success_rate",
    "asof_batter_middle_rate", "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate"
]

STATE_COLS = [
    "cur_succ", "cur_mid", "cur_ball", "cur_rev", "cur_str",
    "cur_bsucc", "cur_bmid", "cur_logn_pitch", "cur_logn_mix", "cur_logn_bat"
]

EX_COLS = (
    [f"x_{r}_{c}" for r in ("cur_succ", "cur_mid") for c in ("adv", "onb", "sh", "bs")]
    + [f"h1_{r}_{c}" for r in ("cur_ball", "cur_rev", "cur_str") for c in ("sh", "bs")]
)

NEW_TM_COLS = (
    [f"tmc_{k}_dev" for k in ("breaking", "fastball", "offspeed")]
    + ["tmc_speed_dev"]
    + [f"tmh_{k}_dev" for k in ("breaking", "fastball", "offspeed")]
    + ["tmh_speed_dev"]
)

COLS = BASE_COLS + STATE_COLS + EX_COLS + ["target_hist_pitcher_count_n", "target_hist_pitcher_count_delta"] + NEW_TM_COLS


def build_features(df_test: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    o = df_test.copy()
    pp = pd.read_csv(MODEL_DIR / "asof_pitcher_id_prior.csv")
    bp = pd.read_csv(MODEL_DIR / "asof_batter_id_prior.csv")
    mp = pd.read_csv(MODEL_DIR / "asof_pitchmix_prior.csv")

    o = o.merge(pp, on="pitcher_id", how="left", validate="many_to_one")
    o = o.merge(bp, on="batter_id", how="left", validate="many_to_one")
    o = o.merge(mp, on="pitcher_id", how="left", validate="many_to_one")

    curp = (o.asof_pitcher_n.fillna(0) - o.prior_n_x.fillna(0)).clip(lower=0)
    curb = (o.asof_batter_n.fillna(0) - o.prior_n_y.fillna(0)).clip(lower=0)
    o["cur_logn_pitch"] = np.log1p(curp)
    o["cur_logn_bat"] = np.log1p(curb)
    o["cur_logn_mix"] = np.log1p((o.asof_pitcher_pitchmix_n.fillna(0) - o.prior_pitchmix_n.fillna(0)).clip(lower=0))

    spec = [
        ("asof_pitcher_success_rate", "cur_succ", "prior_asof_pitcher_success_rate", curp, o.asof_pitcher_n),
        ("asof_pitcher_middle_rate", "cur_mid", "prior_asof_pitcher_middle_rate", curp, o.asof_pitcher_n),
        ("asof_pitcher_ball_rate", "cur_ball", "prior_asof_pitcher_ball_rate", curp, o.asof_pitcher_n),
        ("asof_pitcher_reverse_rate", "cur_rev", "prior_asof_pitcher_reverse_rate", curp, o.asof_pitcher_n),
        ("asof_pitcher_strike_rate", "cur_str", "prior_asof_pitcher_strike_rate", curp, o.asof_pitcher_n),
        ("asof_batter_success_rate", "cur_bsucc", "prior_asof_batter_success_rate", curb, o.asof_batter_n),
        ("asof_batter_middle_rate", "cur_bmid", "prior_asof_batter_middle_rate", curb, o.asof_batter_n),
    ]
    for rc, lb, pc, cur, n in spec:
        o[lb] = ((n.fillna(0) * o[rc].fillna(0) - o[pc].fillna(0)) / cur).where(cur > 0)

    o["adv"] = (o.strikes_before > o.balls_before).astype(float)
    o["onb"] = (o.num_runners_on > 0).astype(float)
    o["sh"] = (o.pitcher_hand.astype(str) == o.batter_hand.astype(str)).astype(float)
    o["bs"] = o.balls_before - o.strikes_before

    full = (o.balls_before == 3) & (o.strikes_before == 2)
    pa = ((o.balls_before == 0) & o.strikes_before.isin([1, 2])) | ((o.balls_before == 1) & (o.strikes_before == 2))
    ba = ((o.balls_before == 2) & (o.strikes_before == 0)) | ((o.balls_before == 3) & o.strikes_before.isin([0, 1]))
    o["count_state"] = np.select([full, pa, ba], ["full_count", "ahead_pitcher", "ahead_batter"], default="neutral")

    for r in ("cur_succ", "cur_mid"):
        for c in ("adv", "onb", "sh", "bs"):
            o[f"x_{r}_{c}"] = o[r] * o[c]
    for r in ("cur_ball", "cur_rev", "cur_str"):
        for c in ("sh", "bs"):
            o[f"h1_{r}_{c}"] = o[r] * o[c]

    lk = pd.read_csv(MODEL_DIR / "pitcher_count_lookup.csv")
    o = o.merge(lk, on=["pitcher_id", "count_state"], how="left", validate="many_to_one")
    o["target_hist_pitcher_count_n"] = o["target_hist_pitcher_count_n"].fillna(0.0)
    o["target_hist_pitcher_count_delta"] = o["target_hist_pitcher_count_delta"].fillna(0.0)

    c = pd.read_csv(MODEL_DIR / "trackman_count_lookup.csv")
    h = pd.read_csv(MODEL_DIR / "trackman_hand_lookup.csv").rename(columns={"batter_hand": "__hand"})
    o = o.merge(c, on=["pitcher_id", "balls_before", "strikes_before"], how="left", validate="many_to_one")
    o["__hand"] = o["batter_hand"].map({1: "Left", 2: "Right"}).fillna(o["batter_hand"].astype(str))
    o = o.merge(h, on=["pitcher_id", "__hand"], how="left", validate="many_to_one").drop(columns="__hand")

    x = o[COLS].copy()
    cats = ["top_bottom", "game_type", "base_state"]
    for col in cats:
        x[col] = x[col].astype(str)
    return x, cats



def main() -> None:
    test_path = DATA_DIR / "test.csv"
    if not test_path.exists():
        raise RuntimeError(f"test.csv not found at: {test_path}")

    test_df = pd.read_csv(test_path, encoding="utf-8-sig")
    X_test, cats = build_features(test_df)
    pool_test = cb.Pool(X_test, cat_features=cats, feature_names=COLS)

    # 1. 6-Seed Baseline COMBO Predictions
    base_preds = []
    for s in SEEDS:
        m = cb.CatBoostClassifier()
        m.load_model(MODEL_DIR / f"baseline_combo_seed_{s}.cbm")
        base_preds.append(m.predict_proba(pool_test)[:, 1])
    pred_baseline = sum(base_preds) / len(base_preds)

    # 2. 6-Seed F-Regime Predictions
    mask_f = test_df.game_type.astype(str).eq("F").to_numpy()
    pred_f = np.zeros(len(test_df), dtype=float)
    if mask_f.any():
        f_pool = cb.Pool(X_test.loc[mask_f], cat_features=cats, feature_names=COLS)
        f_preds = []
        for s in SEEDS:
            m = cb.CatBoostClassifier()
            m.load_model(MODEL_DIR / f"f_regime_seed_{s}.cbm")
            f_preds.append(m.predict_proba(f_pool)[:, 1])
        pred_f[mask_f] = sum(f_preds) / len(f_preds)

    # 3. 6-Seed R-Regime Predictions
    mask_r = test_df.game_type.astype(str).eq("R").to_numpy()
    pred_r = np.zeros(len(test_df), dtype=float)
    if mask_r.any():
        r_pool = cb.Pool(X_test.loc[mask_r], cat_features=cats, feature_names=COLS)
        r_preds = []
        for s in SEEDS:
            m = cb.CatBoostClassifier()
            m.load_model(MODEL_DIR / f"r_regime_seed_{s}.cbm")
            r_preds.append(m.predict_proba(r_pool)[:, 1])
        pred_r[mask_r] = sum(r_preds) / len(r_preds)

    pred_split = np.where(mask_f, pred_f, pred_r)

    # Blend: 0.25 * Baseline_6seed + 0.75 * Split_6seed
    final_pred = BASELINE_WEIGHT * pred_baseline + CANDIDATE_WEIGHT * pred_split


    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    submission_path = OUTPUT_DIR / "submission.csv"
    sub = pd.DataFrame({"row_id": test_df["row_id"], "control_success": final_pred})
    sub.to_csv(submission_path, index=False)
    print(f"[{HERE.name}] Generated {len(sub):,} predictions to: {submission_path}")


if __name__ == "__main__":
    main()
'''
    (CANDIDATE_DIR / "script.py").write_text(script_content, encoding="utf-8")

    manifest = {
        "candidate_id": "REGIME-6SEED-FULL-029",
        "source_full_train": "REGIME-6SEED-FULL-029",
        "feature_count": 81,
        "seeds": SEEDS,
        "models_count": 18,
        "blend_weights": {
            "baseline": 0.25,
            "split": 0.75
        },
        "created_at": "2026-08-20"
    }
    (CANDIDATE_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[BUILD-029] Candidate package complete in: {CANDIDATE_DIR}")


if __name__ == "__main__":
    main()
