#!/usr/bin/env python3
"""Build candidate inference package for REGIME-RCAPACITY-FULL-019."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "model" / "REGIME-RCAPACITY-FULL-019"
CANDIDATE_DIR = ROOT / "candidate" / "REGIME-RCAPACITY-FULL-019"
MODEL_DEST = CANDIDATE_DIR / "model"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    if not SOURCE_DIR.exists():
        raise RuntimeError(f"source model directory missing: {SOURCE_DIR}")

    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DEST.mkdir(parents=True, exist_ok=True)
    (CANDIDATE_DIR / "data").mkdir(exist_ok=True)
    (CANDIDATE_DIR / "output").mkdir(exist_ok=True)

    # 1. Copy model files and lookups
    files_to_copy = [
        "model_baseline_combo.cbm",
        "model_regime_f.cbm",
        "model_regime_r.cbm",
        "feature_columns.json",
        "asof_pitcher_id_prior.csv",
        "asof_batter_id_prior.csv",
        "asof_pitchmix_prior.csv",
        "pitcher_count_lookup.csv",
        "trackman_count_lookup.csv",
        "trackman_hand_lookup.csv",
        "pitcher_id_map_audit.csv",
    ]

    copied_manifest = {}
    for filename in files_to_copy:
        src = SOURCE_DIR / filename
        dst = MODEL_DEST / filename
        if not src.exists():
            raise RuntimeError(f"required source artifact missing: {src}")
        shutil.copy2(src, dst)
        copied_manifest[filename] = {
            "sha256": sha256_file(dst),
            "size_bytes": dst.stat().st_size,
        }

    # 2. Write requirements.txt
    req_path = CANDIDATE_DIR / "requirements.txt"
    req_path.write_text("catboost==1.2.10\nnumpy>=1.26.0\npandas>=2.2.0\n", encoding="utf-8")
    copied_manifest["requirements.txt"] = {
        "sha256": sha256_file(req_path),
        "size_bytes": req_path.stat().st_size,
    }

    # 3. Write script.py
    script_content = '''#!/usr/bin/env python3
"""Production Inference Script for REGIME-RCAPACITY-FULL-019.

Executes 0.25 * Baseline COMBO + 0.75 * Split(F/R) blending.
Row-independent, offline, official-data-only contract.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import catboost as cb
import numpy as np
import pandas as pd

# Support running from submission root or package directory
HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "model").exists() else HERE.parent
MODEL_DIR = ROOT / "model"
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"

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

    for f in EX_COLS:
        z = f.split("_")
        o[f] = o["_".join(z[1:3])] * o[z[3]]

    lk = pd.read_csv(MODEL_DIR / "pitcher_count_lookup.csv")
    o = o.merge(lk, on=["pitcher_id", "count_state"], how="left", validate="many_to_one")
    o[["target_hist_pitcher_count_n", "target_hist_pitcher_count_delta"]] = (
        o[["target_hist_pitcher_count_n", "target_hist_pitcher_count_delta"]].fillna(0)
    )

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
        test_path = ROOT / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError("test.csv not found in data/ or root")

    df_test = pd.read_csv(test_path, encoding="utf-8-sig")
    x_test, cats = build_features(df_test)

    # 1. Load Baseline COMBO Model
    m_baseline = cb.CatBoostClassifier()
    m_baseline.load_model(MODEL_DIR / "model_baseline_combo.cbm")
    pred_baseline = m_baseline.predict_proba(cb.Pool(x_test, cat_features=cats, feature_names=COLS))[:, 1]

    # 2. Load Regime Models
    m_f = cb.CatBoostClassifier()
    m_f.load_model(MODEL_DIR / "model_regime_f.cbm")

    m_r = cb.CatBoostClassifier()
    m_r.load_model(MODEL_DIR / "model_regime_r.cbm")

    # 3. Predict by Regime Split
    mask_f = df_test.game_type.astype(str).eq("F").to_numpy()
    mask_r = df_test.game_type.astype(str).eq("R").to_numpy()

    pred_split = np.zeros(len(df_test), dtype=float)
    if mask_f.any():
        pred_split[mask_f] = m_f.predict_proba(
            cb.Pool(x_test.loc[mask_f], cat_features=cats, feature_names=COLS)
        )[:, 1]
    if mask_r.any():
        pred_split[mask_r] = m_r.predict_proba(
            cb.Pool(x_test.loc[mask_r], cat_features=cats, feature_names=COLS)
        )[:, 1]

    # 4. Final 0.25 : 0.75 Blend
    pred_final = BASELINE_WEIGHT * pred_baseline + CANDIDATE_WEIGHT * pred_split

    if not np.isfinite(pred_final).all() or not ((pred_final >= 0) & (pred_final <= 1)).all():
        raise RuntimeError("prediction finite/range contract failed")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame({"row_id": df_test.row_id, "control_success": pred_final})
    sub_path = OUTPUT_DIR / "submission.csv"
    submission.to_csv(sub_path, index=False, encoding="utf-8")

    summary = {
        "rows": len(submission),
        "features": len(x_test.columns),
        "min": float(pred_final.min()),
        "max": float(pred_final.max()),
        "mean": float(pred_final.mean()),
        "status": "PASS_INDEPENDENT_INFERENCE",
    }
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
'''
    script_path = CANDIDATE_DIR / "script.py"
    script_path.write_text(script_content, encoding="utf-8")
    script_path.chmod(0o755)
    copied_manifest["script.py"] = {
        "sha256": sha256_file(script_path),
        "size_bytes": script_path.stat().st_size,
    }

    # 4. Save bundle metadata
    bundle_meta = {
        "bundle_id": "REGIME-RCAPACITY-FULL-019-BUNDLE",
        "source_experiment": "REGIME-RCAPACITY-FULL-019",
        "official_train_only": True,
        "files": copied_manifest,
    }
    (MODEL_DEST / "bundle_metadata.json").write_text(json.dumps(bundle_meta, ensure_ascii=False, indent=2))
    print(f"Bundle successfully created at: {CANDIDATE_DIR}")
    print(json.dumps(bundle_meta, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
