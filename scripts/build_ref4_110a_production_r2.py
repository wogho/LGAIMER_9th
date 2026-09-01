#!/usr/bin/env python3
"""Build the audited 110A production directory; deliberately does not ZIP it."""
from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "model/REF4-110-ORIGINAL-R2"
SOURCE_109 = ROOT / "model/REF4-SUPER-ENSEMBLE-109C/production_package"
SOURCE_108 = ROOT / "model/REF4-SUPER-ENSEMBLE-108C/production_package/model"
DEST_ROOT = ROOT / "model/REF4-TEMPORAL-CROSSFIT-MOE-110A-R2"
PROD = DEST_ROOT / "production_package"
MODEL = PROD / "model"
ROUTER_SEEDS = (42, 1, 2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


ADVANCED_FUNCTION = r'''

def extract_advanced_physics(df: pd.DataFrame) -> pd.DataFrame:
    velo = df["release_speed"].fillna(142.0).to_numpy(float) if "release_speed" in df.columns else np.full(len(df), 142.0)
    spin = df["spin_rate"].fillna(2200.0).to_numpy(float) if "spin_rate" in df.columns else np.full(len(df), 2200.0)
    pfx_x = df["pfx_x"].fillna(0.0).to_numpy(float) if "pfx_x" in df.columns else np.zeros(len(df))
    pfx_z = df["pfx_z"].fillna(0.0).to_numpy(float) if "pfx_z" in df.columns else np.zeros(len(df))
    rel_x = df["release_pos_x"].fillna(0.0).to_numpy(float) if "release_pos_x" in df.columns else np.zeros(len(df))
    rel_z = df["release_pos_z"].fillna(1.8).to_numpy(float) if "release_pos_z" in df.columns else np.full(len(df), 1.8)
    movement_mag = np.sqrt(pfx_x**2 + pfx_z**2)
    spin_eff_ratio = np.clip(movement_mag / (velo + 1e-5), 0, 1.0)
    release_dist = np.sqrt(rel_x**2 + (rel_z - 1.8)**2)
    balls = df["balls_before"].fillna(0).to_numpy(float) if "balls_before" in df.columns else np.zeros(len(df))
    strikes = df["strikes_before"].fillna(0).to_numpy(float) if "strikes_before" in df.columns else np.zeros(len(df))
    return pd.DataFrame({
        "phys_velo": velo, "phys_spin": spin, "phys_movement_mag": movement_mag,
        "phys_spin_eff": spin_eff_ratio, "phys_release_dist": release_dist,
        "phys_is_2s": (strikes == 2).astype(float), "phys_is_3b": (balls == 3).astype(float),
        "phys_count_pressure": (balls - strikes) * (balls + strikes + 1.0) / 7.0,
    }, index=df.index)
'''


P108_BLOCK = r'''

    # 110A expert snapshots: p107 is the inherited backbone before 108C/109C.
    p107 = np.clip(p.copy(), 1e-5, 1 - 1e-5)
    p108 = p107.copy()
    moe108_r_seeds = [42, 1, 2, 3, 4]
    if regular.any():
        x108r, _ = build_v5_deep_61_features(test[regular].reset_index(drop=True), profile_path=MODEL / "team_asof_profile.json", prior=0.523766)
        prev1_108 = test.loc[regular, "asof_pitcher_prev1_game_success_rate"].to_numpy(float) if "asof_pitcher_prev1_game_success_rate" in test.columns else p_rate_raw[regular]
        prev1_108 = np.where(np.isnan(prev1_108), p_rate_raw[regular], prev1_108)
        batter_108 = test.loc[regular, "asof_batter_success_rate"].fillna(0.523766).to_numpy(float) if "asof_batter_success_rate" in test.columns else np.full(np.sum(regular), 0.523766)
        x108r["form_gap"] = prev1_108 - batter_108
        x108r["anchor_p"] = p107[regular]
        x108r = pd.concat([x108r, extract_advanced_physics(test[regular].reset_index(drop=True))], axis=1)
        members108 = []
        for seed in moe108_r_seeds:
            cb108 = CatBoostRegressor().load_model(str(MODEL / f"moe108_super_resid_cb_seed{seed}.cbm"))
            members108.append(cb108.predict(x108r))
            lg108 = lgb.Booster(model_file=str(MODEL / f"moe108_super_resid_lgb_seed{seed}.txt"))
            members108.append(lg108.predict(x108r))
        p108[regular] = p107[regular] + 0.08 * np.mean(members108, axis=0)
    if futures.any():
        x108f, _ = build_v5_deep_61_features(test[futures].reset_index(drop=True), profile_path=MODEL / "team_asof_profile.json", prior=0.523766)
        x108f["anchor_p"] = p107[futures]
        x108f = pd.concat([x108f, extract_advanced_physics(test[futures].reset_index(drop=True))], axis=1)
        members108f = []
        for seed in [42, 1, 2, 3]:
            cb108f = CatBoostRegressor().load_model(str(MODEL / f"moe108_fut_resid_cb_seed{seed}.cbm"))
            members108f.append(cb108f.predict(x108f))
        p108[futures] = p107[futures] + 0.04 * np.mean(members108f, axis=0)
    p108 = np.clip(p108, 1e-5, 1 - 1e-5)
'''


ROUTER_BLOCK = r'''

    # 110A final temporal router, fitted only from strict 2022-2024 expert OOF.
    p109 = np.clip(p.copy(), 1e-5, 1 - 1e-5)
    router_x = router_features(test.reset_index(drop=True), np.column_stack([p107, p108, p109]))
    router_members = []
    for seed in [42, 1, 2]:
        router_model = CatBoostRegressor().load_model(str(MODEL / f"moe_router_seed{seed}.cbm"))
        router_members.append(router_model.predict(router_x))
    p = p109 + np.mean(router_members, axis=0)
'''


def build_script() -> str:
    script = (SOURCE_109 / "script.py").read_text(encoding="utf-8")
    script = script.replace('from src.v5_deep_61_features import build_v5_deep_61_features\n', 'from src.v5_deep_61_features import build_v5_deep_61_features\nfrom src.ref4_110_router import router_features\n', 1)
    script = script.replace("\ndef main():", ADVANCED_FUNCTION + "\n\ndef main():", 1)
    pocket_marker = '    # 109C: Hyper-Regime Tri-Bridge 15-Model Ensemble (w=0.085)\n'
    if script.count(pocket_marker) != 1:
        raise RuntimeError("109C insertion marker mismatch")
    script = script.replace(pocket_marker, P108_BLOCK + "\n" + pocket_marker, 1)
    final_marker = '    p = np.clip(p, 1e-5, 1 - 1e-5)\n    if len(p) != len(test) or not np.isfinite(p).all():\n'
    if script.count(final_marker) != 1:
        raise RuntimeError("router insertion marker mismatch")
    script = script.replace(final_marker, ROUTER_BLOCK + "\n" + final_marker, 1)
    script = script.replace('"""Offline inference entry point for REF4-SUPER-ENSEMBLE-109C."""', '"""Offline inference entry point for REF4-TEMPORAL-CROSSFIT-MOE-110A-R2."""', 1)
    canonical = '''\nCANONICAL_INPUT_COLUMNS = [\n    "row_id", "season", "game_month", "game_dayofweek", "inning", "top_bottom", "game_type",\n    "balls_before", "strikes_before", "outs_before", "run_top_before", "run_bot_before", "run_total_before",\n    "score_diff_home", "score_diff_pitcher_team", "runner_on_1b", "runner_on_2b", "runner_on_3b",\n    "num_runners_on", "base_state", "home_win_expectancy", "away_win_expectancy", "li", "pitcher_id",\n    "batter_id", "pitcher_hand", "batter_hand", "pitcher_team_id", "batter_team_id", "asof_pitcher_n",\n    "asof_pitcher_success_rate", "asof_pitcher_reverse_rate", "asof_pitcher_middle_rate",\n    "asof_pitcher_ball_rate", "asof_pitcher_strike_rate", "asof_pitcher_prev1_game_success_rate",\n    "asof_pitcher_prev3_game_success_rate", "asof_pitcher_prev5_game_success_rate",\n    "asof_pitcher_prev1_game_middle_rate", "asof_pitcher_prev3_game_middle_rate",\n    "asof_pitcher_prev5_game_middle_rate", "asof_batter_n", "asof_batter_success_rate",\n    "asof_batter_middle_rate", "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate",\n    "asof_pitcher_breaking_rate", "asof_pitcher_offspeed_rate",\n+]\n'''
    script = script.replace('MODEL = ROOT / "model"\n', 'MODEL = ROOT / "model"\n' + canonical, 1)
    script = script.replace('    test = pd.read_csv(ROOT / "data/test.csv", low_memory=False)\n', '    test = pd.read_csv(ROOT / "data/test.csv", low_memory=False)\n    missing_columns = [column for column in CANONICAL_INPUT_COLUMNS if column not in test.columns]\n    if missing_columns:\n        raise RuntimeError(f"missing required input columns: {missing_columns}")\n    test = test[CANONICAL_INPUT_COLUMNS + [column for column in test.columns if column not in CANONICAL_INPUT_COLUMNS]]\n', 1)
    return script


def main() -> None:
    validation = json.loads((AUDIT / "validation_report.json").read_text())
    if validation.get("status") != "AUDIT_VERIFIED" or validation.get("provisional_winner") != "110A":
        raise RuntimeError("110A is not the independently audited winner")
    if DEST_ROOT.exists():
        raise RuntimeError(f"destination already exists: {DEST_ROOT}")
    DEST_ROOT.mkdir(parents=True)
    shutil.copytree(SOURCE_109, PROD)
    copied_108 = []
    for prefix in ("super_resid_cb_seed", "super_resid_lgb_seed", "fut_resid_cb_seed"):
        for source in sorted(SOURCE_108.glob(f"{prefix}*")):
            target = MODEL / f"moe108_{source.name}"
            shutil.copy2(source, target)
            copied_108.append(target)
    if len(copied_108) != 14:
        raise RuntimeError(f"expected 14 108C residual models, got {len(copied_108)}")
    shutil.copy2(ROOT / "src/ref4_110_codex.py", PROD / "src/ref4_110_router.py")

    oof = pd.read_csv(AUDIT / "expert_oof.csv", dtype={"row_id": str})
    raw = pd.read_csv(ROOT / "data/train.csv", low_memory=False)
    raw["row_id"] = raw["row_id"].astype(str)
    raw_index = raw.set_index("row_id", drop=False)
    rows = raw_index.loc[oof["row_id"]].reset_index(drop=True)
    helper_path = ROOT / "src/ref4_110_codex.py"
    spec = __import__("importlib.util").util.spec_from_file_location("ref4_router_fullfit", helper_path)
    helper = __import__("importlib.util").util.module_from_spec(spec)
    sys.modules["ref4_router_fullfit"] = helper
    spec.loader.exec_module(helper)
    x = helper.router_features(rows, oof[["p107", "p108c", "p109c"]].to_numpy(float))
    target = oof["target"].to_numpy(float) - oof["p109c"].to_numpy(float)
    router_models = []
    for seed in ROUTER_SEEDS:
        print(f"router full-fit seed={seed}", flush=True)
        model = CatBoostRegressor(iterations=150, depth=4, learning_rate=0.03, l2_leaf_reg=5, random_seed=seed, thread_count=4, allow_writing_files=False, verbose=False)
        model.fit(x, target)
        path = MODEL / f"moe_router_seed{seed}.cbm"
        model.save_model(path)
        router_models.append(path)

    (PROD / "script.py").write_text(build_script(), encoding="utf-8")
    manifest_path = MODEL / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update({
        "version": "REF4-TEMPORAL-CROSSFIT-MOE-110A-R2",
        "moe_experts": ["107A", "108C", "109C"],
        "moe_router_seeds": list(ROUTER_SEEDS),
        "moe_router_train_seasons": [2022, 2023, 2024],
        "moe_router_train_rows": int(len(oof)),
        "moe_router_source": "strict-forward OOF only",
    })
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    provenance = {
        "status": "PENDING_TECHNICAL_AUDIT",
        "candidate": "110A",
        "source_109_script_sha256": sha256(SOURCE_109 / "script.py"),
        "source_109_manifest_sha256": sha256(SOURCE_109 / "model/manifest.json"),
        "audit_manifest_sha256": sha256(AUDIT / "audit_manifest.json"),
        "expert_oof_sha256": sha256(AUDIT / "expert_oof.csv"),
        "router_train_rows": int(len(oof)),
        "router_train_seasons": oof["season"].value_counts().sort_index().to_dict(),
        "copied_108_models": {path.name: sha256(path) for path in copied_108},
        "router_models": {path.name: sha256(path) for path in router_models},
        "test_read": False,
        "zip_created": False,
    }
    (DEST_ROOT / "build_provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"status": provenance["status"], "router_rows": len(oof), "copied_108_models": len(copied_108), "router_models": len(router_models)}, indent=2))


if __name__ == "__main__":
    main()
