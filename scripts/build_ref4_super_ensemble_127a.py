#!/usr/bin/env python3
"""
Production Packager for REF4-ANCHOR-INVARIANT-R-RESIDUAL-PROD-127A.
Based on:
  - Official Champion Baseline: 113A (Official LB: 1121.9039933605)
  - Audited L4 R-Residual Expert: REF4-127 (Delta Brier: -0.00027052 on 2024, CI < 0)
  - Strict Convex Blend: (1 - w)*p113A + w*clip(p113A + raw_corr, 0.005, 0.995), w = 0.075
  - Bit-exact Frozen 113A Futures (F) regime
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/train.csv"
ZIP_113A = ROOT / "output/submit_ref4_super_ensemble_113A.zip"
EXP_127_DIR = ROOT / "model/REF4-ANCHOR-INVARIANT-R-RESIDUAL-L4-127"
OUT_DIR = ROOT / "model/REF4-ANCHOR-INVARIANT-R-RESIDUAL-PROD-127A/production_package"
ZIP_OUT = ROOT / "output/submit_ref4_super_ensemble_127A.zip"

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 80)
    print("  BUILDING PRODUCTION PACKAGE: REF4-ANCHOR-INVARIANT-R-RESIDUAL-PROD-127A  ")
    print("=" * 80)
    
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Unzip 113A base package
    print(f"Extracting 113A base package from {ZIP_113A}...")
    with zipfile.ZipFile(ZIP_113A, "r") as z:
        z.extractall(OUT_DIR)
        
    out_model = OUT_DIR / "model"
    out_src = OUT_DIR / "src"
    out_model.mkdir(exist_ok=True)
    out_src.mkdir(exist_ok=True)
    
    # 2. Copy the 3 L4 CatBoost models from 127 checkpoint
    cp_models_dir = EXP_127_DIR / "checkpoints/full_l4/2024/models"
    for seed in (17, 42, 777):
        src_cbm = cp_models_dir / f"r_residual_seed{seed}.cbm"
        dst_cbm = out_model / f"r_residual_127_seed{seed}.cbm"
        assert src_cbm.exists(), f"Missing {src_cbm}"
        shutil.copy2(src_cbm, dst_cbm)
        print(f"  • Copied r_residual_127_seed{seed}.cbm ({dst_cbm.stat().st_size / 1024:.1f} KB)")
        
    # 3. Generate and save prior_game_type lookup table
    print("Generating prior_game_type lookup table from train.csv...")
    train = pd.read_csv(DATA_PATH, low_memory=False)
    earlier = train.loc[train["season"].lt(2025)]
    counts = (
        earlier.groupby(["pitcher_id", "season", "game_type"], sort=False, observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    dominant = counts.sort_values("n").groupby(
        ["pitcher_id", "season"], sort=False, observed=True
    ).tail(1)
    latest = dominant.sort_values("season").groupby("pitcher_id", sort=False, observed=True).tail(1)
    prior_map = latest.set_index(latest["pitcher_id"].astype(str))["game_type"].to_dict()
    
    with open(out_model / "prior_game_type_lookup.json", "w", encoding="utf-8") as f:
        json.dump(prior_map, f, indent=2)
    print(f"  • Saved prior_game_type_lookup.json ({len(prior_map)} pitchers)")
    
    # 4. Update manifest.json
    manifest = json.loads((out_model / "manifest.json").read_text(encoding="utf-8")) if (out_model / "manifest.json").exists() else {}
    manifest["pipeline"] = "REF4-ANCHOR-INVARIANT-R-RESIDUAL-PROD-127A"
    manifest["base_backbone"] = "REF4-SUPER-ENSEMBLE-113A (1121.9039933605)"
    manifest["description"] = "113A Backbone + Audited L4 R-Residual Expert (w=0.075 Convex Blend)"
    manifest["created_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    manifest["r_residual_127_weight"] = 0.075
    manifest["r_residual_127_seeds"] = [17, 42, 777]
    with open(out_model / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
        
    # 5. Read existing 113A script.py and integrate 127 R-residual blend
    script_orig = (OUT_DIR / "script.py").read_text(encoding="utf-8")
    
    # Check for build_127_features integration
    r_blend_code = '''
    # =========================================================================
    # 127: Audited Anchor-Invariant R-Residual Expert (NVIDIA L4, w=0.075)
    # =========================================================================
    w_127 = float(meta.get("r_residual_127_weight", 0.075))
    if regular.any() and w_127 > 0.0:
        with open(MODEL / "prior_game_type_lookup.json", encoding="utf-8") as f:
            prior_gt_map = json.load(f)
            
        test_reg = test[regular].reset_index(drop=True)
        p113_reg = p[regular].copy()
        
        # Build exact 55 features for 127 R-residual expert
        feat_127 = test_reg.drop(columns=["row_id", "control_success", "pitcher_id", "batter_id"], errors="ignore").copy()
        pid_series = test_reg["pitcher_id"].astype(str) if "pitcher_id" in test_reg.columns else pd.Series(["NEW"] * len(test_reg))
        p_gt = pid_series.map(prior_gt_map).fillna("NEW").astype(str)
        c_gt = test_reg["game_type"].astype(str) if "game_type" in test_reg.columns else pd.Series(["R"] * len(test_reg))
        
        p_team = test_reg["pitcher_team_id"].astype(str) if "pitcher_team_id" in test_reg.columns else pd.Series(["0"] * len(test_reg))
        b_before = test_reg["balls_before"].astype(str) if "balls_before" in test_reg.columns else pd.Series(["0"] * len(test_reg))
        s_before = test_reg["strikes_before"].astype(str) if "strikes_before" in test_reg.columns else pd.Series(["0"] * len(test_reg))
        p_hand = test_reg["pitcher_hand"].astype(str) if "pitcher_hand" in test_reg.columns else pd.Series(["R"] * len(test_reg))
        b_hand = test_reg["batter_hand"].astype(str) if "batter_hand" in test_reg.columns else pd.Series(["R"] * len(test_reg))
        
        feat_127["prior_game_type"] = p_gt
        feat_127["league_transition"] = p_gt + ">" + c_gt
        feat_127["team_type"] = p_team + "|" + c_gt
        feat_127["count_state"] = b_before + "-" + s_before
        feat_127["hand_matchup"] = p_hand + "-" + b_hand
        feat_127["p113a_strict"] = p113_reg
        
        n_p_val = pd.to_numeric(test_reg["asof_pitcher_n"], errors="coerce").fillna(0).clip(lower=0) if "asof_pitcher_n" in test_reg.columns else pd.Series(0, index=test_reg.index)
        n_b_val = pd.to_numeric(test_reg["asof_batter_n"], errors="coerce").fillna(0).clip(lower=0) if "asof_batter_n" in test_reg.columns else pd.Series(0, index=test_reg.index)
        feat_127["log_pitcher_n"] = np.log1p(n_p_val).astype(np.float32)
        feat_127["log_batter_n"] = np.log1p(n_b_val).astype(np.float32)
        
        prev1_v = pd.to_numeric(test_reg["asof_pitcher_prev1_game_success_rate"], errors="coerce") if "asof_pitcher_prev1_game_success_rate" in test_reg.columns else pd.Series(0.523766, index=test_reg.index)
        prev5_v = pd.to_numeric(test_reg["asof_pitcher_prev5_game_success_rate"], errors="coerce") if "asof_pitcher_prev5_game_success_rate" in test_reg.columns else pd.Series(0.523766, index=test_reg.index)
        feat_127["recent_1_minus_5"] = (prev1_v - prev5_v).fillna(0.0).astype(np.float32)
        
        m_prev1_v = pd.to_numeric(test_reg["asof_pitcher_prev1_game_middle_rate"], errors="coerce") if "asof_pitcher_prev1_game_middle_rate" in test_reg.columns else pd.Series(0.0, index=test_reg.index)
        m_prev5_v = pd.to_numeric(test_reg["asof_pitcher_prev5_game_middle_rate"], errors="coerce") if "asof_pitcher_prev5_game_middle_rate" in test_reg.columns else pd.Series(0.0, index=test_reg.index)
        feat_127["middle_1_minus_5"] = (m_prev1_v - m_prev5_v).fillna(0.0).astype(np.float32)
        
        # Load 127 models
        seeds_127 = meta.get("r_residual_127_seeds", [17, 42, 777])
        preds_127 = []
        for s in seeds_127:
            m_path = MODEL / f"r_residual_127_seed{s}.cbm"
            if m_path.exists():
                cb_127 = CatBoostRegressor()
                cb_127.load_model(str(m_path))
                preds_127.append(safe_cb_predict(cb_127, feat_127))
                
        if preds_127:
            raw_r_corr = np.mean(preds_127, axis=0)
            aux_r = np.clip(p113_reg + raw_r_corr, 0.005, 0.995)
            p[regular] = (1.0 - w_127) * p113_reg + w_127 * aux_r
'''
    # Insert before final monotonic calibration
    target_marker = "    # Monotonic bounded calibration"
    if target_marker in script_orig:
        script_new = script_orig.replace(target_marker, r_blend_code + "\n" + target_marker)
    else:
        # fallback before out = ROOT / "output"
        script_new = script_orig.replace('    out = ROOT / "output"', r_blend_code + '\n    out = ROOT / "output"')
        
    (OUT_DIR / "script.py").write_text(script_new, encoding="utf-8")
    print("  • Updated script.py with 127 R-residual blend")
    
    # 6. Create clean submission ZIP
    print(f"Packaging {OUT_DIR} -> {ZIP_OUT}...")
    ZIP_OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for file in sorted(OUT_DIR.rglob("*")):
            if file.is_file():
                arcname = file.relative_to(OUT_DIR)
                z.write(file, arcname)
                
    zip_mb = ZIP_OUT.stat().st_size / (1024 * 1024)
    zip_sha = sha256(ZIP_OUT)
    print(f"Created {ZIP_OUT} ({zip_mb:.2f} MB, SHA256: {zip_sha})")

if __name__ == "__main__":
    main()
