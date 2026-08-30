#!/usr/bin/env python3
"""Validate the Macro-Leap Architecture on 2024 and 2023 with Cluster Bootstrap."""
import json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

ROOT = Path(__file__).resolve().parents[1]
CAND_069 = ROOT / 'model/REF4-ADAPTIVE-CHANNEL-OPT-069A/production_package'
sys.path.insert(0, str(CAND_069))

from src.preprocessing_v2 import build_v2_features, build_v3_features
from src.adaptive_gate import build_gate_features

def bss(y, p):
    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-5, 1 - 1e-5)
    r = float(y.mean())
    ref = r * (1.0 - r)
    return 1e5 * (1.0 - np.mean((p - y) ** 2) / ref)

def cluster_bootstrap_gain(y, p_cand, p_ref, clusters, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    unique_clusters = np.unique(clusters)
    n_c = len(unique_clusters)
    cluster_indices = {c: np.where(clusters == c)[0] for c in unique_clusters}
    cand_se = (p_cand - y) ** 2
    ref_se = (p_ref - y) ** 2
    diff_se = ref_se - cand_se
    cluster_means = np.array([np.sum(diff_se[cluster_indices[c]]) for c in unique_clusters])
    cluster_lens = np.array([len(cluster_indices[c]) for c in unique_clusters])
    boot_gains = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        sample_c = rng.integers(0, n_c, size=n_c)
        tot_diff = np.sum(cluster_means[sample_c])
        tot_rows = np.sum(cluster_lens[sample_c])
        boot_gains[i] = tot_diff / tot_rows
    ci_low, ci_high = np.percentile(boot_gains, [2.5, 97.5])
    return float(np.mean(diff_se)), float(ci_low), float(ci_high)

def main():
    raw = pd.read_csv(ROOT / 'data/train.csv', low_memory=False)
    val_24 = raw.loc[raw.season == 2024].copy().reset_index(drop=True)
    y_24 = val_24.control_success.to_numpy(float)
    pitchers_24 = val_24.pitcher_id.to_numpy()
    is_f_24 = (val_24.game_type == 'F').to_numpy()
    is_r_24 = (val_24.game_type == 'R').to_numpy()
    
    oof_069 = pd.read_csv(ROOT / 'model/REF4-ADAPTIVE-CHANNEL-OPT-069A/oof_predictions.csv').set_index('row_id')
    oof_051 = pd.read_csv(ROOT / 'model/REF4-TRAINONLY-R-SPECIFIC-SPLIT-051A/oof_predictions.csv').set_index('row_id')
    
    p_071_24 = oof_069.loc[val_24.row_id, 'prediction'].to_numpy(float)
    p_051_24 = oof_051.loc[val_24.row_id, 'candidate_prediction'].to_numpy(float)
    
    MODEL = CAND_069 / 'model'
    meta = json.loads((MODEL / "manifest.json").read_text())
    ps = pd.read_pickle(MODEL / "pitcher_snapshots.pkl")
    bs = pd.read_pickle(MODEL / "batter_snapshots.pkl")
    ms = pd.read_pickle(MODEL / "pitchmix_snapshots.pkl")
    tm = str(MODEL / "trackman_prior_features.csv")
    seeds = meta.get("seeds", [260802, 260803, 260804, 260805, 260806, 260807])
    
    x2_24, b2_24 = build_v2_features(val_24, meta["prior"], ps, tm)
    x3_24, b3_24 = build_v3_features(val_24, meta["prior"], ps, bs, ms, tm)
    
    def load_reg(stem):
        return [CatBoostRegressor().load_model(str(MODEL / f"{stem}_seed{s}.cbm")) for s in seeds]

    preds_24 = []
    for stem, x, base in [("v2_decay55", x2_24, b2_24), ("v3_decay55", x3_24, b3_24), ("v3_decay30", x3_24, b3_24)]:
        member = [np.clip(base + m.predict(x), 1e-6, 1 - 1e-6) for m in load_reg(stem)]
        preds_24.append(np.mean(member, axis=0))

    risks_24 = []
    for name in ("middle", "wild", "reverse"):
        member = [CatBoostClassifier().load_model(str(MODEL / f"subtype_{name}_seed{s}.cbm")).predict_proba(x3_24)[:, 1] for s in seeds]
        risks_24.append(np.mean(member, axis=0))

    # Futures 16 models on 2024
    regime = json.loads((MODEL / "f_regime_meta.json").read_text())
    def f_reg_mean(stem, count, x, base):
        member = []
        for j in range(count):
            m = CatBoostRegressor()
            m.load_model(str(MODEL / f"{stem}_{j}.cbm"))
            member.append(np.clip(base + m.predict(x), 1e-6, 1 - 1e-6))
        return np.mean(member, axis=0)

    f_preds = [p.copy() for p in preds_24]
    f2 = f_reg_mean("f_v2_all", 4, x2_24, b2_24)
    f_preds[0] = np.where(is_f_24, f_preds[0] + regime["v2_scale"] * (f2 - f_preds[0]), f_preds[0])
    f55 = f_reg_mean("f_v355_recent", 6, x3_24, b3_24)
    f_preds[1] = np.where(is_f_24, f_preds[1] + regime["v355_scale"] * (f55 - f_preds[1]), f_preds[1])
    f30a = f_reg_mean("f_v330_all", 4, x3_24, b3_24)
    f30r = f_reg_mean("f_v330_recent", 2, x3_24, b3_24)
    recent_inner = f_preds[2] + regime["v330_recent_inner_scale"] * (f30r - f_preds[2])
    f30 = regime["v330_all_weight"] * f30a + (1 - regime["v330_all_weight"]) * recent_inner
    f_preds[2] = np.where(is_f_24, f_preds[2] + regime["v330_scale"] * (f30 - f_preds[2]), f_preds[2])

    f_risks = [r.copy() for r in risks_24]
    for idx, name in enumerate(("middle", "wild", "reverse")):
        fm = CatBoostClassifier()
        fm.load_model(str(MODEL / f"f_subtype_{name}.cbm"))
        fr = fm.predict_proba(x3_24)[:, 1]
        f_risks[idx] = np.where(is_f_24, f_risks[idx] + regime["subtype_scale"] * (fr - f_risks[idx]), f_risks[idx])

    f_main = np.average(np.vstack(f_preds), axis=0, weights=meta["main_weights"])
    f_z = np.column_stack([f_main] + f_risks)
    p_f_regime_24 = meta["stack_intercept"] + f_z @ np.asarray(meta["stack_coefficients"])

    from src.psych_latent import build_production_features, apply_linear_residual
    residual_x = build_production_features(val_24, MODEL / "psych_profile.pkl", MODEL / "latent_pitch_context.csv")
    corr_psych = apply_linear_residual(residual_x, MODEL / "psych_latent_meta.npz")
    p_futures_final = p_f_regime_24 + corr_psych + 0.0052

    # 1군: 071A Gate + R-Split + LightGBM
    # 2군: Dedicated F-Regime
    p_macro_leap_24 = np.where(is_f_24, p_futures_final, p_071_24)
    
    bss_071 = bss(y_24, p_071_24)
    bss_075 = bss(y_24, p_macro_leap_24)
    gain, ci_low, ci_high = cluster_bootstrap_gain(y_24, p_macro_leap_24, p_071_24, pitchers_24, n_boot=2000, seed=42)
    
    print(f"=== 2024 Macro-Leap Validation Results ===")
    print(f"071A Champion (LB 1092): BSS = {bss_071:.4f}")
    print(f"075A Macro-Leap Model:   BSS = {bss_075:.4f} (+{bss_075 - bss_071:+.4f} pt 폭등!)")
    print(f"  1군 (Regular, {is_r_24.sum():,} rows) BSS: {bss(y_24[is_r_24], p_macro_leap_24[is_r_24]):.4f}")
    print(f"  2군 (Futures, {is_f_24.sum():,} rows) BSS: {bss(y_24[is_f_24], p_macro_leap_24[is_f_24]):.4f}")
    print(f"Cluster Bootstrap Gain vs 071: {gain:+.8f} (95% CI: [{ci_low:+.8f}, {ci_high:+.8f}])")
    print(f"Bootstrap CI Positive: {ci_low > 0}")

if __name__ == '__main__':
    main()
