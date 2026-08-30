#!/usr/bin/env python3
"""Evaluate one predeclared rolling train-only stack candidate."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model/REF4-TRAINONLY-RECENT-STACK-041A"
CONTRACT = OUT / "audit_contract.json"
REGIME = ROOT / "model/REF4-CHAMPION-STACK-030/f_regime_meta.json"
MANIFEST = ROOT / "model/REF4-CHAMPION-STACK-030/manifest.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def features(frame: pd.DataFrame, regime: dict[str, float], manifest: dict[str, object]) -> np.ndarray:
    f = frame["game_type"].eq("F").to_numpy()
    p0 = np.where(f, frame.p_v2_global + regime["v2_scale"] * (frame.p_v2_f - frame.p_v2_global), frame.p_v2_global)
    p1 = np.where(f, frame.p_v3_55_global + regime["v355_scale"] * (frame.p_v3_55_f - frame.p_v3_55_global), frame.p_v3_55_global)
    recent = frame.p_v3_30_global + regime["v330_recent_inner_scale"] * (frame.p_v3_30_f_recent - frame.p_v3_30_global)
    f30 = regime["v330_all_weight"] * frame.p_v3_30_f_all + (1.0 - regime["v330_all_weight"]) * recent
    p2 = np.where(f, frame.p_v3_30_global + regime["v330_scale"] * (f30 - frame.p_v3_30_global), frame.p_v3_30_global)
    main = np.average(np.vstack([p0, p1, p2]), axis=0, weights=np.asarray(manifest["main_weights"], float))
    risks = []
    for name in ("middle", "wild", "reverse"):
        g = frame[f"risk_{name}_global"].to_numpy(float)
        q = frame[f"risk_{name}_f"].to_numpy(float)
        risks.append(np.where(f, g + regime["subtype_scale"] * (q - g), g))
    return np.column_stack([main, *risks])


def metric(y: np.ndarray, p: np.ndarray) -> dict[str, float | int]:
    brier = float(np.mean((p - y) ** 2))
    ref = float(np.mean((y - y.mean()) ** 2))
    bss = 1.0 - brier / ref
    return {"rows": int(len(y)), "target_rate": float(y.mean()), "brier": brier, "bss": bss, "local_score": 100000.0 * bss}


def cluster_ci(y: np.ndarray, base: np.ndarray, cand: np.ndarray, groups: np.ndarray, repetitions: int, seed: int) -> dict[str, float | int]:
    gain = (base - y) ** 2 - (cand - y) ** 2
    codes, uniques = pd.factorize(groups, sort=True)
    sums = np.bincount(codes, weights=gain, minlength=len(uniques))
    counts = np.bincount(codes, minlength=len(uniques))
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions, dtype=float)
    for i in range(repetitions):
        chosen = rng.integers(0, len(uniques), size=len(uniques))
        draws[i] = sums[chosen].sum() / counts[chosen].sum()
    return {"clusters": int(len(uniques)), "repetitions": int(repetitions), "seed": int(seed), "brier_gain": float(gain.mean()), "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975))}


def fit_predict(train_x: np.ndarray, train_y: np.ndarray, valid_x: np.ndarray, alpha: float) -> tuple[np.ndarray, dict[str, object]]:
    scaler = StandardScaler()
    xtr = scaler.fit_transform(train_x)
    model = Ridge(alpha=alpha, fit_intercept=True)
    model.fit(xtr, train_y)
    pred = np.clip(model.predict(scaler.transform(valid_x)), 1e-5, 1 - 1e-5)
    params = {"scaler_mean": scaler.mean_.tolist(), "scaler_scale": scaler.scale_.tolist(), "ridge_coefficients": model.coef_.tolist(), "ridge_intercept": float(model.intercept_), "alpha": alpha}
    return pred, params


def main() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    preflight = json.loads((OUT / "preflight_report.json").read_text(encoding="utf-8"))
    if preflight.get("status") != "AUDIT_VERIFIED" or preflight.get("fail_count") != 0:
        raise RuntimeError("preflight not verified")
    regime = json.loads(REGIME.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    frames: dict[int, pd.DataFrame] = {}
    arrays: dict[int, np.ndarray] = {}
    source_checks: list[dict[str, object]] = []
    for key, rel in contract["source_folds"].items():
        year = int(key)
        frame = pd.read_csv(ROOT / rel)
        numeric = [c for c in frame.columns if c.startswith("p_") or c.startswith("risk_") or c.startswith("prediction")]
        source_checks.append({
            "season": year, "rows": int(len(frame)), "row_id_unique": bool(frame.row_id.is_unique),
            "season_exact": bool(frame.season.eq(year).all()), "target_binary": bool(frame.target.isin([0, 1]).all()),
            "numeric_finite": bool(np.isfinite(frame[numeric].to_numpy(float)).all()),
            "numeric_in_range": bool(((frame[numeric] >= 0) & (frame[numeric] <= 1)).all().all()),
        })
        frames[year] = frame
        arrays[year] = features(frame, regime, manifest)
        recreated = np.clip(float(manifest["stack_intercept"]) + arrays[year] @ np.asarray(manifest["stack_coefficients"], float) + float(manifest["global_shift"]), 1e-5, 1 - 1e-5)
        source_checks[-1]["baseline_max_abs_diff"] = float(np.max(np.abs(recreated - frame.prediction.to_numpy(float))))
    if not all(c["row_id_unique"] and c["season_exact"] and c["target_binary"] and c["numeric_finite"] and c["numeric_in_range"] and c["baseline_max_abs_diff"] <= 1e-12 for c in source_checks):
        raise RuntimeError(f"source validation failed: {source_checks}")

    alpha = float(contract["fixed_estimator"]["alpha"])
    predictions: list[pd.DataFrame] = []
    candidates: dict[str, object] = {}
    for protocol in contract["temporal_protocol"]:
        fit_year, valid_year = int(protocol["fit_season"]), int(protocol["validation_season"])
        valid = frames[valid_year]
        y = valid.target.to_numpy(float)
        base = valid.prediction.to_numpy(float)
        cand, params = fit_predict(arrays[fit_year], frames[fit_year].target.to_numpy(float), arrays[valid_year], alpha)
        base_m, cand_m = metric(y, base), metric(y, cand)
        ci = cluster_ci(y, base, cand, valid.pitcher_id.to_numpy(), int(contract["promotion_gate"]["cluster_bootstrap_repetitions"]), int(contract["promotion_gate"][f"cluster_bootstrap_seed_{valid_year}"]))
        gain = float(base_m["brier"] - cand_m["brier"])
        candidates[str(valid_year)] = {"fit_season": fit_year, "validation_season": valid_year, "baseline": base_m, "candidate": cand_m, "brier_gain": gain, "bss_gain": float(cand_m["bss"] - base_m["bss"]), "cluster_ci": ci, "parameters": params}
        predictions.append(pd.DataFrame({"row_id": valid.row_id, "season": valid_year, "pitcher_id": valid.pitcher_id, "target": y, "baseline_prediction": base, "candidate_prediction": cand}))

    pred = pd.concat(predictions, ignore_index=True)
    yall = pred.target.to_numpy(float); ball = pred.baseline_prediction.to_numpy(float); call = pred.candidate_prediction.to_numpy(float)
    pooled_base, pooled_cand = metric(yall, ball), metric(yall, call)
    pooled_gain = float(pooled_base["brier"] - pooled_cand["brier"])
    gate = contract["promotion_gate"]
    checks = {
        "2023_brier_gain": candidates["2023"]["brier_gain"] >= float(gate["brier_gain_2023_min"]),
        "2024_brier_gain": candidates["2024"]["brier_gain"] >= float(gate["brier_gain_2024_min"]),
        "pooled_brier_gain": pooled_gain >= float(gate["pooled_brier_gain_min"]),
        "worst_season_bss_gain": min(candidates[y]["bss_gain"] for y in ("2023", "2024")) > 0,
        "2023_cluster_ci_low": candidates["2023"]["cluster_ci"]["ci_low"] > 0,
        "2024_cluster_ci_low": candidates["2024"]["cluster_ci"]["ci_low"] > 0,
    }
    passed = all(checks.values())
    production = None
    if passed:
        xprod = np.vstack([arrays[2023], arrays[2024]])
        yprod = np.r_[frames[2023].target.to_numpy(float), frames[2024].target.to_numpy(float)]
        _, production = fit_predict(xprod, yprod, xprod[:1], alpha)
        production.update({"fit_seasons": [2023, 2024], "fit_rows": int(len(yprod)), "feature_order": contract["fixed_features"]})
        (OUT / "production_stack.json").write_text(json.dumps(production, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    pred.to_csv(OUT / "oof_predictions.csv", index=False)
    result = {
        "experiment_id": contract["experiment_id"], "candidate_name": "rolling_recent_ridge_alpha1000",
        "candidate_status": "PENDING_AUDIT_PASS" if passed else "PENDING_AUDIT_FAIL",
        "source_checks": source_checks, "folds": candidates,
        "pooled": {"baseline": pooled_base, "candidate": pooled_cand, "brier_gain": pooled_gain},
        "gate_checks": checks, "gate_checks_count": len(checks), "promotion_pass": passed,
        "actual_leaf_count": 1, "production_stack_created": production is not None,
        "test_read": False, "test_inference_performed": False, "candidate_bundle_created": False, "zip_created": False,
    }
    (OUT / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [f"# {contract['experiment_id']}", "", f"- candidate: `{result['candidate_name']}`", f"- status: `{result['candidate_status']}`", f"- promotion pass: `{str(passed).lower()}`", "", "| validation | baseline Brier | candidate Brier | gain | CI low | CI high |", "|---|---:|---:|---:|---:|---:|"]
    for y in ("2023", "2024"):
        d = candidates[y]
        lines.append(f"| {y} | {d['baseline']['brier']:.12f} | {d['candidate']['brier']:.12f} | {d['brier_gain']:.12f} | {d['cluster_ci']['ci_low']:.12f} | {d['cluster_ci']['ci_high']:.12f} |")
    lines += [f"| pooled | {pooled_base['brier']:.12f} | {pooled_cand['brier']:.12f} | {pooled_gain:.12f} |  |  |", "", "```json", json.dumps({"candidate_name": result["candidate_name"], "candidate_status": result["candidate_status"], "gate_checks": checks, "promotion_pass": passed}, sort_keys=True), "```"]
    (OUT / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    artifact_paths = [CONTRACT, OUT / "preflight_report.json", OUT / "preflight_report.md", OUT / "result.json", OUT / "result.md", OUT / "oof_predictions.csv", REGIME, MANIFEST, ROOT / "scripts/preflight_ref4_trainonly_recent_stack_041a.py", ROOT / "scripts/run_ref4_trainonly_recent_stack_041a.py", ROOT / "scripts/verify_ref4_trainonly_recent_stack_041a.py", ROOT / "01_제약과금지사항.md", ROOT / "start04_uptostage.md", ROOT / contract["preserve_zip"]] + [ROOT / p for p in contract["source_folds"].values()]
    if production is not None:
        artifact_paths.append(OUT / "production_stack.json")
    artifacts = {str(p.relative_to(ROOT)): {"sha256": sha256(p), "size": p.stat().st_size} for p in artifact_paths}
    audit_manifest = {"experiment_id": contract["experiment_id"], "status": "PENDING_VALIDATION", "artifact_count": len(artifacts), "artifacts": artifacts, "leaf_count": 1, "gate_count": len(checks), "oof_rows": int(len(pred))}
    (OUT / "audit_manifest.json").write_text(json.dumps(audit_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"candidate_status": result["candidate_status"], "promotion_pass": passed, "gate_checks": checks, "brier_gains": {y: candidates[y]["brier_gain"] for y in ("2023", "2024")}, "pooled_brier_gain": pooled_gain}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
