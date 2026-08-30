#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.entity_context_split import build_split_features


OUT = ROOT / "model/REF4-TRAINONLY-SPLIT-RESIDUAL-046A"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def metrics(target: np.ndarray, prediction: np.ndarray) -> dict:
    brier = float(np.mean((prediction - target) ** 2))
    reference = float(np.mean((target - target.mean()) ** 2))
    bss = 1.0 - brier / reference
    return {"rows": len(target), "target_rate": float(target.mean()), "brier": brier, "bss": bss, "local_score": 1e5 * bss}


def cluster_ci(target, base, candidate, groups, repetitions, seed):
    gain = (base - target) ** 2 - (candidate - target) ** 2
    codes, unique = pd.factorize(groups, sort=True)
    sums = np.bincount(codes, weights=gain, minlength=len(unique))
    counts = np.bincount(codes, minlength=len(unique))
    rng = np.random.default_rng(seed)
    draws = np.empty(repetitions)
    for idx in range(repetitions):
        sample = rng.integers(0, len(unique), len(unique))
        draws[idx] = sums[sample].sum() / counts[sample].sum()
    return {
        "clusters": len(unique),
        "repetitions": repetitions,
        "seed": seed,
        "brier_gain": float(gain.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def main() -> None:
    started = time.time()
    contract = json.loads((OUT / "audit_contract.json").read_text())
    preflight = json.loads((OUT / "preflight_report.json").read_text())
    assert preflight["status"] == "AUDIT_VERIFIED" and preflight["fail_count"] == 0

    raw_columns = [
        "row_id", "season", "control_success", "pitcher_id", "batter_id", "pitcher_hand",
        "batter_hand", "balls_before", "strikes_before", "inning", "base_state",
    ]
    raw = pd.read_csv(ROOT / contract["official_train"], usecols=raw_columns)
    oof = {int(year): pd.read_csv(ROOT / rel) for year, rel in contract["base_oof"].items()}
    rows = {}
    features = {}
    audits = {}
    shrinkages = tuple(float(value) for value in contract["feature_contract"]["shrinkage_grid"])
    for year in (2022, 2023, 2024):
        current = oof[year][["row_id"]].merge(raw, on="row_id", how="left", validate="one_to_one")
        history = raw.loc[raw.season < year].copy()
        x = build_split_features(history, current, shrinkages)
        rows[year] = current
        features[year] = x
        audits[str(year)] = {
            "history_rows": len(history),
            "history_max_season": int(history.season.max()),
            "feature_rows": len(x),
            "feature_count": x.shape[1],
            "feature_names": x.columns.tolist(),
            "finite": bool(np.isfinite(x.to_numpy()).all()),
        }
        print(f"features {year}: {x.shape}", flush=True)

    alpha = float(contract["ridge_contract"]["alpha"])
    fold_results = {}
    saved_parts = []
    for valid_year, fit_years in ((2023, (2022,)), (2024, (2022, 2023))):
        train_x = pd.concat([features[year] for year in fit_years], ignore_index=True)
        train_target = np.concatenate([
            oof[year].target.to_numpy(float) - oof[year].prediction.to_numpy(float) for year in fit_years
        ])
        fit_weights = np.concatenate([
            np.full(len(oof[year]), 0.55 if valid_year == 2024 and year == 2022 else 1.0) for year in fit_years
        ])
        mean = train_x.mean(axis=0).to_numpy(float)
        std = train_x.std(axis=0).replace(0, 1).fillna(1).to_numpy(float)
        z_train = np.nan_to_num((train_x.to_numpy(float) - mean) / std)
        z_valid = np.nan_to_num((features[valid_year].to_numpy(float) - mean) / std)
        model = Ridge(alpha=alpha, fit_intercept=False)
        model.fit(z_train, train_target, sample_weight=fit_weights)
        correction = model.predict(z_valid)
        target = oof[valid_year].target.to_numpy(float)
        base = oof[valid_year].prediction.to_numpy(float)
        candidate = np.clip(base + correction, 1e-6, 1 - 1e-6)
        base_metrics = metrics(target, base)
        candidate_metrics = metrics(target, candidate)
        ci = cluster_ci(
            target,
            base,
            candidate,
            oof[valid_year].pitcher_id.to_numpy(),
            int(contract["promotion_gate"]["cluster_bootstrap_repetitions"]),
            int(contract["promotion_gate"][f"cluster_bootstrap_seed_{valid_year}"]),
        )
        fold_results[str(valid_year)] = {
            "fit_seasons": list(fit_years),
            "fit_rows": len(train_x),
            "feature_count": train_x.shape[1],
            "alpha": alpha,
            "standardization_mean": mean.tolist(),
            "standardization_std": std.tolist(),
            "ridge_coef": model.coef_.tolist(),
            "baseline": base_metrics,
            "candidate": candidate_metrics,
            "brier_gain": base_metrics["brier"] - candidate_metrics["brier"],
            "bss_gain": candidate_metrics["bss"] - base_metrics["bss"],
            "mean_change": float(np.mean(candidate - base)),
            "mean_abs_change": float(np.mean(np.abs(candidate - base))),
            "max_abs_change": float(np.max(np.abs(candidate - base))),
            "cluster_ci": ci,
        }
        saved_parts.append(pd.DataFrame({
            "row_id": oof[valid_year].row_id,
            "season": valid_year,
            "pitcher_id": oof[valid_year].pitcher_id,
            "target": target,
            "baseline_prediction": base,
            "correction": correction,
            "candidate_prediction": candidate,
        }))
        print(f"fold {valid_year}: gain={fold_results[str(valid_year)]['brier_gain']:.12g}", flush=True)

    prediction_frame = pd.concat(saved_parts, ignore_index=True)
    target = prediction_frame.target.to_numpy(float)
    base = prediction_frame.baseline_prediction.to_numpy(float)
    candidate = prediction_frame.candidate_prediction.to_numpy(float)
    pooled_base = metrics(target, base)
    pooled_candidate = metrics(target, candidate)
    pooled_gain = pooled_base["brier"] - pooled_candidate["brier"]
    gates = {
        "2023_brier_gain_positive": fold_results["2023"]["brier_gain"] > 0,
        "2024_brier_gain_positive": fold_results["2024"]["brier_gain"] > 0,
        "pooled_brier_gain_positive": pooled_gain > 0,
        "worst_season_bss_gain_positive": min(fold_results[y]["bss_gain"] for y in ("2023", "2024")) > 0,
        "2023_cluster_ci_low_positive": fold_results["2023"]["cluster_ci"]["ci_low"] > 0,
        "2024_cluster_ci_low_positive": fold_results["2024"]["cluster_ci"]["ci_low"] > 0,
    }
    promotion = all(gates.values())
    result = {
        "experiment_id": contract["experiment_id"],
        "candidate_name": "fixed_entity_context_split_ridge10000",
        "candidate_status": "PENDING_AUDIT_PASS" if promotion else "PENDING_AUDIT_FAIL",
        "feature_audits": audits,
        "folds": fold_results,
        "pooled": {"baseline": pooled_base, "candidate": pooled_candidate, "brier_gain": pooled_gain},
        "gate_checks": gates,
        "gate_checks_count": len(gates),
        "promotion_pass": promotion,
        "actual_leaf_count": 1,
        "training_performed": True,
        "test_read": False,
        "test_inference_performed": False,
        "production_assets_created": False,
        "candidate_bundle_created": False,
        "zip_created": False,
        "elapsed_seconds": time.time() - started,
    }
    prediction_frame.to_csv(OUT / "oof_predictions.csv", index=False)
    (OUT / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    lines = [
        f"# {contract['experiment_id']}", "", f"- candidate: `{result['candidate_name']}`",
        f"- status: `{result['candidate_status']}`", f"- promotion pass: `{str(promotion).lower()}`", "",
        "| validation | baseline Brier | candidate Brier | gain | CI low | CI high |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for year in ("2023", "2024"):
        fold = fold_results[year]
        lines.append(
            f"| {year} | {fold['baseline']['brier']:.12f} | {fold['candidate']['brier']:.12f} | "
            f"{fold['brier_gain']:.12f} | {fold['cluster_ci']['ci_low']:.12f} | {fold['cluster_ci']['ci_high']:.12f} |"
        )
    lines.append(f"| pooled | {pooled_base['brier']:.12f} | {pooled_candidate['brier']:.12f} | {pooled_gain:.12f} |  |  |")
    (OUT / "result.md").write_text("\n".join(lines) + "\n")

    paths = [
        OUT / "audit_contract.json", OUT / "preflight_report.json", OUT / "preflight_report.md",
        OUT / "result.json", OUT / "result.md", OUT / "oof_predictions.csv",
        ROOT / contract["official_train"], ROOT / contract["preserve_zip"],
        ROOT / "01_제약과금지사항.md",
        ROOT / "src/entity_context_split.py", ROOT / "scripts/preflight_ref4_split_residual_046a.py",
        ROOT / "scripts/run_ref4_split_residual_046a.py", ROOT / "scripts/verify_ref4_split_residual_046a.py",
    ] + [ROOT / p for p in contract["base_oof"].values()] + [ROOT / p for p in contract["base_attestations"]]
    artifacts = {str(path.relative_to(ROOT)): {"sha256": sha256(path), "size": path.stat().st_size} for path in paths}
    manifest = {
        "experiment_id": contract["experiment_id"], "status": "PENDING_VALIDATION",
        "artifact_count": len(artifacts), "artifacts": artifacts, "leaf_count": 1,
        "gate_count": len(gates), "oof_rows": len(prediction_frame), "feature_count": features[2022].shape[1],
    }
    (OUT / "audit_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"candidate_status": result["candidate_status"], "promotion_pass": promotion, "gates": gates, "pooled_brier_gain": pooled_gain}, indent=2))


if __name__ == "__main__":
    main()
