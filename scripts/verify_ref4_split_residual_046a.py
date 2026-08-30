#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model/REF4-TRAINONLY-SPLIT-RESIDUAL-046A"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def independent_keys(frame: pd.DataFrame) -> pd.DataFrame:
    keys = pd.DataFrame(index=frame.index)
    balls = pd.to_numeric(frame["balls_before"], errors="coerce").fillna(0).astype(int)
    strikes = pd.to_numeric(frame["strikes_before"], errors="coerce").fillna(0).astype(int)
    inning = pd.to_numeric(frame["inning"], errors="coerce").fillna(1).clip(1, 10).astype(int)
    keys["pitcher_id"] = frame["pitcher_id"].astype(str)
    keys["batter_id"] = frame["batter_id"].astype(str)
    keys["pitcher_opponent_hand"] = frame["batter_hand"].astype(str)
    keys["batter_opponent_hand"] = frame["pitcher_hand"].astype(str)
    keys["count_state"] = balls.astype(str) + "-" + strikes.astype(str)
    keys["count_ahead"] = np.where(strikes > balls, "ahead", "not_ahead")
    keys["inning_bucket"] = np.where(inning <= 3, "early", np.where(inning <= 6, "middle", "late"))
    keys["base_state"] = frame["base_state"].fillna("___").astype(str)
    return keys


def independent_features(history: pd.DataFrame, rows: pd.DataFrame, shrinkages) -> pd.DataFrame:
    hk = independent_keys(history)
    rk = independent_keys(rows)
    season_rates = history.groupby("season", observed=True)["control_success"].transform("mean")
    hk["relative"] = history.control_success.to_numpy(float) - season_rates.to_numpy(float)
    output = {}
    for entity in ("pitcher_id", "batter_id"):
        for context in ("opponent_hand", "count_state", "count_ahead", "inning_bucket", "base_state"):
            context_key = f"{entity.split('_')[0]}_opponent_hand" if context == "opponent_hand" else context
            cell = hk.groupby([entity, context_key], observed=True)["relative"].agg(total="sum", count="size")
            entity_table = hk.groupby(entity, observed=True)["relative"].agg(entity_total="sum", entity_count="size")
            cell = cell.join(entity_table)
            cell["effect"] = cell.total / cell["count"] - cell.entity_total / cell.entity_count
            lookup = pd.MultiIndex.from_arrays([rk[entity], rk[context_key]])
            effect = np.nan_to_num(lookup.map(cell.effect).to_numpy(float), nan=0.0)
            count = np.nan_to_num(lookup.map(cell["count"]).to_numpy(float), nan=0.0)
            for shrinkage in shrinkages:
                output[f"split_{entity}_{context}_k{int(shrinkage)}"] = effect * count / (count + shrinkage)
    return pd.DataFrame(output, index=rows.index)


def metric(target, prediction):
    brier = float(np.mean((prediction - target) ** 2))
    reference = float(np.mean((target - target.mean()) ** 2))
    return brier, 1.0 - brier / reference


def independent_ci(target, base, candidate, groups, repetitions, seed):
    delta = (base - target) ** 2 - (candidate - target) ** 2
    codes, unique = pd.factorize(groups, sort=True)
    sums = np.bincount(codes, weights=delta, minlength=len(unique))
    counts = np.bincount(codes, minlength=len(unique))
    rng = np.random.default_rng(seed)
    values = np.empty(repetitions)
    for index in range(repetitions):
        chosen = rng.integers(0, len(unique), len(unique))
        values[index] = sums[chosen].sum() / counts[chosen].sum()
    return float(delta.mean()), float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975)), len(unique)


def main() -> None:
    contract = json.loads((OUT / "audit_contract.json").read_text())
    result = json.loads((OUT / "result.json").read_text())
    manifest = json.loads((OUT / "audit_manifest.json").read_text())
    checks = []

    def check(name, passed, actual, expected=None):
        checks.append({"name": name, "checked": True, "pass": bool(passed), "actual": actual, "expected": expected})

    mismatched = []
    for relative, evidence in manifest["artifacts"].items():
        path = ROOT / relative
        if not path.is_file() or sha256(path) != evidence["sha256"] or path.stat().st_size != evidence["size"]:
            mismatched.append(relative)
    check("manifest_artifacts", not mismatched, mismatched, [])
    check("manifest_count", manifest["artifact_count"] == len(manifest["artifacts"]), len(manifest["artifacts"]), manifest["artifact_count"])

    columns = [
        "row_id", "season", "control_success", "pitcher_id", "batter_id", "pitcher_hand",
        "batter_hand", "balls_before", "strikes_before", "inning", "base_state",
    ]
    raw = pd.read_csv(ROOT / contract["official_train"], usecols=columns)
    base_oof = {int(year): pd.read_csv(ROOT / path) for year, path in contract["base_oof"].items()}
    shrinkages = tuple(float(value) for value in contract["feature_contract"]["shrinkage_grid"])
    frames = {}
    for year in (2022, 2023, 2024):
        current = base_oof[year][["row_id"]].merge(raw, on="row_id", how="left", validate="one_to_one")
        x = independent_features(raw.loc[raw.season < year].copy(), current, shrinkages)
        frames[year] = x
        audit = result["feature_audits"][str(year)]
        check(
            f"features_{year}",
            len(x) == audit["feature_rows"] and x.shape[1] == audit["feature_count"]
            and x.columns.tolist() == audit["feature_names"] and np.isfinite(x.to_numpy()).all(),
            {"rows": len(x), "count": x.shape[1], "finite": bool(np.isfinite(x.to_numpy()).all())},
            {"rows": audit["feature_rows"], "count": audit["feature_count"], "finite": True},
        )

    saved = pd.read_csv(OUT / "oof_predictions.csv")
    recomputed_parts = []
    recomputed_folds = {}
    for valid_year, fit_years in ((2023, (2022,)), (2024, (2022, 2023))):
        train_x = pd.concat([frames[year] for year in fit_years], ignore_index=True)
        train_y = np.concatenate([
            base_oof[year].target.to_numpy(float) - base_oof[year].prediction.to_numpy(float) for year in fit_years
        ])
        sample_weight = np.concatenate([
            np.full(len(base_oof[year]), 0.55 if valid_year == 2024 and year == 2022 else 1.0) for year in fit_years
        ])
        mean = train_x.mean(axis=0).to_numpy(float)
        std = train_x.std(axis=0).replace(0, 1).fillna(1).to_numpy(float)
        z_train = np.nan_to_num((train_x.to_numpy(float) - mean) / std)
        z_valid = np.nan_to_num((frames[valid_year].to_numpy(float) - mean) / std)
        ridge = Ridge(alpha=float(contract["ridge_contract"]["alpha"]), fit_intercept=False)
        ridge.fit(z_train, train_y, sample_weight=sample_weight)
        correction = ridge.predict(z_valid)
        target = base_oof[valid_year].target.to_numpy(float)
        base = base_oof[valid_year].prediction.to_numpy(float)
        candidate = np.clip(base + correction, 1e-6, 1 - 1e-6)
        stored = saved.loc[saved.season == valid_year].reset_index(drop=True)
        max_difference = float(np.max(np.abs(candidate - stored.candidate_prediction.to_numpy(float))))
        fold_result = result["folds"][str(valid_year)]
        base_brier, base_bss = metric(target, base)
        candidate_brier, candidate_bss = metric(target, candidate)
        ci = independent_ci(
            target, base, candidate, base_oof[valid_year].pitcher_id.to_numpy(),
            int(contract["promotion_gate"]["cluster_bootstrap_repetitions"]),
            int(contract["promotion_gate"][f"cluster_bootstrap_seed_{valid_year}"]),
        )
        numeric_difference = max(
            max_difference,
            float(np.max(np.abs(mean - np.asarray(fold_result["standardization_mean"])))),
            float(np.max(np.abs(std - np.asarray(fold_result["standardization_std"])))),
            float(np.max(np.abs(ridge.coef_ - np.asarray(fold_result["ridge_coef"])))),
            abs((base_brier - candidate_brier) - fold_result["brier_gain"]),
            abs((candidate_bss - base_bss) - fold_result["bss_gain"]),
            abs(ci[1] - fold_result["cluster_ci"]["ci_low"]),
            abs(ci[2] - fold_result["cluster_ci"]["ci_high"]),
        )
        check(
            f"independent_fold_{valid_year}",
            stored.row_id.tolist() == base_oof[valid_year].row_id.tolist() and numeric_difference <= 1e-12,
            {"rows": len(stored), "max_numeric_difference": numeric_difference},
            {"rows": len(base_oof[valid_year]), "max_numeric_difference": 1e-12},
        )
        recomputed_folds[str(valid_year)] = {
            "brier_gain": base_brier - candidate_brier,
            "bss_gain": candidate_bss - base_bss,
            "ci_low": ci[1],
            "ci_high": ci[2],
        }
        recomputed_parts.append((target, base, candidate))

    target = np.concatenate([part[0] for part in recomputed_parts])
    base = np.concatenate([part[1] for part in recomputed_parts])
    candidate = np.concatenate([part[2] for part in recomputed_parts])
    pooled_gain = float(np.mean((base - target) ** 2) - np.mean((candidate - target) ** 2))
    gates = {
        "2023_brier_gain_positive": recomputed_folds["2023"]["brier_gain"] > 0,
        "2024_brier_gain_positive": recomputed_folds["2024"]["brier_gain"] > 0,
        "pooled_brier_gain_positive": pooled_gain > 0,
        "worst_season_bss_gain_positive": min(recomputed_folds[y]["bss_gain"] for y in ("2023", "2024")) > 0,
        "2023_cluster_ci_low_positive": recomputed_folds["2023"]["ci_low"] > 0,
        "2024_cluster_ci_low_positive": recomputed_folds["2024"]["ci_low"] > 0,
    }
    promotion = all(gates.values())
    check("pooled_metric", abs(pooled_gain - result["pooled"]["brier_gain"]) <= 1e-15, pooled_gain, result["pooled"]["brier_gain"])
    check("gate_recalculation", gates == result["gate_checks"], gates, result["gate_checks"])
    check("counts", result["actual_leaf_count"] == manifest["leaf_count"] == 1 and result["gate_checks_count"] == manifest["gate_count"] == len(gates), [result["actual_leaf_count"], manifest["leaf_count"], result["gate_checks_count"], manifest["gate_count"], len(gates)])
    check("promotion", result["promotion_pass"] is promotion, result["promotion_pass"], promotion)
    check(
        "scope",
        result["training_performed"] is True and not any(result[key] for key in ("test_read", "test_inference_performed", "production_assets_created", "candidate_bundle_created", "zip_created")),
        {key: result[key] for key in ("training_performed", "test_read", "test_inference_performed", "production_assets_created", "candidate_bundle_created", "zip_created")},
    )
    markdown = (OUT / "result.md").read_text()
    check("markdown", result["candidate_name"] in markdown and result["candidate_status"] in markdown and f"`{str(promotion).lower()}`" in markdown, result["candidate_name"])

    failures = [item["name"] for item in checks if not item["pass"]]
    report = {
        "experiment_id": contract["experiment_id"],
        "status": "AUDIT_VERIFIED" if not failures else "AUDIT_FAIL_REPORT",
        "checked_count": len(checks), "pass_count": len(checks) - len(failures),
        "fail_count": len(failures), "mismatch_count": len(failures), "failures": failures,
        "checks": checks, "recomputed_folds": recomputed_folds,
        "pooled_brier_gain": pooled_gain, "expected_gate": gates, "promotion_pass": promotion,
    }
    report_path = OUT / "validation_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    attestation = {
        "experiment_id": contract["experiment_id"], "overall_status": report["status"],
        "performance_status": "PASS" if promotion else "FAIL", "candidate_count": 1,
        "leaf_count": 1, "gate_count": len(gates), "checked_count": len(checks),
        "pass_count": len(checks) - len(failures), "fail_count": len(failures),
        "mismatch_count": len(failures), "audit_manifest_sha256": sha256(OUT / "audit_manifest.json"),
        "validation_report_sha256": sha256(report_path), "validator_sha256": sha256(Path(__file__)),
        "result_sha256": sha256(OUT / "result.json"), "oof_predictions_sha256": sha256(OUT / "oof_predictions.csv"),
        "preserved_zip_sha256": sha256(ROOT / contract["preserve_zip"]), "promotion_pass": promotion,
        "training_performed": True, "test_read": False, "test_inference_performed": False,
        "production_assets_created": False, "candidate_bundle_created": False, "zip_created": False,
    }
    (OUT / "audit_attestation.json").write_text(json.dumps(attestation, indent=2) + "\n")
    print(json.dumps({key: attestation[key] for key in ("overall_status", "performance_status", "checked_count", "pass_count", "fail_count", "mismatch_count", "promotion_pass")}, indent=2))
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
