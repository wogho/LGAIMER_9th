#!/usr/bin/env python3
"""Read-only structural diagnosis of exact REF4 OOF for 2022-2024."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "REF4-OOF-DIAG-034A"
OUT = ROOT / "model" / EXPERIMENT_ID
BASE = ROOT / "model" / "REF4-EXACT-OOF-031A"
OOF22 = ROOT / "model" / "REF4-ADAPTIVE-GATE-031B"
SOURCE = ROOT / "model" / "REF4-CHAMPION-STACK-030"
RAW_PATH = ROOT / "data" / "train.csv"


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def metric(y: np.ndarray, prediction: np.ndarray) -> dict[str, float | int | None]:
    rate = float(y.mean())
    brier = float(np.mean((y - prediction) ** 2))
    denominator = rate * (1.0 - rate)
    bss = float(1.0 - brier / denominator) if denominator > 0 else None
    return {
        "rows": int(len(y)), "target_rate": rate, "prediction_mean": float(prediction.mean()),
        "mean_bias": float(prediction.mean() - rate), "brier": brier,
        "bss": bss, "local_score": 100000.0 * bss if bss is not None else None,
    }


def prior_type_table(raw: pd.DataFrame, year: int) -> pd.Series:
    past = raw.loc[raw["season"].lt(year)]
    counts = past.groupby(["pitcher_id", "season", "game_type"], observed=True).size().rename("n").reset_index()
    dominant = counts.sort_values("n").groupby(["pitcher_id", "season"], observed=True).tail(1)
    latest = dominant.sort_values("season").groupby("pitcher_id", observed=True).tail(1)
    return latest.set_index(latest["pitcher_id"].astype(str))["game_type"].astype(str)


def recombine(oof: pd.DataFrame, regime: dict[str, float], manifest: dict[str, object]) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    is_f = oof["game_type"].eq("F").to_numpy()
    p0g = oof["p_v2_global"].to_numpy(float); p0f = oof["p_v2_f"].to_numpy(float)
    p1g = oof["p_v3_55_global"].to_numpy(float); p1f = oof["p_v3_55_f"].to_numpy(float)
    p2g = oof["p_v3_30_global"].to_numpy(float)
    p2a = oof["p_v3_30_f_all"].to_numpy(float); p2r = oof["p_v3_30_f_recent"].to_numpy(float)
    p0 = np.where(is_f, p0g + regime["v2_scale"] * (p0f - p0g), p0g)
    p1 = np.where(is_f, p1g + regime["v355_scale"] * (p1f - p1g), p1g)
    recent_inner = p2g + regime["v330_recent_inner_scale"] * (p2r - p2g)
    f30 = regime["v330_all_weight"] * p2a + (1.0 - regime["v330_all_weight"]) * recent_inner
    p2 = np.where(is_f, p2g + regime["v330_scale"] * (f30 - p2g), p2g)
    risks = []
    for name in ("middle", "wild", "reverse"):
        global_risk = oof[f"risk_{name}_global"].to_numpy(float)
        f_risk = oof[f"risk_{name}_f"].to_numpy(float)
        risks.append(np.where(is_f, global_risk + regime["subtype_scale"] * (f_risk - global_risk), global_risk))
    main = np.average(np.vstack([p0, p1, p2]), axis=0, weights=np.asarray(manifest["main_weights"], float))
    no_shift = float(manifest["stack_intercept"]) + np.column_stack([main, *risks]) @ np.asarray(manifest["stack_coefficients"], float)
    final = np.clip(no_shift + float(manifest["global_shift"]), 1e-5, 1 - 1e-5)
    components = {
        "p_v2_adjusted": p0, "p_v3_55_adjusted": p1, "p_v3_30_adjusted": p2,
        "risk_middle_adjusted": risks[0], "risk_wild_adjusted": risks[1],
        "risk_reverse_adjusted": risks[2], "prediction_no_shift_raw": no_shift,
    }
    return final, components


def add_slice_rows(rows: list[dict[str, object]], frame: pd.DataFrame, dimension: str) -> None:
    for (year, value), part in frame.groupby(["season", dimension], observed=True, sort=True, dropna=False):
        y = part["target"].to_numpy(float)
        current = part["prediction_current"].to_numpy(float)
        no_shift = part["prediction_no_shift"].to_numpy(float)
        global_only = part["prediction_global_only"].to_numpy(float)
        f075 = part["prediction_f075"].to_numpy(float)
        current_metric = metric(y, current); no_shift_metric = metric(y, no_shift)
        global_metric = metric(y, global_only); f075_metric = metric(y, f075)
        rows.append({
            "dimension": dimension, "value": str(value), "season": int(year),
            "rows": int(len(part)), "target_rate": current_metric["target_rate"],
            "current_prediction_mean": current_metric["prediction_mean"],
            "current_brier": current_metric["brier"], "current_bss": current_metric["bss"],
            "no_shift_brier": no_shift_metric["brier"],
            "shift_removal_brier_gain": float(current_metric["brier"] - no_shift_metric["brier"]),
            "global_only_brier": global_metric["brier"],
            "global_only_brier_gain": float(current_metric["brier"] - global_metric["brier"]),
            "f075_brier": f075_metric["brier"],
            "f075_brier_gain": float(current_metric["brier"] - f075_metric["brier"]),
        })


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    preflight = json.loads((OUT / "preflight_report.json").read_text(encoding="utf-8"))
    if preflight.get("status") != "AUDIT_VERIFIED" or preflight.get("mismatch_count") != 0:
        raise RuntimeError("034A preflight must be AUDIT_VERIFIED")

    manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    current_regime = {key: float(value) for key, value in json.loads((SOURCE / "f_regime_meta.json").read_text(encoding="utf-8")).items()}
    global_regime = dict(current_regime)
    for key in ("v2_scale", "v355_scale", "v330_scale", "subtype_scale"):
        global_regime[key] = 0.0
    f075_regime = dict(current_regime)
    for key in ("v2_scale", "v355_scale", "v330_scale", "subtype_scale"):
        f075_regime[key] *= 0.75

    input_paths = {2022: OOF22 / "oof_2022.csv", 2023: BASE / "oof_2023.csv", 2024: BASE / "oof_2024.csv"}
    mapping_paths = {
        2022: OOF22 / "fold_2022" / "pitcher_trackman_mapping.csv",
        2023: BASE / "fold_2023" / "pitcher_trackman_mapping.csv",
        2024: BASE / "fold_2024" / "pitcher_trackman_mapping.csv",
    }
    raw = pd.read_csv(RAW_PATH, low_memory=False)
    all_parts: list[pd.DataFrame] = []
    component_summary: dict[str, object] = {}
    for year in (2022, 2023, 2024):
        oof = pd.read_csv(input_paths[year], dtype={"row_id": str, "game_type": str, "pitcher_id": str})
        raw_year = raw.loc[raw["season"].eq(year)].reset_index(drop=True)
        if not np.array_equal(oof["row_id"].astype(str).to_numpy(), raw_year["row_id"].astype(str).to_numpy()):
            raise RuntimeError(f"row order mismatch for {year}")
        current, components = recombine(oof, current_regime, manifest)
        global_only, _ = recombine(oof, global_regime, manifest)
        f075, _ = recombine(oof, f075_regime, manifest)
        if float(np.max(np.abs(current - oof["prediction"].to_numpy(float)))) > 1e-12:
            raise RuntimeError(f"current formula mismatch for {year}")
        no_shift = np.clip(oof["prediction_no_shift"].to_numpy(float), 1e-5, 1 - 1e-5)
        prior = prior_type_table(raw, year)
        pitcher = raw_year["pitcher_id"].astype(str)
        prior_type = pitcher.map(prior).fillna("NEW").astype(str)
        trackman_ids = set(pd.read_csv(mapping_paths[year], dtype={"pitcher_id": str})["pitcher_id"].astype(str))
        pitcher_n = pd.to_numeric(raw_year["asof_pitcher_n"], errors="coerce").fillna(0)
        pitcher_n_bin = pd.cut(pitcher_n, [-1, 0, 9, 49, 199, np.inf], labels=["0", "1-9", "10-49", "50-199", "200+"]).astype(str)
        part = pd.DataFrame({
            "row_id": oof["row_id"].astype(str), "season": year,
            "game_type": raw_year["game_type"].astype(str), "pitcher_id": pitcher,
            "target": raw_year["control_success"].to_numpy(float),
            "game_month": pd.to_numeric(raw_year["game_month"], errors="coerce").fillna(-1).astype(int),
            "known_status": np.where(prior_type.eq("NEW"), "NEW", "KNOWN"),
            "prior_type": prior_type, "transition": prior_type + ">" + raw_year["game_type"].astype(str),
            "pitcher_n_bin": pitcher_n_bin, "trackman_status": np.where(pitcher.isin(trackman_ids), "MAPPED", "UNMAPPED"),
            "prediction_current": current, "prediction_no_shift": no_shift,
            "prediction_global_only": global_only, "prediction_f075": f075,
        })
        for name, values in components.items():
            part[name] = values
        all_parts.append(part)
        channels = np.column_stack([components["p_v2_adjusted"], components["p_v3_55_adjusted"], components["p_v3_30_adjusted"]])
        component_summary[str(year)] = {
            name: {"mean": float(values.mean()), "std": float(values.std())}
            for name, values in components.items()
        }
        component_summary[str(year)]["ensemble_disagreement"] = {
            "std_mean": float(channels.std(axis=1).mean()),
            "std_q95": float(np.quantile(channels.std(axis=1), 0.95)),
            "range_mean": float((channels.max(axis=1) - channels.min(axis=1)).mean()),
        }
    diagnostic = pd.concat(all_parts, ignore_index=True)
    diagnostic.to_csv(OUT / "diagnostic_rows.csv", index=False)

    overall: dict[str, object] = {}
    for year in (2022, 2023, 2024):
        part = diagnostic.loc[diagnostic["season"].eq(year)]
        y = part["target"].to_numpy(float)
        variants = {
            "current": part["prediction_current"].to_numpy(float),
            "no_shift": part["prediction_no_shift"].to_numpy(float),
            "global_only": part["prediction_global_only"].to_numpy(float),
            "f075": part["prediction_f075"].to_numpy(float),
        }
        metrics = {name: metric(y, prediction) for name, prediction in variants.items()}
        overall[str(year)] = {
            "metrics": metrics,
            "shift_removal_brier_gain": float(metrics["current"]["brier"] - metrics["no_shift"]["brier"]),
            "global_only_brier_gain": float(metrics["current"]["brier"] - metrics["global_only"]["brier"]),
            "f075_brier_gain": float(metrics["current"]["brier"] - metrics["f075"]["brier"]),
            "f_rows": int(part["game_type"].eq("F").sum()),
        }

    slice_rows: list[dict[str, object]] = []
    for dimension in ("game_type", "known_status", "transition", "pitcher_n_bin", "game_month", "trackman_status"):
        add_slice_rows(slice_rows, diagnostic, dimension)
    slice_frame = pd.DataFrame(slice_rows).sort_values(["dimension", "season", "value"]).reset_index(drop=True)
    slice_frame.to_csv(OUT / "slice_metrics.csv", index=False)

    calibration_rows: list[dict[str, object]] = []
    for year, part in diagnostic.groupby("season", sort=True):
        bins = np.minimum((part["prediction_current"].to_numpy(float) * 10).astype(int), 9)
        temp = part.assign(calibration_bin=bins)
        for bin_id, group in temp.groupby("calibration_bin", sort=True):
            calibration_rows.append({
                "season": int(year), "bin": int(bin_id), "rows": int(len(group)),
                "target_rate": float(group["target"].mean()),
                "prediction_mean": float(group["prediction_current"].mean()),
                "mean_bias": float(group["prediction_current"].mean() - group["target"].mean()),
                "brier": float(np.mean((group["target"] - group["prediction_current"]) ** 2)),
            })
    calibration = pd.DataFrame(calibration_rows)
    calibration.to_csv(OUT / "calibration_bins.csv", index=False)

    cluster_rows: list[dict[str, object]] = []
    cluster_summary: dict[str, object] = {}
    for year, part in diagnostic.groupby("season", sort=True):
        rate = float(part["target"].mean())
        temp = part.assign(
            squared_loss=(part["target"] - part["prediction_current"]) ** 2,
            reference_loss=(part["target"] - rate) ** 2,
        )
        temp["excess_loss"] = temp["squared_loss"] - temp["reference_loss"]
        grouped = temp.groupby("pitcher_id", sort=True).agg(
            rows=("row_id", "size"), target_rate=("target", "mean"),
            prediction_mean=("prediction_current", "mean"), squared_loss=("squared_loss", "sum"),
            reference_loss=("reference_loss", "sum"), excess_loss=("excess_loss", "sum"),
        ).reset_index()
        grouped.insert(0, "season", int(year))
        cluster_rows.extend(grouped.to_dict(orient="records"))
        positive = grouped.loc[grouped["excess_loss"].gt(0), "excess_loss"].sort_values(ascending=False)
        total_positive = float(positive.sum())
        cluster_summary[str(year)] = {
            "clusters": int(len(grouped)), "positive_excess_clusters": int(len(positive)),
            "total_excess_loss": float(grouped["excess_loss"].sum()),
            "total_positive_excess_loss": total_positive,
            "top_10_positive_share": float(positive.head(10).sum() / total_positive) if total_positive > 0 else 0.0,
            "top_20_positive_share": float(positive.head(20).sum() / total_positive) if total_positive > 0 else 0.0,
        }
    pitcher_clusters = pd.DataFrame(cluster_rows)
    pitcher_clusters.to_csv(OUT / "pitcher_clusters.csv", index=False)

    f_slice = slice_frame.loc[(slice_frame["dimension"] == "game_type") & (slice_frame["value"] == "F")].copy()
    f_slice_summary = {
        str(int(row.season)): {
            "rows": int(row.rows), "current_brier": float(row.current_brier),
            "f075_brier": float(row.f075_brier), "f075_brier_gain": float(row.f075_brier_gain),
        }
        for row in f_slice.itertuples(index=False)
    }
    findings = {
        "shift_removal_positive_years": [year for year in ("2022", "2023", "2024") if overall[year]["shift_removal_brier_gain"] > 0],
        "global_only_positive_years": [year for year in ("2022", "2023", "2024") if overall[year]["global_only_brier_gain"] > 0],
        "f075_positive_years": [year for year in ("2022", "2023", "2024") if overall[year]["f075_brier_gain"] > 0],
        "f075_f_slice_positive_years": [year for year in ("2022", "2023", "2024") if f_slice_summary[year]["f075_brier_gain"] > 0],
        "root_cause_status": "DIAGNOSTIC_ONLY_NOT_CAUSAL",
        "next_experiments": ["REF4-SHIFT-034B", "REF4-F-REGIME-032B"],
    }
    result = {
        "experiment_id": EXPERIMENT_ID, "status": "PENDING_AUDIT",
        "scope": "read_only_three_year_oof_diagnosis", "source_official_score": 1068.25021,
        "candidate_count": 0, "actual_leaf_count": 0, "gate_checks_count": 0,
        "model_count": 0, "oof_rows": int(len(diagnostic)), "slice_rows": int(len(slice_frame)),
        "calibration_rows": int(len(calibration)), "pitcher_cluster_rows": int(len(pitcher_clusters)),
        "overall": overall, "f_slice": f_slice_summary,
        "component_summary": component_summary, "cluster_summary": cluster_summary,
        "findings": findings, "test_inference_performed": False,
        "training_performed": False, "zip_created": False, "elapsed_seconds": time.time() - started,
    }
    result_path = OUT / "result.json"
    write_json(result_path, result)
    embedded = json.dumps({
        "candidate_count": 0, "actual_leaf_count": 0, "gate_checks_count": 0,
        "oof_rows": result["oof_rows"], "slice_rows": result["slice_rows"],
        "overall": overall, "f_slice": f_slice_summary, "findings": findings,
    }, sort_keys=True)
    lines = [
        f"# {EXPERIMENT_ID}", "", "- status: `PENDING_AUDIT`", "- candidate leaf count: `0`",
        f"- three-year OOF rows: `{result['oof_rows']}`", f"- slice metric rows: `{result['slice_rows']}`",
        "- training/test/ZIP: `false/false/false`", "",
        "| year | target mean | current mean | current Brier | no-shift gain | global-only gain | F075 gain | F rows |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for year in ("2022", "2023", "2024"):
        item = overall[year]
        current = item["metrics"]["current"]
        lines.append(
            f"| {year} | {current['target_rate']:.12f} | {current['prediction_mean']:.12f} | "
            f"{current['brier']:.12f} | {item['shift_removal_brier_gain']:.12g} | "
            f"{item['global_only_brier_gain']:.12g} | {item['f075_brier_gain']:.12g} | {item['f_rows']} |"
        )
    lines.extend(["", "<!-- RESULT_JSON_BEGIN", embedded, "RESULT_JSON_END -->"])
    result_md = OUT / "result.md"
    result_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    artifact_paths = [
        ROOT / "scripts" / "preflight_ref4_oof_diag_034a.py",
        ROOT / "scripts" / "run_ref4_oof_diag_034a.py",
        ROOT / "scripts" / "verify_ref4_oof_diag_034a.py",
        OUT / "preflight_report.json", OUT / "preflight_report.md",
        OUT / "diagnostic_rows.csv", OUT / "slice_metrics.csv", OUT / "calibration_bins.csv",
        OUT / "pitcher_clusters.csv", result_path, result_md,
        BASE / "audit_manifest.json", BASE / "audit_attestation.json",
        OOF22 / "audit_manifest.json", OOF22 / "audit_attestation.json",
        *input_paths.values(), *mapping_paths.values(), RAW_PATH,
        SOURCE / "manifest.json", SOURCE / "f_regime_meta.json",
        ROOT / "start03_reference.md", ROOT / "01_제약과금지사항.md",
        ROOT / "output" / "submit_ref4_champion_030.zip",
    ]
    records = {
        str(path.relative_to(ROOT)): {"sha256": sha256_path(path), "size": path.stat().st_size}
        for path in artifact_paths
    }
    audit_manifest = {
        "experiment_id": EXPERIMENT_ID, "status": "PENDING_VALIDATION",
        "artifact_count": len(records), "artifacts": records,
        "candidate_count": 0, "leaf_count": 0, "gate_checks_count": 0,
        "model_count": 0, "oof_rows": result["oof_rows"], "slice_rows": result["slice_rows"],
    }
    write_json(OUT / "audit_manifest.json", audit_manifest)
    print(json.dumps({
        "status": "PENDING_AUDIT", "candidate_count": 0, "model_count": 0,
        "oof_rows": result["oof_rows"], "slice_rows": result["slice_rows"],
        "findings": findings, "elapsed_seconds": result["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
