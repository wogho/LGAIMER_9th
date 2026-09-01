#!/usr/bin/env python3
"""Bias-hardened R2 evaluation for TrackMan privileged distillation.

R1 demonstrated that the aligned subset has a large season-dependent mean
residual bias.  R2 changes one mechanism: privileged teacher deltas are
centered within (season, game_type), and unaligned outer-training rows are
included with a zero-correction target.  The student still receives official
pre-pitch features only.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from run_ref4_trackman_privileged_distill_120a import (
    ANCHOR_PATH,
    CALIBRATION_SCALES,
    DELTA_CAP,
    GATE_KINDS,
    OUT,
    STUDENT_CATEGORICAL,
    TM_CATEGORICAL,
    TM_NUMERIC,
    TRACKMAN_PATH,
    TRAIN_PATH,
    brier,
    crossfit_teacher,
    make_model,
    pitcher_bootstrap,
    prepare_features,
    reachability_gate,
)


R2_OUT = OUT / "r2_centered"
UNALIGNED_WEIGHT = 0.25
CENTER_KEYS = ("season", "game_type")


def group_gain(rows: pd.DataFrame, baseline: np.ndarray, candidate: np.ndarray, key: str) -> dict[str, float]:
    result: dict[str, float] = {}
    y = rows["control_success"].to_numpy(float)
    for value in sorted(rows[key].astype(str).unique()):
        mask = rows[key].astype(str).eq(value).to_numpy()
        result[value] = brier(y[mask], baseline[mask]) - brier(y[mask], candidate[mask])
    return result


def metrics(rows: pd.DataFrame, baseline: np.ndarray, candidate: np.ndarray, year: int, label: str) -> dict[str, object]:
    y = rows["control_success"].to_numpy(float)
    base_brier = brier(y, baseline)
    candidate_brier = brier(y, candidate)
    return {
        "label": label,
        "season": year,
        "rows": int(len(rows)),
        "baseline_brier": base_brier,
        "candidate_brier": candidate_brier,
        "brier_gain": base_brier - candidate_brier,
        "mean_absolute_change": float(np.mean(np.abs(candidate - baseline))),
        "max_absolute_change": float(np.max(np.abs(candidate - baseline))),
        "by_game_type": group_gain(rows, baseline, candidate, "game_type"),
        "pitcher_bootstrap": pitcher_bootstrap(
            y, baseline, candidate, rows["pitcher_id"], 120_700 + year
        ),
    }


def main() -> None:
    started = time.time()
    R2_OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(TRAIN_PATH, low_memory=False)
    trackman = pd.read_csv(TRACKMAN_PATH, low_memory=False)
    anchor = pd.read_csv(
        ANCHOR_PATH, usecols=["row_id", "season", "target", "p113a_strict"], low_memory=False
    )
    alignment = pd.read_csv(OUT / "alignment_verified.csv.gz", low_memory=False)

    validation = raw.loc[raw["season"].isin((2022, 2023, 2024))].reset_index(drop=True)
    if not np.array_equal(validation["row_id"].astype(str), anchor["row_id"].astype(str)):
        raise RuntimeError("strict anchor order mismatch")
    validation["p113a_strict"] = anchor["p113a_strict"].to_numpy(float)

    aligned = alignment.merge(
        anchor, on=["row_id", "season"], how="inner", validate="one_to_one"
    )
    official = validation.set_index("row_id", drop=False).loc[aligned["row_id"]].reset_index(drop=True)
    for column in validation.columns:
        if column not in aligned.columns:
            aligned[column] = official[column].to_numpy()
    privileged = trackman.iloc[aligned["trackman_position"].to_numpy(np.int64)].reset_index(drop=True)
    for column in (*TM_CATEGORICAL, *TM_NUMERIC):
        aligned[f"tm_{column}"] = privileged[column].to_numpy()

    excluded = {"row_id", "season", "control_success", "target", "p113a_strict"}
    student_features = [column for column in raw.columns if column not in excluded]
    student_categoricals = [column for column in STUDENT_CATEGORICAL if column in student_features]
    teacher_features = [*student_features, *[f"tm_{column}" for column in (*TM_CATEGORICAL, *TM_NUMERIC)]]
    teacher_categoricals = [*student_categoricals, *[f"tm_{column}" for column in TM_CATEGORICAL]]

    aligned_residual_audit = (
        aligned.groupby(["season", "game_type"], observed=True)
        .apply(
            lambda group: pd.Series(
                {
                    "rows": len(group),
                    "mean_residual": float((group["target"] - group["p113a_strict"]).mean()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    full_residual_audit = (
        validation.groupby(["season", "game_type"], observed=True)
        .apply(
            lambda group: pd.Series(
                {
                    "rows": len(group),
                    "mean_residual": float((group["control_success"] - group["p113a_strict"]).mean()),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    aligned_residual_audit.to_csv(R2_OUT / "aligned_residual_audit.csv", index=False)
    full_residual_audit.to_csv(R2_OUT / "full_residual_audit.csv", index=False)

    report: dict[str, object] = {
        "experiment": "REF4-TRACKMAN-PRIVILEGED-DISTILL-120A-R2",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "single_change": "season_x_game_type centered teacher target plus zero-target unaligned regularization",
        "center_keys": list(CENTER_KEYS),
        "unaligned_weight": UNALIGNED_WEIGHT,
        "test_file_read": False,
        "student_privileged_columns": [column for column in student_features if column.startswith("tm_")],
        "outer": {},
    }
    selected: dict[str, object] | None = None
    prediction_frames = []

    for outer_year in (2023, 2024):
        print(f"[R2] outer={outer_year}: teacher cross-fit", flush=True)
        aligned_fit = aligned.loc[
            aligned["season"].ge(2022) & aligned["season"].lt(outer_year)
        ].reset_index(drop=True)
        teacher_oof, teacher_report = crossfit_teacher(
            aligned_fit, teacher_features, teacher_categoricals, outer_year
        )
        np.savez_compressed(
            R2_OUT / f"teacher_oof_outer_{outer_year}.npz",
            row_id=aligned_fit["row_id"].astype(str).to_numpy(),
            season=aligned_fit["season"].to_numpy(np.int16),
            teacher_delta=teacher_oof,
        )

        target_frame = aligned_fit[["row_id", "season", "game_type"]].copy()
        target_frame["teacher_delta_raw"] = np.clip(teacher_oof, -DELTA_CAP, DELTA_CAP)
        target_frame["center"] = target_frame.groupby(
            list(CENTER_KEYS), observed=True
        )["teacher_delta_raw"].transform("mean")
        target_frame["privileged_target"] = target_frame["teacher_delta_raw"] - target_frame["center"]
        center_audit = (
            target_frame.groupby(list(CENTER_KEYS), observed=True)
            .agg(rows=("row_id", "size"), raw_mean=("teacher_delta_raw", "mean"), centered_mean=("privileged_target", "mean"))
            .reset_index()
        )
        center_audit.to_csv(R2_OUT / f"center_audit_outer_{outer_year}.csv", index=False)

        full_fit = validation.loc[
            validation["season"].ge(2022) & validation["season"].lt(outer_year)
        ].copy()
        full_fit = full_fit.merge(
            target_frame[["row_id", "privileged_target"]],
            on="row_id",
            how="left",
            validate="one_to_one",
        )
        full_fit["has_privileged_target"] = full_fit["privileged_target"].notna()
        full_fit["privileged_target"] = full_fit["privileged_target"].fillna(0.0)
        sample_weight = np.where(full_fit["has_privileged_target"], 1.0, UNALIGNED_WEIGHT)

        eval_rows = validation.loc[validation["season"].eq(outer_year)].reset_index(drop=True)
        x_fit = prepare_features(full_fit, student_features, student_categoricals)
        x_eval = prepare_features(eval_rows, student_features, student_categoricals)
        student = make_model("student", 120_800 + outer_year)
        student.fit(
            x_fit,
            full_fit["privileged_target"].to_numpy(float),
            sample_weight=sample_weight,
            cat_features=student_categoricals,
        )
        raw_delta = student.predict(x_eval)
        train_prediction = student.predict(x_fit)
        # Fixed training-distribution offset; no evaluation-batch aggregate is used.
        prediction_offset = float(np.average(train_prediction, weights=sample_weight))
        raw_delta = raw_delta - prediction_offset
        baseline = eval_rows["p113a_strict"].to_numpy(float)

        if outer_year == 2023:
            grid = []
            for gate_kind in GATE_KINDS:
                gate = reachability_gate(eval_rows, gate_kind)
                for scale in CALIBRATION_SCALES:
                    candidate = np.clip(
                        baseline + gate * np.clip(scale * raw_delta, -DELTA_CAP, DELTA_CAP),
                        0.001,
                        0.999,
                    )
                    item_metrics = metrics(
                        eval_rows, baseline, candidate, outer_year, f"gate={gate_kind},scale={scale}"
                    )
                    group_floor = min(item_metrics["by_game_type"].values())
                    grid.append(
                        {
                            "gate_kind": gate_kind,
                            "scale": scale,
                            "group_floor": group_floor,
                            "metrics": item_metrics,
                        }
                    )
            eligible = [
                item
                for item in grid
                if item["group_floor"] >= -0.00005
                and item["metrics"]["max_absolute_change"] <= DELTA_CAP + 1e-12
            ]
            pool = eligible if eligible else grid
            best = max(pool, key=lambda item: (item["metrics"]["brier_gain"], item["group_floor"]))
            selected = {
                "gate_kind": str(best["gate_kind"]),
                "scale": float(best["scale"]),
                "selection_metrics": best["metrics"],
                "eligible_count": len(eligible),
            }
            chosen = np.clip(
                baseline
                + reachability_gate(eval_rows, selected["gate_kind"])
                * np.clip(selected["scale"] * raw_delta, -DELTA_CAP, DELTA_CAP),
                0.001,
                0.999,
            )
            chosen_metrics = metrics(eval_rows, baseline, chosen, outer_year, "selected_2023")
            report["outer"][str(outer_year)] = {
                "teacher": teacher_report,
                "aligned_fit_rows": int(len(aligned_fit)),
                "full_fit_rows": int(len(full_fit)),
                "prediction_offset": prediction_offset,
                "grid": grid,
                "selected": selected,
                "metrics": chosen_metrics,
            }
        else:
            if selected is None:
                raise RuntimeError("R2 selection missing")
            chosen = np.clip(
                baseline
                + reachability_gate(eval_rows, selected["gate_kind"])
                * np.clip(selected["scale"] * raw_delta, -DELTA_CAP, DELTA_CAP),
                0.001,
                0.999,
            )
            chosen_metrics = metrics(eval_rows, baseline, chosen, outer_year, "frozen_2023_selection")
            report["outer"][str(outer_year)] = {
                "teacher": teacher_report,
                "aligned_fit_rows": int(len(aligned_fit)),
                "full_fit_rows": int(len(full_fit)),
                "prediction_offset": prediction_offset,
                "frozen_selection": selected,
                "metrics": chosen_metrics,
            }

        prediction_frames.append(
            pd.DataFrame(
                {
                    "row_id": eval_rows["row_id"].astype(str),
                    "season": outer_year,
                    "target": eval_rows["control_success"].to_numpy(float),
                    "p113a_strict": baseline,
                    "p120a_r2": chosen,
                    "student_delta_uncalibrated": raw_delta,
                }
            )
        )
        print(json.dumps(chosen_metrics, indent=2), flush=True)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions.to_csv(R2_OUT / "strict_predictions.csv.gz", index=False, compression="gzip")
    summary_by_year = {}
    for year, group in predictions.groupby("season"):
        base_score = brier(group["target"].to_numpy(float), group["p113a_strict"].to_numpy(float))
        candidate_score = brier(group["target"].to_numpy(float), group["p120a_r2"].to_numpy(float))
        summary_by_year[str(int(year))] = {
            "baseline_brier": base_score,
            "candidate_brier": candidate_score,
            "brier_gain": base_score - candidate_score,
        }
    metrics_2024 = report["outer"]["2024"]["metrics"]
    gates = {
        "2023_positive": summary_by_year["2023"]["brier_gain"] > 0.0,
        "2024_gain_at_least_0_00010": summary_by_year["2024"]["brier_gain"] >= 0.00010,
        "2024_bootstrap_ci_low_positive": metrics_2024["pitcher_bootstrap"]["ci_low"] > 0.0,
        "2024_each_game_type_above_minus_0_00005": min(metrics_2024["by_game_type"].values()) >= -0.00005,
        "student_has_zero_privileged_columns": len(report["student_privileged_columns"]) == 0,
        "test_file_not_read": True,
    }
    report["summary"] = {
        "by_year": summary_by_year,
        "gates": gates,
        "promotion_pass": all(gates.values()),
        "decision": "PROMOTE_TO_FINAL_TRAINING" if all(gates.values()) else "REJECT_KEEP_113A",
        "submission_zip_created": False,
        "elapsed_seconds": time.time() - started,
    }
    (R2_OUT / "research_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (R2_OUT / "promotion_decision.json").write_text(
        json.dumps(report["summary"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
