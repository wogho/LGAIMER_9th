#!/usr/bin/env python3
"""Strict research pipeline for REF4-TRACKMAN-PRIVILEGED-DISTILL-120A.

The current-pitch TrackMan columns are *privileged training information*.
They are available to the teacher only.  Every evaluated student consumes the
same official pre-pitch columns that are available at submission inference.

Protocol
--------
* Reconstruct an auditable pitch-level train <-> TrackMan alignment.
* Generate grouped cross-fitted teacher corrections on outer-training seasons.
* Tune a small, predeclared distillation/calibration grid on 2023 only.
* Freeze that choice and evaluate it once on the full 2024 strict holdout.
* Do not read test.csv and do not build a submission package in this script.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor


ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data/train.csv"
TRACKMAN_PATH = ROOT / "data/trackman_history.csv"
ANCHOR_PATH = ROOT / "model/REF4-113A-V66-NESTED-117A/oof_predictions.csv"
PITCHER_MAP_PATH = ROOT / "model/TRACKMAN-MAP-004/pitcher_id_map.csv"
OUT = ROOT / "model/REF4-TRACKMAN-PRIVILEGED-DISTILL-120A"

OUTER_YEARS = (2023, 2024)
TEACHER_FOLDS = 3
TEACHER_ITERATIONS = 350
STUDENT_ITERATIONS = 500
DELTA_CAP = 0.05
DISTILL_ALPHAS = (1.0, 0.75, 0.50)
CALIBRATION_SCALES = (0.25, 0.50, 0.75, 1.00)
GATE_KINDS = ("none", "reliability")

TM_NUMERIC = (
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed",
)
TM_CATEGORICAL = ("tagged_pitch_type", "auto_pitch_type", "pitch_type_group")

STUDENT_CATEGORICAL = (
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.square(y - p)))


def normalize_id(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace(r"\.0$", "", regex=True)


def unit_table(frame: pd.DataFrame, pitcher_team: str, batter_team: str) -> pd.DataFrame:
    grouped = frame.groupby("unit", sort=False)
    return pd.DataFrame(
        {
            "n": grouped.size(),
            "season": grouped["season"].first(),
            "month": grouped["game_month"].first(),
            "dow": grouped["game_dayofweek"].first(),
            "pteam": grouped[pitcher_team].first(),
            "bteam": grouped[batter_team].first(),
        }
    )


def build_verified_alignment(raw: pd.DataFrame, trackman: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    """Recreate the existing official-only unit matcher and verify each pitch."""
    train_columns = [
        "season",
        "game_month",
        "game_dayofweek",
        "pitcher_team_id",
        "batter_team_id",
        "inning",
        "pitcher_id",
    ]
    train = raw.loc[raw["season"].between(2019, 2024), train_columns].copy()
    train["_src"] = train.index.to_numpy(np.int64)
    train = train.reset_index(drop=True)
    train_key = ["season", "game_month", "game_dayofweek", "pitcher_team_id", "batter_team_id"]
    train["_new"] = train.groupby(train_key, sort=False)["inning"].diff().fillna(0).lt(0)
    train["_seg"] = train["_new"].groupby([train[column] for column in train_key], sort=False).cumsum()
    train["unit"] = list(zip(*[train[column] for column in [*train_key, "_seg"]]))

    tm_columns = [
        "season",
        "game_date",
        "trackman_game_id",
        "pitch_no",
        "pitcher_trackman_id",
        "pitcher_team",
        "batter_team",
    ]
    tm = trackman.loc[trackman["season"].between(2019, 2024), tm_columns].copy()
    tm["_src"] = tm.index.to_numpy(np.int64)
    dates = pd.to_datetime(tm["game_date"], format="mixed")
    tm["game_month"] = dates.dt.month
    tm["game_dayofweek"] = dates.dt.dayofweek
    tm = tm.sort_values(["trackman_game_id", "pitch_no"], kind="stable").reset_index(drop=True)
    tm["unit"] = list(zip(tm["trackman_game_id"], tm["pitcher_team"]))

    a = unit_table(train, "pitcher_team_id", "batter_team_id")
    b = unit_table(tm, "pitcher_team", "batter_team")
    first_signature = ["season", "month", "dow", "n"]
    unique_a = a.groupby(first_signature).filter(lambda group: len(group) == 1)
    unique_b = b.groupby(first_signature).filter(lambda group: len(group) == 1)
    first_matches = unique_a.reset_index().merge(
        unique_b.reset_index(), on=first_signature, suffixes=("_a", "_b")
    )

    team_votes: defaultdict[object, Counter] = defaultdict(Counter)
    for row in first_matches.itertuples():
        team_votes[row.pteam_a][row.pteam_b] += 1
        team_votes[row.bteam_a][row.bteam_b] += 1
    team_map = {key: votes.most_common(1)[0][0] for key, votes in team_votes.items()}

    mapped_a = a.copy()
    mapped_a["pteam"] = mapped_a["pteam"].map(team_map)
    mapped_a["bteam"] = mapped_a["bteam"].map(team_map)
    full_signature = ["season", "month", "dow", "pteam", "bteam", "n"]
    unique_a2 = mapped_a.dropna(subset=["pteam", "bteam"]).groupby(full_signature).filter(lambda group: len(group) == 1)
    unique_b2 = b.groupby(full_signature).filter(lambda group: len(group) == 1)
    unit_matches = unique_a2.reset_index().merge(
        unique_b2.reset_index(), on=full_signature, suffixes=("_a", "_b")
    )

    train_groups = train.groupby("unit", sort=False).indices
    tm_groups = tm.groupby("unit", sort=False).indices
    train_source: list[int] = []
    tm_source: list[int] = []
    unit_ids: list[str] = []
    for position, row in enumerate(unit_matches.itertuples()):
        train_positions = train_groups[row.unit_a]
        tm_positions = tm_groups[row.unit_b]
        if len(train_positions) != len(tm_positions):
            continue
        count = len(train_positions)
        train_source.extend(train.iloc[train_positions]["_src"].astype(int).tolist())
        tm_source.extend(tm.iloc[tm_positions]["_src"].astype(int).tolist())
        unit_ids.extend([f"u{position:06d}"] * count)

    train_rows = raw.iloc[train_source].reset_index(drop=True)
    tm_rows = trackman.iloc[tm_source].reset_index(drop=True)
    top_bottom = tm_rows["top_bottom"].map({"Top": "T", "Bottom": "B"}).fillna(tm_rows["top_bottom"])
    core_exact = (
        train_rows["season"].eq(tm_rows["season"])
        & train_rows["inning"].eq(tm_rows["inning"])
        & train_rows["balls_before"].eq(tm_rows["balls_before"])
        & train_rows["strikes_before"].eq(tm_rows["strikes_before"])
        & train_rows["outs_before"].eq(tm_rows["outs_before"])
        & train_rows["top_bottom"].astype(str).eq(top_bottom.astype(str))
    )

    pitcher_map = pd.read_csv(PITCHER_MAP_PATH)
    normalized_map = dict(
        zip(normalize_id(pitcher_map["pitcher_id"]), normalize_id(pitcher_map["pitcher_trackman_id"]))
    )
    expected_pitcher = normalize_id(train_rows["pitcher_id"]).map(normalized_map)
    pitcher_exact = expected_pitcher.fillna("__NO_MAP__").eq(normalize_id(tm_rows["pitcher_trackman_id"]))
    keep = core_exact & pitcher_exact

    alignment = pd.DataFrame(
        {
            "row_id": train_rows.loc[keep, "row_id"].astype(str).to_numpy(),
            "season": train_rows.loc[keep, "season"].astype(int).to_numpy(),
            "train_position": np.asarray(train_source, dtype=np.int64)[keep.to_numpy()],
            "trackman_position": np.asarray(tm_source, dtype=np.int64)[keep.to_numpy()],
            "trackman_id": tm_rows.loc[keep, "trackman_id"].astype(str).to_numpy(),
            "match_unit_id": np.asarray(unit_ids, dtype=object)[keep.to_numpy()],
        }
    )
    if alignment["row_id"].duplicated().any() or alignment["trackman_id"].duplicated().any():
        raise RuntimeError("verified alignment is not one-to-one")

    report = {
        "train_units": int(len(a)),
        "trackman_units": int(len(b)),
        "unique_stage_matches": int(len(first_matches)),
        "full_stage_matches": int(len(unit_matches)),
        "positionally_paired_rows": int(len(train_rows)),
        "core_exact_rows": int(core_exact.sum()),
        "pitcher_exact_rows": int(pitcher_exact.sum()),
        "fully_verified_rows": int(len(alignment)),
        "full_train_coverage": float(len(alignment) / len(raw)),
        "core_exact_rate": float(core_exact.mean()),
        "verified_by_season": {str(k): int(v) for k, v in alignment.groupby("season").size().items()},
    }
    return alignment, report


def prepare_features(frame: pd.DataFrame, columns: list[str], categoricals: list[str]) -> pd.DataFrame:
    result = frame[columns].copy()
    for column in categoricals:
        result[column] = result[column].fillna("__MISSING__").astype(str)
    for column in set(columns).difference(categoricals):
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def make_model(kind: str, seed: int) -> CatBoostRegressor:
    if kind == "teacher":
        return CatBoostRegressor(
            iterations=TEACHER_ITERATIONS,
            depth=7,
            learning_rate=0.04,
            loss_function="RMSE",
            l2_leaf_reg=40.0,
            random_seed=seed,
            random_strength=0.5,
            verbose=False,
            allow_writing_files=False,
            thread_count=-1,
        )
    return CatBoostRegressor(
        iterations=STUDENT_ITERATIONS,
        depth=7,
        learning_rate=0.035,
        loss_function="RMSE",
        l2_leaf_reg=60.0,
        random_seed=seed,
        random_strength=0.75,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def crossfit_teacher(
    fit: pd.DataFrame,
    teacher_features: list[str],
    teacher_categoricals: list[str],
    outer_year: int,
) -> tuple[np.ndarray, dict[str, object]]:
    features = prepare_features(fit, teacher_features, teacher_categoricals)
    residual = fit["target"].to_numpy(float) - fit["p113a_strict"].to_numpy(float)
    hashes = pd.util.hash_pandas_object(fit["match_unit_id"].astype(str), index=False).to_numpy(np.uint64)
    fold_ids = (hashes % TEACHER_FOLDS).astype(np.int8)
    predictions = np.full(len(fit), np.nan, dtype=float)
    fold_report = []
    for fold in range(TEACHER_FOLDS):
        valid_mask = fold_ids == fold
        train_mask = ~valid_mask
        model = make_model("teacher", 120_000 + outer_year * 10 + fold)
        model.fit(features.loc[train_mask], residual[train_mask], cat_features=teacher_categoricals)
        predictions[valid_mask] = model.predict(features.loc[valid_mask])
        fold_report.append(
            {
                "fold": fold,
                "train_rows": int(train_mask.sum()),
                "valid_rows": int(valid_mask.sum()),
                "train_units": int(fit.loc[train_mask, "match_unit_id"].nunique()),
                "valid_units": int(fit.loc[valid_mask, "match_unit_id"].nunique()),
                "unit_overlap": int(
                    len(
                        set(fit.loc[train_mask, "match_unit_id"])
                        & set(fit.loc[valid_mask, "match_unit_id"])
                    )
                ),
            }
        )
    if not np.isfinite(predictions).all():
        raise RuntimeError("teacher cross-fit did not cover every row")
    base = fit["p113a_strict"].to_numpy(float)
    y = fit["target"].to_numpy(float)
    teacher_prediction = np.clip(base + np.clip(predictions, -DELTA_CAP, DELTA_CAP), 0.001, 0.999)
    return predictions, {
        "rows": int(len(fit)),
        "units": int(fit["match_unit_id"].nunique()),
        "folds": fold_report,
        "baseline_brier": brier(y, base),
        "teacher_oof_brier": brier(y, teacher_prediction),
        "teacher_oof_gain": brier(y, base) - brier(y, teacher_prediction),
        "teacher_delta_std": float(np.std(predictions)),
    }


def reachability_gate(frame: pd.DataFrame, kind: str) -> np.ndarray:
    if kind == "none":
        return np.ones(len(frame), dtype=float)
    pitcher_n = frame["asof_pitcher_n"].fillna(0).clip(lower=0).to_numpy(float)
    batter_n = frame["asof_batter_n"].fillna(0).clip(lower=0).to_numpy(float)
    reliability = np.sqrt((pitcher_n / (pitcher_n + 100.0)) * (batter_n / (batter_n + 100.0)))
    return 0.25 + 0.75 * reliability


def pitcher_bootstrap(
    y: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    pitcher: pd.Series,
    seed: int,
    repeats: int = 2000,
) -> dict[str, object]:
    row_gain = np.square(baseline - y) - np.square(candidate - y)
    codes, unique = pd.factorize(pitcher.astype(str), sort=True)
    sums = np.bincount(codes, weights=row_gain, minlength=len(unique))
    sizes = np.bincount(codes, minlength=len(unique))
    rng = np.random.default_rng(seed)
    draws = np.empty(repeats, dtype=float)
    for start in range(0, repeats, 64):
        count = min(64, repeats - start)
        sampled = rng.integers(0, len(unique), size=(count, len(unique)))
        draws[start : start + count] = sums[sampled].sum(axis=1) / sizes[sampled].sum(axis=1)
    return {
        "clusters": int(len(unique)),
        "repeats": repeats,
        "mean_gain": float(draws.mean()),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
    }


def score_candidate(
    year: int,
    rows: pd.DataFrame,
    baseline: np.ndarray,
    candidate: np.ndarray,
    label: str,
) -> dict[str, object]:
    y = rows["control_success"].to_numpy(float)
    base_score = brier(y, baseline)
    candidate_score = brier(y, candidate)
    return {
        "label": label,
        "season": year,
        "rows": int(len(rows)),
        "baseline_brier": base_score,
        "candidate_brier": candidate_score,
        "brier_gain": base_score - candidate_score,
        "mean_absolute_change": float(np.mean(np.abs(candidate - baseline))),
        "max_absolute_change": float(np.max(np.abs(candidate - baseline))),
        "pitcher_bootstrap": pitcher_bootstrap(
            y, baseline, candidate, rows["pitcher_id"], 120_500 + year
        ),
    }


def main() -> None:
    started = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    print("[1/5] Loading official train, TrackMan, and immutable strict 113A anchor", flush=True)
    raw = pd.read_csv(TRAIN_PATH, low_memory=False)
    trackman = pd.read_csv(TRACKMAN_PATH, low_memory=False)
    anchor = pd.read_csv(
        ANCHOR_PATH,
        usecols=["row_id", "season", "target", "p113a_strict"],
        low_memory=False,
    )
    validation = raw.loc[raw["season"].isin((2022, 2023, 2024))].reset_index(drop=True)
    if not np.array_equal(validation["row_id"].astype(str), anchor["row_id"].astype(str)):
        raise RuntimeError("strict 113A anchor row order mismatch")
    if not np.array_equal(validation["control_success"].to_numpy(float), anchor["target"].to_numpy(float)):
        raise RuntimeError("strict 113A anchor target mismatch")
    validation["p113a_strict"] = anchor["p113a_strict"].to_numpy(float)

    print("[2/5] Reconstructing and verifying pitch-level TrackMan alignment", flush=True)
    alignment, alignment_report = build_verified_alignment(raw, trackman)
    alignment.to_csv(OUT / "alignment_verified.csv.gz", index=False, compression="gzip")
    (OUT / "alignment_report.json").write_text(
        json.dumps(alignment_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(alignment_report, indent=2), flush=True)

    aligned = alignment.merge(
        anchor[["row_id", "season", "target", "p113a_strict"]],
        on=["row_id", "season"],
        how="inner",
        validate="one_to_one",
    )
    official_by_row = validation.set_index("row_id", drop=False)
    official = official_by_row.loc[aligned["row_id"]].reset_index(drop=True)
    if not np.array_equal(official["row_id"].astype(str), aligned["row_id"].astype(str)):
        raise RuntimeError("official/alignment row drift")
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

    provenance = {
        "experiment": "REF4-TRACKMAN-PRIVILEGED-DISTILL-120A",
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "official_champion": {"version": "113A", "score": 1121.9039933605},
        "input_hashes": {
            "train": sha256(TRAIN_PATH),
            "trackman": sha256(TRACKMAN_PATH),
            "strict_anchor": sha256(ANCHOR_PATH),
            "pitcher_map": sha256(PITCHER_MAP_PATH),
        },
        "test_file_read": False,
        "student_features": student_features,
        "student_categorical_features": student_categoricals,
        "teacher_privileged_features": [f"tm_{column}" for column in (*TM_CATEGORICAL, *TM_NUMERIC)],
        "protocol": {
            "outer_years": list(OUTER_YEARS),
            "teacher_folds": TEACHER_FOLDS,
            "teacher_group": "matched game-side unit",
            "selection_year": 2023,
            "untouched_confirmation_year": 2024,
            "distill_alphas": list(DISTILL_ALPHAS),
            "calibration_scales": list(CALIBRATION_SCALES),
            "gate_kinds": list(GATE_KINDS),
            "delta_cap": DELTA_CAP,
        },
    }
    (OUT / "provenance.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    report: dict[str, object] = {
        **provenance,
        "alignment": alignment_report,
        "outer_evaluations": {},
    }
    selected: dict[str, object] | None = None
    all_predictions = []

    for outer_year in OUTER_YEARS:
        print(f"[3/5] Outer {outer_year}: cross-fitting privileged teacher", flush=True)
        aligned_fit = aligned.loc[
            aligned["season"].ge(2022) & aligned["season"].lt(outer_year)
        ].reset_index(drop=True)
        eval_rows = validation.loc[validation["season"].eq(outer_year)].reset_index(drop=True)
        teacher_oof, teacher_report = crossfit_teacher(
            aligned_fit, teacher_features, teacher_categoricals, outer_year
        )
        residual = aligned_fit["target"].to_numpy(float) - aligned_fit["p113a_strict"].to_numpy(float)
        student_fit_features = prepare_features(
            aligned_fit, student_features, student_categoricals
        )
        student_eval_features = prepare_features(
            eval_rows, student_features, student_categoricals
        )
        baseline = eval_rows["p113a_strict"].to_numpy(float)

        if outer_year == 2023:
            candidates = []
            for alpha_position, alpha in enumerate(DISTILL_ALPHAS):
                target = alpha * teacher_oof + (1.0 - alpha) * residual
                student = make_model("student", 120_100 + alpha_position)
                student.fit(
                    student_fit_features,
                    target,
                    cat_features=student_categoricals,
                )
                raw_delta = student.predict(student_eval_features)
                for gate_kind in GATE_KINDS:
                    gate = reachability_gate(eval_rows, gate_kind)
                    for scale in CALIBRATION_SCALES:
                        candidate = np.clip(
                            baseline + gate * np.clip(scale * raw_delta, -DELTA_CAP, DELTA_CAP),
                            0.001,
                            0.999,
                        )
                        metrics = score_candidate(
                            outer_year,
                            eval_rows,
                            baseline,
                            candidate,
                            f"alpha={alpha},gate={gate_kind},scale={scale}",
                        )
                        candidates.append(
                            {
                                "alpha": alpha,
                                "gate_kind": gate_kind,
                                "scale": scale,
                                "metrics": metrics,
                            }
                        )
            selected = max(candidates, key=lambda item: item["metrics"]["brier_gain"])
            selected = {
                "alpha": float(selected["alpha"]),
                "gate_kind": str(selected["gate_kind"]),
                "scale": float(selected["scale"]),
                "selection_metrics": selected["metrics"],
            }
            chosen_target = selected["alpha"] * teacher_oof + (1.0 - selected["alpha"]) * residual
            chosen_student = make_model("student", 120_199)
            chosen_student.fit(
                student_fit_features,
                chosen_target,
                cat_features=student_categoricals,
            )
            chosen_delta = chosen_student.predict(student_eval_features)
            chosen_candidate = np.clip(
                baseline
                + reachability_gate(eval_rows, selected["gate_kind"])
                * np.clip(selected["scale"] * chosen_delta, -DELTA_CAP, DELTA_CAP),
                0.001,
                0.999,
            )
            chosen_metrics = score_candidate(
                outer_year, eval_rows, baseline, chosen_candidate, "frozen-selection-refit"
            )
            outer_report = {
                "fit_seasons": sorted(aligned_fit["season"].unique().astype(int).tolist()),
                "teacher": teacher_report,
                "selection_grid": candidates,
                "selected": selected,
                "frozen_selection_refit": chosen_metrics,
            }
        else:
            if selected is None:
                raise RuntimeError("2023 selection was not frozen before 2024 evaluation")
            target = selected["alpha"] * teacher_oof + (1.0 - selected["alpha"]) * residual
            student = make_model("student", 120_200 + outer_year)
            student.fit(student_fit_features, target, cat_features=student_categoricals)
            raw_delta = student.predict(student_eval_features)
            candidate = np.clip(
                baseline
                + reachability_gate(eval_rows, selected["gate_kind"])
                * np.clip(selected["scale"] * raw_delta, -DELTA_CAP, DELTA_CAP),
                0.001,
                0.999,
            )
            chosen_candidate = candidate
            chosen_metrics = score_candidate(
                outer_year, eval_rows, baseline, candidate, "2023-frozen-configuration"
            )

            control = make_model("student", 120_900 + outer_year)
            control.fit(student_fit_features, residual, cat_features=student_categoricals)
            control_delta = control.predict(student_eval_features)
            control_candidate = np.clip(
                baseline
                + reachability_gate(eval_rows, selected["gate_kind"])
                * np.clip(selected["scale"] * control_delta, -DELTA_CAP, DELTA_CAP),
                0.001,
                0.999,
            )
            control_metrics = score_candidate(
                outer_year, eval_rows, baseline, control_candidate, "raw-residual-control"
            )
            outer_report = {
                "fit_seasons": sorted(aligned_fit["season"].unique().astype(int).tolist()),
                "teacher": teacher_report,
                "frozen_configuration": selected,
                "privileged_student": chosen_metrics,
                "raw_residual_control": control_metrics,
                "privileged_minus_control_gain": float(
                    chosen_metrics["brier_gain"] - control_metrics["brier_gain"]
                ),
            }

        report["outer_evaluations"][str(outer_year)] = outer_report
        all_predictions.append(
            pd.DataFrame(
                {
                    "row_id": eval_rows["row_id"].astype(str),
                    "season": outer_year,
                    "target": eval_rows["control_success"].to_numpy(float),
                    "p113a_strict": baseline,
                    "p120a_student": chosen_candidate,
                }
            )
        )
        print(json.dumps(chosen_metrics, indent=2), flush=True)

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(OUT / "strict_predictions_2023_2024.csv.gz", index=False, compression="gzip")
    by_year = {
        int(year): {
            "baseline_brier": brier(group["target"].to_numpy(float), group["p113a_strict"].to_numpy(float)),
            "candidate_brier": brier(group["target"].to_numpy(float), group["p120a_student"].to_numpy(float)),
        }
        for year, group in predictions.groupby("season")
    }
    for values in by_year.values():
        values["brier_gain"] = values["baseline_brier"] - values["candidate_brier"]
    overall_baseline = brier(predictions["target"].to_numpy(float), predictions["p113a_strict"].to_numpy(float))
    overall_candidate = brier(predictions["target"].to_numpy(float), predictions["p120a_student"].to_numpy(float))
    evaluation_2024 = report["outer_evaluations"]["2024"]["privileged_student"]
    gate_checks = {
        "2023_positive": bool(by_year[2023]["brier_gain"] > 0.0),
        "2024_gain_at_least_0_00010": bool(by_year[2024]["brier_gain"] >= 0.00010),
        "overall_positive": bool(overall_baseline - overall_candidate > 0.0),
        "2024_pitcher_bootstrap_ci_low_positive": bool(
            evaluation_2024["pitcher_bootstrap"]["ci_low"] > 0.0
        ),
        "max_change_within_cap": bool(
            max(
                np.max(np.abs(group["p120a_student"] - group["p113a_strict"]))
                for _, group in predictions.groupby("season")
            )
            <= DELTA_CAP + 1e-12
        ),
        "student_uses_no_privileged_features": bool(
            all(not column.startswith("tm_") for column in student_features)
        ),
        "test_file_not_read": True,
    }
    promotion_pass = all(gate_checks.values())
    report["selection"] = selected
    report["summary"] = {
        "by_year": by_year,
        "overall_baseline_brier": overall_baseline,
        "overall_candidate_brier": overall_candidate,
        "overall_brier_gain": overall_baseline - overall_candidate,
        "gate_checks": gate_checks,
        "promotion_pass": promotion_pass,
        "submission_zip_created": False,
        "decision": "PROMOTE_TO_FINAL_TRAINING" if promotion_pass else "REJECT_KEEP_113A",
        "elapsed_seconds": time.time() - started,
    }
    (OUT / "research_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (OUT / "promotion_decision.json").write_text(
        json.dumps(report["summary"], indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print("[5/5] Strict decision", flush=True)
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
