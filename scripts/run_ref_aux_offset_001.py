#!/usr/bin/env python3
"""REF-AUX-OFFSET-001: forward-only auxiliary failure-mode offset study.

This is a research artifact, not a submission builder. It uses official train
data and pre-existing official OOF success predictions only. Coefficients are
fit on one validation season and applied to the next season.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.linear_model import SGDClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.features import load_data

EXP = ROOT / "model" / "REF-AUX-OFFSET-001"
EXPERIMENT_ID = "REF-AUX-OFFSET-001"
AUX_MODEL_NAME = "SGDClassifier(log_loss)"
AUX_ARTIFACTS = []
TRANSITION_ARTIFACTS = []
LABEL_PATH = ROOT / "model" / "REF-AUX-LABEL-001" / "recovered_labels.csv.gz"
TRAIN_PATH = ROOT / "data" / "train.csv"
SEASONS = (2022, 2023, 2024)
SEED = 2024
# Diagnostic-only fixed budget; this is not a final submission model.
PARAMS = dict(loss="log_loss", max_iter=20, tol=1e-4, random_state=SEED, n_jobs=4)
AUX_COLUMNS = [
    "season", "game_month", "inning", "balls_before", "strikes_before", "outs_before",
    "li", "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_middle_rate",
    "asof_pitcher_reverse_rate", "asof_pitcher_ball_rate", "asof_pitcher_strike_rate",
    "asof_pitcher_pitchmix_n", "asof_pitcher_fastball_rate", "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate", "asof_batter_n", "asof_batter_success_rate", "asof_batter_middle_rate",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def fit_offset(p_success: np.ndarray, p_mr: np.ndarray, p_wayoff: np.ndarray, y: np.ndarray, mu_mr: float, mu_wayoff: float) -> tuple[float, float]:
    z = logit(p_success)
    u = logit(p_mr) - mu_mr
    v = logit(p_wayoff) - mu_wayoff

    def nll(w: np.ndarray) -> float:
        q = np.clip(1 / (1 + np.exp(-(z + w[0] * u + w[1] * v))), 1e-8, 1 - 1e-8)
        return float(-np.mean(y * np.log(q) + (1 - y) * np.log(1 - q)))

    result = minimize(nll, np.zeros(2), method="Nelder-Mead", options={"maxiter": 500})
    if not result.success or not np.all(np.isfinite(result.x)):
        raise RuntimeError(f"offset optimization failed: {result.message}")
    return float(result.x[0]), float(result.x[1])


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((y - p) ** 2))


def predict_aux(train_x: pd.DataFrame, train_y: np.ndarray, valid_x: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    model = SGDClassifier(**PARAMS)
    model.fit(train_x, train_y)
    return model.predict_proba(valid_x)[:, 1], model.predict_proba(train_x)[:, 1]


def main() -> None:
    EXP.mkdir(parents=True, exist_ok=True)
    raw = load_data(str(TRAIN_PATH), is_train=True)
    labels = pd.read_csv(LABEL_PATH)
    if not raw["row_id"].equals(labels["row_id"]):
        raise AssertionError("train and recovered labels row_id ordering mismatch")
    recovered_valid = labels[["success", "middle", "reverse"]].notna().all(axis=1).to_numpy()
    mr_values = ((labels["middle"] == 1) | (labels["reverse"] == 1)).to_numpy(dtype=np.float32)
    wayoff_values = ((raw["control_success"].to_numpy() == 0) & (mr_values == 0)).astype("float32")
    labels["mr"] = np.where(recovered_valid, mr_values, np.nan)
    labels["wayoff"] = np.where(recovered_valid, wayoff_values, np.nan)
    features = raw[AUX_COLUMNS].copy()
    features["game_type"] = raw["game_type"].astype("string").eq("R").astype("int8")
    features = features.replace([np.inf, -np.inf], np.nan)

    # Existing main success predictions are official OOF artifacts.
    pred_paths = {
        2022: ROOT / "model/ENS-CATF-LGBMCATR5050-FE001-EW-2022/selective_predictions_2022.csv",
        2023: ROOT / "model/ENS-CATF-LGBMCATR5050-FE001/selective_predictions_2023.csv",
        2024: ROOT / "model/ENS-CATF-LGBMCATR5050-FE001/selective_predictions_2024.csv",
    }
    pred = {s: pd.read_csv(pred_paths[s]) for s in SEASONS}
    for s in SEASONS:
        if not raw.loc[raw["season"].eq(s), "row_id"].reset_index(drop=True).equals(pred[s]["row_id"]):
            raise AssertionError(f"main prediction row order mismatch in {s}")

    season_aux: dict[int, dict[str, np.ndarray]] = {}
    train_means: dict[int, dict[str, float]] = {}
    for season in SEASONS:
        train_mask = raw["season"] < season
        valid_mask = raw["season"] == season
        valid_labels = labels.loc[train_mask, ["mr", "wayoff"]].notna().all(axis=1).to_numpy()
        x_train = features.loc[train_mask].loc[valid_labels]
        x_valid = features.loc[valid_mask]
        train_medians = x_train.median(numeric_only=True)
        x_train = x_train.fillna(train_medians).fillna(0.0)
        x_valid = x_valid.fillna(train_medians).fillna(0.0)
        aux_for_season: dict[str, np.ndarray] = {}
        means_for_season: dict[str, float] = {}
        for name in ("mr", "wayoff"):
            y_aux = labels.loc[train_mask, name].to_numpy(dtype=np.float64)[valid_labels].astype(np.int8)
            p_valid, p_train = predict_aux(x_train, y_aux, x_valid)
            aux_for_season[name] = p_valid
            means_for_season[name] = float(logit(p_train).mean())
        season_aux[season] = aux_for_season
        train_means[season] = means_for_season

    transitions: list[dict[str, object]] = []
    for source, target in ((2022, 2023), (2023, 2024)):
        source_mask = raw["season"] == source
        target_mask = raw["season"] == target
        p_s = pred[source]["pred_selective"].to_numpy(dtype=np.float64)
        p_t = pred[target]["pred_selective"].to_numpy(dtype=np.float64)
        y_s = raw.loc[source_mask, "control_success"].to_numpy(dtype=np.float64)
        y_t = raw.loc[target_mask, "control_success"].to_numpy(dtype=np.float64)
        b, c = fit_offset(p_s, season_aux[source]["mr"], season_aux[source]["wayoff"], y_s, train_means[source]["mr"], train_means[source]["wayoff"])
        z_t = logit(p_t) + b * (logit(season_aux[target]["mr"]) - train_means[target]["mr"]) + c * (logit(season_aux[target]["wayoff"]) - train_means[target]["wayoff"])
        p_adj = 1 / (1 + np.exp(-z_t))
        transition_artifact = EXP / "transition_predictions" / f"{source}_{target}.npz"
        transition_artifact.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(transition_artifact, y=y_t, baseline=p_t, offset=p_adj)
        TRANSITION_ARTIFACTS.append({"path": str(transition_artifact.relative_to(ROOT)), "sha256": sha256(transition_artifact), "fit_season": source, "apply_season": target, "n_rows": int(len(y_t))})
        transitions.append({
            "fit_season": source,
            "apply_season": target,
            "b": b,
            "c": c,
            "mu_source": train_means[source],
            "mu_apply": train_means[target],
            "baseline_brier": brier(y_t, p_t),
            "offset_brier": brier(y_t, p_adj),
            "delta_brier": brier(y_t, p_adj) - brier(y_t, p_t),
            "n_rows": int(len(y_t)),
        })

    report = {
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "runtime": {"python": platform.python_version(), "aux_model": AUX_MODEL_NAME, "seed": SEED},
        "source_scope": "official train.csv, REF-AUX-LABEL-001 labels, existing official OOF success predictions; no test/external data",
        "source_hashes": {"train": sha256(TRAIN_PATH), "labels": sha256(LABEL_PATH), **{f"pred_{s}": sha256(pred_paths[s]) for s in SEASONS}},
        "aux_artifacts": AUX_ARTIFACTS,
        "transition_artifacts": TRANSITION_ARTIFACTS,
        "fixed_contract": {"free_calibration": False, "candidate_count": 1, "new_zip": False, "forward_transitions": [[2022, 2023], [2023, 2024]]},
        "transitions": transitions,
        "status": "PASS" if all(t["delta_brier"] < 0 for t in transitions) else "FAIL_FORWARD_SIGN",
        "submission_status": "HOLD",
    }
    rp = EXP / "validation_report.json"; rp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    att = {"attestation_id": f"{EXPERIMENT_ID}-ATTESTATION", "report_sha256": sha256(rp), "validator_sha256": sha256(Path(__file__)), "transitions": len(transitions), "forward_all_improved": report["status"] == "PASS", "status": report["status"], "submission_status": "HOLD"}
    (EXP / "attestation.json").write_text(json.dumps(att, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
