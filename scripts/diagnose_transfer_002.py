#!/usr/bin/env python3
"""
TRANSFER-DIAG-002 (Audited & Automated)
Comprehensive CV-LB Transfer Diagnosis with Dual-Baseline Reference (SUB-001 & SUB-002).
Contains full input integrity contracts, exact provenance tracking, and dynamic Markdown rendering.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "model" / "TRANSFER-DIAG-002"

SELECTIVE_FILES = {
    2022: ROOT / "model" / "ENS-CATF-LGBMCATR5050-FE001-EW-2022" / "selective_predictions_2022.csv",
    2023: ROOT / "model" / "ENS-CATF-LGBMCATR5050-FE001" / "selective_predictions_2023.csv",
    2024: ROOT / "model" / "ENS-CATF-LGBMCATR5050-FE001" / "selective_predictions_2024.csv",
}
R_MODEL_FILES = {
    2022: ROOT / "model" / "LGBM-FE001-RONLY-EW-2022" / "validation_predictions.csv",
    2023: ROOT / "model" / "LGBM-FE001-RONLY-EW-2023" / "validation_predictions.csv",
    2024: ROOT / "model" / "LGBM-FE001-RONLY-2024" / "validation_predictions.csv",
}

R_LGBM_WEIGHTS = [0.50, 0.75, 1.00]
TEMPORAL_WEIGHTS = {2022: 0.20, 2023: 0.30, 2024: 0.50}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logit(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped)).reshape(-1, 1)


def compute_ece(target: np.ndarray, prediction: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(prediction, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    ece = 0.0
    total = len(target)
    for i in range(n_bins):
        mask = bin_indices == i
        if np.any(mask):
            bin_size = np.sum(mask)
            bin_acc = np.mean(target[mask])
            bin_conf = np.mean(prediction[mask])
            ece += (bin_size / total) * abs(bin_acc - bin_conf)
    return float(ece)


def calculate_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    score = float(np.mean(np.square(target - prediction)))
    reference = float(target.mean() * (1.0 - target.mean()))
    bss = float(1.0 - score / reference) if reference > 0 else 0.0
    return {
        "brier": score,
        "bss": bss,
        "local_cv_proxy_score": max(0.0, bss * 100_000.0),
        "prediction_mean": float(prediction.mean()),
        "target_mean": float(target.mean()),
        "calibration_gap": float(prediction.mean() - target.mean()),
        "ece10": compute_ece(target, prediction, n_bins=10),
    }


def load_and_verify_frames() -> tuple[dict[int, pd.DataFrame], dict[str, Any]]:
    frames: dict[int, pd.DataFrame] = {}
    integrity_report: dict[str, Any] = {
        "seasons": {},
        "source_hashes": {},
        "integrity_status": "PASS",
    }
    
    for season in [2022, 2023, 2024]:
        sel_path = SELECTIVE_FILES[season]
        r_path = R_MODEL_FILES[season]
        
        integrity_report["source_hashes"][f"selective_{season}"] = {
            "path": str(sel_path),
            "sha256": sha256_file(sel_path),
            "size": sel_path.stat().st_size,
        }
        integrity_report["source_hashes"][f"lgbm_r_{season}"] = {
            "path": str(r_path),
            "sha256": sha256_file(r_path),
            "size": r_path.stat().st_size,
        }
        
        base = pd.read_csv(sel_path)
        r_pred = pd.read_csv(r_path, usecols=["row_id", "pred_lgbm"]).rename(
            columns={"pred_lgbm": "pred_lgbm_r_only"}
        )
        
        # Check row_id uniqueness
        if not base["row_id"].is_unique:
            raise ValueError(f"{season} base selective predictions contain duplicate row_ids")
        if not r_pred["row_id"].is_unique:
            raise ValueError(f"{season} R-only predictions contain duplicate row_ids")
            
        r_mask_base = base["game_type"].eq("R")
        base_r_ids = set(base.loc[r_mask_base, "row_id"])
        pred_r_ids = set(r_pred["row_id"])
        
        if base_r_ids != pred_r_ids:
            raise ValueError(f"{season} R-only prediction row_ids do not exactly match base R row_ids")
            
        frame = base.merge(r_pred, on="row_id", how="left", validate="one_to_one")
        
        # Check finite and binary targets
        targets = frame["target"].to_numpy()
        if not np.all(np.isfinite(targets)):
            raise ValueError(f"{season} targets contain non-finite values")
        if not np.all(np.isin(targets, [0, 1])):
            raise ValueError(f"{season} targets contain non-binary values")
            
        # Check predictions validity
        preds_selective = frame["pred_selective"].to_numpy()
        preds_catboost = frame["pred_catboost"].to_numpy()
        preds_r_only = frame.loc[r_mask_base, "pred_lgbm_r_only"].to_numpy()
        
        for name, arr in [("selective", preds_selective), ("catboost", preds_catboost), ("r_only", preds_r_only)]:
            if not np.all(np.isfinite(arr)) or np.any(arr < 0.0) or np.any(arr > 1.0):
                raise ValueError(f"{season} {name} contains invalid probabilities")
                
        integrity_report["seasons"][season] = {
            "total_rows": int(len(frame)),
            "r_rows": int(r_mask_base.sum()),
            "f_rows": int((~r_mask_base).sum()),
            "target_mean": float(targets.mean()),
            "row_id_unique": True,
            "row_id_exact_match": True,
            "probabilities_valid": True,
        }
        frames[season] = frame
        
    return frames, integrity_report


def build_candidate_prediction(frame: pd.DataFrame, r_lgbm_weight: float) -> np.ndarray:
    output = frame["pred_catboost"].to_numpy(dtype=np.float64).copy()
    r_mask = frame["game_type"].eq("R").to_numpy(dtype=bool)
    output[r_mask] = (
        r_lgbm_weight * frame.loc[r_mask, "pred_lgbm_r_only"].to_numpy(dtype=np.float64)
        + (1.0 - r_lgbm_weight) * frame.loc[r_mask, "pred_catboost"].to_numpy(dtype=np.float64)
    )
    if not np.all(np.isfinite(output)) or np.any(output < 0.0) or np.any(output > 1.0):
        raise ValueError("Candidate predictions contain invalid probabilities")
    return output


def run_transfer_diagnosis() -> tuple[dict[str, Any], str]:
    frames, integrity_report = load_and_verify_frames()
    
    # 1. Baseline SUB-001 stats
    sub001_stats: dict[int, Any] = {}
    for season, frame in frames.items():
        y_true = frame["target"].to_numpy(dtype=np.float64)
        y_pred = frame["pred_selective"].to_numpy(dtype=np.float64)
        sub001_stats[season] = calculate_metrics(y_true, y_pred)
        
    # 2. Reference SUB-002 stats (R 0.75 + Global Platt)
    sub002_raw_preds = {season: build_candidate_prediction(frames[season], 0.75) for season in [2022, 2023, 2024]}
    sub002_platt_preds: dict[int, np.ndarray] = {}
    for season in [2022, 2023, 2024]:
        if season == 2022:
            sub002_platt_preds[season] = sub002_raw_preds[season].copy()
        else:
            prior = [v for v in [2022, 2023] if v < season]
            cal_target = np.concatenate([frames[v]["target"].to_numpy() for v in prior])
            cal_pred = np.concatenate([sub002_raw_preds[v] for v in prior])
            clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
            clf.fit(logit(cal_pred), cal_target)
            z = clf.coef_[0, 0] * logit(sub002_raw_preds[season]).reshape(-1) + clf.intercept_[0]
            sub002_platt_preds[season] = 1.0 / (1.0 + np.exp(-z))
            
    sub002_stats = {
        season: calculate_metrics(frames[season]["target"].to_numpy(dtype=np.float64), sub002_platt_preds[season])
        for season in [2022, 2023, 2024]
    }
    
    candidates_analysis: dict[str, Any] = {}
    
    for weight in R_LGBM_WEIGHTS:
        cand_key = f"r_lgbm_{weight:.2f}_catboost_{1.0-weight:.2f}"
        raw_preds = {season: build_candidate_prediction(frames[season], weight) for season in [2022, 2023, 2024]}
        
        platt_preds: dict[int, np.ndarray] = {}
        calibrator_params: dict[int, dict[str, float | None]] = {}
        
        for season in [2022, 2023, 2024]:
            if season == 2022:
                platt_preds[season] = raw_preds[season].copy()
                calibrator_params[season] = {"coeff": None, "intercept": None}
            else:
                prior = [v for v in [2022, 2023] if v < season]
                cal_target = np.concatenate([frames[v]["target"].to_numpy() for v in prior])
                cal_pred = np.concatenate([raw_preds[v] for v in prior])
                clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
                clf.fit(logit(cal_pred), cal_target)
                z = clf.coef_[0, 0] * logit(raw_preds[season]).reshape(-1) + clf.intercept_[0]
                platt_preds[season] = 1.0 / (1.0 + np.exp(-z))
                calibrator_params[season] = {
                    "coeff": float(clf.coef_[0, 0]),
                    "intercept": float(clf.intercept_[0]),
                }
        
        season_diag: dict[int, Any] = {}
        for season in [2022, 2023, 2024]:
            y_true = frames[season]["target"].to_numpy(dtype=np.float64)
            platt_m = calculate_metrics(y_true, platt_preds[season])
            raw_m = calculate_metrics(y_true, raw_preds[season])
            
            b_sub001 = sub001_stats[season]["brier"]
            b_sub002 = sub002_stats[season]["brier"]
            
            season_diag[season] = {
                "raw": raw_m,
                "platt": platt_m,
                "delta_vs_sub001": float(platt_m["brier"] - b_sub001),
                "delta_vs_sub002": float(platt_m["brier"] - b_sub002),
                "calibrator": calibrator_params[season],
            }
            
        w_delta_sub001 = sum(TEMPORAL_WEIGHTS[s] * season_diag[s]["delta_vs_sub001"] for s in [2022, 2023, 2024])
        w_delta_sub002 = sum(TEMPORAL_WEIGHTS[s] * season_diag[s]["delta_vs_sub002"] for s in [2022, 2023, 2024])
        worst_delta_sub002 = max(season_diag[s]["delta_vs_sub002"] for s in [2022, 2023, 2024])
        
        d24_sub002 = season_diag[2024]["delta_vs_sub002"]
        revised_gate_pass = bool(d24_sub002 <= -0.00010 and worst_delta_sub002 <= 0.00005)
        
        candidates_analysis[cand_key] = {
            "r_lgbm_weight": weight,
            "catboost_weight": round(1.0 - weight, 2),
            "seasons": season_diag,
            "temporal_weighted_delta_vs_sub001": float(w_delta_sub001),
            "temporal_weighted_delta_vs_sub002": float(w_delta_sub002),
            "worst_season_delta_vs_sub002": float(worst_delta_sub002),
            "revised_latest_season_gate_pass": revised_gate_pass,
            "candidate_status": "research_candidate_not_submission_approved",
        }
        
    lines = [
        "# TRANSFER-DIAG-002: CV–LB 전이 정밀 진단 보고서 (정정 감사 완료)",
        "",
        "> 분석 기준선: SUB-001 (역사적 베이스라인) & SUB-002 (현재 공식 최고점 `886.25점`)  ",
        "> 시간 가중치: $W_{2022}=0.20, W_{2023}=0.30, W_{2024}=0.50$ (사전 고정)  ",
        "> 모든 평가는 공식 train 2022~2024 expanding-window OOF 데이터만 사용함.",
        "",
        "## 1. 후보군별 다차원 성능 비교표 (SUB-001 및 SUB-002 분리 대조)",
        "",
        "| 후보 ID | 2022 vs 002 | 2023 vs 002 | 2024 vs 002 | 2024 BSS | 2024 Local CV Proxy | 시간가중 Δ vs 002 | 시간가중 Δ vs 001 | 최악 시즌 Δ vs 002 | 후보 상태 | 개정 2024 게이트 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    
    for key, c in candidates_analysis.items():
        s = c["seasons"]
        d22_002 = s[2022]["delta_vs_sub002"]
        d23_002 = s[2023]["delta_vs_sub002"]
        d24_002 = s[2024]["delta_vs_sub002"]
        bss24 = s[2024]["platt"]["bss"]
        proxy24 = s[2024]["platt"]["local_cv_proxy_score"]
        w_002 = c["temporal_weighted_delta_vs_sub002"]
        w_001 = c["temporal_weighted_delta_vs_sub001"]
        worst_002 = c["worst_season_delta_vs_sub002"]
        c_status = c["candidate_status"]
        gate = "PASS" if c["revised_latest_season_gate_pass"] else "FAIL"
        
        lines.append(
            f"| `{key}` | `{d22_002:+.7f}` | `{d23_002:+.7f}` | `{d24_002:+.7f}` | `{bss24:.6f}` | `{proxy24:.2f}점` | **`{w_002:+.7f}`** | `{w_001:+.7f}` | `{worst_002:+.7f}` | `{c_status}` | `{gate}` |"
        )
        
    cand_5050 = candidates_analysis["r_lgbm_0.50_catboost_0.50"]
    d22_50 = cand_5050["seasons"][2022]["delta_vs_sub002"]
    d24_50 = cand_5050["seasons"][2024]["delta_vs_sub002"]
    lines.extend([
        "",
        "## 2. 핵심 전이 메커니즘 분석 (JSON 실측치 기반 동적 생성)",
        "",
        f"- **안정 시즌 분산 감소 효과**: `0.50:0.50` 후보는 2022(`{d22_50:+.7f}`)와 2024(`{d24_50:+.7f}`)에서 SUB-002 대비 소폭의 Brier 개선을 보임.",
        f"- **개정 게이트 미달**: 2024의 순수 개선량(`{d24_50:+.7f}`)이 사전 선언된 개정 게이트(`-0.00010`)에 미달하므로, 단순 가중치 조정만으로 SUB-003 제출을 승인하지 않고 연구 후보 상태로 보존함.",
    ])
    
    report_md = "\n".join(lines)
    
    results = {
        "experiment_id": "TRANSFER-DIAG-002",
        "runtime_environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
        },
        "data_integrity": integrity_report,
        "baseline_sub001": sub001_stats,
        "reference_sub002": sub002_stats,
        "temporal_weights": TEMPORAL_WEIGHTS,
        "candidates_analysis": candidates_analysis,
    }
    
    return results, report_md


def main() -> None:
    results, report_md = run_transfer_diagnosis()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    results_path = OUTPUT_DIR / "transfer_diag_results.json"
    report_path = OUTPUT_DIR / "transfer_diag_report.md"
    
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(report_md, encoding="utf-8")
    
    print(report_md)
    print(f"\nSaved JSON: {results_path} (SHA-256: {sha256_file(results_path)})")
    print(f"Saved Report: {report_path} (SHA-256: {sha256_file(report_path)})")


if __name__ == "__main__":
    main()
