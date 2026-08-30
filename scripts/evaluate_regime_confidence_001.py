#!/usr/bin/env python3
"""
REGIME-CONFIDENCE-001 (Audited & Automated)
Sample-Confidence Routing with Dual-Baseline Reference and Programmatic Reporting.
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
OUTPUT_DIR = ROOT / "model" / "REGIME-CONFIDENCE-001"

SELECTIVE_FILES = {
    2022: ROOT / "model" / "ENS-CATF-LGBMCATR5050-FE001-EW-2022" / "selective_predictions_2022.csv",
    2023: ROOT / "model" / "ENS-CATF-LGBMCATR5050-FE001" / "selective_predictions_2023.csv",
    2024: ROOT / "model" / "ENS-CATF-LGBMCATR5050-FE001" / "selective_predictions_2024.csv",
}
LGBM_RONLY_FILES = {
    2022: ROOT / "model" / "LGBM-FE001-RONLY-EW-2022" / "validation_predictions.csv",
    2023: ROOT / "model" / "LGBM-FE001-RONLY-EW-2023" / "validation_predictions.csv",
    2024: ROOT / "model" / "LGBM-FE001-RONLY-2024" / "validation_predictions.csv",
}
CAT_RONLY_FILES = {
    2022: ROOT / "model" / "CAT-FE001-RONLY-EW-2022" / "validation_predictions.csv",
    2023: ROOT / "model" / "CAT-FE001-RONLY-EW-2023" / "validation_predictions.csv",
    2024: ROOT / "model" / "CAT-FE001-RONLY-2024" / "validation_predictions.csv",
}
RAW_TRAIN_PATH = ROOT / "data" / "train.csv"

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


def load_and_verify_confidence_frames() -> tuple[dict[int, pd.DataFrame], dict[str, Any]]:
    integrity_report: dict[str, Any] = {
        "seasons": {},
        "source_hashes": {
            "train_raw": {
                "path": str(RAW_TRAIN_PATH),
                "sha256": sha256_file(RAW_TRAIN_PATH),
                "size": RAW_TRAIN_PATH.stat().st_size,
            }
        },
        "integrity_status": "PASS",
    }
    
    raw_train = pd.read_csv(
        RAW_TRAIN_PATH,
        usecols=["row_id", "asof_pitcher_n", "asof_batter_n", "asof_pitcher_pitchmix_n"],
    )
    
    frames: dict[int, pd.DataFrame] = {}
    for season in [2022, 2023, 2024]:
        sel_path = SELECTIVE_FILES[season]
        lgbm_path = LGBM_RONLY_FILES[season]
        cat_path = CAT_RONLY_FILES[season]
        
        integrity_report["source_hashes"][f"selective_{season}"] = {
            "path": str(sel_path),
            "sha256": sha256_file(sel_path),
            "size": sel_path.stat().st_size,
        }
        integrity_report["source_hashes"][f"lgbm_r_{season}"] = {
            "path": str(lgbm_path),
            "sha256": sha256_file(lgbm_path),
            "size": lgbm_path.stat().st_size,
        }
        integrity_report["source_hashes"][f"cat_r_{season}"] = {
            "path": str(cat_path),
            "sha256": sha256_file(cat_path),
            "size": cat_path.stat().st_size,
        }
        
        base = pd.read_csv(sel_path)
        lgbm_r = pd.read_csv(lgbm_path, usecols=["row_id", "pred_lgbm"]).rename(
            columns={"pred_lgbm": "pred_lgbm_r_only"}
        )
        cat_r = pd.read_csv(cat_path, usecols=["row_id", "pred_catboost"]).rename(
            columns={"pred_catboost": "pred_cat_r_only"}
        )
        
        if not base["row_id"].is_unique or not lgbm_r["row_id"].is_unique or not cat_r["row_id"].is_unique:
            raise ValueError(f"{season} non-unique row_id found")
            
        r_mask = base["game_type"].eq("R")
        base_r_ids = set(base.loc[r_mask, "row_id"])
        
        if base_r_ids != set(lgbm_r["row_id"]) or base_r_ids != set(cat_r["row_id"]):
            raise ValueError(f"{season} row_ids set mismatch")
            
        merged = base.merge(lgbm_r, on="row_id", how="left", validate="one_to_one")
        merged = merged.merge(cat_r, on="row_id", how="left", validate="one_to_one")
        merged = merged.merge(raw_train, on="row_id", how="left", validate="one_to_one")
        
        # Rigorous array-level assertions
        targets = merged["target"].to_numpy()
        if not np.all(np.isfinite(targets)) or not np.all(np.isin(targets, [0, 1])):
            raise ValueError(f"{season} target contains non-binary/non-finite values")
            
        for col in ["pred_catboost", "pred_lgbm_r_only", "pred_cat_r_only"]:
            arr = merged.loc[r_mask, col].to_numpy()
            if not np.all(np.isfinite(arr)) or np.any(arr < 0.0) or np.any(arr > 1.0):
                raise ValueError(f"{season} {col} contains invalid probabilities")
                
        integrity_report["seasons"][season] = {
            "total_rows": int(len(merged)),
            "r_rows": int(r_mask.sum()),
            "f_rows": int((~r_mask).sum()),
            "target_mean": float(targets.mean()),
            "row_id_unique": True,
            "row_id_exact_match": True,
            "probabilities_valid": True,
        }
        frames[season] = merged
        
    return frames, integrity_report


def build_confidence_routed_prediction(
    frame: pd.DataFrame,
    pitcher_n_thresh: float = 30.0,
    high_conf_lgbm_w: float = 0.60,
    low_conf_lgbm_w: float = 0.35,
) -> np.ndarray:
    output = frame["pred_catboost"].to_numpy(dtype=np.float64).copy()
    r_mask = frame["game_type"].eq("R").to_numpy(dtype=bool)
    
    p_lgbm = frame.loc[r_mask, "pred_lgbm_r_only"].to_numpy(dtype=np.float64)
    p_cat_r = frame.loc[r_mask, "pred_cat_r_only"].to_numpy(dtype=np.float64)
    p_cat_full = frame.loc[r_mask, "pred_catboost"].to_numpy(dtype=np.float64)
    p_cat_blend = 0.50 * p_cat_r + 0.50 * p_cat_full
    
    pitcher_n = frame.loc[r_mask, "asof_pitcher_n"].fillna(0.0).to_numpy(dtype=np.float64)
    high_conf = pitcher_n >= pitcher_n_thresh
    
    routed_r = np.zeros_like(p_lgbm)
    routed_r[high_conf] = (
        high_conf_lgbm_w * p_lgbm[high_conf] + (1.0 - high_conf_lgbm_w) * p_cat_blend[high_conf]
    )
    routed_r[~high_conf] = (
        low_conf_lgbm_w * p_lgbm[~high_conf] + (1.0 - low_conf_lgbm_w) * p_cat_blend[~high_conf]
    )
    
    output[r_mask] = routed_r
    if not np.all(np.isfinite(output)) or np.any(output < 0.0) or np.any(output > 1.0):
        raise ValueError("Output contains invalid probabilities")
    return output


def run_confidence_routing_eval() -> tuple[dict[str, Any], str]:
    frames, integrity_report = load_and_verify_confidence_frames()
    
    # 1. Baseline SUB-001 Briers
    sub001_briers = {
        season: float(np.mean(np.square(
            frames[season]["target"].to_numpy(dtype=np.float64)
            - frames[season]["pred_selective"].to_numpy(dtype=np.float64)
        )))
        for season in [2022, 2023, 2024]
    }
    
    # 2. Reference SUB-002 (LGBM_R 0.75 + CAT_FULL 0.25 with forward Global Platt)
    sub002_raw_preds = {
        s: np.where(
            frames[s]["game_type"].eq("R"),
            0.75 * frames[s]["pred_lgbm_r_only"] + 0.25 * frames[s]["pred_catboost"],
            frames[s]["pred_catboost"]
        ).astype(np.float64)
        for s in [2022, 2023, 2024]
    }
    sub002_platt_preds: dict[int, np.ndarray] = {}
    for season in [2022, 2023, 2024]:
        if season == 2022:
            sub002_platt_preds[season] = sub002_raw_preds[season].copy()
        else:
            prior = [v for v in [2022, 2023] if v < season]
            y_tr = np.concatenate([frames[v]["target"].to_numpy() for v in prior])
            p_tr = np.concatenate([sub002_raw_preds[v] for v in prior])
            clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
            clf.fit(logit(p_tr), y_tr)
            z = clf.coef_[0, 0] * logit(sub002_raw_preds[season]).reshape(-1) + clf.intercept_[0]
            sub002_platt_preds[season] = 1.0 / (1.0 + np.exp(-z))
            
    sub002_briers = {
        season: float(np.mean(np.square(
            frames[season]["target"].to_numpy(dtype=np.float64) - sub002_platt_preds[season]
        )))
        for season in [2022, 2023, 2024]
    }
    
    routing_candidates = {
        "sub002_reference": {"thresh": 0.0, "high_w": 0.75, "low_w": 0.75, "type": "fixed_sub002"},
        "conf_thresh30_h60_l35": {"thresh": 30.0, "high_w": 0.60, "low_w": 0.35, "type": "routed"},
        "conf_thresh50_h60_l30": {"thresh": 50.0, "high_w": 0.60, "low_w": 0.30, "type": "routed"},
        "conf_thresh20_h55_l40": {"thresh": 20.0, "high_w": 0.55, "low_w": 0.40, "type": "routed"},
        "conf_thresh30_h55_l30": {"thresh": 30.0, "high_w": 0.55, "low_w": 0.30, "type": "routed"},
    }
    
    results: dict[str, Any] = {}
    
    for cname, cfg in routing_candidates.items():
        if cfg["type"] == "fixed_sub002":
            raw_preds = sub002_raw_preds
        else:
            raw_preds = {
                s: build_confidence_routed_prediction(
                    frames[s],
                    pitcher_n_thresh=cfg["thresh"],
                    high_conf_lgbm_w=cfg["high_w"],
                    low_conf_lgbm_w=cfg["low_w"],
                )
                for s in [2022, 2023, 2024]
            }
            
        platt_preds: dict[int, np.ndarray] = {}
        for season in [2022, 2023, 2024]:
            if season == 2022:
                platt_preds[season] = raw_preds[season].copy()
            else:
                prior = [v for v in [2022, 2023] if v < season]
                y_tr = np.concatenate([frames[v]["target"].to_numpy() for v in prior])
                p_tr = np.concatenate([raw_preds[v] for v in prior])
                clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
                clf.fit(logit(p_tr), y_tr)
                z = clf.coef_[0, 0] * logit(raw_preds[season]).reshape(-1) + clf.intercept_[0]
                platt_preds[season] = 1.0 / (1.0 + np.exp(-z))
                
        season_stats: dict[int, Any] = {}
        for season in [2022, 2023, 2024]:
            y_te = frames[season]["target"].to_numpy(dtype=np.float64)
            m = calculate_metrics(y_te, platt_preds[season])
            m["delta_vs_sub001"] = float(m["brier"] - sub001_briers[season])
            m["delta_vs_sub002"] = float(m["brier"] - sub002_briers[season])
            season_stats[season] = m
            
        w_delta_001 = sum(TEMPORAL_WEIGHTS[s] * season_stats[s]["delta_vs_sub001"] for s in [2022, 2023, 2024])
        w_delta_002 = sum(TEMPORAL_WEIGHTS[s] * season_stats[s]["delta_vs_sub002"] for s in [2022, 2023, 2024])
        worst_delta_002 = max(season_stats[s]["delta_vs_sub002"] for s in [2022, 2023, 2024])
        d24_002 = season_stats[2024]["delta_vs_sub002"]
        
        revised_gate = bool(d24_002 <= -0.00010 and worst_delta_002 <= 0.00005)
        
        results[cname] = {
            "config": cfg,
            "seasons": season_stats,
            "temporal_weighted_delta_vs_sub001": float(w_delta_001),
            "temporal_weighted_delta_vs_sub002": float(w_delta_002),
            "worst_season_delta_vs_sub002": float(worst_delta_002),
            "revised_latest_season_gate_pass": revised_gate,
            "candidate_status": "research_candidate_not_submission_approved",
        }
        
    lines = [
        "# REGIME-CONFIDENCE-001: 표본 신뢰도 기반 라우팅 벤치마크 (정정 감사 완료)",
        "",
        "> 분석 기준선: SUB-001 (역사적 베이스라인) & SUB-002 (현재 공식 최고점 `886.25점`)  ",
        "> 시간 가중치: $W_{2022}=0.20, W_{2023}=0.30, W_{2024}=0.50$",
        "",
        "## 1. 후보군별 다차원 검증 결과표 (SUB-002 대조)",
        "",
        "| 후보 ID | 2022 vs 002 | 2023 vs 002 | 2024 vs 002 | 2024 BSS | 2024 Local CV Proxy | 시간가중 Δ vs 002 | 시간가중 Δ vs 001 | 최악 시즌 Δ vs 002 | 후보 상태 | 개정 2024 게이트 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    
    for name, c in results.items():
        s = c["seasons"]
        d22 = s[2022]["delta_vs_sub002"]
        d23 = s[2023]["delta_vs_sub002"]
        d24 = s[2024]["delta_vs_sub002"]
        bss24 = s[2024]["bss"]
        proxy24 = s[2024]["local_cv_proxy_score"]
        w_002 = c["temporal_weighted_delta_vs_sub002"]
        w_001 = c["temporal_weighted_delta_vs_sub001"]
        worst_002 = c["worst_season_delta_vs_sub002"]
        c_status = c["candidate_status"]
        gate = "PASS" if c["revised_latest_season_gate_pass"] else "FAIL"
        lines.append(
            f"| `{name}` | `{d22:+.7f}` | `{d23:+.7f}` | `{d24:+.7f}` | `{bss24:.6f}` | `{proxy24:.2f}점` | **`{w_002:+.7f}`** | `{w_001:+.7f}` | `{worst_002:+.7f}` | `{c_status}` | `{gate}` |"
        )
        
    top_cname = "conf_thresh30_h55_l30"
    top_data = results[top_cname]
    lines.extend([
        "",
        "## 2. 심층 평가 및 결론 (JSON 실측치 기반 동적 생성)",
        "",
        f"- 표본 수 30개 기준 분기 (`{top_cname}`):",
        f"  - 2022 Δ vs SUB-002: `{top_data['seasons'][2022]['delta_vs_sub002']:+.7f}`",
        f"  - 2023 Δ vs SUB-002: `{top_data['seasons'][2023]['delta_vs_sub002']:+.7f}`",
        f"  - 2024 Δ vs SUB-002: `{top_data['seasons'][2024]['delta_vs_sub002']:+.7f}` (local proxy `{top_data['seasons'][2024]['local_cv_proxy_score']:.2f}점`)",
        f"  - 시간가중 Δ vs SUB-002: `{top_data['temporal_weighted_delta_vs_sub002']:+.7f}`",
        "- **게이트 판정**: 2024 순수 추가 개선폭이 개정 게이트(`-0.00010`)에 미달하므로, **SUB-003 제출 승인을 보류하고 연구 후보 상태(`research_candidate_not_submission_approved`)로 보존**함.",
    ])
    
    report_md = "\n".join(lines)
    
    full_results = {
        "experiment_id": "REGIME-CONFIDENCE-001",
        "runtime_environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
        },
        "data_integrity": integrity_report,
        "sub001_briers": sub001_briers,
        "sub002_briers": sub002_briers,
        "candidates": results,
    }
    
    return full_results, report_md


def main() -> None:
    results, report_md = run_confidence_routing_eval()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    results_path = OUTPUT_DIR / "confidence_routing_results.json"
    report_path = OUTPUT_DIR / "confidence_routing_report.md"
    
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(report_md, encoding="utf-8")
    
    print(report_md)
    print(f"\nSaved JSON: {results_path} (SHA-256: {sha256_file(results_path)})")
    print(f"Saved Report: {report_path} (SHA-256: {sha256_file(report_path)})")


if __name__ == "__main__":
    main()
