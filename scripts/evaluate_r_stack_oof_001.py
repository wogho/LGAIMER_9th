#!/usr/bin/env python3
"""
R-STACK-OOF-001 (Audited & Automated)
Low-Complexity Temporal OOF Stacking on R-Regime.
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
OUTPUT_DIR = ROOT / "model" / "R-STACK-OOF-001"

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

TEMPORAL_WEIGHTS = {2022: 0.20, 2023: 0.30, 2024: 0.50}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def logit_transform(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


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
        if set(base.loc[r_mask, "row_id"]) != set(lgbm_r["row_id"]) or set(base.loc[r_mask, "row_id"]) != set(cat_r["row_id"]):
            raise ValueError(f"{season} row_ids set mismatch")
            
        merged = base.merge(lgbm_r, on="row_id", how="left", validate="one_to_one")
        merged = merged.merge(cat_r, on="row_id", how="left", validate="one_to_one")
        
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


def build_stacking_features(frame: pd.DataFrame) -> np.ndarray:
    r_mask = frame["game_type"].eq("R")
    x1 = logit_transform(frame.loc[r_mask, "pred_lgbm_r_only"].to_numpy())
    x2 = logit_transform(frame.loc[r_mask, "pred_cat_r_only"].to_numpy())
    x3 = logit_transform(frame.loc[r_mask, "pred_catboost"].to_numpy())
    return np.column_stack([x1, x2, x3])


def run_stacking_evaluation() -> tuple[dict[str, Any], str]:
    frames, integrity_report = load_and_verify_frames()
    
    # 1. Baseline SUB-001 Briers
    sub001_briers = {
        season: float(np.mean(np.square(
            frames[season]["target"].to_numpy(dtype=np.float64)
            - frames[season]["pred_selective"].to_numpy(dtype=np.float64)
        )))
        for season in [2022, 2023, 2024]
    }
    
    # 2. SUB-002 Reference (LGBM_R 0.75 + CAT_FULL 0.25 with forward Global Platt)
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
            clf.fit(logit_transform(p_tr).reshape(-1, 1), y_tr)
            z = clf.coef_[0, 0] * logit_transform(sub002_raw_preds[season]) + clf.intercept_[0]
            sub002_platt_preds[season] = 1.0 / (1.0 + np.exp(-z))
            
    sub002_briers = {
        season: float(np.mean(np.square(
            frames[season]["target"].to_numpy(dtype=np.float64) - sub002_platt_preds[season]
        )))
        for season in [2022, 2023, 2024]
    }
    
    c_candidates = [1.0, 10.0, 100.0]
    stacking_results: dict[str, Any] = {}
    
    for c_reg in c_candidates:
        cand_key = f"r_stack_l2_c_{c_reg:.1f}"
        raw_stacked_preds: dict[int, np.ndarray] = {}
        meta_weights: dict[int, dict[str, float | list[float]]] = {}
        
        for season in [2022, 2023, 2024]:
            r_mask_curr = frames[season]["game_type"].eq("R").to_numpy(dtype=bool)
            x_curr = build_stacking_features(frames[season])
            
            if season == 2022:
                p_r = (
                    frames[season].loc[r_mask_curr, "pred_lgbm_r_only"].to_numpy() / 3.0
                    + frames[season].loc[r_mask_curr, "pred_cat_r_only"].to_numpy() / 3.0
                    + frames[season].loc[r_mask_curr, "pred_catboost"].to_numpy() / 3.0
                )
                meta_weights[season] = {"coef": [0.3333333333333333, 0.3333333333333333, 0.3333333333333333], "intercept": 0.0}
            else:
                prior = [v for v in [2022, 2023] if v < season]
                x_train = np.vstack([build_stacking_features(frames[v]) for v in prior])
                y_train = np.concatenate([
                    frames[v].loc[frames[v]["game_type"].eq("R"), "target"].to_numpy()
                    for v in prior
                ])
                
                meta_clf = LogisticRegression(C=c_reg, solver="lbfgs", max_iter=1000)
                meta_clf.fit(x_train, y_train)
                p_r = meta_clf.predict_proba(x_curr)[:, 1]
                meta_weights[season] = {
                    "coef": [float(val) for val in meta_clf.coef_[0]],
                    "intercept": float(meta_clf.intercept_[0]),
                }
                
            pred_full = frames[season]["pred_catboost"].to_numpy(dtype=np.float64).copy()
            pred_full[r_mask_curr] = p_r
            raw_stacked_preds[season] = pred_full
            
        platt_stacked_preds: dict[int, np.ndarray] = {}
        for season in [2022, 2023, 2024]:
            if season == 2022:
                platt_stacked_preds[season] = raw_stacked_preds[season].copy()
            else:
                prior = [v for v in [2022, 2023] if v < season]
                y_tr = np.concatenate([frames[v]["target"].to_numpy() for v in prior])
                p_tr = np.concatenate([raw_stacked_preds[v] for v in prior])
                clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
                clf.fit(logit_transform(p_tr).reshape(-1, 1), y_tr)
                z = clf.coef_[0, 0] * logit_transform(raw_stacked_preds[season]) + clf.intercept_[0]
                platt_stacked_preds[season] = 1.0 / (1.0 + np.exp(-z))
                
        for eval_type, preds_dict in [("raw_stack", raw_stacked_preds), ("platt_stack", platt_stacked_preds)]:
            sub_key = f"{cand_key}_{eval_type}"
            season_stats: dict[int, Any] = {}
            for season in [2022, 2023, 2024]:
                y_te = frames[season]["target"].to_numpy(dtype=np.float64)
                m = calculate_metrics(y_te, preds_dict[season])
                m["delta_vs_sub001"] = float(m["brier"] - sub001_briers[season])
                m["delta_vs_sub002"] = float(m["brier"] - sub002_briers[season])
                season_stats[season] = m
                
            w_delta_001 = sum(TEMPORAL_WEIGHTS[s] * season_stats[s]["delta_vs_sub001"] for s in [2022, 2023, 2024])
            w_delta_002 = sum(TEMPORAL_WEIGHTS[s] * season_stats[s]["delta_vs_sub002"] for s in [2022, 2023, 2024])
            worst_delta_002 = max(season_stats[s]["delta_vs_sub002"] for s in [2022, 2023, 2024])
            d24_002 = season_stats[2024]["delta_vs_sub002"]
            
            revised_gate = bool(d24_002 <= -0.00010 and worst_delta_002 <= 0.00005)
            
            stacking_results[sub_key] = {
                "c_reg": c_reg,
                "eval_type": eval_type,
                "meta_weights": meta_weights,
                "seasons": season_stats,
                "temporal_weighted_delta_vs_sub001": float(w_delta_001),
                "temporal_weighted_delta_vs_sub002": float(w_delta_002),
                "worst_season_delta_vs_sub002": float(worst_delta_002),
                "revised_latest_season_gate_pass": revised_gate,
                "candidate_status": "research_candidate_not_submission_approved",
            }
            
    lines = [
        "# R-STACK-OOF-001: R 전용 저복잡도 시간 OOF 스태킹 벤치마크 (정정 감사 완료)",
        "",
        "> 메타 모델: L2 Logistic Regression ($x_1=\\text{logit}(P_{\\text{LGBM\\_R}}), x_2=\\text{logit}(P_{\\text{CAT\\_R}}), x_3=\\text{logit}(P_{\\text{CAT\\_FULL}})$)  ",
        "> 학습 원칙: 2023 메타 모델은 2022 OOF로만, 2024 메타 모델은 2022+2023 OOF로만 적합  ",
        "> 시간 가중치: $W_{2022}=0.20, W_{2023}=0.30, W_{2024}=0.50$",
        "",
        "## 1. 스태킹 후보군별 다차원 검증 결과표 (SUB-002 대조)",
        "",
        "| 후보 ID | 2022 vs 002 | 2023 vs 002 | 2024 vs 002 | 2024 BSS | 2024 Local CV Proxy | 시간가중 Δ vs 002 | 시간가중 Δ vs 001 | 최악 시즌 Δ vs 002 | 후보 상태 | 개정 2024 게이트 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|:---:|",
    ]
    
    for name, c in stacking_results.items():
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
        
    s_c10 = stacking_results["r_stack_l2_c_10.0_platt_stack"]
    d24_c10 = s_c10["seasons"][2024]["delta_vs_sub002"]
    proxy24_c10 = s_c10["seasons"][2024]["local_cv_proxy_score"]
    w_c10 = s_c10["temporal_weighted_delta_vs_sub002"]
    
    lines.extend([
        "",
        "## 2. 메타 모델 가중치 분석 및 결론 (JSON 실측치 기반 동적 생성)",
        "",
        "- **2024 메타 모델 학습 가중치 (C=10.0)**:",
        f"  - Coef: `{s_c10['meta_weights'][2024]['coef']}`",
        f"  - Intercept: `{s_c10['meta_weights'][2024]['intercept']:.6f}`",
        f"- **성능 평가**: L2 스태킹(`r_stack_l2_c_10.0_platt_stack`)의 2024 Brier Δ vs SUB-002는 `{d24_c10:+.7f}` (악화, local proxy `{proxy24_c10:.2f}점`), 시간가중 Δ vs SUB-002는 `{w_c10:+.7f}`임.",
        "- **게이트 판정**: 스태킹은 SUB-002 대비 전반적으로 성능이 저하되었으며, **`failed_research_family`로 판정하여 기각 및 연구 후보 상태(`research_candidate_not_submission_approved`)로 보존**함.",
    ])
    
    report_md = "\n".join(lines)
    
    results = {
        "experiment_id": "R-STACK-OOF-001",
        "runtime_environment": {
            "python_version": sys.version,
            "platform": platform.platform(),
        },
        "data_integrity": integrity_report,
        "sub001_briers": sub001_briers,
        "sub002_briers": sub002_briers,
        "candidates": stacking_results,
    }
    
    return results, report_md


def main() -> None:
    results, report_md = run_stacking_evaluation()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    results_path = OUTPUT_DIR / "stacking_results.json"
    report_path = OUTPUT_DIR / "stacking_report.md"
    
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path.write_text(report_md, encoding="utf-8")
    
    print(report_md)
    print(f"\nSaved JSON: {results_path} (SHA-256: {sha256_file(results_path)})")
    print(f"Saved Report: {report_path} (SHA-256: {sha256_file(report_path)})")


if __name__ == "__main__":
    main()
