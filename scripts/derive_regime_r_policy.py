#!/usr/bin/env python3
"""Derive the locked final training policy for REGIME-R-001."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCES = [
    ROOT / "model" / "LGBM-FE001-RONLY-EW-2022",
    ROOT / "model" / "LGBM-FE001-RONLY-EW-2023",
    ROOT / "model" / "LGBM-FE001-RONLY-2024",
]
OUTPUT_DIR = ROOT / "model" / "REGIME-R-FINAL-POLICY-001"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    sources = []
    reference_features = None
    reference_params = None
    for directory in SOURCES:
        metadata_path = directory / "metadata.json"
        features_path = directory / "feature_columns.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        features = json.loads(features_path.read_text(encoding="utf-8"))
        if metadata.get("game_type_filter") != "R":
            raise ValueError(f"R-only source가 아닙니다: {directory}")
        if metadata["feature_count"] != 60 or len(features) != 60:
            raise ValueError(f"FE-001 60피처 source가 아닙니다: {directory}")
        if reference_features is None:
            reference_features = features
            reference_params = metadata["hyperparameters"]
        elif features != reference_features or metadata["hyperparameters"] != reference_params:
            raise ValueError("R-only Holdout 사이의 피처/파라미터 계약이 다릅니다")
        sources.append(
            {
                "experiment_id": metadata["exp_id"],
                "directory": str(directory),
                "train_seasons": metadata["train_seasons"],
                "valid_season": int(metadata["valid_season"]),
                "best_iteration": int(metadata["best_iteration"]),
                "brier": float(metadata["metrics"]["lgbm"]["brier_score"]),
                "metadata_sha256": sha256_file(metadata_path),
                "feature_columns_sha256": sha256_file(features_path),
            }
        )
    sources.sort(key=lambda item: item["valid_season"])
    if [item["valid_season"] for item in sources] != [2022, 2023, 2024]:
        raise ValueError("R-only Holdout 시즌 계약이 다릅니다")
    selected = sources[-1]
    policy = {
        "policy_id": "REGIME-R-FINAL-POLICY-001",
        "model_family": "lightgbm",
        "feature_contract": "FE-001-60",
        "training_filter": {"column": "game_type", "value": "R"},
        "selection_method": "latest_temporal_holdout_with_post_regime_training_context",
        "selected_source": selected,
        "selected_iterations": selected["best_iteration"],
        "sources": sources,
        "future_season": 2025,
        "final_training_contract": {
            "train_seasons": [2019, 2020, 2021, 2022, 2023, 2024],
            "game_type_filter": "R",
            "n_estimators": selected["best_iteration"],
            "validation_set": None,
            "early_stopping": False,
            "active_model_sync": False,
        },
        "regime_contract": {
            "F": {"catboost": 1.0},
            "R": {"lightgbm_r_only": 0.75, "catboost": 0.25},
            "postprocess": "global_platt",
        },
        "policy_gate_pass": True,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "regime_r_policy.json"
    md_path = OUTPUT_DIR / "regime_r_policy.md"
    json_path.write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# REGIME-R 최종 학습 정책",
        "",
        f"- 선택 source: `{selected['experiment_id']}`",
        f"- 고정 trees: `{selected['best_iteration']}`",
        "- 학습 행: 2019~2024 `game_type=R`",
        "- validation/early stopping: 없음",
        "- 활성 제출 동기화: 없음",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"))
    print(f"Policy SHA-256: {sha256_file(json_path)}")


if __name__ == "__main__":
    main()
