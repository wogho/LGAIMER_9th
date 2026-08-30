#!/usr/bin/env python3
"""Derive a reproducible final LightGBM iteration policy from time Holdouts."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = [
    ROOT / "model" / "LGBM-FE001-EW-2022",
    ROOT / "model" / "LGBM-FE001-EW-2023",
    ROOT / "model" / "LGBM-FE001-2024",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Derive final LightGBM trees from strict temporal metadata"
    )
    parser.add_argument(
        "--source-dir",
        type=Path,
        action="append",
        dest="source_dirs",
        help="Repeat for each Holdout model; defaults to 2022, 2023, 2024",
    )
    parser.add_argument("--regime-start-season", type=int, default=2023)
    parser.add_argument("--final-train-seasons", default="2019,2020,2021,2022,2023,2024")
    parser.add_argument("--future-season", type=int, default=2025)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "model" / "LGBM-FE001-FINAL-POLICY-001",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sources(source_dirs: list[Path]) -> list[dict[str, Any]]:
    sources: list[dict[str, Any]] = []
    reference_features: list[str] | None = None
    reference_hyperparameters: dict[str, Any] | None = None
    reference_categorical: list[str] | None = None
    for directory in source_dirs:
        metadata_path = directory / "metadata.json"
        features_path = directory / "feature_columns.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        features = json.loads(features_path.read_text(encoding="utf-8"))
        valid_season = int(metadata["valid_season"])
        expected_train_seasons = list(range(2019, valid_season))
        if not str(metadata["exp_id"]).startswith("LGBM-FE001"):
            raise ValueError(f"LightGBM FE-001 metadata가 아닙니다: {directory}")
        if metadata["train_seasons"] != expected_train_seasons:
            raise ValueError(f"strict expanding-window 계약이 아닙니다: {directory}")
        if metadata["feature_count"] != 60 or len(features) != 60:
            raise ValueError(f"FE-001 60피처 계약이 아닙니다: {directory}")
        if metadata["features"] != features:
            raise ValueError(f"metadata와 피처 JSON이 다릅니다: {directory}")
        if int(metadata["best_iteration"]) < 1:
            raise ValueError(f"best iteration이 잘못되었습니다: {directory}")
        if metadata["excluded_features"] or metadata["season_weight_decay"] != 1.0:
            raise ValueError(f"기준 FE-001 설정이 아닙니다: {directory}")
        if reference_features is None:
            reference_features = features
            reference_hyperparameters = metadata["hyperparameters"]
            reference_categorical = metadata["categorical_features"]
        elif features != reference_features:
            raise ValueError("Holdout 모델 사이의 피처 계약이 다릅니다")
        elif metadata["hyperparameters"] != reference_hyperparameters:
            raise ValueError("Holdout 모델 사이의 LightGBM 설정이 다릅니다")
        elif metadata["categorical_features"] != reference_categorical:
            raise ValueError("Holdout 모델 사이의 categorical 계약이 다릅니다")
        sources.append(
            {
                "experiment_id": metadata["exp_id"],
                "directory": str(directory),
                "train_seasons": metadata["train_seasons"],
                "valid_season": valid_season,
                "best_iteration": int(metadata["best_iteration"]),
                "brier": float(metadata["metrics"]["lgbm"]["brier_score"]),
                "bss": float(metadata["metrics"]["lgbm"]["bss"]),
                "metadata_sha256": sha256_file(metadata_path),
                "feature_columns_sha256": sha256_file(features_path),
            }
        )
    sources.sort(key=lambda item: item["valid_season"])
    seasons = [item["valid_season"] for item in sources]
    if seasons != list(range(min(seasons), max(seasons) + 1)):
        raise ValueError("Holdout 시즌이 연속적이지 않습니다")
    return sources


def render_markdown(policy: dict[str, Any]) -> str:
    lines = [
        "# LightGBM FE-001 최종 반복 수 정책",
        "",
        f"- 선택 방식: `{policy['selection_method']}`",
        f"- 체제 시작 시즌: `{policy['regime_start_season']}`",
        f"- 선택 source Holdout: `{policy['selected_source']['valid_season']}`",
        f"- 최종 고정 trees: `{policy['selected_iterations']}`",
        "- 전체 학습 시 early stopping 및 tree 수 확대/축소 없음",
        "",
        "| 검증 시즌 | 학습 시즌 | Best iteration | Brier | 선택 적격 |",
        "|---:|---|---:|---:|---|",
    ]
    eligible_seasons = set(policy["eligible_source_seasons"])
    for source in policy["sources"]:
        train_label = f"{source['train_seasons'][0]}~{source['train_seasons'][-1]}"
        eligible = "예" if source["valid_season"] in eligible_seasons else "아니오"
        lines.append(
            f"| {source['valid_season']} | {train_label} | {source['best_iteration']} | "
            f"{source['brier']:.9f} | {eligible} |"
        )
    lines.extend(
        [
            "",
            "## 근거",
            "",
            "- 최종 2019~2024 학습은 2023 체제 전환 이후 시즌을 모두 포함한다.",
            "- 체제 시작 시즌을 학습 데이터에 포함한 시간 Holdout만 최종 iteration source로 허용한다.",
            "- 적격 source 중 미래 시즌에 가장 가까운 최신 Holdout의 best iteration을 사용한다.",
            "- 서로 다른 정보 체제에서 나온 iteration의 평균이나 중앙값은 사용하지 않는다.",
            "",
            "## 다음 단계 계약",
            "",
            f"- 학습 시즌: `{policy['final_training_contract']['train_seasons']}`",
            f"- n_estimators: `{policy['final_training_contract']['n_estimators']}`",
            "- validation/early stopping: 사용 안 함",
            "- 후보 모델과 격리된 디렉터리에 저장하고 명시적 승인 전 활성화하지 않음",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    sources = load_sources(args.source_dirs or DEFAULT_SOURCES)
    final_train_seasons = [
        int(value.strip())
        for value in args.final_train_seasons.split(",")
        if value.strip()
    ]
    if final_train_seasons != list(range(2019, args.future_season)):
        raise ValueError("최종 학습 시즌은 2019부터 미래 시즌 직전까지 연속이어야 합니다")
    if args.regime_start_season not in final_train_seasons:
        raise ValueError("체제 시작 시즌이 최종 학습 범위에 없습니다")

    eligible = [
        source
        for source in sources
        if args.regime_start_season in source["train_seasons"]
        and source["valid_season"] > args.regime_start_season
    ]
    if not eligible:
        raise ValueError("체제 시작 시즌을 학습에 포함한 Holdout source가 없습니다")
    selected = max(eligible, key=lambda item: item["valid_season"])
    iterations = [source["best_iteration"] for source in sources]
    policy = {
        "policy_id": "LGBM-FE001-FINAL-POLICY-001",
        "model_family": "lightgbm",
        "feature_contract": "FE-001-60",
        "selection_method": "latest_temporal_holdout_with_post_regime_training_context",
        "regime_start_season": args.regime_start_season,
        "future_season": args.future_season,
        "eligible_source_seasons": [source["valid_season"] for source in eligible],
        "selected_source": selected,
        "selected_iterations": selected["best_iteration"],
        "iteration_scaling_after_full_data": False,
        "aggregate_iterations_not_used": {
            "mean": statistics.mean(iterations),
            "median": statistics.median(iterations),
            "minimum": min(iterations),
            "maximum": max(iterations),
            "range": max(iterations) - min(iterations),
            "reason": "different temporal information regimes must not be averaged",
        },
        "sources": sources,
        "final_training_contract": {
            "train_seasons": final_train_seasons,
            "n_estimators": selected["best_iteration"],
            "validation_set": None,
            "early_stopping": False,
            "active_model_sync": False,
        },
        "risk_flags": [
            "holdout_best_iteration_range_is_large",
            "2023_regime_onset_fold_is_not_representative_of_final_training_context",
            "candidate_must_remain_isolated_until_ensemble_contract_passes",
        ],
        "policy_gate_pass": True,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = args.output_dir / "lightgbm_iteration_policy.json"
    report_path = args.output_dir / "lightgbm_iteration_policy.md"
    policy_path.write_text(
        json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path.write_text(render_markdown(policy), encoding="utf-8")
    print(report_path.read_text(encoding="utf-8"))
    print(f"Saved policy: {policy_path}")
    print(f"Policy SHA-256: {sha256_file(policy_path)}")


if __name__ == "__main__":
    main()
