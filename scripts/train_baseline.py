#!/usr/bin/env python3
"""
train_baseline.py — 시즌 기반 검증(2024 Holdout) 및 LightGBM 베이스라인 학습 파이프라인

실행:
  python scripts/train_baseline.py --exp-id LGBM-001

출력:
  model/LGBM-001/
    ├── model.txt                 # LightGBM 네이티브 가중치
    ├── feature_columns.json      # 피처 순서/목록
    ├── metadata.json             # 실험 설정 및 점수 메트릭
    └── validation_predictions.csv # 2024 검증 예측값
"""
import argparse
import json
import os
import resource
import shutil
import sys
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

# Root path configuration
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.insert(0, root)

from src.features import (
    ID_COL,
    TARGET_COL,
    brier_score,
    brier_skill_score,
    build_features,
    build_features_fe002,
    hackathon_score,
    load_data,
)
from src.target_aggregates import (
    build_batter_target_history,
    build_pitcher_count_state_target_history,
    build_pitcher_scoring_pos_target_history,
    build_pitcher_target_history,
)


def get_peak_memory_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def main():
    parser = argparse.ArgumentParser(description="Train LightGBM Baseline with Season-based Validation")
    parser.add_argument("--train-path", default=os.path.join(root, "data", "train.csv"), help="Path to train.csv")
    parser.add_argument("--exp-id", default="LGBM-001", help="Experiment identifier")
    parser.add_argument("--train-seasons", default="2019,2020,2021,2022,2023", help="Comma-separated train seasons")
    parser.add_argument("--valid-season", type=int, default=2024, help="Holdout validation season")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="LightGBM learning rate")
    parser.add_argument("--num-leaves", type=int, default=63, help="LightGBM num_leaves")
    parser.add_argument(
        "--min-child-samples",
        type=int,
        default=100,
        help="LightGBM min_child_samples (default: 100)",
    )
    parser.add_argument("--n-estimators", type=int, default=1500, help="Maximum boosting rounds")
    parser.add_argument("--early-stopping", type=int, default=50, help="Early stopping rounds")
    parser.add_argument(
        "--game-type-filter",
        default="",
        help="학습·검증 행을 단일 game_type(F 또는 R)으로 제한",
    )
    parser.add_argument(
        "--fe002-groups",
        default="",
        help="현재 행 기반 FE-002 그룹 쉼표 목록: state,form,support",
    )
    parser.add_argument(
        "--exclude-features",
        default="",
        help="모델 입력에서 제외할 쉼표 구분 피처 목록 (예: season,pitcher_id)",
    )
    parser.add_argument(
        "--season-weight-decay",
        type=float,
        default=1.0,
        help=(
            "최근 학습 시즌의 가중치를 1로 두고 과거 1년마다 곱할 감쇠율. "
            "1.0은 가중치 미적용, 범위는 0 초과 1 이하"
        ),
    )
    parser.add_argument(
        "--pitcher-target-history",
        action="store_true",
        help=(
            "이전 시즌 타깃만 사용하는 투수별 과거 성공률·표본 수 피처를 추가하고 "
            "검증/추론용 train-only lookup을 저장"
        ),
    )
    parser.add_argument(
        "--batter-target-history",
        action="store_true",
        help=(
            "이전 시즌 타깃만 사용하는 타자별 과거 상대 성공률·표본 수 피처를 "
            "추가하고 검증/추론용 train-only lookup을 저장"
        ),
    )
    parser.add_argument(
        "--pitcher-count-target-deviation",
        action="store_true",
        help=(
            "이전 시즌 타깃만 사용하는 투수×count_state 표본 수와 "
            "투수 전체 대비 조건부 성공률 편차를 추가"
        ),
    )
    parser.add_argument(
        "--pitcher-scoring-pos-target-deviation",
        action="store_true",
        help=(
            "이전 시즌 타깃만 사용하는 투수×득점권 여부 표본 수와 "
            "투수 전체 대비 조건부 성공률 편차를 추가"
        ),
    )
    parser.add_argument(
        "--aggregate-smoothing",
        type=float,
        default=100.0,
        help="투수 타깃 집계의 Bayesian smoothing 강도 (default: 100)",
    )
    parser.add_argument(
        "--sync-root-model",
        action="store_true",
        help=(
            "검증 산출물을 활성 제출 모델(model/model.txt)로 명시적으로 동기화. "
            "기본값은 비활성화하여 탐색 실험의 덮어쓰기를 방지"
        ),
    )
    args = parser.parse_args()

    t_start = time.time()
    train_seasons = [int(s.strip()) for s in args.train_seasons.split(",")]
    valid_season = args.valid_season
    excluded_features = [
        feature.strip()
        for feature in args.exclude_features.split(",")
        if feature.strip()
    ]
    fe002_groups = [
        group.strip() for group in args.fe002_groups.split(",") if group.strip()
    ]
    if len(fe002_groups) != len(set(fe002_groups)):
        parser.error("--fe002-groups에 중복 그룹이 있습니다")
    if args.game_type_filter not in {"", "F", "R"}:
        parser.error("--game-type-filter는 F, R 또는 빈 값이어야 합니다")
    if len(excluded_features) != len(set(excluded_features)):
        parser.error("--exclude-features에 중복 피처가 있습니다")
    if args.min_child_samples < 1:
        parser.error("--min-child-samples는 1 이상이어야 합니다")
    if not np.isfinite(args.aggregate_smoothing) or args.aggregate_smoothing <= 0:
        parser.error("--aggregate-smoothing은 0보다 큰 유한값이어야 합니다")
    if not 0.0 < args.season_weight_decay <= 1.0:
        parser.error("--season-weight-decay는 0 초과 1 이하이어야 합니다")
    aggregate_flags = [
        args.pitcher_target_history,
        args.batter_target_history,
        args.pitcher_count_target_deviation,
        args.pitcher_scoring_pos_target_deviation,
    ]
    if sum(aggregate_flags) > 1:
        parser.error(
            "단일 변경 실험을 위해 타깃 집계 옵션은 하나만 사용할 수 있습니다"
        )
    if any(aggregate_flags) and args.sync_root_model:
        parser.error(
            "train-only 집계 실험은 활성 추론 lookup 계약이 확정되기 전 "
            "--sync-root-model을 사용할 수 없습니다"
        )

    print("=" * 70)
    print(f"  LG Aimers — Baseline Model Training & Validation [{args.exp_id}]")
    print("=" * 70)
    print(f"  학습 시즌: {train_seasons}")
    print(f"  검증 시즌: {valid_season} (Holdout)")
    print(f"  랜덤 시드: {args.seed}")
    print(f"  제외 피처: {excluded_features or '없음'}")
    print(f"  game_type 필터: {args.game_type_filter or '없음'}")
    print(f"  FE-002 그룹: {fe002_groups or '없음'}")
    print(f"  min_child_samples: {args.min_child_samples}")
    print(f"  투수 타깃 과거 집계: {args.pitcher_target_history}")
    print(f"  타자 타깃 과거 집계: {args.batter_target_history}")
    print(f"  투수×count_state 조건부 편차: {args.pitcher_count_target_deviation}")
    print(f"  투수×득점권 조건부 편차: {args.pitcher_scoring_pos_target_deviation}")
    if any(aggregate_flags):
        print(f"  집계 smoothing: {args.aggregate_smoothing}")
    print(f"  시즌 가중치 감쇠율: {args.season_weight_decay}")
    print(f"  데이터 경로: {args.train_path}")

    # 1. 전체 학습 데이터 로드
    print("\n[1/6] 데이터 로드 중...")
    if not os.path.isfile(args.train_path):
        print(f"❌ [에러] 파일이 존재하지 않습니다: {args.train_path}")
        sys.exit(1)

    df_all = load_data(args.train_path, is_train=True)

    # 2. 시즌별 데이터 분포 통계
    print("\n[2/6] 시즌별 데이터 및 타깃 분포 확인:")
    season_stats = df_all.groupby("season").agg(
        rows=("row_id", "count"),
        pos_count=(TARGET_COL, "sum"),
        pos_ratio=(TARGET_COL, "mean")
    ).reset_index()
    print(f"{'Season':>8} | {'Rows':>12} | {'Success Count':>14} | {'Success Ratio (%)':>18}")
    print("-" * 60)
    for _, row in season_stats.iterrows():
        print(f"{int(row['season']):>8} | {int(row['rows']):>12,} | {int(row['pos_count']):>14,} | {row['pos_ratio']*100:>17.2f}%")

    # 3. 시즌 기준 Train / Validation 분할
    print("\n[3/6] 시즌 분할 및 피처 생성...")
    train_mask = df_all["season"].isin(train_seasons)
    valid_mask = df_all["season"] == valid_season

    train_df = df_all[train_mask].copy()
    valid_df = df_all[valid_mask].copy()
    if args.game_type_filter:
        train_df = train_df.loc[
            train_df["game_type"].astype("string").eq(args.game_type_filter)
        ].copy()
        valid_df = valid_df.loc[
            valid_df["game_type"].astype("string").eq(args.game_type_filter)
        ].copy()
        if train_df.empty or valid_df.empty:
            raise ValueError("game_type 필터 적용 후 학습 또는 검증 행이 없습니다")

    y_train = train_df[TARGET_COL].to_numpy(dtype=np.int8)
    y_valid = valid_df[TARGET_COL].to_numpy(dtype=np.int8)

    if fe002_groups:
        X_train = build_features_fe002(train_df, fe002_groups)
        X_valid = build_features_fe002(valid_df, fe002_groups)
    else:
        X_train = build_features(train_df)
        X_valid = build_features(valid_df)

    aggregate_lookup = None
    aggregate_metadata = None
    aggregate_entity_label = None
    if any(aggregate_flags):
        if args.pitcher_target_history:
            aggregate_builder = build_pitcher_target_history
            aggregate_entity_label = "pitcher"
        elif args.batter_target_history:
            aggregate_builder = build_batter_target_history
            aggregate_entity_label = "batter"
        elif args.pitcher_count_target_deviation:
            aggregate_builder = build_pitcher_count_state_target_history
            aggregate_entity_label = "pitcher_count"
        else:
            aggregate_builder = build_pitcher_scoring_pos_target_history
            aggregate_entity_label = "pitcher_scoring_pos"
        (
            train_aggregate_features,
            valid_aggregate_features,
            aggregate_lookup,
            aggregate_metadata,
        ) = aggregate_builder(
            train_df,
            valid_df,
            smoothing=args.aggregate_smoothing,
        )
        X_train = pd.concat([X_train, train_aggregate_features], axis=1)
        X_valid = pd.concat([X_valid, valid_aggregate_features], axis=1)
        print(
            "  - train-only 집계 피처 추가: "
            f"{aggregate_metadata['feature_columns']} | "
            f"valid 신규 ID 행 비율={aggregate_metadata['valid_unseen_row_rate']:.4f}"
        )

    missing_exclusions = [
        feature for feature in excluded_features if feature not in X_train.columns
    ]
    if missing_exclusions:
        raise ValueError(f"제외 대상 피처가 모델 입력에 없습니다: {missing_exclusions}")
    if excluded_features:
        X_train = X_train.drop(columns=excluded_features)
        X_valid = X_valid.drop(columns=excluded_features)

    feature_names = list(X_train.columns)
    print(f"  - 학습셋 크기: {len(X_train):,} 행 × {len(feature_names)} 피처 (시즌: {train_seasons})")
    print(f"  - 검증셋 크기: {len(X_valid):,} 행 × {len(feature_names)} 피처 (시즌: {valid_season})")

    # 범주형 컬럼 확인
    cat_cols = [c for c in X_train.columns if str(X_train[c].dtype) == "category" or X_train[c].dtype == "object"]
    for c in cat_cols:
        X_train[c] = X_train[c].astype("category")
        X_valid[c] = X_valid[c].astype("category")
    print(f"  - 범주형 피처 ({len(cat_cols)}개): {cat_cols}")

    latest_train_season = max(train_seasons)
    train_season_values = train_df["season"].to_numpy(dtype=np.int16)
    train_weights = np.power(
        args.season_weight_decay,
        latest_train_season - train_season_values,
    ).astype(np.float64)
    season_weight_map = {
        str(season): float(args.season_weight_decay ** (latest_train_season - season))
        for season in sorted(train_seasons)
    }
    print(f"  - 시즌별 학습 가중치: {season_weight_map}")

    # 4. 베이스라인 기준선 평가
    print("\n[4/6] 기준선 모델 점수 측정...")
    # 4-1. 상수 모델 (Train 타깃 평균)
    train_mean = float(np.mean(y_train))
    valid_mean = float(np.mean(y_valid))
    pred_const = np.full_like(y_valid, fill_value=train_mean, dtype=np.float64)
    bs_const = brier_score(y_valid, pred_const)
    bss_const = brier_skill_score(y_valid, pred_const)
    score_const = hackathon_score(y_valid, pred_const)

    print(f"  [상수 모델] (p_train={train_mean:.5f}, p_valid={valid_mean:.5f})")
    print(
        f"    - Brier Score: {bs_const:.6f} | BSS: {bss_const:.6f} "
        f"| 로컬 환산 점수: {score_const:.2f}"
    )

    # 5. LightGBM 모델 학습
    print("\n[5/6] LightGBM 모델 학습 시작...")
    lgb_params = {
        "objective": "binary",
        "metric": "binary_logloss",
        "boosting_type": "gbdt",
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "max_depth": -1,
        "min_child_samples": args.min_child_samples,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "random_state": args.seed,
        "n_jobs": -1,
        "verbose": -1,
    }

    dtrain = lgb.Dataset(
        X_train,
        label=y_train,
        weight=train_weights if args.season_weight_decay < 1.0 else None,
        categorical_feature=cat_cols,
        free_raw_data=False,
    )
    dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain, categorical_feature=cat_cols, free_raw_data=False)

    callbacks = [
        lgb.early_stopping(stopping_rounds=args.early_stopping, verbose=True),
        lgb.log_evaluation(period=50),
    ]

    t_train_start = time.time()
    booster = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=args.n_estimators,
        valid_sets=[dtrain, dvalid],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )
    t_train_elapsed = time.time() - t_train_start
    best_iteration = booster.best_iteration
    print(f"  ✅ 학습 완료 (최적 트리 수: {best_iteration}, 소요 시간: {t_train_elapsed:.1f}초)")

    # 5-1. 검증 예측 및 평가
    pred_lgbm = booster.predict(X_valid, num_iteration=best_iteration)
    bs_lgbm = brier_score(y_valid, pred_lgbm)
    bss_lgbm = brier_skill_score(y_valid, pred_lgbm)
    score_lgbm = hackathon_score(y_valid, pred_lgbm)

    print("\n" + "=" * 70)
    print(f"  🏆 모델별 {valid_season} Holdout 검증 결과 비교")
    print("=" * 70)
    print(f"{'Model':<25} | {'Brier Score':<12} | {'BSS':<10} | {'Local Score':<15} | {'Status'}")
    print("-" * 75)
    print(f"{'Train Mean Constant':<25} | {bs_const:<12.6f} | {bss_const:<10.6f} | {score_const:<15.2f} | Operational Ref")
    print(f"{f'LightGBM ({args.exp_id})':<25} | {bs_lgbm:<12.6f} | {bss_lgbm:<10.6f} | {score_lgbm:<15.2f} | Validation Candidate")
    print("=" * 75)

    # 6. 산출물 및 메타데이터 저장
    print("\n[6/6] 실험 산출물 저장 중...")
    exp_dir = os.path.join(root, "model", args.exp_id)
    os.makedirs(exp_dir, exist_ok=True)

    model_txt_path = os.path.join(exp_dir, "model.txt")
    booster.save_model(model_txt_path)
    print(f"  ✅ 모델 가중치 저장: {model_txt_path} ({os.path.getsize(model_txt_path) / 1e6:.2f} MB)")

    # 피처 목록 저장
    feat_json_path = os.path.join(exp_dir, "feature_columns.json")
    with open(feat_json_path, "w", encoding="utf-8") as f:
        json.dump(feature_names, f, indent=2, ensure_ascii=False)
    print(f"  ✅ 피처 목록 저장 ({len(feature_names)}개): {feat_json_path}")

    if aggregate_lookup is not None and aggregate_metadata is not None:
        aggregate_lookup_path = os.path.join(
            exp_dir,
            f"{aggregate_entity_label}_target_lookup.csv",
        )
        aggregate_lookup.to_csv(aggregate_lookup_path, index=False)
        aggregate_metadata = {
            **aggregate_metadata,
            "lookup_file": os.path.basename(aggregate_lookup_path),
        }
        aggregate_metadata_path = os.path.join(exp_dir, "target_aggregate.json")
        with open(aggregate_metadata_path, "w", encoding="utf-8") as f:
            json.dump(aggregate_metadata, f, indent=2, ensure_ascii=False)
        print(
            "  ✅ train-only 엔터티 lookup 저장: "
            f"{aggregate_lookup_path} ({len(aggregate_lookup):,} 엔터티)"
        )
        print(f"  ✅ 집계 계약 저장: {aggregate_metadata_path}")

    # 메타데이터 저장
    total_elapsed = time.time() - t_start
    peak_mem = get_peak_memory_mb()
    metadata = {
        "exp_id": args.exp_id,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "train_seasons": train_seasons,
        "valid_season": valid_season,
        "seed": args.seed,
        "train_rows": int(len(X_train)),
        "valid_rows": int(len(X_valid)),
        "feature_count": int(len(feature_names)),
        "features": feature_names,
        "excluded_features": excluded_features,
        "feature_contract": "FE-002" if fe002_groups else "FE-001",
        "fe002_groups": fe002_groups,
        "game_type_filter": args.game_type_filter or None,
        "categorical_features": cat_cols,
        "season_weight_decay": args.season_weight_decay,
        "season_weight_map": season_weight_map,
        "target_aggregate": aggregate_metadata,
        "best_iteration": int(best_iteration),
        "hyperparameters": lgb_params,
        "metrics": {
            "constant_baseline": {"brier_score": bs_const, "bss": bss_const, "score": score_const},
            "lgbm": {"brier_score": bs_lgbm, "bss": bss_lgbm, "score": score_lgbm},
        },
        "training_time_sec": round(t_train_elapsed, 2),
        "total_time_sec": round(total_elapsed, 2),
        "peak_memory_mb": round(peak_mem, 1),
    }

    meta_json_path = os.path.join(exp_dir, "metadata.json")
    with open(meta_json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  ✅ 메타데이터 저장: {meta_json_path}")

    # 검증 예측 결과 CSV 저장
    val_pred_df = pd.DataFrame({
        "row_id": valid_df["row_id"].values,
        "season": valid_df["season"].values,
        "target": y_valid,
        "pred_constant": pred_const,
        "pred_lgbm": pred_lgbm,
    })
    val_pred_csv_path = os.path.join(exp_dir, "validation_predictions.csv")
    val_pred_df.to_csv(val_pred_csv_path, index=False)
    print(f"  ✅ 검증 예측값 저장: {val_pred_csv_path} ({len(val_pred_df):,} 행)")

    # 루트 model/ 디렉토리에 동기화 (script.py 및 제출용)
    if args.sync_root_model:
        root_model_dir = os.path.join(root, "model")
        shutil.copy2(model_txt_path, os.path.join(root_model_dir, "model.txt"))
        shutil.copy2(feat_json_path, os.path.join(root_model_dir, "feature_columns.json"))
        print(f"  ✅ 루트 model/ 디렉토리에 model.txt 및 feature_columns.json 동기화 완료")
    else:
        print("  ℹ️ 활성 제출 모델은 변경하지 않음 (--sync-root-model 미지정)")

    print("\n" + "=" * 70)
    print(f"  🎉 [{args.exp_id}] 학습 및 검증 완료! (로컬 환산 점수: {score_lgbm:.2f}점)")
    print(f"     총 소요 시간: {total_elapsed:.1f}초 | 최대 메모리: {peak_mem:.1f} MB")
    print("=" * 70)


if __name__ == "__main__":
    main()
