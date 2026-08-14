"""
config.py — 프로젝트 경로 및 설정 상수 관리
"""
import os

# ──────────────────────────────────────────────
# 경로 설정 (모두 상대 경로 → 평가 서버 호환)
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(PROJECT_ROOT, "data")
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
TRACKMAN_PATH = os.path.join(DATA_DIR, "trackman_history.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
SUBMISSION_PATH = os.path.join(OUTPUT_DIR, "submission.csv")

# ──────────────────────────────────────────────
# 학습 설정
# ──────────────────────────────────────────────
RANDOM_SEED = 42
N_FOLDS = 5

# 시즌 기반 Validation (2024 시즌을 holdout으로 사용)
TRAIN_SEASONS = [2019, 2020, 2021, 2022, 2023]
VALID_SEASON = 2024
TEST_SEASON = 2025

# ──────────────────────────────────────────────
# LightGBM 기본 파라미터
# ──────────────────────────────────────────────
LGBM_DEFAULT_PARAMS = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 100,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "n_estimators": 2000,
    "random_state": RANDOM_SEED,
    "verbose": -1,
    "n_jobs": -1,
}

# ──────────────────────────────────────────────
# 평가 서버 제약 조건
# ──────────────────────────────────────────────
EVAL_SERVER = {
    "os": "Ubuntu 22.04.5 LTS",
    "python": "3.11.15",
    "cpu": "6 vCPU",
    "ram": "28 GB",
    "gpu": "NVIDIA L4 (22.4 GiB VRAM)",
    "cuda": "12.8",
    "timeout_min": 10,
    "test_rows": 245_000,
    "max_zip_gb": 10,
    "submissions_per_day": 5,
}
