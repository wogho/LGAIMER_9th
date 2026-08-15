"""
features.py — 학습 & 추론 공유 피처 엔지니어링 모듈

⚠️ 핵심 규칙:
  - 학습용 build_features()와 제출 script.py의 동명 함수를 항상 동기화
  - test 행 간 cross-referencing 절대 금지 (groupby, rolling, freq encoding 등)
  - 추론 중 test 기반 fit/집계/보정 금지
"""

import pandas as pd
import numpy as np

ID_COL = "row_id"
TARGET_COL = "control_success"

# ──────────────────────────────────────────────
# 1. 메모리 최적화 dtype 매핑
# ──────────────────────────────────────────────
DTYPE_MAP = {
    "season": "int16",
    "game_month": "int8",
    "game_dayofweek": "int8",
    "inning": "int8",
    "top_bottom": "category",
    "game_type": "category",
    "balls_before": "int8",
    "strikes_before": "int8",
    "outs_before": "int8",
    "run_top_before": "int16",
    "run_bot_before": "int16",
    "run_total_before": "int16",
    "score_diff_home": "int16",
    "score_diff_pitcher_team": "int16",
    "runner_on_1b": "int8",
    "runner_on_2b": "int8",
    "runner_on_3b": "int8",
    "num_runners_on": "int8",
    "base_state": "category",
    "home_win_expectancy": "float32",
    "away_win_expectancy": "float32",
    "li": "float32",
    "pitcher_id": "int32",
    "batter_id": "int32",
    "pitcher_hand": "category",
    "batter_hand": "category",
    "pitcher_team_id": "int16",
    "batter_team_id": "int16",
    "asof_pitcher_n": "float32",
    "asof_pitcher_success_rate": "float32",
    "asof_pitcher_reverse_rate": "float32",
    "asof_pitcher_middle_rate": "float32",
    "asof_pitcher_ball_rate": "float32",
    "asof_pitcher_strike_rate": "float32",
    "asof_pitcher_prev1_game_success_rate": "float32",
    "asof_pitcher_prev3_game_success_rate": "float32",
    "asof_pitcher_prev5_game_success_rate": "float32",
    "asof_pitcher_prev1_game_middle_rate": "float32",
    "asof_pitcher_prev3_game_middle_rate": "float32",
    "asof_pitcher_prev5_game_middle_rate": "float32",
    "asof_batter_n": "float32",
    "asof_batter_success_rate": "float32",
    "asof_batter_middle_rate": "float32",
    "asof_pitcher_pitchmix_n": "float32",
    "asof_pitcher_fastball_rate": "float32",
    "asof_pitcher_breaking_rate": "float32",
    "asof_pitcher_offspeed_rate": "float32",
}


# ──────────────────────────────────────────────
# 2. 데이터 로딩
# ──────────────────────────────────────────────
def load_data(path: str, is_train: bool = True) -> pd.DataFrame:
    """메모리 효율적 데이터 로딩."""
    dtype = {k: v for k, v in DTYPE_MAP.items()}
    if is_train:
        dtype[TARGET_COL] = "int8"

    df = pd.read_csv(path, encoding="utf-8-sig", dtype=dtype)
    print(f"  Loaded {path}: {df.shape[0]:,} rows × {df.shape[1]} cols | "
          f"mem={df.memory_usage(deep=True).sum() / 1e6:.1f} MB")
    return df


# ──────────────────────────────────────────────
# 3. 피처 엔지니어링 (현재 베이스라인 계약)
# ──────────────────────────────────────────────
def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """현재 베이스라인의 모델 입력 피처 생성.

    현재는 식별자와 정답만 제외한다. 파생 피처를 추가할 때는 현재 행의
    값만 사용하고 script.py의 build_features()도 함께 변경한 뒤
    scripts/verify_features.py를 통과해야 한다.
    """
    drop_columns = [ID_COL]
    if TARGET_COL in df.columns:
        drop_columns.append(TARGET_COL)
    return df.drop(columns=drop_columns)


# ──────────────────────────────────────────────
# 4. 평가 메트릭
# ──────────────────────────────────────────────
def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Brier Score."""
    return float(np.mean((y_true - y_pred) ** 2))


def brier_skill_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    BSS (Brier Skill Score).
    baseline = mean(y_true), 0 이면 baseline 수준, 양수일수록 좋음.
    """
    bs = brier_score(y_true, y_pred)
    r = np.mean(y_true)
    bs_ref = r * (1.0 - r)
    return 1.0 - (bs / bs_ref) if bs_ref > 0 else 0.0


def hackathon_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    해커톤 공식 점수: max(0, 100000 * BSS).
    """
    bss = brier_skill_score(y_true, y_pred)
    return max(0.0, 100_000 * bss)
