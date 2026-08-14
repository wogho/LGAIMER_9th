"""
features.py — 학습 & 추론 공유 피처 엔지니어링 모듈

⚠️ 핵심 규칙:
  - 학습 노트북과 script.py 양쪽에서 동일한 build_features()를 import
  - test 행 간 cross-referencing 절대 금지 (groupby, rolling, freq encoding 등)
  - 2025 시즌 데이터 사용 금지
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
# 3. 피처 엔지니어링 (학습 == 추론 동일)
# ──────────────────────────────────────────────
ASOF_COLS = [c for c in DTYPE_MAP if c.startswith("asof_")]


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    모델 입력 피처 생성.

    ⚠️ 이 함수는 train/test 모두에서 동일하게 호출되어야 합니다.
    추가 피처를 만들 때 반드시 여기에 추가하세요.
    """
    feat = df.drop(columns=[ID_COL] + ([TARGET_COL] if TARGET_COL in df.columns else []))

    # --- (a) asof_* 결측 플래그 ---
    for col in ASOF_COLS:
        if col in feat.columns:
            feat[f"{col}_isna"] = feat[col].isna().astype("int8")

    # --- (b) 파생 피처 예시 (필요 시 활성화) ---
    # feat["count_state"] = feat["balls_before"] * 10 + feat["strikes_before"]
    # feat["pitcher_batter_same_hand"] = (feat["pitcher_hand"] == feat["batter_hand"]).astype("int8")

    return feat


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
