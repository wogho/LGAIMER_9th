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

COUNT_STATE_CATEGORIES = ["ahead_pitcher", "ahead_batter", "neutral", "full_count"]
MATCHUP_CATEGORIES = ["1_1", "1_2", "2_1", "2_2", "unknown"]
EXACT_COUNT_CATEGORIES = [
    "0-0", "0-1", "0-2", "1-0", "1-1", "1-2",
    "2-0", "2-1", "2-2", "3-0", "3-1", "3-2",
]
SCORE_MARGIN_CATEGORIES = [
    "behind_4+", "behind_1_3", "tied", "ahead_1_3", "ahead_4+"
]
ASOF_MISSING_COLUMNS = [
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]
DERIVED_FEATURE_COLUMNS = [
    "count_state",
    "scoring_pos_runner",
    "matchup_platoon",
    "high_leverage_pressure",
    "late_inning_flag",
    "pitcher_prev1_success_delta",
    "pitcher_prev3_success_delta",
    "control_to_middle_ratio",
    "pitchmix_fastball_bias",
    "batter_control_pressure",
    "asof_missing_count",
    "asof_any_missing",
    "asof_cold_start_flag",
]
FE002_GROUP_COLUMNS = {
    "state": [
        "exact_count",
        "score_margin_state",
        "close_game_flag",
        "leverage_log1p",
        "late_high_leverage",
    ],
    "form": [
        "pitcher_recent_slope_1_3",
        "pitcher_recent_slope_3_5",
        "pitcher_recent_range",
        "pitcher_batter_success_gap",
    ],
    "support": [
        "asof_pitcher_log_n",
        "asof_batter_log_n",
        "asof_pitchmix_log_n",
        "pitchmix_concentration",
        "pitchmix_entropy",
    ],
}

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


def derive_count_state(df: pd.DataFrame) -> pd.Categorical:
    """Derive the shared pre-pitch count-state categories from current-row data."""

    required = {"balls_before", "strikes_before"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"count_state 생성 필수 컬럼이 없습니다: {sorted(missing)}")
    balls = df["balls_before"]
    strikes = df["strikes_before"]
    full_count = balls.eq(3) & strikes.eq(2)
    pitcher_ahead = (
        (balls.eq(0) & strikes.isin([1, 2]))
        | (balls.eq(1) & strikes.eq(2))
    )
    batter_ahead = (
        (balls.eq(2) & strikes.eq(0))
        | (balls.eq(3) & strikes.isin([0, 1]))
    )
    count_state = np.select(
        [full_count, pitcher_ahead, batter_ahead],
        ["full_count", "ahead_pitcher", "ahead_batter"],
        default="neutral",
    )
    return pd.Categorical(count_state, categories=COUNT_STATE_CATEGORIES)


def derive_scoring_pos_runner(df: pd.DataFrame) -> pd.Series:
    """Return whether second or third base is occupied using current-row data."""

    required = {"runner_on_2b", "runner_on_3b"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"scoring_pos_runner 생성 필수 컬럼이 없습니다: {sorted(missing)}"
        )
    return (df["runner_on_2b"].eq(1) | df["runner_on_3b"].eq(1)).astype("int8")


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
    """기본 컬럼과 현재 행만 사용하는 FE-001 파생 피처를 생성한다."""
    drop_columns = [ID_COL]
    if TARGET_COL in df.columns:
        drop_columns.append(TARGET_COL)
    features = df.drop(columns=drop_columns).copy()

    features["count_state"] = derive_count_state(features)
    features["scoring_pos_runner"] = derive_scoring_pos_runner(features)

    pitcher_hand = features["pitcher_hand"].astype("string")
    batter_hand = features["batter_hand"].astype("string")
    matchup = (pitcher_hand + "_" + batter_hand).fillna("unknown")
    matchup = matchup.where(matchup.isin(MATCHUP_CATEGORIES), "unknown")
    features["matchup_platoon"] = pd.Categorical(
        matchup,
        categories=MATCHUP_CATEGORIES,
    )

    features["high_leverage_pressure"] = (
        features["li"].astype("float32")
        * features["score_diff_pitcher_team"].abs().astype("float32")
    ).astype("float32")
    features["late_inning_flag"] = features["inning"].ge(7).astype("int8")
    features["pitcher_prev1_success_delta"] = (
        features["asof_pitcher_prev1_game_success_rate"]
        - features["asof_pitcher_success_rate"]
    ).astype("float32")
    features["pitcher_prev3_success_delta"] = (
        features["asof_pitcher_prev3_game_success_rate"]
        - features["asof_pitcher_success_rate"]
    ).astype("float32")
    features["control_to_middle_ratio"] = (
        features["asof_pitcher_success_rate"]
        / (features["asof_pitcher_middle_rate"] + np.float32(1e-5))
    ).astype("float32")
    features["pitchmix_fastball_bias"] = (
        features["asof_pitcher_fastball_rate"]
        - features["asof_pitcher_breaking_rate"]
        - features["asof_pitcher_offspeed_rate"]
    ).astype("float32")
    features["batter_control_pressure"] = (
        features["asof_batter_middle_rate"]
        / (features["asof_batter_success_rate"] + np.float32(1e-5))
    ).astype("float32")

    missing_count = features[ASOF_MISSING_COLUMNS].isna().sum(axis=1)
    features["asof_missing_count"] = missing_count.astype("int8")
    features["asof_any_missing"] = missing_count.gt(0).astype("int8")
    features["asof_cold_start_flag"] = (
        features["asof_pitcher_n"].le(0)
        | features["asof_batter_n"].le(0)
        | features["asof_pitcher_pitchmix_n"].le(0)
    ).astype("int8")

    return features


def build_features_fe002(
    df: pd.DataFrame,
    groups: list[str] | tuple[str, ...] = ("state", "form", "support"),
) -> pd.DataFrame:
    """Add current-row-only FE-002 candidates without changing the FE-001 contract."""

    groups = list(groups)
    if not groups or len(groups) != len(set(groups)):
        raise ValueError("FE-002 그룹이 비어 있거나 중복되었습니다")
    unexpected = sorted(set(groups) - set(FE002_GROUP_COLUMNS))
    if unexpected:
        raise ValueError(f"지원하지 않는 FE-002 그룹입니다: {unexpected}")

    features = build_features(df)
    score = features["score_diff_pitcher_team"]

    if "state" in groups:
        exact_count = (
            features["balls_before"].astype("string")
            + "-"
            + features["strikes_before"].astype("string")
        )
        features["exact_count"] = pd.Categorical(
            exact_count,
            categories=EXACT_COUNT_CATEGORIES,
        )
        score_margin = np.select(
            [score.le(-4), score.between(-3, -1), score.eq(0), score.between(1, 3)],
            ["behind_4+", "behind_1_3", "tied", "ahead_1_3"],
            default="ahead_4+",
        )
        features["score_margin_state"] = pd.Categorical(
            score_margin,
            categories=SCORE_MARGIN_CATEGORIES,
        )
        features["close_game_flag"] = score.abs().le(1).astype("int8")
        features["leverage_log1p"] = np.log1p(
            features["li"].clip(lower=0).astype("float32")
        ).astype("float32")
        features["late_high_leverage"] = (
            features["li"].astype("float32")
            * features["late_inning_flag"].astype("float32")
        ).astype("float32")

    if "form" in groups:
        prev1 = features["asof_pitcher_prev1_game_success_rate"]
        prev3 = features["asof_pitcher_prev3_game_success_rate"]
        prev5 = features["asof_pitcher_prev5_game_success_rate"]
        features["pitcher_recent_slope_1_3"] = (prev1 - prev3).astype("float32")
        features["pitcher_recent_slope_3_5"] = (prev3 - prev5).astype("float32")
        recent = pd.concat([prev1, prev3, prev5], axis=1)
        features["pitcher_recent_range"] = (
            recent.max(axis=1, skipna=False) - recent.min(axis=1, skipna=False)
        ).astype("float32")
        features["pitcher_batter_success_gap"] = (
            features["asof_pitcher_success_rate"]
            - features["asof_batter_success_rate"]
        ).astype("float32")

    if "support" in groups:
        features["asof_pitcher_log_n"] = np.log1p(
            features["asof_pitcher_n"].clip(lower=0)
        ).astype("float32")
        features["asof_batter_log_n"] = np.log1p(
            features["asof_batter_n"].clip(lower=0)
        ).astype("float32")
        features["asof_pitchmix_log_n"] = np.log1p(
            features["asof_pitcher_pitchmix_n"].clip(lower=0)
        ).astype("float32")
        rate_columns = [
            "asof_pitcher_fastball_rate",
            "asof_pitcher_breaking_rate",
            "asof_pitcher_offspeed_rate",
        ]
        rates = features[rate_columns].astype("float32").clip(lower=0, upper=1)
        any_missing = rates.isna().any(axis=1)
        concentration = rates.pow(2).sum(axis=1, skipna=False)
        safe_rates = rates.clip(lower=np.float32(1e-7))
        entropy = -(rates * np.log(safe_rates)).sum(axis=1, skipna=False)
        features["pitchmix_concentration"] = concentration.mask(any_missing).astype(
            "float32"
        )
        features["pitchmix_entropy"] = entropy.mask(any_missing).astype("float32")

    expected_new = [column for group in groups for column in FE002_GROUP_COLUMNS[group]]
    missing = [column for column in expected_new if column not in features.columns]
    if missing:
        raise AssertionError(f"FE-002 파생 피처가 생성되지 않았습니다: {missing}")
    return features


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
