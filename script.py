import hashlib
import json
import os

import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"
CATEGORICAL_COLS = [
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "count_state",
    "matchup_platoon",
]
COUNT_STATE_CATEGORIES = ["ahead_pitcher", "ahead_batter", "neutral", "full_count"]
MATCHUP_CATEGORIES = ["1_1", "1_2", "2_1", "2_2", "unknown"]
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


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_contract(path):
    with open(path, "r", encoding="utf-8") as file:
        contract = json.load(file)
    if contract.get("schema_version") != 1:
        raise ValueError("지원하지 않는 앙상블 계약 버전입니다")
    if contract.get("feature_count") != 60 or contract.get("future_season") != 2025:
        raise ValueError("FE-001 또는 미래 시즌 계약이 다릅니다")
    expected_rule = {
        "F": {"lightgbm": 0.0, "catboost": 1.0},
        "R": {"lightgbm": 0.5, "catboost": 0.5},
    }
    if contract.get("selection_rule") != expected_rule:
        raise ValueError("고정 선택형 앙상블 규칙이 다릅니다")
    if contract.get("tree_counts") != {"lightgbm": 100, "catboost": 259}:
        raise ValueError("두 모델의 tree 수 계약이 다릅니다")
    if contract.get("active_model_sync") is not False:
        raise ValueError("격리 후보 계약이 아닙니다")
    return contract


def load_feature_columns(path, expected_count):
    with open(path, "r", encoding="utf-8") as file:
        columns = json.load(file)
    if not isinstance(columns, list) or len(columns) != expected_count:
        raise ValueError("feature_columns.json 개수가 계약과 다릅니다")
    if len(columns) != len(set(columns)):
        raise ValueError("feature_columns.json에 중복 피처가 있습니다")
    if ID_COL in columns or TARGET_COL in columns:
        raise ValueError("피처 계약에 ID 또는 정답 컬럼이 포함되어 있습니다")
    return columns


def load_inputs(test_path, sample_path):
    test = pd.read_csv(test_path, encoding="utf-8-sig", dtype=DTYPE_MAP)
    sample = pd.read_csv(sample_path, encoding="utf-8-sig")
    if ID_COL not in test.columns or TARGET_COL in test.columns:
        raise ValueError("test 입력 컬럼 계약이 다릅니다")
    if list(sample.columns) != [ID_COL, TARGET_COL]:
        raise ValueError("sample_submission 컬럼 계약이 다릅니다")
    if test[ID_COL].isna().any() or test[ID_COL].duplicated().any():
        raise ValueError("test row_id에 결측 또는 중복이 있습니다")
    if sample[ID_COL].isna().any() or sample[ID_COL].duplicated().any():
        raise ValueError("sample_submission row_id에 결측 또는 중복이 있습니다")
    if set(test[ID_COL].tolist()) != set(sample[ID_COL].tolist()):
        raise ValueError("test와 sample_submission row_id 집합이 다릅니다")
    if not test["season"].eq(2025).all():
        raise ValueError("test에 2025 이외 시즌이 있습니다")
    return test, sample


def build_features(frame, feature_columns):
    features = frame.drop(columns=[ID_COL]).copy()
    if TARGET_COL in features.columns:
        features = features.drop(columns=[TARGET_COL])
    expected_raw = [
        column for column in feature_columns if column not in DERIVED_FEATURE_COLUMNS
    ]
    actual_raw = list(features.columns)
    missing_raw = [column for column in expected_raw if column not in actual_raw]
    unexpected_raw = [column for column in actual_raw if column not in expected_raw]
    if missing_raw or unexpected_raw:
        raise ValueError(
            f"추론 원본 피처 계약 불일치: missing={missing_raw}, unexpected={unexpected_raw}"
        )

    balls = features["balls_before"]
    strikes = features["strikes_before"]
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
    features["count_state"] = pd.Categorical(
        count_state, categories=COUNT_STATE_CATEGORIES
    )
    features["scoring_pos_runner"] = (
        features["runner_on_2b"].eq(1) | features["runner_on_3b"].eq(1)
    ).astype("int8")

    pitcher_hand = features["pitcher_hand"].astype("string")
    batter_hand = features["batter_hand"].astype("string")
    matchup = (pitcher_hand + "_" + batter_hand).fillna("unknown")
    matchup = matchup.where(matchup.isin(MATCHUP_CATEGORIES), "unknown")
    features["matchup_platoon"] = pd.Categorical(
        matchup, categories=MATCHUP_CATEGORIES
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
    missing = [column for column in feature_columns if column not in features.columns]
    if missing:
        raise ValueError(f"파생 피처 계약 불일치: missing={missing}")
    return features.loc[:, feature_columns].copy()


def apply_selective_rule(game_type, pred_lightgbm, pred_catboost):
    game_type = pd.Series(game_type, copy=False).astype("string").to_numpy(dtype=str)
    pred_lightgbm = np.asarray(pred_lightgbm, dtype=np.float64)
    pred_catboost = np.asarray(pred_catboost, dtype=np.float64)
    if not (len(game_type) == len(pred_lightgbm) == len(pred_catboost)):
        raise ValueError("선택 규칙 입력 행 수가 다릅니다")
    unexpected = sorted(set(game_type) - {"F", "R"})
    if unexpected:
        raise ValueError(f"지원하지 않는 game_type입니다: {unexpected}")
    if not np.isfinite(pred_lightgbm).all() or not np.isfinite(pred_catboost).all():
        raise ValueError("단일 모델 예측에 NaN 또는 Inf가 있습니다")
    prediction = np.where(
        game_type == "F",
        pred_catboost,
        0.5 * pred_lightgbm + 0.5 * pred_catboost,
    )
    if not np.isfinite(prediction).all() or not (
        (prediction >= 0.0) & (prediction <= 1.0)
    ).all():
        raise ValueError("선택형 예측이 확률 범위를 벗어났습니다")
    return prediction


def merge_submission(sample, ids, prediction):
    prediction_by_id = pd.Series(
        np.asarray(prediction, dtype=np.float64),
        index=pd.Series(ids).to_numpy(),
    )
    output = sample[[ID_COL]].copy()
    output[TARGET_COL] = output[ID_COL].map(prediction_by_id)
    if output[TARGET_COL].isna().any():
        raise ValueError("제출 순서 결합 후 누락 예측이 있습니다")
    if not np.isfinite(output[TARGET_COL]).all():
        raise ValueError("제출 예측에 NaN 또는 Inf가 있습니다")
    if not output[TARGET_COL].between(0.0, 1.0).all():
        raise ValueError("제출 예측이 확률 범위를 벗어났습니다")
    return output


def main():
    test_path = "./data/test.csv"
    sample_path = "./data/sample_submission.csv"
    output_path = "./output/submission.csv"
    contract_path = "./model/ensemble_contract.json"
    contract = load_contract(contract_path)
    model_files = contract["model_files"]
    model_paths = {
        "lightgbm": os.path.join("./model", model_files["lightgbm"]),
        "catboost": os.path.join("./model", model_files["catboost"]),
        "feature_columns": os.path.join("./model", model_files["feature_columns"]),
    }
    for key, path in model_paths.items():
        if not os.path.isfile(path):
            raise FileNotFoundError(f"필수 {key} 파일이 없습니다: {path}")
        if sha256_file(path) != contract["model_sha256"][key]:
            raise ValueError(f"{key} 파일 해시가 계약과 다릅니다")

    feature_columns = load_feature_columns(
        model_paths["feature_columns"], contract["feature_count"]
    )
    test, sample = load_inputs(test_path, sample_path)
    ids = test[ID_COL].copy()
    game_type = test["game_type"].copy()
    base_features = build_features(test, feature_columns)

    import lightgbm as lgb
    import catboost as cb

    lightgbm_model = lgb.Booster(model_file=model_paths["lightgbm"])
    if lightgbm_model.num_trees() != contract["tree_counts"]["lightgbm"]:
        raise ValueError("LightGBM tree 수가 계약과 다릅니다")
    if lightgbm_model.feature_name() != feature_columns:
        raise ValueError("LightGBM 피처 이름·순서가 계약과 다릅니다")
    lightgbm_features = base_features.copy()
    for column in CATEGORICAL_COLS:
        lightgbm_features[column] = lightgbm_features[column].astype("category")
    pred_lightgbm = lightgbm_model.predict(
        lightgbm_features,
        num_iteration=contract["tree_counts"]["lightgbm"],
    )

    catboost_model = cb.CatBoostClassifier()
    catboost_model.load_model(model_paths["catboost"], format="cbm")
    if catboost_model.tree_count_ != contract["tree_counts"]["catboost"]:
        raise ValueError("CatBoost tree 수가 계약과 다릅니다")
    if catboost_model.feature_names_ != feature_columns:
        raise ValueError("CatBoost 피처 이름·순서가 계약과 다릅니다")
    catboost_features = base_features.copy()
    for column in CATEGORICAL_COLS:
        catboost_features[column] = (
            catboost_features[column].astype("string").fillna("<NA>").astype(str)
        )
    catboost_pool = cb.Pool(
        catboost_features,
        cat_features=CATEGORICAL_COLS,
        feature_names=feature_columns,
    )
    pred_catboost = catboost_model.predict_proba(catboost_pool)[:, 1]
    prediction = apply_selective_rule(game_type, pred_lightgbm, pred_catboost)
    submission = merge_submission(sample, ids, prediction)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved {output_path}: rows={len(submission)}")


if __name__ == "__main__":
    main()
