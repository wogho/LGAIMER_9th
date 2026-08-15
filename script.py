# script.py
import os

import joblib
import numpy as np
import pandas as pd

ID_COL = "row_id"
TARGET_COL = "control_success"


def load_test(path):
    """평가 데이터 로드 및 추론 입력 계약 검증."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    if ID_COL not in df.columns:
        raise ValueError(f"test 데이터에 {ID_COL} 컬럼이 없음: {list(df.columns)[:5]}")
    if TARGET_COL in df.columns:
        raise ValueError(f"test 데이터에 정답 컬럼 {TARGET_COL}이 포함되어 있음")
    if df[ID_COL].isna().any():
        raise ValueError("test 데이터의 row_id에 결측값이 있음")
    if df[ID_COL].duplicated().any():
        raise ValueError("test 데이터의 row_id에 중복값이 있음")
    return df


def load_sample_submission(path):
    """sample_submission.csv 로드 및 출력 계약 검증."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    expected = [ID_COL, TARGET_COL]
    if list(df.columns) != expected:
        raise ValueError(f"sample_submission 컬럼 {list(df.columns)} != {expected}")
    if df[ID_COL].isna().any():
        raise ValueError("sample_submission의 row_id에 결측값이 있음")
    if df[ID_COL].duplicated().any():
        raise ValueError("sample_submission의 row_id에 중복값이 있음")
    return df


def build_features(df):
    """학습과 동일한 행별 피처 추출.

    현재 베이스라인은 row_id만 제외한다. 이 함수에는 현재 행의 값만
    사용하는 파생 피처만 추가할 수 있으며, test 배치 통계는 금지한다.
    """
    return df.drop(columns=[ID_COL])


def merge_predictions(sub, ids, preds):
    """예측을 제출 순서에 맞추되 누락·중복·비정상 값을 즉시 차단."""
    ids = pd.Series(ids, name=ID_COL)
    predictions = np.asarray(preds, dtype=np.float64)

    if len(ids) != len(predictions):
        raise ValueError(
            f"ID 수와 예측 수 불일치: ids={len(ids)}, preds={len(predictions)}"
        )
    if ids.isna().any() or ids.duplicated().any():
        raise ValueError("예측 ID에 결측 또는 중복값이 있음")
    if sub[ID_COL].isna().any() or sub[ID_COL].duplicated().any():
        raise ValueError("제출 ID에 결측 또는 중복값이 있음")
    if set(ids.tolist()) != set(sub[ID_COL].tolist()):
        missing = set(sub[ID_COL]) - set(ids)
        extra = set(ids) - set(sub[ID_COL])
        raise ValueError(
            f"예측 ID와 제출 ID 불일치: missing={len(missing)}, extra={len(extra)}"
        )
    if not np.isfinite(predictions).all():
        raise ValueError("예측값에 NaN 또는 Inf가 있음")
    if ((predictions < 0.0) | (predictions > 1.0)).any():
        raise ValueError("예측 확률이 [0, 1] 범위를 벗어남")

    pred_by_id = pd.Series(predictions, index=ids.to_numpy())
    result = sub.copy()
    result[TARGET_COL] = result[ID_COL].map(pred_by_id)
    if result[TARGET_COL].isna().any():
        raise ValueError("제출 순서 정렬 후 누락된 예측이 있음")
    return result


def save_submission(path, sub):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sub.to_csv(path, index=False, encoding="utf-8")


def main():
    test_path = "./data/test.csv"
    sample_sub_path = "./data/sample_submission.csv"
    model_path = "./model/rf.pkl"
    output_path = "./output/submission.csv"

    print("Load model...")
    model = joblib.load(model_path)
    print(f" OK. n_features={getattr(model, 'n_features_in_', '?')}")

    print("Load test data...")
    test = load_test(test_path)
    sub = load_sample_submission(sample_sub_path)
    if len(test) != len(sub):
        raise ValueError(f"test/submission 행 수 불일치: {len(test)} != {len(sub)}")
    print(f" test={len(test)}  submission={len(sub)}")

    print("Build features...")
    ids = test[ID_COL].copy()
    features = build_features(test)
    print(f" features={features.shape[1]}")

    print("Inference model...")
    predictions = (
        model.predict_proba(features)[:, 1]
        if len(features)
        else np.array([], dtype=np.float64)
    )
    print(f" preds={len(predictions)}")

    print("Build submission...")
    submission = merge_predictions(sub, ids, predictions)
    save_submission(output_path, submission)
    print(f"✅ Saved: {output_path} (rows={len(submission)})")


if __name__ == "__main__":
    main()
