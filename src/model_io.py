"""
model_io.py — 모델 저장/로드 유틸리티 (Pickle-free 안전 직렬화)

⚠️ 핵심 원칙:
  - Python 3.12 (ARM64) → Python 3.11 (x86_64) 간 pickle 비호환 위험 회피
  - LightGBM/XGBoost/CatBoost 각각의 네이티브 포맷 사용
  - scikit-learn 객체(Calibrator 등)만 제한적으로 joblib 허용

사용 예:
    from src.model_io import save_model, load_model

    # 저장
    save_model(lgb_model, "model/lgbm_v1.txt", model_type="lightgbm")

    # 로드
    model = load_model("model/lgbm_v1.txt", model_type="lightgbm")
"""

import os
import warnings
import json
import numpy as np


# ──────────────────────────────────────────────
# LightGBM
# ──────────────────────────────────────────────
def save_lightgbm(model, path: str):
    """LightGBM 모델을 네이티브 텍스트 포맷으로 저장."""
    import lightgbm as lgb

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # LGBMClassifier/LGBMRegressor → 내부 Booster 추출
    if hasattr(model, "booster_"):
        booster = model.booster_
    elif isinstance(model, lgb.Booster):
        booster = model
    else:
        raise TypeError(f"지원하지 않는 LightGBM 모델 타입: {type(model)}")

    booster.save_model(path)
    print(f"  ✅ LightGBM 저장: {path} ({os.path.getsize(path) / 1e6:.2f} MB)")


def load_lightgbm(path: str):
    """LightGBM 모델을 네이티브 텍스트 포맷에서 로드."""
    import lightgbm as lgb

    booster = lgb.Booster(model_file=path)
    print(f"  ✅ LightGBM 로드: {path}")
    return booster


# ──────────────────────────────────────────────
# XGBoost
# ──────────────────────────────────────────────
def save_xgboost(model, path: str):
    """XGBoost 모델을 JSON 포맷으로 저장."""
    import xgboost as xgb

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    if hasattr(model, "get_booster"):
        booster = model.get_booster()
    elif isinstance(model, xgb.Booster):
        booster = model
    else:
        raise TypeError(f"지원하지 않는 XGBoost 모델 타입: {type(model)}")

    booster.save_model(path)
    print(f"  ✅ XGBoost 저장: {path} ({os.path.getsize(path) / 1e6:.2f} MB)")


def load_xgboost(path: str):
    """XGBoost 모델을 JSON 포맷에서 로드."""
    import xgboost as xgb

    booster = xgb.Booster()
    booster.load_model(path)
    print(f"  ✅ XGBoost 로드: {path}")
    return booster


# ──────────────────────────────────────────────
# CatBoost
# ──────────────────────────────────────────────
def save_catboost(model, path: str):
    """CatBoost 모델을 네이티브 바이너리(.cbm)로 저장."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    model.save_model(path)
    print(f"  ✅ CatBoost 저장: {path} ({os.path.getsize(path) / 1e6:.2f} MB)")


def load_catboost(path: str, task: str = "classification"):
    """CatBoost 모델을 네이티브 바이너리에서 로드."""
    if task == "classification":
        from catboost import CatBoostClassifier as Cls
    else:
        from catboost import CatBoostRegressor as Cls

    model = Cls()
    model.load_model(path)
    print(f"  ✅ CatBoost 로드: {path}")
    return model


# ──────────────────────────────────────────────
# Isotonic Calibrator (sklearn — joblib 허용)
# ──────────────────────────────────────────────
def save_calibrator(calibrator, path: str):
    """
    sklearn IsotonicRegression 보정기 저장.

    ⚠️ joblib 직렬화를 사용하므로 Python 메이저 버전 차이 시 주의.
    가능하면 보정 테이블(X_, y_)을 별도 CSV/JSON으로 저장하는 것을 권장.
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    # 안전 모드: 보정 테이블을 JSON으로 저장
    if path.endswith(".json"):
        data = {
            "X_thresholds": calibrator.X_thresholds_.tolist(),
            "y_thresholds": calibrator.y_thresholds_.tolist(),
            "X_min": float(calibrator.X_min_),
            "X_max": float(calibrator.X_max_),
            "increasing": bool(calibrator.increasing_),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  ✅ Calibrator 저장 (JSON): {path}")
    else:
        import joblib
        joblib.dump(calibrator, path)
        warnings.warn(
            "joblib 포맷은 Python 버전 간 호환이 불안정합니다. "
            ".json 확장자로 저장하는 것을 권장합니다.",
            UserWarning,
        )
        print(f"  ⚠️ Calibrator 저장 (joblib): {path}")


def load_calibrator(path: str):
    """sklearn IsotonicRegression 보정기 로드."""
    if path.endswith(".json"):
        from sklearn.isotonic import IsotonicRegression

        with open(path, "r") as f:
            data = json.load(f)

        cal = IsotonicRegression(out_of_bounds="clip")
        cal.X_thresholds_ = np.array(data["X_thresholds"])
        cal.y_thresholds_ = np.array(data["y_thresholds"])
        cal.X_min_ = data["X_min"]
        cal.X_max_ = data["X_max"]
        cal.increasing_ = data["increasing"]
        cal.f_ = None  # predict 시 X_thresholds_/y_thresholds_ 사용
        print(f"  ✅ Calibrator 로드 (JSON): {path}")
        return cal
    else:
        import joblib
        cal = joblib.load(path)
        print(f"  ✅ Calibrator 로드 (joblib): {path}")
        return cal


# ──────────────────────────────────────────────
# 통합 인터페이스
# ──────────────────────────────────────────────
def save_model(model, path: str, model_type: str = "lightgbm"):
    """
    모델 저장 통합 인터페이스.

    model_type: "lightgbm" | "xgboost" | "catboost" | "calibrator"
    """
    dispatch = {
        "lightgbm": save_lightgbm,
        "xgboost": save_xgboost,
        "catboost": save_catboost,
        "calibrator": save_calibrator,
    }
    fn = dispatch.get(model_type)
    if fn is None:
        raise ValueError(f"지원하지 않는 model_type: {model_type}. "
                         f"가능: {list(dispatch.keys())}")
    fn(model, path)


def load_model(path: str, model_type: str = "lightgbm", **kwargs):
    """
    모델 로드 통합 인터페이스.

    model_type: "lightgbm" | "xgboost" | "catboost" | "calibrator"
    """
    dispatch = {
        "lightgbm": load_lightgbm,
        "xgboost": load_xgboost,
        "catboost": lambda p: load_catboost(p, **kwargs),
        "calibrator": load_calibrator,
    }
    fn = dispatch.get(model_type)
    if fn is None:
        raise ValueError(f"지원하지 않는 model_type: {model_type}. "
                         f"가능: {list(dispatch.keys())}")
    return fn(path)
