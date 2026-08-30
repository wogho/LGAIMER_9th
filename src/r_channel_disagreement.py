"""Row-local REF4 channel-disagreement features; no batch statistics."""
from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_NAMES = (
    "main_delta_v2",
    "main_delta_v355",
    "main_delta_v330",
    "risk_middle_centered",
    "risk_wild_centered",
    "risk_reverse_centered",
    "main_std",
    "main_range",
    "base_margin",
    "base_confidence",
)


def build_channel_features(predictions, risks, base_prediction) -> pd.DataFrame:
    main = np.column_stack([np.asarray(x, dtype=float) for x in predictions])
    risk = np.column_stack([np.asarray(x, dtype=float) for x in risks])
    base = np.asarray(base_prediction, dtype=float)
    if main.shape[1] != 3 or risk.shape[1] != 3 or len(base) != len(main):
        raise ValueError("expected three main and three risk channels with aligned rows")
    main_mean = main.mean(axis=1)
    risk_mean = risk.mean(axis=1)
    values = np.column_stack([
        main[:, 0] - main_mean,
        main[:, 1] - main_mean,
        main[:, 2] - main_mean,
        risk[:, 0] - risk_mean,
        risk[:, 1] - risk_mean,
        risk[:, 2] - risk_mean,
        main.std(axis=1),
        main.max(axis=1) - main.min(axis=1),
        base - 0.5,
        np.abs(base - 0.5),
    ])
    return pd.DataFrame(values, columns=FEATURE_NAMES)


def build_oof_channel_features(frame: pd.DataFrame) -> pd.DataFrame:
    return build_channel_features(
        [frame["p_v2_global"], frame["p_v3_55_global"], frame["p_v3_30_global"]],
        [frame["risk_middle_global"], frame["risk_wild_global"], frame["risk_reverse_global"]],
        frame["prediction"],
    )
