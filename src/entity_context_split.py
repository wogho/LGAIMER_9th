"""Train-only entity-by-context partial-pooling residual features."""
from __future__ import annotations

import numpy as np
import pandas as pd


SHRINKAGES = (50.0, 200.0, 800.0, 3200.0)


def add_context_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    balls = pd.to_numeric(frame["balls_before"], errors="coerce").fillna(0).astype(int)
    strikes = pd.to_numeric(frame["strikes_before"], errors="coerce").fillna(0).astype(int)
    inning = pd.to_numeric(frame["inning"], errors="coerce").fillna(1).clip(1, 10).astype(int)
    out["pitcher_id"] = frame["pitcher_id"].astype(str)
    out["batter_id"] = frame["batter_id"].astype(str)
    out["pitcher_opponent_hand"] = frame["batter_hand"].astype(str)
    out["batter_opponent_hand"] = frame["pitcher_hand"].astype(str)
    out["count_state"] = balls.astype(str) + "-" + strikes.astype(str)
    out["count_ahead"] = np.where(strikes > balls, "ahead", "not_ahead")
    out["inning_bucket"] = pd.cut(
        inning, bins=[0, 3, 6, 10], labels=["early", "middle", "late"]
    ).astype(str)
    out["base_state"] = frame["base_state"].fillna("___").astype(str)
    return out


def feature_specs() -> list[tuple[str, str]]:
    contexts = ("opponent_hand", "count_state", "count_ahead", "inning_bucket", "base_state")
    return [(entity, context) for entity in ("pitcher_id", "batter_id") for context in contexts]


def build_split_features(
    history: pd.DataFrame,
    rows: pd.DataFrame,
    shrinkages: tuple[float, ...] = SHRINKAGES,
) -> pd.DataFrame:
    """Build features using only labelled history and row-local lookup keys."""
    hkeys = add_context_keys(history)
    rkeys = add_context_keys(rows)
    season_mean = history.groupby("season", observed=True)["control_success"].transform("mean")
    hkeys["relative_target"] = history["control_success"].to_numpy(float) - season_mean.to_numpy(float)
    out: dict[str, np.ndarray] = {}
    for entity, context in feature_specs():
        context_key = f"{entity.split('_')[0]}_opponent_hand" if context == "opponent_hand" else context
        cell = hkeys.groupby([entity, context_key], observed=True)["relative_target"].agg(["sum", "size"])
        own = hkeys.groupby(entity, observed=True)["relative_target"].agg(["sum", "size"])
        cell = cell.join(own, rsuffix="_entity")
        cell["effect"] = cell["sum"] / cell["size"] - cell["sum_entity"] / cell["size_entity"]
        lookup = pd.MultiIndex.from_arrays([rkeys[entity], rkeys[context_key]])
        effect = np.nan_to_num(lookup.map(cell["effect"]).to_numpy(float), nan=0.0)
        seen = np.nan_to_num(lookup.map(cell["size"]).to_numpy(float), nan=0.0)
        for shrinkage in shrinkages:
            name = f"split_{entity}_{context}_k{int(shrinkage)}"
            out[name] = effect * seen / (seen + shrinkage)
    return pd.DataFrame(out, index=rows.index)


def build_split_profile(history: pd.DataFrame) -> pd.DataFrame:
    """Serialize all train-only cell effects as a portable long table."""
    keys = add_context_keys(history)
    season_mean = history.groupby("season", observed=True)["control_success"].transform("mean")
    keys["relative_target"] = history["control_success"].to_numpy(float) - season_mean.to_numpy(float)
    parts = []
    for entity, context in feature_specs():
        context_key = f"{entity.split('_')[0]}_opponent_hand" if context == "opponent_hand" else context
        cell = keys.groupby([entity, context_key], observed=True)["relative_target"].agg(["sum", "size"])
        own = keys.groupby(entity, observed=True)["relative_target"].agg(["sum", "size"])
        cell = cell.join(own, rsuffix="_entity").reset_index()
        cell["effect"] = cell["sum"] / cell["size"] - cell["sum_entity"] / cell["size_entity"]
        parts.append(pd.DataFrame({
            "entity_name": entity,
            "context_name": context,
            "entity_value": cell[entity].astype(str),
            "context_value": cell[context_key].astype(str),
            "effect": cell["effect"].to_numpy(float),
            "count": cell["size"].to_numpy(int),
        }))
    return pd.concat(parts, ignore_index=True)


def apply_split_profile(
    rows: pd.DataFrame,
    profile: pd.DataFrame,
    shrinkages: tuple[float, ...] = SHRINKAGES,
) -> pd.DataFrame:
    """Apply a saved profile using only the current rows' lookup keys."""
    keys = add_context_keys(rows)
    out: dict[str, np.ndarray] = {}
    for entity, context in feature_specs():
        context_key = f"{entity.split('_')[0]}_opponent_hand" if context == "opponent_hand" else context
        table = profile.loc[
            profile.entity_name.eq(entity) & profile.context_name.eq(context),
            ["entity_value", "context_value", "effect", "count"],
        ].copy()
        table["entity_value"] = table.entity_value.astype(str)
        table["context_value"] = table.context_value.astype(str)
        table = table.set_index(["entity_value", "context_value"])
        lookup = pd.MultiIndex.from_arrays([keys[entity].astype(str), keys[context_key].astype(str)])
        effect = np.nan_to_num(lookup.map(table.effect).to_numpy(float), nan=0.0)
        count = np.nan_to_num(lookup.map(table["count"]).to_numpy(float), nan=0.0)
        for shrinkage in shrinkages:
            out[f"split_{entity}_{context}_k{int(shrinkage)}"] = effect * count / (count + shrinkage)
    return pd.DataFrame(out, index=rows.index)


def apply_linear_split(features: pd.DataFrame, meta_path) -> np.ndarray:
    asset = np.load(meta_path, allow_pickle=False)
    names = [str(value) for value in asset["feature_names"]]
    values = features.reindex(columns=names).to_numpy(float)
    standardized = np.nan_to_num((values - asset["mean"]) / asset["std"])
    return standardized @ asset["coef"]


def compound_feature_specs() -> list[tuple[str, str, str]]:
    return [
        ("pitcher_id", "opponent_hand", "count_state"),
        ("pitcher_id", "opponent_hand", "base_state"),
        ("pitcher_id", "count_state", "inning_bucket"),
        ("pitcher_id", "count_state", "base_state"),
        ("batter_id", "opponent_hand", "count_state"),
        ("batter_id", "opponent_hand", "base_state"),
    ]


def build_compound_split_features(
    history: pd.DataFrame,
    rows: pd.DataFrame,
    shrinkages: tuple[float, ...] = SHRINKAGES,
) -> pd.DataFrame:
    """Build zero-centred entity effects inside fixed two-way contexts."""
    hkeys = add_context_keys(history)
    rkeys = add_context_keys(rows)
    season_mean = history.groupby("season", observed=True)["control_success"].transform("mean")
    hkeys["relative_target"] = history["control_success"].to_numpy(float) - season_mean.to_numpy(float)
    out: dict[str, np.ndarray] = {}
    for entity, left, right in compound_feature_specs():
        prefix = entity.split("_")[0]
        left_key = f"{prefix}_opponent_hand" if left == "opponent_hand" else left
        right_key = f"{prefix}_opponent_hand" if right == "opponent_hand" else right
        compound = left + "__" + right
        h_context = hkeys[left_key].astype(str) + "|" + hkeys[right_key].astype(str)
        r_context = rkeys[left_key].astype(str) + "|" + rkeys[right_key].astype(str)
        work = pd.DataFrame({"entity": hkeys[entity].astype(str), "context": h_context, "relative": hkeys["relative_target"]})
        cell = work.groupby(["entity", "context"], observed=True)["relative"].agg(["sum", "size"])
        own = work.groupby("entity", observed=True)["relative"].agg(["sum", "size"])
        cell = cell.join(own, rsuffix="_entity")
        cell["effect"] = cell["sum"] / cell["size"] - cell["sum_entity"] / cell["size_entity"]
        lookup = pd.MultiIndex.from_arrays([rkeys[entity].astype(str), r_context])
        effect = np.nan_to_num(lookup.map(cell.effect).to_numpy(float), nan=0.0)
        seen = np.nan_to_num(lookup.map(cell["size"]).to_numpy(float), nan=0.0)
        for shrinkage in shrinkages:
            out[f"compound_{entity}_{compound}_k{int(shrinkage)}"] = effect * seen / (seen + shrinkage)
    return pd.DataFrame(out, index=rows.index)
