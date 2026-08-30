"""Train-only entity-by-game-state partial-pooling residual features."""
from __future__ import annotations

import numpy as np
import pandas as pd


SHRINKAGES = (50.0, 200.0, 800.0, 3200.0)
CONTEXTS = (
    "exact_inning",
    "outs_before",
    "score_diff_pitcher_bucket",
    "leverage_bucket",
    "pitcher_win_expectancy_bucket",
    "run_total_bucket",
    "season_phase",
    "pitcher_home",
)


def add_game_state_keys(frame: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    out["pitcher_id"] = frame["pitcher_id"].astype(str)
    out["batter_id"] = frame["batter_id"].astype(str)
    inning = pd.to_numeric(frame["inning"], errors="coerce").fillna(1).clip(1, 10).astype(int)
    outs = pd.to_numeric(frame["outs_before"], errors="coerce").fillna(0).clip(0, 2).astype(int)
    score = pd.to_numeric(frame["score_diff_pitcher_team"], errors="coerce").fillna(0)
    leverage = pd.to_numeric(frame["li"], errors="coerce").fillna(1).clip(lower=0)
    total = pd.to_numeric(frame["run_total_before"], errors="coerce").fillna(0).clip(lower=0)
    month = pd.to_numeric(frame["game_month"], errors="coerce").fillna(6).astype(int)
    top_value = frame["top_bottom"].fillna("T").astype(str).str.upper()
    top = top_value.eq("T") | top_value.str.startswith("TOP")
    home_we = pd.to_numeric(frame["home_win_expectancy"], errors="coerce").fillna(50.0)
    away_we = pd.to_numeric(frame["away_win_expectancy"], errors="coerce").fillna(50.0)
    pitcher_we = np.where(top, home_we, away_we)
    out["exact_inning"] = np.where(inning.ge(10), "10+", inning.astype(str))
    out["outs_before"] = outs.astype(str)
    out["score_diff_pitcher_bucket"] = pd.cut(score, [-np.inf, -4, -2, -1, 0, 1, 3, np.inf], labels=["lt-4", "-4:-2", "-2:-1", "-1:0", "0:1", "1:3", "ge3"], right=False).astype(str)
    out["leverage_bucket"] = pd.cut(leverage, [-np.inf, 0.5, 1.0, 2.0, np.inf], labels=["low", "medium", "high", "very_high"], right=False).astype(str)
    out["pitcher_win_expectancy_bucket"] = pd.cut(pitcher_we, [-np.inf, 25, 45, 55, 75, np.inf], labels=["very_low", "low", "even", "high", "very_high"], right=False).astype(str)
    out["run_total_bucket"] = pd.cut(total, [-np.inf, 1, 3, 6, np.inf], labels=["0", "1:2", "3:5", "6+"], right=False).astype(str)
    out["season_phase"] = pd.cut(month, [-np.inf, 5, 7, np.inf], labels=["early", "middle", "late"], right=False).astype(str)
    out["pitcher_home"] = np.where(top, "home", "away")
    return out


def feature_specs() -> list[tuple[str, str]]:
    return [(entity, context) for entity in ("pitcher_id", "batter_id") for context in CONTEXTS]


def build_game_state_features(history: pd.DataFrame, rows: pd.DataFrame, shrinkages: tuple[float, ...] = SHRINKAGES) -> pd.DataFrame:
    hkeys = add_game_state_keys(history)
    rkeys = add_game_state_keys(rows)
    season_mean = history.groupby("season", observed=True)["control_success"].transform("mean")
    hkeys["relative_target"] = history["control_success"].to_numpy(float) - season_mean.to_numpy(float)
    out: dict[str, np.ndarray] = {}
    for entity, context in feature_specs():
        cell = hkeys.groupby([entity, context], observed=True)["relative_target"].agg(["sum", "size"])
        own = hkeys.groupby(entity, observed=True)["relative_target"].agg(["sum", "size"])
        cell = cell.join(own, rsuffix="_entity")
        cell["effect"] = cell["sum"] / cell["size"] - cell["sum_entity"] / cell["size_entity"]
        lookup = pd.MultiIndex.from_arrays([rkeys[entity], rkeys[context]])
        effect = np.nan_to_num(lookup.map(cell["effect"]).to_numpy(float), nan=0.0)
        seen = np.nan_to_num(lookup.map(cell["size"]).to_numpy(float), nan=0.0)
        for shrinkage in shrinkages:
            out[f"state_{entity}_{context}_k{int(shrinkage)}"] = effect * seen / (seen + shrinkage)
    return pd.DataFrame(out, index=rows.index)


def build_game_state_profile(history: pd.DataFrame) -> pd.DataFrame:
    keys = add_game_state_keys(history)
    season_mean = history.groupby("season", observed=True)["control_success"].transform("mean")
    keys["relative_target"] = history["control_success"].to_numpy(float) - season_mean.to_numpy(float)
    parts = []
    for entity, context in feature_specs():
        cell = keys.groupby([entity, context], observed=True)["relative_target"].agg(["sum", "size"])
        own = keys.groupby(entity, observed=True)["relative_target"].agg(["sum", "size"])
        cell = cell.join(own, rsuffix="_entity").reset_index()
        cell["effect"] = cell["sum"] / cell["size"] - cell["sum_entity"] / cell["size_entity"]
        parts.append(pd.DataFrame({"entity_name": entity, "context_name": context, "entity_value": cell[entity].astype(str), "context_value": cell[context].astype(str), "effect": cell["effect"].to_numpy(float), "count": cell["size"].to_numpy(int)}))
    return pd.concat(parts, ignore_index=True)


def apply_game_state_profile(rows: pd.DataFrame, profile: pd.DataFrame, shrinkages: tuple[float, ...] = SHRINKAGES) -> pd.DataFrame:
    keys = add_game_state_keys(rows)
    out: dict[str, np.ndarray] = {}
    for entity, context in feature_specs():
        table = profile.loc[profile.entity_name.eq(entity) & profile.context_name.eq(context), ["entity_value", "context_value", "effect", "count"]].copy()
        table["entity_value"] = table["entity_value"].astype(str)
        table["context_value"] = table["context_value"].astype(str)
        table = table.set_index(["entity_value", "context_value"])
        lookup = pd.MultiIndex.from_arrays([keys[entity].astype(str), keys[context].astype(str)])
        effect = np.nan_to_num(lookup.map(table.effect).to_numpy(float), nan=0.0)
        count = np.nan_to_num(lookup.map(table["count"]).to_numpy(float), nan=0.0)
        for shrinkage in shrinkages:
            out[f"state_{entity}_{context}_k{int(shrinkage)}"] = effect * count / (count + shrinkage)
    return pd.DataFrame(out, index=rows.index)
