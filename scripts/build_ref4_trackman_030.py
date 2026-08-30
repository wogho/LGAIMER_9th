#!/usr/bin/env python3
"""Build Trackman prior summaries exactly as 4th Repo."""
from __future__ import annotations

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "model" / "REF4-CHAMPION-STACK-030"
OUT.mkdir(parents=True, exist_ok=True)

cols = ["pitcher_id", "season", "game_month", "game_dayofweek", "pitcher_hand"]
tcols = ["pitcher_trackman_id", "season", "game_month", "game_dayofweek", "pitcher_hand"]
tr = pd.read_csv(ROOT / "data/train.csv", usecols=cols)
tmkey = pd.read_csv(ROOT / "data/trackman_history.csv", usecols=tcols)

tr["key"] = tr.season.astype(str) + "_" + tr.game_month.astype(str) + "_" + tr.game_dayofweek.astype(str)
tmkey["key"] = tmkey.season.astype(str) + "_" + tmkey.game_month.astype(str) + "_" + tmkey.game_dayofweek.astype(str)
A = tr.groupby(["pitcher_id", "key"]).size().unstack(fill_value=0)
B = tmkey.groupby(["pitcher_trackman_id", "key"]).size().unstack(fill_value=0)
allcols = A.columns.union(B.columns)
A = A.reindex(columns=allcols, fill_value=0)
B = B.reindex(columns=allcols, fill_value=0)

sim = (normalize(csr_matrix(A.values)) @ normalize(csr_matrix(B.values)).T).toarray()
ha = tr.groupby("pitcher_id").pitcher_hand.first().reindex(A.index).values
hb = tmkey.groupby("pitcher_trackman_id").pitcher_hand.first().reindex(B.index).map({"Left": 1, "Right": 2}).values
sim[ha[:, None] != hb[None, :]] = -1

best_j = sim.argmax(1)
reverse_best = sim.argmax(0)
ordered = np.sort(sim, axis=1)
confidence = sim[np.arange(len(A)), best_j]
margin = ordered[:, -1] - ordered[:, -2]

ok = (reverse_best[best_j] == np.arange(len(A))) & (confidence >= 0.90) & (margin >= 0.03)
mapping = pd.DataFrame({
    "pitcher_id": A.index[ok],
    "pitcher_trackman_id": B.index[best_j[ok]],
    "mapping_similarity": confidence[ok],
    "mapping_margin": margin[ok]
})
mapping.to_csv(OUT / "pitcher_trackman_mapping.csv", index=False)
print("Mapped", len(mapping), "pitchers. Train row coverage:", tr.pitcher_id.isin(mapping.pitcher_id).mean())

metrics = ["rel_speed", "spin_rate", "induced_vert_break", "horz_break", "extension", "rel_height", "rel_side", "zone_speed"]
tm = pd.read_csv(ROOT / "data/trackman_history.csv", usecols=["pitcher_trackman_id", "season", "pitch_type_group", *metrics])
rows = []
for rec in mapping.itertuples(index=False):
    player = tm[tm.pitcher_trackman_id == rec.pitcher_trackman_id]
    for target_season in range(2019, 2026):
        hist = player[player.season < target_season]
        row = {
            "pitcher_id": str(rec.pitcher_id),
            "season": target_season,
            "tm_mapping_similarity": rec.mapping_similarity,
            "tm_n": len(hist)
        }
        for c in metrics:
            row[f"tm_{c}_mean"] = hist[c].mean()
            row[f"tm_{c}_std"] = hist[c].std()
            last = hist[hist.season == target_season - 1][c]
            earlier = hist[hist.season < target_season - 1][c]
            row[f"tm_last_{c}_mean"] = last.mean()
            row[f"tm_last_{c}_std"] = last.std()
            row[f"tm_{c}_trend"] = last.mean() - earlier.mean()
        rates = hist.pitch_type_group.value_counts(normalize=True)
        for p in ["fastball", "breaking", "offspeed", "other"]:
            row[f"tm_{p}_rate"] = rates.get(p, np.nan)
        last_hist = hist[hist.season == target_season - 1]
        row["tm_last_n"] = len(last_hist)
        last_rates = last_hist.pitch_type_group.value_counts(normalize=True)
        for p in ["fastball", "breaking", "offspeed", "other"]:
            row[f"tm_last_{p}_rate"] = last_rates.get(p, np.nan)
            subset = hist[hist.pitch_type_group == p]
            recent_subset = last_hist[last_hist.pitch_type_group == p]
            for c in ["rel_speed", "spin_rate", "induced_vert_break", "horz_break"]:
                row[f"tm_{p}_{c}_mean"] = subset[c].mean()
                row[f"tm_last_{p}_{c}_mean"] = recent_subset[c].mean()
        rows.append(row)

table = pd.DataFrame(rows)
table.to_csv(OUT / "trackman_prior_features.csv", index=False)
print("Saved Trackman prior features table:", table.shape)
