#!/usr/bin/env python3
"""REF-AUX-OFFSET-CAT-001: fixed CatBoost confirmation run.

Wraps the prior forward-only experiment with a fixed CatBoost auxiliary model;
no hyperparameter or submission search is performed.
"""

from pathlib import Path
import hashlib
import sys

import catboost as cb

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import run_ref_aux_offset_001 as base

base.EXP = ROOT / "model" / "REF-AUX-OFFSET-CAT-001"
base.EXPERIMENT_ID = "REF-AUX-OFFSET-CAT-001"
base.AUX_MODEL_NAME = "CatBoostClassifier(iterations=100,depth=6)"
base.AUX_ARTIFACTS = []
base.TRANSITION_ARTIFACTS = []
_call_index = 0
_call_names = [(s, n) for s in (2022, 2023, 2024) for n in ("mr", "wayoff")]
base.PARAMS = dict(
    iterations=100,
    learning_rate=0.05,
    depth=6,
    loss_function="Logloss",
    verbose=False,
    thread_count=6,
    allow_writing_files=False,
    random_seed=2024,
)


def predict_aux(train_x, train_y, valid_x):
    global _call_index
    categories = [c for c in train_x.columns if str(train_x[c].dtype) in {"category", "object"}]
    train_x = train_x.copy()
    valid_x = valid_x.copy()
    for col in categories:
        train_x[col] = train_x[col].astype("string").fillna("<NA>").astype(str)
        valid_x[col] = valid_x[col].astype("string").fillna("<NA>").astype(str)
    model = cb.CatBoostClassifier(**base.PARAMS)
    model.fit(cb.Pool(train_x, label=train_y, cat_features=categories), verbose=False)
    season, name = _call_names[_call_index]
    _call_index += 1
    artifact = base.EXP / "aux_models" / f"catboost_{season}_{name}.cbm"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(artifact))
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    base.AUX_ARTIFACTS.append({"path": str(artifact.relative_to(base.ROOT)), "sha256": digest, "fit_season": season, "target": name})
    return (
        model.predict_proba(cb.Pool(valid_x, cat_features=categories))[:, 1],
        model.predict_proba(cb.Pool(train_x, cat_features=categories))[:, 1],
    )


base.predict_aux = predict_aux
base.main()
