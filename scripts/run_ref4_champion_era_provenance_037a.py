#!/usr/bin/env python3
"""Read-only provenance and package audit of REF4-CHAMPION-STACK-030."""
from __future__ import annotations

import hashlib
import json
import pickle
import re
import time
import zipfile
from pathlib import Path

import catboost as cb
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "REF4-CHAMPION-ERA-PROVENANCE-037A"
OUT = ROOT / "model" / EXPERIMENT_ID
SOURCE = ROOT / "model" / "REF4-CHAMPION-STACK-030"
CANDIDATE = ROOT / "candidate" / "REF4-CHAMPION-STACK-030"
ZIP = ROOT / "output" / "submit_ref4_champion_030.zip"
TRAIN_SCRIPT = ROOT / "scripts" / "train_and_package_ref4_champion_030.py"
TRACKMAN_SCRIPT = ROOT / "scripts" / "build_ref4_trackman_030.py"
PPT = ROOT / "solution" / "LG_Aimers_솔루션_PPT_Phase2.pptx"
SEEDS = [260802, 260803, 260804, 260805, 260806, 260807]


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def expected_components() -> dict[str, dict[str, object]]:
    specs: dict[str, dict[str, object]] = {}
    def add(name: str, trees: int, seed: int, loss: str, depth: int, thread: int, scope: str, feature_group: str) -> None:
        specs[name] = {"expected_trees": trees, "expected_seed": seed, "expected_loss": loss, "expected_depth": depth, "expected_thread_count": thread, "intended_fit_scope": scope, "feature_group": feature_group}
    for seed in SEEDS:
        add(f"v2_decay55_seed{seed}.cbm", 140, seed, "RMSE", 8, 3, "TRAIN_2019_2024_ALL_DECAY55", "v2")
        add(f"v3_decay55_seed{seed}.cbm", 220, seed, "RMSE", 8, 3, "TRAIN_2019_2024_ALL_DECAY55", "v3")
        add(f"v3_decay30_seed{seed}.cbm", 199, seed, "RMSE", 8, 3, "TRAIN_2019_2024_ALL_DECAY30", "v3")
        add(f"subtype_middle_seed{seed}.cbm", 100, seed, "Logloss", 7, 3, "TRAIN_2019_2024_RECOVERED_DECAY30", "v3")
        add(f"subtype_wild_seed{seed}.cbm", 190, seed + 100, "Logloss", 7, 3, "TRAIN_2019_2024_RECOVERED_DECAY30", "v3")
        add(f"subtype_reverse_seed{seed}.cbm", 230, seed + 200, "Logloss", 7, 3, "TRAIN_2019_2024_RECOVERED_DECAY30", "v3")
    for index in range(4):
        add(f"f_v2_all_{index}.cbm", 140, 968000 + index, "RMSE", 8, 3, "TRAIN_2019_2024_F_ALL_DECAY55", "v2")
        add(f"f_v330_all_{index}.cbm", 199, 968200 + index, "RMSE", 8, 3, "TRAIN_2019_2024_F_ALL_DECAY30", "v3")
    for index in range(6):
        add(f"f_v355_recent_{index}.cbm", 220, 968100 + index, "RMSE", 8, 3, "TRAIN_2024_F_ONLY", "v3")
    for index in range(2):
        add(f"f_v330_recent_{index}.cbm", 199, 968300 + index, "RMSE", 8, 3, "TRAIN_2024_F_ONLY", "v3")
    for index, (name, trees) in enumerate((("middle", 100), ("wild", 190), ("reverse", 230))):
        add(f"f_subtype_{name}.cbm", trees, 968400 + index, "Logloss", 7, 3, "TRAIN_2019_2024_F_RECOVERED_DECAY30", "v3")
    add("transition_gate.cbm", 10, 0, "RMSE", 3, 3, "SYNTHETIC_TWO_ROWS_DISABLED_SCALE_ZERO", "transition")
    return specs


def frame_equal(csv_path: Path, pickle_path: Path, id_column: str) -> dict[str, object]:
    csv = pd.read_csv(csv_path, dtype={id_column: str})
    stored = pd.read_pickle(pickle_path)
    stored[id_column] = stored[id_column].astype(str)
    failures: list[str] = []
    if list(csv.columns) != list(stored.columns):
        failures.append("columns")
    if len(csv) != len(stored):
        failures.append("rows")
    if not failures:
        for column in csv.columns:
            if pd.api.types.is_numeric_dtype(csv[column]):
                if not np.allclose(csv[column].to_numpy(float), pd.to_numeric(stored[column], errors="coerce").to_numpy(float), rtol=0, atol=1e-10, equal_nan=True):
                    failures.append(column)
            elif not np.array_equal(csv[column].astype(str).to_numpy(), stored[column].astype(str).to_numpy()):
                failures.append(column)
    return {"rows": int(len(csv)), "columns": int(len(csv.columns)), "match": not failures, "failures": failures, "min_season": int(csv["season"].min()), "max_season": int(csv["season"].max()), "season_2025_rows": int(csv["season"].eq(2025).sum())}


def recompute_prior(raw: pd.DataFrame) -> dict[str, str]:
    counts = raw.loc[raw["season"].lt(2025)].groupby(["pitcher_id", "season", "game_type"], observed=True).size().rename("n").reset_index()
    dominant = counts.sort_values("n").groupby(["pitcher_id", "season"], observed=True).tail(1)
    latest = dominant.sort_values("season").groupby("pitcher_id", observed=True).tail(1)
    return latest.set_index(latest["pitcher_id"].astype(str))["game_type"].astype(str).to_dict()


def main() -> None:
    started = time.time()
    preflight = json.loads((OUT / "preflight_report.json").read_text(encoding="utf-8"))
    contract = json.loads((OUT / "audit_contract.json").read_text(encoding="utf-8"))
    if preflight["status"] != "AUDIT_VERIFIED" or preflight["mismatch_count"] != 0:
        raise RuntimeError("037A preflight must be AUDIT_VERIFIED")
    specs = expected_components()
    source_code = TRAIN_SCRIPT.read_text(encoding="utf-8")
    trackman_code = TRACKMAN_SCRIPT.read_text(encoding="utf-8")
    code_intent_checks = {
        "official_train_loaded": 'raw = pd.read_csv(ROOT / "data" / "train.csv"' in source_code,
        "max_season_weight_anchor": "int(season.max()) - season" in source_code,
        "f_2024_mask": "f_2024_mask = f_mask & (season == 2024)" in source_code,
        "recent_v355_uses_f_2024": 'm.fit(x3.loc[f_2024_mask], (y - base3)[f_2024_mask]' in source_code,
        "recent_v330_uses_f_2024": source_code.count('m.fit(x3.loc[f_2024_mask], (y - base3)[f_2024_mask]') >= 2,
        "f_all_uses_f_mask": "m.fit(x3.loc[f_mask], (y - base3)[f_mask]" in source_code and "m.fit(x2.loc[f_mask], (y - base2)[f_mask]" in source_code,
        "prior_cutoff_2025": "prior_type_table(raw_for_lookup, 2025)" in source_code,
        "transition_disabled": '"transition_scale": 0.0' in source_code,
        "trackman_strict_past": "hist = player[player.season < target_season]" in trackman_code,
        "trackman_targets_through_2025": "for target_season in range(2019, 2026)" in trackman_code,
    }

    component_rows: list[dict[str, object]] = []
    feature_groups: dict[str, list[list[str]]] = {"v2": [], "v3": [], "transition": []}
    for name, expected in sorted(specs.items()):
        path = SOURCE / name
        model = cb.CatBoost()
        model.load_model(path)
        params = model.get_all_params()
        metadata = dict(model.get_metadata())
        embedded_params = json.loads(metadata["params"])
        embedded_thread_count = embedded_params.get("flat_params", {}).get("thread_count", embedded_params.get("system_options", {}).get("thread_count", -1))
        feature_names = list(model.feature_names_)
        feature_groups[str(expected["feature_group"])].append(feature_names)
        actual = {"trees": int(model.tree_count_), "seed": int(params.get("random_seed", -1)), "loss": str(params.get("loss_function")), "depth": int(params.get("depth", -1)), "thread_count": int(embedded_thread_count)}
        param_matches = {key: actual[key] == expected[f"expected_{key}"] for key in ("trees", "seed", "loss", "depth", "thread_count")}
        candidate_path = CANDIDATE / "model" / name
        component_rows.append({"component": name, **expected, "actual_trees": actual["trees"], "actual_seed": actual["seed"], "actual_loss": actual["loss"], "actual_depth": actual["depth"], "actual_thread_count": actual["thread_count"], "feature_count": len(feature_names), "train_finish_time": metadata.get("train_finish_time"), "source_sha256": sha256_path(path), "candidate_sha256": sha256_path(candidate_path) if candidate_path.is_file() else None, "candidate_match": candidate_path.is_file() and sha256_path(path) == sha256_path(candidate_path), "parameter_match": all(param_matches.values()), "parameter_mismatches": "|".join(key for key, passed in param_matches.items() if not passed)})
    components = pd.DataFrame(component_rows).sort_values("component").reset_index(drop=True)
    components.to_csv(OUT / "component_inventory.csv", index=False)
    feature_group_checks = {group: all(names == collections[0] for names in collections[1:]) for group, collections in feature_groups.items()}

    expected_zip_sources: dict[str, Path] = {"script.py": CANDIDATE / "script.py", "requirements.txt": CANDIDATE / "requirements.txt", "solution/LG_Aimers_솔루션_PPT_Phase2.pptx": PPT}
    for path in sorted((CANDIDATE / "src").glob("*.py")):
        expected_zip_sources[f"src/{path.name}"] = path
    for path in sorted((CANDIDATE / "model").iterdir()):
        if path.is_file():
            expected_zip_sources[f"model/{path.name}"] = path
    package_rows: list[dict[str, object]] = []
    with zipfile.ZipFile(ZIP) as archive:
        names = archive.namelist()
        for arcname in sorted(set(names) | set(expected_zip_sources)):
            local_path = expected_zip_sources.get(arcname)
            zip_present = arcname in names
            local_present = local_path is not None and local_path.is_file()
            local_hash = sha256_path(local_path) if local_present else None
            zip_data = archive.read(arcname) if zip_present else None
            zip_hash = sha256_bytes(zip_data) if zip_data is not None else None
            source_match: bool | None = None
            if arcname.startswith("model/") and local_present:
                source_path = SOURCE / Path(arcname).name
                source_match = source_path.is_file() and sha256_path(source_path) == local_hash
            package_rows.append({"arcname": arcname, "local_path": str(local_path.relative_to(ROOT)) if local_path else None, "local_present": local_present, "zip_present": zip_present, "local_size": local_path.stat().st_size if local_present else None, "zip_size": len(zip_data) if zip_data is not None else None, "local_sha256": local_hash, "zip_sha256": zip_hash, "zip_local_match": local_present and zip_present and local_hash == zip_hash, "source_model_match": source_match})
    package = pd.DataFrame(package_rows).sort_values("arcname").reset_index(drop=True)
    package.to_csv(OUT / "package_inventory.csv", index=False)

    snapshot_report = {"pitcher_snapshots": frame_equal(SOURCE / "pitcher_snapshots.csv", SOURCE / "pitcher_snapshots.pkl", "pitcher_id"), "batter_snapshots": frame_equal(SOURCE / "batter_snapshots.csv", SOURCE / "batter_snapshots.pkl", "batter_id"), "pitchmix_snapshots": frame_equal(SOURCE / "pitchmix_snapshots.csv", SOURCE / "pitchmix_snapshots.pkl", "pitcher_id")}
    raw = pd.read_csv(ROOT / "data" / "train.csv", dtype={"pitcher_id": str}, usecols=["pitcher_id", "season", "game_type"], low_memory=False)
    expected_prior = recompute_prior(raw)
    with (SOURCE / "prior_type.pkl").open("rb") as handle:
        pkl_prior = {str(key): str(value) for key, value in pickle.load(handle).items()}
    csv_prior_frame = pd.read_csv(SOURCE / "prior_type.csv", dtype={"pitcher_id": str, "prior_type": str})
    csv_prior = csv_prior_frame.set_index("pitcher_id")["prior_type"].astype(str).to_dict()
    prior_report = {"expected_count": len(expected_prior), "pkl_count": len(pkl_prior), "csv_count": len(csv_prior), "pkl_matches_recomputed": pkl_prior == expected_prior, "csv_matches_recomputed": csv_prior == expected_prior, "pkl_csv_match": pkl_prior == csv_prior, "cutoff": 2025, "source_max_season": int(raw["season"].max())}
    trackman = pd.read_csv(SOURCE / "trackman_prior_features.csv", low_memory=False)
    trackman_source = pd.read_csv(ROOT / "data" / "trackman_history.csv", usecols=["season"])
    trackman_report = {"rows": len(trackman), "min_target_season": int(trackman["season"].min()), "max_target_season": int(trackman["season"].max()), "target_2025_rows": int(trackman["season"].eq(2025).sum()), "source_min_season": int(trackman_source["season"].min()), "source_max_season": int(trackman_source["season"].max()), "source_strictly_before_2025": bool(int(trackman_source["season"].max()) < 2025)}
    lookup_snapshot_report = {"snapshots": snapshot_report, "prior_type": prior_report, "trackman": trackman_report}
    write_json(OUT / "lookup_snapshot_report.json", lookup_snapshot_report)

    basic_manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8"))
    binding_candidates = sorted(path.name for path in SOURCE.iterdir() if path.is_file() and ("audit" in path.name.lower() or "attest" in path.name.lower() or "build_manifest" in path.name.lower()))
    binding_report = {"binding_evidence_files": binding_candidates, "binding_evidence_count": len(binding_candidates), "manifest_has_train_sha256": any("train" in key.lower() and "sha" in key.lower() for key in basic_manifest), "manifest_has_training_script_sha256": any("script" in key.lower() and "sha" in key.lower() for key in basic_manifest), "per_model_source_binding_available": False}

    parameter_mismatches = components.loc[~components["parameter_match"], ["component", "parameter_mismatches", "expected_thread_count", "actual_thread_count"]].to_dict(orient="records")
    package_mismatches = package.loc[~package["zip_local_match"] | package["source_model_match"].eq(False), ["arcname", "zip_local_match", "source_model_match"]].to_dict(orient="records")
    snapshot_failures = [name for name, value in snapshot_report.items() if not value["match"]]
    semantic_failures = []
    if not all(code_intent_checks.values()): semantic_failures.append("code_intent")
    if not all(feature_group_checks.values()): semantic_failures.append("feature_groups")
    if snapshot_failures: semantic_failures.append("snapshots")
    if not all((prior_report["pkl_matches_recomputed"], prior_report["csv_matches_recomputed"], prior_report["pkl_csv_match"])): semantic_failures.append("prior_type")
    if not trackman_report["source_strictly_before_2025"] or trackman_report["max_target_season"] != 2025: semantic_failures.append("trackman_cutoff")
    issues: list[dict[str, object]] = []
    if parameter_mismatches: issues.append({"code": "CBM_PARAMETER_MISMATCH", "count": len(parameter_mismatches), "details": parameter_mismatches})
    if package_mismatches: issues.append({"code": "PACKAGE_BYTE_MISMATCH", "count": len(package_mismatches), "details": package_mismatches})
    if semantic_failures: issues.append({"code": "SEMANTIC_PROVENANCE_MISMATCH", "count": len(semantic_failures), "details": semantic_failures})
    if not binding_report["per_model_source_binding_available"]: issues.append({"code": "MISSING_MODEL_INPUT_BINDING", "count": 1, "details": binding_report})
    if parameter_mismatches or package_mismatches or semantic_failures:
        provenance_status = "AUDIT_FAIL_PROVENANCE"
    elif not binding_report["per_model_source_binding_available"]:
        provenance_status = "AUDIT_INCOMPLETE"
    else:
        provenance_status = "AUDIT_VERIFIED"
    counts = {"audited_cbm_count": len(components), "parameter_match_count": int(components["parameter_match"].sum()), "parameter_mismatch_count": int((~components["parameter_match"]).sum()), "source_file_count": len([path for path in SOURCE.iterdir() if path.is_file()]), "candidate_file_count": len([path for path in CANDIDATE.rglob("*") if path.is_file()]), "zip_member_count": len(package), "package_match_count": int(package["zip_local_match"].sum()), "package_mismatch_count": len(package_mismatches), "issue_count": len(issues), "binding_evidence_count": binding_report["binding_evidence_count"]}
    result = {"experiment_id": EXPERIMENT_ID, "execution_status": "PENDING_AUDIT", "provenance_status": provenance_status, "source_official_score": 1068.25021, "code_intent_checks": code_intent_checks, "feature_group_checks": feature_group_checks, "binding_report": binding_report, "lookup_snapshot_report": lookup_snapshot_report, "counts": counts, "issues": issues, "candidate_count": 0, "actual_leaf_count": 0, "gate_checks_count": 0, "model_count_created": 0, "test_read_performed": False, "test_inference_performed": False, "training_performed": False, "zip_created": False, "elapsed_seconds": time.time() - started}
    write_json(OUT / "result.json", result)
    embedded_value = {key: value for key, value in result.items() if key != "elapsed_seconds"}
    embedded = json.dumps(embedded_value, sort_keys=True)
    lines = [f"# {EXPERIMENT_ID}", "", "- execution status: `PENDING_AUDIT`", f"- provenance status: `{provenance_status}`", f"- audited CBM: `{counts['audited_cbm_count']}`", f"- parameter match/mismatch: `{counts['parameter_match_count']}/{counts['parameter_mismatch_count']}`", f"- ZIP member match/mismatch: `{counts['package_match_count']}/{counts['package_mismatch_count']}`", f"- binding evidence: `{counts['binding_evidence_count']}`", "- candidate/leaf/gate/new-model: `0/0/0/0`", "- test-read/test-inference/training/ZIP: `false/false/false/false`", "", "<!-- RESULT_JSON_BEGIN", embedded, "RESULT_JSON_END -->"]
    (OUT / "result.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    artifact_paths = [ROOT / "scripts" / "preflight_ref4_champion_era_provenance_037a.py", ROOT / "scripts" / "run_ref4_champion_era_provenance_037a.py", ROOT / "scripts" / "verify_ref4_champion_era_provenance_037a.py", OUT / "audit_contract.json", OUT / "preflight_report.json", OUT / "preflight_report.md", OUT / "component_inventory.csv", OUT / "package_inventory.csv", OUT / "lookup_snapshot_report.json", OUT / "result.json", OUT / "result.md", TRAIN_SCRIPT, TRACKMAN_SCRIPT, ROOT / "data" / "train.csv", ROOT / "data" / "trackman_history.csv", ROOT / "start03_reference.md", ROOT / "01_제약과금지사항.md", ZIP]
    artifact_paths.extend(sorted(path for path in SOURCE.iterdir() if path.is_file()))
    artifact_paths.extend(sorted(path for path in CANDIDATE.rglob("*") if path.is_file() and "__pycache__" not in path.parts))
    unique_paths = list(dict.fromkeys(artifact_paths))
    records = {str(path.relative_to(ROOT)): {"sha256": sha256_path(path), "size": path.stat().st_size} for path in unique_paths}
    audit = {"experiment_id": EXPERIMENT_ID, "status": "PENDING_VALIDATION", "artifact_count": len(records), "artifacts": records, "candidate_count": 0, "leaf_count": 0, "gate_checks_count": 0, "model_count_created": 0, **counts}
    write_json(OUT / "audit_manifest.json", audit)
    print(json.dumps({"experiment_id": EXPERIMENT_ID, "execution_status": "PENDING_AUDIT", "provenance_status": provenance_status, **counts, "elapsed_seconds": result["elapsed_seconds"]}, indent=2))


if __name__ == "__main__":
    main()
