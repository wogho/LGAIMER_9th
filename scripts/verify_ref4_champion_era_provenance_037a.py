#!/usr/bin/env python3
"""Independent validator for REF4-CHAMPION-ERA-PROVENANCE-037A."""
from __future__ import annotations

import hashlib
import json
import pickle
import re
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


def specs() -> dict[str, dict[str, object]]:
    result: dict[str, dict[str, object]] = {}
    def add(name: str, trees: int, seed: int, loss: str, depth: int, scope: str, group: str) -> None:
        result[name] = {"expected_trees": trees, "expected_seed": seed, "expected_loss": loss, "expected_depth": depth, "expected_thread_count": 3, "intended_fit_scope": scope, "feature_group": group}
    for seed in SEEDS:
        add(f"v2_decay55_seed{seed}.cbm", 140, seed, "RMSE", 8, "TRAIN_2019_2024_ALL_DECAY55", "v2")
        add(f"v3_decay55_seed{seed}.cbm", 220, seed, "RMSE", 8, "TRAIN_2019_2024_ALL_DECAY55", "v3")
        add(f"v3_decay30_seed{seed}.cbm", 199, seed, "RMSE", 8, "TRAIN_2019_2024_ALL_DECAY30", "v3")
        add(f"subtype_middle_seed{seed}.cbm", 100, seed, "Logloss", 7, "TRAIN_2019_2024_RECOVERED_DECAY30", "v3")
        add(f"subtype_wild_seed{seed}.cbm", 190, seed + 100, "Logloss", 7, "TRAIN_2019_2024_RECOVERED_DECAY30", "v3")
        add(f"subtype_reverse_seed{seed}.cbm", 230, seed + 200, "Logloss", 7, "TRAIN_2019_2024_RECOVERED_DECAY30", "v3")
    for index in range(4):
        add(f"f_v2_all_{index}.cbm", 140, 968000 + index, "RMSE", 8, "TRAIN_2019_2024_F_ALL_DECAY55", "v2")
        add(f"f_v330_all_{index}.cbm", 199, 968200 + index, "RMSE", 8, "TRAIN_2019_2024_F_ALL_DECAY30", "v3")
    for index in range(6): add(f"f_v355_recent_{index}.cbm", 220, 968100 + index, "RMSE", 8, "TRAIN_2024_F_ONLY", "v3")
    for index in range(2): add(f"f_v330_recent_{index}.cbm", 199, 968300 + index, "RMSE", 8, "TRAIN_2024_F_ONLY", "v3")
    for index, (name, trees) in enumerate((("middle", 100), ("wild", 190), ("reverse", 230))): add(f"f_subtype_{name}.cbm", trees, 968400 + index, "Logloss", 7, "TRAIN_2019_2024_F_RECOVERED_DECAY30", "v3")
    add("transition_gate.cbm", 10, 0, "RMSE", 3, "SYNTHETIC_TWO_ROWS_DISABLED_SCALE_ZERO", "transition")
    return result


def compare_frames(actual: pd.DataFrame, expected: pd.DataFrame) -> dict[str, object]:
    failures: list[str] = []
    maximum = 0.0
    if list(actual.columns) != list(expected.columns): return {"failures": ["columns"], "max_abs_diff": 0.0}
    if len(actual) != len(expected): return {"failures": ["rows"], "max_abs_diff": 0.0}
    for column in expected:
        if pd.api.types.is_numeric_dtype(expected[column]):
            left = pd.to_numeric(actual[column], errors="coerce").to_numpy(float); right = pd.to_numeric(expected[column], errors="coerce").to_numpy(float)
            if not np.array_equal(np.isfinite(left), np.isfinite(right)): failures.append(f"{column}:finite"); continue
            finite = np.isfinite(right); diff = float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else 0.0; maximum = max(maximum, diff)
            if diff > 1e-10: failures.append(f"{column}:{diff}")
        elif not np.array_equal(actual[column].fillna("<NA>").astype(str).to_numpy(), expected[column].fillna("<NA>").astype(str).to_numpy()): failures.append(f"{column}:values")
    return {"failures": failures, "max_abs_diff": maximum}


def compare_objects(actual: object, expected: object, path: str = "") -> tuple[float, list[str]]:
    maximum = 0.0; failures: list[str] = []
    if isinstance(actual, dict) and isinstance(expected, dict):
        if set(actual) != set(expected): return 0.0, [f"{path}:keys"]
        for key in actual:
            diff, child = compare_objects(actual[key], expected[key], f"{path}.{key}" if path else str(key)); maximum = max(maximum, diff); failures.extend(child)
    elif isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected): failures.append(f"{path}:length")
        else:
            for index, (left, right) in enumerate(zip(actual, expected)):
                diff, child = compare_objects(left, right, f"{path}[{index}]"); maximum = max(maximum, diff); failures.extend(child)
    elif isinstance(actual, (int, float)) and not isinstance(actual, bool) and isinstance(expected, (int, float)) and not isinstance(expected, bool):
        diff = abs(float(actual) - float(expected)); maximum = max(maximum, diff)
        if diff > 1e-10: failures.append(f"{path}:{diff}")
    elif actual != expected: failures.append(f"{path}:{actual!r}!={expected!r}")
    return maximum, failures


def snapshot_check(stem: str, id_column: str) -> dict[str, object]:
    csv = pd.read_csv(SOURCE / f"{stem}.csv", dtype={id_column: str}); pkl = pd.read_pickle(SOURCE / f"{stem}.pkl"); pkl[id_column] = pkl[id_column].astype(str); failures: list[str] = []
    if list(csv.columns) != list(pkl.columns): failures.append("columns")
    if len(csv) != len(pkl): failures.append("rows")
    if not failures:
        for column in csv:
            if pd.api.types.is_numeric_dtype(csv[column]):
                if not np.allclose(csv[column].to_numpy(float), pd.to_numeric(pkl[column], errors="coerce").to_numpy(float), rtol=0, atol=1e-10, equal_nan=True): failures.append(column)
            elif not np.array_equal(csv[column].astype(str).to_numpy(), pkl[column].astype(str).to_numpy()): failures.append(column)
    return {"rows": int(len(csv)), "columns": int(len(csv.columns)), "match": not failures, "failures": failures, "min_season": int(csv["season"].min()), "max_season": int(csv["season"].max()), "season_2025_rows": int(csv["season"].eq(2025).sum())}


def main() -> None:
    checks: list[dict[str, object]] = []; mismatches: list[str] = []
    def check(name: str, passed: bool, actual: object) -> None:
        checks.append({"name": name, "checked": True, "passed": bool(passed), "actual": actual})
        if not passed: mismatches.append(name)
    required = [OUT / name for name in ("audit_manifest.json", "audit_contract.json", "component_inventory.csv", "package_inventory.csv", "lookup_snapshot_report.json", "result.json", "result.md")]
    for path in required: check(f"exists:{path.name}", path.is_file(), path.is_file())
    audit = json.loads((OUT / "audit_manifest.json").read_text(encoding="utf-8")); contract = json.loads((OUT / "audit_contract.json").read_text(encoding="utf-8")); result = json.loads((OUT / "result.json").read_text(encoding="utf-8"))
    artifact_failures: list[str] = []
    for relative, record in audit["artifacts"].items():
        path = ROOT / relative
        if not path.is_file(): artifact_failures.append(f"missing:{relative}"); continue
        if sha256_path(path) != record["sha256"]: artifact_failures.append(f"sha256:{relative}")
        if path.stat().st_size != record["size"]: artifact_failures.append(f"size:{relative}")
    check("manifest_artifact_hashes", not artifact_failures, artifact_failures); check("manifest_artifact_count", len(audit["artifacts"]) == audit["artifact_count"], {"actual": len(audit["artifacts"]), "recorded": audit["artifact_count"]})
    check("diagnostic_only_counts", contract["candidate_count"] == contract["actual_leaf_count"] == contract["gate_checks_count"] == contract["model_count_created"] == result["candidate_count"] == result["actual_leaf_count"] == result["gate_checks_count"] == result["model_count_created"] == 0, {key: result[key] for key in ("candidate_count", "actual_leaf_count", "gate_checks_count", "model_count_created")})
    check("no_mutating_actions", result["test_read_performed"] is False and result["test_inference_performed"] is False and result["training_performed"] is False and result["zip_created"] is False, {key: result[key] for key in ("test_read_performed", "test_inference_performed", "training_performed", "zip_created")})

    expected_specs = specs(); actual_names = {path.name for path in SOURCE.glob("*.cbm")}; check("component_name_set", actual_names == set(expected_specs), {"missing": sorted(set(expected_specs) - actual_names), "extra": sorted(actual_names - set(expected_specs))})
    rows: list[dict[str, object]] = []; groups: dict[str, list[list[str]]] = {"v2": [], "v3": [], "transition": []}
    for name, expected in sorted(expected_specs.items()):
        model = cb.CatBoost(); model.load_model(SOURCE / name); params = model.get_all_params(); metadata = dict(model.get_metadata()); embedded_params = json.loads(metadata["params"]); embedded_thread_count = embedded_params.get("flat_params", {}).get("thread_count", embedded_params.get("system_options", {}).get("thread_count", -1)); names = list(model.feature_names_); groups[str(expected["feature_group"])].append(names)
        actual = {"trees": int(model.tree_count_), "seed": int(params.get("random_seed", -1)), "loss": str(params.get("loss_function")), "depth": int(params.get("depth", -1)), "thread_count": int(embedded_thread_count)}; matches = {key: actual[key] == expected[f"expected_{key}"] for key in actual}; candidate_path = CANDIDATE / "model" / name
        rows.append({"component": name, **expected, "actual_trees": actual["trees"], "actual_seed": actual["seed"], "actual_loss": actual["loss"], "actual_depth": actual["depth"], "actual_thread_count": actual["thread_count"], "feature_count": len(names), "train_finish_time": metadata.get("train_finish_time"), "source_sha256": sha256_path(SOURCE / name), "candidate_sha256": sha256_path(candidate_path) if candidate_path.is_file() else None, "candidate_match": candidate_path.is_file() and sha256_path(SOURCE / name) == sha256_path(candidate_path), "parameter_match": all(matches.values()), "parameter_mismatches": "|".join(key for key, passed in matches.items() if not passed)})
    expected_components = pd.DataFrame(rows).sort_values("component").reset_index(drop=True); stored_components = pd.read_csv(OUT / "component_inventory.csv", dtype={"component": str, "actual_loss": str, "expected_loss": str, "parameter_mismatches": str}); stored_components["parameter_mismatches"] = stored_components["parameter_mismatches"].fillna(""); component_compare = compare_frames(stored_components, expected_components); check("component_inventory_recomputed", not component_compare["failures"], component_compare)
    feature_checks = {group: all(value == collections[0] for value in collections[1:]) for group, collections in groups.items()}

    expected_sources: dict[str, Path] = {"script.py": CANDIDATE / "script.py", "requirements.txt": CANDIDATE / "requirements.txt", "solution/LG_Aimers_솔루션_PPT_Phase2.pptx": PPT}
    for path in sorted((CANDIDATE / "src").glob("*.py")): expected_sources[f"src/{path.name}"] = path
    for path in sorted((CANDIDATE / "model").iterdir()):
        if path.is_file(): expected_sources[f"model/{path.name}"] = path
    package_rows: list[dict[str, object]] = []
    with zipfile.ZipFile(ZIP) as archive:
        archive_names = archive.namelist()
        for arcname in sorted(set(archive_names) | set(expected_sources)):
            local = expected_sources.get(arcname); in_zip = arcname in archive_names; local_ok = local is not None and local.is_file(); local_hash = sha256_path(local) if local_ok else None; data = archive.read(arcname) if in_zip else None; zip_hash = sha256_bytes(data) if data is not None else None; source_match = None
            if arcname.startswith("model/") and local_ok: source_path = SOURCE / Path(arcname).name; source_match = source_path.is_file() and sha256_path(source_path) == local_hash
            package_rows.append({"arcname": arcname, "local_path": str(local.relative_to(ROOT)) if local else None, "local_present": local_ok, "zip_present": in_zip, "local_size": local.stat().st_size if local_ok else None, "zip_size": len(data) if data is not None else None, "local_sha256": local_hash, "zip_sha256": zip_hash, "zip_local_match": local_ok and in_zip and local_hash == zip_hash, "source_model_match": source_match})
    expected_package = pd.DataFrame(package_rows).sort_values("arcname").reset_index(drop=True); stored_package = pd.read_csv(OUT / "package_inventory.csv", dtype={"arcname": str, "local_path": str}); package_compare = compare_frames(stored_package, expected_package); check("package_inventory_recomputed", not package_compare["failures"], package_compare)

    snapshot_report = {"pitcher_snapshots": snapshot_check("pitcher_snapshots", "pitcher_id"), "batter_snapshots": snapshot_check("batter_snapshots", "batter_id"), "pitchmix_snapshots": snapshot_check("pitchmix_snapshots", "pitcher_id")}
    raw = pd.read_csv(ROOT / "data" / "train.csv", dtype={"pitcher_id": str}, usecols=["pitcher_id", "season", "game_type"], low_memory=False); counts_prior = raw.loc[raw["season"].lt(2025)].groupby(["pitcher_id", "season", "game_type"], observed=True).size().rename("n").reset_index(); dominant = counts_prior.sort_values("n").groupby(["pitcher_id", "season"], observed=True).tail(1); latest = dominant.sort_values("season").groupby("pitcher_id", observed=True).tail(1); expected_prior = latest.set_index(latest["pitcher_id"].astype(str))["game_type"].astype(str).to_dict()
    with (SOURCE / "prior_type.pkl").open("rb") as handle: pkl_prior = {str(key): str(value) for key, value in pickle.load(handle).items()}
    csv_prior_frame = pd.read_csv(SOURCE / "prior_type.csv", dtype={"pitcher_id": str, "prior_type": str}); csv_prior = csv_prior_frame.set_index("pitcher_id")["prior_type"].astype(str).to_dict(); prior_report = {"expected_count": len(expected_prior), "pkl_count": len(pkl_prior), "csv_count": len(csv_prior), "pkl_matches_recomputed": pkl_prior == expected_prior, "csv_matches_recomputed": csv_prior == expected_prior, "pkl_csv_match": pkl_prior == csv_prior, "cutoff": 2025, "source_max_season": int(raw["season"].max())}
    tm = pd.read_csv(SOURCE / "trackman_prior_features.csv", low_memory=False); tm_source = pd.read_csv(ROOT / "data" / "trackman_history.csv", usecols=["season"]); tm_report = {"rows": len(tm), "min_target_season": int(tm["season"].min()), "max_target_season": int(tm["season"].max()), "target_2025_rows": int(tm["season"].eq(2025).sum()), "source_min_season": int(tm_source["season"].min()), "source_max_season": int(tm_source["season"].max()), "source_strictly_before_2025": bool(int(tm_source["season"].max()) < 2025)}; expected_lookup = {"snapshots": snapshot_report, "prior_type": prior_report, "trackman": tm_report}; stored_lookup = json.loads((OUT / "lookup_snapshot_report.json").read_text(encoding="utf-8")); lookup_diff, lookup_failures = compare_objects(stored_lookup, expected_lookup); check("lookup_snapshot_recomputed", not lookup_failures, {"max_abs_diff": lookup_diff, "failures": lookup_failures})

    code = TRAIN_SCRIPT.read_text(encoding="utf-8"); tm_code = TRACKMAN_SCRIPT.read_text(encoding="utf-8"); code_checks = {"official_train_loaded": 'raw = pd.read_csv(ROOT / "data" / "train.csv"' in code, "max_season_weight_anchor": "int(season.max()) - season" in code, "f_2024_mask": "f_2024_mask = f_mask & (season == 2024)" in code, "recent_v355_uses_f_2024": 'm.fit(x3.loc[f_2024_mask], (y - base3)[f_2024_mask]' in code, "recent_v330_uses_f_2024": code.count('m.fit(x3.loc[f_2024_mask], (y - base3)[f_2024_mask]') >= 2, "f_all_uses_f_mask": "m.fit(x3.loc[f_mask], (y - base3)[f_mask]" in code and "m.fit(x2.loc[f_mask], (y - base2)[f_mask]" in code, "prior_cutoff_2025": "prior_type_table(raw_for_lookup, 2025)" in code, "transition_disabled": '"transition_scale": 0.0' in code, "trackman_strict_past": "hist = player[player.season < target_season]" in tm_code, "trackman_targets_through_2025": "for target_season in range(2019, 2026)" in tm_code}
    basic_manifest = json.loads((SOURCE / "manifest.json").read_text(encoding="utf-8")); binding_files = sorted(path.name for path in SOURCE.iterdir() if path.is_file() and ("audit" in path.name.lower() or "attest" in path.name.lower() or "build_manifest" in path.name.lower())); binding = {"binding_evidence_files": binding_files, "binding_evidence_count": len(binding_files), "manifest_has_train_sha256": any("train" in key.lower() and "sha" in key.lower() for key in basic_manifest), "manifest_has_training_script_sha256": any("script" in key.lower() and "sha" in key.lower() for key in basic_manifest), "per_model_source_binding_available": False}
    parameter_mismatches = expected_components.loc[~expected_components["parameter_match"], ["component", "parameter_mismatches", "expected_thread_count", "actual_thread_count"]].to_dict(orient="records"); package_mismatches = expected_package.loc[~expected_package["zip_local_match"] | expected_package["source_model_match"].eq(False), ["arcname", "zip_local_match", "source_model_match"]].to_dict(orient="records"); semantic = []
    if not all(code_checks.values()): semantic.append("code_intent")
    if not all(feature_checks.values()): semantic.append("feature_groups")
    if any(not value["match"] for value in snapshot_report.values()): semantic.append("snapshots")
    if not all((prior_report["pkl_matches_recomputed"], prior_report["csv_matches_recomputed"], prior_report["pkl_csv_match"])): semantic.append("prior_type")
    if not tm_report["source_strictly_before_2025"] or tm_report["max_target_season"] != 2025: semantic.append("trackman_cutoff")
    issues: list[dict[str, object]] = []
    if parameter_mismatches: issues.append({"code": "CBM_PARAMETER_MISMATCH", "count": len(parameter_mismatches), "details": parameter_mismatches})
    if package_mismatches: issues.append({"code": "PACKAGE_BYTE_MISMATCH", "count": len(package_mismatches), "details": package_mismatches})
    if semantic: issues.append({"code": "SEMANTIC_PROVENANCE_MISMATCH", "count": len(semantic), "details": semantic})
    if not binding["per_model_source_binding_available"]: issues.append({"code": "MISSING_MODEL_INPUT_BINDING", "count": 1, "details": binding})
    provenance = "AUDIT_FAIL_PROVENANCE" if parameter_mismatches or package_mismatches or semantic else ("AUDIT_INCOMPLETE" if not binding["per_model_source_binding_available"] else "AUDIT_VERIFIED")
    counts = {"audited_cbm_count": len(expected_components), "parameter_match_count": int(expected_components["parameter_match"].sum()), "parameter_mismatch_count": int((~expected_components["parameter_match"]).sum()), "source_file_count": len([path for path in SOURCE.iterdir() if path.is_file()]), "candidate_file_count": len([path for path in CANDIDATE.rglob("*") if path.is_file()]), "zip_member_count": len(expected_package), "package_match_count": int(expected_package["zip_local_match"].sum()), "package_mismatch_count": len(package_mismatches), "issue_count": len(issues), "binding_evidence_count": binding["binding_evidence_count"]}
    expected_result = {"experiment_id": EXPERIMENT_ID, "execution_status": "PENDING_AUDIT", "provenance_status": provenance, "source_official_score": 1068.25021, "code_intent_checks": code_checks, "feature_group_checks": feature_checks, "binding_report": binding, "lookup_snapshot_report": expected_lookup, "counts": counts, "issues": issues, "candidate_count": 0, "actual_leaf_count": 0, "gate_checks_count": 0, "model_count_created": 0, "test_read_performed": False, "test_inference_performed": False, "training_performed": False, "zip_created": False}; stored_result = {key: value for key, value in result.items() if key != "elapsed_seconds"}; result_diff, result_failures = compare_objects(stored_result, expected_result); check("result_recomputed", not result_failures, {"max_abs_diff": result_diff, "failures": result_failures}); check("manifest_counts", all(audit[key] == counts[key] for key in counts), {"audit": {key: audit[key] for key in counts}, "expected": counts})
    markdown = (OUT / "result.md").read_text(encoding="utf-8"); match = re.search(r"<!-- RESULT_JSON_BEGIN\n(.*?)\nRESULT_JSON_END -->", markdown, re.DOTALL); embedded = json.loads(match.group(1)) if match else None; check("json_markdown_embedded", embedded == expected_result, embedded == expected_result)
    validation_status = "AUDIT_VERIFIED" if not mismatches else "FAIL"; report = {"experiment_id": EXPERIMENT_ID, "validation_status": validation_status, "provenance_status": provenance if validation_status == "AUDIT_VERIFIED" else "AUDIT_INCOMPLETE", "checked_count": len(checks), "passed_count": sum(bool(row["passed"]) for row in checks), "mismatch_count": len(mismatches), "mismatches": mismatches, "candidate_count": 0, "actual_leaf_count": 0, "gate_checks_count": 0, "model_count_created": 0, "counts": counts, "issues": issues, "checks": checks}; report_path = OUT / "validation_report.json"; write_json(report_path, report)
    attestation = {"experiment_id": EXPERIMENT_ID, "status": validation_status, "provenance_status": report["provenance_status"], "manifest_sha256": sha256_path(OUT / "audit_manifest.json"), "validation_report_sha256": sha256_path(report_path), "validator_sha256": sha256_path(Path(__file__).resolve()), "checked_count": report["checked_count"], "passed_count": report["passed_count"], "mismatch_count": report["mismatch_count"], "candidate_count": 0, "actual_leaf_count": 0, "gate_checks_count": 0, "model_count_created": 0, **counts}; write_json(OUT / "audit_attestation.json", attestation); print(json.dumps(attestation, indent=2))
    if mismatches: raise SystemExit(1)


if __name__ == "__main__": main()
