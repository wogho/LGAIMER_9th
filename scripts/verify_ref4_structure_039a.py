#!/usr/bin/env python3
"""Independent full re-export verifier for 039A structure comparison."""
from __future__ import annotations

import hashlib
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import catboost as cb
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ID = "REF4-CHAMPION-REPRO-STRUCTURE-039A"
ORIGINAL = ROOT / "model" / "REF4-CHAMPION-STACK-030"
REPRO = ROOT / "model" / "REF4-CHAMPION-REPRO-038A"
OUT = ROOT / "model" / EXPERIMENT_ID
ZIP = ROOT / "output" / "submit_ref4_champion_030.zip"
EXPECTED_ZIP_SHA256 = "ca6af4bdf26d4bc79d503cdc6a09b0057a7ea9bdd77e29012f9e459a50cd66b8"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def normalize(model_json: dict[str, Any]) -> dict[str, Any]:
    info = model_json.get("model_info", {})
    return {
        "features_info": model_json.get("features_info"),
        "ctr_data": model_json.get("ctr_data"),
        "oblivious_trees": model_json.get("oblivious_trees"),
        "scale_and_bias": model_json.get("scale_and_bias"),
        "class_params": info.get("class_params"),
    }


def recursive_difference(left: Any, right: Any, path: str = "$") -> dict[str, Any] | None:
    if type(left) is not type(right):
        return {"path": path, "reason": "type", "left": type(left).__name__, "right": type(right).__name__}
    if isinstance(left, dict):
        if set(left) != set(right):
            return {"path": path, "reason": "keys", "left_only": sorted(set(left) - set(right)), "right_only": sorted(set(right) - set(left))}
        for key in sorted(left):
            difference = recursive_difference(left[key], right[key], f"{path}.{key}")
            if difference:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return {"path": path, "reason": "length", "left": len(left), "right": len(right)}
        for index in range(len(left)):
            difference = recursive_difference(left[index], right[index], f"{path}[{index}]")
            if difference:
                return difference
        return None
    if left != right:
        return {"path": path, "reason": "value", "left": repr(left)[:500], "right": repr(right)[:500]}
    return None


def payload_counts(payload: dict[str, Any]) -> dict[str, int]:
    info = payload.get("features_info") or {}
    trees = payload.get("oblivious_trees") or []
    return {
        "tree_count": len(trees),
        "split_count": sum(len(tree.get("splits", [])) for tree in trees),
        "leaf_value_count": sum(len(tree.get("leaf_values", [])) for tree in trees),
        "leaf_weight_count": sum(len(tree.get("leaf_weights", [])) for tree in trees),
        "float_feature_count": len(info.get("float_features", [])),
        "categorical_feature_count": len(info.get("categorical_features", [])),
        "ctr_definition_count": len(info.get("ctrs", [])),
        "ctr_table_count": len(payload.get("ctr_data") or {}),
    }


def export(path: Path, destination: Path) -> tuple[cb.CatBoost, dict[str, Any]]:
    model = cb.CatBoost()
    model.load_model(path)
    model.save_model(destination, format="json")
    return model, json.loads(destination.read_text())


def add_check(checks: list[dict[str, Any]], check_id: str, passed: bool, actual: Any) -> None:
    checks.append({"check_id": check_id, "checked": True, "status": "PASS" if passed else "FAIL", "actual": actual})


def main() -> None:
    manifest = json.loads((OUT / "audit_manifest.json").read_text())
    contract = json.loads((OUT / "audit_contract.json").read_text())
    preflight = json.loads((OUT / "preflight_report.json").read_text())
    result = json.loads((OUT / "result.json").read_text())
    checks: list[dict[str, Any]] = []

    artifact_mismatches = []
    for relative, expected in manifest.get("artifacts", {}).items():
        path = ROOT / relative
        actual_hash = sha256_file(path) if path.is_file() else None
        actual_size = path.stat().st_size if path.is_file() else None
        if actual_hash != expected["sha256"] or actual_size != expected["size"]:
            artifact_mismatches.append({"path": relative, "expected_hash": expected["sha256"], "actual_hash": actual_hash})
    add_check(checks, "manifest_artifact_hashes", not artifact_mismatches, artifact_mismatches)
    add_check(checks, "manifest_artifact_count", manifest.get("artifact_count") == len(manifest.get("artifacts", {})), {
        "declared": manifest.get("artifact_count"), "actual": len(manifest.get("artifacts", {}))
    })
    add_check(checks, "preflight_verified", preflight.get("status") == "AUDIT_VERIFIED" and preflight.get("fail_count") == 0, {
        "status": preflight.get("status"), "fail_count": preflight.get("fail_count")
    })
    add_check(checks, "contract_counts", contract.get("expected_model_pair_count") == 56 and contract.get("expected_json_export_count") == 112 and len(contract.get("semantic_fields", [])) == 5, {
        "pairs": contract.get("expected_model_pair_count"), "exports": contract.get("expected_json_export_count"),
        "semantic_fields": len(contract.get("semantic_fields", []))
    })

    original_names = {path.name for path in ORIGINAL.glob("*.cbm")}
    repro_names = {path.name for path in REPRO.glob("*.cbm")}
    add_check(checks, "source_model_sets", original_names == repro_names and len(original_names) == 56, {
        "original": len(original_names), "repro": len(repro_names),
        "left_only": sorted(original_names - repro_names), "right_only": sorted(repro_names - original_names)
    })

    rows: list[dict[str, Any]] = []
    independent_differences: list[dict[str, Any]] = []
    export_count = 0
    with tempfile.TemporaryDirectory(prefix="ref4_039a_verify_") as temporary:
        temp = Path(temporary)
        for index, name in enumerate(sorted(original_names & repro_names), 1):
            original_model, original_json = export(ORIGINAL / name, temp / "left.json")
            repro_model, repro_json = export(REPRO / name, temp / "right.json")
            export_count += 2
            left = normalize(original_json)
            right = normalize(repro_json)
            left_bytes, right_bytes = canonical_bytes(left), canonical_bytes(right)
            difference = recursive_difference(left, right)
            left_counts, right_counts = payload_counts(left), payload_counts(right)
            left_metadata, right_metadata = original_json.get("model_info", {}), repro_json.get("model_info", {})
            row = {
                "model": name,
                "semantic_match": difference is None and left_bytes == right_bytes,
                "original_semantic_sha256": sha256_bytes(left_bytes),
                "repro_semantic_sha256": sha256_bytes(right_bytes),
                "canonical_size_original": len(left_bytes),
                "canonical_size_repro": len(right_bytes),
                "loaded_tree_count_original": int(original_model.tree_count_),
                "loaded_tree_count_repro": int(repro_model.tree_count_),
                "loaded_feature_count_original": len(original_model.feature_names_),
                "loaded_feature_count_repro": len(repro_model.feature_names_),
                "model_kind": "classifier" if left["class_params"] is not None else "regressor",
                "first_mismatch_path": "" if difference is None else difference["path"],
                "original_model_info_sha256": sha256_bytes(canonical_bytes(left_metadata)),
                "repro_model_info_sha256": sha256_bytes(canonical_bytes(right_metadata)),
                "model_info_exact_match": left_metadata == right_metadata,
                **{f"original_{key}": value for key, value in left_counts.items()},
                **{f"repro_{key}": value for key, value in right_counts.items()},
            }
            rows.append(row)
            if not row["semantic_match"]:
                independent_differences.append({"model": name, "first_difference": difference})
            if index % 8 == 0 or index == len(original_names & repro_names):
                print(f"[{EXPERIMENT_ID} verifier] recomputed {index}/{len(original_names & repro_names)} pairs", flush=True)

    verification = pd.DataFrame(rows).sort_values("model").reset_index(drop=True)
    verification.to_csv(OUT / "verification_inventory.csv", index=False)
    executor = pd.read_csv(OUT / "structure_inventory.csv", keep_default_na=False)
    executor = executor.sort_values("model").reset_index(drop=True)
    inventory_mismatches: list[dict[str, Any]] = []
    if list(executor.columns) != list(verification.columns) or len(executor) != len(verification):
        inventory_mismatches.append({
            "reason": "shape_or_columns", "executor_shape": list(executor.shape),
            "verification_shape": list(verification.shape), "executor_columns": list(executor.columns),
            "verification_columns": list(verification.columns),
        })
    else:
        for column in executor.columns:
            if column in ("semantic_match", "model_info_exact_match"):
                left_values = executor[column].astype(str).str.lower().tolist()
                right_values = verification[column].astype(str).str.lower().tolist()
            elif column in ("model", "model_kind", "first_mismatch_path", "original_semantic_sha256", "repro_semantic_sha256", "original_model_info_sha256", "repro_model_info_sha256"):
                left_values = executor[column].astype(str).tolist()
                right_values = verification[column].astype(str).tolist()
            else:
                left_values = pd.to_numeric(executor[column]).astype("int64").tolist()
                right_values = pd.to_numeric(verification[column]).astype("int64").tolist()
            if left_values != right_values:
                positions = [index for index, (left, right) in enumerate(zip(left_values, right_values)) if left != right]
                inventory_mismatches.append({"column": column, "mismatch_count": len(positions), "first_positions": positions[:10]})
    add_check(checks, "executor_inventory_recomputed", not inventory_mismatches, inventory_mismatches)
    add_check(checks, "json_export_count", export_count == 112, export_count)
    match_count = int(verification["semantic_match"].sum())
    add_check(checks, "semantic_structure_56", match_count == 56 and not independent_differences, {
        "match_count": match_count, "mismatch_count": len(independent_differences), "differences": independent_differences
    })
    add_check(checks, "loaded_tree_feature_parity", bool(((verification.loaded_tree_count_original == verification.loaded_tree_count_repro) & (verification.loaded_feature_count_original == verification.loaded_feature_count_repro)).all()), {
        "tree_mismatch_count": int((verification.loaded_tree_count_original != verification.loaded_tree_count_repro).sum()),
        "feature_mismatch_count": int((verification.loaded_feature_count_original != verification.loaded_feature_count_repro).sum()),
    })

    result_md = (OUT / "result.md").read_text()
    begin, end = "<!-- RESULT_JSON_BEGIN\n", "\nRESULT_JSON_END -->"
    embedded = None
    if result_md.count(begin) == 1 and result_md.count(end) == 1:
        embedded = json.loads(result_md.split(begin, 1)[1].split(end, 1)[0])
    add_check(checks, "result_markdown_parity", embedded == result, "exact" if embedded == result else "mismatch")
    expected_structure_status = "STRUCTURE_EQUIVALENT" if match_count == 56 and not independent_differences else "AUDIT_FAIL_STRUCTURE"
    result_expected = (
        result.get("model_pair_count") == 56
        and result.get("json_export_count") == 112
        and result.get("semantic_field_count") == 5
        and result.get("structure_match_count") == match_count
        and result.get("structure_mismatch_count") == len(independent_differences)
        and result.get("structure_status") == expected_structure_status
    )
    add_check(checks, "result_json_recomputed", result_expected, {
        "recorded_status": result.get("structure_status"), "expected_status": expected_structure_status,
        "recorded_match": result.get("structure_match_count"), "expected_match": match_count,
    })
    zero_counts = all(result.get(key) == 0 for key in ("candidate_count", "actual_leaf_count", "gate_count"))
    false_operations = all(result.get(key) is False for key in (
        "test_read_performed", "test_inference_performed", "training_performed", "model_modified",
        "candidate_created", "zip_created", "submission_performed",
    ))
    add_check(checks, "zero_candidate_leaf_gate", zero_counts, {key: result.get(key) for key in ("candidate_count", "actual_leaf_count", "gate_count")})
    add_check(checks, "forbidden_operations_absent", false_operations, {key: result.get(key) for key in (
        "test_read_performed", "test_inference_performed", "training_performed", "model_modified", "candidate_created", "zip_created", "submission_performed"
    )})
    candidate_exists = (ROOT / "candidate" / EXPERIMENT_ID).exists()
    zip_matches = sorted(str(path.relative_to(ROOT)) for path in (ROOT / "output").glob("*039A*.zip"))
    add_check(checks, "filesystem_no_candidate_or_zip", not candidate_exists and not zip_matches, {
        "candidate_exists": candidate_exists, "zip_matches": zip_matches
    })
    champion_hash = sha256_file(ZIP)
    add_check(checks, "champion_zip_preserved", champion_hash == EXPECTED_ZIP_SHA256, champion_hash)
    add_check(checks, "manifest_counts_recomputed", manifest.get("model_pair_count") == 56 and manifest.get("json_export_count") == 112 and manifest.get("structure_match_count") == match_count and manifest.get("structure_mismatch_count") == len(independent_differences), {
        "pairs": manifest.get("model_pair_count"), "exports": manifest.get("json_export_count"),
        "matches": manifest.get("structure_match_count"), "mismatches": manifest.get("structure_mismatch_count")
    })

    failed_checks = [row["check_id"] for row in checks if row["status"] != "PASS"]
    if expected_structure_status != "STRUCTURE_EQUIVALENT":
        final_status = "AUDIT_FAIL_STRUCTURE"
    elif failed_checks:
        final_status = "AUDIT_FAIL_PROVENANCE"
    else:
        final_status = "AUDIT_VERIFIED"
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": final_status,
        "structure_status": expected_structure_status,
        "check_count": len(checks),
        "pass_count": len(checks) - len(failed_checks),
        "fail_count": len(failed_checks),
        "mismatch_count": len(failed_checks),
        "failed_checks": failed_checks,
        "model_pair_count": len(verification),
        "json_export_count": export_count,
        "structure_match_count": match_count,
        "structure_mismatch_count": len(independent_differences),
        "candidate_count": 0,
        "leaf_count": 0,
        "gate_count": 0,
        "test_read_performed": False,
        "test_inference_performed": False,
        "training_performed": False,
        "zip_created": False,
        "checks": checks,
        "artifacts": {
            "audit_manifest_sha256": sha256_file(OUT / "audit_manifest.json"),
            "structure_inventory_sha256": sha256_file(OUT / "structure_inventory.csv"),
            "verification_inventory_sha256": sha256_file(OUT / "verification_inventory.csv"),
            "result_sha256": sha256_file(OUT / "result.json"),
        },
    }
    write_path = OUT / "validation_report.json"
    write_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    validator_path = Path(__file__).resolve()
    attestation = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": final_status,
        "structure_status": expected_structure_status,
        "audit_manifest_sha256": sha256_file(OUT / "audit_manifest.json"),
        "validation_report_sha256": sha256_file(write_path),
        "validator_sha256": sha256_file(validator_path),
        "verification_inventory_sha256": sha256_file(OUT / "verification_inventory.csv"),
        "check_count": len(checks),
        "pass_count": len(checks) - len(failed_checks),
        "fail_count": len(failed_checks),
        "mismatch_count": len(failed_checks),
        "model_pair_count": len(verification),
        "json_export_count": export_count,
        "structure_match_count": match_count,
        "structure_mismatch_count": len(independent_differences),
        "candidate_count": 0,
        "leaf_count": 0,
        "gate_count": 0,
        "test_read_performed": False,
        "test_inference_performed": False,
        "training_performed": False,
        "zip_created": False,
    }
    (OUT / "audit_attestation.json").write_text(json.dumps(attestation, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    print(json.dumps({
        "status": final_status, "structure_status": expected_structure_status,
        "checks": len(checks), "passed": len(checks) - len(failed_checks),
        "mismatches": len(failed_checks), "model_pairs": len(verification),
        "json_exports": export_count, "structure_matches": match_count,
        "structure_mismatches": len(independent_differences),
    }, indent=2))
    if final_status != "AUDIT_VERIFIED":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
