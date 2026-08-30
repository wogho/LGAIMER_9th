#!/usr/bin/env python3
"""Exact inference-semantic JSON structure comparison for 56 CBM pairs."""
from __future__ import annotations

import hashlib
import json
import tempfile
import time
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    return {"sha256": sha256_file(path), "size": path.stat().st_size}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n")


def semantic_payload(model_json: dict[str, Any]) -> dict[str, Any]:
    return {
        "features_info": model_json.get("features_info"),
        "ctr_data": model_json.get("ctr_data"),
        "oblivious_trees": model_json.get("oblivious_trees"),
        "scale_and_bias": model_json.get("scale_and_bias"),
        "class_params": model_json.get("model_info", {}).get("class_params"),
    }


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def first_mismatch(left: Any, right: Any, path: str = "$") -> dict[str, Any] | None:
    if type(left) is not type(right):
        return {"path": path, "reason": "type", "original": type(left).__name__, "repro": type(right).__name__}
    if isinstance(left, dict):
        left_keys, right_keys = set(left), set(right)
        if left_keys != right_keys:
            return {
                "path": path, "reason": "keys", "missing_in_repro": sorted(left_keys - right_keys),
                "extra_in_repro": sorted(right_keys - left_keys),
            }
        for key in sorted(left):
            mismatch = first_mismatch(left[key], right[key], f"{path}.{key}")
            if mismatch:
                return mismatch
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return {"path": path, "reason": "length", "original": len(left), "repro": len(right)}
        for index, (left_value, right_value) in enumerate(zip(left, right)):
            mismatch = first_mismatch(left_value, right_value, f"{path}[{index}]")
            if mismatch:
                return mismatch
        return None
    if left != right:
        return {"path": path, "reason": "value", "original": repr(left)[:500], "repro": repr(right)[:500]}
    return None


def counts(payload: dict[str, Any]) -> dict[str, int]:
    feature_info = payload.get("features_info") or {}
    trees = payload.get("oblivious_trees") or []
    return {
        "tree_count": len(trees),
        "split_count": sum(len(tree.get("splits", [])) for tree in trees),
        "leaf_value_count": sum(len(tree.get("leaf_values", [])) for tree in trees),
        "leaf_weight_count": sum(len(tree.get("leaf_weights", [])) for tree in trees),
        "float_feature_count": len(feature_info.get("float_features", [])),
        "categorical_feature_count": len(feature_info.get("categorical_features", [])),
        "ctr_definition_count": len(feature_info.get("ctrs", [])),
        "ctr_table_count": len(payload.get("ctr_data") or {}),
    }


def load_json(model_path: Path, json_path: Path) -> tuple[cb.CatBoost, dict[str, Any]]:
    model = cb.CatBoost()
    model.load_model(model_path)
    model.save_model(json_path, format="json")
    return model, json.loads(json_path.read_text())


def verify_input_binding(binding: dict[str, Any]) -> list[dict[str, Any]]:
    mismatches: list[dict[str, Any]] = []
    groups = ["original_models", "repro_models", "upstream_artifacts", "champion_zip", "governing_documents"]
    for group in groups:
        records = binding[group] if isinstance(binding[group], list) else [binding[group]]
        for record in records:
            path = ROOT / record["path"]
            actual = sha256_file(path) if path.is_file() else None
            if actual != record["sha256"] or (path.is_file() and path.stat().st_size != record["size"]):
                mismatches.append({"path": record["path"], "expected": record["sha256"], "actual": actual})
    return mismatches


def main() -> None:
    started = time.time()
    preflight = json.loads((OUT / "preflight_report.json").read_text())
    contract = json.loads((OUT / "audit_contract.json").read_text())
    binding = json.loads((OUT / "input_binding.json").read_text())
    if preflight.get("status") != "AUDIT_VERIFIED" or preflight.get("fail_count") != 0:
        raise RuntimeError("Preflight is not AUDIT_VERIFIED")
    binding_mismatches = verify_input_binding(binding)
    if binding_mismatches:
        raise RuntimeError(f"Input binding changed after preflight: {binding_mismatches}")

    original_names = {path.name for path in ORIGINAL.glob("*.cbm")}
    repro_names = {path.name for path in REPRO.glob("*.cbm")}
    if original_names != repro_names or len(original_names) != contract["expected_model_pair_count"]:
        raise RuntimeError("Model name set changed after preflight")

    rows: list[dict[str, Any]] = []
    mismatches: list[dict[str, Any]] = []
    json_export_count = 0
    with tempfile.TemporaryDirectory(prefix="ref4_039a_") as temp_dir:
        temp = Path(temp_dir)
        for index, name in enumerate(sorted(original_names), 1):
            original_model, original_json = load_json(ORIGINAL / name, temp / "original.json")
            repro_model, repro_json = load_json(REPRO / name, temp / "repro.json")
            json_export_count += 2
            original_payload = semantic_payload(original_json)
            repro_payload = semantic_payload(repro_json)
            original_bytes = canonical_bytes(original_payload)
            repro_bytes = canonical_bytes(repro_payload)
            mismatch = first_mismatch(original_payload, repro_payload)
            original_counts = counts(original_payload)
            repro_counts = counts(repro_payload)
            original_metadata = original_json.get("model_info", {})
            repro_metadata = repro_json.get("model_info", {})
            row = {
                "model": name,
                "semantic_match": mismatch is None and original_bytes == repro_bytes,
                "original_semantic_sha256": sha256_bytes(original_bytes),
                "repro_semantic_sha256": sha256_bytes(repro_bytes),
                "canonical_size_original": len(original_bytes),
                "canonical_size_repro": len(repro_bytes),
                "loaded_tree_count_original": int(original_model.tree_count_),
                "loaded_tree_count_repro": int(repro_model.tree_count_),
                "loaded_feature_count_original": len(original_model.feature_names_),
                "loaded_feature_count_repro": len(repro_model.feature_names_),
                "model_kind": "classifier" if original_payload["class_params"] is not None else "regressor",
                "first_mismatch_path": "" if mismatch is None else mismatch["path"],
                "original_model_info_sha256": sha256_bytes(canonical_bytes(original_metadata)),
                "repro_model_info_sha256": sha256_bytes(canonical_bytes(repro_metadata)),
                "model_info_exact_match": original_metadata == repro_metadata,
                **{f"original_{key}": value for key, value in original_counts.items()},
                **{f"repro_{key}": value for key, value in repro_counts.items()},
            }
            rows.append(row)
            if mismatch is not None or original_bytes != repro_bytes:
                mismatches.append({"model": name, "first_mismatch": mismatch, "row": row})
            if index % 8 == 0 or index == len(original_names):
                print(f"[{EXPERIMENT_ID}] compared {index}/{len(original_names)} pairs", flush=True)

    inventory = pd.DataFrame(rows).sort_values("model").reset_index(drop=True)
    inventory.to_csv(OUT / "structure_inventory.csv", index=False)
    write_json(OUT / "structure_mismatches.json", {
        "experiment_id": EXPERIMENT_ID, "mismatch_count": len(mismatches), "mismatches": mismatches,
    })
    structure_match_count = int(inventory["semantic_match"].sum())
    structure_status = "STRUCTURE_EQUIVALENT" if structure_match_count == 56 and not mismatches else "AUDIT_FAIL_STRUCTURE"
    result = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "execution_status": "PENDING_VALIDATION",
        "structure_status": structure_status,
        "model_pair_count": len(inventory),
        "json_export_count": json_export_count,
        "semantic_field_count": len(contract["semantic_fields"]),
        "structure_match_count": structure_match_count,
        "structure_mismatch_count": len(mismatches),
        "model_info_exact_match_count": int(inventory["model_info_exact_match"].sum()),
        "candidate_count": 0,
        "actual_leaf_count": 0,
        "gate_count": 0,
        "test_read_performed": False,
        "test_inference_performed": False,
        "training_performed": False,
        "model_modified": False,
        "candidate_created": False,
        "zip_created": False,
        "submission_performed": False,
        "official_score_preserved": 1068.25021,
        "champion_zip_sha256": sha256_file(ZIP),
        "elapsed_seconds": time.time() - started,
    }
    write_json(OUT / "result.json", result)
    embedded = json.dumps(result, ensure_ascii=False, sort_keys=True)
    (OUT / "result.md").write_text(
        f"# {EXPERIMENT_ID}\n\n"
        f"- execution status: `{result['execution_status']}`\n"
        f"- structure status: `{structure_status}`\n"
        f"- model pairs: `{len(inventory)}`\n"
        f"- JSON exports: `{json_export_count}`\n"
        f"- match/mismatch: `{structure_match_count}/{len(mismatches)}`\n"
        "- candidate/leaf/gate: `0/0/0`\n"
        "- test/training/model-write/ZIP: `false/false/false/false`\n\n"
        f"<!-- RESULT_JSON_BEGIN\n{embedded}\nRESULT_JSON_END -->\n"
    )

    artifact_paths = [
        ROOT / "scripts" / "preflight_ref4_structure_039a.py",
        ROOT / "scripts" / "run_ref4_structure_039a.py",
        ROOT / "scripts" / "verify_ref4_structure_039a.py",
        OUT / "audit_contract.json", OUT / "contract_snapshot.md", OUT / "input_binding.json",
        OUT / "preflight_report.json", OUT / "preflight_report.md",
        OUT / "structure_inventory.csv", OUT / "structure_mismatches.json",
        OUT / "result.json", OUT / "result.md", ZIP,
    ]
    artifact_paths.extend(sorted(ORIGINAL.glob("*.cbm")))
    artifact_paths.extend(sorted(REPRO.glob("*.cbm")))
    records = {str(path.relative_to(ROOT)): file_record(path) for path in artifact_paths}
    audit = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "status": "PENDING_VALIDATION",
        "artifact_count": len(records),
        "artifacts": records,
        "model_pair_count": len(inventory),
        "json_export_count": json_export_count,
        "structure_match_count": structure_match_count,
        "structure_mismatch_count": len(mismatches),
        "candidate_count": 0,
        "leaf_count": 0,
        "gate_count": 0,
        "test_read_performed": False,
        "test_inference_performed": False,
        "training_performed": False,
        "zip_created": False,
    }
    write_json(OUT / "audit_manifest.json", audit)
    print(json.dumps({
        "execution_status": result["execution_status"], "structure_status": structure_status,
        "model_pairs": len(inventory), "json_exports": json_export_count,
        "matches": structure_match_count, "mismatches": len(mismatches),
        "elapsed_seconds": result["elapsed_seconds"],
    }, indent=2))


if __name__ == "__main__":
    main()
