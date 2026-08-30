#!/usr/bin/env python3
"""Build the isolated, deterministic REGIME-R submission ZIP."""

from __future__ import annotations

import hashlib
import json
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SCRIPT = ROOT / "candidate" / "selective_submission" / "script.py"
SOURCE_REQUIREMENTS = ROOT / "candidate" / "selective_submission" / "requirements.txt"
E2E_CONTRACT = ROOT / "model" / "REGIME-R-001-FINAL-E2E" / "candidate_contract.json"
STAGING = ROOT / "candidate" / "regime_r_submission"
OUTPUT_ZIP = ROOT / "output" / "candidates" / "submit_regime_r_candidate.zip"
MANIFEST = ROOT / "output" / "candidates" / "submit_regime_r_candidate_manifest.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def replace_once(text: str, old: str, new: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"source script replacement count must be 1, got {count}: {old[:60]!r}")
    return text.replace(old, new)


def build_script() -> str:
    text = SOURCE_SCRIPT.read_text(encoding="utf-8")
    text = replace_once(
        text,
        '''    expected_rule = {
        "F": {"lightgbm": 0.0, "catboost": 1.0},
        "R": {"lightgbm": 0.5, "catboost": 0.5},
    }''',
        '''    expected_rule = {
        "F": {"lightgbm": 0.0, "catboost": 1.0},
        "R": {"lightgbm": 0.75, "catboost": 0.25},
    }''',
    )
    text = replace_once(
        text,
        '''    if contract.get("tree_counts") != {"lightgbm": 100, "catboost": 259}:
        raise ValueError("두 모델의 tree 수 계약이 다릅니다")''',
        '''    if contract.get("tree_counts") != {"lightgbm": 110, "catboost": 259}:
        raise ValueError("두 모델의 tree 수 계약이 다릅니다")
    calibration = contract.get("calibration", {})
    if calibration.get("method") != "platt_logit" or calibration.get("scope") != "all_rows":
        raise ValueError("전역 Platt 보정 계약이 다릅니다")''',
    )
    old_function = '''def apply_selective_rule(game_type, pred_lightgbm, pred_catboost):
    game_type = pd.Series(game_type, copy=False).astype("string").to_numpy(dtype=str)
    pred_lightgbm = np.asarray(pred_lightgbm, dtype=np.float64)
    pred_catboost = np.asarray(pred_catboost, dtype=np.float64)
    if not (len(game_type) == len(pred_lightgbm) == len(pred_catboost)):
        raise ValueError("선택 규칙 입력 행 수가 다릅니다")
    unexpected = sorted(set(game_type) - {"F", "R"})
    if unexpected:
        raise ValueError(f"지원하지 않는 game_type입니다: {unexpected}")
    if not np.isfinite(pred_lightgbm).all() or not np.isfinite(pred_catboost).all():
        raise ValueError("단일 모델 예측에 NaN 또는 Inf가 있습니다")
    prediction = np.where(
        game_type == "F",
        pred_catboost,
        0.5 * pred_lightgbm + 0.5 * pred_catboost,
    )
    if not np.isfinite(prediction).all() or not (
        (prediction >= 0.0) & (prediction <= 1.0)
    ).all():
        raise ValueError("선택형 예측이 확률 범위를 벗어났습니다")
    return prediction
'''
    new_function = '''def apply_regime_rule(game_type, pred_lightgbm, pred_catboost, calibration):
    game_type = pd.Series(game_type, copy=False).astype("string").to_numpy(dtype=str)
    pred_lightgbm = np.asarray(pred_lightgbm, dtype=np.float64)
    pred_catboost = np.asarray(pred_catboost, dtype=np.float64)
    if not (len(game_type) == len(pred_lightgbm) == len(pred_catboost)):
        raise ValueError("레짐 규칙 입력 행 수가 다릅니다")
    unexpected = sorted(set(game_type) - {"F", "R"})
    if unexpected:
        raise ValueError(f"지원하지 않는 game_type입니다: {unexpected}")
    if not np.isfinite(pred_lightgbm).all() or not np.isfinite(pred_catboost).all():
        raise ValueError("단일 모델 예측에 NaN 또는 Inf가 있습니다")
    raw = np.where(
        game_type == "F",
        pred_catboost,
        0.75 * pred_lightgbm + 0.25 * pred_catboost,
    )
    epsilon = float(calibration["clip_epsilon"])
    clipped = np.clip(raw, epsilon, 1.0 - epsilon)
    logit = np.log(clipped / (1.0 - clipped))
    z = float(calibration["coefficient"]) * logit + float(calibration["intercept"])
    prediction = 1.0 / (1.0 + np.exp(-z))
    if not np.isfinite(prediction).all() or not (
        (prediction >= 0.0) & (prediction <= 1.0)
    ).all():
        raise ValueError("REGIME-R 예측이 확률 범위를 벗어났습니다")
    return prediction
'''
    text = replace_once(text, old_function, new_function)
    text = replace_once(
        text,
        "prediction = apply_selective_rule(game_type, pred_lightgbm, pred_catboost)",
        "prediction = apply_regime_rule(game_type, pred_lightgbm, pred_catboost, contract[\"calibration\"])",
    )
    return text


def build_contract(source: dict) -> dict:
    contract = {
        "schema_version": 1,
        "experiment_id": source["experiment_id"],
        "future_season": source["future_season"],
        "feature_count": source["feature_count"],
        "selection_rule": {
            "F": {"lightgbm": 0.0, "catboost": 1.0},
            "R": {"lightgbm": 0.75, "catboost": 0.25},
        },
        "calibration": source["calibration"],
        "tree_counts": {"lightgbm": 110, "catboost": 259},
        "model_files": {
            "lightgbm": "lightgbm_r_only_model.txt",
            "catboost": "catboost_model.cbm",
            "feature_columns": "feature_columns.json",
        },
        "model_sha256": {
            "lightgbm": source["model_sha256"]["lightgbm_r_only"],
            "catboost": source["model_sha256"]["catboost"],
            "feature_columns": source["model_sha256"]["feature_columns"],
        },
        "source_contract_sha256": source["source_contract_sha256"],
        "test_distribution_used": False,
        "external_data_used": False,
        "active_model_sync": False,
    }
    return contract


def deterministic_zip(files: list[tuple[Path, str]]) -> None:
    OUTPUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for source, arcname in sorted(files, key=lambda item: item[1]):
            info = zipfile.ZipInfo(arcname, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, source.read_bytes())


def main() -> None:
    source = json.loads(E2E_CONTRACT.read_text(encoding="utf-8"))
    if source.get("active_model_sync") is not False:
        raise ValueError("격리 E2E 계약이 아닙니다")
    script_text = build_script()
    contract = build_contract(source)
    model_dir = STAGING / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "script": STAGING / "script.py",
        "requirements": STAGING / "requirements.txt",
        "contract": model_dir / "ensemble_contract.json",
        "lightgbm": model_dir / contract["model_files"]["lightgbm"],
        "catboost": model_dir / contract["model_files"]["catboost"],
        "feature_columns": model_dir / contract["model_files"]["feature_columns"],
    }
    paths["script"].write_text(script_text, encoding="utf-8")
    shutil.copyfile(SOURCE_REQUIREMENTS, paths["requirements"])
    paths["contract"].write_text(json.dumps(contract, ensure_ascii=False, indent=2), encoding="utf-8")
    shutil.copyfile(source["model_files"]["lightgbm_r_only"], paths["lightgbm"])
    shutil.copyfile(source["model_files"]["catboost"], paths["catboost"])
    shutil.copyfile(source["model_files"]["feature_columns"], paths["feature_columns"])
    for key in ("lightgbm", "catboost", "feature_columns"):
        if sha256_file(paths[key]) != contract["model_sha256"][key]:
            raise ValueError(f"staged {key} hash mismatch")
    files = [
        (paths["script"], "script.py"),
        (paths["requirements"], "requirements.txt"),
        (paths["contract"], "model/ensemble_contract.json"),
        (paths["lightgbm"], f"model/{paths['lightgbm'].name}"),
        (paths["catboost"], f"model/{paths['catboost'].name}"),
        (paths["feature_columns"], f"model/{paths['feature_columns'].name}"),
    ]
    deterministic_zip(files)
    with zipfile.ZipFile(OUTPUT_ZIP) as archive:
        members = archive.namelist()
        if members != sorted(arcname for _, arcname in files):
            raise ValueError("ZIP member contract mismatch")
        if any(name.startswith("/") or ".." in Path(name).parts for name in members):
            raise ValueError("ZIP contains unsafe member")
    manifest = {
        "experiment_id": source["experiment_id"],
        "zip_path": str(OUTPUT_ZIP),
        "zip_sha256": sha256_file(OUTPUT_ZIP),
        "zip_bytes": OUTPUT_ZIP.stat().st_size,
        "members": members,
        "member_sha256": {arcname: sha256_file(path) for path, arcname in files},
        "active_model_sync": False,
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
