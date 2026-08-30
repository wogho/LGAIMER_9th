#!/usr/bin/env python3
"""Build a non-circular audit manifest for the fixed transition experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "model" / "REGIME-RCAPACITY-TRANSITION-018"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_entry(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.relative_to(ROOT)),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, nargs="+", required=True)
    parser.add_argument("--name", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seasons = sorted(set(args.seasons))
    if not seasons or any(season not in (2022, 2023, 2024) for season in seasons):
        raise ValueError("seasons must be drawn from 2022, 2023, 2024")
    common_files = [
        ROOT / "data" / "train.csv",
        ROOT / "data" / "trackman_history.csv",
        ROOT / "model" / "TRACKMAN-MAP-004" / "pitcher_id_map.csv",
        ROOT / "model" / "COMBO-RESID3-OOF-007" / "oof_predictions.csv",
        ROOT / "scripts" / "run_regime_rcapacity_transition_018.py",
        ROOT / "scripts" / "build_regime_rcapacity_audit_018.py",
        ROOT / "scripts" / "verify_regime_rcapacity_transition_018.py",
        ROOT / "scripts" / "screen_regime_split_015.py",
        ROOT / "scripts" / "build_combo_full_candidate_002.py",
        ROOT / "scripts" / "screen_trackman_context_003.py",
        ROOT / "src" / "asof_state_features.py",
        ROOT / "src" / "target_aggregates.py",
        ROOT / "output" / "submit_combo_spread_005.zip",
        ROOT / "output" / "candidates" / "submit_regime_r_candidate.zip",
        ROOT / "output" / "submit_final_selective.zip",
    ]
    fold_files: dict[str, dict[str, dict[str, object]]] = {}
    for season in seasons:
        if season == 2022:
            fold_dir = ROOT / "model" / "REGIME-SPLIT-015-RCAPACITY-EXCLUDE2020"
            fold_files[str(season)] = {
                "predictions": file_entry(fold_dir / "predictions.csv"),
                "report": file_entry(fold_dir / "screen_report.json"),
            }
        else:
            fold_dir = ROOT / "model" / f"REGIME-RCAPACITY-TRANSITION-018-{season}"
            fold_files[str(season)] = {
                "predictions": file_entry(fold_dir / "paired_predictions.csv"),
                "report": file_entry(fold_dir / "screen_report.json"),
            }
    manifest = {
        "audit_id": args.name,
        "experiment_id": "REGIME-RCAPACITY-TRANSITION-018",
        "seasons": seasons,
        "official_train_only": True,
        "test_used": False,
        "external_data_used": False,
        "leaf_candidate_ids": ["baseline25_split75"],
        "fixed_blend_weights": {"baseline": 0.25, "split_rcapacity": 0.75},
        "required_relative_improvement": 0.02,
        "preserved_zip_expected_sha256": {
            "output/submit_combo_spread_005.zip": "9cb19954ad313eef4a05b2cb2b1ce339a765caf79c42c601836c9d9ef5cac946",
            "output/candidates/submit_regime_r_candidate.zip": "62de6d960c770cca03dc3bd9a0abac4d2364ae96426d95dd4292f5cd71993aa8",
            "output/submit_final_selective.zip": "22dc61a85a5f6ea26e81645b6e21eed3c59584435c0a172b98779159f2b997ff",
        },
        "common_files": [file_entry(path) for path in common_files],
        "fold_files": fold_files,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "status": "MANIFEST_READY_VALIDATION_PENDING",
    }
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = AUDIT_DIR / f"{args.name}_manifest.json"
    if output_path.exists():
        raise RuntimeError(f"refusing to overwrite existing manifest: {output_path}")
    output_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(output_path)


if __name__ == "__main__":
    main()
