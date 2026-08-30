#!/usr/bin/env python3
"""Build a deterministic, isolated research ZIP for REF-AUX-OFFSET-CAT-OOF-003."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = ROOT / "candidate" / "REF-AUX-OFFSET-CAT-OOF-003"
OUT = ROOT / "output" / "candidates" / "submit_ref_aux_offset_cat_oof_003.zip"
FILES = [
    ("script.py", CANDIDATE / "script.py"),
    ("requirements.txt", CANDIDATE / "requirements.txt"),
    ("model/catboost_model.cbm", CANDIDATE / "model/catboost_model.cbm"),
    ("model/ensemble_contract.json", CANDIDATE / "model/ensemble_contract.json"),
    ("model/feature_columns.json", CANDIDATE / "model/feature_columns.json"),
    ("model/lightgbm_model.txt", CANDIDATE / "model/lightgbm_model.txt"),
    ("model/aux_feature_medians.json", CANDIDATE / "model/aux_feature_medians.json"),
    ("model/catboost_2024_mr.cbm", CANDIDATE / "model/catboost_2024_mr.cbm"),
    ("model/catboost_2024_wayoff.cbm", CANDIDATE / "model/catboost_2024_wayoff.cbm"),
    ("LG_Aimers_솔루션_PPT_Phase2.pptx", ROOT / "output" / "LG_Aimers_솔루션_PPT_Phase2.pptx"),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    missing = [str(path) for _, path in FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for arcname, path in sorted(FILES):
            info = zipfile.ZipInfo(arcname, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            zf.writestr(info, path.read_bytes())
    with zipfile.ZipFile(OUT) as zf:
        members = zf.namelist()
        if members != [name for name, _ in sorted(FILES)]:
            raise AssertionError(f"archive members mismatch: {members}")
        bad = zf.testzip()
        if bad:
            raise AssertionError(f"archive CRC failure: {bad}")
    manifest = {
        "experiment_id": "REF-AUX-OFFSET-CAT-OOF-003",
        "zip_path": str(OUT),
        "zip_sha256": sha256(OUT),
        "zip_bytes": OUT.stat().st_size,
        "members": [name for name, _ in sorted(FILES)],
        "member_sha256": {name: sha256(path) for name, path in FILES},
        "submission_status": "HOLD",
    }
    (OUT.with_suffix(".manifest.json")).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
