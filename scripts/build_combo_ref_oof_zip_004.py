from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAND = ROOT / "candidate" / "COMBO-REF-OOF-004"
OUT = ROOT / "output" / "submit_combo_ref_oof_004.zip"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    files = [("script.py", CAND / "script.py"), ("combo_infer.py", CAND / "combo_infer.py"),
             ("ref_aux.py", CAND / "ref_aux.py"), ("requirements.txt", CAND / "requirements.txt"),
             ("LG_Aimers_솔루션_PPT_Phase2.pptx", ROOT / "output" / "LG_Aimers_솔루션_PPT_Phase2.pptx")]
    files += [(f"model/{p.relative_to(CAND / 'model')}", p) for p in sorted((CAND / 'model').rglob('*')) if p.is_file()]
    missing = [str(p) for _, p in files if not p.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for arc, path in sorted(files):
            info = zipfile.ZipInfo(arc, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            z.writestr(info, path.read_bytes())
    with zipfile.ZipFile(OUT) as z:
        if z.testzip() is not None:
            raise AssertionError("ZIP CRC failure")
    record = {"experiment_id": "COMBO-REF-OOF-004", "zip_path": str(OUT),
              "zip_sha256": sha256(OUT), "members": [a for a, _ in sorted(files)],
              "member_sha256": {a: sha256(p) for a, p in files}, "submission_status": "HOLD"}
    (OUT.with_suffix('.manifest.json')).write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
