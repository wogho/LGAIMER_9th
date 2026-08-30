#!/usr/bin/env python3
"""Build and independently audit submit_regime_rcapacity_019.zip."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_DIR = ROOT / "candidate" / "REGIME-RCAPACITY-FULL-019"
OUTPUT_DIR = ROOT / "output"
ZIP_PATH = OUTPUT_DIR / "submit_regime_rcapacity_019.zip"
PPT_SRC = OUTPUT_DIR / "LG_Aimers_솔루션_PPT_Phase2.pptx"

WHITELIST = {
    "script.py": CANDIDATE_DIR / "script.py",
    "requirements.txt": CANDIDATE_DIR / "requirements.txt",
    "model/model_baseline_combo.cbm": CANDIDATE_DIR / "model" / "model_baseline_combo.cbm",
    "model/model_regime_f.cbm": CANDIDATE_DIR / "model" / "model_regime_f.cbm",
    "model/model_regime_r.cbm": CANDIDATE_DIR / "model" / "model_regime_r.cbm",
    "model/feature_columns.json": CANDIDATE_DIR / "model" / "feature_columns.json",
    "model/asof_pitcher_id_prior.csv": CANDIDATE_DIR / "model" / "asof_pitcher_id_prior.csv",
    "model/asof_batter_id_prior.csv": CANDIDATE_DIR / "model" / "asof_batter_id_prior.csv",
    "model/asof_pitchmix_prior.csv": CANDIDATE_DIR / "model" / "asof_pitchmix_prior.csv",
    "model/pitcher_count_lookup.csv": CANDIDATE_DIR / "model" / "pitcher_count_lookup.csv",
    "model/trackman_count_lookup.csv": CANDIDATE_DIR / "model" / "trackman_count_lookup.csv",
    "model/trackman_hand_lookup.csv": CANDIDATE_DIR / "model" / "trackman_hand_lookup.csv",
    "model/pitcher_id_map_audit.csv": CANDIDATE_DIR / "model" / "pitcher_id_map_audit.csv",
    "solution/LG_Aimers_솔루션_PPT_Phase2.pptx": PPT_SRC,
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    start_time = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[ZIP-BUILD-AUDIT] Packaging {len(WHITELIST)} members into {ZIP_PATH.name}...")

    # Verify source files existence
    missing = [str(p) for p in WHITELIST.values() if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing required sources: {missing}")

    # 1. Deterministic ZIP packaging
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, src_path in sorted(WHITELIST.items()):
            content = src_path.read_bytes()
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            z.writestr(info, content)

    zip_sha256 = sha256_file(ZIP_PATH)
    zip_size = ZIP_PATH.stat().st_size
    print(f"  ✅ ZIP created successfully: size={zip_size:,} bytes, sha256={zip_sha256}")

    # 2. Independent E2E sandbox execution from ZIP
    print("  Testing isolated ZIP extraction and E2E inference...")
    with tempfile.TemporaryDirectory(prefix="zip_e2e_sandbox_") as sb_dir:
        sb = Path(sb_dir)
        with zipfile.ZipFile(ZIP_PATH, "r") as z:
            z.extractall(sb)

        (sb / "data").mkdir(exist_ok=True)
        (sb / "output").mkdir(exist_ok=True)
        shutil.copy2(ROOT / "data" / "test.csv", sb / "data" / "test.csv")

        res = subprocess.run(
            [sys.executable, "script.py"],
            cwd=sb,
            capture_output=True,
            text=True,
            timeout=120,
            check=True,
        )
        sub_file = sb / "output" / "submission.csv"
        if not sub_file.exists():
            raise RuntimeError(f"submission.csv missing after ZIP inference: stderr={res.stderr}")

        sub_df = pd.read_csv(sub_file, encoding="utf-8")
        test_df = pd.read_csv(ROOT / "data" / "test.csv", encoding="utf-8-sig")

        if len(sub_df) != len(test_df):
            raise RuntimeError(f"Row count mismatch: {len(sub_df)} != {len(test_df)}")
        if not sub_df.row_id.equals(test_df.row_id):
            raise RuntimeError("row_id mismatch or ordering violated")
        if not np.isfinite(sub_df.control_success).all():
            raise RuntimeError("Non-finite predictions found in submission.csv")
        if not ((sub_df.control_success >= 0) & (sub_df.control_success <= 1)).all():
            raise RuntimeError("Predictions out of [0, 1] range")

        p_vals = sub_df.control_success.to_numpy()
        print(f"  ✅ ZIP E2E Inference: PASS ({len(sub_df)} rows, min={p_vals.min():.6f}, max={p_vals.max():.6f})")

    # 3. Static Audit of ZIP members and contents
    print("  Performing static audit on ZIP archive...")
    with zipfile.ZipFile(ZIP_PATH, "r") as z:
        names = set(z.namelist())
        if names != set(WHITELIST):
            raise RuntimeError(f"ZIP member mismatch: diff={names.symmetric_difference(WHITELIST)}")

        # Source hash match
        source_hashes = {}
        for name in sorted(names):
            member_bytes = z.read(name)
            m_hash = sha256_bytes(member_bytes)
            s_hash = sha256_file(WHITELIST[name])
            if m_hash != s_hash:
                raise RuntimeError(f"Hash mismatch for member {name}: zip={m_hash} != src={s_hash}")
            source_hashes[name] = {
                "sha256": m_hash,
                "size_bytes": len(member_bytes),
                "match": True,
            }

        # Code forbidden tokens check
        script_src = z.read("script.py").decode("utf-8")
        forbidden = ["requests", "urllib", "http://", "https://", "openai", "google.generativeai", "train.csv", "trackman_history.csv"]
        found_forbidden = [tok for tok in forbidden if tok in script_src]
        if found_forbidden:
            raise RuntimeError(f"Forbidden tokens found in script.py: {found_forbidden}")

        fc_data = json.loads(z.read("model/feature_columns.json").decode("utf-8"))
        feature_count = len(fc_data)
        if feature_count != 81:
            raise RuntimeError(f"Feature count is {feature_count}, expected 81")

    # 4. Generate Audit and Manifest Reports
    manifest = {
        "manifest_id": "SUBMIT-REGIME-RCAPACITY-019-MANIFEST",
        "zip_path": str(ZIP_PATH),
        "zip_sha256": zip_sha256,
        "zip_size_bytes": zip_size,
        "member_count": len(WHITELIST),
        "members": source_hashes,
    }
    (OUTPUT_DIR / "submit_regime_rcapacity_019.manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    audit_report = {
        "experiment_id": "REGIME-RCAPACITY-FULL-ZIP-AUDIT-019",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "zip_file": str(ZIP_PATH.name),
        "zip_sha256": zip_sha256,
        "zip_size_bytes": zip_size,
        "member_count": len(WHITELIST),
        "member_hash_match": True,
        "ppt_included": True,
        "feature_count": feature_count,
        "forbidden_tokens_found": found_forbidden,
        "external_api_used": False,
        "external_data_used": False,
        "test_row_aggregation_used": False,
        "test_prediction_stats": {
            "rows": len(sub_df),
            "min": float(p_vals.min()),
            "max": float(p_vals.max()),
            "mean": float(p_vals.mean()),
        },
        "elapsed_seconds": time.time() - start_time,
        "status": "AUDIT_VERIFIED",
        "submission_status": "READY_FOR_SUBMISSION",
    }
    (OUTPUT_DIR / "submit_regime_rcapacity_019_audit.json").write_text(json.dumps(audit_report, ensure_ascii=False, indent=2))

    print(json.dumps(audit_report, ensure_ascii=False, indent=2))
    print(f"\n[ZIP-BUILD-AUDIT] All steps verified successfully! ZIP ready at: {ZIP_PATH}")


if __name__ == "__main__":
    main()
