#!/usr/bin/env python3
"""Build isolated 119A/B/C production packages on the immutable 113A base."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "model/REF4-DISJOINT-EB-113A/production_package"
RESEARCH = ROOT / "model/REF4-119-RESEARCH"
SOURCE_119 = ROOT / "src/ref4_119.py"

CANDIDATES = {
    "119A": {
        "name": "ABS-ERA-RESIDUAL-119A",
        "function": "apply_abs_era_119a",
        "assets": ("abs_era_119a_meta.json", "abs_era_119a_table.csv"),
    },
    "119B": {
        "name": "LATENT-PITCH-MARGINAL-MOE-119B",
        "function": "apply_latent_moe_119b",
        "assets": ("latent_moe_119b_meta.json", "latent_moe_119b_models.npz", "latent_pitch_119b.csv"),
    },
    "119C": {
        "name": "TRACKMAN-QUANTILE-DRIFT-119C",
        "function": "apply_quantile_drift_119c",
        "assets": ("quantile_drift_119c_meta.json", "quantile_drift_119c_model.npz", "trackman_quantile_119c.csv"),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def replace_hardlink(path: Path, content: str) -> None:
    if path.exists():
        path.unlink()
    path.write_text(content, encoding="utf-8")


def main() -> None:
    if not BASE.exists():
        raise FileNotFoundError(BASE)
    report = json.loads((RESEARCH / "comparison_report.json").read_text(encoding="utf-8"))
    base_script = (BASE / "script.py").read_text(encoding="utf-8")
    marker = "    # Monotonic bounded calibration\n    p = np.clip(p, 0.02, 0.98)"
    if base_script.count(marker) != 1:
        raise RuntimeError("113A production insertion marker mismatch")

    build_report = {
        "base": "REF4-DISJOINT-EB-113A",
        "base_script_sha256": sha256(BASE / "script.py"),
        "research_report_sha256": sha256(RESEARCH / "comparison_report.json"),
        "packages": {},
    }
    for version, config in CANDIDATES.items():
        out = ROOT / f"model/REF4-{config['name']}/production_package"
        if out.exists():
            raise FileExistsError(f"refusing to overwrite existing candidate: {out}")
        out.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(BASE, out, copy_function=os.link)

        source_target = out / "src/ref4_119.py"
        shutil.copy2(SOURCE_119, source_target)
        for asset in config["assets"]:
            shutil.copy2(RESEARCH / asset, out / "model" / asset)

        manifest_path = out / "model/manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "pipeline": config["name"],
                "base_pipeline": "REF4-DISJOINT-EB-113A",
                "official_base_score": 1121.9039933605,
                "candidate_gate_pass": bool(report[version]["gate_pass"]),
                "candidate_enabled": bool(report[version]["gate_pass"]),
                "research_report_sha256": build_report["research_report_sha256"],
                "safety_policy": "bounded row-local correction; disabled candidate is exact 113A rollback",
            }
        )
        replace_hardlink(manifest_path, json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")

        function = config["function"]
        insertion = (
            f"    # {config['name']}: offline-fitted, row-independent bounded correction\n"
            f"    from src.ref4_119 import {function}\n"
            f"    p = {function}(test.reset_index(drop=True), p, MODEL)\n\n"
            "    # Monotonic bounded calibration\n"
            "    p = np.clip(p, 0.02, 0.98)"
        )
        replace_hardlink(out / "script.py", base_script.replace(marker, insertion))

        zip_path = ROOT / f"output/submit_ref4_super_ensemble_{version}.zip"
        if zip_path.exists():
            raise FileExistsError(f"refusing to overwrite existing ZIP: {zip_path}")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(out.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(out))
        build_report["packages"][version] = {
            "name": config["name"],
            "gate_pass": bool(report[version]["gate_pass"]),
            "package": str(out.relative_to(ROOT)),
            "zip": str(zip_path.relative_to(ROOT)),
            "zip_size": zip_path.stat().st_size,
            "zip_sha256": sha256(zip_path),
        }
        print(json.dumps(build_report["packages"][version], ensure_ascii=False), flush=True)

    (RESEARCH / "build_report.json").write_text(json.dumps(build_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
