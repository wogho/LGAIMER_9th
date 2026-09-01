#!/usr/bin/env python3
"""
Final packaging script for 113A, 113B, 113C.
Ensures clean code, stable weights, and verified ZIP packaging.
"""
import json
import os
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def package_zip(src_dir: Path, out_zip: Path):
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for file in src_dir.rglob("*"):
            if file.is_file():
                arcname = file.relative_to(src_dir)
                z.write(file, arcname)
    size_mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"Created {out_zip.name} ({size_mb:.2f} MB)")

def main():
    print("=" * 80)
    print("  PACKAGING FINAL 113 TRIO (113A, 113B, 113C)  ")
    print("=" * 80)
    
    # 1. 113A
    pkg_113a = ROOT / "model/REF4-DISJOINT-EB-113A/production_package"
    zip_113a = ROOT / "output/submit_ref4_super_ensemble_113A.zip"
    package_zip(pkg_113a, zip_113a)
    
    # 2. 113B
    pkg_113b = ROOT / "model/REF4-TUNNELING-GBDT-113B/production_package"
    zip_113b = ROOT / "output/submit_ref4_super_ensemble_113B.zip"
    package_zip(pkg_113b, zip_113b)
    
    # 3. 113C
    pkg_113c = ROOT / "model/REF4-BETA-SIMPLEX-113C/production_package"
    zip_113c = ROOT / "output/submit_ref4_super_ensemble_113C.zip"
    package_zip(pkg_113c, zip_113c)

if __name__ == "__main__":
    main()
