#!/usr/bin/env python3
"""Build Production Package 079A with R-decoupled residual weight w_lgb=0.05 based on 078A champion backbone."""
import gc, hashlib, json, os, shutil, sys, time, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / 'model/REF4-R-RESIDUAL-DECOUPLE-079A'
PROD_DIR = EXP_DIR / 'production_package'
MODEL_DIR = PROD_DIR / 'model'
SRC_DIR = PROD_DIR / 'src'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
SRC_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_078_MODEL = ROOT / 'model/REF4-ADAPTIVE-CHANNEL-OPT-078A/production_package/model'
SOURCE_078_SRC = ROOT / 'model/REF4-ADAPTIVE-CHANNEL-OPT-078A/production_package/src'
SOURCE_078_SCRIPT = ROOT / 'model/REF4-ADAPTIVE-CHANNEL-OPT-078A/production_package/script.py'
SOURCE_078_REQS = ROOT / 'model/REF4-ADAPTIVE-CHANNEL-OPT-078A/production_package/requirements.txt'

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    t0 = time.time()
    print("=== Step 1: Copying 078A Assets, Script, and Sources ===")
    for f in SOURCE_078_MODEL.iterdir():
        if f.is_file():
            shutil.copy2(f, MODEL_DIR / f.name)
            
    for f in SOURCE_078_SRC.iterdir():
        if f.is_file():
            shutil.copy2(f, SRC_DIR / f.name)
            
    shutil.copy2(SOURCE_078_SCRIPT, PROD_DIR / 'script.py')
    shutil.copy2(SOURCE_078_REQS, PROD_DIR / 'requirements.txt')
    
    # Update manifest.json with version 079A, gate_scale = 0.08, and r_expert_lgbm_weight = 0.05
    manifest_path = MODEL_DIR / 'manifest.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    manifest['version'] = 'REF4-R-RESIDUAL-DECOUPLE-079A'
    manifest['gate_scale'] = 0.08
    manifest['r_expert_lgbm_weight'] = 0.05
    manifest['notes'] = "078A Champion Backbone (1094.54 LB) with R-Season Decoupled Residual w_lgb=0.05"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(f"Updated manifest.json: gate_scale = {manifest['gate_scale']}, r_expert_lgbm_weight = {manifest['r_expert_lgbm_weight']}")
    
    print("=== Step 2: Packaging submit_ref4_r_residual_decouple_079.zip ===")
    zip_path = ROOT / 'output/submit_ref4_r_residual_decouple_079.zip'
    if zip_path.exists():
        zip_path.unlink()
        
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.write(PROD_DIR / 'requirements.txt', 'requirements.txt')
        zf.write(PROD_DIR / 'script.py', 'script.py')
        for root_dir, _, files in os.walk(MODEL_DIR):
            for file in files:
                abs_f = Path(root_dir) / file
                rel_f = abs_f.relative_to(PROD_DIR)
                zf.write(abs_f, str(rel_f))
        for root_dir, _, files in os.walk(SRC_DIR):
            for file in files:
                abs_f = Path(root_dir) / file
                rel_f = abs_f.relative_to(PROD_DIR)
                zf.write(abs_f, str(rel_f))
                
    zip_hash = sha256_file(zip_path)
    zip_size = zip_path.stat().st_size
    print(f"\nCreated ZIP: {zip_path.name}")
    print(f"Size: {zip_size:,} bytes ({zip_size / (1024*1024):.2f} MB)")
    print(f"SHA-256: {zip_hash}")
    
    with zipfile.ZipFile(zip_path, 'r') as zf:
        names = zf.namelist()
        print(f"ZIP check: 'requirements.txt' in zip: {'requirements.txt' in names}")
        print(f"ZIP check: 'script.py' in zip: {'script.py' in names}")
        print(f"ZIP check total files: {len(names)}")
        
    report = {
        'experiment_id': 'REF4-R-RESIDUAL-DECOUPLE-079A',
        'status': 'PACKAGED_AND_READY',
        'zip_filename': zip_path.name,
        'zip_size_bytes': zip_size,
        'sha256': zip_hash,
        'has_requirements_txt': True,
        'gate_scale': 0.08,
        'r_expert_lgbm_weight': 0.05,
        'elapsed_seconds': time.time() - t0
    }
    (EXP_DIR / 'production_package_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f"Completed in {time.time() - t0:.2f}s")

if __name__ == '__main__':
    main()
