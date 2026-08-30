#!/usr/bin/env python3
"""Build Full-Strength Adaptive Channel Opt Package 072A (gate_scale=0.75)."""
import gc, hashlib, json, os, shutil, sys, time, zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXP_DIR = ROOT / 'model/REF4-ADAPTIVE-CHANNEL-OPT-072A'
PROD_DIR = EXP_DIR / 'production_package'
MODEL_DIR = PROD_DIR / 'model'
SRC_DIR = PROD_DIR / 'src'
MODEL_DIR.mkdir(parents=True, exist_ok=True)
SRC_DIR.mkdir(parents=True, exist_ok=True)

SOURCE_069_MODEL = ROOT / 'model/REF4-ADAPTIVE-CHANNEL-OPT-069A/production_package/model'
SOURCE_069_SRC = ROOT / 'model/REF4-ADAPTIVE-CHANNEL-OPT-069A/production_package/src'
SOURCE_069_SCRIPT = ROOT / 'model/REF4-ADAPTIVE-CHANNEL-OPT-069A/production_package/script.py'

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    t0 = time.time()
    print("=== Step 1: Copying Assets, Script, and Sources ===")
    for f in SOURCE_069_MODEL.iterdir():
        if f.is_file():
            shutil.copy2(f, MODEL_DIR / f.name)
            
    for f in SOURCE_069_SRC.iterdir():
        if f.is_file():
            shutil.copy2(f, SRC_DIR / f.name)
            
    shutil.copy2(SOURCE_069_SCRIPT, PROD_DIR / 'script.py')
    
    # Update manifest for 072A: gate_scale = 0.75
    manifest = json.loads((MODEL_DIR / "manifest.json").read_text(encoding="utf-8"))
    manifest["adaptive_gate"] = True
    manifest["gate_scale"] = 0.75  # Full-strength proven gate scale!
    manifest["gate_bias_offset"] = 0.00848698  # Exact zero-centering
    manifest["r_expert_lgbm"] = True
    manifest["r_expert_lgbm_weight"] = 0.02
    manifest["r_split_table"] = True
    manifest["f_psych_latent"] = True
    manifest["global_shift"] = 0.0052
    (MODEL_DIR / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    
    # Write requirements.txt
    req_content = "numpy>=1.24.0\npandas>=2.0.0\ncatboost>=1.2.0\nlightgbm>=4.0.0\n"
    (PROD_DIR / "requirements.txt").write_text(req_content)
    
    print("=== Step 2: Packaging submit_ref4_adaptive_channel_opt_072.zip ===")
    zip_path = ROOT / 'output/submit_ref4_adaptive_channel_opt_072.zip'
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
        'experiment_id': 'REF4-ADAPTIVE-CHANNEL-OPT-072A',
        'status': 'PACKAGED_AND_READY',
        'zip_filename': zip_path.name,
        'zip_size_bytes': zip_size,
        'sha256': zip_hash,
        'gate_scale': 0.75,
        'gate_bias_offset': 0.00848698,
        'has_requirements_txt': True,
        'elapsed_seconds': time.time() - t0
    }
    (EXP_DIR / 'production_package_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(f"Completed in {time.time() - t0:.2f}s")

if __name__ == '__main__':
    main()
