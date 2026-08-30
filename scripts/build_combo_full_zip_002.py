#!/usr/bin/env python3
from pathlib import Path
import hashlib, json, shutil, subprocess, tempfile, zipfile
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]; B = ROOT / "model/COMBO-FULL-002"; OUT = ROOT / "output"; OUT.mkdir(exist_ok=True)
FILES = {"script.py": ROOT/"scripts/infer_combo_full_isolated_002.py", "requirements.txt": ROOT/"requirements_submit.txt", "solution/LG_Aimers_솔루션_PPT_Phase2.pptx": ROOT/"output/LG_Aimers_솔루션_PPT_Phase2.pptx", "model/COMBO-FULL-002/model.cbm": B/"model.cbm", "model/COMBO-FULL-002/feature_columns.json": B/"feature_columns.json", "model/COMBO-FULL-002/pitcher_count_lookup.csv": B/"pitcher_count_lookup.csv", "model/COMBO-FULL-002/asof_pitcher_id_prior.csv": B/"asof_pitcher_id_prior.csv", "model/COMBO-FULL-002/asof_batter_id_prior.csv": B/"asof_batter_id_prior.csv", "model/COMBO-FULL-002/asof_pitchmix_prior.csv": B/"asof_pitchmix_prior.csv"}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    missing = [str(p) for p in FILES.values() if not p.is_file()]; assert not missing, missing
    zpath = OUT / "submit_combo_full_002.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for name, p in sorted(FILES.items()):
            info = zipfile.ZipInfo(name, date_time=(2020,1,1,0,0,0)); info.compress_type = zipfile.ZIP_DEFLATED; z.writestr(info, p.read_bytes())
    with zipfile.ZipFile(zpath) as z: assert set(z.namelist()) == set(FILES)
    sandbox = Path(tempfile.mkdtemp(prefix="combo_full_zip_")); (sandbox/"data").mkdir(); (sandbox/"output").mkdir()
    try:
        with zipfile.ZipFile(zpath) as z: z.extractall(sandbox)
        shutil.copy2(ROOT/"data/test.csv", sandbox/"data/test.csv")
        r = subprocess.run([str(ROOT/".venv/bin/python"), "script.py"], cwd=sandbox, capture_output=True, text=True, timeout=180); assert r.returncode == 0, r.stderr
        pred = pd.read_csv(sandbox/"model/COMBO-FULL-002/isolated_submission.csv"); sample = pd.read_csv(ROOT/"data/sample_submission.csv"); assert len(pred) == len(sample) and pred.row_id.is_unique and pred.control_success.between(0,1).all()
        report = {"experiment_id":"COMBO-FULL-ZIP-002", "zip":str(zpath), "zip_sha256":sha(zpath), "members":sorted(FILES), "member_count":len(FILES), "ppt_included":True, "sandbox_rows":len(pred), "status":"PASS_ZIP_E2E", "submission_status":"HOLD"}; (OUT/"submit_combo_full_002_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)); print(json.dumps(report, ensure_ascii=False, indent=2))
    finally: shutil.rmtree(sandbox, ignore_errors=True)
if __name__ == "__main__": main()
