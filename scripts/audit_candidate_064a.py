#!/usr/bin/env python3
"""Audit row-independence and package submit_ref4_lgbm_r_expert_064.zip."""
import hashlib, json, os, shutil, subprocess, sys, tempfile, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CAND_064 = ROOT / 'candidate/REF4-LGBM-R-EXPERT-064A'
OUT_ZIP = ROOT / 'output/submit_ref4_lgbm_r_expert_064.zip'
PYTHON = str(ROOT / '.venv-submit/bin/python')

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def make_eval_frame(n_rows: int = 1500, seed: int = 42) -> pd.DataFrame:
    columns = pd.read_csv(ROOT / "data/test.csv", nrows=1).columns
    train = pd.read_csv(ROOT / "data/train.csv", low_memory=False)
    rows = train[train.season == 2024].sample(n_rows, random_state=seed).reset_index(drop=True)
    rows["season"] = 2025
    rows["row_id"] = [f"TEST_{i:06d}" for i in range(len(rows))]
    return rows[columns]

def predict(workdir: Path, frame: pd.DataFrame) -> pd.Series:
    (workdir / "data").mkdir(exist_ok=True)
    frame.to_csv(workdir / "data/test.csv", index=False)
    done = subprocess.run([PYTHON, "script.py"], cwd=workdir, capture_output=True, text=True)
    if done.returncode != 0:
        raise SystemExit(f"inference failed:\n{done.stdout}\n{done.stderr}")
    out = pd.read_csv(workdir / "output/submission.csv")
    return out.set_index("row_id").control_success

def test_row_independence(workdir: Path, n_rows: int = 1500, singles: int = 8, tol: float = 1e-5):
    print(f"\n--- Testing Row Independence on {workdir.name} ---")
    frame = make_eval_frame(n_rows, seed=42)
    full = predict(workdir, frame)
    
    checks = []
    # 1. Half file
    half = frame.iloc[: len(frame) // 2]
    checks.append(("half file", full.reindex(half.row_id), predict(workdir, half)))
    
    # 2. Shuffled file
    shuffled = frame.sample(frac=1.0, random_state=43)
    checks.append(("shuffled file", full.reindex(shuffled.row_id), predict(workdir, shuffled)))
    
    # 3. N rows completely alone
    picks = frame.iloc[:: max(1, len(frame) // singles)].head(singles)
    alone = pd.concat([predict(workdir, frame.iloc[[i]]) for i in picks.index])
    checks.append((f"{len(picks)} rows alone", full.reindex(alone.index), alone))
    
    worst = 0.0
    for name, ref, other in checks:
        gap = float(np.max(np.abs(ref.to_numpy(float) - other.to_numpy(float))))
        worst = max(worst, gap)
        verdict = "동일" if gap <= 1e-9 else ("부동소수점 오차" if gap <= tol else "누수")
        print(f"  {name:<22} rows={len(other):5d}  max|diff| = {gap:.3e}   {verdict}")
        
    assert worst <= tol, f"FAIL: max diff {worst:.3e} exceeds tolerance {tol}"
    print(f"PASS: Row-independence verified (worst diff: {worst:.3e} <= {tol})\n")
    return worst

def main():
    t0 = time.time()
    # 1. Test candidate directly in temporary copy
    with tempfile.TemporaryDirectory(prefix="candcheck-") as tmpdir:
        tmp_path = Path(tmpdir)
        for item in CAND_064.iterdir():
            if item.is_dir():
                shutil.copytree(item, tmp_path / item.name)
            else:
                shutil.copy2(item, tmp_path / item.name)
        worst_cand = test_row_independence(tmp_path)
        
    # 2. Package ZIP
    print(f"--- Packaging ZIP: {OUT_ZIP.name} ---")
    if OUT_ZIP.exists():
        OUT_ZIP.unlink()
        
    with zipfile.ZipFile(OUT_ZIP, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for root_dir, dirs, files in os.walk(CAND_064):
            for file in files:
                full_p = Path(root_dir) / file
                rel_p = full_p.relative_to(CAND_064)
                z.write(full_p, arcname=str(rel_p))
                
    zip_size = OUT_ZIP.stat().st_size
    zip_sha = sha256_file(OUT_ZIP)
    print(f"ZIP created: {OUT_ZIP} ({zip_size / (1024*1024):.2f} MB, SHA256={zip_sha})")
    
    # 3. Test ZIP extraction and inference
    with tempfile.TemporaryDirectory(prefix="zipcheck-") as tmpdir:
        tmp_path = Path(tmpdir)
        with zipfile.ZipFile(OUT_ZIP) as z:
            z.extractall(tmp_path)
        worst_zip = test_row_independence(tmp_path)
        
    report = {
        'candidate_dir': str(CAND_064),
        'zip_file': str(OUT_ZIP),
        'zip_sha256': zip_sha,
        'zip_size_bytes': zip_size,
        'row_independence_worst_diff_candidate': worst_cand,
        'row_independence_worst_diff_zip': worst_zip,
        'row_independence_status': 'PASS',
        'elapsed_seconds': time.time() - t0
    }
    
    out_rep = ROOT / 'model/REF4-TRAINONLY-R-LGBM-CONSERVATIVE-063A/production_package_report.json'
    out_rep.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print("Production package report saved at", out_rep)
    print(f"Completed in {time.time() - t0:.2f}s")

if __name__ == '__main__':
    main()
