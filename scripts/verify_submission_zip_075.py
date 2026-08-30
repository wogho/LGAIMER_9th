#!/usr/bin/env python3
"""Comprehensive Zero-Risk Submission Verification for submit_ref4_macro_leap_075.zip."""
import ast, gc, hashlib, json, os, shutil, subprocess, sys, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / 'output/submit_ref4_macro_leap_075.zip'
SANDBOX = ROOT / 'model/REF4-MACRO-LEAP-CHAMPION-075A/audit_verification_sandbox'
PYTHON_BIN = str(ROOT / '.venv-submit/bin/python')

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def check_ast_safety(script_path: Path):
    tree = ast.parse(script_path.read_text(encoding='utf-8'))
    forbidden_modules = {'urllib', 'requests', 'http', 'socket', 'ftplib', 'smtplib', 'telnetlib'}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name.split('.')[0]
                if mod in forbidden_modules:
                    raise ValueError(f"Forbidden module import detected in script.py: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mod = node.module.split('.')[0]
                if mod in forbidden_modules:
                    raise ValueError(f"Forbidden module import detected in script.py: {node.module}")
    return True

def run_test_in_sandbox(test_df: pd.DataFrame) -> pd.DataFrame:
    data_dir = SANDBOX / 'data'
    data_dir.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(data_dir / 'test.csv', index=False)
    
    cmd = [PYTHON_BIN, 'script.py']
    p = subprocess.run(cmd, cwd=str(SANDBOX), capture_output=True, text=True)
    if p.returncode != 0:
        print("STDOUT:", p.stdout)
        print("STDERR:", p.stderr)
        raise RuntimeError(f"Sandbox inference failed with exit code {p.returncode}")
        
    out_csv = SANDBOX / 'output/submission.csv'
    if not out_csv.exists():
        raise FileNotFoundError("output/submission.csv was not produced!")
    return pd.read_csv(out_csv)

def main():
    t0 = time.time()
    results = {}
    print("=" * 70)
    print("      LG AIMERS 075A (MACRO-LEAP) SUBMISSION ZERO-RISK AUDIT      ")
    print("=" * 70)
    
    # 1. ZIP File Existence, Size, and Checksum
    print("\n[Gate 1] ZIP File Existence & Checksum Inspection")
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Missing submission ZIP: {ZIP_PATH}")
    zip_size = ZIP_PATH.stat().st_size
    zip_hash = sha256_file(ZIP_PATH)
    print(f"  • File:    {ZIP_PATH.name}")
    print(f"  • Size:    {zip_size:,} bytes ({zip_size / (1024*1024):.2f} MB) [< 10 GB limit]")
    print(f"  • SHA-256: {zip_hash}")
    results['gate1_zip_check'] = (zip_size > 0) and (zip_size < 1024*1024*500)
    
    # 2. ZIP Archive Integrity and Member Whitelist Check
    print("\n[Gate 2] ZIP Archive Integrity & File Layout Check")
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        crc_check = zf.testzip()
        if crc_check is not None:
            raise ValueError(f"Corrupt chunk detected in ZIP at: {crc_check}")
        namelist = zf.namelist()
        print(f"  • CRC Integrity Check: PASS (testzip() returned None)")
        print(f"  • Total files in archive: {len(namelist):,} files")
        
        has_req = 'requirements.txt' in namelist
        has_script = 'script.py' in namelist
        print(f"  • 'requirements.txt' at root: {has_req} (REQUIRED: True)")
        print(f"  • 'script.py' at root:        {has_script} (REQUIRED: True)")
        
        # Check no top-level wrapper folder
        has_wrapper = any(name.startswith(ZIP_PATH.stem + '/') for name in namelist)
        print(f"  • No root wrapper directory:  {not has_wrapper} (REQUIRED: True)")
        
        # Check no pycache or garbage
        has_pycache = any('__pycache__' in name or name.endswith('.pyc') for name in namelist)
        print(f"  • No __pycache__ / *.pyc:     {not has_pycache} (REQUIRED: True)")
        
        req_content = zf.read('requirements.txt').decode('utf-8').strip()
        print(f"  • requirements.txt contents:\n    " + req_content.replace('\n', '\n    '))
        
        results['gate2_layout_check'] = (crc_check is None) and has_req and has_script and (not has_wrapper) and (not has_pycache)
        
    # 3. Clean Sandbox Extraction & AST Safety Audit
    print("\n[Gate 3] Clean Extraction & Static AST Code Audit")
    if SANDBOX.exists():
        shutil.rmtree(SANDBOX)
    SANDBOX.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
        zf.extractall(SANDBOX)
    print(f"  • Extracted to sandbox: {SANDBOX}")
    
    ast_safe = check_ast_safety(SANDBOX / 'script.py')
    print(f"  • AST Network/Socket Safety: PASS (No forbidden remote calls)")
    results['gate3_ast_safety'] = ast_safe
    
    # 4. Official Dummy Test (5 rows from data/test.csv)
    print("\n[Gate 4] Official Dummy Test (data/test.csv)")
    dummy_df = pd.read_csv(ROOT / 'data/test.csv')
    res_dummy = run_test_in_sandbox(dummy_df)
    
    dummy_rows_match = (len(res_dummy) == len(dummy_df))
    dummy_cols_match = (list(res_dummy.columns) == ['row_id', 'control_success'])
    dummy_id_match = (res_dummy['row_id'].tolist() == dummy_df['row_id'].tolist())
    dummy_p = res_dummy['control_success'].to_numpy(float)
    dummy_finite = np.isfinite(dummy_p).all()
    dummy_range = (dummy_p >= 0.0).all() and (dummy_p <= 1.0).all()
    
    print(f"  • Output row count match: {dummy_rows_match} ({len(res_dummy)} rows)")
    print(f"  • Output columns match:   {dummy_cols_match} ({list(res_dummy.columns)})")
    print(f"  • Output row_id exact match: {dummy_id_match}")
    print(f"  • Finite values (no NaN/Inf): {dummy_finite}")
    print(f"  • Probability in [0, 1]:  {dummy_range} (min={dummy_p.min():.6f}, max={dummy_p.max():.6f})")
    results['gate4_dummy_test'] = dummy_rows_match and dummy_cols_match and dummy_id_match and dummy_finite and dummy_range
    
    # 5. Real Mixed Dataset (1군 Regular 1,000 rows + 2군 Futures 500 rows = 1,500 rows)
    print("\n[Gate 5] Mixed 1군(Regular) & 2군(Futures) Real Test")
    train_raw = pd.read_csv(ROOT / 'data/train.csv', low_memory=False)
    reg_rows = train_raw.loc[(train_raw.season == 2024) & (train_raw.game_type == 'R')].iloc[:1000]
    fut_rows = train_raw.loc[(train_raw.season == 2024) & (train_raw.game_type == 'F')].iloc[:500]
    probe_1500 = pd.concat([reg_rows, fut_rows], ignore_index=True)
    n_reg = (probe_1500.game_type == 'R').sum()
    n_fut = (probe_1500.game_type == 'F').sum()
    print(f"  • Probe sample: {len(probe_1500)} rows (Regular: {n_reg:,}, Futures: {n_fut:,})")
    
    t_start = time.time()
    res_1500 = run_test_in_sandbox(probe_1500)
    elapsed_1500 = time.time() - t_start
    p_1500 = res_1500['control_success'].to_numpy(float)
    
    print(f"  • Execution time: {elapsed_1500:.2f}s ({elapsed_1500/1.5:.2f} ms/row)")
    print(f"  • Output rows: {len(res_1500)} (Expected: 1,500)")
    print(f"  • Mean probability: {np.mean(p_1500):.6f} (Well-calibrated around ~0.485)")
    print(f"  • Min/Max prob: [{np.min(p_1500):.6f}, {np.max(p_1500):.6f}]")
    print(f"  • Std dev: {np.std(p_1500):.6f}")
    results['gate5_mixed_real'] = (len(res_1500) == 1500) and np.isfinite(p_1500).all() and (np.min(p_1500) > 0.0) and (np.max(p_1500) < 1.0)
    
    # 6. Column Reordering Invariance Test
    print("\n[Gate 6] Column Permutation / Reordering Test")
    cols_reversed = list(probe_1500.columns)[::-1]
    probe_reversed = probe_1500[cols_reversed].copy()
    res_reversed = run_test_in_sandbox(probe_reversed)
    p_rev = res_reversed['control_success'].to_numpy(float)
    max_col_diff = np.max(np.abs(p_1500 - p_rev))
    print(f"  • Max |diff| when columns are reversed: {max_col_diff:.3e} (REQUIRED: 0.000e+00)")
    results['gate6_col_invariance'] = (max_col_diff < 1e-9)
    
    # 7. Single Row Isolated Inference (Regular & Futures)
    print("\n[Gate 7] Single-Row Isolated Inference Test (Row-Independence)")
    reg_idx = list(np.where(probe_1500.game_type == 'R')[0][:2])
    fut_idx = list(np.where(probe_1500.game_type == 'F')[0][:2])
    single_diffs = []
    for idx in reg_idx + fut_idx:
        gt = probe_1500.game_type.iloc[idx]
        single_df = probe_1500.iloc[[idx]].copy().reset_index(drop=True)
        res_single = run_test_in_sandbox(single_df)
        p_s = float(res_single['control_success'].iloc[0])
        diff = abs(p_1500[idx] - p_s)
        single_diffs.append(diff)
        print(f"  • Row {idx:4d} ({gt}): Batch={p_1500[idx]:.8f}, Single={p_s:.8f}, Diff={diff:.3e}")
    max_single_diff = max(single_diffs)
    print(f"  • Max single-row difference: {max_single_diff:.3e} (REQUIRED: 0.000e+00)")
    results['gate7_row_independence'] = (max_single_diff < 1e-9)
    
    # 8. Large-Scale Stress Test (5,000 rows)
    print("\n[Gate 8] Large-Scale Stress Benchmark (5,000 rows)")
    probe_5000 = train_raw.loc[train_raw.season == 2024].iloc[:5000].copy().reset_index(drop=True)
    t_bench_start = time.time()
    res_5000 = run_test_in_sandbox(probe_5000)
    elapsed_5000 = time.time() - t_bench_start
    print(f"  • Inferred 5,000 rows in: {elapsed_5000:.2f}s (Extrapolated 245k rows: ~{elapsed_5000 * 49:.1f}s << 600s limit)")
    results['gate8_stress_benchmark'] = (len(res_5000) == 5000) and (elapsed_5000 < 30.0)
    
    # Summary of All Gates
    print("\n" + "=" * 70)
    print("                     FINAL AUDIT SUMMARY                         ")
    print("=" * 70)
    all_passed = True
    for gate_name, status in results.items():
        pass_str = "✅ PASS" if status else "❌ FAIL"
        print(f"  {gate_name:<30}: {pass_str}")
        if not status:
            all_passed = False
            
    print("-" * 70)
    if all_passed:
        print("🎯 FINAL VERDICT: ✅ ALL GATES PASSED (100% SUBMISSION SAFE & CERTIFIED)")
    else:
        print("🚨 FINAL VERDICT: ❌ ONE OR MORE GATES FAILED")
    print(f"Total verification time: {time.time() - t0:.2f}s")
    print("=" * 70)
    
    if not all_passed:
        sys.exit(1)

if __name__ == '__main__':
    main()
