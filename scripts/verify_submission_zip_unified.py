#!/usr/bin/env python3
"""
Strict 8-Gate Submission ZIP Verifier.
Enforces all official DACON constraints:
  - Total inference time for 245,789 rows must be strictly < 8.0 min (10 min official limit with 20% margin)
  - Process return code must be 0 for all gates
  - Row independence & Column permutation invariance: Max |diff| < 1e-9
  - No network / socket calls
"""
import ast
import os
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
PYTHON_BIN = sys.executable
TRAIN_PATH = ROOT / "data/train.csv"

def log_gate(gate_num: int, gate_name: str, passed: bool, details: str):
    tag = "[PASS]" if passed else "[FAIL]"
    print(f"  • [Gate {gate_num}] {gate_name:<34} : {tag} {details}")
    if not passed:
        raise AssertionError(f"Gate {gate_num} FAILED: {gate_name} - {details}")

def verify_zip(zip_path: Path):
    print("=" * 80)
    print(f"  STRICT 8-GATE VERIFICATION SUITE: {zip_path.name}")
    print("=" * 80)
    
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP file not found: {zip_path}")
        
    sandbox = Path(tempfile.mkdtemp(prefix="gate_audit_"))
    try:
        # Gate 1: ZIP Integrity & File Size
        zip_size_mb = zip_path.stat().st_size / (1024 * 1024)
        log_gate(1, "ZIP Integrity & Size Check", zip_size_mb <= 500.0, f"{zip_size_mb:.2f} MB (Limit: 500MB)")
        
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(sandbox)
            namelist = z.namelist()
            
        # Gate 2: Root Layout Verification
        has_script = "script.py" in namelist
        has_req = "requirements.txt" in namelist
        has_model = any(f.startswith("model/") for f in namelist)
        has_root = has_script and has_req and has_model
        log_gate(2, "Root Layout Verification", has_root, "script.py, requirements.txt, model/ found at root")
        
        # Gate 3: Static AST Security Check
        script_code = (sandbox / "script.py").read_text(encoding="utf-8")
        tree = ast.parse(script_code)
        suspicious = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for n in node.names:
                    if n.name in ["socket", "requests", "urllib", "http", "ftplib"]:
                        suspicious.append(n.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module in ["socket", "requests", "urllib", "http", "ftplib"]:
                    suspicious.append(node.module)
        log_gate(3, "Static AST Security Check", len(suspicious) == 0, f"{len(suspicious)} forbidden network modules")
        
        data_dir = sandbox / "data"
        out_dir = sandbox / "output"
        data_dir.mkdir(exist_ok=True)
        out_dir.mkdir(exist_ok=True)
        
        # Gate 4: Official Dummy Schema Test (Real Schema)
        raw_5 = pd.read_csv(TRAIN_PATH, nrows=5, low_memory=False)
        dummy_df = raw_5.drop(columns=["control_success"], errors="ignore").copy()
        dummy_df["row_id"] = [f"dummy_{i}" for i in range(len(dummy_df))]
        dummy_df.to_csv(data_dir / "test.csv", index=False)
        
        p = subprocess.run([PYTHON_BIN, "script.py"], cwd=str(sandbox), capture_output=True, text=True)
        if p.returncode != 0:
            print("STDERR:", p.stderr)
            raise RuntimeError("Dummy inference failed with non-zero exit code")
        sub_dummy = pd.read_csv(out_dir / "submission.csv")
        p_dum = sub_dummy["control_success"].to_numpy()
        valid_dum = (len(sub_dummy) == 5) and (not np.isnan(p_dum).any()) and (np.all((p_dum >= 0.0) & (p_dum <= 1.0)))
        log_gate(4, "Official Dummy Schema Test", valid_dum, f"5 rows returned, finite in [0, 1]")
        
        # Gate 5: Mixed 1·2군 Real Benchmark
        raw_train = pd.read_csv(TRAIN_PATH, low_memory=False)
        reg_part = raw_train[raw_train["game_type"] == "R"].head(1000).drop(columns=["control_success"], errors="ignore")
        fut_part = raw_train[raw_train["game_type"] == "F"].head(500).drop(columns=["control_success"], errors="ignore")
        sample_df = pd.concat([reg_part, fut_part], ignore_index=True)
        sample_df["row_id"] = [f"real_{i}" for i in range(len(sample_df))]
        sample_df.to_csv(data_dir / "test.csv", index=False)
        
        t0 = time.time()
        p = subprocess.run([PYTHON_BIN, "script.py"], cwd=str(sandbox), capture_output=True, text=True)
        t_sample = time.time() - t0
        if p.returncode != 0:
            print("STDERR:", p.stderr)
            raise RuntimeError("Mixed benchmark failed")
        sub_sample = pd.read_csv(out_dir / "submission.csv")
        p_samp = sub_sample["control_success"].to_numpy()
        valid_sample = (len(sub_sample) == 1500) and (not np.isnan(p_samp).any()) and np.all((p_samp >= 0.0) & (p_samp <= 1.0))
        ms_per_row = (t_sample / 1500.0) * 1000.0
        log_gate(5, "Mixed 1·2군 Real Benchmark", valid_sample, f"1,500 rows in {t_sample:.2f}s ({ms_per_row:.2f} ms/row)")
        
        # Gate 6: Column Permutation Invariance
        perm_cols = list(np.random.RandomState(42).permutation(sample_df.columns))
        sample_df[perm_cols].to_csv(data_dir / "test.csv", index=False)
        p = subprocess.run([PYTHON_BIN, "script.py"], cwd=str(sandbox), capture_output=True, text=True)
        if p.returncode != 0:
            print("PERMUTATION STDERR:", p.stderr)
            raise RuntimeError("Permutation run failed")
        sub_perm = pd.read_csv(out_dir / "submission.csv")
        perm_diff = np.max(np.abs(sub_sample["control_success"].to_numpy() - sub_perm["control_success"].to_numpy()))
        log_gate(6, "Column Permutation Invariance", perm_diff < 1e-9, f"Max |diff| = {perm_diff:.3e}")
        
        # Gate 7: Single-Row Isolated Inference
        sample_rows = sample_df.head(8)
        single_preds = []
        for idx in range(len(sample_rows)):
            sample_rows.iloc[[idx]].to_csv(data_dir / "test.csv", index=False)
            p = subprocess.run([PYTHON_BIN, "script.py"], cwd=str(sandbox), capture_output=True, text=True)
            if p.returncode != 0:
                raise RuntimeError(f"Single row {idx} failed")
            s_out = pd.read_csv(out_dir / "submission.csv")
            single_preds.append(s_out["control_success"].iloc[0])
        orig_preds = sub_sample.iloc[:8]["control_success"].to_numpy()
        single_diff = np.max(np.abs(orig_preds - np.array(single_preds)))
        log_gate(7, "Single-Row Isolated Inference", single_diff < 1e-9, f"Max |diff| across 8 rows = {single_diff:.3e}")
        
        # Gate 8: 5,000-Row Stress Benchmark & Strict <8.0 min Extrapolated Limit
        raw_5k = pd.read_csv(TRAIN_PATH, nrows=5000, low_memory=False)
        raw_5k["row_id"] = [f"stress_{i}" for i in range(len(raw_5k))]
        raw_5k.to_csv(data_dir / "test.csv", index=False)
        
        t0 = time.time()
        p = subprocess.run([PYTHON_BIN, "script.py"], cwd=str(sandbox), capture_output=True, text=True)
        t_5k = time.time() - t0
        if p.returncode != 0:
            print("STDERR:", p.stderr)
            raise RuntimeError("5K stress run failed")
        sub_5k = pd.read_csv(out_dir / "submission.csv")
        if len(sub_5k) != 5000 or sub_5k["control_success"].isna().any():
            raise RuntimeError("5K stress output invalid")
            
        extrapolated_full = (t_5k / 5000.0) * 245789.0 / 60.0
        log_gate(8, "5,000-Row Strict Time Limit", extrapolated_full < 8.0, 
                 f"5,000 rows in {t_5k:.2f}s -> Full 245,789 test: {extrapolated_full:.2f} min (Strict Limit: <8.0 min)")
        
        print("\n" + "=" * 80)
        print(f"🏆 ALL 8 GATES PASSED STRICT AUDIT: {zip_path.name}")
        print("=" * 80)
        return True
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/verify_submission_zip_unified.py <zip_path>")
        sys.exit(1)
    verify_zip(Path(sys.argv[1]))
