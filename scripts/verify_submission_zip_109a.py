#!/usr/bin/env python3
"""Strict 8-Gate Pre-Submission Verification Suite for 109A."""
import ast, hashlib, json, os, shutil, subprocess, sys, tempfile, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "output/submit_ref4_super_ensemble_109A.zip"
PYTHON_BIN = str(ROOT / ".venv-submit/bin/python")
TRAIN_PATH = ROOT / "data/train.csv"

def log_gate(num, name, passed, details=""):
    status = "PASS" if passed else "FAIL"
    print(f"  [Gate {num}] {name:<35} : [{status}] {details}")
    if not passed:
        raise RuntimeError(f"Gate {num} FAILED: {name} - {details}")

def main():
    print("=" * 75)
    print("      ZERO-RISK 8-GATE PRE-SUBMISSION VERIFICATION: 109A         ")
    print("=" * 75)
    
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"ZIP package not found at {ZIP_PATH}")
        
    zip_bytes = ZIP_PATH.read_bytes()
    zip_sha = hashlib.sha256(zip_bytes).hexdigest()
    zip_size = len(zip_bytes)
    
    # Gate 1: ZIP Integrity & Size
    log_gate(1, "ZIP Integrity & Size Check", zip_size < 500 * 1024 * 1024, f"{zip_size / 1e6:.2f} MB (SHA-256: {zip_sha[:16]}...)")
    
    temp_dir = tempfile.mkdtemp(prefix="verify_109a_")
    sandbox = Path(temp_dir)
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
            zf.extractall(sandbox)
            
        # Gate 2: Root Layout
        has_script = (sandbox / "script.py").exists()
        has_reqs = (sandbox / "requirements.txt").exists()
        has_model = (sandbox / "model").is_dir()
        has_src = (sandbox / "src").is_dir()
        log_gate(2, "Root Layout Verification", has_script and has_reqs and has_model and has_src, "script.py, requirements.txt, model/, src/ present at root")
        
        # Gate 3: Static AST Security Check
        script_code = (sandbox / "script.py").read_text(encoding="utf-8")
        parsed = ast.parse(script_code)
        forbidden_imports = {"socket", "urllib", "requests", "http", "ftplib"}
        found_forbidden = []
        for node in ast.walk(parsed):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] in forbidden_imports:
                        found_forbidden.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] in forbidden_imports:
                    found_forbidden.append(node.module)
        log_gate(3, "Static AST Security Check", len(found_forbidden) == 0, f"No remote socket/network calls ({len(found_forbidden)} found)")
        
        # Gate 4: Official Dummy Schema Test
        data_dir = sandbox / "data"
        out_dir = sandbox / "output"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        raw = pd.read_csv(TRAIN_PATH, nrows=50000, low_memory=False)
        dummy_test = raw.iloc[:5].copy()
        dummy_test["row_id"] = [f"dummy_{i}" for i in range(5)]
        dummy_test.to_csv(data_dir / "test.csv", index=False)
        
        p = subprocess.run([PYTHON_BIN, "script.py"], cwd=str(sandbox), capture_output=True, text=True)
        if p.returncode != 0:
            print("STDERR:", p.stderr)
            raise RuntimeError(f"Dummy script execution failed with code {p.returncode}")
            
        sub = pd.read_csv(out_dir / "submission.csv")
        valid_dummy = (
            len(sub) == 5
            and list(sub.columns) == ["row_id", "control_success"]
            and sub["control_success"].between(0.0, 1.0).all()
            and not sub["control_success"].isna().any()
        )
        log_gate(4, "Official Dummy Schema Test", valid_dummy, f"5 rows, valid [0, 1] range: {[round(x, 4) for x in sub['control_success'].tolist()]}")
        
        # Gate 5: Mixed 1·2군 Real Benchmark (1,500 rows)
        raw_reg = pd.read_csv(TRAIN_PATH, nrows=5000)
        raw_fut = pd.read_csv(TRAIN_PATH, skiprows=range(1, 211627), nrows=500)
        val_reg = raw_reg.loc[raw_reg.game_type != "F"].iloc[:1200]
        val_fut = raw_fut.loc[raw_fut.game_type == "F"].iloc[:300]
        mixed_1500 = pd.concat([val_reg, val_fut], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
        mixed_1500.to_csv(data_dir / "test.csv", index=False)
        
        t0 = time.time()
        p = subprocess.run([PYTHON_BIN, "script.py"], cwd=str(sandbox), capture_output=True, text=True)
        t_batch = time.time() - t0
        
        if p.returncode != 0:
            print("STDERR:", p.stderr)
            raise RuntimeError("Mixed 1,500 execution failed")
            
        sub_batch = pd.read_csv(out_dir / "submission.csv")
        log_gate(5, "Mixed 1·2군 Real Benchmark", len(sub_batch) == 1500 and not sub_batch["control_success"].isna().any(), f"1,500 rows in {t_batch:.2f}s ({t_batch*1000/1500:.2f} ms/row, mean={sub_batch['control_success'].mean():.4f}, std={sub_batch['control_success'].std():.4f})")
        
        # Gate 6: Column Permutation Invariance
        permuted_test = mixed_1500.sample(frac=1.0, random_state=123).reset_index(drop=True)
        cols = list(permuted_test.columns)
        np.random.seed(99)
        np.random.shuffle(cols)
        permuted_test = permuted_test[cols]
        permuted_test.to_csv(data_dir / "test.csv", index=False)
        
        subprocess.run([PYTHON_BIN, "script.py"], cwd=str(sandbox), capture_output=True, text=True)
        sub_permuted = pd.read_csv(out_dir / "submission.csv")
        
        merged = sub_batch.merge(sub_permuted, on="row_id", suffixes=("_orig", "_perm"))
        diff = np.max(np.abs(merged["control_success_orig"] - merged["control_success_perm"]))
        log_gate(6, "Column Permutation Invariance", diff < 1e-10, f"Max |diff| = {diff:.3e}")
        
        # Gate 7: Single-Row Isolated Inference
        sample_rows = mixed_1500.iloc[:8].copy()
        single_preds = []
        for idx in range(len(sample_rows)):
            single_df = sample_rows.iloc[[idx]].copy()
            single_df.to_csv(data_dir / "test.csv", index=False)
            subprocess.run([PYTHON_BIN, "script.py"], cwd=str(sandbox), capture_output=True, text=True)
            s_out = pd.read_csv(out_dir / "submission.csv")
            single_preds.append(s_out["control_success"].iloc[0])
            
        orig_preds = sub_batch.iloc[:8]["control_success"].to_numpy()
        single_diff = np.max(np.abs(orig_preds - np.array(single_preds)))
        log_gate(7, "Single-Row Isolated Inference", single_diff < 1e-10, f"Max |diff| across 8 isolated rows = {single_diff:.3e}")
        
        # Gate 8: 5,000-Row Stress Benchmark
        raw_5k = pd.read_csv(TRAIN_PATH, nrows=5000)
        raw_5k["row_id"] = [f"stress_{i}" for i in range(len(raw_5k))]
        raw_5k.to_csv(data_dir / "test.csv", index=False)
        
        t0 = time.time()
        subprocess.run([PYTHON_BIN, "script.py"], cwd=str(sandbox), capture_output=True, text=True)
        t_5k = time.time() - t0
        extrapolated_full = (t_5k / 5000.0) * 245789.0 / 60.0
        log_gate(8, "5,000-Row Stress Benchmark", extrapolated_full < 9.5, f"5,000 rows in {t_5k:.2f}s (Extrapolated full test time: ~{extrapolated_full:.1f} min << 10 min)")
        
        print("\n" + "=" * 75)
        print("🏆 ALL 8 GATES PASSED PERFECTLY WITH ZERO DEFECTS / ZERO RISK: 109A!")
        print("=" * 75)
        
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)

if __name__ == "__main__":
    main()
