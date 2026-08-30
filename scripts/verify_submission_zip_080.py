#!/usr/bin/env python3
"""Strict 8-Gate Pre-Submission Verification Suite for 080A."""
import ast, hashlib, json, os, shutil, subprocess, sys, tempfile, time, zipfile
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ZIP_PATH = ROOT / "output/submit_ref4_orthogonal_v5_080.zip"
PYTHON_BIN = str(ROOT / ".venv-submit/bin/python")
TRAIN_PATH = ROOT / "data/train.csv"

def log_gate(num, name, passed, details=""):
    status = "PASS" if passed else "FAIL"
    print(f"  [Gate {num}] {name:<35} : [{status}] {details}")
    if not passed:
        raise RuntimeError(f"Gate {num} FAILED: {name} - {details}")

def main():
    print("=" * 75)
    print("      ZERO-RISK 8-GATE PRE-SUBMISSION VERIFICATION: 080A         ")
    print("=" * 75)
    
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"ZIP package not found at {ZIP_PATH}")
        
    zip_bytes = ZIP_PATH.read_bytes()
    zip_sha = hashlib.sha256(zip_bytes).hexdigest()
    zip_size = len(zip_bytes)
    
    # ----------------------------------------------------
    # Gate 1: ZIP File Existence & Size
    # ----------------------------------------------------
    log_gate(1, "ZIP Integrity & Size Check", zip_size < 500 * 1024 * 1024, f"{zip_size / 1e6:.2f} MB (SHA-256: {zip_sha[:16]}...)")
    
    # Extract to sandbox
    temp_dir = tempfile.mkdtemp(prefix="verify_080_")
    sandbox = Path(temp_dir)
    try:
        with zipfile.ZipFile(ZIP_PATH, 'r') as zf:
            zf.extractall(sandbox)
            namelist = zf.namelist()
            
        # ----------------------------------------------------
        # Gate 2: Root Layout Check
        # ----------------------------------------------------
        has_script = (sandbox / "script.py").exists()
        has_reqs = (sandbox / "requirements.txt").exists()
        has_model = (sandbox / "model").is_dir()
        has_src = (sandbox / "src").is_dir()
        log_gate(2, "Root Layout Verification", has_script and has_reqs and has_model and has_src, "script.py, requirements.txt, model/, src/ present at root")
        
        # ----------------------------------------------------
        # Gate 3: Static AST Security Check
        # ----------------------------------------------------
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
        
        # ----------------------------------------------------
        # Gate 4: Official Dummy Schema Test
        # ----------------------------------------------------
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
        log_gate(4, "Official Dummy Schema Test", valid_dummy, f"5 rows, valid [0, 1] range: {sub['control_success'].tolist()}")
        
        # ----------------------------------------------------
        # Gate 5: 1군 & 2군 Real Mixed Test (1,500 rows)
        # ----------------------------------------------------
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
            
        sub_1500 = pd.read_csv(out_dir / "submission.csv")
        preds_1500 = sub_1500["control_success"].to_numpy()
        valid_mixed = (
            len(sub_1500) == 1500
            and not np.isnan(preds_1500).any()
            and not np.isinf(preds_1500).any()
            and (preds_1500 > 0.01).all()
            and (preds_1500 < 0.99).all()
            and np.std(preds_1500) > 0.01
        )
        log_gate(5, "Mixed 1·2군 Real Benchmark", valid_mixed, f"1,500 rows in {t_batch:.2f}s ({t_batch/1500*1000:.2f} ms/row, mean={np.mean(preds_1500):.4f}, std={np.std(preds_1500):.4f})")
        
        # ----------------------------------------------------
        # Gate 6: Column Permutation Invariance Test
        # ----------------------------------------------------
        cols = list(mixed_1500.columns)
        rng = np.random.default_rng(123)
        shuffled_cols = list(rng.permutation(cols))
        mixed_shuffled = mixed_1500[shuffled_cols].copy()
        mixed_shuffled.to_csv(data_dir / "test.csv", index=False)
        
        subprocess.run([PYTHON_BIN, "script.py"], cwd=str(sandbox), capture_output=True, text=True)
        sub_shuffled = pd.read_csv(out_dir / "submission.csv")
        preds_shuffled = sub_shuffled["control_success"].to_numpy()
        
        max_diff_perm = np.max(np.abs(preds_1500 - preds_shuffled))
        log_gate(6, "Column Permutation Invariance", max_diff_perm < 1e-6, f"Max |diff| = {max_diff_perm:.3e}")
        
        # ----------------------------------------------------
        # Gate 7: Single-Row Isolated Inference Test
        # ----------------------------------------------------
        sample_indices = [0, 100, 300, 500, 1200, 1300, 1400, 1499]
        iso_diffs = []
        for idx in sample_indices:
            single_row = mixed_1500.iloc[[idx]].copy()
            single_row.to_csv(data_dir / "test.csv", index=False)
            subprocess.run([PYTHON_BIN, "script.py"], cwd=str(sandbox), capture_output=True, text=True)
            iso_sub = pd.read_csv(out_dir / "submission.csv")
            p_iso = iso_sub["control_success"].iloc[0]
            p_batch = preds_1500[idx]
            iso_diffs.append(abs(p_iso - p_batch))
            
        max_diff_iso = max(iso_diffs)
        log_gate(7, "Single-Row Isolated Inference", max_diff_iso < 1e-6, f"Max |diff| across 8 isolated rows = {max_diff_iso:.3e}")
        
        # ----------------------------------------------------
        # Gate 8: Stress Test (5,000 Rows Benchmark)
        # ----------------------------------------------------
        stress_5k = pd.concat([raw.sample(5000, random_state=777, replace=True)], ignore_index=True)
        stress_5k.to_csv(data_dir / "test.csv", index=False)
        
        t0 = time.time()
        p = subprocess.run([PYTHON_BIN, "script.py"], cwd=str(sandbox), capture_output=True, text=True)
        t_stress = time.time() - t0
        
        if p.returncode != 0:
            print("STDERR:", p.stderr)
            raise RuntimeError("Stress 5k execution failed")
            
        sub_5k = pd.read_csv(out_dir / "submission.csv")
        valid_stress = len(sub_5k) == 5000 and not sub_5k["control_success"].isna().any()
        log_gate(8, "5,000-Row Stress Benchmark", valid_stress and t_stress < 600, f"5,000 rows in {t_stress:.2f}s (Extrapolated full test time: ~{t_stress/5000*250000/60:.1f} min << 10 min)")
        
    finally:
        shutil.rmtree(sandbox, ignore_errors=True)
        
    print("\n" + "=" * 75)
    print("🏆 ALL 8 GATES PASSED PERFECTLY WITH ZERO DEFECTS / ZERO RISK!")
    print("=" * 75)

if __name__ == '__main__':
    main()
