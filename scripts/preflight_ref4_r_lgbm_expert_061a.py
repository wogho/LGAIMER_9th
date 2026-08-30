#!/usr/bin/env python3
"""Preflight audit for REF4-TRAINONLY-R-LGBM-EXPERT-061A."""
import hashlib, json, sys, time
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'model/REF4-TRAINONLY-R-LGBM-EXPERT-061A'

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def main():
    t0 = time.time()
    contract_path = OUT / 'audit_contract.json'
    assert contract_path.exists(), "audit_contract.json missing"
    c = json.loads(contract_path.read_text(encoding='utf-8'))
    
    checks = []
    
    # 1. Train integrity
    train_path = ROOT / c['official_train']
    assert train_path.exists(), "train.csv missing"
    train_df = pd.read_csv(train_path, usecols=['row_id', 'season', 'game_type', 'control_success', 'pitcher_id'])
    assert len(train_df) == 1475092, f"Train row count mismatch: {len(train_df)}"
    checks.append({'check': 'train_rows', 'pass': True, 'count': len(train_df)})
    
    # 2. Base OOF files
    for yr, rel_p in c['base_oof'].items():
        p = ROOT / rel_p
        assert p.exists(), f"Base OOF missing: {p}"
        oof_df = pd.read_csv(p, usecols=['row_id', 'season', 'game_type', 'target', 'pitcher_id'])
        checks.append({'check': f'base_oof_{yr}', 'pass': True, 'rows': len(oof_df), 'sha': sha256_file(p)})
        
    # 3. 051A baseline predictions
    p_051 = ROOT / c['curr_051_predictions']
    assert p_051.exists(), f"051 predictions missing: {p_051}"
    df_051 = pd.read_csv(p_051)
    checks.append({'check': '051_predictions', 'pass': True, 'rows': len(df_051), 'sha': sha256_file(p_051)})
    
    # 4. F psych predictions
    p_f = ROOT / c['f_psych_predictions']
    assert p_f.exists(), f"F psych predictions missing: {p_f}"
    df_f = pd.read_csv(p_f)
    checks.append({'check': 'f_psych_predictions', 'pass': True, 'rows': len(df_f), 'sha': sha256_file(p_f)})
    
    # 5. Preserved champion zip
    zip_p = ROOT / c['preserve_zip']
    assert zip_p.exists(), f"Preserve zip missing: {zip_p}"
    checks.append({'check': 'preserve_zip', 'pass': True, 'sha': sha256_file(zip_p)})
    
    report = {
        'experiment_id': c['experiment_id'],
        'status': 'AUDIT_VERIFIED',
        'checked_count': len(checks),
        'pass_count': len(checks),
        'fail_count': 0,
        'checks': checks,
        'elapsed_seconds': time.time() - t0
    }
    (OUT / 'preflight_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    (OUT / 'preflight_report.md').write_text(f"# Preflight Report: {c['experiment_id']}\n- status: `AUDIT_VERIFIED`\n- checks: {len(checks)}/{len(checks)} PASS\n")
    print(f"061A Preflight completed: {len(checks)} checks PASS in {time.time()-t0:.2f}s")

if __name__ == '__main__':
    main()
