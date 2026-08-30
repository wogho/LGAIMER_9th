#!/usr/bin/env python3
"""Independent verification for REF4-TRAINONLY-R-BASE-EXPERT-059A."""
import hashlib, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'model/REF4-TRAINONLY-R-BASE-EXPERT-059A'

def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def metric(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    b = float(np.mean((p - y) ** 2))
    ref = float(np.mean((y - y.mean()) ** 2))
    q = 1.0 - b / ref
    return {'rows': len(y), 'target_rate': float(y.mean()), 'brier': b, 'bss': q, 'local_score': 1e5 * q}

def main():
    t0 = time.time()
    c = json.loads((OUT / 'audit_contract.json').read_text(encoding='utf-8'))
    res = json.loads((OUT / 'result.json').read_text(encoding='utf-8'))
    manifest = json.loads((OUT / 'audit_manifest.json').read_text(encoding='utf-8'))
    
    checks = []
    
    # 1. Manifest artifact hashes
    mismatch_artifacts = []
    for rel_p, meta in manifest['artifacts'].items():
        full_p = ROOT / rel_p
        if not full_p.exists():
            mismatch_artifacts.append((rel_p, 'missing'))
        else:
            cur_sha = sha256_file(full_p)
            if cur_sha != meta['sha256']:
                mismatch_artifacts.append((rel_p, cur_sha, meta['sha256']))
    checks.append({'name': 'manifest_artifacts', 'pass': len(mismatch_artifacts) == 0, 'mismatches': mismatch_artifacts})
    
    # 2. Recompute metrics from oof_predictions.csv
    oof_p = OUT / 'oof_predictions.csv'
    assert oof_p.exists(), "oof_predictions.csv missing"
    df = pd.read_csv(oof_p)
    assert len(df) == 499032, f"OOF rows mismatch: {len(df)}"
    checks.append({'name': 'oof_row_count', 'pass': True, 'rows': len(df)})
    
    y = df['target'].to_numpy(float)
    base = df['baseline_prediction'].to_numpy(float)
    p051 = df['curr_051_prediction'].to_numpy(float)
    cand = df['candidate_prediction'].to_numpy(float)
    
    m_base = metric(y, base)
    m_051 = metric(y, p051)
    m_cand = metric(y, cand)
    
    gain_base = m_base['brier'] - m_cand['brier']
    gain_051 = m_051['brier'] - m_cand['brier']
    
    diff_gain_base = abs(gain_base - res['pooled']['brier_gain_vs_base'])
    diff_gain_051 = abs(gain_051 - res['pooled']['brier_gain_vs_051'])
    checks.append({'name': 'metric_recalculation', 'pass': diff_gain_base < 1e-12 and diff_gain_051 < 1e-12})
    
    # 3. Model files
    m2023 = OUT / 'r_expert_2023.cbm'
    m2024 = OUT / 'r_expert_2024.cbm'
    checks.append({'name': 'model_files', 'pass': m2023.exists() and m2024.exists()})
    
    # 4. Promotion gate evaluation
    gate_checks = res['gate_checks']
    promotion = all(gate_checks.values())
    checks.append({'name': 'promotion_consistency', 'pass': promotion == res['promotion_pass']})
    
    all_pass = all(chk['pass'] for chk in checks)
    status = 'AUDIT_VERIFIED' if all_pass else 'AUDIT_FAIL'
    
    report = {
        'experiment_id': c['experiment_id'],
        'status': status,
        'checked_count': len(checks),
        'pass_count': sum(1 for chk in checks if chk['pass']),
        'fail_count': sum(1 for chk in checks if not chk['pass']),
        'mismatch_count': len(mismatch_artifacts),
        'checks': checks,
        'pooled_brier_gain_vs_base': gain_base,
        'pooled_brier_gain_vs_051': gain_051,
        'promotion_pass': promotion,
        'elapsed_seconds': time.time() - t0
    }
    
    (OUT / 'validation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    attestation = {
        'experiment_id': c['experiment_id'],
        'overall_status': status,
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'validation_report_sha256': sha256_file(OUT / 'validation_report.json'),
        'promotion_pass': promotion
    }
    (OUT / 'audit_attestation.json').write_text(json.dumps(attestation, ensure_ascii=False, indent=2) + '\n')
    print(f"Verification completed: status={status}, promotion_pass={promotion}")

if __name__ == '__main__':
    main()
