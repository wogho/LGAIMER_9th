#!/usr/bin/env python3
from pathlib import Path
import json,hashlib
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256();h.update(p.read_bytes());return h.hexdigest()
def main():
    reports=[ROOT/'model/COMBO-E2E-001/e2e_report.json',ROOT/'model/COMBO-E2E-001/provenance_report.json',ROOT/'output/submit_combo_e2e_001_report.json',ROOT/'output/submit_combo_e2e_001_audit.json']
    data={p.stem:json.loads(p.read_text()) for p in reports};rules=ROOT/'01_제약과금지사항.md';z=ROOT/'output/submit_combo_e2e_001.zip'
    out={'readiness_id':'COMBO-UPLOAD-READINESS-001','rule_doc_sha256':sha(rules),'zip_sha256':sha(z),'e2e_status':data['e2e_report'].get('status'),'provenance_status':data['provenance_report'].get('status'),'zip_e2e_status':data['submit_combo_e2e_001_report'].get('status'),'zip_audit_status':data['submit_combo_e2e_001_audit'].get('status'),'active_submission_changed':False,'status':'HOLD','reason':'research candidate; active SUB-002 unchanged'}
    (ROOT/'output/combo_upload_readiness_001.json').write_text(json.dumps(out,ensure_ascii=False,indent=2));print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
