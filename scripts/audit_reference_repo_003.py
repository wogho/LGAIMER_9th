#!/usr/bin/env python3
"""Audit github_reference/3번 레포 without importing or executing its artifacts."""
from __future__ import annotations
import ast, csv, hashlib, json, re, zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]; REPO=ROOT/'github_reference/3번 레포'; OUT=ROOT/'model/REF-REPO-AUDIT-003'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 files=[p for p in REPO.rglob('*') if p.is_file() and '.git' not in p.parts]
 z=REPO/'submissions/cand_asof_xl.zip'; members=[]
 with zipfile.ZipFile(z) as f: members=f.namelist()
 script=(REPO/'submissions/cand_asof_xl.zip')
 text=(REPO/'script.py').read_text(encoding='utf-8')
 forbidden_tokens=[t for t in ('requests','urllib','httpx','openai','google.generativeai','test.mean','test.groupby','test.rolling') if t in text]
 manifest=(REPO/'SUBMISSION_MANIFEST.csv').read_text(encoding='utf-8')
 score_rows=[line for line in manifest.splitlines() if 'cand_asof_xl.zip' in line]
 report={'experiment_id':'REF-REPO-AUDIT-003','repo':'https://github.com/hoo743-ui/LG_Aimers09','head_commit':'4659b2f','file_count':len(files),'bytes':sum(p.stat().st_size for p in files),'license_files':[p.name for p in REPO.iterdir() if p.name.lower().startswith(('license','copying'))],'latest_candidate':{'path':'submissions/cand_asof_xl.zip','sha256':sha(z),'bytes':z.stat().st_size,'members':members,'manifest_lines':score_rows},'mechanism':{'asof_state_decomposition':True,'current_state_formula':'cur_n = asof_n - prior_n; cur_rate = (asof_n*asof_rate - prior_events)/cur_n','feature_layers':['D current 13','X context interactions 8','H1 level expansion 6'],'catboost':'depth6 l2=100 lr=0.02 iterations=1200 seeds=3 border_count=32','post_adjustment':'four nested deviation tables; fixed alpha=1.09; train-only center'},'official_data_scope':{'trackman_history_used':True,'train_target_used_for_prior_tables':True,'test_batch_aggregation_in_inference':False},'static_script_flags':{'network_or_external_api_tokens':forbidden_tokens,'direct_test_aggregation_tokens':[]},'status':'AUDITED_REFERENCE_ONLY','copy_or_submit':'PROHIBITED'}
 OUT.mkdir(parents=True,exist_ok=True); (OUT/'audit_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
