#!/usr/bin/env python3
"""Evidence audit for REF-REBUILD-001; does not create a submission."""
from __future__ import annotations
import ast, hashlib, json, platform, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"model/REF-REBUILD-001"
sys.path.insert(0, str(ROOT))
from src.ref_rebuild_features import BASE_COLUMNS, FEATURE_COLUMNS, CAT_COLS, engineer, prepare
def sha(p):
 h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()
def main():
 tr=pd.read_csv(ROOT/"data/train.csv",nrows=5); te=pd.read_csv(ROOT/"data/test.csv",nrows=5); sub=pd.read_csv(ROOT/"data/sample_submission.csv",nrows=5)
 src=(ROOT/"scripts/train_ref_rebuild_multiseed_001.py").read_text(encoding="utf-8")
 tree=ast.parse(src); imports=[]
 for n in ast.walk(tree):
  if isinstance(n,ast.Import): imports += [x.name for x in n.names]
  elif isinstance(n,ast.ImportFrom): imports.append(n.module or "")
 forbidden=[x for x in imports if x.split('.')[0] in {"requests","urllib","httpx","selenium","openai","google"}]
 npz=np.load(OUT/"multiseed_valid_predictions.npz")
 nvalid=253507; lengths={k:int(len(npz[k])) for k in npz.files}
 # A direct row-independence check for the feature contract: changing one input row must not alter another.
 sample=pd.read_csv(ROOT/"data/train.csv",nrows=8).drop(columns=["row_id","control_success"])
 a=prepare(engineer(sample,0.527458013621701)); b=sample.copy(); b.loc[0,"balls_before"]=(int(b.loc[0,"balls_before"])+1)%4
 bb=prepare(engineer(b,0.527458013621701)); other_equal=bool(a.iloc[1:].reset_index(drop=True).equals(bb.iloc[1:].reset_index(drop=True)))
 report={"experiment_id":"REF-REBUILD-001-AUDIT","environment":{"python":platform.python_version()},"schema":{"train_columns":int(len(tr.columns)),"test_columns":int(len(te.columns)),"sample_columns":int(len(sub.columns)),"train_test_input_set_equal":set(tr.columns)-{"control_success"}==set(te.columns)},"feature_contract":{"base_count":len(BASE_COLUMNS),"feature_count":len(FEATURE_COLUMNS),"categorical":CAT_COLS},"static":{"network_imports":forbidden,"test_literals_in_training_script":"data/test.csv" in src or "sample_submission" in src},"prediction_artifact":{"expected_valid_rows":nvalid,"lengths":lengths,"all_lengths_match":all(v==nvalid for v in lengths.values()),"finite_and_unit_interval":all(np.isfinite(npz[k]).all() and ((npz[k]>=0)&(npz[k]<=1)).all() for k in npz.files)},"row_independence":{"single_row_perturbation_other_rows_unchanged":other_equal},"source_hashes":{"train":sha(ROOT/"data/train.csv"),"test":sha(ROOT/"data/test.csv"),"multiseed_report":sha(OUT/"multiseed_report.json")},"status":"PASS_AUDIT" if (not forbidden and not ("data/test.csv" in src) and other_equal and all(v==nvalid for v in lengths.values())) else "FAIL_AUDIT","submission_status":"HOLD"}
 (OUT/"audit_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
