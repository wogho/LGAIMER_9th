#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,shutil,subprocess,tempfile,zipfile
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];B=ROOT/'model/COMBO-TM-FULL-006';OUT=ROOT/'output';OUT.mkdir(exist_ok=True)
FILES={'script.py':ROOT/'scripts/infer_combo_tm_full_isolated_006.py','requirements.txt':ROOT/'requirements_submit.txt','model/COMBO-TM-FULL-006/model.cbm':B/'model.cbm','model/COMBO-TM-FULL-006/feature_columns.json':B/'feature_columns.json','model/COMBO-TM-FULL-006/pitcher_count_lookup.csv':B/'pitcher_count_lookup.csv','model/COMBO-TM-FULL-006/asof_pitcher_id_prior.csv':B/'asof_pitcher_id_prior.csv','model/COMBO-TM-FULL-006/asof_batter_id_prior.csv':B/'asof_batter_id_prior.csv','model/COMBO-TM-FULL-006/asof_pitchmix_prior.csv':B/'asof_pitchmix_prior.csv','model/COMBO-TM-FULL-006/trackman_count_lookup.csv':B/'trackman_count_lookup.csv','model/COMBO-TM-FULL-006/trackman_hand_lookup.csv':B/'trackman_hand_lookup.csv','solution/LG_Aimers_솔루션_PPT_Phase2.pptx':ROOT/'output/LG_Aimers_솔루션_PPT_Phase2.pptx'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 missing=[str(p) for p in FILES.values() if not p.is_file()];assert not missing,missing;zpath=OUT/'submit_combo_tm_full_006.zip'
 with zipfile.ZipFile(zpath,'w',zipfile.ZIP_DEFLATED) as z:
  for n,p in sorted(FILES.items()):i=zipfile.ZipInfo(n,date_time=(2020,1,1,0,0,0));i.compress_type=zipfile.ZIP_DEFLATED;z.writestr(i,p.read_bytes())
 sb=Path(tempfile.mkdtemp(prefix='tm_full_zip_'));(sb/'data').mkdir();(sb/'output').mkdir()
 try:
  with zipfile.ZipFile(zpath) as z:z.extractall(sb)
  shutil.copy2(ROOT/'data/test.csv',sb/'data/test.csv');r=subprocess.run([str(ROOT/'.venv/bin/python'),'script.py'],cwd=sb,capture_output=True,text=True,timeout=180);assert r.returncode==0,r.stderr;p=pd.read_csv(sb/'model/COMBO-TM-FULL-006/isolated_submission.csv');s=pd.read_csv(ROOT/'data/sample_submission.csv');assert len(p)==len(s) and p.row_id.is_unique and p.control_success.between(0,1).all();q={'experiment_id':'COMBO-TM-FULL-ZIP-006','zip':str(zpath),'zip_sha256':sha(zpath),'member_count':len(FILES),'members':sorted(FILES),'ppt_included':True,'sandbox_rows':len(p),'status':'PASS_ZIP_E2E','submission_status':'HOLD'};(OUT/'submit_combo_tm_full_006_report.json').write_text(json.dumps(q,ensure_ascii=False,indent=2));print(json.dumps(q,ensure_ascii=False,indent=2))
 finally:shutil.rmtree(sb,ignore_errors=True)
if __name__=='__main__':main()
