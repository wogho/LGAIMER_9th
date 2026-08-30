#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1]
ZIP=ROOT/'output/submit_combo_tm_full_006.zip'; B=ROOT/'model/COMBO-TM-FULL-006'
EXPECTED={'script.py':ROOT/'scripts/infer_combo_tm_full_isolated_006.py','requirements.txt':ROOT/'requirements_submit.txt','model/COMBO-TM-FULL-006/model.cbm':B/'model.cbm','model/COMBO-TM-FULL-006/feature_columns.json':B/'feature_columns.json','model/COMBO-TM-FULL-006/pitcher_count_lookup.csv':B/'pitcher_count_lookup.csv','model/COMBO-TM-FULL-006/asof_pitcher_id_prior.csv':B/'asof_pitcher_id_prior.csv','model/COMBO-TM-FULL-006/asof_batter_id_prior.csv':B/'asof_batter_id_prior.csv','model/COMBO-TM-FULL-006/asof_pitchmix_prior.csv':B/'asof_pitchmix_prior.csv','model/COMBO-TM-FULL-006/trackman_count_lookup.csv':B/'trackman_count_lookup.csv','model/COMBO-TM-FULL-006/trackman_hand_lookup.csv':B/'trackman_hand_lookup.csv','solution/LG_Aimers_솔루션_PPT_Phase2.pptx':ROOT/'output/LG_Aimers_솔루션_PPT_Phase2.pptx'}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    with zipfile.ZipFile(ZIP) as z:
        names=set(z.namelist()); assert names==set(EXPECTED)
        src=z.read('script.py').decode(); bad=[x for x in ['requests','urllib','http://','https://','openai','google.generativeai','train.csv','trackman_history.csv','groupby('] if x in src]; assert not bad
        assert {n:hashlib.sha256(z.read(n)).hexdigest() for n in sorted(names)}=={n:sha(p) for n,p in EXPECTED.items()}
        fc=len(json.loads(z.read('model/COMBO-TM-FULL-006/feature_columns.json'))); assert fc==81
    r={'experiment_id':'COMBO-TM-FULL-ZIP-AUDIT-006','zip_sha256':sha(ZIP),'member_count':len(names),'member_hash_match':True,'ppt_included':True,'feature_count':fc,'forbidden_tokens':bad,'external_api_used':False,'external_data_used':False,'test_row_aggregation_used':False,'status':'AUDIT_VERIFIED','submission_status':'HOLD'}
    (ROOT/'output/submit_combo_tm_full_006_audit.json').write_text(json.dumps(r,ensure_ascii=False,indent=2)); print(json.dumps(r,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
