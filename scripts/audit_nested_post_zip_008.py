#!/usr/bin/env python3
from pathlib import Path
import hashlib,json,zipfile
ROOT=Path(__file__).resolve().parents[1];ZIP=ROOT/'output/submit_nested_post_008.zip'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 with zipfile.ZipFile(ZIP) as z:
  names=set(z.namelist());src=z.read('script.py').decode();bad=[x for x in ['requests','urllib','http://','https://','openai','google.generativeai','train.csv','trackman_history.csv','groupby('] if x in src];assert not bad;assert len(names)==15;assert 'solution/LG_Aimers_솔루션_PPT_Phase2.pptx' in names;fc=len(json.loads(z.read('model/COMBO-TM-POST-FULL-008/feature_columns.json')));assert fc==81
 r={'experiment_id':'NESTED-POST-ZIP-AUDIT-008','zip_sha256':sha(ZIP),'member_count':len(names),'ppt_included':True,'feature_count':fc,'forbidden_tokens':bad,'external_api_used':False,'external_data_used':False,'test_row_aggregation_used':False,'status':'AUDIT_VERIFIED','submission_status':'HOLD'};(ROOT/'output/submit_nested_post_008_audit.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps(r,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
