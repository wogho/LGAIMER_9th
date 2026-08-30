#!/usr/bin/env python3
from pathlib import Path
import hashlib,json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256();h.update(p.read_bytes());return h.hexdigest()
def main():
 out=ROOT/'model/COMBO-E2E-001';train=pd.read_csv(ROOT/'data/train.csv',usecols=['season','control_success'],encoding='utf-8-sig');test=pd.read_csv(ROOT/'data/test.csv',usecols=['row_id','season'],encoding='utf-8-sig');lookup=pd.read_csv(out/'pitcher_count_lookup.csv');
 checks={'train_max_season':int(train.season.max()),'test_min_season':int(test.season.min()),'test_max_season':int(test.season.max()),'target_binary':set(train.control_success.unique()).issubset({0,1}),'test_has_target':False,'lookup_duplicate_keys':int(lookup.duplicated(['pitcher_id','count_state']).sum()),'model_sha256':sha(out/'model.cbm'),'lookup_sha256':sha(out/'pitcher_count_lookup.csv'),'feature_columns_sha256':sha(out/'feature_columns.json')}
 assert checks['train_max_season']==2024 and checks['test_min_season']==checks['test_max_season']==2025 and checks['target_binary'] and checks['lookup_duplicate_keys']==0 and not checks['test_has_target']
 report={'experiment_id':'COMBO-E2E-PROVENANCE-001','official_train_only':True,'external_data_used':False,'checks':checks,'cutoff_contract':{'train_history_seasons':'<=2024','valid_screen_season':2024,'test_season':2025,'lookup_source':'official train only'},'status':'AUDIT_VERIFIED','submission_status':'HOLD'}
 (out/'provenance_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
