#!/usr/bin/env python3
from pathlib import Path
import json,sys
import catboost as cb
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.asof_state_features import add_state_for_cutoff,add_state_walkforward
from src.target_aggregates import build_pitcher_count_state_target_history
from scripts.build_combo_full_candidate_002 import COLS as BASE_COLS,extras,prep,prior_bundle
from scripts.screen_trackman_context_003 import load_tm,context_tables,attach,NEW
OUT=ROOT/'model/COMBO-TM-FULL-006'
def main():
    raw=pd.read_csv(ROOT/'data/train.csv',encoding='utf-8-sig');test=pd.read_csv(ROOT/'data/test.csv',encoding='utf-8-sig');tm,mapdf=load_tm();
    train_base=add_state_walkforward(raw.drop(columns=['row_id','control_success']),2025);test_base=add_state_for_cutoff(test.drop(columns=['row_id']),raw.drop(columns=['row_id','control_success']))
    at,ate,lookup,_=build_pitcher_count_state_target_history(raw,test.assign(control_success=0),smoothing=100.0)
    xt=pd.concat([extras(train_base.reset_index(drop=True)),at.reset_index(drop=True)],axis=1);xe=pd.concat([extras(test_base.reset_index(drop=True)),ate.reset_index(drop=True)],axis=1)
    # Build season-specific train context and retain the 2025 inference tables.
    parts=[]
    for s in sorted(raw.season.unique()):
        ix=raw.season.eq(s); parts.append(attach(xt.loc[ix.to_numpy()].copy(),tm,int(s)))
    xt=pd.concat(parts,axis=0).sort_index();c25,h25=context_tables(tm,2025);xe=xe.copy();xe['__hand']=xe['batter_hand'].map({1:'Left',2:'Right'}).fillna(xe['batter_hand'].astype(str));xe=xe.join(c25,on=['pitcher_id','balls_before','strikes_before']).join(h25,on=['pitcher_id','__hand']).drop(columns='__hand')
    cols=BASE_COLS+NEW;xt=xt[cols];xe=xe[cols];xt,cats=prep(xt);xe,_=prep(xe);m=cb.CatBoostClassifier(iterations=300,learning_rate=.05,depth=6,l2_leaf_reg=5,thread_count=-1,random_seed=42,allow_writing_files=False,verbose=False);m.fit(cb.Pool(xt,label=raw.control_success.to_numpy(),cat_features=cats,feature_names=cols));OUT.mkdir(parents=True,exist_ok=True);prior_bundle(raw,OUT)
    OUT.mkdir(parents=True,exist_ok=True);m.save_model(OUT/'model.cbm');(OUT/'feature_columns.json').write_text(json.dumps(cols,ensure_ascii=False,indent=2));lookup.to_csv(OUT/'pitcher_count_lookup.csv',index=False);c25.reset_index().to_csv(OUT/'trackman_count_lookup.csv',index=False);h25.reset_index().to_csv(OUT/'trackman_hand_lookup.csv',index=False);mapdf.to_csv(OUT/'pitcher_id_map_audit.csv',index=False);pred=m.predict_proba(cb.Pool(xe,cat_features=cats,feature_names=cols))[:,1];pd.DataFrame({'row_id':test.row_id,'control_success':pred}).to_csv(OUT/'test_predictions.csv',index=False);r={'experiment_id':'COMBO-TM-FULL-006','official_train_only':True,'test_used_for_training':False,'external_data_used':False,'mapping_source':'model/TRACKMAN-MAP-004/pitcher_id_map.csv','mapping_rows':len(mapdf),'train_rows':len(raw),'test_rows':len(test),'feature_count':len(cols),'trackman_feature_count':len(NEW),'tree_count':m.tree_count_,'trackman_count_lookup_rows':len(c25),'trackman_hand_lookup_rows':len(h25),'prediction_min':float(pred.min()),'prediction_max':float(pred.max()),'status':'PASS_FULL_TRAIN_MODEL','submission_status':'HOLD'};(OUT/'full_report.json').write_text(json.dumps(r,ensure_ascii=False,indent=2));print(json.dumps(r,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
