#!/usr/bin/env python3
"""Build a research-only residual-3 ZIP from official train OOF tables."""
from pathlib import Path
import hashlib, json, zipfile
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'candidate/COMBO-SPREAD-005'
EXP=ROOT/'model/COMBO-RESID3-OOF-007'
OUT=ROOT/'output/submit_combo_resid3_007.zip'
WEIGHT=0.10
AXES=[('hand',['pitcher_id','pitcher_hand','batter_hand'],1000.0),('strikes',['pitcher_id','strikes_before'],1000.0),('runners',['pitcher_id','num_runners_on'],2000.0)]

def sha(p):
 h=hashlib.sha256(); h.update(p.read_bytes()); return h.hexdigest()
def main():
 o=pd.read_csv(EXP/'oof_predictions.csv'); o['resid']=o.target-o.pred
 table_files=[]
 for name,key,k in AXES:
  g=o.groupby(key,dropna=False,observed=True)['resid'].agg(['mean','count']); g['adj']=g['mean']*g['count']/(g['count']+k); p=EXP/f'resid_{name}.csv'; g[['adj']].reset_index().to_csv(p,index=False); table_files.append((name,p))
 src=(BASE/'combo_infer.py').read_text(encoding='utf-8')
 marker="p=model.predict_proba(cb.Pool(x,cat_features=cats,feature_names=COLS))[:,1];"
 if marker not in src: raise RuntimeError('baseline inference marker missing')
 inject=marker+"\n for fn,key in [('resid_hand',['pitcher_id','pitcher_hand','batter_hand']),('resid_strikes',['pitcher_id','strikes_before']),('resid_runners',['pitcher_id','num_runners_on'])]:\n  q=pd.read_csv(HERE/f'{fn}.csv'); o=o.merge(q,on=key,how='left',validate='many_to_one'); p=np.clip(p+0.10*o.adj.fillna(0).to_numpy(),0,1); o=o.drop(columns='adj')\n "
 script=src.replace(marker,inject)
 files={'script.py':script.encode(),'requirements.txt':(BASE/'requirements.txt').read_bytes()}
 for p in sorted((BASE/'model').rglob('*')):
  if p.is_file(): files[f'model/{p.relative_to(BASE/"model")}']=p.read_bytes()
 for name,p in table_files: files[f'resid_{name}.csv']=p.read_bytes()
 ppt=ROOT/'output/LG_Aimers_솔루션_PPT_Phase2.pptx'
 if ppt.is_file(): files['LG_Aimers_솔루션_PPT_Phase2.pptx']=ppt.read_bytes()
 with zipfile.ZipFile(OUT,'w',zipfile.ZIP_DEFLATED) as z:
  for name,data in sorted(files.items()): z.writestr(name,data)
 with zipfile.ZipFile(OUT) as z: assert z.testzip() is None
 manifest={'experiment_id':'COMBO-RESID3-OOF-007','zip_path':str(OUT),'zip_sha256':sha(OUT),'members':sorted(files),'official_train_only':True,'fixed_weight':WEIGHT,'submission_status':'HOLD','audit_status':'PENDING_INDEPENDENT_ZIP_AUDIT'}
 (OUT.with_suffix('.manifest.json')).write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8'); print(json.dumps(manifest,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
