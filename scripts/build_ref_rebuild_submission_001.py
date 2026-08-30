#!/usr/bin/env python3
"""Build a research-only REF-REBUILD-001 submission package (not submitted)."""
from __future__ import annotations
import json, shutil, sys, zipfile
from pathlib import Path
import catboost as cb
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.ref_rebuild_features import CAT_COLS, FEATURE_COLUMNS, engineer, prepare
OUT=ROOT/"candidate/REF-REBUILD-001"; ZIP=ROOT/"output/candidates/submit_ref_rebuild_001.zip"
def main():
 raw=pd.read_csv(ROOT/"data/train.csv",encoding="utf-8-sig"); y=raw.control_success.to_numpy(np.int8); gm=float(y.mean())
 x=prepare(engineer(raw.drop(columns=["row_id","control_success"]),gm)); ci=[x.columns.get_loc(c) for c in CAT_COLS]; pool=cb.Pool(x,label=y,cat_features=ci,feature_names=FEATURE_COLUMNS)
 labels=pd.read_csv(ROOT/"model/REF-AUX-LABEL-001/recovered_labels.csv.gz",usecols=["middle","reverse"]); ym=((labels.middle.eq(1))|(labels.reverse.eq(1))).to_numpy(np.int8); yw=((y==0)&(ym==0)).astype(np.int8)
 if OUT.exists(): shutil.rmtree(OUT)
 (OUT/"model").mkdir(parents=True); (OUT/"data").mkdir()
 for name,target,it in (("success",y,260),("mr",ym,330),("wayoff",yw,60)):
  p=cb.CatBoostClassifier(iterations=it,learning_rate=.05,depth=6,loss_function="Logloss",thread_count=12,random_seed=42,allow_writing_files=False,verbose=False)
  p.fit(cb.Pool(x,label=target,cat_features=ci,feature_names=FEATURE_COLUMNS),verbose=False); p.save_model(str(OUT/"model"/f"{name}.cbm"))
 (OUT/"feature_columns.json").write_text(json.dumps(FEATURE_COLUMNS,ensure_ascii=False,indent=2),encoding="utf-8")
 (OUT/"contract.json").write_text(json.dumps({"experiment_id":"REF-REBUILD-001","feature_count":57,"categorical_features":CAT_COLS,"global_mean":gm,"models":{"success":"model/success.cbm","mr":"model/mr.cbm","wayoff":"model/wayoff.cbm"}},ensure_ascii=False,indent=2),encoding="utf-8")
 runtime='''import json, os\nfrom pathlib import Path\nimport numpy as np, pandas as pd\nfrom catboost import CatBoostClassifier\nCAT_COLS=%r\nBASE=%r\nFEATURES=%r\ndef eng(f,gm):\n d=f.loc[:,BASE].copy()\n for w in ("pitcher","batter"):\n  n=d[f"asof_{w}_n"].fillna(0.); r=d[f"asof_{w}_success_rate"].fillna(gm); d[f"smoothed_{w}_success_rate"]=(n*r+30*gm)/(n+30)\n d["platoon_advantage"]=(d.pitcher_hand.astype("string")==d.batter_hand.astype("string")).astype("int8"); d["count_advantage"]=d.strikes_before-d.balls_before; d["count_state"]=d.balls_before.astype("string")+"-"+d.strikes_before.astype("string"); d["recent_control_momentum"]=d.asof_pitcher_prev1_game_success_rate-d.asof_pitcher_success_rate; d["form_trend_5_1"]=d.asof_pitcher_prev1_game_success_rate-d.asof_pitcher_prev5_game_success_rate; d["is_home"]=d.top_bottom.astype("string").eq("T").astype("int8"); d["pitcher_win_expectancy"]=np.where(d.is_home.eq(1),d.home_win_expectancy,d.away_win_expectancy); d["is_coldstart_pitcher"]=d.asof_pitcher_n.isna().astype("int8")\n for c in CAT_COLS:d[c]=d[c].astype("string").fillna("<NA>").astype(str)\n return d\ndef main():\n root=Path(__file__).resolve().parent; test=pd.read_csv(root/"data/test.csv",encoding="utf-8-sig"); sample=pd.read_csv(root/"data/sample_submission.csv",encoding="utf-8-sig"); contract=json.loads((root/"contract.json").read_text()); x=eng(test.drop(columns=["row_id"]),contract["global_mean"]); ci=[x.columns.get_loc(c) for c in CAT_COLS]; ps=[]\n for n in ("success","mr","wayoff"):\n  m=CatBoostClassifier(); m.load_model(str(root/contract["models"][n])); ps.append(m.predict_proba(x)[:,1])\n p=np.clip(ps[0],1e-6,1-1e-6); z=np.log(p/(1-p))-0.05*(ps[1]-ps[1].mean())+0.05*(ps[2]-ps[2].mean()); out=sample.copy(); out["control_success"]=1/(1+np.exp(-z)); out.to_csv(root/"output/submission.csv",index=False)\nif __name__=="__main__":main()\n''' % (CAT_COLS, __import__('src.ref_rebuild_features',fromlist=['BASE_COLUMNS']).BASE_COLUMNS, FEATURE_COLUMNS)
 (OUT/"script.py").write_text(runtime,encoding="utf-8"); (OUT/"requirements.txt").write_text("catboost==1.2.10\nnumpy==1.26.4\npandas==2.0.3\n",encoding="utf-8")
 (OUT/"data/test.csv").write_bytes((ROOT/"data/test.csv").read_bytes()); (OUT/"data/sample_submission.csv").write_bytes((ROOT/"data/sample_submission.csv").read_bytes())
 ZIP.parent.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(ZIP,"w",zipfile.ZIP_DEFLATED) as z:
  for p in sorted(OUT.rglob("*")):
   if p.is_file(): z.write(p,p.relative_to(OUT))
 print(json.dumps({"zip":str(ZIP),"bytes":ZIP.stat().st_size,"models":3},indent=2))
if __name__=="__main__":main()
