#!/usr/bin/env python3
"""One-seed forward transition screen for REF-REBUILD-001."""
from __future__ import annotations
import json, platform, sys
from pathlib import Path
import catboost as cb
import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.ref_rebuild_features import CAT_COLS, FEATURE_COLUMNS, engineer, prepare
TRAIN = ROOT / "data" / "train.csv"
LABELS = ROOT / "model" / "REF-AUX-LABEL-001" / "recovered_labels.csv.gz"
OUT = ROOT / "model" / "REF-REBUILD-001"

def logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6); return np.log(p / (1-p))
def brier(y,p): return float(np.mean((np.asarray(y)-np.asarray(p))**2))
def fit_offset(ps, pm, pw, y, mm, mw):
    z,u,v = logit(ps), logit(pm)-mm, logit(pw)-mw
    def nll(w):
        q = 1/(1+np.exp(-(z+w[0]*u+w[1]*v))); q=np.clip(q,1e-8,1-1e-8)
        return float(-np.mean(y*np.log(q)+(1-y)*np.log(1-q)))
    return minimize(nll,[0.,0.],method="Nelder-Mead").x

def main():
    raw = pd.read_csv(TRAIN, encoding="utf-8-sig")
    lab = pd.read_csv(LABELS, usecols=["row_id","middle","reverse"])
    if not raw.row_id.equals(lab.row_id): raise AssertionError("row order")
    ys = raw.control_success.to_numpy(np.int8)
    valid_lab = lab[["middle","reverse"]].notna().all(axis=1).to_numpy()
    ym = ((lab.middle.eq(1)) | (lab.reverse.eq(1))).to_numpy(np.int8)
    yw = ((ys==0)&(ym==0)).astype(np.int8)
    preds = {}
    for season in (2022, 2023, 2024):
        tr = raw.season.lt(season).to_numpy() & valid_lab; va = raw.season.eq(season).to_numpy()
        gm=float(ys[tr].mean()); x=prepare(engineer(raw.drop(columns=["row_id","control_success"]),gm))
        tx,vx=x.loc[tr],x.loc[va]; ci=[x.columns.get_loc(c) for c in CAT_COLS]
        out={}
        for name,y in (("success",ys),("mr",ym),("wayoff",yw)):
            tp=cb.Pool(tx,label=y[tr],cat_features=ci,feature_names=FEATURE_COLUMNS)
            vp=cb.Pool(vx,label=y[va],cat_features=ci,feature_names=FEATURE_COLUMNS)
            m=cb.CatBoostClassifier(iterations=180,learning_rate=.05,depth=6,loss_function="Logloss",eval_metric="Logloss",thread_count=12,random_seed=42,allow_writing_files=False,verbose=False,early_stopping_rounds=40)
            m.fit(tp,eval_set=vp,verbose=False); out[name]=m.predict_proba(vp)[:,1]
        preds[season]={"pred":out,"y":ys[va]}
    trans=[]
    for src,tgt in ((2022,2023),(2023,2024)):
        # offset is fitted on source predictions and applied unchanged to target.
        s,t=preds[src],preds[tgt]
        mm=float(logit(s["pred"]["mr"]).mean()); mw=float(logit(s["pred"]["wayoff"]).mean())
        bc=fit_offset(s["pred"]["success"],s["pred"]["mr"],s["pred"]["wayoff"],s["y"],mm,mw)
        mt=float(logit(t["pred"]["mr"]).mean()); wt=float(logit(t["pred"]["wayoff"]).mean())
        adj=1/(1+np.exp(-(logit(t["pred"]["success"])+bc[0]*(logit(t["pred"]["mr"])-mt)+bc[1]*(logit(t["pred"]["wayoff"])-wt))))
        trans.append({"fit_season":src,"apply_season":tgt,"b":float(bc[0]),"c":float(bc[1]),"baseline_brier":brier(t["y"],t["pred"]["success"]),"offset_brier":brier(t["y"],adj),"delta_brier":brier(t["y"],adj)-brier(t["y"],t["pred"]["success"]),"rows":int(len(t["y"]))})
    report={"experiment_id":"REF-REBUILD-001-TRANSITION-001","environment":{"python":platform.python_version(),"catboost":cb.__version__},"feature_count":len(FEATURE_COLUMNS),"official_train_only":True,"test_used":False,"external_data_used":False,"transitions":trans,"status":"PASS" if all(x["delta_brier"]<0 for x in trans) else "FAIL_FORWARD_SIGN","submission_status":"HOLD"}
    OUT.mkdir(parents=True,exist_ok=True); (OUT/"transition_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=="__main__": main()
