#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);args=ap.parse_args()
 lb45=json.loads((ROOT/'model/REF4-F-GATED-PSYCH-LATENT-ZIP-045A/leaderboard_result.json').read_text());lb50=json.loads((ROOT/'model/REF4-DISJOINT-RSPLIT-FPSYCH-ZIP-050A/leaderboard_result.json').read_text());r43=json.loads((ROOT/'model/REF4-TRAINONLY-F-GATED-PSYCH-LATENT-043A/result.json').read_text());r47=json.loads((ROOT/'model/REF4-TRAINONLY-R-GATED-SPLIT-RESIDUAL-047A/result.json').read_text());r48=json.loads((ROOT/'model/REF4-TRAINONLY-DISJOINT-RSPLIT-FPSYCH-048A/result.json').read_text())
 def local_gain(r):return r['pooled']['candidate']['local_score']-r['pooled']['baseline']['local_score']
 lb030=lb45['previous_official_score'];d45=lb45['official_score']-lb030;d50=lb50['official_score']-lb45['official_score'];dall=lb50['official_score']-lb030;g43=local_gain(r43);g47=local_gain(r47);g48=local_gain(r48)
 out={'status':'AGGREGATE_DIAGNOSIS_ONLY','leaderboard':{'030':lb030,'045':lb45['official_score'],'050':lb50['official_score'],'045_vs_030':d45,'050_vs_045':d50,'050_vs_030':dall,'gap_to_1100':1100-lb50['official_score']},'oof_local_score_gains':{'F_psych_043':g43,'R_split_047':g47,'disjoint_048':g48},'aggregate_transfer_ratios':{'F_psych_LB_delta_over_OOF_local_gain':d45/g43,'R_split_LB_delta_over_OOF_local_gain':d50/g47,'combined_LB_delta_over_OOF_local_gain':dall/g48},'bindings':{'zip_045_sha256':lb45['zip_sha256'],'zip_050_sha256':lb50['zip_sha256'],'result_043_sha256':sha(ROOT/'model/REF4-TRAINONLY-F-GATED-PSYCH-LATENT-043A/result.json'),'result_047_sha256':sha(ROOT/'model/REF4-TRAINONLY-R-GATED-SPLIT-RESIDUAL-047A/result.json'),'result_048_sha256':sha(ROOT/'model/REF4-TRAINONLY-DISJOINT-RSPLIT-FPSYCH-048A/result.json')},'leaderboard_derived_tuning_authorized':False}
 p=ROOT/args.out;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
