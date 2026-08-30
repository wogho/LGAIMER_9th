#!/usr/bin/env python3
"""Fail-closed verifier for the mandatory >=5% internal metric gate."""
from pathlib import Path
import json
ROOT=Path(__file__).resolve().parents[1]
R=ROOT/'model/COMBO-RESID3-OOF-007/residual3_results.json'
def main():
 d=json.loads(R.read_text()); rows=[x for x in d['season_results'] if x['season'] in (2022,2023,2024)]
 checks=[]
 for x in rows:
  b=float(x['base_metric']); c=float(x['resid3']['metric']); rel=(c/b-1.0) if b else float('-inf')
  checks.append({'season':x['season'],'baseline_metric':b,'candidate_metric':c,'relative_improvement':rel,'pass':rel>=0.05})
 out={'experiment_id':'EXTREME-METRIC-GATE-008','required_relative_improvement':0.05,'checks':checks,'checked_count':len(checks),'pass_count':sum(int(x['pass']) for x in checks),'status':'EXTREME_GATE_PASS' if checks and all(x['pass'] for x in checks) else 'EXTREME_GATE_FAIL','zip_approval':False}
 (ROOT/'model/COMBO-RESID3-OOF-007/extreme_gate_008.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)); print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
