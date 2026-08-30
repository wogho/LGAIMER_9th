#!/usr/bin/env python3
from pathlib import Path
import hashlib, json, zipfile
ROOT = Path(__file__).resolve().parents[1]; ZIP = ROOT / "output/submit_combo_full_002.zip"; B = ROOT / "model/COMBO-FULL-002"
EXPECTED = {"script.py":ROOT/"scripts/infer_combo_full_isolated_002.py", "requirements.txt":ROOT/"requirements_submit.txt", "solution/LG_Aimers_솔루션_PPT_Phase2.pptx":ROOT/"output/LG_Aimers_솔루션_PPT_Phase2.pptx", "model/COMBO-FULL-002/model.cbm":B/"model.cbm", "model/COMBO-FULL-002/feature_columns.json":B/"feature_columns.json", "model/COMBO-FULL-002/pitcher_count_lookup.csv":B/"pitcher_count_lookup.csv", "model/COMBO-FULL-002/asof_pitcher_id_prior.csv":B/"asof_pitcher_id_prior.csv", "model/COMBO-FULL-002/asof_batter_id_prior.csv":B/"asof_batter_id_prior.csv", "model/COMBO-FULL-002/asof_pitchmix_prior.csv":B/"asof_pitchmix_prior.csv"}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert ZIP.is_file()
    with zipfile.ZipFile(ZIP) as z:
        members = set(z.namelist()); assert members == set(EXPECTED)
        source = z.read("script.py").decode("utf-8")
        forbidden = [x for x in ["requests", "urllib", "http://", "https://", "openai", "google.generativeai", "train.csv", "groupby(", "test.groupby"] if x in source]
        assert not forbidden, forbidden
        hashes = {n: hashlib.sha256(z.read(n)).hexdigest() for n in sorted(members)}; source_hashes = {n: sha(p) for n,p in EXPECTED.items()}; assert hashes == source_hashes
        feature_count = len(json.loads(z.read("model/COMBO-FULL-002/feature_columns.json"))); assert feature_count == 73
    report = {"experiment_id":"COMBO-FULL-ZIP-AUDIT-002", "zip_sha256":sha(ZIP), "member_count":len(members), "member_hash_match":True, "ppt_included":True, "feature_count":feature_count, "forbidden_tokens":forbidden, "external_api_used":False, "external_data_used":False, "test_row_aggregation_used":False, "status":"AUDIT_VERIFIED", "submission_status":"HOLD"}
    (ROOT/"output/submit_combo_full_002_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)); print(json.dumps(report, ensure_ascii=False, indent=2))
if __name__ == "__main__": main()
