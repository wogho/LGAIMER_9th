import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF4_SCRIPT = ROOT / 'github_reference/4번 레포/final/inference/script.py'
PROD_071_SCRIPT = ROOT / 'model/REF4-ADAPTIVE-CHANNEL-OPT-071A/production_package/script.py'

print("=== REF4 Script Line Count ===", len(REF4_SCRIPT.read_text().splitlines()))
print("=== 071A Script Line Count ===", len(PROD_071_SCRIPT.read_text().splitlines()))

# Let's inspect manifest files
manifest_071 = json.loads((ROOT / 'model/REF4-ADAPTIVE-CHANNEL-OPT-071A/production_package/model/manifest.json').read_text())
print("\n071A Manifest Keys & Values:")
for k, v in manifest_071.items():
    print(f"  {k}: {v}")

