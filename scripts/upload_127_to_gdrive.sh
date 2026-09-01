#!/usr/bin/env bash
set -euo pipefail

RCLONE_CONF="/home/ubuntu/.config/rclone/rclone.conf"
BASE="lgaimer126_gdrive:L4_EXPERIMENTS/REF4-ANCHOR-INVARIANT-R-RESIDUAL-L4-127"

echo "=== Uploading REF4-127 files to Google Drive ==="

echo "[1/6] Uploading data/train.csv (368 MB)..."
rclone copyto data/train.csv "$BASE/input/data/train.csv" \
    --config "$RCLONE_CONF" \
    --checksum --check-first --transfers 1 -v

echo "[2/6] Uploading model/REF4-113A-V66-NESTED-117A/oof_predictions.csv (96 MB)..."
rclone copyto model/REF4-113A-V66-NESTED-117A/oof_predictions.csv "$BASE/input/anchor/oof_predictions.csv" \
    --config "$RCLONE_CONF" \
    --checksum --check-first --transfers 1 -v

echo "[3/6] Uploading colab/REF4_127_L4_CODE.zip..."
rclone copyto colab/REF4_127_L4_CODE.zip "$BASE/input/code/REF4_127_L4_CODE.zip" \
    --config "$RCLONE_CONF" \
    --checksum --check-first --transfers 1 -v

echo "[4/6] Uploading colab/REF4_127_L4.ipynb..."
rclone copyto colab/REF4_127_L4.ipynb "$BASE/input/code/REF4_127_L4.ipynb" \
    --config "$RCLONE_CONF" \
    --checksum --check-first --transfers 1 -v

echo "[5/6] Uploading model/REF4-ANCHOR-INVARIANT-R-RESIDUAL-L4-127/audit_contract.json..."
rclone copyto model/REF4-ANCHOR-INVARIANT-R-RESIDUAL-L4-127/audit_contract.json "$BASE/input/code/audit_contract.json" \
    --config "$RCLONE_CONF" \
    --checksum --check-first --transfers 1 -v

echo "[6/6] Uploading model/REF4-ANCHOR-INVARIANT-R-RESIDUAL-L4-127/SHA256SUMS.input..."
rclone copyto model/REF4-ANCHOR-INVARIANT-R-RESIDUAL-L4-127/SHA256SUMS.input "$BASE/manifest/SHA256SUMS.input" \
    --config "$RCLONE_CONF" \
    --checksum --check-first --transfers 1 -v

echo "=== All 6 files uploaded successfully! ==="
