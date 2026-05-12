#!/usr/bin/env bash
# Local-side pre-flight check for the vast.ai run (R2 backend, boto3).
# Verifies that:
#   - boto3 can head the configured R2 bucket
#   - boto3 can put/get/delete a probe object under the upload prefix
#   - gdown can resolve the three Drive file IDs
#
# Run this on your laptop BEFORE renting a vast.ai box.
#
# Usage:
#   bash scripts/local_r2_test.sh
#   DOWNLOAD_CKPTS=1 bash scripts/local_r2_test.sh   # actually pull both ckpt archives
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# shellcheck disable=SC1091
if [ -f "$SCRIPT_DIR/r2.env" ]; then
    source "$SCRIPT_DIR/r2.env"
fi

: "${R2_ENDPOINT:?missing R2_ENDPOINT (populate scripts/r2.env)}"
: "${R2_BUCKET:?missing R2_BUCKET}"
: "${AWS_ACCESS_KEY_ID:?missing AWS_ACCESS_KEY_ID}"
: "${AWS_SECRET_ACCESS_KEY:?missing AWS_SECRET_ACCESS_KEY}"

REG_CKPT_FILE_ID="1KEEY8q8ZPtGVeVUdmJl-aGtNXtUHy28"
VANILLA_CKPT_FILE_ID="1JaR9IOuHnlgDcofajzNT_Q62CLbva1wy"
RAW_FILE_ID="1ETEzZhm5s5XAp1tKUtSj9zq4Ic-7tben"

pass() { printf '\033[1;32m  ✓\033[0m %s\n' "$*"; }
fail() { printf '\033[1;31m  ✗\033[0m %s\n' "$*" >&2; exit 1; }

echo "[1/3] python + boto3 presence"
command -v python3 >/dev/null || fail "python3 not installed"
python3 -c "import boto3" 2>/dev/null || fail "boto3 not installed. pip install boto3"
pass "boto3 importable ($(python3 -c 'import boto3; print(boto3.__version__)'))"

echo "[2/3] R2 head_bucket + put/get/delete probe"
python3 "$SCRIPT_DIR/r2_util.py" probe || fail "R2 probe failed — check creds, bucket, endpoint."
pass "R2 round-trip OK"

echo "[3/3] gdown can reach ckpt file IDs"
command -v gdown >/dev/null || fail "gdown not installed. pip install gdown"
for fid in "$REG_CKPT_FILE_ID" "$VANILLA_CKPT_FILE_ID" "$RAW_FILE_ID"; do
    if gdown "https://drive.google.com/uc?id=$fid" -O /dev/null --no-cookies 2>&1 | head -n 5 | grep -qi "permission\|cannot retrieve\|not found"; then
        fail "Drive file $fid not accessible"
    fi
    pass "Drive file $fid reachable"
done

if [ "${DOWNLOAD_CKPTS:-0}" = "1" ]; then
    echo "[extra] full ckpt download (DOWNLOAD_CKPTS=1)"
    mkdir -p /tmp/vast_run_probe
    gdown "https://drive.google.com/uc?id=$REG_CKPT_FILE_ID"     -O /tmp/vast_run_probe/reg.tar.gz
    gdown "https://drive.google.com/uc?id=$VANILLA_CKPT_FILE_ID" -O /tmp/vast_run_probe/vanilla.zip
    ls -lh /tmp/vast_run_probe/
    pass "both ckpt archives downloaded into /tmp/vast_run_probe/"
fi

echo
echo "All checks passed. Safe to launch scripts/vast_run.sh on vast.ai."
