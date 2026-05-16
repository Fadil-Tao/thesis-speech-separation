#!/usr/bin/env bash
# Pull vast.ai run artifacts from R2 → local checkpoints/.
# Self-contained: scripts/r2_download.py loads scripts/r2.env + uses uv inline deps.
#
# Usage:
#   bash scripts/r2_download.sh           # both runs
#   bash scripts/r2_download.sh reg       # reg-transfer only
#   bash scripts/r2_download.sh vanilla   # vanilla only
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec uv run --with boto3 "$SCRIPT_DIR/r2_download.py" --which "${1:-both}"
