#!/usr/bin/env bash
# Upload all paper-faith-strict checkpoints to R2 under k16-s8 naming.
# Usage: bash scripts/r2_upload_all.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# shellcheck disable=SC1091
source "$SCRIPT_DIR/r2.env"

CKPT_ROOT="$PROJECT_ROOT/checkpoints/paper-faith-strict"

if [[ ! -d "$CKPT_ROOT" ]]; then
    echo "ERROR: $CKPT_ROOT not found"
    exit 1
fi

UPLOADED=0
SKIPPED=0

for dir in "$CKPT_ROOT"/*/; do
    name="$(basename "$dir")"
    if [[ ! -f "$dir/best_model.pth" ]]; then
        echo "SKIP $name (no best_model.pth)"
        SKIPPED=$((SKIPPED + 1))
        continue
    fi
    dest="${name}-k16-s8"
    echo "=== Uploading $name -> $dest ==="
    python "$SCRIPT_DIR/r2_util.py" upload "$dir" "$dest" \
        --include "best_model.pth" \
        --include "training_history.json" \
        --include "config.json" \
        --include "*.png"
    UPLOADED=$((UPLOADED + 1))
done

echo ""
echo "Done. Uploaded $UPLOADED model(s), skipped $SKIPPED."
