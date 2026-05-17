#!/usr/bin/env bash
# Zip each eval result dir per model, then upload to R2.
#
# Layout:
#   inference/eval/results/<model>/        → inference/eval/zips/<model>.zip
#   R2 dest: $R2_PREFIX/eval/<model>.zip
#
# Usage:
#   bash inference/eval/zip_and_upload.sh                   # all
#   bash inference/eval/zip_and_upload.sh 2speaker-skim     # specific
#   NO_UPLOAD=1 bash inference/eval/zip_and_upload.sh       # zip only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

RESULTS_DIR="inference/eval/results"
ZIPS_DIR="inference/eval/zips"
mkdir -p "$ZIPS_DIR"

if [ $# -gt 0 ]; then
    MODELS=("$@")
else
    MODELS=()
    for d in "$RESULTS_DIR"/*/; do
        [ -d "$d" ] || continue
        MODELS+=("$(basename "$d")")
    done
fi

if [ ${#MODELS[@]} -eq 0 ]; then
    echo "[err] no model result dirs in $RESULTS_DIR" >&2
    exit 1
fi

echo "[zip] models: ${MODELS[*]}"

for m in "${MODELS[@]}"; do
    src="$RESULTS_DIR/$m"
    zip_path="$ZIPS_DIR/$m.zip"
    if [ ! -d "$src" ]; then
        echo "[skip] $m: result dir missing"
        continue
    fi
    echo "[zip] $m → $zip_path"
    rm -f "$zip_path"
    (cd "$RESULTS_DIR" && zip -rq "$PROJECT_ROOT/$zip_path" "$m")
    size_mb=$(du -m "$zip_path" | cut -f1)
    echo "      size: ${size_mb} MB"
done

if [ "${NO_UPLOAD:-0}" = "1" ]; then
    echo "[done] NO_UPLOAD=1, skipping R2 upload"
    exit 0
fi

if [ -f scripts/r2.env ]; then
    # shellcheck disable=SC1091
    . scripts/r2.env
fi

: "${R2_PREFIX:?R2_PREFIX not set (source scripts/r2.env)}"

for m in "${MODELS[@]}"; do
    zip_path="$ZIPS_DIR/$m.zip"
    [ -f "$zip_path" ] || continue
    echo "[upload] $m → s3://.../$R2_PREFIX/eval/$m.zip"
    python scripts/r2_util.py upload \
        "$ZIPS_DIR" \
        "$R2_PREFIX/eval" \
        --include "$m.zip"
done

echo "[done]"
