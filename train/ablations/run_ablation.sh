#!/usr/bin/env bash
# Run K-size ablation sweep for baseline SkiM 2-speaker.
#
# Trains baseline SkiM at multiple segment_size (K) values, all else fixed.
# Each run gets its own checkpoint dir under checkpoints/ablations/k-size/.
#
# Usage:
#   bash train/ablations/run_ablation.sh           # default sweep, 30 epochs each
#   EPOCHS=50 bash train/ablations/run_ablation.sh # longer per run
#   K_VALUES="20 100" bash train/ablations/run_ablation.sh
#   OVERLAP=1 bash train/ablations/run_ablation.sh # enable seg_overlap

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY="${SCRIPT_DIR}/k-size/train_k_ablation.py"

EPOCHS="${EPOCHS:-30}"
K_VALUES="${K_VALUES:-20 50 100 150}"
OVERLAP="${OVERLAP:-0}"

OVERLAP_FLAG=""
if [ "${OVERLAP}" = "1" ]; then
  OVERLAP_FLAG="--seg-overlap"
fi

cd "${PROJECT_ROOT}"

echo "=========================================="
echo "K-size ablation sweep"
echo "  K values : ${K_VALUES}"
echo "  Epochs   : ${EPOCHS}"
echo "  Overlap  : ${OVERLAP}"
echo "=========================================="

for K in ${K_VALUES}; do
  echo ""
  echo ">>> Running K=${K} ${OVERLAP_FLAG}"
  python "${PY}" --k "${K}" --num-epochs "${EPOCHS}" ${OVERLAP_FLAG} \
    || { echo "FAILED at K=${K}"; exit 1; }
done

echo ""
echo "=========================================="
echo "Sweep complete. Results in:"
echo "  checkpoints/ablations/k-size/"
echo "Compare best_val_loss across runs:"
echo "  for d in checkpoints/ablations/k-size/*/; do"
echo "    echo \"\$d: \$(jq .best_si_snr \$d/config.json)\""
echo "  done"
echo "=========================================="
