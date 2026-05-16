#!/usr/bin/env bash
# Single-command vast.ai runner for K-size ablation (baseline SkiM 2spk).
#
# Steps:
#   1. env + deps (mirrors scripts/vast_run.sh)
#   2. R2 creds load + verify
#   3. raw dataset (TITML-IDN) download/extract  ← shared with vast_run.sh
#   4. synthetic 2spk dataset generation         ← differs (2spk, not 3spk)
#   5. K-size ablation sweep over baseline SkiM 2spk
#   6. upload each ablation dir to R2
#
# Usage:
#   bash scripts/vast_run_ablation.sh
#   EPOCHS=50 bash scripts/vast_run_ablation.sh
#   K_VALUES="20 100" bash scripts/vast_run_ablation.sh
#   OVERLAP=1 bash scripts/vast_run_ablation.sh
#   DRYRUN_DATASET_ONLY=1 bash scripts/vast_run_ablation.sh  # stop after step 4
#
# Required before launching on vast.ai:
#   - scripts/r2.env present (or matching env vars exported)
#   - GPU available
set -euo pipefail

# -----------------------------------------------------------------------------
# 1. Env
# -----------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# shellcheck disable=SC1091
if [ -f "$SCRIPT_DIR/r2.env" ]; then
    source "$SCRIPT_DIR/r2.env"
fi

R2_ENDPOINT="${R2_ENDPOINT:?missing R2_ENDPOINT (populate scripts/r2.env or export it)}"
R2_BUCKET="${R2_BUCKET:?missing R2_BUCKET}"
R2_PREFIX="${R2_PREFIX:-vast-checkpoint}"
: "${AWS_ACCESS_KEY_ID:?missing AWS_ACCESS_KEY_ID}"
: "${AWS_SECRET_ACCESS_KEY:?missing AWS_SECRET_ACCESS_KEY}"
export AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-auto}"

SYNTH_DIR="dataset/synthetic/TITML-2spk-v2"
RAW_DIR="dataset/raw/TTML-IDN"
ZIPS_DIR="dataset/zips"

RAW_FILE_ID="1ETEzZhm5s5XAp1tKUtSj9zq4Ic-7tben"
RAW_PASSWORD="Hwd9m2x_d3ig"

# Ablation params (override via env)
EPOCHS="${EPOCHS:-30}"
K_VALUES="${K_VALUES:-20 50 100 150}"
OVERLAP="${OVERLAP:-0}"

OVERLAP_FLAG=""
OVERLAP_TAG="F"
if [ "${OVERLAP}" = "1" ]; then
    OVERLAP_FLAG="--seg-overlap"
    OVERLAP_TAG="T"
fi

log()  { printf '\n\033[1;36m[vast_ablation]\033[0m %s\n' "$*"; }
fail() { printf '\n\033[1;31m[vast_ablation] FATAL:\033[0m %s\n' "$*" >&2; exit 1; }

# -----------------------------------------------------------------------------
# 2. Python env + deps
# -----------------------------------------------------------------------------
log "Preparing Python env"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip wheel >/dev/null

if ! python -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    log "Installing torch (CUDA build)"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
fi

log "Installing project deps"
pip install \
    espnet==202304 espnet_model_zoo \
    soundfile librosa==0.9.2 \
    mir_eval pesq pystoi \
    matplotlib tqdm gdown pyzipper \
    boto3

# -----------------------------------------------------------------------------
# 3. R2 verify
# -----------------------------------------------------------------------------
log "Verifying R2 access (bucket=${R2_BUCKET})"
python "$SCRIPT_DIR/r2_util.py" probe >/dev/null \
    || fail "R2 probe failed — check creds/endpoint/bucket."

# -----------------------------------------------------------------------------
# 4. Raw dataset (TITML-IDN)
# -----------------------------------------------------------------------------
mkdir -p "$ZIPS_DIR" "$RAW_DIR"
if [ ! -d "$RAW_DIR/Speech" ]; then
    log "Downloading TITML-IDN raw zip"
    gdown "https://drive.google.com/uc?id=$RAW_FILE_ID" -O "$ZIPS_DIR/TITML-IDN.zip" \
        || fail "Raw dataset download failed."
    log "Extracting raw zip (AES)"
    python - <<PYEOF
import pyzipper, os
os.makedirs("$RAW_DIR", exist_ok=True)
with pyzipper.AESZipFile("$ZIPS_DIR/TITML-IDN.zip") as zf:
    zf.pwd = b"$RAW_PASSWORD"
    zf.extractall("$RAW_DIR")
print("extracted")
PYEOF
    if [ ! -d "$RAW_DIR/Speech" ]; then
        NEST="$(find "$RAW_DIR" -mindepth 2 -maxdepth 2 -type d -name Speech | head -n 1)"
        if [ -n "$NEST" ]; then
            mv "$(dirname "$NEST")"/* "$RAW_DIR"/
        fi
    fi
    [ -d "$RAW_DIR/Speech" ] || fail "Raw extract did not yield $RAW_DIR/Speech."
else
    log "Raw dataset already present, skipping"
fi

# -----------------------------------------------------------------------------
# 5. Synthetic 2spk dataset
# -----------------------------------------------------------------------------
if [ ! -f "$SYNTH_DIR/dataset_info.json" ]; then
    log "Generating synthetic 2-speaker dataset → $SYNTH_DIR"
    python dataset/generator/titml_mix_generator_2spk.py \
        --titml-dir "$PROJECT_ROOT/$RAW_DIR" \
        --output-dir "$PROJECT_ROOT/$SYNTH_DIR" \
        || fail "Synthetic 2spk dataset generation failed."
else
    log "Synthetic 2spk dataset already present, skipping"
fi

if [ "${DRYRUN_DATASET_ONLY:-0}" = "1" ]; then
    log "DRYRUN_DATASET_ONLY=1 — stopping after dataset prep."
    exit 0
fi

# -----------------------------------------------------------------------------
# 6. K-size ablation sweep
# -----------------------------------------------------------------------------
upload_results() {
    local src_dir="$1"
    local run_name="$2"
    local dest_prefix="${R2_PREFIX}/ablations/${run_name}"
    log "Uploading $src_dir → s3://${R2_BUCKET}/${dest_prefix}/"
    python "$SCRIPT_DIR/r2_util.py" upload "$src_dir" "$dest_prefix" \
        --include "best_model.pth" \
        --include "training_history.json" \
        --include "training_curves.png" \
        --include "config.json" \
        --include "run.log" \
        --include "checkpoint_epoch_*.pth" \
        || fail "Upload failed for $run_name."
}

log "Starting K-size ablation: K=[${K_VALUES}], epochs=${EPOCHS}, overlap=${OVERLAP}"

export PYTHONUNBUFFERED=1   # ensure training prints/tqdm stream live through `tee`

for K in ${K_VALUES}; do
    TAG="k${K}_overlap${OVERLAP_TAG}"
    CKPT_DIR="checkpoints/ablations/k-size/${TAG}"
    mkdir -p "$CKPT_DIR"

    log "Run K=${K} ${OVERLAP_FLAG} → ${CKPT_DIR}"

    python -u train/ablations/k-size/train_k_ablation.py \
        --k "${K}" \
        --num-epochs "${EPOCHS}" \
        --dataset-dir "$PROJECT_ROOT/$SYNTH_DIR" \
        --raw-dir "$PROJECT_ROOT/$RAW_DIR" \
        --checkpoint-dir "$PROJECT_ROOT/$CKPT_DIR" \
        ${OVERLAP_FLAG} \
        2>&1 | tee "$CKPT_DIR/run.log" \
        || fail "Ablation run failed at K=${K}."

    log "K=${K} done → upload to R2"
    upload_results "$CKPT_DIR" "k-size/${TAG}"
done

log "DONE. All K-ablation runs uploaded under s3://${R2_BUCKET}/${R2_PREFIX}/ablations/k-size/"

# Uncomment to auto-shutdown the vast.ai box on completion:
# sudo shutdown -h now
