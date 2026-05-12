#!/usr/bin/env bash
# Single-command vast.ai run:
#   1. env + deps
#   2. R2 creds load + verify
#   3. raw dataset download/extract
#   4. synthetic 3spk dataset generation
#   5. ckpt downloads (reg-transfer .tar.gz + vanilla v3 .zip)
#   6. resume SkiM-Attention v3 γ-reg transfer training, upload to R2
#   7. resume SkiM-Attention v3 vanilla training, upload to R2
#
# Usage:
#   bash scripts/vast_run.sh                      # full run
#   DRYRUN_CKPT_ONLY=1 bash scripts/vast_run.sh   # stop after step 5 (smoke-test)
#
# Required before launching on vast.ai:
#   - scripts/r2.env present (or matching env vars exported in shell)
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

REG_DIR="checkpoints/3speaker/skim-attention-v3-reg-transfer"
VANILLA_DIR="checkpoints/3speaker/skim-attention-v3"
SYNTH_DIR="dataset/synthetic/TITML-3spk-v2"
RAW_DIR="dataset/raw/TTML-IDN"
ZIPS_DIR="dataset/zips"

RAW_FILE_ID="1ETEzZhm5s5XAp1tKUtSj9zq4Ic-7tben"
RAW_PASSWORD="Hwd9m2x_d3ig"
REG_CKPT_FILE_ID="1KEEY8q8ZPtGVeVUdmJl-aGtNXtUHy28"
VANILLA_CKPT_FILE_ID="1JaR9IOuHnlgDcofajzNT_Q62CLbva1wy"

log()  { printf '\n\033[1;36m[vast_run]\033[0m %s\n' "$*"; }
fail() { printf '\n\033[1;31m[vast_run] FATAL:\033[0m %s\n' "$*" >&2; exit 1; }

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

# torch: skip reinstall if CUDA build already present
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
# 4. Raw dataset
# -----------------------------------------------------------------------------
mkdir -p "$ZIPS_DIR" "$RAW_DIR"
if [ ! -d "$RAW_DIR/Speech" ]; then
    log "Downloading TITML-IDN raw zip"
    gdown "https://drive.google.com/uc?id=$RAW_FILE_ID" -O "$ZIPS_DIR/TITML-IDN.zip" || fail "Raw dataset download failed."
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
# 5. Synthetic 3spk dataset
# -----------------------------------------------------------------------------
if [ ! -f "$SYNTH_DIR/dataset_info.json" ]; then
    log "Generating synthetic 3-speaker dataset to $SYNTH_DIR"
    python dataset/generator/titml_mix_generator_3spk.py \
        --titml-dir "$PROJECT_ROOT/$RAW_DIR" \
        --output-dir "$PROJECT_ROOT/$SYNTH_DIR" \
        || fail "Synthetic dataset generation failed."
else
    log "Synthetic dataset already present, skipping"
fi

# -----------------------------------------------------------------------------
# 6. Checkpoint downloads
# -----------------------------------------------------------------------------
mkdir -p "$REG_DIR" "$VANILLA_DIR"

download_and_extract() {
    local file_id="$1"
    local archive_path="$2"
    local dest_dir="$3"
    local archive_type="$4"   # tar.gz | zip

    if [ -f "$dest_dir/best_model.pth" ]; then
        log "ckpt already present at $dest_dir/best_model.pth — skipping download"
        return 0
    fi

    log "Downloading ckpt → $archive_path"
    gdown "https://drive.google.com/uc?id=$file_id" -O "$archive_path" || fail "ckpt download failed for $file_id."

    log "Extracting $archive_path → $dest_dir"
    case "$archive_type" in
        tar.gz) tar -xzf "$archive_path" -C "$dest_dir" || fail "tar extract failed." ;;
        zip)    unzip -o -q "$archive_path" -d "$dest_dir" || fail "unzip failed." ;;
        *)      fail "Unknown archive type $archive_type" ;;
    esac

    if [ ! -f "$dest_dir/best_model.pth" ]; then
        local nested
        nested="$(find "$dest_dir" -mindepth 2 -name best_model.pth | head -n 1)"
        if [ -n "$nested" ]; then
            log "Flattening nested archive layout"
            mv "$(dirname "$nested")"/* "$dest_dir"/ 2>/dev/null || true
        fi
    fi

    [ -f "$dest_dir/best_model.pth" ] || fail "best_model.pth missing in $dest_dir after extract."
}

download_and_extract "$REG_CKPT_FILE_ID"     "$ZIPS_DIR/skim-attention-v3-reg-transfer.tar.gz" "$REG_DIR"     "tar.gz"
download_and_extract "$VANILLA_CKPT_FILE_ID" "$ZIPS_DIR/skim-attention-v3-3spk-30epochs.zip"   "$VANILLA_DIR" "zip"

if [ "${DRYRUN_CKPT_ONLY:-0}" = "1" ]; then
    log "DRYRUN_CKPT_ONLY=1 — stopping after ckpt extraction."
    exit 0
fi

# -----------------------------------------------------------------------------
# 7. Run 1 — SkiM-Attention v3 γ-reg transfer (resume)
# -----------------------------------------------------------------------------
upload_results() {
    local src_dir="$1"
    local run_name="$2"
    local dest_prefix="${R2_PREFIX}/${run_name}"
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

log "Run 1/2 — SkiM-Attention v3 γ-reg transfer (resume)"
python train/3speaker/skim-attention-v3-reg/train_skim_attention_v3_reg_3spk_transfer.py \
    --resume-from "$REG_DIR/best_model.pth" \
    2>&1 | tee "$REG_DIR/run.log"
upload_results "$REG_DIR" "skim-attention-v3-reg-transfer"

# -----------------------------------------------------------------------------
# 8. Run 2 — SkiM-Attention v3 vanilla (resume)
# -----------------------------------------------------------------------------
log "Run 2/2 — SkiM-Attention v3 vanilla (resume)"
python train/3speaker/skim-attention-v3/train_skim_attention_v3_3spk.py \
    --resume-from "$VANILLA_DIR/best_model.pth" \
    2>&1 | tee "$VANILLA_DIR/run.log"
upload_results "$VANILLA_DIR" "skim-attention-v3"

log "DONE. Both runs uploaded under s3://${R2_BUCKET}/${R2_PREFIX}/"

# Uncomment to auto-shutdown the vast.ai box on completion:
# sudo shutdown -h now
