#!/usr/bin/env bash
# Bootstrap a fresh box (vast.ai or local) for training:
#   1. Python venv + deps (torch CUDA, ESPnet, audio libs)
#   2. TITML-IDN raw download + AES extract
#   3. TITML-2spk-v2 synthetic dataset generation
#   4. TITML-3spk-v2 synthetic dataset generation
#
# No R2, no training, no checkpoint downloads. Pure dataset prep.
#
# Usage:
#   bash scripts/setup_datasets.sh
#   SKIP_2SPK=1 bash scripts/setup_datasets.sh   # only 3spk
#   SKIP_3SPK=1 bash scripts/setup_datasets.sh   # only 2spk
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

RAW_DIR="dataset/raw/TTML-IDN"
SYNTH_2SPK="dataset/synthetic/TITML-2spk-v2"
SYNTH_3SPK="dataset/synthetic/TITML-3spk-v2"
ZIPS_DIR="dataset/zips"

RAW_FILE_ID="1ETEzZhm5s5XAp1tKUtSj9zq4Ic-7tben"
RAW_PASSWORD="Hwd9m2x_d3ig"

log()  { printf '\n\033[1;36m[setup]\033[0m %s\n' "$*"; }
fail() { printf '\n\033[1;31m[setup] FATAL:\033[0m %s\n' "$*" >&2; exit 1; }

# -----------------------------------------------------------------------------
# 1. Python env + deps
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
    matplotlib tqdm gdown pyzipper

# -----------------------------------------------------------------------------
# 2. Raw TITML-IDN
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
# 3. Synthetic 2spk
# -----------------------------------------------------------------------------
if [ "${SKIP_2SPK:-0}" != "1" ]; then
    if [ ! -f "$SYNTH_2SPK/dataset_info.json" ]; then
        log "Generating synthetic 2-speaker dataset → $SYNTH_2SPK"
        python dataset/generator/titml_mix_generator_2spk.py \
            --titml-dir "$PROJECT_ROOT/$RAW_DIR" \
            --output-dir "$PROJECT_ROOT/$SYNTH_2SPK" \
            || fail "2spk dataset generation failed."
    else
        log "2spk synthetic already present, skipping"
    fi
else
    log "SKIP_2SPK=1 — skipping 2spk gen"
fi

# -----------------------------------------------------------------------------
# 4. Synthetic 3spk
# -----------------------------------------------------------------------------
if [ "${SKIP_3SPK:-0}" != "1" ]; then
    if [ ! -f "$SYNTH_3SPK/dataset_info.json" ]; then
        log "Generating synthetic 3-speaker dataset → $SYNTH_3SPK"
        python dataset/generator/titml_mix_generator_3spk.py \
            --titml-dir "$PROJECT_ROOT/$RAW_DIR" \
            --output-dir "$PROJECT_ROOT/$SYNTH_3SPK" \
            || fail "3spk dataset generation failed."
    else
        log "3spk synthetic already present, skipping"
    fi
else
    log "SKIP_3SPK=1 — skipping 3spk gen"
fi

log "DONE."
log "  venv:      .venv/  (source .venv/bin/activate)"
log "  raw:       $RAW_DIR/"
log "  2spk:      $SYNTH_2SPK/"
log "  3spk:      $SYNTH_3SPK/"
