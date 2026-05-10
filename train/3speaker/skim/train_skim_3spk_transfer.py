# -*- coding: utf-8 -*-
"""Training script for SkiM 3-Speaker model with Transfer Learning.

This script trains a SkiM (Skipping Memory LSTM) model for 3-speaker speech separation
using transfer learning from a pretrained 2-speaker model.

Usage:
    python train_skim_3spk_transfer.py [--resume-from CHECKPOINT] [--num-epochs N]

The script will:
1. Load the TITML-3spk-v2 dataset (DynamicMixDataset for train)
2. Load pretrained weights from 2-speaker SkiM model
3. Reinitialize only the output layer (2->3 speakers dimension change)
4. Train with lower learning rate (1e-4) to protect pretrained weights
5. Save checkpoints and training curves
6. Evaluate on test set
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "train"))

from utils.paths import get_raw_dir, get_synthetic_dir, get_checkpoint_dir

# Import shared dataset utilities
from datasets_utils import (
    build_utterance_split, DynamicMixDataset, IndonesianMixDataset
)

# ESPnet imports
from espnet2.enh.encoder.conv_encoder import ConvEncoder
from espnet2.enh.decoder.conv_decoder import ConvDecoder
from espnet2.enh.espnet_model import ESPnetEnhancementModel
from espnet2.enh.loss.criterions.time_domain import SISNRLoss
from espnet2.enh.loss.wrappers.pit_solver import PITSolver

# Import SkiM separator
from implementation.skim.skim_separator import SkiMSeparator


# =============================================================================
# Configuration
# =============================================================================

MODEL_CONFIG = {
    "encoder": {
        "channel": 256,
        "kernel_size": 32,
        "stride": 16,
    },
    "decoder": {
        "channel": 256,
        "kernel_size": 32,
        "stride": 16,
    },
    "separator": {
        "input_dim": 256,
        "causal": False,
        "num_spk": 3,
        "predict_noise": False,
        "nonlinear": "relu",
        "layer": 4,
        "unit": 256,
        "segment_size": 20,
        "dropout": 0.1,
        "mem_type": "hc",
        "seg_overlap": False,
    },
}

TRAIN_CONFIG = {
    "batch_size": 8,
    "num_epochs": 100,
    "learning_rate": 1e-4,  # 10x smaller for transfer learning
    "weight_decay": 1e-5,
    "gradient_clip": 5.0,
    "seed": 42,
}

TRANSFER_CONFIG = {
    "pretrained_path": "checkpoints/2speaker/skim/best_model.pth",
    "description": "Transfer from SkiM 2-speaker to 3-speaker",
}

# Paths
# Paths — env-aware. Override via $TSS_RAW_DIR / $TSS_SYNTHETIC_DIR /
# $TSS_CHECKPOINT_DIR / $TSS_PROJECT_ROOT.
DATASET_DIR = get_synthetic_dir("TITML-3spk-v2")
RAW_DIR = get_raw_dir()
CHECKPOINT_DIR = get_checkpoint_dir("3speaker", "skim-transfer")

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Transfer Learning
# =============================================================================


def load_pretrained_weights(model, pretrained_path, device):
    """Load pretrained 2-speaker weights, reinitializing shape-mismatched layers.

    ``pretrained_path`` may be absolute, or relative to project root.
    Override at runtime via $TSS_PRETRAINED_PATH.
    """
    env_override = os.environ.get("TSS_PRETRAINED_PATH")
    if env_override:
        pretrained_full_path = Path(env_override).expanduser().resolve()
    else:
        p = Path(pretrained_path).expanduser()
        pretrained_full_path = p if p.is_absolute() else (project_root / p).resolve()

    if not pretrained_full_path.exists():
        print(f"\n⚠️  Warning: Pretrained model not found at {pretrained_full_path}")
        print("Training from scratch...")
        return model

    print(f"\n📥 Loading pretrained weights from: {pretrained_path}")
    checkpoint = torch.load(pretrained_full_path, map_location=device, weights_only=False)
    pretrained_dict = checkpoint["model_state_dict"]
    model_dict = model.state_dict()

    compatible_dict = {}
    reinitialized_layers = []
    skipped_layers = []

    for k, v in pretrained_dict.items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                compatible_dict[k] = v
            else:
                reinitialized_layers.append((k, v.shape, model_dict[k].shape))
        else:
            skipped_layers.append(k)

    model_dict.update(compatible_dict)
    model.load_state_dict(model_dict, strict=False)

    print("\n" + "=" * 60)
    print("Transfer Learning Summary")
    print("=" * 60)
    print(f"✓ Loaded: {len(compatible_dict)} layers from pretrained model")

    if reinitialized_layers:
        print(f"\n🔄 Reinitialized {len(reinitialized_layers)} layer(s) due to shape mismatch:")
        for name, old_shape, new_shape in reinitialized_layers:
            print(f"   {name}: {old_shape} → {new_shape}")

    if skipped_layers:
        print(f"\n⚠️  Skipped {len(skipped_layers)} layer(s) not in target model")

    print("=" * 60)
    return model


# =============================================================================
# Model Building
# =============================================================================


def build_model(device, use_transfer=True):
    """Build the SkiM 3-speaker model with optional transfer learning."""
    print("\n" + "=" * 60)
    print("Building SkiM 3-Speaker Model (Transfer Learning)")
    print("=" * 60)

    encoder = ConvEncoder(
        channel=MODEL_CONFIG["encoder"]["channel"],
        kernel_size=MODEL_CONFIG["encoder"]["kernel_size"],
        stride=MODEL_CONFIG["encoder"]["stride"],
    )
    print(f"✓ Encoder: Conv1D ({MODEL_CONFIG['encoder']['channel']} channels)")

    separator = SkiMSeparator(
        input_dim=MODEL_CONFIG["separator"]["input_dim"],
        causal=MODEL_CONFIG["separator"]["causal"],
        num_spk=MODEL_CONFIG["separator"]["num_spk"],
        predict_noise=MODEL_CONFIG["separator"]["predict_noise"],
        nonlinear=MODEL_CONFIG["separator"]["nonlinear"],
        layer=MODEL_CONFIG["separator"]["layer"],
        unit=MODEL_CONFIG["separator"]["unit"],
        segment_size=MODEL_CONFIG["separator"]["segment_size"],
        dropout=MODEL_CONFIG["separator"]["dropout"],
        mem_type=MODEL_CONFIG["separator"]["mem_type"],
        seg_overlap=MODEL_CONFIG["separator"]["seg_overlap"],
    )
    print(
        f"✓ Separator: SkiM ({MODEL_CONFIG['separator']['layer']} layers, "
        f"{MODEL_CONFIG['separator']['unit']} units, "
        f"{MODEL_CONFIG['separator']['num_spk']} speakers)"
    )

    decoder = ConvDecoder(
        channel=MODEL_CONFIG["decoder"]["channel"],
        kernel_size=MODEL_CONFIG["decoder"]["kernel_size"],
        stride=MODEL_CONFIG["decoder"]["stride"],
    )
    print(f"✓ Decoder: ConvTranspose1D")

    criterion = SISNRLoss()
    pit_wrapper = PITSolver(criterion=criterion)
    print(f"✓ Loss: SI-SNR with PIT")

    model = ESPnetEnhancementModel(
        encoder=encoder,
        separator=separator,
        decoder=decoder,
        mask_module=None,
        loss_wrappers=[pit_wrapper],
        loss_type="si_snr",
    )

    model = model.to(device)

    if use_transfer:
        model = load_pretrained_weights(model, TRANSFER_CONFIG["pretrained_path"], device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")
    return model


# =============================================================================
# Training Functions
# =============================================================================


def save_training_history(train_losses, val_losses, out_dir):
    """Save per-epoch loss history to a lightweight JSON file."""
    with open(out_dir / "training_history.json", "w") as f:
        json.dump({"train_losses": train_losses, "val_losses": val_losses}, f)


def load_training_history(out_dir):
    """Load per-epoch loss history from JSON if it exists."""
    history_path = out_dir / "training_history.json"
    if history_path.exists():
        with open(history_path) as f:
            data = json.load(f)
        return data.get("train_losses", []), data.get("val_losses", [])
    return [], []


def save_training_curves(train_losses, val_losses, out_dir):
    if not train_losses:
        return
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", marker="o", markersize=3)
    plt.plot(val_losses, label="Val Loss", marker="s", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Negative SI-SNR)")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot([-l for l in train_losses], label="Train SI-SNR", marker="o", markersize=3)
    plt.plot([-l for l in val_losses], label="Val SI-SNR", marker="s", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("SI-SNR (dB)")
    plt.title("Training and Validation SI-SNR")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close()


def train_epoch(model, train_loader, optimizer, scaler, device, epoch):
    """Train for one epoch with AMP."""
    model.train()
    total_loss = 0
    num_batches = len(train_loader)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        mix = batch["mix"].to(device)
        s1 = batch["s1"].to(device)
        s2 = batch["s2"].to(device)
        s3 = batch["s3"].to(device)

        batch_size = mix.size(0)
        mix_lengths = torch.full((batch_size,), mix.size(1), dtype=torch.long, device=device)
        ref_lengths = mix_lengths.clone()

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
            loss, stats, weight = model(
                speech_mix=mix,
                speech_mix_lengths=mix_lengths,
                speech_ref1=s1,
                speech_ref1_lengths=ref_lengths,
                speech_ref2=s2,
                speech_ref2_lengths=ref_lengths,
                speech_ref3=s3,
                speech_ref3_lengths=ref_lengths,
            )

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n⚠️ Warning: NaN/Inf loss at batch {batch_idx}, skipping...")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG["gradient_clip"])

