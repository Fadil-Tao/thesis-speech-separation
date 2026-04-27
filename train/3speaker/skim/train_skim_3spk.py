# -*- coding: utf-8 -*-
"""Training script for SkiM 3-Speaker model.

This script trains a SkiM (Skipping Memory LSTM) model for 3-speaker speech separation
using the TITML-3spk synthetic dataset.

Usage:
    python train_skim_3spk.py

The script will:
1. Load the TITML-3spk dataset
2. Build the SkiM model with ESPnet framework
3. Train with SI-SNR loss and PIT
4. Save checkpoints and training curves
5. Evaluate on test set
"""

import os
import sys
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import librosa
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root and user site-packages to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "train"))

# Import shared dataset utilities
from datasets_utils import (
    build_utterance_split, DynamicMixDataset, IndonesianMixDataset
)

# ESPnet imports
from espnet2.enh.encoder.conv_encoder import ConvEncoder
from implementation.conv_encoder_abs import ConvEncoderAbs
from espnet2.enh.decoder.conv_decoder import ConvDecoder
from espnet2.enh.espnet_model import ESPnetEnhancementModel
from espnet2.enh.loss.criterions.time_domain import SISNRLoss
from espnet2.enh.loss.wrappers.pit_solver import PITSolver

# Import SkiM separator
from implementation.skim.skim_separator import SkiMSeparator


# =============================================================================
# Configuration
# =============================================================================

# Model configuration - Optimized for TITML dataset (~5M parameters)
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
        "causal": False,  # Non-causal for better offline separation quality
        "num_spk": 3,  # 3 speakers
        "predict_noise": False,
        "nonlinear": "relu",
        "layer": 4,
        "unit": 256,
        "segment_size": 20,
        "dropout": 0.2,
        "mem_type": "hc",
        "seg_overlap": False,
    },
}

# Training configuration
TRAIN_CONFIG = {
    "batch_size": 8,  # Increased from 4 for more stable gradients
    "num_epochs": 100,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "gradient_clip": 5.0,
    "patience": 20,  # Increased from 10 to handle noisy val curves
    "seed": 42,
}

# Paths
DATASET_DIR = project_root / "dataset" / "synthetic" / "TITML-3spk-v2"
RAW_DIR = project_root / "dataset" / "raw" / "TTML-IDN"
CHECKPOINT_DIR = project_root / "checkpoints" / "3speaker" / "skim"

# Create checkpoint directory
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def resolve_resume_path(resume_from):
    """Resolve checkpoint path from absolute/relative/shorthand input."""
    input_path = Path(resume_from)

    # 1) Absolute path
    if input_path.is_absolute() and input_path.exists():
        return input_path

    candidates = []

    # 2) Relative to project root as provided
    candidates.append(project_root / input_path)

    # 3) Relative to checkpoints root if user passed 2speaker/... or 3speaker/...
    if input_path.parts and input_path.parts[0] in {"2speaker", "3speaker"}:
        candidates.append(project_root / "checkpoints" / input_path)

    # 4) Relative to default 3-speaker SkiM checkpoint dir
    candidates.append(CHECKPOINT_DIR / input_path.name)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    # 5) Filename-only fallback: search across checkpoints/**
    if len(input_path.parts) == 1:
        matches = sorted((project_root / "checkpoints").glob(f"**/{input_path.name}"))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            match_list = "\n".join([f"- {m}" for m in matches[:10]])
            raise FileNotFoundError(
                "Multiple checkpoints matched your filename. "
                "Please pass one full path:\n"
                f"{match_list}"
            )

    searched = "\n".join([f"- {p}" for p in candidates])
    raise FileNotFoundError(
        "Checkpoint not found. Searched these paths:\n"
        f"{searched}\n"
        "Hint: use a path like checkpoints/3speaker/skim/checkpoint_epoch_30.pth"
    )


# =============================================================================
# Dataset
# =============================================================================


# =============================================================================
# Model Building
# =============================================================================


def build_model(device):
    """Build the SkiM model."""
    print("\n" + "=" * 60)
    print("Building SkiM 3-Speaker Model")
    print("=" * 60)

    # Encoder
    encoder = ConvEncoder(
        channel=MODEL_CONFIG["encoder"]["channel"],
        kernel_size=MODEL_CONFIG["encoder"]["kernel_size"],
        stride=MODEL_CONFIG["encoder"]["stride"],
    )
    print(f"✓ Encoder: Conv1D ({MODEL_CONFIG['encoder']['channel']} channels)")

    # Separator
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

    # Decoder
    decoder = ConvDecoder(
        channel=MODEL_CONFIG["decoder"]["channel"],
        kernel_size=MODEL_CONFIG["decoder"]["kernel_size"],
        stride=MODEL_CONFIG["decoder"]["stride"],
    )
    print(f"✓ Decoder: ConvTranspose1D")

    # Loss function
    criterion = SISNRLoss()
    pit_wrapper = PITSolver(criterion=criterion)
    print(f"✓ Loss: SI-SNR with PIT")

    # Full model
    model = ESPnetEnhancementModel(
        encoder=encoder,
        separator=separator,
        decoder=decoder,
        mask_module=None,
        loss_wrappers=[pit_wrapper],
        loss_type="si_snr",
        # num_spk removed - using loss_wrappers instead
    )

    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel parameters: {num_params:,}")

    return model


# =============================================================================
# Training Functions
# =============================================================================


def train_epoch(model, train_loader, optimizer, scaler, device, epoch):
    """Train for one epoch with AMP."""
    model.train()
    total_loss = 0
    num_batches = len(train_loader)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        mix = batch["mix"].to(device)
        s1 = batch["s1"].to(device)
        s2 = batch["s2"].to(device)
        s3 = batch["s3"].to(device)

        # Prepare inputs for ESPnet model
        batch_size = mix.size(0)
        mix_lengths = torch.full(
            (batch_size,), mix.size(1), dtype=torch.long, device=device
        )

        speech_ref1 = s1
        speech_ref2 = s2
        speech_ref3 = s3
        ref_lengths = mix_lengths.clone()

        optimizer.zero_grad()

        # Forward pass with AMP
        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
            loss, stats, weight = model(
                speech_mix=mix,
                speech_mix_lengths=mix_lengths,
                speech_ref1=speech_ref1,
                speech_ref1_lengths=ref_lengths,
                speech_ref2=speech_ref2,
                speech_ref2_lengths=ref_lengths,
                speech_ref3=speech_ref3,
                speech_ref3_lengths=ref_lengths,
            )

        # Check for NaN
        if torch.isnan(loss) or torch.isinf(loss):
            print(
                f"\n⚠️ Warning: NaN/Inf loss detected at batch {batch_idx}, skipping..."
            )
            continue

        # Backward pass with gradient scaling
        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), TRAIN_CONFIG["gradient_clip"]
        )

        # Check gradient norm
        if grad_norm > 10.0:
            print(
                f"\n⚠️ Warning: Large gradient norm ({grad_norm:.2f}), clipping applied"
            )

        scaler.step(optimizer)
        scaler.update()

        # Accumulate loss
        total_loss += loss.item()

        # Update progress bar with both loss and SI-SNR
        si_snr_db = -loss.item()  # Convert negative loss to positive SI-SNR
        pbar.set_postfix(
            {"loss": f"{loss.item():.4f}", "SI-SNR": f"{si_snr_db:.2f} dB"}
        )

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, val_loader, device, epoch):
    """Validate the model with AMP."""
    model.eval()
    total_loss = 0
    num_batches = len(val_loader)

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        for batch in pbar:
            # Move to device
            mix = batch["mix"].to(device)
            s1 = batch["s1"].to(device)
            s2 = batch["s2"].to(device)
            s3 = batch["s3"].to(device)

            # Prepare inputs
            batch_size = mix.size(0)
            mix_lengths = torch.full(
                (batch_size,), mix.size(1), dtype=torch.long, device=device
            )

            speech_ref1 = s1
            speech_ref2 = s2
            speech_ref3 = s3
            ref_lengths = mix_lengths.clone()

            # Forward pass with AMP
            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                loss, stats, weight = model(
                    speech_mix=mix,
                    speech_mix_lengths=mix_lengths,
                    speech_ref1=speech_ref1,
                    speech_ref1_lengths=ref_lengths,
                    speech_ref2=speech_ref2,
                    speech_ref2_lengths=ref_lengths,
                    speech_ref3=speech_ref3,
                    speech_ref3_lengths=ref_lengths,
                )

            total_loss += loss.item()

            # Update progress bar with both loss and SI-SNR
            val_si_snr_db = -loss.item()
            pbar.set_postfix(
                {
                    "val_loss": f"{loss.item():.4f}",
                    "val_SI-SNR": f"{val_si_snr_db:.2f} dB",
                }
            )

    avg_loss = total_loss / num_batches
    return avg_loss


# =============================================================================
# Main Training Loop
# =============================================================================


def main(resume_from=None, num_epochs=None):
    """Main training function."""
    # Set random seeds
    random.seed(TRAIN_CONFIG["seed"])
    np.random.seed(TRAIN_CONFIG["seed"])
    torch.manual_seed(TRAIN_CONFIG["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(TRAIN_CONFIG["seed"])

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build utterance-level split from raw dataset
    print("\nBuilding utterance-level train/dev/test split...")
    train_utts, dev_utts, test_utts = build_utterance_split(
        RAW_DIR, seed=TRAIN_CONFIG["seed"], train_ratio=0.8, dev_ratio=0.1
    )

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = DynamicMixDataset(
        utterances_by_speaker=train_utts,
        num_speakers=3,
        target_duration=5.0,
        target_sr=16000,
        snr_range=(-5.0, 5.0),
        epoch_size=28800,
        gender_balance=True,
        augment=True,
    )
    dev_dataset = IndonesianMixDataset(
        split="dev", dataset_dir=DATASET_DIR, num_speakers=3, augment=False
    , target_duration=5.0)
    test_dataset = IndonesianMixDataset(
        split="test", dataset_dir=DATASET_DIR, num_speakers=3, augment=False
    , target_duration=5.0)

    # Create data loaders with optimized settings
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
    )

    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Dev batches: {len(dev_loader)}")
    print(f"✓ Test batches: {len(test_loader)}")

    # Build model
    model = build_model(device)

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAIN_CONFIG["learning_rate"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=TRAIN_CONFIG["weight_decay"],
    )

    # AMP Gradient Scaler
    scaler = torch.amp.GradScaler("cuda")

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,  # Increased from 3 to prevent premature LR decay
        min_lr=1e-6,
        verbose=True,
    )

    # Training loop
    best_val_loss = float("inf")
    start_epoch = 1
    target_num_epochs = (
        num_epochs if num_epochs is not None else TRAIN_CONFIG["num_epochs"]
    )

    if resume_from is not None:
        resume_path = resolve_resume_path(resume_from)

        print(f"\nLoading checkpoint: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "scaler_state_dict" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])

        best_val_loss = checkpoint.get("best_val_loss", checkpoint.get("val_loss", best_val_loss))
        start_epoch = checkpoint.get("epoch", 0) + 1

        print(
            f"✓ Resumed from epoch {start_epoch - 1}. "
            f"Continuing until epoch {target_num_epochs}."
        )

    if start_epoch > target_num_epochs:
        print(
            f"Checkpoint is already at epoch {start_epoch - 1}, "
            f"which is >= target epoch {target_num_epochs}."
        )
        return

    patience_counter = 0
    train_losses = []
    val_losses = []

    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    for epoch in range(start_epoch, target_num_epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, scaler, device, epoch)
        train_losses.append(train_loss)

        # Validate
        val_loss = validate(model, dev_loader, device, epoch)
        val_losses.append(val_loss)

        # Scheduler step
        scheduler.step(val_loss)

        # Print progress
        print(
            f"Epoch {epoch:3d}: Train Loss = {train_loss:.4f}, "
            f"Val Loss = {val_loss:.4f}, "
            f"Val SI-SNR = {-val_loss:.2f} dB"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save checkpoint
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "best_val_loss": best_val_loss,
                "config": MODEL_CONFIG,
            }
            torch.save(checkpoint, CHECKPOINT_DIR / "best_model.pth")
            print(f"  ✓ Best model saved (SI-SNR: {-val_loss:.2f} dB)")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= TRAIN_CONFIG["patience"]:
            print(f"\n⚠️ Early stopping triggered after {epoch} epochs")
            print(f"Best validation SI-SNR: {-best_val_loss:.2f} dB")
            break

        # Save periodic checkpoint
        if epoch % 10 == 0:
            checkpoint_path = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "config": MODEL_CONFIG,
                },
                checkpoint_path,
            )
            print(f"  ✓ Checkpoint saved: epoch_{epoch}.pth")

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best validation SI-SNR: {-best_val_loss:.2f} dB")

    # Plot training curves
    plt.figure(figsize=(12, 5))

    # Loss curves
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", marker="o", markersize=3)
    plt.plot(val_losses, label="Val Loss", marker="s", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Negative SI-SNR)")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    # SI-SNR curves
    plt.subplot(1, 2, 2)
    plt.plot([-l for l in train_losses], label="Train SI-SNR", marker="o", markersize=3)
    plt.plot([-l for l in val_losses], label="Val SI-SNR", marker="s", markersize=3)
    plt.xlabel("Epoch")
    plt.ylabel("SI-SNR (dB)")
    plt.title("Training and Validation SI-SNR")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(CHECKPOINT_DIR / "training_curves.png", dpi=150)
    print(f"✓ Training curves saved")

    # Save config
    with open(CHECKPOINT_DIR / "config.json", "w") as f:
        json.dump(
            {
                "model_config": MODEL_CONFIG,
                "train_config": TRAIN_CONFIG,
                "best_val_loss": best_val_loss,
                "best_si_snr": -best_val_loss,
            },
            f,
            indent=2,
        )
    print(f"✓ Config saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SkiM 3-speaker model")
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Total epochs to train up to (default from TRAIN_CONFIG)",
    )
    args = parser.parse_args()

    main(resume_from=args.resume_from, num_epochs=args.num_epochs)
