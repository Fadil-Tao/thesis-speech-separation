# -*- coding: utf-8 -*-
"""Training script for SkiM Attention 3-Speaker model.

This script trains a SkiM Attention model (with Multi-Head Self-Attention)
for 3-speaker speech separation using the TITML-3spk synthetic dataset.

Usage:
    python train_skim_attention_3spk.py

The script will:
1. Load the TITML-3spk dataset
2. Build the SkiM Attention model with ESPnet framework
3. Train with SI-SNR loss and PIT
4. Save checkpoints and training curves
5. Evaluate on test set
"""

import os
import sys
import json
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# ESPnet imports
from espnet2.enh.encoder.conv_encoder import ConvEncoder
from espnet2.enh.decoder.conv_decoder import ConvDecoder
from espnet2.enh.espnet_model import ESPnetEnhancementModel
from espnet2.enh.loss.criterions.time_domain import SISNRLoss
from espnet2.enh.loss.wrappers.pit_solver import PITSolver

# Import SkiM Attention separator
from implementation.skim_attention.skim_attention_separator import (
    SkiMAttentionSeparator,
)


# =============================================================================
# Configuration
# =============================================================================

# Model configuration - Small-Causal variant with Attention
MODEL_CONFIG = {
    "encoder": {
        "channel": 512,
        "kernel_size": 32,
        "stride": 16,
    },
    "decoder": {
        "channel": 512,
        "kernel_size": 32,
        "stride": 16,
    },
    "separator": {
        "input_dim": 512,
        "causal": True,
        "num_spk": 3,  # 3 speakers
        "predict_noise": False,
        "nonlinear": "relu",
        "layer": 3,
        "unit": 512,
        "segment_size": 20,
        "dropout": 0.0,
        "mem_type": "hc",
        "seg_overlap": False,
        "num_heads": 8,  # Attention heads: 512 / 8 = 64 dims per head
    },
}

# Training configuration
TRAIN_CONFIG = {
    "batch_size": 4,
    "num_epochs": 100,
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "gradient_clip": 5.0,
    "patience": 10,
    "seed": 42,
}

# Paths
DATASET_DIR = project_root / "dataset" / "synthetic" / "TITML-3spk"
CHECKPOINT_DIR = project_root / "checkpoints" / "3speaker" / "skim-attention"

# Create checkpoint directory
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Dataset
# =============================================================================


class IndonesianMixDataset(Dataset):
    """Dataset for Indonesian speech mixtures (3 speakers)."""

    def __init__(self, split="train", dataset_dir=DATASET_DIR):
        self.split = split
        self.dataset_dir = Path(dataset_dir)
        self.split_dir = self.dataset_dir / split

        # Get all mixture files
        self.mix_files = sorted(list((self.split_dir / "mix").glob("*.wav")))

        print(f"[{split}] Loaded {len(self.mix_files)} mixtures")

    def __len__(self):
        return len(self.mix_files)

    def __getitem__(self, idx):
        mix_file = self.mix_files[idx]
        file_id = mix_file.stem

        # Load mixture
        mix, sr = sf.read(mix_file)

        # Load sources (3 speakers)
        s1, _ = sf.read(self.split_dir / "s1" / f"{file_id}.wav")
        s2, _ = sf.read(self.split_dir / "s2" / f"{file_id}.wav")
        s3, _ = sf.read(self.split_dir / "s3" / f"{file_id}.wav")

        return {
            "mix": torch.FloatTensor(mix),
            "s1": torch.FloatTensor(s1),
            "s2": torch.FloatTensor(s2),
            "s3": torch.FloatTensor(s3),
            "file_id": file_id,
        }


def collate_fn(batch):
    """Collate function for DataLoader."""
    # Find max length in batch
    max_len = max([b["mix"].shape[0] for b in batch])

    # Pad sequences
    mix_batch = []
    s1_batch = []
    s2_batch = []
    s3_batch = []
    file_ids = []

    for b in batch:
        mix = b["mix"]
        s1 = b["s1"]
        s2 = b["s2"]
        s3 = b["s3"]

        # Pad if necessary
        if mix.shape[0] < max_len:
            pad_len = max_len - mix.shape[0]
            mix = torch.nn.functional.pad(mix, (0, pad_len))
            s1 = torch.nn.functional.pad(s1, (0, pad_len))
            s2 = torch.nn.functional.pad(s2, (0, pad_len))
            s3 = torch.nn.functional.pad(s3, (0, pad_len))

        mix_batch.append(mix)
        s1_batch.append(s1)
        s2_batch.append(s2)
        s3_batch.append(s3)
        file_ids.append(b["file_id"])

    return {
        "mix": torch.stack(mix_batch),
        "s1": torch.stack(s1_batch),
        "s2": torch.stack(s2_batch),
        "s3": torch.stack(s3_batch),
        "file_id": file_ids,
    }


# =============================================================================
# Model Building
# =============================================================================


def build_model(device):
    """Build the SkiM Attention model."""
    print("\n" + "=" * 60)
    print("Building SkiM Attention 3-Speaker Model")
    print("=" * 60)

    # Encoder
    encoder = ConvEncoder(
        channel=MODEL_CONFIG["encoder"]["channel"],
        kernel_size=MODEL_CONFIG["encoder"]["kernel_size"],
        stride=MODEL_CONFIG["encoder"]["stride"],
    )
    print(f"✓ Encoder: Conv1D ({MODEL_CONFIG['encoder']['channel']} channels)")

    # Separator with Attention
    separator = SkiMAttentionSeparator(
        input_dim=MODEL_CONFIG["separator"]["input_dim"],
        causal=MODEL_CONFIG["separator"]["causal"],
        # num_spk removed - using loss_wrappers instead
        predict_noise=MODEL_CONFIG["separator"]["predict_noise"],
        nonlinear=MODEL_CONFIG["separator"]["nonlinear"],
        layer=MODEL_CONFIG["separator"]["layer"],
        unit=MODEL_CONFIG["separator"]["unit"],
        segment_size=MODEL_CONFIG["separator"]["segment_size"],
        dropout=MODEL_CONFIG["separator"]["dropout"],
        num_heads=MODEL_CONFIG["separator"]["num_heads"],
        mem_type=MODEL_CONFIG["separator"]["mem_type"],
        seg_overlap=MODEL_CONFIG["separator"]["seg_overlap"],
    )
    print(
        f"✓ Separator: SkiM Attention ({MODEL_CONFIG['separator']['layer']} layers, "
        f"{MODEL_CONFIG['separator']['unit']} units, "
        f"{MODEL_CONFIG['separator']['num_heads']} heads, "
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


def train_epoch(model, train_loader, optimizer, device, epoch):
    """Train for one epoch."""
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

        # Forward pass
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

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), TRAIN_CONFIG["gradient_clip"]
        )

        optimizer.step()

        # Accumulate loss
        total_loss += loss.item()

        # Update progress bar
        # Update progress bar with both loss and SI-SNR
        si_snr_db = -loss.item()  # Convert negative loss to positive SI-SNR
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "SI-SNR": f"{si_snr_db:.2f} dB"})

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, val_loader, device, epoch):
    """Validate the model."""
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

            # Forward pass
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
            pbar.set_postfix({"val_loss": f"{loss.item():.4f}", "val_SI-SNR": f"{val_si_snr_db:.2f} dB"})

    avg_loss = total_loss / num_batches
    return avg_loss


# =============================================================================
# Main Training Loop
# =============================================================================


def main():
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

    # Create datasets
    print("\nLoading datasets...")
    train_dataset = IndonesianMixDataset("train")
    dev_dataset = IndonesianMixDataset("dev")
    test_dataset = IndonesianMixDataset("test")

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
    )
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=TRAIN_CONFIG["batch_size"],
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
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

    # Scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=3,
        verbose=True,
    )

    # Training loop
    best_val_loss = float("inf")
    patience_counter = 0
    train_losses = []
    val_losses = []

    print("\n" + "=" * 60)
    print("Starting Training")
    print("=" * 60)

    for epoch in range(1, TRAIN_CONFIG["num_epochs"] + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
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
                    "train_loss": train_loss,
                    "val_loss": val_loss,
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
    main()
