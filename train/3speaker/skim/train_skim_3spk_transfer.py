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
    "patience": 20,
    "seed": 42,
}

TRANSFER_CONFIG = {
    "pretrained_path": "checkpoints/2speaker/skim/best_model.pth",
    "description": "Transfer from SkiM 2-speaker to 3-speaker",
}

# Paths
DATASET_DIR = project_root / "dataset" / "synthetic" / "TITML-3spk-v2"
RAW_DIR = project_root / "dataset" / "raw" / "TTML-IDN"
CHECKPOINT_DIR = project_root / "checkpoints" / "3speaker" / "skim-transfer"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Transfer Learning
# =============================================================================


def load_pretrained_weights(model, pretrained_path, device):
    """Load pretrained 2-speaker weights, reinitializing shape-mismatched layers."""
    pretrained_full_path = project_root / pretrained_path

    if not pretrained_full_path.exists():
        print(f"\n⚠️  Warning: Pretrained model not found at {pretrained_full_path}")
        print("Training from scratch...")
        return model

    print(f"\n📥 Loading pretrained weights from: {pretrained_path}")
    checkpoint = torch.load(pretrained_full_path, map_location=device)
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

        if grad_norm > 10.0:
            print(f"\n⚠️ Warning: Large gradient norm ({grad_norm:.2f}), clipping applied")

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}", "SI-SNR": f"{-loss.item():.2f} dB"})

    return total_loss / num_batches


def validate(model, val_loader, device, epoch):
    """Validate the model with AMP."""
    model.eval()
    total_loss = 0
    num_batches = len(val_loader)

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]")
        for batch in pbar:
            mix = batch["mix"].to(device)
            s1 = batch["s1"].to(device)
            s2 = batch["s2"].to(device)
            s3 = batch["s3"].to(device)

            batch_size = mix.size(0)
            mix_lengths = torch.full((batch_size,), mix.size(1), dtype=torch.long, device=device)
            ref_lengths = mix_lengths.clone()

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

            total_loss += loss.item()
            pbar.set_postfix({"val_loss": f"{loss.item():.4f}", "val_SI-SNR": f"{-loss.item():.2f} dB"})

    return total_loss / num_batches


# =============================================================================
# Main Training Loop
# =============================================================================


def main(resume_from=None, num_epochs=None):
    random.seed(TRAIN_CONFIG["seed"])
    np.random.seed(TRAIN_CONFIG["seed"])
    torch.manual_seed(TRAIN_CONFIG["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(TRAIN_CONFIG["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n" + "=" * 60)
    print("Transfer Learning Configuration")
    print("=" * 60)
    print(f"Source: {TRANSFER_CONFIG['pretrained_path']}")
    print(f"Target: 3-speaker separation")
    print(f"Learning Rate: {TRAIN_CONFIG['learning_rate']} (10x smaller)")
    print("=" * 60)

    # Build utterance-level split from raw dataset
    print("\nBuilding utterance-level train/dev/test split...")
    train_utts, dev_utts, test_utts = build_utterance_split(
        RAW_DIR, seed=TRAIN_CONFIG["seed"], train_ratio=0.8, dev_ratio=0.1
    )

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
        split="dev", dataset_dir=DATASET_DIR, num_speakers=3, augment=False, target_duration=5.0
    )
    test_dataset = IndonesianMixDataset(
        split="test", dataset_dir=DATASET_DIR, num_speakers=3, augment=False, target_duration=5.0
    )

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

    model = build_model(device, use_transfer=True)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAIN_CONFIG["learning_rate"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=TRAIN_CONFIG["weight_decay"],
    )

    scaler = torch.amp.GradScaler("cuda")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=True,
    )

    best_val_loss = float("inf")
    start_epoch = 1
    target_num_epochs = num_epochs if num_epochs is not None else TRAIN_CONFIG["num_epochs"]

    if resume_from is not None:
        resume_path = CHECKPOINT_DIR / resume_from
        print(f"\nLoading checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        best_val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", best_val_loss))
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"✓ Resumed from epoch {start_epoch - 1}.")

    patience_counter = 0
    train_losses = []
    val_losses = []
    if resume_from is not None and "train_losses" in ckpt:
        train_losses = ckpt["train_losses"]
        val_losses = ckpt["val_losses"]

    print("\n" + "=" * 60)
    print("Starting Transfer Learning Training")
    print("=" * 60)

    try:
        for epoch in range(start_epoch, target_num_epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, scaler, device, epoch)
            train_losses.append(train_loss)

            val_loss = validate(model, dev_loader, device, epoch)
            val_losses.append(val_loss)

            scheduler.step(val_loss)

            print(
                f"Epoch {epoch:3d}: Train Loss = {train_loss:.4f}, "
                f"Val Loss = {val_loss:.4f}, "
                f"Val SI-SNR = {-val_loss:.2f} dB"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
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
                        "transfer_config": TRANSFER_CONFIG,
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                    },
                    CHECKPOINT_DIR / "best_model.pth",
                )
                print(f"  ✓ Best model saved (SI-SNR: {-val_loss:.2f} dB)")
            else:
                patience_counter += 1

            if patience_counter >= TRAIN_CONFIG["patience"]:
                print(f"\n⚠️ Early stopping triggered after {epoch} epochs")
                print(f"Best validation SI-SNR: {-best_val_loss:.2f} dB")
                break

            if epoch % 10 == 0:
                checkpoint_path = CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "train_losses": train_losses,
                        "val_losses": val_losses,
                    },
                    checkpoint_path,
                )
                print(f"  ✓ Checkpoint saved: epoch_{epoch}.pth")

            save_training_curves(train_losses, val_losses, CHECKPOINT_DIR)

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    finally:
        if train_losses:
            print("\n" + "=" * 60)
            print(f"Best validation SI-SNR: {-best_val_loss:.2f} dB")
            save_training_curves(train_losses, val_losses, CHECKPOINT_DIR)
            print(f"✓ Training curves saved")
            with open(CHECKPOINT_DIR / "config.json", "w") as f:
                json.dump(
                    {
                        "model_config": MODEL_CONFIG,
                        "train_config": TRAIN_CONFIG,
                        "transfer_config": TRANSFER_CONFIG,
                        "best_val_loss": best_val_loss,
                        "best_si_snr": -best_val_loss,
                    },
                    f,
                    indent=2,
                )
            print(f"✓ Config saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SkiM 3-Speaker Transfer Learning model")
    parser.add_argument("--resume-from", type=str, default=None,
                        help="Checkpoint filename to resume from (e.g. checkpoint_epoch_30.pth)")
    parser.add_argument("--num-epochs", type=int, default=None,
                        help="Total number of epochs to train (overrides config)")
    args = parser.parse_args()
    main(resume_from=args.resume_from, num_epochs=args.num_epochs)
