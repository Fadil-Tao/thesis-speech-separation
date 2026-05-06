# -*- coding: utf-8 -*-
"""SkiM Attention v3 — 3-Speaker Transfer Learning from γ-regularized 2spk.

Same as the standard v3 3spk transfer script except the pretrained source is
the γ-regularized 2spk checkpoint (Option A from the SkiM Attention v3 transfer
diagnostics: gates pulled toward 0 during pretraining → less 2spk-specific
specialization → cleaner transfer).

Pretrained source: checkpoints/2speaker/skim-attention-v3-reg/best_model.pth
Output dir:        checkpoints/3speaker/skim-attention-v3-reg-transfer/

Usage:
    uv run python train/3speaker/skim-attention-v3-reg/train_skim_attention_v3_reg_3spk_transfer.py
    uv run python train/3speaker/skim-attention-v3-reg/train_skim_attention_v3_reg_3spk_transfer.py --two-stage
"""

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

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "train"))

from datasets_utils import (
    build_utterance_split, DynamicMixDataset, IndonesianMixDataset
)

from espnet2.enh.encoder.conv_encoder import ConvEncoder
from espnet2.enh.decoder.conv_decoder import ConvDecoder
from espnet2.enh.espnet_model import ESPnetEnhancementModel
from espnet2.enh.loss.criterions.time_domain import SISNRLoss
from espnet2.enh.loss.wrappers.pit_solver import PITSolver

from implementation.skim_attention_v3.skim_attention_v3_separator import (
    SkiMAttentionV3Separator,
)


# =============================================================================
# Configuration
# =============================================================================

MODEL_CONFIG = {
    "encoder": {"channel": 256, "kernel_size": 32, "stride": 16},
    "decoder": {"channel": 256, "kernel_size": 32, "stride": 16},
    "separator": {
        "input_dim": 256,
        "causal": False,
        "num_spk": 3,
        "predict_noise": False,
        "nonlinear": "relu",
        "layer": 4,
        "unit": 256,
        "segment_size": 20,
        "dropout": 0.2,
        "mem_type": "hc",
        "seg_overlap": False,
        "num_heads": 4,
    },
}

TRAIN_CONFIG = {
    "batch_size": 8,
    "num_epochs": 100,
    "learning_rate": 1e-4,
    "weight_decay": 1e-5,
    "gradient_clip": 5.0,
    "seed": 42,
}

TRANSFER_CONFIG = {
    "pretrained_path": "checkpoints/2speaker/skim-attention-v3-reg/best_model.pth",
    "description": "Transfer from γ-regularized SkiM Attention v3 2-speaker → 3-speaker",
}

DATASET_DIR = project_root / "dataset" / "synthetic" / "TITML-3spk-v2"
RAW_DIR = project_root / "dataset" / "raw" / "TTML-IDN"
CHECKPOINT_DIR = project_root / "checkpoints" / "3speaker" / "skim-attention-v3-reg-transfer"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Transfer Learning
# =============================================================================


def load_pretrained_weights(model, pretrained_path, device):
    pretrained_full_path = project_root / pretrained_path

    if not pretrained_full_path.exists():
        print(f"\n⚠️  Pretrained model not found at {pretrained_full_path}")
        print("Run the γ-regularized 2spk pretraining first:")
        print("  uv run python train/2speaker/skim-attention-v3-reg/train_skim_attention_v3_reg_2spk.py")
        sys.exit(1)

    print(f"\n📥 Loading pretrained weights from: {pretrained_path}")
    checkpoint = torch.load(pretrained_full_path, map_location=device)
    pretrained_dict = checkpoint["model_state_dict"]
    model_dict = model.state_dict()

    compatible_dict = {}
    reinitialized_layers = []

    for k, v in pretrained_dict.items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                compatible_dict[k] = v
            else:
                reinitialized_layers.append((k, v.shape, model_dict[k].shape))

    model_dict.update(compatible_dict)
    model.load_state_dict(model_dict, strict=False)

    print(f"✓ Loaded {len(compatible_dict)} layers")
    if reinitialized_layers:
        print(f"🔄 Reinitialized {len(reinitialized_layers)} layer(s) due to shape mismatch:")
        for name, old_shape, new_shape in reinitialized_layers:
            print(f"   {name}: {old_shape} → {new_shape}")

    # Echo the γ values inherited from the regularized pretraining so the user
    # can confirm gates are near 0 (vs. the unregularized run where γ_attn[3] ≈ -0.45).
    print("\nInherited γ values (from regularized 2spk):")
    for name, p in model.named_parameters():
        if "gamma_attn" in name or "gamma_ffn" in name:
            print(f"  {name}: {p.data.item():+.4f}")

    return model


# =============================================================================
# Model Building
# =============================================================================


def build_model(device, use_transfer=True):
    print("\n" + "=" * 60)
    print("Building SkiM Attention v3 (γ-reg TL) 3-Speaker Model")
    print("=" * 60)

    encoder = ConvEncoder(**MODEL_CONFIG["encoder"])
    separator = SkiMAttentionV3Separator(**MODEL_CONFIG["separator"])
    decoder = ConvDecoder(**MODEL_CONFIG["decoder"])

    criterion = SISNRLoss()
    pit_wrapper = PITSolver(criterion=criterion)

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
# Training
# =============================================================================


def save_training_history(train_losses, val_losses, out_dir):
    with open(out_dir / "training_history.json", "w") as f:
        json.dump({"train_losses": train_losses, "val_losses": val_losses}, f)


def load_training_history(out_dir):
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
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot([-l for l in train_losses], label="Train SI-SNR", marker="o", markersize=3)
    plt.plot([-l for l in val_losses], label="Val SI-SNR", marker="s", markersize=3)
    plt.xlabel("Epoch"); plt.ylabel("SI-SNR (dB)"); plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close()


def train_epoch(model, train_loader, optimizer, scaler, device, epoch):
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
            loss, _, _ = model(
                speech_mix=mix, speech_mix_lengths=mix_lengths,
                speech_ref1=s1, speech_ref1_lengths=ref_lengths,
                speech_ref2=s2, speech_ref2_lengths=ref_lengths,
                speech_ref3=s3, speech_ref3_lengths=ref_lengths,
            )

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n⚠️ NaN/Inf loss at batch {batch_idx}, skipping...")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG["gradient_clip"])
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pbar.set_postfix({"loss": f"{loss.item():.4f}",
                          "SI-SNR": f"{-loss.item():.2f} dB"})

    return total_loss / num_batches


def validate(model, val_loader, device, epoch):
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
                loss, _, _ = model(
                    speech_mix=mix, speech_mix_lengths=mix_lengths,
                    speech_ref1=s1, speech_ref1_lengths=ref_lengths,
                    speech_ref2=s2, speech_ref2_lengths=ref_lengths,
                    speech_ref3=s3, speech_ref3_lengths=ref_lengths,
                )

            total_loss += loss.item()
            pbar.set_postfix({"val_loss": f"{loss.item():.4f}",
                              "val_SI-SNR": f"{-loss.item():.2f} dB"})

    return total_loss / num_batches


# =============================================================================
# Stage helpers (Option C compatibility — works on top of Option A pretrain)
# =============================================================================


def reset_layerscale_gates(model):
    zeroed = 0
    for name, param in model.named_parameters():
        if "gamma_attn" in name or "gamma_ffn" in name:
            param.data.zero_()
            zeroed += 1
    print(f"  ✓ Reset {zeroed} LayerScale gates to 0.0")
    return model


def is_stage1_param(name):
    return ("gamma_attn" in name or "gamma_ffn" in name or "output_fc.1" in name)


def freeze_for_stage1(model):
    n_train = 0
    for name, param in model.named_parameters():
        if is_stage1_param(name):
            param.requires_grad = True
            n_train += param.numel()
        else:
            param.requires_grad = False
    return model, n_train


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True
    return model


# =============================================================================
# Main
# =============================================================================


def main(resume_from=None, num_epochs=None, reset_gates=False,
         two_stage=False, stage1_epochs=5, stage1_lr=1e-3):
    random.seed(TRAIN_CONFIG["seed"])
    np.random.seed(TRAIN_CONFIG["seed"])
    torch.manual_seed(TRAIN_CONFIG["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(TRAIN_CONFIG["seed"])

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\n" + "=" * 60)
    print("Transfer Learning Configuration (Option A: γ-regularized source)")
    print("=" * 60)
    print(f"Source: {TRANSFER_CONFIG['pretrained_path']}")
    print(f"Target: 3-speaker separation")
    print(f"Learning Rate: {TRAIN_CONFIG['learning_rate']}")
    print("=" * 60)

    print("\nBuilding utterance-level train/dev/test split...")
    train_utts, dev_utts, test_utts = build_utterance_split(
        RAW_DIR, seed=TRAIN_CONFIG["seed"], train_ratio=0.8, dev_ratio=0.1
    )

    print("\nLoading datasets...")
    train_dataset = DynamicMixDataset(
        utterances_by_speaker=train_utts,
        num_speakers=3, target_duration=5.0, target_sr=16000,
        snr_range=(-5.0, 5.0), epoch_size=28800,
        gender_balance=True, augment=True,
    )
    dev_dataset = IndonesianMixDataset(
        split="dev", dataset_dir=DATASET_DIR, num_speakers=3,
        augment=False, target_duration=5.0,
    )
    test_dataset = IndonesianMixDataset(
        split="test", dataset_dir=DATASET_DIR, num_speakers=3,
        augment=False, target_duration=5.0,
    )

    train_loader = DataLoader(train_dataset, batch_size=TRAIN_CONFIG["batch_size"],
                              shuffle=True, num_workers=8, pin_memory=True,
                              persistent_workers=True)
    dev_loader = DataLoader(dev_dataset, batch_size=TRAIN_CONFIG["batch_size"],
                            shuffle=False, num_workers=8, pin_memory=True,
                            persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=TRAIN_CONFIG["batch_size"],
                             shuffle=False, num_workers=8, pin_memory=True,
                             persistent_workers=True)

    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Dev batches: {len(dev_loader)}")
    print(f"✓ Test batches: {len(test_loader)}")

    model = build_model(device, use_transfer=True)

    best_val_loss = float("inf")
    start_epoch = 1
    target_num_epochs = num_epochs if num_epochs is not None else TRAIN_CONFIG["num_epochs"]
    ckpt = None

    if resume_from is not None:
        resume_path = project_root / resume_from
        print(f"\nLoading checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        best_val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss", best_val_loss))
        start_epoch = ckpt.get("epoch", 0) + 1
        print(f"✓ Resumed from epoch {start_epoch - 1}.")

    if reset_gates:
        print("\n🔄 Resetting LayerScale gates (--reset-gates):")
        model = reset_layerscale_gates(model)

    in_stage1 = two_stage and start_epoch <= stage1_epochs
    if in_stage1:
        model, n_stage1 = freeze_for_stage1(model)
        print(f"\n🔒 Stage 1: backbone frozen, training {n_stage1} params (γ gates + output_fc.1) for {stage1_epochs} epochs at LR={stage1_lr}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    current_lr = stage1_lr if in_stage1 else TRAIN_CONFIG["learning_rate"]
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=current_lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=TRAIN_CONFIG["weight_decay"],
    )
    print(f"✓ Optimizer: LR={current_lr}")

    scaler = torch.amp.GradScaler("cuda")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6, verbose=True,
    )

    if ckpt is not None:
        if "optimizer_state_dict" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            except (ValueError, KeyError):
                print("  ⚠️ Optimizer state incompatible, using fresh optimizer")
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])

    train_losses, val_losses = load_training_history(CHECKPOINT_DIR)
    if not train_losses and resume_from is not None and ckpt is not None and "train_losses" in ckpt:
        train_losses = ckpt["train_losses"]
        val_losses = ckpt["val_losses"]
    if start_epoch > 1 and len(train_losses) >= start_epoch - 1:
        train_losses = train_losses[:start_epoch - 1]
        val_losses = val_losses[:start_epoch - 1]

    print("\n" + "=" * 60)
    print("Starting Transfer Learning Training (γ-reg source)")
    print("=" * 60)

    try:
        for epoch in range(start_epoch, target_num_epochs + 1):
            if two_stage and in_stage1 and epoch > stage1_epochs:
                print(f"\n🔓 Stage 2: unfreezing backbone, switching to LR={TRAIN_CONFIG['learning_rate']}")
                model = unfreeze_all(model)
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=TRAIN_CONFIG["learning_rate"],
                    betas=(0.9, 0.999),
                    eps=1e-8,
                    weight_decay=TRAIN_CONFIG["weight_decay"],
                )
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6, verbose=True,
                )
                in_stage1 = False

            train_loss = train_epoch(model, train_loader, optimizer, scaler, device, epoch)
            train_losses.append(train_loss)

            val_loss = validate(model, dev_loader, device, epoch)
            val_losses.append(val_loss)

            scheduler.step(val_loss)

            print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.4f}, "
                  f"Val Loss = {val_loss:.4f}, "
                  f"Val SI-SNR = {-val_loss:.2f} dB")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({
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
                }, CHECKPOINT_DIR / "best_model.pth")
                print(f"  ✓ Best model saved (SI-SNR: {-val_loss:.2f} dB)")

            if epoch % 10 == 0:
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "config": MODEL_CONFIG,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                }, CHECKPOINT_DIR / f"checkpoint_epoch_{epoch}.pth")
                print(f"  ✓ Checkpoint saved: epoch_{epoch}.pth")

            save_training_curves(train_losses, val_losses, CHECKPOINT_DIR)
            save_training_history(train_losses, val_losses, CHECKPOINT_DIR)

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        if train_losses:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "train_loss": train_losses[-1],
                "val_loss": val_losses[-1] if val_losses else float("inf"),
                "best_val_loss": best_val_loss,
                "train_losses": train_losses,
                "val_losses": val_losses,
            }, CHECKPOINT_DIR / "checkpoint_interrupted.pth")
            print(f"  ✓ Interrupted checkpoint saved (epoch {epoch})")
    finally:
        if train_losses:
            print("\n" + "=" * 60)
            print(f"Best validation SI-SNR: {-best_val_loss:.2f} dB")
            save_training_curves(train_losses, val_losses, CHECKPOINT_DIR)
            with open(CHECKPOINT_DIR / "config.json", "w") as f:
                json.dump({
                    "model_config": MODEL_CONFIG,
                    "train_config": TRAIN_CONFIG,
                    "transfer_config": TRANSFER_CONFIG,
                    "best_val_loss": best_val_loss,
                    "best_si_snr": -best_val_loss,
                }, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="3spk TL from γ-regularized v3 2spk pretrain (Option A)"
    )
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--reset-gates", action="store_true", default=False,
                        help="Zero LayerScale gates after loading pretrained weights")
    parser.add_argument("--two-stage", action="store_true", default=False,
                        help="Stage 1: train only γ + output head; Stage 2: full fine-tune")
    parser.add_argument("--stage1-epochs", type=int, default=5)
    parser.add_argument("--stage1-lr", type=float, default=1e-3)
    args = parser.parse_args()

    main(resume_from=args.resume_from, num_epochs=args.num_epochs,
         reset_gates=args.reset_gates,
         two_stage=args.two_stage, stage1_epochs=args.stage1_epochs,
         stage1_lr=args.stage1_lr)
