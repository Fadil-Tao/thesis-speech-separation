# -*- coding: utf-8 -*-
"""SkiM Attention v3 — 2-Speaker pretraining with γ-gate L2 regularization.

Identical to the standard v3 2spk training script except for an explicit L2
penalty on the LayerScale gates (γ_attn, γ_ffn). The default v3 model trained
without this penalty learned strongly negative γ_attn values — a 2spk-specific
*subtractive* denoising filter — that does not transfer well to 3spk.

This variant pulls γ toward 0 during pretraining so the architecture cannot
over-rely on task-specific gate behaviour, producing a more transferable
backbone for the 3spk fine-tune.

Output: checkpoints/2speaker/skim-attention-v3-reg/best_model.pth

Usage:
    uv run python train/2speaker/skim-attention-v3-reg/train_skim_attention_v3_reg_2spk.py
    uv run python train/2speaker/skim-attention-v3-reg/train_skim_attention_v3_reg_2spk.py --gate-reg-lambda 5e-3
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
        "num_spk": 2,
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
    "learning_rate": 1e-3,
    "weight_decay": 1e-5,
    "gradient_clip": 5.0,
    "seed": 42,
    "gate_reg_lambda": 1e-3,  # L2 penalty on γ_attn / γ_ffn
}

DATASET_DIR = project_root / "dataset" / "synthetic" / "TITML-2spk-v2"
RAW_DIR = project_root / "dataset" / "raw" / "TTML-IDN"
CHECKPOINT_DIR = project_root / "checkpoints" / "2speaker" / "skim-attention-v3-reg"

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Model Building
# =============================================================================


def build_model(device):
    print("\n" + "=" * 60)
    print("Building SkiM Attention v3 (γ-regularized) 2-Speaker Model")
    print("=" * 60)

    encoder = ConvEncoder(
        channel=MODEL_CONFIG["encoder"]["channel"],
        kernel_size=MODEL_CONFIG["encoder"]["kernel_size"],
        stride=MODEL_CONFIG["encoder"]["stride"],
    )

    separator = SkiMAttentionV3Separator(
        input_dim=MODEL_CONFIG["separator"]["input_dim"],
        causal=MODEL_CONFIG["separator"]["causal"],
        num_spk=MODEL_CONFIG["separator"]["num_spk"],
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

    decoder = ConvDecoder(
        channel=MODEL_CONFIG["decoder"]["channel"],
        kernel_size=MODEL_CONFIG["decoder"]["kernel_size"],
        stride=MODEL_CONFIG["decoder"]["stride"],
    )

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

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Cache γ parameter references once so train_epoch doesn't re-iterate every step.
    gate_params = [p for n, p in model.named_parameters()
                   if ("gamma_attn" in n or "gamma_ffn" in n)]
    print(f"Gate params under L2 penalty: {len(gate_params)} scalars")
    return model, gate_params


# =============================================================================
# Training
# =============================================================================


def gate_l2_penalty(gate_params):
    """Sum of squares over γ_attn / γ_ffn scalars."""
    if not gate_params:
        return torch.tensor(0.0)
    return sum((p ** 2).sum() for p in gate_params)


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
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.subplot(1, 2, 2)
    plt.plot([-l for l in train_losses], label="Train SI-SNR", marker="o", markersize=3)
    plt.plot([-l for l in val_losses], label="Val SI-SNR", marker="s", markersize=3)
    plt.xlabel("Epoch"); plt.ylabel("SI-SNR (dB)"); plt.title("SI-SNR")
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close()


def train_epoch(model, train_loader, optimizer, scaler, device, epoch,
                gate_params, gate_reg_lambda):
    model.train()
    total_loss = 0
    num_batches = len(train_loader)

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch_idx, batch in enumerate(pbar):
        mix = batch["mix"].to(device)
        s1 = batch["s1"].to(device)
        s2 = batch["s2"].to(device)

        batch_size = mix.size(0)
        mix_lengths = torch.full((batch_size,), mix.size(1), dtype=torch.long, device=device)
        ref_lengths = mix_lengths.clone()

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
            sisnr_loss, _, _ = model(
                speech_mix=mix,
                speech_mix_lengths=mix_lengths,
                speech_ref1=s1,
                speech_ref1_lengths=ref_lengths,
                speech_ref2=s2,
                speech_ref2_lengths=ref_lengths,
            )
            # γ regularizer is computed in fp32 outside autocast scope of MHA — but
            # since γ is a small set of leaf scalars, fp32 here is cheap and stable.
            reg = gate_reg_lambda * gate_l2_penalty(gate_params)
            loss = sisnr_loss + reg

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"\n⚠️ NaN/Inf loss at batch {batch_idx}, skipping...")
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), TRAIN_CONFIG["gradient_clip"])
        scaler.step(optimizer)
        scaler.update()

        total_loss += sisnr_loss.item()  # log raw SI-SNR, not penalized
        pbar.set_postfix({"loss": f"{sisnr_loss.item():.4f}",
                          "SI-SNR": f"{-sisnr_loss.item():.2f} dB",
                          "reg": f"{reg.item():.4f}"})

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

            batch_size = mix.size(0)
            mix_lengths = torch.full((batch_size,), mix.size(1), dtype=torch.long, device=device)
            ref_lengths = mix_lengths.clone()

            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu"):
                loss, _, _ = model(
                    speech_mix=mix,
                    speech_mix_lengths=mix_lengths,
                    speech_ref1=s1,
                    speech_ref1_lengths=ref_lengths,
                    speech_ref2=s2,
                    speech_ref2_lengths=ref_lengths,
                )

            total_loss += loss.item()
            pbar.set_postfix({"val_loss": f"{loss.item():.4f}",
                              "val_SI-SNR": f"{-loss.item():.2f} dB"})

    return total_loss / num_batches


# =============================================================================
# Main
# =============================================================================


def main(resume_from=None, num_epochs=None, gate_reg_lambda=None):
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

    if gate_reg_lambda is not None:
        TRAIN_CONFIG["gate_reg_lambda"] = gate_reg_lambda
    print(f"γ-gate L2 lambda: {TRAIN_CONFIG['gate_reg_lambda']}")

    print("\nBuilding utterance-level train/dev/test split...")
    train_utts, dev_utts, test_utts = build_utterance_split(
        RAW_DIR, seed=TRAIN_CONFIG["seed"], train_ratio=0.8, dev_ratio=0.1
    )

    print("\nLoading datasets...")
    train_dataset = DynamicMixDataset(
        utterances_by_speaker=train_utts,
        num_speakers=2,
        target_duration=5.0,
        target_sr=16000,
        snr_range=(-5.0, 5.0),
        epoch_size=28800,
        gender_balance=True,
        augment=True,
    )
    dev_dataset = IndonesianMixDataset(
        split="dev", dataset_dir=DATASET_DIR, num_speakers=2, augment=False,
        target_duration=5.0,
    )
    test_dataset = IndonesianMixDataset(
        split="test", dataset_dir=DATASET_DIR, num_speakers=2, augment=False,
        target_duration=5.0,
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

    model, gate_params = build_model(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=TRAIN_CONFIG["learning_rate"],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=TRAIN_CONFIG["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda")
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6, verbose=True,
    )

    best_val_loss = float("inf")
    start_epoch = 1
    target_num_epochs = num_epochs if num_epochs is not None else TRAIN_CONFIG["num_epochs"]
    ckpt = None

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

    train_losses, val_losses = load_training_history(CHECKPOINT_DIR)
    if not train_losses and resume_from is not None and ckpt is not None and "train_losses" in ckpt:
        train_losses = ckpt["train_losses"]
        val_losses = ckpt["val_losses"]
    if start_epoch > 1 and len(train_losses) >= start_epoch - 1:
        train_losses = train_losses[:start_epoch - 1]
        val_losses = val_losses[:start_epoch - 1]

    print("\n" + "=" * 60)
    print("Starting Training (γ-regularized v3)")
    print("=" * 60)

    try:
        for epoch in range(start_epoch, target_num_epochs + 1):
            train_loss = train_epoch(
                model, train_loader, optimizer, scaler, device, epoch,
                gate_params, TRAIN_CONFIG["gate_reg_lambda"],
            )
            train_losses.append(train_loss)

            val_loss = validate(model, dev_loader, device, epoch)
            val_losses.append(val_loss)

            scheduler.step(val_loss)

            with torch.no_grad():
                gate_max = max((p.abs().max().item() for p in gate_params), default=0.0)
            print(f"Epoch {epoch:3d}: Train Loss = {train_loss:.4f}, "
                  f"Val Loss = {val_loss:.4f}, "
                  f"Val SI-SNR = {-val_loss:.2f} dB, "
                  f"max|γ| = {gate_max:.4f}")

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
                    "train_config": TRAIN_CONFIG,
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
                    "best_val_loss": best_val_loss,
                    "best_si_snr": -best_val_loss,
                }, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train γ-regularized SkiM Attention v3 2-Speaker model"
    )
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--num-epochs", type=int, default=None)
    parser.add_argument("--gate-reg-lambda", type=float, default=None,
                        help="L2 penalty coefficient on γ_attn / γ_ffn (default: 1e-3)")
    args = parser.parse_args()
    main(resume_from=args.resume_from, num_epochs=args.num_epochs,
         gate_reg_lambda=args.gate_reg_lambda)
