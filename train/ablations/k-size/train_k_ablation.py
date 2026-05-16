# -*- coding: utf-8 -*-
"""K-size (segment_size) ablation for baseline SkiM 2-speaker.

Reuses the baseline training script (train/2speaker/skim/train_skim_2spk.py),
overriding only `separator.segment_size` and optionally `separator.seg_overlap`.
Each ablation run writes to its own checkpoint dir:
    checkpoints/ablations/k-size/k{K}_overlap{T|F}/

Usage:
    python train_k_ablation.py --k 100
    python train_k_ablation.py --k 50 --seg-overlap --num-epochs 30
"""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "train" / "2speaker" / "skim"))

import train_skim_2spk as base
from utils.paths import get_checkpoint_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, required=True,
                   help="segment_size for SkiM separator")
    p.add_argument("--seg-overlap", action="store_true",
                   help="Enable 50%% segment overlap (default off)")
    p.add_argument("--num-epochs", type=int, default=30,
                   help="Epochs per ablation run (default 30, short for ranking)")
    p.add_argument("--dataset-dir", type=str, default=None)
    p.add_argument("--raw-dir", type=str, default=None)
    p.add_argument("--checkpoint-dir", type=str, default=None,
                   help="Override output dir (default: checkpoints/ablations/k-size/k{K}_overlap{T|F})")
    p.add_argument("--resume-from", type=str, default=None)
    args = p.parse_args()

    base.MODEL_CONFIG["separator"]["segment_size"] = args.k
    base.MODEL_CONFIG["separator"]["seg_overlap"] = args.seg_overlap

    if args.checkpoint_dir:
        ckpt_dir = args.checkpoint_dir
    else:
        tag = f"k{args.k}_overlap{'T' if args.seg_overlap else 'F'}"
        ckpt_dir = str(get_checkpoint_dir("ablations", "k-size") / tag)

    print(f"\n[K-ablation] K={args.k}  seg_overlap={args.seg_overlap}")
    print(f"[K-ablation] checkpoint_dir={ckpt_dir}\n")

    base.main(
        resume_from=args.resume_from,
        num_epochs=args.num_epochs,
        dataset_dir=args.dataset_dir,
        raw_dir=args.raw_dir,
        checkpoint_dir=ckpt_dir,
    )


if __name__ == "__main__":
    main()
