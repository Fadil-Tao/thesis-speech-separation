# -*- coding: utf-8 -*-
"""Paper-faithful SkiM 3-speaker transfer (from paper-faith 2spk skim).

Overrides separator.segment_size=150, separator.dropout=0.1.
Pretrained path defaults to paper-faith 2spk SkiM checkpoint; override via
$TSS_PRETRAINED_PATH or --pretrained-path.

Usage:
    # after 2spk paper-faith skim done:
    python train/paper-faith/3speaker/train_skim_transfer_paperfaith.py
"""
import argparse
import os
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "train" / "3speaker" / "skim"))

import train_skim_3spk_transfer as base
from utils.paths import get_checkpoint_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-epochs", type=int, default=100)
    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--resume-from", type=str, default=None)
    p.add_argument("--pretrained-path", type=str, default=None,
                   help="Path to 2spk pretrained checkpoint "
                        "(default: paper-faith 2spk skim best_model.pth)")
    args = p.parse_args()

    base.MODEL_CONFIG["separator"]["segment_size"] = 150
    base.MODEL_CONFIG["separator"]["dropout"] = 0.1

    # Default pretrained path: paper-faith 2spk skim
    pretrained_default = str(
        get_checkpoint_dir("paper-faith", "2speaker-skim") / "best_model.pth"
    )
    pretrained = args.pretrained_path or os.environ.get(
        "TSS_PRETRAINED_PATH", pretrained_default
    )
    os.environ["TSS_PRETRAINED_PATH"] = pretrained
    base.TRANSFER_CONFIG["pretrained_path"] = pretrained

    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else get_checkpoint_dir(
        "paper-faith", "3speaker-skim-transfer"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    base.CHECKPOINT_DIR = ckpt_dir

    print(f"\n[paper-faith][3spk-skim-transfer] K=150 dropout=0.1")
    print(f"  pretrained = {pretrained}")
    print(f"  out        = {ckpt_dir}\n")

    base.main(resume_from=args.resume_from, num_epochs=args.num_epochs)


if __name__ == "__main__":
    main()
