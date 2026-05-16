# -*- coding: utf-8 -*-
"""Paper-faithful SkiM 2-speaker.

Overrides only `separator.segment_size` and `separator.dropout` to match
SkiM paper Sec 3.3, leaving everything else identical to the baseline at
train/2speaker/skim/train_skim_2spk.py.

Usage:
    python train/paper-faith/2speaker/train_skim_paperfaith.py
    python train/paper-faith/2speaker/train_skim_paperfaith.py --num-epochs 30
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
    p.add_argument("--num-epochs", type=int, default=100)
    p.add_argument("--dataset-dir", type=str, default=None)
    p.add_argument("--raw-dir", type=str, default=None)
    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--resume-from", type=str, default=None)
    args = p.parse_args()

    base.MODEL_CONFIG["separator"]["segment_size"] = 150
    base.MODEL_CONFIG["separator"]["dropout"] = 0.1

    ckpt_dir = args.checkpoint_dir or str(
        get_checkpoint_dir("paper-faith", "2speaker-skim")
    )

    print(f"\n[paper-faith][2spk-skim] K=150 dropout=0.1 → {ckpt_dir}\n")

    base.main(
        resume_from=args.resume_from,
        num_epochs=args.num_epochs,
        dataset_dir=args.dataset_dir,
        raw_dir=args.raw_dir,
        checkpoint_dir=ckpt_dir,
    )


if __name__ == "__main__":
    main()
