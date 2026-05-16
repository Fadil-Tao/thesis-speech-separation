# -*- coding: utf-8 -*-
"""Paper-faithful SkiM 3-speaker (cold start).

Overrides separator.segment_size=150, separator.dropout=0.1.

Usage:
    python train/paper-faith/3speaker/train_skim_paperfaith.py
"""
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "train" / "3speaker" / "skim"))

import train_skim_3spk as base
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
        get_checkpoint_dir("paper-faith", "3speaker-skim")
    )

    print(f"\n[paper-faith][3spk-skim] K=150 dropout=0.1 → {ckpt_dir}\n")

    base.main(
        resume_from=args.resume_from,
        num_epochs=args.num_epochs,
        dataset_dir=args.dataset_dir,
        raw_dir=args.raw_dir,
        checkpoint_dir=ckpt_dir,
    )


if __name__ == "__main__":
    main()
