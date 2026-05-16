# -*- coding: utf-8 -*-
"""Paper-faithful SkiM-Attention v3 3-speaker (cold start).

Overrides separator.segment_size=150, separator.dropout=0.1.

Usage:
    python train/paper-faith/3speaker/train_skim_attention_v3_paperfaith.py
"""
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "train" / "3speaker" / "skim-attention-v3"))

import train_skim_attention_v3_3spk as base
from utils.paths import get_checkpoint_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-epochs", type=int, default=100)
    p.add_argument("--checkpoint-dir", type=str, default=None)
    args = p.parse_args()

    base.MODEL_CONFIG["separator"]["segment_size"] = 150
    base.MODEL_CONFIG["separator"]["dropout"] = 0.1
    base.TRAIN_CONFIG["num_epochs"] = args.num_epochs

    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else get_checkpoint_dir(
        "paper-faith", "3speaker-skim-attention-v3"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    base.CHECKPOINT_DIR = ckpt_dir

    print(f"\n[paper-faith][3spk-v3] K=150 dropout=0.1 epochs={args.num_epochs} → {ckpt_dir}\n")

    base.main()


if __name__ == "__main__":
    main()
