# -*- coding: utf-8 -*-
"""Paper-faithful SkiM-Attention v3 (non-reg) 3-speaker transfer.

Overrides separator.segment_size=150, separator.dropout=0.1.
Pretrained path defaults to paper-faith 2spk v3 (non-reg) checkpoint.

Usage:
    # after 2spk paper-faith v3 done:
    python train/paper-faith/3speaker/train_skim_attention_v3_transfer_paperfaith.py
"""
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "train" / "3speaker" / "skim-attention-v3"))

import train_skim_attention_v3_3spk_transfer as base
from utils.paths import get_checkpoint_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-epochs", type=int, default=100)
    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--resume-from", type=str, default=None)
    p.add_argument("--pretrained-path", type=str, default=None,
                   help="2spk pretrained ckpt path "
                        "(default: paper-faith 2spk v3 best_model.pth)")
    p.add_argument("--reset-gates", action="store_true")
    args = p.parse_args()

    base.MODEL_CONFIG["separator"]["segment_size"] = 150
    base.MODEL_CONFIG["separator"]["dropout"] = 0.1

    pretrained_default = str(
        get_checkpoint_dir("paper-faith", "2speaker-skim-attention-v3")
        / "best_model.pth"
    )
    pretrained = args.pretrained_path or pretrained_default
    base.TRANSFER_CONFIG["pretrained_path"] = pretrained

    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else get_checkpoint_dir(
        "paper-faith", "3speaker-skim-attention-v3-transfer"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    base.CHECKPOINT_DIR = ckpt_dir

    print(f"\n[paper-faith][3spk-v3-transfer] K=150 dropout=0.1")
    print(f"  pretrained = {pretrained}")
    print(f"  out        = {ckpt_dir}\n")

    base.main(
        resume_from=args.resume_from,
        num_epochs=args.num_epochs,
        reset_gates=args.reset_gates,
    )


if __name__ == "__main__":
    main()
