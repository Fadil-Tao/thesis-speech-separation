# -*- coding: utf-8 -*-
"""Paper-faithful SkiM-Attention v3-reg 2-speaker (γ-regularized).

Overrides separator.segment_size=150, separator.dropout=0.1.
γ-regularization (gate_reg_lambda) unchanged from baseline.

Usage:
    python train/paper-faith/2speaker/train_skim_attention_v3_reg_paperfaith.py
"""
import argparse
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "train" / "2speaker" / "skim-attention-v3-reg"))

import train_skim_attention_v3_reg_2spk as base
from utils.paths import get_checkpoint_dir


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-epochs", type=int, default=100)
    p.add_argument("--checkpoint-dir", type=str, default=None)
    p.add_argument("--resume-from", type=str, default=None)
    p.add_argument("--gate-reg-lambda", type=float, default=None,
                   help="Override γ-reg lambda (default uses base config)")
    args = p.parse_args()

    base.MODEL_CONFIG["separator"]["segment_size"] = 150
    base.MODEL_CONFIG["separator"]["dropout"] = 0.1

    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else get_checkpoint_dir(
        "paper-faith", "2speaker-skim-attention-v3-reg"
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    base.CHECKPOINT_DIR = ckpt_dir

    print(f"\n[paper-faith][2spk-v3-reg] K=150 dropout=0.1 → {ckpt_dir}\n")

    base.main(
        resume_from=args.resume_from,
        num_epochs=args.num_epochs,
        gate_reg_lambda=args.gate_reg_lambda,
    )


if __name__ == "__main__":
    main()
