"""Plot val loss curves per num_spk group, all paper-faith models overlaid.

Reads training_history.json from checkpoints_from_r2/paper-faith/<model>/.
Outputs high-resolution PNG  figures suitable for thesis Bab 4.5.1.

Usage:
    python analysis/plot_val_loss.py
    python analysis/plot_val_loss.py --metric sisnr   # plot as SI-SNR (negate)
    python analysis/plot_val_loss.py --root path/to/checkpoints
"""
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CKPT_ROOT = PROJECT_ROOT / "checkpoints_from_r2" / "paper-faith"
OUT_DIR = PROJECT_ROOT / "analysis" / "figures"

# Group model names by num_spk + display label + color
GROUPS = {
    2: [
        ("2speaker-skim",                   "SkiM",                 "#1f77b4"),
        ("2speaker-skim-attention-v3-reg",  "SkiM-Attention v3",    "#d62728"),
    ],
    3: [
        ("3speaker-skim",                          "SkiM ",                  "#1f77b4"),
        ("3speaker-skim-attention-v3",             "SkiM-Attention",     "#d62728"),
        ("3speaker-skim-transfer",                 "SkiM (transfer)",              "#2ca02c"),
        ("3speaker-skim-attention-v3-reg-transfer","SkiM-Attention (transfer)", "#ff7f0e"),
    ],
}


def load_history(ckpt_root: Path, name: str):
    path = ckpt_root / name / "training_history.json"
    if not path.exists():
        print(f"  [miss] {path}")
        return None
    data = json.loads(path.read_text())
    return data.get("val_losses", [])


def plot_group(num_spk: int, models, ckpt_root: Path, metric: str, out_dir: Path):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=200)

    plotted = 0
    for name, label, color in models:
        val = load_history(ckpt_root, name)
        if not val:
            continue
        val = np.array(val, dtype=np.float64)
        if metric == "sisnr":
            y = -val
            ylabel = "Val SI-SNR (dB)"
        else:
            y = val
            ylabel = "Val Loss (negatif SI-SNR)"
        epochs = np.arange(1, len(y) + 1)
        ax.plot(
            epochs, y,
            label=label, color=color,
            linewidth=1.8, alpha=0.95,
        )
        # marker best epoch
        if metric == "sisnr":
            best_idx = int(np.argmax(y))
        else:
            best_idx = int(np.argmin(y))
        ax.scatter(
            [epochs[best_idx]], [y[best_idx]],
            color=color, s=60, zorder=5,
            edgecolors="black", linewidths=0.8,
        )
        plotted += 1

    if plotted == 0:
        print(f"[skip] no data for {num_spk}-speaker group")
        plt.close(fig)
        return

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(
        f"Kurva Validasi Pelatihan — {num_spk}-Speaker (Paper-Faith)",
        fontsize=13, pad=12,
    )
    ax.grid(True, which="both", alpha=0.3, linestyle="--")
    ax.minorticks_on()
    ax.grid(True, which="minor", alpha=0.15, linestyle=":")
    ax.legend(
        loc="lower right" if metric == "sisnr" else "upper right",
        fontsize=10, frameon=True, framealpha=0.9,
    )
    ax.tick_params(axis="both", which="major", labelsize=10)

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"val_{metric}_{num_spk}spk"
    fig.tight_layout()
    fig.savefig(f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=DEFAULT_CKPT_ROOT)
    p.add_argument("--metric", choices=["loss", "sisnr"], default="loss",
                   help="'loss' plot raw negative-SISNR loss (default); 'sisnr' plot as positive SI-SNR")
    p.add_argument("--out", type=Path, default=OUT_DIR)
    args = p.parse_args()

    if not args.root.exists():
        raise SystemExit(f"ckpt root not found: {args.root}")

    print(f"ckpt_root: {args.root}")
    print(f"metric:    {args.metric}")
    print(f"out_dir:   {args.out}\n")

    for num_spk, models in GROUPS.items():
        print(f"=== {num_spk}-speaker ===")
        plot_group(num_spk, models, args.root, args.metric, args.out)
        print()


if __name__ == "__main__":
    main()
