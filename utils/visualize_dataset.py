"""Waveform visualization for 3-speaker test dataset."""

import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TEST_DIR = os.path.join(PROJECT_ROOT, "dataset/synthetic/3speaker/test")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "visualization")

N_SAMPLES = 4
SEED = 42


def load_wav(path):
    audio, sr = sf.read(path)
    return audio, sr


def plot_sample(sample_id, axes_row):
    mix, sr = load_wav(os.path.join(TEST_DIR, "mix", sample_id))
    s1, _   = load_wav(os.path.join(TEST_DIR, "s1",  sample_id))
    s2, _   = load_wav(os.path.join(TEST_DIR, "s2",  sample_id))
    s3, _   = load_wav(os.path.join(TEST_DIR, "s3",  sample_id))

    t = np.linspace(0, len(mix) / sr, len(mix))

    spk_tracks = [
        ("S1", s1, "#e03030"),
        ("S2", s2, "#28a745"),
        ("S3", s3, "#1a7dcc"),
    ]

    # Mix panel: overlay all 3 speakers
    ax_mix = axes_row[0]
    for label, audio, color in spk_tracks:
        ax_mix.plot(t, audio, linewidth=0.4, color=color, alpha=0.7, label=label)
    ax_mix.set_title(f"Mix (overlay) — {sample_id}", fontsize=8)
    ax_mix.set_xlabel("Time (s)", fontsize=7)
    ax_mix.set_ylabel("Amplitude", fontsize=7)
    ax_mix.tick_params(labelsize=6)
    ax_mix.set_ylim(-1.05, 1.05)
    ax_mix.grid(True, linewidth=0.3, alpha=0.5)
    ax_mix.legend(fontsize=6, loc="upper right")

    # Individual speaker panels
    for ax, (label, audio, color) in zip(axes_row[1:], spk_tracks):
        ax.plot(t, audio, linewidth=0.4, color=color)
        ax.set_title(f"{label} — {sample_id}", fontsize=8)
        ax.set_xlabel("Time (s)", fontsize=7)
        ax.set_ylabel("Amplitude", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.set_ylim(-1.05, 1.05)
        ax.grid(True, linewidth=0.3, alpha=0.5)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_files = sorted(os.listdir(os.path.join(TEST_DIR, "mix")))
    random.seed(SEED)
    selected = random.sample(all_files, N_SAMPLES)

    fig, axes = plt.subplots(N_SAMPLES, 4, figsize=(18, N_SAMPLES * 3))
    fig.suptitle("3-Speaker Test Dataset — Waveform Visualization", fontsize=13, y=1.01)

    for i, sample_id in enumerate(selected):
        plot_sample(sample_id, axes[i])

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "3spk_test_waveforms.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
