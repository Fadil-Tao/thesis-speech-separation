"""Full test-set SI-SNR / SI-SNRi evaluation across all model checkpoints.

Outputs per-model mean SI-SNR (estimate vs ref, PIT-aligned) and SI-SNRi
(improvement over input mix). Also dumps per-file CSV for variance inspection.
"""
import gc
import json
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from espnet2.enh.encoder.conv_encoder import ConvEncoder  # noqa: E402
from espnet2.enh.decoder.conv_decoder import ConvDecoder  # noqa: E402
from implementation.skim.skim_separator import SkiMSeparator  # noqa: E402
from implementation.skim_attention_v3.skim_attention_v3_separator import (  # noqa: E402
    SkiMAttentionV3Separator,
)

SAMPLE_RATE = 16000
TARGET_LEN = SAMPLE_RATE * 5
NUM_SPK = 3
EPS = 1e-8

TEST_ROOT = ROOT / "dataset/synthetic/3speaker/test"
CKPT_ROOT = ROOT / "checkpoints/3speaker"
OUT_DIR = ROOT / "inference/results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "skim":                            "skim",
    "skim-transfer":                   "skim",
    "skim-attention-v3":               "v3",
    "skim-attention-v3-reg-transfer":  "v3",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")


def normalize_length(x):
    if len(x) > TARGET_LEN:
        return x[:TARGET_LEN]
    if len(x) < TARGET_LEN:
        return np.pad(x, (0, TARGET_LEN - len(x)))
    return x


def load_wav(p):
    a, sr = sf.read(p)
    assert sr == SAMPLE_RATE
    return normalize_length(a.astype(np.float32))


def si_snr(est, ref):
    est = est - est.mean()
    ref = ref - ref.mean()
    s_target = np.dot(est, ref) / (np.dot(ref, ref) + EPS) * ref
    e_noise = est - s_target
    return 10 * np.log10((np.dot(s_target, s_target) + EPS) / (np.dot(e_noise, e_noise) + EPS))


def best_pit_sisnr(ests, refs):
    """Return mean SI-SNR over speakers under best permutation."""
    best = -1e9
    for perm in permutations(range(len(refs))):
        v = np.mean([si_snr(ests[perm[i]], refs[i]) for i in range(len(refs))])
        if v > best:
            best = v
    return best


def build(variant, ckpt_path):
    cfg = json.loads((ckpt_path.parent / "config.json").read_text())["model_config"]
    enc = ConvEncoder(**cfg["encoder"])
    dec = ConvDecoder(**cfg["decoder"])
    sep_cfg = dict(cfg["separator"])
    if variant == "skim":
        sep = SkiMSeparator(**sep_cfg)
    elif variant == "v3":
        sep = SkiMAttentionV3Separator(**sep_cfg)
    else:
        raise ValueError(variant)
    enc, sep, dec = enc.to(device).eval(), sep.to(device).eval(), dec.to(device).eval()
    ckpt = torch.load(ckpt_path, map_location=device)
    sd = ckpt.get("model_state_dict", ckpt)
    enc_sd = {k.replace("encoder.", "", 1): v for k, v in sd.items() if k.startswith("encoder.")}
    sep_sd = {k.replace("separator.", "", 1): v for k, v in sd.items() if k.startswith("separator.")}
    dec_sd = {k.replace("decoder.", "", 1): v for k, v in sd.items() if k.startswith("decoder.")}
    enc.load_state_dict(enc_sd)
    sep.load_state_dict(sep_sd)
    dec.load_state_dict(dec_sd)
    return enc, sep, dec


@torch.no_grad()
def separate(enc, sep, dec, mix_np):
    mix = torch.from_numpy(mix_np).unsqueeze(0).to(device)
    lengths = torch.tensor([mix.size(1)], dtype=torch.long, device=device)
    feats, flens = enc(mix, lengths)
    masked, _, _ = sep(feats, flens)
    out = []
    for m in masked:
        wav, _ = dec(m, lengths)
        out.append(wav.squeeze(0).cpu().numpy().astype(np.float32))
    return out


def main():
    import os
    max_files = int(os.environ.get("MAX_FILES", "0")) or None
    mix_files = sorted((TEST_ROOT / "mix").glob("*.wav"))
    if max_files:
        mix_files = mix_files[:max_files]
    print(f"test mixes: {len(mix_files)}")

    summary = {}
    for name, variant in MODELS.items():
        ckpt = CKPT_ROOT / name / "best_model.pth"
        if not ckpt.exists():
            print(f"[skip] {name}")
            continue
        csv_path = OUT_DIR / f"{name}_per_file.csv"
        if os.environ.get("SKIP_DONE") and csv_path.exists():
            n_done = sum(1 for _ in csv_path.open()) - 1
            if n_done >= len(mix_files):
                print(f"[done] {name} ({n_done} rows)")
                continue
        print(f"\n=== {name} ({variant}) ===")
        enc, sep, dec = build(variant, ckpt)

        sisnrs, sisnris = [], []
        csv_path = OUT_DIR / f"{name}_per_file.csv"
        csv_f = csv_path.open("w", buffering=1)
        csv_f.write("file_id,sisnr,sisnri\n")
        for mp in tqdm(mix_files, desc=name):
            fid = mp.stem
            mix_np = load_wav(mp)
            refs = [load_wav(TEST_ROOT / f"s{i}" / f"{fid}.wav") for i in range(1, NUM_SPK + 1)]
            ests = separate(enc, sep, dec, mix_np)
            ests = [normalize_length(e) for e in ests]
            sep_si = best_pit_sisnr(ests, refs)
            mix_si = np.mean([si_snr(mix_np, r) for r in refs])
            sisnrs.append(sep_si)
            sisnris.append(sep_si - mix_si)
            csv_f.write(f"{fid},{sep_si:.4f},{sep_si - mix_si:.4f}\n")
        csv_f.close()

        arr_si = np.array(sisnrs)
        arr_sii = np.array(sisnris)
        summary[name] = {
            "mean_sisnr":  float(arr_si.mean()),
            "std_sisnr":   float(arr_si.std()),
            "mean_sisnri": float(arr_sii.mean()),
            "std_sisnri":  float(arr_sii.std()),
            "median_sisnr": float(np.median(arr_si)),
            "n": len(arr_si),
        }
        print(f"  mean SI-SNR : {arr_si.mean():.3f} ± {arr_si.std():.3f} dB")
        print(f"  mean SI-SNRi: {arr_sii.mean():.3f} ± {arr_sii.std():.3f} dB")

        del enc, sep, dec
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2))
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
