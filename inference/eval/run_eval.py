"""Paper-faith eval pipeline.

For each model in best-model-list.txt:
  1. Download best_model.pth from Google Drive (if not present)
  2. Run on full test set (matching num_spk dataset)
  3. Save per-file SI-SNR/SI-SNRi/STOI/PESQ CSV (all 3600)
  4. Save mixture + estimated + ground-truth WAVs (first 450 only)
  5. Append model stats to summary.json

Metrics:
  SI-SNR  scale-invariant SDR (PIT-aligned best permutation)
  SI-SNRi SI-SNR improvement over input mixture
  STOI    short-time objective intelligibility (0..1, extended=False)
  PESQ    perceptual quality, wideband mode 'wb' (needs 16 kHz)
STOI/PESQ are NOT permutation-invariant -> computed on PIT-aligned estimates.
Requires: pip install pystoi pesq

Model name parsing:
  prefix "2speaker-" | "3speaker-"        → num_spk
  contains "skim-attention" (any suffix)  → arch = "v3"
  else                                    → arch = "skim"

Config hardcoded to paper-faith (K=150 dropout=0.1).

Usage:
  python inference/eval/run_eval.py
  python inference/eval/run_eval.py --models 2speaker-skim 3speaker-skim
  AUDIO_LIMIT=100 python inference/eval/run_eval.py   # override save-audio count
"""
import argparse
import gc
import json
import os
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
from pystoi import stoi as _stoi
from pesq import pesq as _pesq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from espnet2.enh.encoder.conv_encoder import ConvEncoder  # noqa: E402
from espnet2.enh.decoder.conv_decoder import ConvDecoder  # noqa: E402
from implementation.skim.skim_separator import SkiMSeparator  # noqa: E402
from implementation.skim_attention_v3.skim_attention_v3_separator import (  # noqa: E402
    SkiMAttentionV3Separator,
)

SAMPLE_RATE = 16000
TARGET_LEN = SAMPLE_RATE * 5
EPS = 1e-8

EVAL_DIR = Path(__file__).resolve().parent
MODEL_LIST = EVAL_DIR / "best-model-list.txt"
CKPT_CACHE = EVAL_DIR / "ckpts"
RESULTS_DIR = EVAL_DIR / "results"
AUDIO_LIMIT_DEFAULT = 450

TEST_ROOT = {
    2: ROOT / "dataset/synthetic/TITML-2spk-v2/test",
    3: ROOT / "dataset/synthetic/TITML-3spk-v2/test",
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Paper-faith config (K=150, dropout=0.1, layer=4, unit=256)
def build_config(num_spk: int, arch: str) -> dict:
    cfg = {
        "encoder": {"channel": 256, "kernel_size": 16, "stride": 8},
        "decoder": {"channel": 256, "kernel_size": 16, "stride": 8},
        "separator": {
            "input_dim": 256,
            "causal": False,
            "num_spk": num_spk,
            "layer": 4,
            "unit": 256,
            "segment_size": 150,
            "dropout": 0.1,
            "mem_type": "hc",
            "seg_overlap": False,
            "nonlinear": "relu",
        },
    }
    if arch == "v3":
        cfg["separator"]["num_heads"] = 4
    return cfg


def parse_model_name(name: str) -> tuple[int, str]:
    if name.startswith("2speaker"):
        num_spk = 2
    elif name.startswith("3speaker"):
        num_spk = 3
    else:
        raise ValueError(f"cannot parse num_spk from {name}")
    arch = "v3" if "skim-attention" in name else "skim"
    return num_spk, arch


def read_model_list() -> list[tuple[str, str]]:
    out = []
    for line in MODEL_LIST.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, gid = line.split("=", 1)
        out.append((name.strip(), gid.strip()))
    return out


def ensure_checkpoint(name: str, gdrive_id: str) -> Path:
    target = CKPT_CACHE / name / "best_model.pth"
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    import gdown
    url = f"https://drive.google.com/uc?id={gdrive_id}"
    print(f"[download] {name} ← {url}")
    gdown.download(url, str(target), quiet=False)
    if not target.exists():
        raise RuntimeError(f"download failed for {name}")
    return target


def load_wav(p: Path) -> np.ndarray:
    a, sr = sf.read(p)
    assert sr == SAMPLE_RATE, f"{p}: sr={sr}"
    a = a.astype(np.float32)
    if len(a) > TARGET_LEN:
        return a[:TARGET_LEN]
    if len(a) < TARGET_LEN:
        return np.pad(a, (0, TARGET_LEN - len(a)))
    return a


def si_snr(est: np.ndarray, ref: np.ndarray) -> float:
    est = est - est.mean()
    ref = ref - ref.mean()
    s_target = np.dot(est, ref) / (np.dot(ref, ref) + EPS) * ref
    e_noise = est - s_target
    return float(
        10 * np.log10(
            (np.dot(s_target, s_target) + EPS) / (np.dot(e_noise, e_noise) + EPS)
        )
    )


def best_pit(ests, refs) -> tuple[float, tuple[int, ...]]:
    """Return mean SI-SNR under best permutation + the permutation."""
    best_val = -1e9
    best_perm = None
    for perm in permutations(range(len(refs))):
        v = float(np.mean([si_snr(ests[perm[i]], refs[i]) for i in range(len(refs))]))
        if v > best_val:
            best_val = v
            best_perm = perm
    return best_val, best_perm


def stoi_score(ref: np.ndarray, est: np.ndarray) -> float:
    """STOI (0..1). ref=bersih, est=hasil pisah yang sudah diselaraskan PIT."""
    try:
        return float(_stoi(ref, est, SAMPLE_RATE, extended=False))
    except Exception:
        return float("nan")


def pesq_score(ref: np.ndarray, est: np.ndarray) -> float:
    """PESQ wideband (mode 'wb', butuh 16 kHz). nan jika PESQ gagal (sinyal diam/degenerasi)."""
    try:
        return float(_pesq(SAMPLE_RATE, ref, est, "wb"))
    except Exception:
        return float("nan")


def build_model(num_spk: int, arch: str, ckpt_path: Path):
    cfg = build_config(num_spk, arch)
    enc = ConvEncoder(**cfg["encoder"])
    dec = ConvDecoder(**cfg["decoder"])
    if arch == "skim":
        sep = SkiMSeparator(**cfg["separator"])
    elif arch == "v3":
        sep = SkiMAttentionV3Separator(**cfg["separator"])
    else:
        raise ValueError(arch)

    enc, sep, dec = enc.to(device).eval(), sep.to(device).eval(), dec.to(device).eval()
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = state.get("model_state_dict", state)

    def split_sd(prefix):
        return {k.replace(f"{prefix}.", "", 1): v for k, v in sd.items() if k.startswith(f"{prefix}.")}

    enc.load_state_dict(split_sd("encoder"))
    sep.load_state_dict(split_sd("separator"))
    dec.load_state_dict(split_sd("decoder"))
    return enc, sep, dec


@torch.no_grad()
def separate(enc, sep, dec, mix_np: np.ndarray) -> list[np.ndarray]:
    mix = torch.from_numpy(mix_np).unsqueeze(0).to(device)
    lengths = torch.tensor([mix.size(1)], dtype=torch.long, device=device)
    feats, flens = enc(mix, lengths)
    masked, _, _ = sep(feats, flens)
    out = []
    for m in masked:
        wav, _ = dec(m, lengths)
        out.append(wav.squeeze(0).cpu().numpy().astype(np.float32))
    return out


def save_audio_set(out_dir: Path, fid: str, mix_np, refs, ests, perm):
    """Save mixture + reordered estimates (aligned to GT) + GTs into out_dir/<fid>/."""
    d = out_dir / fid
    d.mkdir(parents=True, exist_ok=True)
    sf.write(d / "mixture.wav", mix_np, SAMPLE_RATE)
    for i, ref in enumerate(refs, start=1):
        sf.write(d / f"s{i}_gt.wav", ref, SAMPLE_RATE)
    for i in range(len(refs)):
        est_aligned = ests[perm[i]]
        if len(est_aligned) > TARGET_LEN:
            est_aligned = est_aligned[:TARGET_LEN]
        elif len(est_aligned) < TARGET_LEN:
            est_aligned = np.pad(est_aligned, (0, TARGET_LEN - len(est_aligned)))
        sf.write(d / f"s{i+1}_est.wav", est_aligned, SAMPLE_RATE)


def normalize_len(x):
    if len(x) > TARGET_LEN:
        return x[:TARGET_LEN]
    if len(x) < TARGET_LEN:
        return np.pad(x, (0, TARGET_LEN - len(x)))
    return x


def eval_model(name: str, gdrive_id: str, audio_limit: int) -> dict:
    num_spk, arch = parse_model_name(name)
    test_root = TEST_ROOT[num_spk]
    if not (test_root / "mix").exists():
        raise FileNotFoundError(f"test dataset not found: {test_root}")

    ckpt_path = ensure_checkpoint(name, gdrive_id)
    enc, sep, dec = build_model(num_spk, arch, ckpt_path)

    out_dir = RESULTS_DIR / name
    audio_dir = out_dir / "audio"
    out_dir.mkdir(parents=True, exist_ok=True)
    audio_dir.mkdir(parents=True, exist_ok=True)

    mix_files = sorted((test_root / "mix").glob("*.wav"))
    print(f"\n=== {name} (num_spk={num_spk}, arch={arch}, test={len(mix_files)}) ===")

    csv_path = out_dir / "per_file.csv"
    csv_f = csv_path.open("w", buffering=1)
    csv_f.write("file_id,sisnr,sisnri,mix_sisnr,stoi,pesq,perm\n")

    sisnrs, sisnris, stois, pesqs = [], [], [], []
    for idx, mp in enumerate(tqdm(mix_files, desc=name)):
        fid = mp.stem
        mix_np = load_wav(mp)
        refs = [load_wav(test_root / f"s{i}" / f"{fid}.wav") for i in range(1, num_spk + 1)]
        ests = [normalize_len(e) for e in separate(enc, sep, dec, mix_np)]

        sep_si, perm = best_pit(ests, refs)
        mix_si = float(np.mean([si_snr(mix_np, r) for r in refs]))
        sisnri = sep_si - mix_si

        # Selaraskan estimasi ke ref pakai permutasi PIT terbaik (est utk ref i = ests[perm[i]]).
        # STOI/PESQ tak permutation-invariant, jadi WAJIB pakai estimasi yang sudah diselaraskan.
        aligned = [ests[perm[i]] for i in range(num_spk)]
        stoi_m = float(np.nanmean([stoi_score(refs[i], aligned[i]) for i in range(num_spk)]))
        pesq_m = float(np.nanmean([pesq_score(refs[i], aligned[i]) for i in range(num_spk)]))

        sisnrs.append(sep_si)
        sisnris.append(sisnri)
        stois.append(stoi_m)
        pesqs.append(pesq_m)
        csv_f.write(
            f"{fid},{sep_si:.4f},{sisnri:.4f},{mix_si:.4f},"
            f"{stoi_m:.4f},{pesq_m:.4f},{'-'.join(map(str, perm))}\n"
        )

        if idx < audio_limit:
            save_audio_set(audio_dir, fid, mix_np, refs, ests, perm)

    csv_f.close()

    arr_si = np.array(sisnrs)
    arr_sii = np.array(sisnris)
    arr_stoi = np.array(stois)
    arr_pesq = np.array(pesqs)
    stats = {
        "model": name,
        "num_spk": num_spk,
        "arch": arch,
        "n_eval": int(len(arr_si)),
        "n_audio_saved": int(min(audio_limit, len(mix_files))),
        "mean_sisnr": float(arr_si.mean()),
        "std_sisnr": float(arr_si.std()),
        "median_sisnr": float(np.median(arr_si)),
        "mean_sisnri": float(arr_sii.mean()),
        "std_sisnri": float(arr_sii.std()),
        "mean_stoi": float(np.nanmean(arr_stoi)),
        "std_stoi": float(np.nanstd(arr_stoi)),
        "mean_pesq": float(np.nanmean(arr_pesq)),
        "std_pesq": float(np.nanstd(arr_pesq)),
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"  mean SI-SNR : {stats['mean_sisnr']:.3f} ± {stats['std_sisnr']:.3f} dB")
    print(f"  mean SI-SNRi: {stats['mean_sisnri']:.3f} ± {stats['std_sisnri']:.3f} dB")
    print(f"  mean STOI   : {stats['mean_stoi']:.4f} ± {stats['std_stoi']:.4f}")
    print(f"  mean PESQ   : {stats['mean_pesq']:.4f} ± {stats['std_pesq']:.4f}")

    del enc, sep, dec
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="*", default=None, help="subset of model names to eval")
    p.add_argument("--audio-limit", type=int, default=int(os.environ.get("AUDIO_LIMIT", AUDIO_LIMIT_DEFAULT)))
    args = p.parse_args()

    print(f"device: {device}")
    print(f"audio_limit: {args.audio_limit}")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_CACHE.mkdir(parents=True, exist_ok=True)

    entries = read_model_list()
    if args.models:
        wanted = set(args.models)
        entries = [(n, g) for n, g in entries if n in wanted]
        if not entries:
            sys.exit(f"no matching models in list: {args.models}")

    summary = {}
    summary_path = RESULTS_DIR / "summary.json"
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text())
        except Exception:
            summary = {}

    for name, gid in entries:
        try:
            stats = eval_model(name, gid, args.audio_limit)
            summary[name] = stats
            summary_path.write_text(json.dumps(summary, indent=2))
        except Exception as e:
            print(f"[FAIL] {name}: {e}")
            summary[name] = {"error": str(e)}
            summary_path.write_text(json.dumps(summary, indent=2))

    print("\n=== FINAL SUMMARY ===")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
