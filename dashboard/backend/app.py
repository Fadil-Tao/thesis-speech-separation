# -*- coding: utf-8 -*-
"""
Dashboard Backend - Flask API for Speech Separation Model Evaluation
"""

import os
import sys
import json
import torch
import numpy as np
import soundfile as sf
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from werkzeug.utils import secure_filename
import tempfile

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# ESPnet imports
from espnet2.enh.encoder.conv_encoder import ConvEncoder
from espnet2.enh.decoder.conv_decoder import ConvDecoder
from espnet2.enh.espnet_model import ESPnetEnhancementModel
from espnet2.enh.loss.criterions.time_domain import SISNRLoss
from espnet2.enh.loss.wrappers.pit_solver import PITSolver

# Import separators
from implementation.skim_attention.skim_attention_separator import (
    SkiMAttentionSeparator,
)
from implementation.skim.skim_separator import SkiMSeparator

app = Flask(__name__)
CORS(app)

# Configuration
UPLOAD_FOLDER = Path(tempfile.gettempdir()) / "dashboard_uploads"
UPLOAD_FOLDER.mkdir(exist_ok=True)
app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB max

# Model registry
MODEL_CONFIGS = {
    "skim-2spk": {
        "path": project_root / "checkpoints" / "2speaker" / "skim" / "best_model.pth",
        "num_spk": 2,
        "separator_class": SkiMSeparator,
        "config": {
            "input_dim": 256,
            "causal": False,
            "num_spk": 2,
            "predict_noise": False,
            "nonlinear": "relu",
            "layer": 4,
            "unit": 256,
            "segment_size": 20,
            "dropout": 0.2,
            "mem_type": "hc",
            "seg_overlap": False,
        },
    },
    "skim-attention-2spk": {
        "path": project_root
        / "checkpoints"
        / "2speaker"
        / "skim-attention"
        / "best_model.pth",
        "num_spk": 2,
        "separator_class": SkiMAttentionSeparator,
        "config": {
            "input_dim": 256,
            "causal": False,
            "num_spk": 2,
            "predict_noise": False,
            "nonlinear": "relu",
            "layer": 4,
            "unit": 256,
            "segment_size": 20,
            "dropout": 0.2,
            "mem_type": "hc",
            "seg_overlap": False,
            "num_heads": 4,
        },
    },
    "skim-3spk": {
        "path": project_root / "checkpoints" / "3speaker" / "skim" / "best_model.pth",
        "num_spk": 3,
        "separator_class": SkiMSeparator,
        "config": {
            "input_dim": 256,
            "causal": False,
            "num_spk": 3,
            "predict_noise": False,
            "nonlinear": "relu",
            "layer": 4,
            "unit": 256,
            "segment_size": 20,
            "dropout": 0.2,
            "mem_type": "hc",
            "seg_overlap": False,
        },
    },
    "skim-attention-3spk": {
        "path": project_root
        / "checkpoints"
        / "3speaker"
        / "skim-attention"
        / "best_model.pth",
        "num_spk": 3,
        "separator_class": SkiMAttentionSeparator,
        "config": {
            "input_dim": 256,
            "causal": False,
            "num_spk": 3,
            "predict_noise": False,
            "nonlinear": "relu",
            "layer": 4,
            "unit": 256,
            "segment_size": 20,
            "dropout": 0.2,
            "mem_type": "hc",
            "seg_overlap": False,
            "num_heads": 4,
        },
    },
}

# Global model cache
loaded_models = {}


def build_model(model_key, device):
    """Build and load model, deriving num_spk from the checkpoint config."""
    config = MODEL_CONFIGS[model_key]
    checkpoint_path = config["path"]

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Infer actual num_spk from the output layer weight shape
    # output_fc.1.weight shape is [num_spk * input_dim, input_dim, 1]
    state = checkpoint["model_state_dict"]
    out_key = "separator.skim.output_fc.1.weight"
    if out_key in state:
        input_dim = config["config"]["input_dim"]
        num_spk = state[out_key].shape[0] // input_dim
        print(f"Inferred num_spk={num_spk} from checkpoint weights")
    else:
        num_spk = config["num_spk"]
        print(f"Using configured num_spk={num_spk}")

    sep_kwargs = {**config["config"], "num_spk": num_spk}

    encoder = ConvEncoder(channel=256, kernel_size=32, stride=16)
    separator = config["separator_class"](**sep_kwargs)
    decoder = ConvDecoder(channel=256, kernel_size=32, stride=16)

    criterion = SISNRLoss()
    pit_wrapper = PITSolver(criterion=criterion)

    model = ESPnetEnhancementModel(
        encoder=encoder,
        separator=separator,
        decoder=decoder,
        mask_module=None,
        loss_wrappers=[pit_wrapper],
        loss_type="si_snr",
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from {checkpoint_path} (num_spk={num_spk})")

    model = model.to(device)
    model.eval()

    return model, num_spk


def get_model(model_key):
    """Get or load model."""
    if model_key not in loaded_models:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, num_spk = build_model(model_key, device)
        loaded_models[model_key] = {
            "model": model,
            "num_spk": num_spk,
            "device": device,
        }
    return loaded_models[model_key]


def normalize_audio(separated, mixture):
    """Scale each separated signal so its RMS matches the mixture's RMS."""
    mix_rms = np.sqrt(np.mean(mixture ** 2) + 1e-10)
    out = []
    for s in separated:
        s_rms = np.sqrt(np.mean(s ** 2) + 1e-10)
        out.append(s * (mix_rms / s_rms))
    return out


def calculate_si_snr(estimate, reference):
    """Calculate SI-SNR in dB."""
    # Ensure same length
    min_len = min(len(estimate), len(reference))
    estimate = estimate[:min_len]
    reference = reference[:min_len]

    # Remove mean
    estimate = estimate - np.mean(estimate)
    reference = reference - np.mean(reference)

    # Target
    target = np.sum(estimate * reference) / (np.sum(reference**2) + 1e-10) * reference

    # Noise
    noise = estimate - target

    # SI-SNR
    si_snr = 10 * np.log10(np.sum(target**2) / (np.sum(noise**2) + 1e-10) + 1e-10)

    return si_snr


def calculate_stoi(estimate, reference, fs=16000):
    """Calculate STOI (Short-Time Objective Intelligibility)."""
    try:
        from pystoi import stoi

        # Ensure same length
        min_len = min(len(estimate), len(reference))
        estimate = estimate[:min_len]
        reference = reference[:min_len]
        return stoi(reference, estimate, fs, extended=False)
    except ImportError:
        # Fallback if pystoi not installed
        return None


# API Routes
@app.route("/api/models", methods=["GET"])
def list_models():
    """List available models."""
    models = []
    for key, config in MODEL_CONFIGS.items():
        exists = config["path"].exists()
        models.append(
            {
                "id": key,
                "name": key.replace("-", " ").title(),
                "num_speakers": config["num_spk"],
                "exists": exists,
                "path": str(config["path"]),
            }
        )
    return jsonify({"models": models})


@app.route("/api/datasets", methods=["GET"])
def list_datasets():
    """List available datasets (auto-discovers all TITML-* dirs)."""
    datasets = []
    dataset_dir = project_root / "dataset" / "synthetic"

    if not dataset_dir.exists():
        return jsonify({"datasets": []})

    for dataset_path in sorted(dataset_dir.iterdir()):
        if not dataset_path.is_dir() or not dataset_path.name.startswith("TITML-"):
            continue

        splits = {}
        for split in ["train", "dev", "test"]:
            split_path = dataset_path / split / "mix"
            if split_path.exists():
                splits[split] = len(list(split_path.glob("*.wav")))

        # Load metadata (try metadata.json first, then dataset_info.json)
        info = {}
        for info_name in ["metadata.json", "train/metadata.json", "dataset_info.json"]:
            info_path = dataset_path / info_name
            if info_path.exists():
                with open(info_path) as f:
                    info = json.load(f)
                break

        datasets.append(
            {
                "id": dataset_path.name.lower(),
                "name": dataset_path.name,
                "splits": splits,
                "info": info,
            }
        )

    return jsonify({"datasets": datasets})


@app.route("/api/dataset/<dataset_id>/samples", methods=["GET"])
def get_dataset_samples(dataset_id):
    """Get samples from dataset."""
    split = request.args.get("split", "test")
    limit = int(request.args.get("limit", 20))

    # Resolve dataset path: dataset_id is the lowercase name e.g. "titml-2spk-v2"
    dataset_dir = project_root / "dataset" / "synthetic"
    dataset_name = next(
        (d.name for d in dataset_dir.iterdir()
         if d.is_dir() and d.name.lower() == dataset_id),
        None,
    )
    if dataset_name is None:
        return jsonify({"error": f"Dataset '{dataset_id}' not found"}), 404
    dataset_path = dataset_dir / dataset_name / split

    samples = []
    mix_files = sorted(list((dataset_path / "mix").glob("*.wav")))[:limit]

    for mix_file in mix_files:
        file_id = mix_file.stem
        sample = {"id": file_id, "mix": str(mix_file)}

        # Check for sources
        for i in range(1, 4):
            src_path = dataset_path / f"s{i}" / f"{file_id}.wav"
            if src_path.exists():
                sample[f"s{i}"] = str(src_path)

        samples.append(sample)

    return jsonify({"samples": samples, "split": split, "total": len(mix_files)})


@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    """Evaluate audio with selected model."""
    data = request.json
    model_key = data.get("model")
    audio_path = data.get("audio_path")

    if not model_key or not audio_path:
        return jsonify({"error": "Missing model or audio_path"}), 400

    try:
        # Load model
        model_info = get_model(model_key)
        model = model_info["model"]
        num_spk = model_info["num_spk"]
        device = model_info["device"]

        # Load audio
        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Convert to tensor
        audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(device)
        lengths = torch.LongTensor([len(audio)]).to(device)

        # Separate using forward_enhance (no PIT reordering)
        with torch.no_grad():
            speech_pre, _, _, _ = model.forward_enhance(audio_tensor, lengths)
            separated = [s.squeeze(0).cpu().numpy() for s in speech_pre[:num_spk]]

        # Normalize output RMS to match input (relu masks are unbounded)
        separated = normalize_audio(separated, audio)

        # Save separated audio
        results = []
        for i, sep_audio in enumerate(separated):
            output_path = (
                UPLOAD_FOLDER / f"separated_spk{i + 1}_{Path(audio_path).stem}.wav"
            )
            sf.write(output_path, sep_audio, sr)
            results.append(
                {
                    "speaker": i + 1,
                    "path": str(output_path),
                    "url": f"/api/audio/{output_path.name}",
                }
            )

        # Calculate metrics with best-permutation matching
        metrics = {}
        dataset_path = Path(audio_path).parent.parent
        refs = []
        for i in range(1, num_spk + 1):
            gt_path = dataset_path / f"s{i}" / Path(audio_path).name
            if gt_path.exists():
                gt_audio, _ = sf.read(gt_path)
                if len(gt_audio.shape) > 1:
                    gt_audio = np.mean(gt_audio, axis=1)
                refs.append((i, gt_audio))

        if len(refs) == num_spk:
            from itertools import permutations
            ref_audios = [r[1] for r in refs]
            best_perm = max(
                permutations(range(num_spk)),
                key=lambda p: sum(calculate_si_snr(separated[p[j]], ref_audios[j]) for j in range(num_spk))
            )
            for j in range(num_spk):
                sep_audio = separated[best_perm[j]]
                gt_audio = ref_audios[j]
                si_snr = calculate_si_snr(sep_audio, gt_audio)
                stoi_score = calculate_stoi(sep_audio, gt_audio, sr)
                metrics[f"spk{j + 1}"] = {
                    "si_snr": round(float(si_snr), 2),
                    "stoi": round(float(stoi_score), 3) if stoi_score else None,
                }

        return jsonify(
            {
                "success": True,
                "separated": results,
                "metrics": metrics,
                "num_speakers": num_spk,
            }
        )

    except Exception as e:
        import traceback

        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/audio/<filename>")
def serve_audio(filename):
    """Serve audio file."""
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


@app.route("/api/audio/dataset/<path:filepath>")
def serve_dataset_audio(filepath):
    """Serve dataset audio file."""
    full_path = project_root / filepath
    if full_path.exists():
        return send_from_directory(full_path.parent, full_path.name)
    return jsonify({"error": "File not found"}), 404


@app.route("/api/evaluate-batch", methods=["POST"])
def evaluate_batch():
    """Evaluate multiple samples from dataset automatically."""
    data = request.json
    model_key = data.get("model")
    dataset_id = data.get("dataset")
    split = data.get("split", "test")
    num_samples = int(data.get("num_samples", 10))

    if not model_key or not dataset_id:
        return jsonify({"error": "Missing model or dataset"}), 400

    try:
        # Load model
        model_info = get_model(model_key)
        model = model_info["model"]
        num_spk = model_info["num_spk"]
        device = model_info["device"]

        # Resolve dataset path
        dataset_dir = project_root / "dataset" / "synthetic"
        dataset_name = next(
            (d.name for d in dataset_dir.iterdir()
             if d.is_dir() and d.name.lower() == dataset_id),
            None,
        )
        if dataset_name is None:
            return jsonify({"error": f"Dataset '{dataset_id}' not found"}), 404
        dataset_path = dataset_dir / dataset_name / split
        mix_files = sorted(list((dataset_path / "mix").glob("*.wav")))[:num_samples]

        results = []
        total_metrics = {"si_snr": [], "stoi": []}

        for mix_file in mix_files:
            file_id = mix_file.stem

            # Load audio
            audio, sr = sf.read(mix_file)
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=1)

            # Convert to tensor
            audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(device)
            lengths = torch.LongTensor([len(audio)]).to(device)

            # Separate using forward_enhance (no PIT reordering)
            with torch.no_grad():
                speech_pre, _, _, _ = model.forward_enhance(audio_tensor, lengths)
                separated = [s.squeeze(0).cpu().numpy() for s in speech_pre[:num_spk]]

            # Normalize output RMS to match input (relu masks are unbounded)
            separated = normalize_audio(separated, audio)

            # Load ground-truth references
            refs = []
            for i in range(1, num_spk + 1):
                gt_path = dataset_path / f"s{i}" / f"{file_id}.wav"
                if gt_path.exists():
                    gt_audio, _ = sf.read(gt_path)
                    if len(gt_audio.shape) > 1:
                        gt_audio = np.mean(gt_audio, axis=1)
                    refs.append(gt_audio)

            # Best-permutation SI-SNR (mirrors PIT so scores are fair)
            sample_metrics = {"file_id": file_id, "speakers": {}}
            if len(refs) == num_spk and len(separated) == num_spk:
                from itertools import permutations
                best_perm = None
                best_sum = -1e9
                for perm in permutations(range(num_spk)):
                    s = sum(calculate_si_snr(separated[perm[j]], refs[j]) for j in range(num_spk))
                    if s > best_sum:
                        best_sum = s
                        best_perm = perm

                for j in range(num_spk):
                    sep_audio = separated[best_perm[j]]
                    gt_audio = refs[j]
                    si_snr = calculate_si_snr(sep_audio, gt_audio)
                    stoi_score = calculate_stoi(sep_audio, gt_audio, sr)
                    sample_metrics["speakers"][f"spk{j + 1}"] = {
                        "si_snr": round(float(si_snr), 2),
                        "stoi": round(float(stoi_score), 3) if stoi_score else None,
                    }
                    total_metrics["si_snr"].append(si_snr)
                    if stoi_score:
                        total_metrics["stoi"].append(stoi_score)

            results.append(sample_metrics)

        # Calculate averages
        avg_metrics = {}
        if total_metrics["si_snr"]:
            avg_metrics["avg_si_snr"] = round(
                float(np.mean(total_metrics["si_snr"])), 2
            )
            avg_metrics["std_si_snr"] = round(float(np.std(total_metrics["si_snr"])), 2)
        if total_metrics["stoi"]:
            avg_metrics["avg_stoi"] = round(float(np.mean(total_metrics["stoi"])), 3)
            avg_metrics["std_stoi"] = round(float(np.std(total_metrics["stoi"])), 3)

        return jsonify(
            {
                "success": True,
                "model": model_key,
                "dataset": dataset_id,
                "split": split,
                "num_evaluated": len(results),
                "results": results,
                "average_metrics": avg_metrics,
            }
        )

    except Exception as e:
        import traceback

        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/training", methods=["GET"])
def get_training_info():
    """Return training config and loss history for all checkpoint dirs."""
    checkpoints_root = project_root / "checkpoints"
    results = []

    for model_key, cfg in MODEL_CONFIGS.items():
        ckpt_dir = cfg["path"].parent
        entry = {
            "model": model_key,
            "checkpoint_dir": str(ckpt_dir),
            "best_model_exists": cfg["path"].exists(),
            "config": None,
            "best_epoch": None,
            "best_val_si_snr": None,
            "checkpoints": [],
        }

        # Load config.json saved by training script
        config_path = ckpt_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                saved = json.load(f)
            entry["config"] = saved.get("train_config")
            entry["best_val_si_snr"] = saved.get("best_si_snr")

        # Best model metadata
        if cfg["path"].exists():
            try:
                ckpt = torch.load(cfg["path"], map_location="cpu")
                entry["best_epoch"] = ckpt.get("epoch")
                if entry["best_val_si_snr"] is None:
                    val_loss = ckpt.get("val_loss")
                    if val_loss is not None:
                        entry["best_val_si_snr"] = round(-float(val_loss), 2)
            except Exception:
                pass

        # List periodic checkpoints
        if ckpt_dir.exists():
            entry["checkpoints"] = sorted(
                p.name for p in ckpt_dir.glob("checkpoint_epoch_*.pth")
            )

        results.append(entry)

    return jsonify({"training": results})


@app.route("/api/training/<model_key>/image")
def get_training_image(model_key):
    """Serve training_curves.png for a model."""
    if model_key not in MODEL_CONFIGS:
        return jsonify({"error": "Unknown model"}), 404
    ckpt_dir = MODEL_CONFIGS[model_key]["path"].parent
    img_path = ckpt_dir / "training_curves.png"
    if not img_path.exists():
        return jsonify({"error": "No training curves image found"}), 404
    return send_from_directory(str(ckpt_dir), "training_curves.png")


@app.route("/api/waveform", methods=["POST"])
def get_waveform():
    """Get waveform data for visualization."""
    data = request.json
    audio_path = data.get("audio_path")
    points = int(data.get("points", 1000))

    if not audio_path:
        return jsonify({"error": "Missing audio_path"}), 400

    try:
        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        # Downsample for visualization
        if len(audio) > points:
            indices = np.linspace(0, len(audio) - 1, points, dtype=int)
            audio = audio[indices]

        # Normalize
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val

        duration = len(audio) / sr

        return jsonify(
            {
                "waveform": audio.tolist(),
                "sample_rate": sr,
                "duration": duration,
                "points": len(audio),
            }
        )

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/")
def index():
    """Serve the dashboard."""
    return send_from_directory("../frontend", "index.html")


@app.route("/<path:path>")
def serve_static(path):
    """Serve static files."""
    return send_from_directory("../frontend", path)


if __name__ == "__main__":
    print("Starting Speech Separation Dashboard...")
    print(f"Project root: {project_root}")
    print(f"Upload folder: {UPLOAD_FOLDER}")
    app.run(host="0.0.0.0", port=5000, debug=True)
