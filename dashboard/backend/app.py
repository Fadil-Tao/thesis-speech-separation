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
            "dropout": 0.1,
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
            "dropout": 0.1,
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
            "dropout": 0.1,
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
            "dropout": 0.1,
            "mem_type": "hc",
            "seg_overlap": False,
            "num_heads": 4,
        },
    },
}

# Global model cache
loaded_models = {}


def build_model(model_key, device):
    """Build and load model."""
    config = MODEL_CONFIGS[model_key]

    # Encoder
    encoder = ConvEncoder(channel=256, kernel_size=32, stride=16)

    # Separator
    separator = config["separator_class"](**config["config"])

    # Decoder
    decoder = ConvDecoder(channel=256, kernel_size=32, stride=16)

    # Loss
    criterion = SISNRLoss()
    pit_wrapper = PITSolver(criterion=criterion)

    # Full model
    model = ESPnetEnhancementModel(
        encoder=encoder,
        separator=separator,
        decoder=decoder,
        mask_module=None,
        loss_wrappers=[pit_wrapper],
        loss_type="si_snr",
    )

    # Load checkpoint
    checkpoint_path = config["path"]
    if checkpoint_path.exists():
        try:
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            print(f"Loaded checkpoint from {checkpoint_path}")
        except Exception as e:
            print(f"Warning: Could not load checkpoint from {checkpoint_path}: {e}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")

    model = model.to(device)
    model.eval()

    return model, config["num_spk"]


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
    """List available datasets."""
    datasets = []
    dataset_dir = project_root / "dataset" / "synthetic"

    for dataset_name in ["TITML-2spk", "TITML-3spk"]:
        dataset_path = dataset_dir / dataset_name
        if dataset_path.exists():
            # Count samples
            splits = {}
            for split in ["train", "dev", "test"]:
                split_path = dataset_path / split / "mix"
                if split_path.exists():
                    count = len(list(split_path.glob("*.wav")))
                    splits[split] = count

            # Load metadata if exists
            info_path = dataset_path / "dataset_info.json"
            info = {}
            if info_path.exists():
                with open(info_path) as f:
                    info = json.load(f)

            datasets.append(
                {
                    "id": dataset_name.lower(),
                    "name": dataset_name,
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

    dataset_name = "TITML-2spk" if "2spk" in dataset_id else "TITML-3spk"
    dataset_path = project_root / "dataset" / "synthetic" / dataset_name / split

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

        # Inference using ESPnet model forward
        with torch.no_grad():
            # Prepare reference dict for PIT
            ref_dict = {}
            dataset_path = Path(audio_path).parent.parent
            for i in range(1, num_spk + 1):
                gt_path = dataset_path / f"s{i}" / Path(audio_path).name
                if gt_path.exists():
                    gt_audio, _ = sf.read(gt_path)
                    if len(gt_audio.shape) > 1:
                        gt_audio = np.mean(gt_audio, axis=1)
                    ref_dict[f"speech_ref{i}"] = (
                        torch.FloatTensor(gt_audio).unsqueeze(0).to(device)
                    )
                    ref_dict[f"speech_ref{i}_lengths"] = torch.LongTensor(
                        [len(gt_audio)]
                    ).to(device)

            # Forward pass through full model
            outputs = model.forward(
                speech_mix=audio_tensor, speech_mix_lengths=lengths, **ref_dict
            )

            # Get separated speech from stats
            separated = []
            if hasattr(model, "stats"):
                for i in range(1, num_spk + 1):
                    key = f"wav_spk{i}"
                    if key in model.stats:
                        sep_audio = model.stats[key].squeeze(0).cpu().numpy()
                        separated.append(sep_audio)

            # Fallback: if no separated audio in stats, try to get from model output
            if len(separated) == 0:
                # Use forward_enhance which properly handles the full pipeline
                speech_pre, _, _, _ = model.forward_enhance(audio_tensor, lengths)
                for i, sep in enumerate(speech_pre[:num_spk]):
                    separated.append(sep.squeeze(0).cpu().numpy())

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

        # Calculate metrics if ground truth available
        metrics = {}
        dataset_path = Path(audio_path).parent.parent
        for i in range(1, num_spk + 1):
            gt_path = dataset_path / f"s{i}" / Path(audio_path).name
            if gt_path.exists():
                gt_audio, _ = sf.read(gt_path)
                sep_audio = separated[i - 1]

                si_snr = calculate_si_snr(sep_audio, gt_audio)
                stoi_score = calculate_stoi(sep_audio, gt_audio, sr)

                metrics[f"spk{i}"] = {
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

        # Get samples
        dataset_name = "TITML-2spk" if "2spk" in dataset_id else "TITML-3spk"
        dataset_path = project_root / "dataset" / "synthetic" / dataset_name / split
        mix_files = sorted(list((dataset_path / "mix").glob("*.wav")))[:num_samples]

        results = []
        total_metrics = {"si_snr": [], "stoi": []}

        for mix_file in mix_files:
            file_id = mix_file.stem

            # Load audio
            audio, sr = sf.read(mix_file)
            if len(audio.shape) > 1:
                audio = np.mean(audio.shape, axis=1)

            # Convert to tensor
            audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(device)
            lengths = torch.LongTensor([len(audio)]).to(device)

            # Inference using ESPnet model forward
            with torch.no_grad():
                # Prepare reference dict for PIT
                ref_dict = {}
                for i in range(1, num_spk + 1):
                    gt_path = dataset_path / f"s{i}" / f"{file_id}.wav"
                    if gt_path.exists():
                        gt_audio, _ = sf.read(gt_path)
                        if len(gt_audio.shape) > 1:
                            gt_audio = np.mean(gt_audio, axis=1)
                        ref_dict[f"speech_ref{i}"] = (
                            torch.FloatTensor(gt_audio).unsqueeze(0).to(device)
                        )
                        ref_dict[f"speech_ref{i}_lengths"] = torch.LongTensor(
                            [len(gt_audio)]
                        ).to(device)

                # Forward pass through full model
                outputs = model.forward(
                    speech_mix=audio_tensor, speech_mix_lengths=lengths, **ref_dict
                )

                # Get separated speech from stats
                separated = []
                if hasattr(model, "stats"):
                    for i in range(1, num_spk + 1):
                        key = f"wav_spk{i}"
                        if key in model.stats:
                            sep_audio = model.stats[key].squeeze(0).cpu().numpy()
                            separated.append(sep_audio)

                # Fallback: if no separated audio in stats, try to get from model output
                if len(separated) == 0:
                    # Use forward_enhance which properly handles the full pipeline
                    speech_pre, _, _, _ = model.forward_enhance(audio_tensor, lengths)
                    for i, sep in enumerate(speech_pre[:num_spk]):
                        separated.append(sep.squeeze(0).cpu().numpy())

            # Calculate metrics
            sample_metrics = {"file_id": file_id, "speakers": {}}
            for i in range(1, num_spk + 1):
                gt_path = dataset_path / f"s{i}" / f"{file_id}.wav"
                if gt_path.exists():
                    gt_audio, _ = sf.read(gt_path)
                    sep_audio = separated[i - 1]

                    si_snr = calculate_si_snr(sep_audio, gt_audio)
                    stoi_score = calculate_stoi(sep_audio, gt_audio, sr)

                    sample_metrics["speakers"][f"spk{i}"] = {
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
