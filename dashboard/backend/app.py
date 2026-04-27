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
    "skim-3spk-transfer": {
        "path": project_root / "checkpoints" / "3speaker" / "skim-transfer" / "best_model.pth",
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
    "skim-attention-3spk-transfer": {
        "path": project_root
        / "checkpoints"
        / "3speaker"
        / "skim-attention-transfer"
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

MODEL_DISPLAY_NAMES = {
    "skim-2spk": "SkiM 2-Spk",
    "skim-attention-2spk": "SkiM Attention 2-Spk",
    "skim-3spk": "SkiM 3-Spk",
    "skim-attention-3spk": "SkiM Attention 3-Spk",
    "skim-3spk-transfer": "SkiM 3-Spk (Transfer)",
    "skim-attention-3spk-transfer": "SkiM Attention 3-Spk (Transfer)",
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


def _run_evaluate(model_key, audio_path):
    """Shared evaluation logic used by /api/evaluate and /api/evaluate/random."""
    from itertools import permutations as _perms
    try:
        model_info = get_model(model_key)
        model = model_info["model"]
        num_spk = model_info["num_spk"]
        device = model_info["device"]

        audio, sr = sf.read(audio_path)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)

        audio_tensor = torch.FloatTensor(audio).unsqueeze(0).to(device)
        lengths = torch.LongTensor([len(audio)]).to(device)

        with torch.no_grad():
            speech_pre, _, _, _ = model.forward_enhance(audio_tensor, lengths)
            separated = [s.squeeze(0).cpu().numpy() for s in speech_pre[:num_spk]]

        separated = normalize_audio(separated, audio)

        results = []
        for i, sep_audio in enumerate(separated):
            output_path = UPLOAD_FOLDER / f"separated_spk{i + 1}_{Path(audio_path).stem}.wav"
            sf.write(output_path, sep_audio, sr)
            results.append({
                "speaker": i + 1,
                "path": str(output_path),
                "url": f"/api/audio/{output_path.name}",
            })

        metrics = {}
        dataset_path = Path(audio_path).parent.parent
        refs = []
        for i in range(1, num_spk + 1):
            gt_path = dataset_path / f"s{i}" / Path(audio_path).name
            if gt_path.exists():
                gt_audio, _ = sf.read(gt_path)
                if len(gt_audio.shape) > 1:
                    gt_audio = np.mean(gt_audio, axis=1)
                refs.append(gt_audio)

        if len(refs) == num_spk:
            best_perm = max(
                _perms(range(num_spk)),
                key=lambda p: sum(calculate_si_snr(separated[p[j]], refs[j]) for j in range(num_spk))
            )
            for j in range(num_spk):
                si_snr = calculate_si_snr(separated[best_perm[j]], refs[j])
                stoi_score = calculate_stoi(separated[best_perm[j]], refs[j], sr)
                metrics[f"spk{j + 1}"] = {
                    "si_snr": round(float(si_snr), 2),
                    "stoi": round(float(stoi_score), 3) if stoi_score else None,
                }

        return jsonify({
            "success": True,
            "sample_id": Path(audio_path).stem,
            "audio_path": audio_path,
            "separated": results,
            "metrics": metrics,
            "num_speakers": num_spk,
        })

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "traceback": traceback.format_exc()}), 500


@app.route("/api/evaluate", methods=["POST"])
def evaluate():
    """Evaluate audio with selected model."""
    data = request.json
    model_key = data.get("model")
    audio_path = data.get("audio_path")
    if not model_key or not audio_path:
        return jsonify({"error": "Missing model or audio_path"}), 400
    return _run_evaluate(model_key, audio_path)


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
    """Return training info keyed by model_key for the frontend training page."""
    results = {}

    for model_key, cfg in MODEL_CONFIGS.items():
        ckpt_dir = cfg["path"].parent
        entry = {
            "name": MODEL_DISPLAY_NAMES.get(model_key, model_key),
            "best_model_exists": cfg["path"].exists(),
            "best_epoch": None,
            "best_val_loss": None,
            "best_si_snr": None,
            "has_curves": (ckpt_dir / "training_curves.png").exists(),
            "checkpoints": [],
        }

        config_path = ckpt_dir / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                saved = json.load(f)
            entry["best_si_snr"] = saved.get("best_si_snr")
            if entry["best_si_snr"] is not None:
                entry["best_val_loss"] = -entry["best_si_snr"]

        if cfg["path"].exists():
            try:
                ckpt = torch.load(cfg["path"], map_location="cpu", weights_only=False)
                entry["best_epoch"] = ckpt.get("epoch")
                if entry["best_si_snr"] is None:
                    val_loss = ckpt.get("best_val_loss", ckpt.get("val_loss"))
                    if val_loss is not None:
                        entry["best_val_loss"] = round(float(val_loss), 4)
                        entry["best_si_snr"] = round(-float(val_loss), 2)
            except Exception:
                pass

        if ckpt_dir.exists():
            entry["checkpoints"] = sorted(
                p.name for p in ckpt_dir.glob("checkpoint_epoch_*.pth")
            )

        results[model_key] = entry

    return jsonify(results)


@app.route("/api/dataset/<dataset_id>/speakers", methods=["GET"])
def get_speakers(dataset_id):
    """Return unique speaker IDs found in a dataset split's metadata."""
    split = request.args.get("split", "test")
    dataset_dir = project_root / "dataset" / "synthetic"
    dataset_name = next(
        (d.name for d in dataset_dir.iterdir() if d.is_dir() and d.name.lower() == dataset_id),
        None,
    )
    if dataset_name is None:
        return jsonify({"error": f"Dataset '{dataset_id}' not found"}), 404

    metadata_path = dataset_dir / dataset_name / split / "metadata.json"
    if not metadata_path.exists():
        return jsonify({"speakers": []})

    with open(metadata_path) as f:
        meta = json.load(f)

    speakers = set()
    samples = meta if isinstance(meta, list) else meta.get("samples", [])
    for s in samples:
        for key in ("speakers", "speaker_ids", "spk_ids"):
            if key in s:
                for spk in s[key]:
                    speakers.add(spk)
                break

    return jsonify({"speakers": sorted(speakers)})


@app.route("/api/evaluate/random", methods=["POST"])
def evaluate_random():
    """Pick a random sample from a dataset split and evaluate it."""
    import random as _random
    data = request.json
    model_key = data.get("model")
    dataset_id = data.get("dataset")
    split = data.get("split", "test")
    speaker_filter = data.get("speaker")

    if not model_key or not dataset_id:
        return jsonify({"error": "Missing model or dataset"}), 400

    dataset_dir = project_root / "dataset" / "synthetic"
    dataset_name = next(
        (d.name for d in dataset_dir.iterdir() if d.is_dir() and d.name.lower() == dataset_id),
        None,
    )
    if dataset_name is None:
        return jsonify({"error": f"Dataset '{dataset_id}' not found"}), 404

    mix_dir = dataset_dir / dataset_name / split / "mix"
    if not mix_dir.exists():
        return jsonify({"error": f"No mix directory for split '{split}'"}), 404

    candidates = list(mix_dir.glob("*.wav"))

    # Filter by speaker if requested and metadata exists
    if speaker_filter and candidates:
        metadata_path = dataset_dir / dataset_name / split / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path) as f:
                meta = json.load(f)
            samples = meta if isinstance(meta, list) else meta.get("samples", [])
            matched = set()
            for s in samples:
                for key in ("speakers", "speaker_ids", "spk_ids"):
                    if key in s and speaker_filter in s[key]:
                        sid = s.get("id") or s.get("filename") or s.get("stem")
                        if sid:
                            matched.add(Path(sid).stem if "." in str(sid) else sid)
                        break
            filtered = [f for f in candidates if f.stem in matched]
            if filtered:
                candidates = filtered

    if not candidates:
        return jsonify({"error": "No samples found"}), 404

    chosen = _random.choice(candidates)

    # Delegate to the same logic as /api/evaluate by constructing an internal request
    return _run_evaluate(model_key, str(chosen))


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


@app.route("/api/training/<model_key>/recover-curves", methods=["POST"])
def recover_training_curves(model_key):
    """Reconstruct training_curves.png from whatever checkpoints exist."""
    if model_key not in MODEL_CONFIGS:
        return jsonify({"error": "Unknown model"}), 404

    ckpt_dir = MODEL_CONFIGS[model_key]["path"].parent
    points = {}

    # Collect loss values from periodic checkpoints
    for p in sorted(ckpt_dir.glob("checkpoint_epoch_*.pth")):
        try:
            c = torch.load(p, map_location="cpu", weights_only=False)
            epoch = c.get("epoch")
            if epoch is None:
                continue
            if "train_losses" in c and c["train_losses"]:
                # Full history stored — use it and stop scanning
                train_losses = c["train_losses"]
                val_losses = c["val_losses"]
                epochs = list(range(1, len(train_losses) + 1))
                _write_curves_png(epochs, train_losses, val_losses, ckpt_dir)
                return jsonify({"recovered": len(train_losses), "source": "full_history"})
            if "train_loss" in c and "val_loss" in c:
                points[epoch] = (c["train_loss"], c["val_loss"])
        except Exception:
            continue

    # Also check best model
    best_path = MODEL_CONFIGS[model_key]["path"]
    if best_path.exists():
        try:
            c = torch.load(best_path, map_location="cpu", weights_only=False)
            if "train_losses" in c and c["train_losses"]:
                train_losses = c["train_losses"]
                val_losses = c["val_losses"]
                epochs = list(range(1, len(train_losses) + 1))
                _write_curves_png(epochs, train_losses, val_losses, ckpt_dir)
                return jsonify({"recovered": len(train_losses), "source": "full_history"})
            epoch = c.get("epoch")
            if epoch and "train_loss" in c and "val_loss" in c:
                points[epoch] = (c["train_loss"], c["val_loss"])
        except Exception:
            pass

    if not points:
        return jsonify({"error": "No usable checkpoint data found"}), 404

    epochs = sorted(points)
    train_losses = [points[e][0] for e in epochs]
    val_losses = [points[e][1] for e in epochs]
    _write_curves_png(epochs, train_losses, val_losses, ckpt_dir)
    return jsonify({"recovered": len(epochs), "source": "sparse_checkpoints", "epochs": epochs})


def _write_curves_png(epochs, train_losses, val_losses, out_dir):
    """Write training_curves.png to out_dir."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label="Train Loss", marker="o", markersize=4)
    plt.plot(epochs, val_losses, label="Val Loss", marker="s", markersize=4)
    plt.xlabel("Epoch")
    plt.ylabel("Loss (Negative SI-SNR)")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, [-l for l in train_losses], label="Train SI-SNR", marker="o", markersize=4)
    plt.plot(epochs, [-l for l in val_losses], label="Val SI-SNR", marker="s", markersize=4)
    plt.xlabel("Epoch")
    plt.ylabel("SI-SNR (dB)")
    plt.title("Training and Validation SI-SNR")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / "training_curves.png", dpi=150)
    plt.close()


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


VIZ_DIR = project_root / "visualization" / "dataset-vizualization"


def _resolve_viz_folder(dataset_id):
    """Map dataset_id (e.g. 'titml-2spk') to the viz sub-folder (e.g. 'TITML-2spk')."""
    if not VIZ_DIR.exists():
        return None
    for d in VIZ_DIR.iterdir():
        if d.is_dir() and d.name.lower() == dataset_id.replace("-", "_").lower().replace("_", "-"):
            return d
        if d.is_dir() and d.name.lower() == dataset_id.lower():
            return d
        # Try matching TITML-2spk against titml-2spk-v2 etc.
        if d.is_dir() and dataset_id.startswith(d.name.lower()):
            return d
    # Try prefix match the other way
    for d in VIZ_DIR.iterdir():
        if d.is_dir() and d.name.lower().replace("-", "") == dataset_id.lower().replace("-", ""):
            return d
    return None


@app.route("/api/dataset/<dataset_id>/viz/list", methods=["GET"])
def list_viz_images(dataset_id):
    """Return available visualization image categories for a dataset."""
    viz_folder = _resolve_viz_folder(dataset_id)
    available = {}

    # Dataset-level comparison (top-level)
    comparison_img = VIZ_DIR / "dataset_comparison.png"
    if comparison_img.exists():
        available["comparison"] = True

    if viz_folder is None:
        return jsonify({"available": available, "has_viz": bool(available)})

    for split in ["train", "dev", "test"]:
        if (viz_folder / f"comparison_grid_{split}.png").exists():
            available.setdefault("comparison_grid", []).append(split)
        if (viz_folder / f"audio_properties_{split}.png").exists():
            available.setdefault("audio_properties", []).append(split)

    if (viz_folder / "dataset_statistics.png").exists():
        available["statistics"] = True

    return jsonify({"available": available, "has_viz": bool(available)})


@app.route("/api/dataset/<dataset_id>/viz/<image_name>", methods=["GET"])
def get_viz_image(dataset_id, image_name):
    """Serve a visualization PNG for a dataset."""
    # Sanitize
    if ".." in image_name or "/" in image_name:
        return jsonify({"error": "Invalid image name"}), 400

    if not image_name.endswith(".png"):
        image_name += ".png"

    # dataset_comparison lives at top level
    if image_name == "dataset_comparison.png":
        if (VIZ_DIR / image_name).exists():
            return send_from_directory(str(VIZ_DIR), image_name)
        return jsonify({"error": "Not found"}), 404

    viz_folder = _resolve_viz_folder(dataset_id)
    if viz_folder is None:
        return jsonify({"error": "No visualizations found for this dataset"}), 404

    img_path = viz_folder / image_name
    if not img_path.exists():
        return jsonify({"error": f"{image_name} not found"}), 404

    return send_from_directory(str(viz_folder), image_name)


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
