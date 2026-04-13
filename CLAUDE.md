# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Speech separation research project implementing **SkiM** (Skipping Memory LSTM) and **SkiM Attention** (SkiM with Multi-Head Self-Attention replacing MemLSTM) models for Indonesian speech separation. Uses the synthetic TITML-IDN-mix dataset and the ESPnet framework.

## Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Train models
python train/2speaker/skim/train_skim_2spk.py
python train/2speaker/skim-attention/train_skim_attention_2spk.py
python train/3speaker/skim/train_skim_3spk.py
python train/3speaker/skim-attention/train_skim_attention_3spk.py

# Monitor training
tail -f checkpoints/2speaker/skim/training.log

# Generate dataset
python dataset/generator/titml_mix_generator_2spk.py
python dataset/generator/titml_mix_generator_3spk.py

# Syntax check
python -m py_compile path/to/script.py

# Run tests
python -m pytest tests/ -v
python -m pytest path/to/test_file.py::test_function_name -v
```

## Architecture

### Model Stack (ESPnet-wrapped)

Each trained model is assembled from three components wired into `ESPnetEnhancementModel`:

1. **Encoder** — `ConvEncoder` (Conv1D, kernel=32, stride=16, channel=256 or 512)
2. **Separator** — either `SkiMSeparator` or `SkiMAttentionSeparator` (see below)
3. **Decoder** — `ConvDecoder` (ConvTranspose1D, matching encoder settings)

Loss: SI-SNR via `SISNRLoss` + Permutation Invariant Training via `PITSolver`.

### Core Implementations (`implementation/`)

| File | Class | Key Role |
|---|---|---|
| `skim/skim.py` | `MemLSTM`, `SegLSTM`, `SkiM` | Base SkiM network |
| `skim/skim_separator.py` | `SkiMSeparator` | ESPnet-compatible separator wrapper |
| `skim_attention/skim_attention.py` | `MemAttention`, `SegLSTM`, `SkiM` | Attention variant (drops MemLSTM) |
| `skim_attention/skim_attention_separator.py` | `SkiMAttentionSeparator` | ESPnet-compatible separator wrapper |

**SkiM** alternates `SegLSTM` blocks (process frames within a segment) with `MemLSTM` blocks (propagate hidden/cell states across segments). **SkiM Attention** replaces `MemLSTM` with `MemAttention` (Multi-Head Self-Attention + FFN + LayerNorm).

### Dataset (`dataset/`)

- **Raw source:** `dataset/raw/TTML-IDN/` — 20 Indonesian speakers (11M, 9F), WAV @ 16 kHz
- **Synthetic mixtures:** `dataset/synthetic/TITML-2spk/` and `TITML-3spk/`
  - 36,000 mixtures each (28800 train / 3600 dev / 3600 test)
  - 5-second clips, SNR −5 to +5 dB, time offset 0–1 s
  - Structure: `mix/`, `s1/`, `s2/` (and `s3/` for 3-speaker)
- Generators: `dataset/generator/titml_mix_generator_{2,3}spk.py`

### Training Scripts

All scripts follow the same pattern:
1. Parse config dicts at the top (`MODEL_CONFIG`, `TRAIN_CONFIG`)
2. `IndonesianMixDataset` class handles loading and padding/truncation to fixed length
3. Adam optimizer + `ReduceLROnPlateau` scheduler (factor=0.5, patience=3)
4. Gradient clipping at 5.0
5. Save `best_model.pth` on val loss improvement; periodic `checkpoint_epoch_N.pth` every 10 epochs
6. Evaluate SI-SNR on test set after training

Scripts add `project_root` (4 levels up from the script) to `sys.path` so imports from `implementation/` resolve correctly.

## Key Configuration Defaults

```python
# Recommended small-causal config (docs use channel=512; train scripts use 256)
MODEL_CONFIG = {
    'encoder':   {'channel': 256, 'kernel_size': 32, 'stride': 16},
    'decoder':   {'channel': 256, 'kernel_size': 32, 'stride': 16},
    'separator': {
        'input_dim': 256, 'causal': False, 'num_spk': 2,
        'layer': 4, 'unit': 256, 'segment_size': 20, 'dropout': 0.1,
        'mem_type': 'hc', 'seg_overlap': False,
        # SkiM Attention only:
        'num_heads': 8,  # must divide unit evenly
    }
}
TRAIN_CONFIG = {'batch_size': 4, 'num_epochs': 100, 'learning_rate': 1e-3,
                'weight_decay': 1e-5, 'gradient_clip': 5.0, 'patience': 10, 'seed': 42}
```

## Dependencies

```bash
pip install torch torchvision torchaudio
pip install espnet==202304 espnet_model_zoo
pip install soundfile librosa
pip install mir_eval pesq pystoi
pip install matplotlib tqdm
```

GPU is required for practical training. Set `CUDA_VISIBLE_DEVICES=0` to select a specific GPU.

## Reference Implementations

`past-reference/` contains earlier notebook-converted scripts (`skim_titml_2mix_raw.py`, `skim_attention_titml_2mix_raw.py`) — useful as working examples but not the canonical implementations.
