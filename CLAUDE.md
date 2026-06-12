# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Speech separation research project implementing **SkiM** (Skipping Memory LSTM) and **SkiM Attention** (SkiM with an added intra-segment Multi-Head Self-Attention block; MemLSTM retained) models for Indonesian speech separation. Uses the synthetic TITML-IDN-mix dataset and the ESPnet framework.

## Commands

```bash
# Activate virtual environment
source .venv/bin/activate

# Train models — train/train/ holds the most up-to-date scripts
# 2-speaker (pretrain source for transfer learning)
python train/train/2speaker/skim/train_skim_2spk.py
python train/train/2speaker/skim-attention/train_skim_attention_2spk.py
# 3-speaker from scratch
python train/train/3speaker/skim/train_skim_3spk.py
python train/train/3speaker/skim-attention/train_skim_attention_3spk.py
# 3-speaker transfer learning from 2-speaker pretrained
python train/train/3speaker/skim-transfer-learning/train_skim_transfer_3spk.py
python train/train/3speaker/skim-attention-transfer-learning/train_skim_attention_transfer_3spk.py

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

1. **Encoder** — `ConvEncoder` (Conv1D, kernel=16, stride=8, channel=256)
2. **Separator** — either `SkiMSeparator` or `SkiMAttentionV3Separator` (see below)
3. **Decoder** — `ConvDecoder` (ConvTranspose1D, matching encoder settings)

Loss: SI-SNR via `SISNRLoss` + Permutation Invariant Training via `PITSolver`.

### Core Implementations (`implementation/`)

| File | Class | Key Role |
|---|---|---|
| `skim/skim.py` | `MemLSTM`, `SegLSTM`, `SkiM` | Base SkiM network |
| `skim/skim_separator.py` | `SkiMSeparator` | ESPnet-compatible separator wrapper |
| `skim_attention_v3/skim_attention_v3.py` | `MemLSTM`, `SegLSTM`, `SegAttention`, `SkiM` | **Canonical thesis Attention variant** (used by training) |
| `skim_attention_v3/skim_attention_v3_separator.py` | `SkiMAttentionV3Separator` | ESPnet-compatible separator wrapper |
| `skim_attention/skim_attention.py` | `MemAttention`, `SegLSTM`, `SkiM` | **Old v1 variant — replaces MemLSTM with MemAttention. NOT the thesis model.** |

**SkiM** alternates `SegLSTM` blocks (process frames within a segment) with `MemLSTM` blocks (propagate hidden/cell states across segments).

**SkiM Attention (thesis model = v3)** does **NOT** replace MemLSTM. It **adds** a Multi-Head Self-Attention block (`SegAttention`) at the **intra-segment** level, inserted after each `SegLSTM`. Block flow: `SegLSTM → SegAttention (NEW) → [MemLSTM if not last block]`. `SegAttention` is a pre-norm Transformer block (MHSA + FFN + LayerNorm) with LayerScale gates (`gamma_attn`, `gamma_ffn`) initialized to 0, so it starts as identity and grows only if intra-segment attention helps. MemLSTM is retained.

⚠️ Do **not** describe the thesis contribution as "replacing MemLSTM" — that is the deprecated v1 variant. The thesis adds intra-segment MHSA. Variants `skim_attention`, `skim_attention_v2a`, `skim_attention_v2b` are experimental; `skim_attention_v3` is canonical.

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
3. Adam optimizer + `ExponentialLR` scheduler (gamma=0.97, stepped every epoch)
4. Gradient clipping at 5.0
5. Save `best_model.pth` on val loss improvement; periodic `checkpoint_epoch_N.pth` every 10 epochs
6. Evaluate SI-SNR on test set after training

Scripts add `project_root` (4 levels up from the script) to `sys.path` so imports from `implementation/` resolve correctly.

## Key Configuration Defaults

```python
# Canonical config — matches the up-to-date scripts under train/train/
MODEL_CONFIG = {
    'encoder':   {'channel': 256, 'kernel_size': 16, 'stride': 8},
    'decoder':   {'channel': 256, 'kernel_size': 16, 'stride': 8},
    'separator': {
        'input_dim': 256, 'causal': False, 'num_spk': 2,  # 3 for 3-speaker
        'predict_noise': False, 'nonlinear': 'relu',
        'layer': 4, 'unit': 256, 'segment_size': 150, 'dropout': 0.1,
        'mem_type': 'hc', 'seg_overlap': False,
        # SkiM Attention (v3) only:
        'num_heads': 4,  # 256 / 4 = 64 dims per head; must divide unit evenly
    }
}
TRAIN_CONFIG = {'batch_size': 8, 'num_epochs': 100, 'learning_rate': 1e-3,
                'weight_decay': 0.0, 'gradient_clip': 5.0, 'seed': 42}
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
