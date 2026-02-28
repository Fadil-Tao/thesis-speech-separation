# AGENTS.md - Agentic Coding Guidelines

## Project Overview

This is a **speech separation** research project using PyTorch and ESPnet. It implements SkiMAttention (Skipping Memory with Multi-Head Self-Attention) models for Indonesian speech separation on the TITML-IDN-mix dataset.

## Build/Test/Lint Commands

Since this is a research project without a formal build system:

```bash
# Run a Python script
python path/to/script.py

# Run a single test (when tests exist)
python -m pytest path/to/test_file.py::test_function_name -v

# Run all tests in a directory
python -m pytest tests/ -v

# Check Python syntax
python -m py_compile script.py

# Install dependencies (when requirements.txt exists)
pip install -r requirements.txt

# Install ESPnet dependencies (required for training)
pip install espnet==202304 espnet_model_zoo
pip install soundfile librosa pesq pystoi mir_eval
```

## Code Style Guidelines

### Python Style

- **Indentation**: 4 spaces (no tabs)
- **Line length**: Follow PEP 8 (79 chars for code, 72 for docstrings)
- **Quotes**: Use single quotes for strings, double quotes for docstrings
- **Encoding**: UTF-8 (include `# -*- coding: utf-8 -*-` at top of files)

### Imports

Order imports as follows:

```python
# 1. Standard library
import json
import os
import random
from pathlib import Path
from collections import defaultdict

# 2. Third-party packages
import numpy as np
import soundfile as sf
import librosa
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# 3. ESPnet imports
from espnet2.enh.encoder.conv_encoder import ConvEncoder
from espnet2.enh.decoder.conv_decoder import ConvDecoder
from espnet2.enh.espnet_model import ESPnetEnhancementModel

# 4. Local/project imports
from dataset.generator.titml_mix_generator_3spk import TITMLMixGenerator3Spk
```

### Naming Conventions

- **Classes**: `PascalCase` (e.g., `SkiMAttentionSeparator`, `IndonesianMixDataset`)
- **Functions/Methods**: `snake_case` (e.g., `train_epoch`, `calculate_si_snr`)
- **Variables**: `snake_case` (e.g., `batch_size`, `num_epochs`)
- **Constants**: `UPPER_CASE` (e.g., `LEARNING_RATE`, `NUM_EPOCHS`)
- **Private methods**: `_leading_underscore` (e.g., `_pad_or_truncate`)
- **Type hints**: Use for function parameters and return types

### Type Hints

Use type hints for function signatures:

```python
from typing import Dict, List, Optional, Tuple, Union

def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> float:
    ...
```

### Documentation

- Use triple-double quotes for docstrings
- Include Args and Returns sections for public functions
- Keep inline comments concise with `# Comment` format

```python
def forward(self, hc, S, causal=False):
    """Multi-Head Self-Attention replacing Mem-LSTM in SkiM.
    
    Args:
        hc: Tuple of (h, c) hidden states
        S: Number of segments
        causal: Whether to use causal masking
        
    Returns:
        Tuple of updated (h, c) states
    """
```

### Error Handling

- Use specific exceptions (e.g., `FileNotFoundError`, `ValueError`)
- Include descriptive error messages
- Use try/except for file I/O operations

```python
try:
    audio, sr = sf.read(file_path)
except Exception as e:
    print(f"Error loading {file_path}: {e}")
    return None
```

### Model Configuration

Store model configs as dictionaries with clear structure:

```python
model_config = {
    'encoder': {'channel': 512, 'kernel_size': 32, 'stride': 16},
    'decoder': {'channel': 512, 'kernel_size': 32, 'stride': 16},
    'separator': {
        'input_dim': 512,
        'causal': True,
        'num_spk': 2,
        'layer': 3,
        'unit': 512,
        'num_heads': 8,
    },
}
```

### Training Loop Pattern

Follow the established training pattern:

```python
def train_epoch(model, train_loader, optimizer, device, epoch):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader, desc=f'Epoch {epoch}'):
        # Move to device
        mix = batch['mix'].to(device)
        # Forward pass
        loss, stats, weight = model(...)
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIP)
        optimizer.step()
```

## Project Structure

```
speech-separation/
├── dataset/
│   ├── generator/        # Dataset generation scripts
│   ├── raw/             # Raw audio data
│   └── zips/            # Compressed datasets
├── implementation/       # Model implementations
├── train/
│   ├── 2speaker/        # 2-speaker training scripts
│   └── 3speaker/        # 3-speaker training scripts
├── checkpoints/         # Model checkpoints (gitignored)
├── past-reference/      # Reference implementations
└── visualization/       # Visualization scripts
```

## Key Dependencies

- PyTorch (with CUDA support)
- ESPnet 202304
- soundfile, librosa (audio I/O)
- numpy, matplotlib (data processing)
- mir_eval, pesq, pystoi (evaluation metrics)

## Notes for Agents

- This is a research codebase, not production software
- Many scripts are Jupyter notebook conversions (`.ipynb` → `.py`)
- GPU is required for training; CPU-only for inference is possible but slow
- Dataset paths are hardcoded and need adjustment per environment
- Checkpoints are saved to Google Drive paths in original scripts
