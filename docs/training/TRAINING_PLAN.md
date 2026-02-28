# Speech Separation Training Plan

## Project Overview

This document outlines the training plan for four speech separation models using the TITML-IDN synthetic dataset:

1. **SkiM 2-Speaker** (Baseline)
2. **SkiM 3-Speaker** (Extended baseline)
3. **SkiM Attention 2-Speaker** (Multi-head self-attention variant)
4. **SkiM Attention 3-Speaker** (Extended attention variant)

## Model Architectures

### SkiM (Skipping Memory LSTM)
- **Core Components:**
  - `MemLSTM`: Processes hidden/cell states between segments using LSTM
  - `SegLSTM`: Processes individual segments
  - Skipping memory mechanism for long-range dependencies

### SkiM Attention
- **Core Components:**
  - `MemAttention`: Replaces MemLSTM with Multi-Head Self-Attention + FFN + LayerNorm
  - `SegLSTM`: Same segment processing as SkiM
  - Configurable attention heads (default: 8 for 512 hidden size)

## Dataset Configuration

### TITML-2spk (2-Speaker Dataset)
- **Location:** `dataset/synthetic/TITML-2spk/`
- **Total Mixtures:** 36,000 (50 hours)
  - Train: 28,800 mixtures (40 hours)
  - Dev: 3,600 mixtures (5 hours)
  - Test: 3,600 mixtures (5 hours)
- **Structure:** mix/, s1/, s2/

### TITML-3spk (3-Speaker Dataset)
- **Location:** `dataset/synthetic/TITML-3spk/`
- **Total Mixtures:** 36,000 (50 hours)
  - Train: 28,800 mixtures (40 hours)
  - Dev: 3,600 mixtures (5 hours)
  - Test: 3,600 mixtures (5 hours)
- **Structure:** mix/, s1/, s2/, s3/

### Common Dataset Features
- Sample Rate: 16,000 Hz
- Clip Duration: 5.0 seconds
- SNR Range: -5 to +5 dB
- Time Offset: 0-1 seconds
- Speakers: 20 total (11 male, 9 female)
- Gender Balance: 70% mixed-gender combinations

## Model Configuration

### Small-Causal Variant (Recommended)
```python
model_config = {
    'encoder': {
        'channel': 512,
        'kernel_size': 32,
        'stride': 16,
    },
    'decoder': {
        'channel': 512,
        'kernel_size': 32,
        'stride': 16,
    },
    'separator': {
        'input_dim': 512,
        'causal': True,
        'num_spk': 2,  # or 3 for 3-speaker models
        'predict_noise': False,
        'nonlinear': 'relu',
        'layer': 3,
        'unit': 512,
        'segment_size': 20,
        'dropout': 0.0,
        'mem_type': 'hc',
        'seg_overlap': False,
        # For SkiM Attention only:
        'num_heads': 8,  # 512 / 8 = 64 dims per head
    }
}
```

## Training Hyperparameters

### Common Settings
```python
BATCH_SIZE = 4
NUM_EPOCHS = 100
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
GRADIENT_CLIP = 5.0
PATIENCE = 10  # Early stopping patience
```

### Optimizer & Scheduler
```python
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=3
)
```

### Loss Function
- **SI-SNR** (Scale-Invariant Signal-to-Noise Ratio) with PIT (Permutation Invariant Training)

## Directory Structure

```
speech-separation/
├── train/
│   ├── 2speaker/
│   │   ├── skim/
│   │   │   └── train_skim_2spk.py
│   │   └── skim-attention/
│   │       └── train_skim_attention_2spk.py
│   └── 3speaker/
│       ├── skim/
│       │   └── train_skim_3spk.py
│       └── skim-attention/
│           └── train_skim_attention_3spk.py
├── checkpoints/
│   ├── 2speaker/
│   │   ├── skim/
│   │   │   ├── best_model.pth
│   │   │   ├── checkpoint_epoch_*.pth
│   │   │   ├── training_curves.png
│   │   │   └── training.log
│   │   └── skim-attention/
│   │       ├── best_model.pth
│   │       ├── checkpoint_epoch_*.pth
│   │       ├── training_curves.png
│   │       └── training.log
│   └── 3speaker/
│       ├── skim/
│       │   └── [same structure]
│       └── skim-attention/
│           └── [same structure]
└── docs/
    └── training/
        ├── README.md
        ├── MODELS.md
        └── CONFIGS.md
```

## Training Scripts

### Script Locations
1. **SkiM 2-Speaker:** `train/2speaker/skim/train_skim_2spk.py`
2. **SkiM Attention 2-Speaker:** `train/2speaker/skim-attention/train_skim_attention_2spk.py`
3. **SkiM 3-Speaker:** `train/3speaker/skim/train_skim_3spk.py`
4. **SkiM Attention 3-Speaker:** `train/3speaker/skim-attention/train_skim_attention_3spk.py`

### Script Components
Each training script should include:

1. **Imports & Setup**
   - ESPnet components
   - Model implementations from `implementation/`
   - Dataset loaders
   - Training utilities

2. **Dataset Class**
   - `IndonesianMixDataset` with padding/truncation
   - Support for 2 or 3 speakers

3. **Model Building**
   - Encoder (Conv1D)
   - Separator (SkiM or SkiM Attention)
   - Decoder (ConvTranspose1D)
   - ESPnet wrapper with PIT

4. **Training Loop**
   - Epoch-based training
   - Validation after each epoch
   - Early stopping
   - Learning rate scheduling
   - Checkpointing every 10 epochs

5. **Evaluation**
   - SI-SNR metric computation
   - SDR metric (optional, requires mir_eval)
   - Test set evaluation

6. **Logging & Visualization**
   - Training curves (loss and SI-SNR)
   - Progress bars
   - Model statistics

## Checkpointing Strategy

### Files Saved
1. **best_model.pth** - Best model based on validation loss
   - epoch
   - model_state_dict
   - optimizer_state_dict
   - train_loss
   - val_loss
   - best_val_loss
   - config

2. **checkpoint_epoch_*.pth** - Periodic checkpoints (every 10 epochs)
   - epoch
   - model_state_dict
   - optimizer_state_dict
   - train_loss
   - val_loss

3. **training_curves.png** - Training visualization
   - Loss curves
   - SI-SNR curves

4. **training.log** - Complete training log

### Checkpoint Directory
```
checkpoints/
├── 2speaker/
│   ├── skim/
│   │   ├── best_model.pth
│   │   ├── checkpoint_epoch_10.pth
│   │   ├── checkpoint_epoch_20.pth
│   │   ├── ...
│   │   ├── training_curves.png
│   │   └── training.log
│   └── skim-attention/
│       └── [same structure]
└── 3speaker/
    ├── skim/
    │   └── [same structure]
    └── skim-attention/
        └── [same structure]
```

## Implementation Files

### Core Models
- **SkiM:** `implementation/skim/skim.py`
  - `MemLSTM` class
  - `SegLSTM` class
  - `SkiM` class

- **SkiM Separator:** `implementation/skim/skim-separator.py`
  - `SkiMSeparator` class (ESPnet-compatible)

- **SkiM Attention:** `implementation/skim-attention/skim-attention.py`
  - `MemAttention` class
  - `SegLSTM` class
  - `SkiM` class

- **SkiM Attention Separator:** `implementation/skim-attention/skim-attention-separator.py`
  - `SkiMAttentionSeparator` class (ESPnet-compatible)

### Reference Implementations
- **SkiM Training:** `past-reference/skim_titml_2mix_raw.py`
- **SkiM Attention Training:** `past-reference/skim_attention_titml_2mix_raw.py`

## Training Order Recommendation

1. **SkiM 2-Speaker** (Baseline)
   - Validate training pipeline
   - Establish baseline metrics

2. **SkiM Attention 2-Speaker** (Variant)
   - Compare with SkiM baseline
   - Evaluate attention mechanism impact

3. **SkiM 3-Speaker** (Extended)
   - Test scalability to more speakers
   - May require hyperparameter tuning

4. **SkiM Attention 3-Speaker** (Extended Variant)
   - Compare with SkiM 3-speaker
   - Final comparison across all models

## Expected Results

### Training Time (Estimated)
- **2-Speaker Models:** ~2-3 hours per model (on GPU)
- **3-Speaker Models:** ~3-4 hours per model (on GPU)

### Expected SI-SNR Performance
- **SkiM 2-Speaker:** ~10-15 dB
- **SkiM Attention 2-Speaker:** ~12-18 dB (improvement expected)
- **SkiM 3-Speaker:** ~8-12 dB (harder task)
- **SkiM Attention 3-Speaker:** ~10-15 dB

## Dependencies

```bash
# Core dependencies
pip install torch torchvision torchaudio
pip install espnet==202304
pip install espnet_model_zoo

# Audio processing
pip install soundfile librosa

# Evaluation metrics
pip install mir_eval pesq pystoi

# Visualization
pip install matplotlib tqdm
```

## Usage

### Training a Model

```bash
# Activate virtual environment
source .venv/bin/activate

# Train SkiM 2-Speaker
python train/2speaker/skim/train_skim_2spk.py

# Train SkiM Attention 2-Speaker
python train/2speaker/skim-attention/train_skim_attention_2spk.py

# Train SkiM 3-Speaker
python train/3speaker/skim/train_skim_3spk.py

# Train SkiM Attention 3-Speaker
python train/3speaker/skim-attention/train_skim_attention_3spk.py
```

### Monitoring Training

Check training logs:
```bash
tail -f checkpoints/2speaker/skim/training.log
```

View training curves:
```bash
# Open checkpoints/2speaker/skim/training_curves.png
```

## Notes for Agents

1. **Model Selection:** Use implementations from `implementation/` folder, not inline definitions
2. **Dataset Paths:** Use absolute paths or paths relative to project root
3. **Checkpointing:** Always save best model and periodic checkpoints
4. **Logging:** Log all hyperparameters and configurations
5. **Reproducibility:** Set random seeds for reproducibility
6. **GPU Memory:** Monitor GPU memory usage, reduce batch size if needed
7. **Early Stopping:** Use patience=10 to prevent overfitting
8. **Evaluation:** Always evaluate on test set after training completes

## Future Work

- Transfer learning experiments (2spk → 3spk)
- Different model sizes (Small, Medium, Large)
- Real-time inference optimization
- Cross-dataset evaluation

## References

- SkiM Paper: "SkiM: Skipping Memory LSTM for Low-Latency Real-Time Continuous Speech Separation" (https://arxiv.org/abs/2201.10800)
- ESPnet: https://espnet.github.io/espnet/
- TITML-IDN Dataset: Indonesian speech dataset
