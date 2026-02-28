# Training Infrastructure - Complete Summary

## ✅ Created Files

### 1. Documentation (docs/training/)

#### TRAINING_PLAN.md
Comprehensive training plan including:
- Model architectures (SkiM and SkiM Attention)
- Dataset configurations (TITML-2spk and TITML-3spk)
- Training hyperparameters
- Directory structure
- Checkpointing strategy
- Training order recommendations
- Expected results

#### MODELS.md
Detailed model documentation:
- Architecture diagrams
- Component descriptions (MemLSTM, MemAttention, SegLSTM)
- Implementation details
- Model comparisons
- Usage examples
- ESPnet integration

#### CONFIGS.md
Configuration reference:
- Model configurations (Small, Medium, Large)
- Training hyperparameters
- Optimizer and scheduler settings
- Dataset configurations
- Checkpoint configurations
- Hardware settings

#### README.md
Quick reference guide for the training documentation

### 2. Training Scripts (train/)

#### 2-Speaker Models

**train/2speaker/skim/train_skim_2spk.py**
- Trains SkiM model for 2-speaker separation
- Uses TITML-2spk dataset
- Saves checkpoints to: `checkpoints/2speaker/skim/`

**train/2speaker/skim-attention/train_skim_attention_2spk.py**
- Trains SkiM Attention model for 2-speaker separation
- Uses TITML-2spk dataset
- Saves checkpoints to: `checkpoints/2speaker/skim-attention/`

#### 3-Speaker Models

**train/3speaker/skim/train_skim_3spk.py**
- Trains SkiM model for 3-speaker separation
- Uses TITML-3spk dataset
- Saves checkpoints to: `checkpoints/3speaker/skim/`

**train/3speaker/skim-attention/train_skim_attention_3spk.py**
- Trains SkiM Attention model for 3-speaker separation
- Uses TITML-3spk dataset
- Saves checkpoints to: `checkpoints/3speaker/skim-attention/`

## 📁 Directory Structure

```
speech-separation/
├── docs/
│   └── training/
│       ├── README.md              # Quick reference
│       ├── TRAINING_PLAN.md       # Complete training plan
│       ├── MODELS.md              # Model architecture docs
│       └── CONFIGS.md             # Configuration reference
│
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
│
└── checkpoints/
    ├── 2speaker/
    │   ├── skim/
    │   │   ├── best_model.pth
    │   │   ├── checkpoint_epoch_*.pth
    │   │   ├── training_curves.png
    │   │   └── config.json
    │   └── skim-attention/
    │       └── [same structure]
    └── 3speaker/
        ├── skim/
        │   └── [same structure]
        └── skim-attention/
            └── [same structure]
```

## 🔧 Configuration

### Model Configuration (Small-Causal)
```python
MODEL_CONFIG = {
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
        'num_spk': 2,  # or 3
        'layer': 3,
        'unit': 512,
        'segment_size': 20,
        'num_heads': 8,  # Only for Attention
    }
}
```

### Training Configuration
```python
TRAIN_CONFIG = {
    'batch_size': 4,
    'num_epochs': 100,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'gradient_clip': 5.0,
    'patience': 10,
}
```

## 🚀 Usage

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

### Expected Training Time
- 2-Speaker models: ~2-3 hours each
- 3-Speaker models: ~3-4 hours each
- Total: ~12-14 hours for all 4 models

### Expected Performance
- **SkiM 2-Speaker:** ~10-15 dB SI-SNR
- **SkiM Attention 2-Speaker:** ~12-18 dB SI-SNR
- **SkiM 3-Speaker:** ~8-12 dB SI-SNR
- **SkiM Attention 3-Speaker:** ~10-15 dB SI-SNR

## 📊 Training Features

Each training script includes:

1. **Dataset Loading**
   - Custom `IndonesianMixDataset` class
   - Handles 2 or 3 speakers
   - Automatic padding and collation

2. **Model Building**
   - ESPnet-compatible model construction
   - Encoder → Separator → Decoder pipeline
   - PIT (Permutation Invariant Training) loss

3. **Training Loop**
   - Epoch-based training
   - Validation after each epoch
   - Progress bars with tqdm
   - Gradient clipping

4. **Optimization**
   - Adam optimizer
   - ReduceLROnPlateau scheduler
   - Early stopping (patience=10)

5. **Checkpointing**
   - Best model saved automatically
   - Periodic checkpoints every 10 epochs
   - Complete state dict (model, optimizer, scheduler)

6. **Logging & Visualization**
   - Training curves (loss and SI-SNR)
   - Config saved as JSON
   - Progress printed to console

## 📝 Key Differences Between Scripts

### 2-Speaker vs 3-Speaker
- Dataset path: `TITML-2spk` vs `TITML-3spk`
- `num_spk`: 2 vs 3
- Additional `s3` source in dataset class
- Additional `speech_ref3` in forward pass

### SkiM vs SkiM Attention
- Separator import: `SkiMSeparator` vs `SkiMAttentionSeparator`
- Additional `num_heads` parameter (8)
- Different checkpoint directories

## 🎯 Training Order Recommendation

1. **SkiM 2-Speaker** - Validate pipeline, establish baseline
2. **SkiM Attention 2-Speaker** - Compare with baseline
3. **SkiM 3-Speaker** - Test scalability
4. **SkiM Attention 3-Speaker** - Final comparison

## 📦 Dependencies

```bash
# Core
pip install torch torchvision torchaudio
pip install espnet==202304
pip install espnet_model_zoo

# Audio
pip install soundfile librosa

# Metrics
pip install mir_eval pesq pystoi

# Viz
pip install matplotlib tqdm
```

## 🔍 Monitoring Training

Check logs:
```bash
tail -f checkpoints/2speaker/skim/training.log
```

View training curves:
```bash
# Open checkpoints/2speaker/skim/training_curves.png
```

## 🎓 Notes for Future Agents

1. **Model Selection:** Use implementations from `implementation/` folder
2. **Dataset Paths:** Use absolute paths or paths relative to project root
3. **Checkpointing:** Always saves best model and periodic checkpoints
4. **Logging:** All configs logged for reproducibility
5. **Reproducibility:** Random seeds set for consistency
6. **GPU Memory:** Monitor with `nvidia-smi`, reduce batch size if OOM
7. **Early Stopping:** Patience=10 prevents overfitting

## 📚 References

- SkiM Paper: https://arxiv.org/abs/2201.10800
- ESPnet: https://espnet.github.io/espnet/
- TITML-IDN Dataset: Indonesian speech dataset

---

**All training infrastructure is ready!** You can now run the training scripts as needed. The scripts are self-contained and will handle everything from data loading to checkpointing automatically.
