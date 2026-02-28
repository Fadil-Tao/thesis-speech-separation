# Training Documentation

This directory contains comprehensive documentation for training speech separation models.

## Files

### TRAINING_PLAN.md
Complete training plan including:
- Model architectures (SkiM and SkiM Attention)
- Dataset configurations (TITML-2spk and TITML-3spk)
- Training hyperparameters
- Directory structure
- Checkpointing strategy
- Training order recommendations
- Expected results

### MODELS.md
Detailed model documentation:
- Architecture diagrams
- Component descriptions
- Implementation details
- Model comparisons
- Usage examples
- ESPnet integration

### CONFIGS.md
Configuration reference:
- Model configurations (Small, Medium, Large)
- Training hyperparameters
- Optimizer and scheduler settings
- Dataset configurations
- Checkpoint configurations
- Hardware settings

## Quick Start

1. **Read TRAINING_PLAN.md** for overview
2. **Check MODELS.md** for architecture details
3. **Reference CONFIGS.md** for configuration options

## Training Scripts Location

Training scripts are located in:
```
train/
├── 2speaker/
│   ├── skim/
│   │   └── train_skim_2spk.py
│   └── skim-attention/
│       └── train_skim_attention_2spk.py
└── 3speaker/
    ├── skim/
    │   └── train_skim_3spk.py
    └── skim-attention/
        └── train_skim_attention_3spk.py
```

## Checkpoints Location

Model checkpoints are saved to:
```
checkpoints/
├── 2speaker/
│   ├── skim/
│   │   ├── best_model.pth
│   │   ├── checkpoint_epoch_*.pth
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

## Models to Train

1. **SkiM 2-Speaker** - Baseline model
2. **SkiM 3-Speaker** - Extended baseline
3. **SkiM Attention 2-Speaker** - Attention variant
4. **SkiM Attention 3-Speaker** - Extended attention variant

## Dataset Information

- **TITML-2spk:** 36,000 mixtures (50 hours), 2 speakers
- **TITML-3spk:** 36,000 mixtures (50 hours), 3 speakers
- Location: `dataset/synthetic/`

## Expected Training Time

- 2-Speaker models: ~2-3 hours each
- 3-Speaker models: ~3-4 hours each
- Total: ~12-14 hours for all 4 models

## Expected Performance

- **SkiM 2-Speaker:** ~10-15 dB SI-SNR
- **SkiM Attention 2-Speaker:** ~12-18 dB SI-SNR
- **SkiM 3-Speaker:** ~8-12 dB SI-SNR
- **SkiM Attention 3-Speaker:** ~10-15 dB SI-SNR

## Dependencies

See TRAINING_PLAN.md for complete dependency list.

## Notes

- All models use ESPnet framework
- SI-SNR loss with PIT (Permutation Invariant Training)
- Early stopping with patience=10
- Checkpointing every 10 epochs
