# Training Configuration Reference

## Model Configurations

### Small-Causal (Recommended for Initial Training)

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
        'num_spk': 2,  # Change to 3 for 3-speaker
        'predict_noise': False,
        'nonlinear': 'relu',
        'layer': 3,
        'unit': 512,
        'segment_size': 20,
        'dropout': 0.0,
        'mem_type': 'hc',
        'seg_overlap': False,
    }
}
```

**Parameters:**
- `channel`: Number of filters in encoder/decoder (512)
- `kernel_size`: Window size in samples (32 = 2ms at 16kHz)
- `stride`: Hop size in samples (16 = 50% overlap)
- `layer`: Number of SkiM blocks (3 for Small variant)
- `unit`: LSTM hidden dimension (512)
- `segment_size`: Segment length for SegLSTM (20 frames)
- `mem_type`: Memory state type ('hc' = hidden + cell)

### SkiM Attention Configuration

Additional parameter for SkiM Attention:
```python
MODEL_CONFIG['separator']['num_heads'] = 8  # 512 / 8 = 64 dims per head
```

**Attention Parameters:**
- `num_heads`: Number of attention heads (must divide hidden_size evenly)
- `hidden_size`: Must be divisible by num_heads (512 / 8 = 64)

## Training Hyperparameters

### Default Configuration

```python
TRAIN_CONFIG = {
    'batch_size': 4,
    'num_epochs': 100,
    'learning_rate': 1e-3,
    'weight_decay': 1e-5,
    'gradient_clip': 5.0,
    'patience': 10,  # Early stopping
    'seed': 42,
}
```

### Optimizer Settings

```python
OPTIMIZER_CONFIG = {
    'type': 'Adam',
    'lr': 1e-3,
    'betas': (0.9, 0.999),
    'eps': 1e-8,
    'weight_decay': 1e-5,
}
```

### Learning Rate Scheduler

```python
SCHEDULER_CONFIG = {
    'type': 'ReduceLROnPlateau',
    'mode': 'min',
    'factor': 0.5,
    'patience': 3,
    'verbose': True,
}
```

## Dataset Configuration

### 2-Speaker Dataset

```python
DATASET_CONFIG_2SPK = {
    'name': 'TITML-2spk',
    'path': 'dataset/synthetic/TITML-2spk',
    'num_speakers': 2,
    'sample_rate': 16000,
    'clip_duration': 5.0,
    'splits': {
        'train': 28800,
        'dev': 3600,
        'test': 3600,
    }
}
```

### 3-Speaker Dataset

```python
DATASET_CONFIG_3SPK = {
    'name': 'TITML-3spk',
    'path': 'dataset/synthetic/TITML-3spk',
    'num_speakers': 3,
    'sample_rate': 16000,
    'clip_duration': 5.0,
    'splits': {
        'train': 28800,
        'dev': 3600,
        'test': 3600,
    }
}
```

## Checkpoint Configuration

### Checkpoint Saving

```python
CHECKPOINT_CONFIG = {
    'save_every': 10,  # Save every N epochs
    'keep_last_n': 3,  # Keep only last N checkpoints (optional)
    'save_best': True,  # Always save best model
}
```

### Checkpoint Contents

```python
checkpoint = {
    'epoch': epoch,
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'train_loss': train_loss,
    'val_loss': val_loss,
    'best_val_loss': best_val_loss,
    'config': MODEL_CONFIG,
}
```

## Hardware Configuration

### GPU Settings

```python
GPU_CONFIG = {
    'device': 'cuda',  # or 'cpu'
    'mixed_precision': False,  # Set True for AMP (not implemented yet)
    'num_workers': 4,  # DataLoader workers
}
```

### Memory Management

For limited GPU memory:
```python
# Reduce batch size
TRAIN_CONFIG['batch_size'] = 2

# Reduce model size
MODEL_CONFIG['separator']['layer'] = 2
MODEL_CONFIG['separator']['unit'] = 256
MODEL_CONFIG['encoder']['channel'] = 256
MODEL_CONFIG['decoder']['channel'] = 256
```

## Model Variants

### Small (Current)
```python
layer=3, unit=512, channel=512
```

### Medium
```python
layer=4, unit=512, channel=512
```

### Large
```python
layer=6, unit=512, channel=512
```

### Tiny (for testing)
```python
layer=2, unit=256, channel=256
```

## Loss Function Configuration

### SI-SNR Loss

```python
LOSS_CONFIG = {
    'type': 'si_snr',
    'eps': 1e-8,
    'pit': True,  # Permutation Invariant Training
}
```

### Alternative Losses (for future experiments)

```python
# L1 Loss in time domain
LOSS_CONFIG = {'type': 'l1'}

# L1 Loss in frequency domain
LOSS_CONFIG = {'type': 'freq_l1'}

# Combined loss
LOSS_CONFIG = {
    'type': 'combined',
    'si_snr_weight': 0.8,
    'l1_weight': 0.2,
}
```

## Evaluation Metrics

### Primary Metrics
- **SI-SNR**: Scale-Invariant Signal-to-Noise Ratio (main metric)
- **SDR**: Signal-to-Distortion Ratio (using mir_eval)

### Secondary Metrics
- **PESQ**: Perceptual Evaluation of Speech Quality
- **STOI**: Short-Time Objective Intelligibility

## Configuration by Model

### SkiM 2-Speaker
```python
MODEL_CONFIG['separator']['num_spk'] = 2
# Use SkiMSeparator from implementation/skim/
```

### SkiM 3-Speaker
```python
MODEL_CONFIG['separator']['num_spk'] = 3
# Use SkiMSeparator from implementation/skim/
```

### SkiM Attention 2-Speaker
```python
MODEL_CONFIG['separator']['num_spk'] = 2
MODEL_CONFIG['separator']['num_heads'] = 8
# Use SkiMAttentionSeparator from implementation/skim-attention/
```

### SkiM Attention 3-Speaker
```python
MODEL_CONFIG['separator']['num_spk'] = 3
MODEL_CONFIG['separator']['num_heads'] = 8
# Use SkiMAttentionSeparator from implementation/skim-attention/
```

## Configuration Files Location

Save configurations in:
```
checkpoints/{2speaker,3speaker}/{skim,skim-attention}/
├── config.json          # Model and training config
├── best_model.pth       # Best model checkpoint
└── training.log         # Training log
```

## Loading Configuration

```python
import json

# Load config
with open('checkpoints/2speaker/skim/config.json', 'r') as f:
    config = json.load(f)

# Reconstruct model
model_config = config['model_config']
train_config = config['train_config']
```

## Environment Variables

Optional environment variables:
```bash
export CUDA_VISIBLE_DEVICES=0  # Select GPU
export OMP_NUM_THREADS=4       # CPU threads
export PYTHONHASHSEED=42       # Reproducibility
```

## Notes

1. **Reproducibility**: Always set random seeds for consistent results
2. **GPU Memory**: Monitor with `nvidia-smi`, adjust batch size if OOM
3. **Checkpointing**: Saves disk space by keeping only necessary checkpoints
4. **Logging**: All configs should be logged for reproducibility
