# Model Architecture Documentation

## Overview

This project implements two speech separation architectures:
1. **SkiM** (Skipping Memory LSTM) - Baseline model
2. **SkiM Attention** - SkiM with Multi-Head Self-Attention replacing MemLSTM

Both models support 2-speaker and 3-speaker separation tasks.

---

## SkiM (Skipping Memory LSTM)

### Architecture

```
Input Mixture
    ↓
[Conv Encoder] - 1D Convolution (512 channels, kernel=32, stride=16)
    ↓
[SegLSTM] × N - Segment-level LSTM processing
    ↓
[MemLSTM] - Memory LSTM for cross-segment dependencies
    ↓
[Conv Decoder] - Transposed Convolution (reconstruction)
    ↓
Separated Sources
```

### Components

#### 1. ConvEncoder
- **Type:** 1D Convolution
- **Purpose:** Encode waveform into latent representation
- **Parameters:**
  - `channel`: 512 (latent dimension)
  - `kernel_size`: 32 (2ms window at 16kHz)
  - `stride`: 16 (50% overlap)
- **Output:** [batch, channels, time_frames]

#### 2. SegLSTM (Segment LSTM)
- **Purpose:** Process individual segments independently
- **Parameters:**
  - `input_size`: 512
  - `hidden_size`: 512
  - `segment_size`: 20 frames
- **Processing:** Divides input into segments, applies LSTM to each

#### 3. MemLSTM (Memory LSTM)
- **Purpose:** Process hidden/cell states across segments
- **Parameters:**
  - `hidden_size`: 512
  - `mem_type`: 'hc' (process both hidden and cell states)
  - `bidirectional`: False (causal mode)
- **Mechanism:** 
  - Takes (h, c) from SegLSTM
  - Applies LSTM across segment dimension
  - Returns updated (h, c)
  - Implements "skipping" memory mechanism

#### 4. ConvDecoder
- **Type:** Transposed 1D Convolution
- **Purpose:** Reconstruct waveform from latent representation
- **Parameters:** Match encoder configuration

### Key Features

1. **Causal Processing**: Suitable for real-time applications
2. **Segment-wise Processing**: Efficient handling of long sequences
3. **Memory Mechanism**: Captures long-range dependencies via MemLSTM
4. **Skipping Connection**: Direct information flow between segments

### Implementation Files

- **Core:** `implementation/skim/skim.py`
  - `MemLSTM` class (lines 13-110)
  - `SegLSTM` class (lines 113-180)
  - `SkiM` class (lines 183-401)

- **ESPnet Wrapper:** `implementation/skim/skim-separator.py`
  - `SkiMSeparator` class
  - Integrates with ESPnet enhancement framework

---

## SkiM Attention

### Architecture

```
Input Mixture
    ↓
[Conv Encoder] - Same as SkiM
    ↓
[SegLSTM] × N - Same as SkiM
    ↓
[MemAttention] - Multi-Head Self-Attention + FFN
    ↓
[Conv Decoder] - Same as SkiM
    ↓
Separated Sources
```

### Components

#### 1. MemAttention (Replaces MemLSTM)
- **Purpose:** Process hidden/cell states using attention mechanism
- **Parameters:**
  - `hidden_size`: 512
  - `num_heads`: 8 (64 dimensions per head)
  - `mem_type`: 'hc'
- **Architecture:**
  ```
  Input: (h, c) from SegLSTM
      ↓
  [Multi-Head Self-Attention]
      ↓
  [Add & LayerNorm]
      ↓
  [Feed-Forward Network]
      ↓
  [Add & LayerNorm]
      ↓
  Output: Updated (h, c)
  ```

#### 2. Attention Mechanism
- **Type:** Multi-Head Self-Attention
- **Heads:** 8 (configurable)
- **Dimensions per head:** 512 / 8 = 64
- **Causal Masking:** Prevents attending to future segments

#### 3. Feed-Forward Network (FFN)
- **Structure:** Linear(512→2048) → ReLU → Dropout → Linear(2048→512)
- **Purpose:** Non-linear transformation after attention

### Key Differences from SkiM

| Feature | SkiM | SkiM Attention |
|---------|------|----------------|
| Memory Module | MemLSTM | MemAttention |
| Mechanism | Recurrent | Attention-based |
| Long-range Dependencies | Sequential | Direct (all-to-all) |
| Computational Complexity | O(T) | O(T²) per segment |
| Parallelization | Limited | Better |

### Implementation Files

- **Core:** `implementation/skim-attention/skim-attention.py`
  - `MemAttention` class (lines 7-99)
  - `SegLSTM` class (lines 102-180)
  - `SkiM` class (lines 183-233)

- **ESPnet Wrapper:** `implementation/skim-attention/skim-attention-separator.py`
  - `SkiMAttentionSeparator` class

---

## Model Comparison

### Performance Expectations

| Model | 2-Speaker SI-SNR | 3-Speaker SI-SNR | Training Time |
|-------|------------------|------------------|---------------|
| SkiM | ~10-15 dB | ~8-12 dB | ~2-3 hours |
| SkiM Attention | ~12-18 dB | ~10-15 dB | ~2-3 hours |

### Advantages

**SkiM:**
- Lower computational complexity
- Proven architecture
- Good for real-time applications

**SkiM Attention:**
- Better long-range dependency modeling
- Parallelizable attention mechanism
- Expected performance improvement

### Trade-offs

**SkiM:**
- Sequential processing limits parallelization
- May struggle with very long dependencies

**SkiM Attention:**
- Higher memory usage
- Quadratic complexity with sequence length
- Requires careful tuning of attention heads

---

## ESPnet Integration

Both models integrate with ESPnet's enhancement framework:

### ESPnetEnhancementModel

```python
from espnet2.enh.espnet_model import ESPnetEnhancementModel

model = ESPnetEnhancementModel(
    encoder=encoder,
    separator=separator,
    decoder=decoder,
    loss_fn=loss_fn,
    num_spk=num_spk,
)
```

### PIT (Permutation Invariant Training)

Both models use PIT to handle the permutation problem:
- Loss computed for all permutations
- Minimum loss permutation selected
- Ensures correct speaker assignment

### Loss Functions

**SI-SNR Loss:**
```python
from espnet2.enh.loss.criterions.time_domain import SISNRLoss

loss_fn = SISNRLoss()
```

**PIT Wrapper:**
```python
from espnet2.enh.loss.wrappers.pit_solver import PITSolver

pit_solver = PITSolver(criterion=loss_fn)
```

---

## Model Sizes

### Small (Recommended)
```python
layer=3, unit=512, channel=512
Parameters: ~15M
```

### Medium
```python
layer=4, unit=512, channel=512
Parameters: ~20M
```

### Large
```python
layer=6, unit=512, channel=512
Parameters: ~30M
```

---

## Usage Examples

### Building SkiM Model

```python
from espnet2.enh.encoder.conv_encoder import ConvEncoder
from espnet2.enh.decoder.conv_decoder import ConvDecoder
from implementation.skim.skim_separator import SkiMSeparator
from espnet2.enh.espnet_model import ESPnetEnhancementModel

# Encoder
encoder = ConvEncoder(channel=512, kernel_size=32, stride=16)

# Separator
separator = SkiMSeparator(
    input_dim=512,
    causal=True,
    num_spk=2,
    layer=3,
    unit=512,
    segment_size=20,
    mem_type='hc',
)

# Decoder
decoder = ConvDecoder(channel=512, kernel_size=32, stride=16)

# Full model
model = ESPnetEnhancementModel(
    encoder=encoder,
    separator=separator,
    decoder=decoder,
    num_spk=2,
)
```

### Building SkiM Attention Model

```python
from implementation.skim_attention.skim_attention_separator import SkiMAttentionSeparator

# Separator with attention
separator = SkiMAttentionSeparator(
    input_dim=512,
    causal=True,
    num_spk=2,
    layer=3,
    unit=512,
    segment_size=20,
    num_heads=8,  # Attention heads
    mem_type='hc',
)
```

---

## References

1. **SkiM Paper:** 
   - "SkiM: Skipping Memory LSTM for Low-Latency Real-Time Continuous Speech Separation"
   - https://arxiv.org/abs/2201.10800

2. **Attention Mechanism:**
   - "Attention Is All You Need" (Transformer)
   - https://arxiv.org/abs/1706.03762

3. **ESPnet:**
   - https://espnet.github.io/espnet/

---

## Notes for Training

1. **Memory Management:**
   - SkiM Attention uses more GPU memory due to attention matrices
   - Reduce batch size if OOM errors occur

2. **Attention Heads:**
   - Must divide hidden_size evenly (512 / 8 = 64)
   - More heads = more parallel attention but smaller per-head dimension

3. **Causal Mode:**
   - Both models support causal processing
   - Essential for real-time applications
   - Attention uses causal masking

4. **Checkpoint Compatibility:**
   - SkiM and SkiM Attention checkpoints are NOT interchangeable
   - Different state dict structures
