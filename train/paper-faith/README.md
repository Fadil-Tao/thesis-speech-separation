# Paper-Faithful Training

Drop-in wrappers around the baseline training scripts, with two separator
hyperparameters overridden to match the SkiM paper (Li et al., ICASSP 2022,
Sec 3.3):

  - `separator.segment_size`: 20 → **150**
  - `separator.dropout`:      0.2 → **0.1**

Everything else (encoder, optimizer, loss, dataset split, transfer logic,
γ-regularization, etc.) is **identical** to the corresponding baseline script.

## Cases (7)

### 2-speaker
| Script | Base | Output dir |
|---|---|---|
| `2speaker/train_skim_paperfaith.py` | `train/2speaker/skim/` | `checkpoints/paper-faith/2speaker-skim/` |
| `2speaker/train_skim_attention_v3_paperfaith.py` | `train/2speaker/skim-attention-v3/` | `checkpoints/paper-faith/2speaker-skim-attention-v3/` |
| `2speaker/train_skim_attention_v3_reg_paperfaith.py` | `train/2speaker/skim-attention-v3-reg/` | `checkpoints/paper-faith/2speaker-skim-attention-v3-reg/` |

### 3-speaker
| Script | Base | Output dir |
|---|---|---|
| `3speaker/train_skim_paperfaith.py` | `train/3speaker/skim/` | `checkpoints/paper-faith/3speaker-skim/` |
| `3speaker/train_skim_attention_v3_paperfaith.py` | `train/3speaker/skim-attention-v3/` | `checkpoints/paper-faith/3speaker-skim-attention-v3/` |
| `3speaker/train_skim_transfer_paperfaith.py` | `train/3speaker/skim/train_skim_3spk_transfer.py` | `checkpoints/paper-faith/3speaker-skim-transfer/` |
| `3speaker/train_skim_attention_v3_reg_transfer_paperfaith.py` | `train/3speaker/skim-attention-v3-reg/` | `checkpoints/paper-faith/3speaker-skim-attention-v3-reg-transfer/` |

### Why 7 not 6
The 6 evaluated models are 2spk skim, 2spk v3, 3spk skim (cold), 3spk v3 (cold),
3spk skim-transfer, 3spk v3-reg-transfer. The 7th (2spk v3-reg) is the
pretrain for the v3-reg-transfer run.

## Defaults

- `--num-epochs`: **100** (matches paper Sec 3.3)
- All output dirs auto-resolve via `utils.paths.get_checkpoint_dir`.

## Run order

Pretrains first (2spk), then 3spk. Transfer scripts auto-point at the paper-faith
2spk checkpoints.

```bash
# === 2-speaker pretrains (run in parallel if multi-GPU) ===
python train/paper-faith/2speaker/train_skim_paperfaith.py
python train/paper-faith/2speaker/train_skim_attention_v3_paperfaith.py
python train/paper-faith/2speaker/train_skim_attention_v3_reg_paperfaith.py

# === 3-speaker cold start ===
python train/paper-faith/3speaker/train_skim_paperfaith.py
python train/paper-faith/3speaker/train_skim_attention_v3_paperfaith.py

# === 3-speaker transfer (needs 2spk done first) ===
python train/paper-faith/3speaker/train_skim_transfer_paperfaith.py
python train/paper-faith/3speaker/train_skim_attention_v3_reg_transfer_paperfaith.py
```

## Common overrides

```bash
# shorter run
python train/paper-faith/2speaker/train_skim_paperfaith.py --num-epochs 30

# resume
python train/paper-faith/2speaker/train_skim_paperfaith.py \
    --resume-from checkpoint_epoch_50.pth

# custom checkpoint dir
python train/paper-faith/2speaker/train_skim_paperfaith.py \
    --checkpoint-dir /tmp/my-run

# transfer with custom pretrained path
python train/paper-faith/3speaker/train_skim_transfer_paperfaith.py \
    --pretrained-path /path/to/other-2spk-ckpt.pth
```

## What this directory is for

Provides a citeable "paper-aligned" reference number for each model alongside
the existing baseline (K=20 dropout=0.2) and the K-size ablation results.
Comparison table for thesis:

```
                            baseline   K-ablation winner   paper-faith
2spk skim                    14.17 dB        TBD             TBD
2spk v3                      14.35 dB        n/a             TBD
...
```
