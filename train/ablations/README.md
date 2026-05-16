# Ablations

Ablation studies for the **baseline SkiM 2-speaker** model only
(`train/2speaker/skim/train_skim_2spk.py`). Not run on attention variants
or transfer-learning models.

## Structure

```
train/ablations/
├── README.md                              ← this file
├── run_ablation.sh                        ← sweep runner
├── k-size/
│   └── train_k_ablation.py                ← K (segment_size) ablation
└── justifications/
    └── CONFIG_JUSTIFICATIONS.md           ← per-param rationale
```

## K-size ablation

Tests whether `separator.segment_size` (K) of 20 — inherited from prototype
without documented justification — is appropriate.

K controls local context length per Seg-LSTM:
```
local_context_ms = K · encoder_stride / sample_rate
                 = K · 16 / 16000
                 = K ms  (at our config)
```

Hypothesis: K=20 (= 20 ms ≈ sub-phoneme) is too short. Paper uses K=150,
ESPnet yaml uses K=250 (with seg_overlap=True).

### Run sweep — local

```bash
# default: K ∈ {20, 50, 100, 150}, 30 epochs each, no seg_overlap
bash train/ablations/run_ablation.sh

# enable 50% segment overlap
OVERLAP=1 bash train/ablations/run_ablation.sh

# longer per-run for confirmatory runs
EPOCHS=100 K_VALUES="20 100" bash train/ablations/run_ablation.sh
```

### Run sweep — vast.ai (full pipeline incl. dataset prep + R2 upload)

```bash
# requires scripts/r2.env with R2 creds
bash scripts/vast_run_ablation.sh

# overrides
EPOCHS=50 K_VALUES="20 100" OVERLAP=1 bash scripts/vast_run_ablation.sh
DRYRUN_DATASET_ONLY=1 bash scripts/vast_run_ablation.sh   # stop after dataset gen
```

This handles:
1. Python env + CUDA torch install
2. R2 verify
3. TITML-IDN raw download (gdown + AES unzip)
4. TITML-2spk-v2 synthetic gen
5. K-ablation sweep
6. Per-run R2 upload to `s3://${R2_BUCKET}/${R2_PREFIX}/ablations/k-size/k{K}_overlap{T|F}/`

### Single run

```bash
python train/ablations/k-size/train_k_ablation.py --k 100 --num-epochs 30
```

### Output

Each run writes to:
```
checkpoints/ablations/k-size/k{K}_overlap{T|F}/
    ├── best_model.pth
    ├── training_history.json
    ├── training_curves.png
    └── config.json
```

### Compare results

```bash
for d in checkpoints/ablations/k-size/*/; do
    name=$(basename "$d")
    sisnr=$(jq -r .best_si_snr "$d/config.json" 2>/dev/null || echo "?")
    echo "$name: $sisnr dB"
done
```

## Notes

- Ablation runs are **short (30 epochs)** by default — for ranking only.
  Promote the winner to a full 100-epoch run before reporting in the thesis.
- All ablation runs share dataset, seed, all other hyperparameters with the
  baseline. Only K (and optionally seg_overlap) varies.
- Scope: 2-speaker baseline SkiM only. No 3-speaker, no attention variants,
  no transfer learning. Justification: the K choice originates from the
  baseline; if K is the bottleneck, fixing it on the baseline first
  establishes the right value before re-applying everywhere.
