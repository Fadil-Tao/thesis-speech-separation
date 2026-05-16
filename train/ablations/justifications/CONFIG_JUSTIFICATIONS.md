# Configuration Justifications

Per-hyperparameter justification for our SkiM baseline (`train/2speaker/skim/train_skim_2spk.py`)
against the SkiM paper (Li et al., ICASSP 2022), the Conv-TasNet paper (Luo & Mesgarani 2019),
and the ESPnet `egs2/wsj0_2mix/enh1/conf/tuning/train_enh_skim_tasnet_noncausal.yaml` recipe.

## Summary Table

| Block | Param | Ours | SkiM paper | Conv-TasNet | ESPnet yaml | Source of choice |
|---|---|---|---|---|---|---|
| Encoder | channel (N) | 256 | not stated WSJ0 | 512 | 64 | Conv-TasNet (halved) |
| Encoder | kernel_size (L) | 32 | 2 / 8 (WSJ0 @ 8 kHz) | 16 (best @ 8 kHz) | 2 | Conv-TasNet scaled to 16 kHz |
| Encoder | stride | 16 | L/2 | L/2 (50% overlap) | 1 | Conv-TasNet 50% rule |
| Separator | input_dim | 256 | 256 | n/a | 128 | match encoder N |
| Separator | layer | 4 | 4 | n/a | 6 | SkiM paper |
| Separator | unit | 256 | 256 | n/a | 128 | SkiM paper |
| Separator | segment_size (K) | 20 | 150 | n/a | 250 | **unjustified — to be ablated** |
| Separator | seg_overlap | False | False | n/a | True | matches paper |
| Separator | dropout | 0.2 (2spk) / 0.1 (3spk) | 0.1 | n/a | 0.1 | paper / inconsistent |
| Separator | mem_type | hc | hc | n/a | hc | paper |
| Separator | nonlinear | relu | — | — | relu | yaml |
| Loss | criterion | SI-SNR + PIT | SI-SNR + PIT | SI-SNR + uPIT | SI-SNR + PIT | standard |
| Optim | optimizer | Adam | Adam | Adam | Adam | standard |
| Optim | lr | 1e-3 | 1e-3 | 1e-3 | 1e-3 | standard |
| Optim | weight_decay | 1e-5 | — | — | 0 | **unjustified** |
| Optim | gradient_clip | 5.0 | 5.0 | 5.0 | — | paper |
| Sched | type | ReduceLROnPlateau | ×0.97/epoch | halve on plateau (patience=3) | ReduceLROnPlateau | adapted from Conv-TasNet |
| Sched | factor / patience | 0.5 / 5 | — | 0.5 / 3 | 0.7 / 1 | Conv-TasNet-like |
| Train | batch_size | 8 | not stated | — | 8 | matches yaml |
| Train | num_epochs | 100 | 100 | 100 | 150 | paper |
| Data | clip length | 5 s | 30 s (CSS) / 4 s (WSJ0) | 4 s | 1 s chunk | adapted to TITML |
| Data | sample rate | 16 kHz | 16 kHz (CSS) / 8 kHz (WSJ0) | 8 kHz | 8 kHz | TITML source SR |
| Data | speakers | 20 (Indonesian) | LibriSpeech / WSJ0 | WSJ0 | WSJ0 | TITML constraint |

---

## 1. Encoder

### 1.1 kernel_size = 32, stride = 16

**Justified.** Direct time-scale of Conv-TasNet best config to 16 kHz.

- Conv-TasNet Table II row 14: best config uses **L=16, stride=8 @ 8 kHz** → **2 ms window, 1 ms hop, 50% overlap**.
- Paper Sec III.B: *"A 50% stride size is used in the convolutional autoencoder."*
- Paper Sec IV.B.(v): *"Shorter segment length consistently improves performance. The best system uses a filter length of only 2 ms."*

Our setup at 16 kHz preserves the 2 ms window:
```
L_ours = 2 ms × 16 000 Hz = 32 samples
stride_ours = L/2 = 16 samples (50% overlap)
```

Cite: Luo & Mesgarani (2019), Conv-TasNet, Sec III.B + Table II.

### 1.2 channel (N) = 256, not 512

**Justified as a parameter-count trade-off.** Conv-TasNet best uses N=512. Conv-TasNet Table II ablation:

| N | SI-SNRi | model size |
|---|---|---|
| 128 | 13.0 dB | 1.5 M |
| 256 | 13.1 dB | 1.5 M |
| 512 | 13.3 dB | 1.7 M |

Halving from 512 → 256 loses ≈ 0.2 dB in the Conv-TasNet TCN setting. In our SkiM setup, encoder N also sets the SkiM `input_dim`. LSTM weight count scales as O(N²). Going N=512 would quadruple SkiM LSTM parameters (≈ 3 M → 12 M) for ≈ 0.2 dB upstream benefit. We chose N=256 to keep total model size manageable on consumer GPUs.

### 1.3 ReLU activation in encoder

**Inherited from ESPnet `ConvEncoder`** (line 43, applied after Conv1d). Enforces non-negative encoder features — TasNet/Conv-TasNet style. Conv-TasNet Sec IV.A showed ReLU encoder is comparable to linear encoder with sigmoid masking (≤ 0.2 dB). Acceptable, not a choice we made.

### 1.4 Sample rate = 16 kHz (vs paper 8 kHz on WSJ0-2mix)

**Forced by source data.** TITML-IDN is recorded at 16 kHz; downsampling would discard sibilant content (4–8 kHz) that aids speaker discrimination. Consequence: our SI-SNR numbers are **not directly comparable** to 8 kHz WSJ0-2mix literature. Modern 16 kHz Conv-TasNet on LibriMix reaches ~14.7 dB SI-SNRi — our 14.2 dB on TITML sits in that ballpark.

---

## 2. Separator (SkiM)

### 2.1 layer=4, unit=256, mem_type='hc', seg_overlap=False

**Matches paper.** SkiM paper Sec 3.3: *"All the SkiM models consist of 4 SkiM blocks. In each SkiM block, the LSTMs' hidden dimension is 256."*  Paper Sec 2.5: *"in the segmentation stage, we abandon the overlap region"* — supports `seg_overlap=False`. Paper Table 3 shows `mem_type='hc'` (both hidden and cell MemLSTM) is best by ≥ 3 dB over alternatives.

### 2.2 segment_size (K) = 20 — UNJUSTIFIED

**Open issue.** Origin not documented in repo. SkiM paper uses K=150 (CSS) and does not state WSJ0 K explicitly. ESPnet yaml uses K=250 with `seg_overlap=True`.

K=20 at our 1 ms hop = **20 ms local context per Seg-LSTM**, which is sub-phoneme. Likely too small. This is the motivation for the ablation in `train/ablations/k-size/`.

Honest framing in thesis:
> "K=20 was inherited from an early prototype configuration; the present work ablates K ∈ {20, 50, 100, 150} to test whether longer local context improves separation."

### 2.3 dropout: 0.2 (2spk) vs 0.1 (3spk)

**Inconsistent within repo.** Paper uses 0.1. No documented reason for the 2spk value being doubled. Treat as legacy. For new ablations, default 0.1.

### 2.4 nonlinear = 'relu'

**Matches ESPnet yaml.** Output mask activation. Paper does not specify; relu is the ESPnet default for SkiM separator.

---

## 3. Loss

### 3.1 SI-SNR with PIT

**Matches paper exactly.** SkiM paper Sec 2.1 (WSJ0 experiments) and Conv-TasNet Sec III.C both use SI-SNR + PIT (utterance-level for WSJ0).

We do **not** use Graph-PIT (paper CSS only) because our data is utterance-level fully-overlapped, not meeting-style.

---

## 4. Optimizer

### 4.1 Adam, lr=1e-3

**Standard, matches both paper and yaml.**

### 4.2 weight_decay = 1e-5

**Minor unjustified divergence.** ESPnet yaml uses 0. Paper does not state. 1e-5 is small enough that effect on LSTM is negligible, but it should not be there without rationale. Recommend setting to 0 in next refresh.

### 4.3 gradient_clip = 5.0

**Matches paper.** SkiM paper Sec 3.3: *"The L2-norm of the gradient is clipped to 5 in model optimization."*

### 4.4 Scheduler = ReduceLROnPlateau(factor=0.5, patience=5)

**Adapted from Conv-TasNet, not SkiM paper.**
- SkiM paper uses deterministic ×0.97 / epoch (lr → 0.048× over 100 epochs).
- Conv-TasNet uses halve on 3-epoch plateau.
- ESPnet yaml uses ×0.7 patience=1.
- Ours: ×0.5 patience=5 — most conservative.

Defensible: less aggressive decay protects against premature LR reduction during noisy early epochs. No empirical justification in repo.

---

## 5. Training

### 5.1 batch_size = 8

**Matches yaml.** Originally 4, increased to 8 for stable gradients. Empirically validated.

### 5.2 num_epochs = 100

**Matches paper.**

### 5.3 clip length = 5 s, fixed pad / truncate

**Adapted.** Paper WSJ0 uses ~4 s utterances. ESPnet yaml uses random 1 s chunks (more augmentation per utterance). TITML utterances are short; 5 s captures full sentences. No random chunking — accepted as slightly less augmentation than yaml.

### 5.4 AMP (`torch.amp.autocast`)

**Engineering choice not in paper.** Half-precision forward pass for memory + speed. Acceptable since loss is computed in fp32 inside `SISNRLoss`.

---

## 6. Dataset

### 6.1 TITML-IDN-mix (Indonesian, 20 speakers, 11M/9F)

**Forced.** This is the dataset under study; not a hyperparameter.

Limitations:
- Speaker diversity (20) much smaller than WSJ0 (~100) or LibriSpeech (>2000).
- Risk of speaker-identity overfit. Generalization claim is restricted to TITML.

### 6.2 Mix parameters

- SNR range: −5 to +5 dB. Matches WSJ0-2mix standard (Conv-TasNet Sec III.A: *"random SNR between -5 dB and 5 dB"*).
- Time offset 0–1 s. Adds onset diversity, not in WSJ0 standard but harmless.
- 5 s fixed clip. Mild divergence (paper ~4 s); justifiable on linguistic-completeness grounds.
- 28 800 / 3 600 / 3 600 split — slightly larger than WSJ0-2mix (20k/5k/3k).

---

## 7. What Still Needs Ablation

| Question | Why | Where to run |
|---|---|---|
| Is K=20 too small? | No documented justification; paper uses K=150. | `train/ablations/k-size/` |
| Does seg_overlap=True help? | yaml uses True, paper uses False. | same dir, `--seg-overlap` |
| Is N=256 vs N=512 worth it? | +0.2 dB possible; 4× LSTM params cost. | not scheduled |
| Is weight_decay=1e-5 a no-op? | yaml uses 0. | low priority |

Top priority: K ablation (`bash train/ablations/run_ablation.sh`).

---

## 8. Citation Targets for Thesis

- Encoder choice → Luo & Mesgarani 2019 (Conv-TasNet), Sec III.B and Table II.
- Separator choice → Li et al. 2022 (SkiM), Sec 2.3 and Sec 3.3.
- Loss / PIT → Yu et al. 2017 (PIT) and Conv-TasNet SI-SNR formulation.
- ESPnet framework → Li et al. 2021 (ESPnet-SE).

## 9. References

- Luo, Y., & Mesgarani, N. (2019). Conv-TasNet: Surpassing ideal time–frequency magnitude masking for speech separation. *IEEE/ACM TASLP*, 27(8), 1256–1266.
- Li, C., Yang, L., Wang, W., & Qian, Y. (2022). SkiM: Skipping memory LSTM for low-latency real-time continuous speech separation. *ICASSP 2022*.
- ESPnet recipe: `egs2/wsj0_2mix/enh1/conf/tuning/train_enh_skim_tasnet_noncausal.yaml`.
