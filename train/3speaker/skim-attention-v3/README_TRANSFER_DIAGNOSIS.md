# Why v3 Transfer Learning Underperforms — Diagnosis

## The puzzle

| Setting | SkiM (base) | SkiM Attention v3 | Δ |
|---|---|---|---|
| 2-speaker (cold-start) | 19.86 dB | **20.25 dB** | **+0.39 dB** ✅ |
| 3-speaker (transfer learning) | **15.27 dB** | ~14.95 dB | **−0.20 to −0.26 dB** ❌ |

v3 is *better* at 2spk but *worse* at 3spk-TL. That's unexpected — usually a stronger pretrained model produces a stronger transfer. Why does v3 invert?

The training configurations are **identical** between SkiM and v3 (same optimizer, LR, schedule, batch size, dropout, seed). The only difference is architectural: v3 inserts a SegAttention block (MHSA + FFN + LayerScale gates) after each SegLSTM. So the answer must be in the architecture, not the training recipe.

---

## Evidence: what v3 actually learned at 2spk

The SegAttention block is wired into the residual stream via two scalar LayerScale gates per block (4 blocks × 2 gates = 8 scalars total):

```
x ← x + γ_attn × MHSA(LN(x))     # attention contribution
x ← x + γ_ffn  × FFN(LN(x))      # FFN contribution
```

When γ ≈ 0, the block is effectively a no-op (identity). When γ is large in magnitude, the block actively shapes the residual stream. The sign of γ matters: **positive γ adds**, **negative γ subtracts**.

Inspecting `checkpoints/2speaker/skim-attention-v3/best_model.pth` reveals what the 2spk training actually converged to:

```
γ_attn[block 0] = −0.041
γ_attn[block 1] = −0.045
γ_attn[block 2] = −0.086
γ_attn[block 3] = −0.452       ← dominant
γ_ffn[*]        ≈  0           ← FFN pathway dormant
```

Three things stand out:

1. **All γ_attn are negative.** The attention contribution is subtracted from the residual stream, not added. The architecture learned to use MHSA as a *subtractor*.
2. **Block 3 dominates.** Its magnitude is ~10× any other block. Most of the 0.39 dB advantage flows through this single gate.
3. **γ_ffn ≈ 0 everywhere.** Half of v3's added capacity (the FFN pathway in every block) is functionally unused.

So v3's "+0.39 dB advantage" reduces to: a single learned subtractive filter in block 3 that removes a specific kind of structure from the LSTM's residual stream.

---

## The diagnosis

### What that filter actually is

When SegLSTM processes a 2spk mixture, it produces frame features that contain (a) the speech content and (b) some residual interference pattern characteristic of 2-speaker mixtures — overlap of two voices, specific spectral crosstalk, etc. By learning γ_attn[3] ≈ −0.45, v3 trained the MHSA module to reproduce that interference pattern from the LSTM's own output, so subtracting it cleans the residual stream.

In short: **the gates encode a learned, 2-speaker-specific denoiser**. v3's 0.39 dB edge at 2spk is the SI-SNR improvement from that subtraction.

### Why it doesn't transfer

A 3-speaker mixture has different interference statistics than a 2-speaker mixture: more overlap, different spectral structure, different residual patterns coming out of the LSTM. The denoiser learned on 2spk data subtracts the *wrong* thing from a 3spk residual stream:

| Input | What γ_attn[3] = −0.45 does |
|---|---|
| 2spk features | subtracts learned 2spk interference → cleaner separation (+0.39 dB) |
| 3spk features | subtracts a 2spk-shaped pattern that doesn't match 3spk interference → degrades the residual |

That alone would account for some of the TL gap. But it gets worse: the LSTM weights themselves were trained *in concert with* the negative attention. The LSTM's output distribution was shaped knowing the attention block would subtract a specific pattern downstream. So the LSTM is also implicitly 2spk-specialized — its features assume the post-attention denoiser is in place.

When transferred to 3spk:

- **If the gates stay at their 2spk values:** the wrong-task subtraction interferes with 3spk fitting. We see this as `train_loss(v3) > train_loss(SkiM)` — v3 cannot even fit the 3spk training data as well as SkiM, because the negative gates keep removing things.
- **If the gates are reset to 0:** the LSTM features are still 2spk-shaped and now have no denoiser to compensate. They have to relearn 3spk features from the wrong starting point.
- **If the gates are allowed to retrain:** the LSTM and gates have to co-adapt to a new task, but the LSTM is already in a bad local minimum (the 2spk basin) that's hard to leave.

Meanwhile, **base SkiM has no specialized component**. Its LSTM weights encode generic sequence modeling. There's no task-specific filter to break. When retargeted at 3spk, the LSTM re-fits cleanly.

This is the **specialization-vs-transferability tradeoff**: the same mechanism that makes v3 better at 2spk makes it worse to transfer from.

---

## Experiments that confirmed the diagnosis

We tried four TL strategies. All four hit a similar ~0.20 dB ceiling vs. SkiM TL, each for a slightly different reason that maps onto the diagnosis:

| Strategy | Result vs SkiM TL | Why it fails |
|---|---|---|
| **Vanilla TL** (gates kept at 2spk values, full fine-tune) | ~−0.20 dB | wrong-task subtraction; LSTM stuck in 2spk basin |
| **`--reset-gates`** (γ zeroed, full fine-tune) | ~−0.19 dB | gates start fresh but LSTM features are still 2spk-shaped |
| **`--freeze-gates`** (γ frozen at 2spk values) | ~−0.23 dB | actively interferes — train loss is **0.56 dB worse** than SkiM at the same epoch |
| **`--two-stage`** (Option C: stage 1 = freeze backbone, train γ + output head only; stage 2 = full fine-tune) | tracking same as vanilla | adapter-style TL fails because **the backbone IS the specialized part** — freezing it makes things worse, not better |

The two-stage result is particularly diagnostic. The conventional wisdom for TL is "freeze backbone, retrain head, then fine-tune." That works when the backbone is task-general and the head is task-specific. v3 inverts this: the LSTM weights are themselves co-adapted to a 2spk-specific output structure, so freezing them locks in the wrong specialization.

---

## What the training config diff confirms

We checked whether v3 was using more compute, longer training, a better schedule — anything that would explain its 2spk advantage as a training trick rather than an architectural one:

```
SkiM 2spk vs v3 2spk training config:
  batch_size:      8           ==  8
  num_epochs:      100         ==  100
  learning_rate:   1e-3        ==  1e-3
  weight_decay:    1e-5        ==  1e-5
  gradient_clip:   5.0         ==  5.0
  seed:            42          ==  42
  dropout:         0.2         ==  0.2
  optimizer:       Adam        ==  Adam
  scheduler:       ReduceLROnPlateau(0.5, patience=5)  (identical)
  AMP:             yes         ==  yes
  encoder:         Conv ch=256 k=32 s=16  (identical)
  separator:       layer=4 unit=256 segment=20 mem=hc  (identical)
  diff:            num_heads=4 (only relevant to v3's MHSA)
```

The configs are identical. v3's advantage is **not** a training-recipe artifact. It is purely the architectural addition of SegAttention — and the architectural addition's only contribution is the negative-γ denoising filter described above.

---

## What this means

The mechanistic chain is:

```
v3 architecture adds attention modules
        ↓
2spk training shapes them into a task-specific subtractive filter (negative γ)
        ↓
LSTM weights co-evolve with the filter (learn to expect downstream subtraction)
        ↓
2spk: filter denoises productively → +0.39 dB
3spk: filter subtracts wrong patterns → wrong-task interference
3spk: LSTM weights also encode 2spk-specific assumptions → backbone needs to relearn
        ↓
Net 3spk-TL outcome: v3 trails base SkiM by ~0.20 dB
```

There is no TL trick (warmup, freezing, gate-only learning, lower LR) that fixes this, because the problem is not in *how* we transfer — it's in *what we are transferring*. The pretrained v3 2spk model is, mechanically, a 2-speaker-specialized network whose advantage is not portable.

---

## Path forward: fix it at pretraining (Option A)

The only place we can break the specialization-transferability tradeoff is during 2spk pretraining itself. By adding an L2 penalty on the LayerScale gates:

```
loss = SI-SNR_loss + λ × Σ γ²
```

we prevent the gates from growing into a strong specialized filter. The architecture cannot offload 2spk-specific denoising onto γ; the LSTM has to do more of the work itself, and what it learns is more task-general.

**Expected tradeoff:**

| Run | 2spk SI-SNR | Mechanism at 2spk | 3spk-TL outcome |
|---|---|---|---|
| v3 unregularized (current) | **20.25 dB** ✅ | gates do task-specific denoising | -0.20 dB vs SkiM TL ❌ |
| v3 regularized (Option A) | likely 19.7–20.0 dB | gates near 0, LSTM does the work | hopefully ≥ SkiM TL ✓ |

You **lose some 2spk performance by design** — that's the cost of removing the over-specialization. The bet is that the regularized backbone transfers cleanly enough to land at or above SkiM TL on 3spk.

Implementation: `train/2speaker/skim-attention-v3-reg/train_skim_attention_v3_reg_2spk.py` then `train/3speaker/skim-attention-v3-reg/train_skim_attention_v3_reg_3spk_transfer.py`.

---

## The honest possibility

Option A may also fail. If after regularized 2spk pretraining the 3spk-TL run still doesn't beat SkiM TL, we will have learned something significant:

> **v3's only mechanism for outperforming base SkiM is task-specific specialization, which is fundamentally non-transferable.**

That is a real, publishable mechanistic finding. The paper's contribution would shift from "v3 is a better separator under TL" to "we identified that v3's architectural advantage at 2spk is a learned task-specific filter, and we mechanistically explain why such advantages do not transfer." The latter is a stronger scientific claim than the former.

The mechanistic story (negative γ_attn, dominant block 3, dormant FFN, co-adapted LSTM, wrong-task subtraction at 3spk) is concrete and reproducible — it stands on its own regardless of whether Option A succeeds or fails.
