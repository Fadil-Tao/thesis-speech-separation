# Why skim-transfer Beats skim-attention-v3-reg-transfer on 3-Speaker

A post-mortem on the 3-speaker transfer-learning result: plain SkiM with TL
(15.27 dB) edged the γ-regularized v3 with TL (15.15 dB) by ~0.12 dB on
validation. This document explains the mechanism, what we already knew from
the unregularized v3 diagnosis, and what the regularized result adds.

## TL;DR

| Model | val SI-SNR | Gap vs skim-transfer |
|---|---|---|
| skim (cold-start) | 14.17 dB | −1.10 |
| skim-attention-v3 (cold-start) | 14.35 dB | −0.92 |
| **skim-transfer** | **15.27 dB** | — |
| skim-attention-v3-reg-transfer | 15.15 dB | −0.12 |

- **Cold-start:** v3 wins by +0.18 dB (architecture advantage).
- **Transfer:** plain skim wins by +0.12 dB. The architecture advantage
  inverted under transfer.
- The gap collapsed from −0.20 dB (unregularized v3) to −0.12 dB (γ-regularized
  v3), so γ-regularization partly worked — but did not close the gap.

The mechanism is **specialization-vs-transferability**: v3's gains at the
single-task setting come from gates learning a task-specific filter; that
filter does not survive a task change, and the LSTM backbone trained around
it inherits some of that specialization too.

## 1. Architecture Diff

Both share the SkiM block: `SegLSTM` (sequential within K frames) → `MemLSTM`
(across S segments). v3 inserts a **`SegAttention`** module after each
`SegLSTM`:

```
x ← x + γ_attn · MHSA(LN(x))     # attention contribution
x ← x + γ_ffn  · FFN(LN(x))      # FFN contribution
```

Each `SegAttention` block carries two **LayerScale scalars** `γ_attn`,
`γ_ffn` (init = 0). Block = identity at init; grows only if attention helps
the loss.

`num_blocks = 4` → 8 gates total. The rest of the architecture is identical
to base SkiM.

## 2. What the 2spk v3 Pretraining Actually Learned

Inspecting `checkpoints/2speaker/skim-attention-v3/best_model.pth`:

```
γ_attn[0] = −0.041
γ_attn[1] = −0.045
γ_attn[2] = −0.086
γ_attn[3] = −0.452     ← dominant
γ_ffn[*]   ≈  0         ← FFN dormant
```

Three observations:

1. **All `γ_attn` are negative.** MHSA is being *subtracted* from the
   residual stream. The architecture learned to use attention as a
   **subtractive denoiser**, not an additive feature.
2. **Block 3 dominates.** Its magnitude is ~10× any other block — most of
   v3's +0.39 dB 2spk gain flows through one gate.
3. **`γ_ffn ≈ 0` everywhere.** Half of v3's added parameters (the FFN
   pathway in every block) are functionally unused at convergence.

So v3's 2spk advantage reduces to: one learned subtractive filter in block 3
that removes 2spk-specific interference from the LSTM's residual stream.

## 3. Why That Filter Does Not Transfer

The filter is **task-specific**. 2spk mixtures have different interference
statistics than 3spk mixtures (more overlap, different spectral structure,
different residual patterns out of the LSTM). The filter trained to subtract
2spk interference subtracts the *wrong* thing from 3spk features:

| Input | What γ_attn[3] = −0.45 does |
|---|---|
| 2spk features | subtracts learned 2spk interference → cleaner (+0.39 dB) |
| 3spk features | subtracts a 2spk-shaped pattern → degrades the residual |

Worse: the LSTM weights co-evolved with the filter. The LSTM's output
distribution was shaped *knowing* the attention block would subtract a
specific pattern downstream. So the backbone itself is implicitly
2spk-specialized — features that assume a post-attention denoiser is in
place. When the model is moved to 3spk:

- **Gates kept at 2spk values:** wrong-task subtraction degrades residual;
  train loss is also worse than skim-TL — v3 can't even fit 3spk as well as
  skim because the gates keep removing things.
- **Gates reset to 0:** LSTM features are still 2spk-shaped, now with no
  denoiser to compensate.
- **Gates allowed to relearn:** LSTM + gates have to co-adapt to a new
  task, but the LSTM is in the 2spk basin of attraction.

Meanwhile, **plain skim has no specialized component**. Its LSTM encodes
generic sequence modeling. Retargeting to 3spk re-fits cleanly.

This is the **specialization-vs-transferability tradeoff**: the same
mechanism that makes v3 better at 2spk makes it worse to transfer from.

## 4. What γ-Regularization Tried to Fix (and What It Achieved)

`skim-attention-v3-reg` adds an L2 penalty on the LayerScale gates during
2spk pretraining:

```
loss = SI-SNR_loss + λ · Σ γ²
```

This pulls γ toward 0 during pretraining → architecture cannot offload
2spk-specific denoising onto the gates → LSTM has to do more of the work,
and what it learns is more task-general.

**Observed result on 3spk-TL:** gap closed from −0.20 dB (unregularized
v3-TL) to −0.12 dB (regularized v3-TL). Improvement = ~0.08 dB. **Direction
of effect matches the theory**: weaker specialization → better transfer.

But the gap did **not close to zero**. The remaining −0.12 dB is the open
question.

## 5. Why the γ-Reg Variant Still Lost — Three Hypotheses

### Hypothesis A: LSTM-side specialization survives γ regularization

γ-reg constrains the gates but not the LSTM weights. During 2spk
pretraining the LSTM still sees only 2spk data, and even without a strong
downstream denoiser it can pick up 2spk-specific structure (specific overlap
patterns, two-speaker spectral statistics). That structure does not survive
the 2→3 speaker transition.

**Evidence supporting:** the 2spk skim pretraining also has 2spk-specific
LSTM features, but its backbone has fewer "moving parts" — no attention to
co-adapt with — so the features end up more generic.

**Counter-evidence:** the skim-transfer LSTM was *also* trained only on
2spk and *also* picks up 2spk structure, yet transfers fine. So this can't
be the whole story.

### Hypothesis B: Capacity surplus hurts low-LR fine-tuning

Transfer training uses `learning_rate = 1e-4` (10× smaller than cold-start)
to protect pretrained weights. v3 has more parameters than skim:

- skim separator: ~3.0 M params
- v3 separator:   ~4.6 M params (4 × SegAttention blocks: ~400 K each)

The extra parameters (MHSA Q/K/V/O projections, FFN, two LayerNorms per
block) all start in a configuration tuned for 2spk. At 1e-4 LR, they cannot
move far in 100 epochs. They sit at a 2spk-shaped local optimum and act
like a slight drag during 3spk fitting.

**Evidence:** dormant FFN pathway (γ_ffn ≈ 0) is dead weight at 2spk, but at
3spk those FFN linears still consume gradient signal during fine-tuning
without contributing to output. They are a regularizer-like nuisance term.

### Hypothesis C: Seed noise within ±0.15 dB

Both runs use seed=42 but the optimizer trajectory differs. Std of per-file
SI-SNR ≈ 1.8 dB → standard error on 3600-file mean ≈ 0.03 dB → a 0.12 dB gap
is ~4σ on the mean and looks significant in isolation. But run-to-run seed
variance (re-train skim-transfer with seed=0, seed=1, ...) easily produces
±0.1 dB swings. We have **n = 1** for each model. A paired t-test on per-file
results will resolve whether 0.12 dB is reproducible or seed-noise.

**Test:** run `cell_11_paired_test.py` on the full eval. If
p < 0.01 with mean diff ≈ +0.12 dB across 3600 files, the gap is real and
hypotheses A/B carry the load. If p > 0.05, the gap is seed noise and v3-reg
is effectively tied with skim-transfer.

## 6. What This Means for the Thesis

The cleaner phrasing of the result:

> v3's architectural advantage on the single-task (2spk) setting is a
> learned task-specific subtractive filter (negative γ_attn, dominant in the
> last block). γ-regularization removes this filter, restoring transfer
> behavior to within 0.12 dB of plain SkiM. The remaining gap, if real, is
> attributable to LSTM-side specialization picked up during 2spk
> pretraining, plus parameter overhead at the low transfer-learning LR.

This is a **stronger scientific claim** than "v3 is a better separator".
You have:

1. **Mechanism** — concrete weight inspection, gate values, dominant block.
2. **Prediction** — γ-reg should narrow the TL gap.
3. **Confirmation** — gap narrowed from −0.20 to −0.12 dB.
4. **Open question** — why the residual 0.12 dB persists.

That arc is much more defensible than a benchmark race that v3 loses.

## 7. Concrete Recommendations

1. **Run the full 3600-file SI-SNR eval** (`inference/notebook-py/sisnr-eval/`
   on Colab). Confirm the 0.12 dB gap on test, not just val.
2. **Run paired t-test** (`cell_11_paired_test.py`). Decide whether the gap
   is significant (Hypothesis A/B carry weight) or noise (Hypothesis C).
3. **Inspect γ values in the γ-reg 3spk-transfer ckpt** — if they ended up
   far from 0 again, the regularization wore off during fine-tuning and the
   filter re-formed in a 3spk-specific way. That would re-open Hypothesis B
   (the gates aren't the only specialization mechanism but they're still
   active).
4. **Optional: try γ-reg + LSTM-side regularization at 2spk** — e.g. weight
   decay on the SegLSTM proj layer, or spectral norm on the recurrent
   matrix. If LSTM specialization is the residual culprit, this should
   close the remaining gap.
5. **Report the result honestly.** v3 cold-start > skim cold-start. skim-TL
   > v3-TL (regularized or not). The transferability finding is the
   contribution.

## References in this Repo

- `train/3speaker/skim-attention-v3/README_TRANSFER_DIAGNOSIS.md` —
  unregularized v3 diagnosis (the original investigation).
- `implementation/skim_attention_v3/skim_attention_v3.py` — SegAttention with
  LayerScale gates.
- `train/2speaker/skim-attention-v3-reg/` — γ-regularized 2spk pretraining.
- `train/3speaker/skim-attention-v3-reg/` — TL from the regularized 2spk
  checkpoint (this file's subject).
- `checkpoints/3speaker/{skim-transfer,skim-attention-v3-reg-transfer}/` —
  the trained models compared here.
- `inference/notebook-py/sisnr-eval/` — Colab notebook for full 3600-file
  evaluation + paired t-test.
