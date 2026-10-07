# Experiment plan and compute budget

This is the plan for experiments A to E. The budget is based on speeds measured by
the smoke test on Kaggle's 2x T4. Results go in `REPORT.md` as runs finish. Any
number below marked "estimate" is a plan, not a result.

## 1. What the smoke test measured

The smoke test (`results/smoke-2` on the `results` branch, 7 October) ran the whole
pipeline on a tiny slice of data and passed every step in 12.5 minutes.

**The first attempt failed.** The first smoke run (`results/smoke`) hung in its data
step until Kaggle stopped it 12 hours later. That cost about 12 of the 30 weekly GPU
hours. Cause: the data step started worker processes by forking (copying) a process
that had live background threads from the Hugging Face download. The copies
deadlocked. The fix starts workers fresh ("spawn"), raises an error if a worker dies
or hangs, and the runner now stops any job that prints nothing for 30 minutes or
passes its own time limit.

**Hardware and software:** 2x Tesla T4 (15 GB each), PyTorch 2.10, CUDA 12.8,
Python 3.12, 4 CPU cores, 31 GB RAM. All-reduce between the two GPUs (summing
gradients across GPUs) runs at about 4 GB/s with default settings.

**Exact resume on GPU.** On CPU, the tests show that stopping and resuming gives
bit-identical losses. On the T4s it is not bit-identical, and it can't be with
default settings: some GPU kernels add numbers in a varying order, so two identical
runs from scratch already differ. In 200 steps:

| Comparison | Largest loss difference |
| --- | --- |
| Two identical runs, no stop | 0.0137 |
| Straight run vs. run stopped at step 100 and resumed from the Hub | 0.0121 |

The resume adds no difference beyond this run-to-run noise. The learning rate
and loss scale match exactly at every step after the resume.

**Speed.** Measured with fp16, both GPUs and DDP (DistributedDataParallel, one model
copy per GPU), with sequence length 1024. MFU (model FLOPs utilization) is the
share of the GPU's peak math speed that goes into useful model math. It is
measured against the T4's fp16 peak of 65 TFLOPS.

| Model | d_model x layers | Params (total / non-embedding) | tok/s, eager | tok/s, compiled | Compile speedup | MFU, compiled |
| --- | --- | --- | --- | --- | --- | --- |
| s1 | 128 x 4 | 2.9M / 0.8M | 243,209 | 624,696 | 2.57x | 11.4% |
| s2 | 256 x 6 | 8.9M / 4.7M | 134,930 | 269,650 | 2.00x | 14.9% |
| s3 | 384 x 8 | 20.5M / 14.2M | 80,492 | 141,151 | 1.75x | 17.4% |
| s4 | 512 x 8 | 33.7M / 25.3M | 57,794 | 100,873 | 1.75x | 19.6% |
| s5 | 640 x 10 | 59.4M / 49.0M | 36,581 | 59,452 | 1.63x | 19.9% |
| s6 | 768 x 12 | 97.5M / 85.0M | 24,522 | 37,324 | 1.52x | 20.1% |

Other measurements, all on s4 without compile:
- **fp16 vs fp32:** 57,794 vs 21,442 tok/s, so fp16 is 2.7x faster. In fp32 the MFU is
  33% of the T4's much lower fp32 peak (8.1 TFLOPS). The fp16 tensor-core peak is
  hard to reach on a T4, because the card is limited to 70 W and slows its clock
  under load.
- **1 vs 2 GPUs:** 33,188 vs 57,794 tok/s. That's 1.74x, or 87% of perfect scaling.
- `torch.compile` also lowers peak memory (s4: 5.7 GB vs 9.2 GB). Compiling takes
  about 30 seconds per run.

Decision: every experiment run uses fp16, both GPUs and `torch.compile`.

**Data and evaluation.** The data step cleaned one FineWeb-Edu file: 18,350
documents, 3 removed for broken characters, and 0 exact duplicates. FineWeb-Edu is
already deduplicated, so that is expected. The text came out at 4.28 bytes per
token. The tiny smoke model (7.3M params, 13M tokens) reached validation loss 6.02
(1.99 bits per byte), and its samples are word salad, as expected that early.

**Bugs the smoke test found (both fixed):**
1. HellaSwag could not be scored: the original GitHub file now returns 404. It is now
   downloaded from the Hugging Face copy at a fixed revision, and the code checks
   that all 10,042 examples are there.
2. The data manifest counted UTF-8 bytes as "characters kept", so it showed more
   characters kept than read.

## 2. Compute budget

Kaggle gives about 30 GPU hours a week. As far as I know it counts hours the
session is open, so 2x T4 costs the same as one GPU. (The quota bar in the
notebook editor shows the real number.) The runner trains for about 10.5 hours
per session.

Estimated hours = training tokens / compiled tok/s x 1.08 (evaluation, per-layer
stats and checkpoints), plus 4 minutes per run (start-up, compile, final
evaluation, upload).

| Experiment | Runs | Estimated GPU hours |
| --- | --- | --- |
| A. Scaling: LR sweeps (s1, s2, s3), ladder s4, s5, s6, IsoFLOP check | 12 + 3 + 3 | 31 |
| B. Ablations at s3, 2 seeds | 11 new (the baseline seed 1 is reused from A) | 10.5 |
| C. Stability: LR sensitivity at s2, base vs. QK-norm vs. z-loss | 8 | 2.2 |
| D. Efficiency: benchmark, batch-size study, fp32 training run | 1 bench + 4 | 1.8 |
| E. Evaluation: built into every run, plus a GPT-2 reference | 1 | 0.3 |
| **Total** | | **about 46** |

That's about 1.5 weeks of quota. In calendar time it's more like 2 to 3 weeks,
because each stage waits for the results of the one before it.

**Largest model: keep it at about 100M (s6, 97.5M parameters).** It is the most
expensive single run: about 16 hours, so two Kaggle sessions with a resume in
between. That's about a third of the budget. It is still worth it:
- It extends the scaling fit to 1.5 orders of magnitude in total parameters (2.9M
  to 97.5M), or 2 orders in non-embedding parameters (0.8M to 85M). The largest
  point pins down the fitted exponent more than any other.
- The dataset (2.6B training tokens) covers its 1.95B tokens without repeating any.
- A bigger model, say 150M, would need about 3B tokens (a bigger data job) and
  about 35 hours: more than a week of quota for one point.
- If the quota gets tight, s6 can drop to 10 tokens per parameter (about 8 hours).
  It would then be reported as off the ladder, not as a ladder point.

## 3. Shared settings

Unless a run changes them:
- **Model:** RoPE, RMSNorm, pre-norm, SwiGLU, tied input/output embeddings, no bias,
  no dropout, head size 64, context 1024, vocab 16,384.
- **Optimizer:** AdamW (beta1 0.9, beta2 0.95, weight decay 0.1), gradient clipping
  at 1.0, warmup for 5% of steps, then cosine decay to 10% of the peak LR.
- **Precision and speed:** fp16 with a loss scaler, `torch.compile`, 2 GPUs.
- **Batch:** 64k tokens per step for s1 and s2, 128k for s3 and s4, 256k for s5 and
  s6. Small models train on few tokens, so they need small batches to get enough
  optimizer steps.
- **Training length:** 20 tokens per parameter (total parameters), Chinchilla's
  rule of thumb for the best loss at a fixed compute budget.
- **Evaluation:** validation loss about 20 times per run (1M tokens each). At the
  end: loss, perplexity and bits per byte on the full validation set (up to 20M
  tokens), zero-shot HellaSwag (all 10,042 examples), and sample texts.

The run configs live in `configs/exp/`. `scripts/make_configs.py` writes them from
one table, and a test checks they are up to date.

## 4. The experiments

### A. Scaling laws

**Question:** how does validation loss fall as the model gets bigger, when each size
gets 20 tokens per parameter? How does that compare with Chinchilla?

**Learning rate first.** The best learning rate depends on model size, and a badly
tuned size would bend the curve. So the three smallest sizes get a sweep with
factor-of-2 steps:

| Size | Learning rates | Estimated hours per run |
| --- | --- | --- |
| s1 | 1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2 | 0.1 |
| s2 | 1e-3, 2e-3, 4e-3, 8e-3 | 0.27 |
| s3 | 1e-3, 2e-3, 4e-3 | 0.94 |

For each size I fit a parabola to loss vs. log(LR) to find the best LR. Then I fit
log(best LR) against log(params) and extend that line to s4, s5 and s6. If the
best LR sits at the edge of a grid, I add a point. The best run of each sweep is
that size's point on the ladder, so no run is wasted. If the line is unclear, one
extra s4 run at a second LR (2 hours) checks it.

**Ladder:** s1 to s6, at their best LRs.

**Fit:** L(N) = E + A / N^alpha, where L is the final validation loss, N the
parameter count, and E the loss an infinitely large model would reach. The fit
is done twice, with N = total parameters and N = non-embedding parameters. With a
16k vocabulary the embedding is 72% of s1 but only 13% of s6, so the choice of N
changes the curve. Kaplan et al. used non-embedding parameters and Chinchilla used
total parameters; this is one reason their answers differ.

**Chinchilla check (IsoFLOP).** Hold the compute fixed at what s3 used,
C = 6 x N x D = 5.0e16 FLOPs (N parameters, D training tokens), and train other sizes
on whatever number of tokens that compute buys:

| Model | Tokens | Tokens per parameter | Estimated hours |
| --- | --- | --- | --- |
| s2 | 943M | 106 | 1.1 |
| s3 | 409M | 20 | (the ladder run) |
| s4 | 248M | 7.4 | 0.8 |
| s5 | 141M | 2.4 | 0.8 |

If Chinchilla's rule holds at this tiny scale, s3 (20 tokens per parameter) should
be at or near the bottom of the curve. If the bottom is clearly elsewhere, that is
a finding. (Each run decays its learning rate over its own length, which matters
for this comparison.)

**Not comparable:** absolute loss values against Chinchilla or GPT-2. The data and
tokenizer differ. Exponents and the best tokens-per-parameter ratio can be compared;
raw losses cannot. Bits per byte can be compared across tokenizers on the same text,
which is what experiment E uses.

### B. Architecture ablations

**Question:** which of the modern choices actually help at this size?

All runs use s3 (20.5M parameters, 409M tokens) at its best LR from A. Each
variant changes one thing:

| Variant | Change |
| --- | --- |
| baseline | RoPE, RMSNorm, pre-norm, SwiGLU, warmup |
| learned-pos | learned position embeddings instead of RoPE |
| layernorm | LayerNorm instead of RMSNorm |
| post-norm | norm after each sub-layer (the original Transformer) instead of before |
| gelu | GELU MLP (4x width) instead of SwiGLU (8/3x width): the same parameter count to within 1% |
| no-warmup | learning rate starts at its peak |

Each variant runs with 2 seeds; the seed changes both the weight init and the
data order. That's 12 runs; baseline seed 1 is the s3 ladder run, so 11 are new.
Every run trains on exactly the same number of tokens.

**Reporting:** final validation loss for each seed and their mean, plus loss curves.
With 2 seeds there is no real statistics. I'll report a difference only if it is
larger than the gap between the two baseline seeds; anything smaller counts as
"no clear difference".

What the literature suggests (a guess, not a result): RoPE beats learned positions;
RMSNorm vs. LayerNorm matters mostly for speed; post-norm is worse and may need
warmup to train at all; SwiGLU is a bit better at matched size.

### C. Training stability

**Question:** what goes wrong when the learning rate is too high, can the logs show
why, and does a fix help?

This follows the method of Wortsman et al. (2023), "Small-scale proxies for
large-scale Transformer training instabilities": push small models to high
learning rates and watch for the same failures that hit large models.

Using s2 (16 minutes per run), extend the LR grid upward and compare three versions:

| Version | Learning rates |
| --- | --- |
| base | 1e-3 to 8e-3 (from A), plus 1.6e-2 and 3.2e-2 |
| QK-norm (queries and keys normalized before attention) | 2e-3, 8e-3, 1.6e-2, 3.2e-2 |
| z-loss 1e-4 (a small penalty that keeps the softmax normalizer near 1) | 1.6e-2, 3.2e-2 |

**Result:** final loss vs. learning rate for each version (an "LR sensitivity"
curve). A fix that works keeps the loss low over a wider range of learning rates.

**Diagnosis:** the high-LR runs log per-layer stats every 25 steps:
- the largest attention logit (the score before softmax)
- attention entropy (how spread out attention is)
- activation sizes
- gradient and weight norms
- loss-scaler skips

These can tell apart different failures. Attention logits growing until attention
collapses onto one token is what QK-norm should fix. Output logits drifting is what
z-loss should fix. On a T4 there is a third cause: fp16 overflow, which shows up as
skipped steps.

Diverged runs stop on their own after 100 bad steps in a row, so they cost little.

### D. Efficiency

**Question:** where does the speed come from, and what does it cost?

Already measured by the smoke benchmark: fp16 vs fp32, compile vs eager for every
size, and 1 vs 2 GPUs (s4, eager).

New:
- **Efficiency benchmark** (`configs/bench/efficiency.yaml`, about 20 minutes):
  - 1 vs 2 GPUs for every size with compile. Does DDP scale worse for small models,
    where the gradient sync is a bigger share of each step?
  - micro-batch size 4 to 32 on s4
  - micro-batch 16 on s6 (it ran at 8)
  - fp32 with compile
- **Batch size study:** s2 at 32k, 64k (ladder), 128k and 256k tokens per step, all on
  the same total tokens and at s2's best LR. Bigger batches run faster per token but
  give fewer optimizer steps. This shows where that starts to cost loss.
- **fp16 vs fp32 training:** one s2 run in fp32. Does fp16 with a loss scaler reach
  the same loss?

### E. Evaluation

Every run already records:
- validation loss, perplexity and bits per byte (bits per byte doesn't depend on the
  tokenizer)
- HellaSwag zero-shot accuracy (acc, and acc_norm, which divides by the ending's
  length in bytes); random guessing scores 25%
- sample texts for four fixed prompts

The report will show:
- loss and HellaSwag vs. model size and compute
- samples from s1 to s6 side by side
- an honest note: at under 100M parameters and 2B tokens, HellaSwag is expected to
  sit only a few points above chance, so validation loss is the more useful number
  at this scale

**Reference point (not built yet):** load the public GPT-2 small weights (124M) into
our own model class. It uses learned positions, LayerNorm, GELU and biases, which
the model already supports. Then score it with our own evaluation code:
- HellaSwag: a check that our scorer gives the known number for a known model
- bits per byte on our validation set: a fair comparison between our models and
  GPT-2 on the same text

This can run on a free CPU session.

## 5. Order of work

Each stage starts after I've read the results of the one before. You say "check
results", I update the queue, and you merge and run.

| Stage | Jobs | Estimated GPU hours | Status |
| --- | --- | --- | --- |
| 0 | Real dataset, `data-fineweb-edu-16k` (CPU session, free) | 0 | queued |
| 1 | `bench-efficiency`; LR sweeps s1, s2, s3 (12 runs) | 4.7 | queued |
| 2 | Ablations at s3 (11 runs) | 10.5 | waits for stage 1 (needs s3's LR) |
| 3 | s4, s5; IsoFLOP (3 runs); stability (8 runs); batch study and fp32 run (4 runs) | 14.6 | waits for stage 1 (needs the LR rule) |
| 4 | s6 (two sessions); GPT-2 reference | 16 | waits for stage 3 |

Results will be filled in as they come in. Until then, every result is "not run yet".
