# gptlab: a small GPT trained from scratch

This project trains small GPT-style language models (about 1M to 100M
parameters) from scratch and runs real experiments on them: scaling laws,
architecture ablations, training stability, speed, and evaluation.

Everything runs on free hardware: CPU tests in the cloud, and GPU training on
Kaggle (2x NVIDIA T4). The model, training loop and evaluation code are written
from scratch in PyTorch. Only small helper libraries are used (`tokenizers` for
fast BPE training, `huggingface_hub` for storage, numpy, pyarrow, matplotlib).

## Summary

Not run yet. Results, plots and the full write-up will appear here and in
`REPORT.md` once the runs have finished. Every number in this repo will come
from a run log on the `results` branch.

## What is built

**Data** (`gptlab/data/`)
- Source: [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu), `sample-10BT`.
  It is web text filtered by a classifier for educational value, and it is
  openly licensed (ODC-By). Its authors report that models trained on it score
  higher on knowledge and reasoning benchmarks (such as MMLU and ARC) than models
  trained on the same number of unfiltered web tokens. That matters when compute
  is tight. It is also already language-filtered and near-duplicate-filtered, so
  our own cleaning can stay light.
- The files store long runs of documents from the same web crawl. So we read
  blocks of about 1000 documents in a seeded random order, and every shard
  mixes all crawls.
- Cleaning: Unicode normalization, and removal of pages that are too short, too
  long, mostly non-letters, or broken (control or replacement characters). Exact
  duplicates are removed by hash. Every removal is counted with its reason and a
  few examples in `manifest.json`.
- Train/validation split by a hash of each document (0.5% to validation). The
  tokenizer is trained on training documents only, and the training loader
  refuses validation shards.
- Tokenizer: byte-level BPE with **16,384** tokens. (BPE, byte pair encoding,
  builds a vocabulary by repeatedly merging the most frequent pair of adjacent
  symbols, starting from the 256 bytes.) Why 16k and not 32k: with our model
  widths (128 to 768), a 32k vocabulary puts most of the smallest models'
  parameters and compute into the embedding and output layers. The data job
  also trains 8k, 16k and 32k tokenizers and measures bytes per token on
  held-out text, so the trade-off is shown with real numbers (not run yet).
- Tokens are stored as uint16 shards with a checked header.

**Model** (`gptlab/model.py`): a decoder-only transformer. The config can switch
positional encoding (learned or RoPE), norm type (LayerNorm or RMSNorm), norm
placement (pre or post), MLP type (GELU or SwiGLU, with matched parameter
counts), weight tying, and QK-norm. It uses GPT-2 style init with scaled
residual projections, and PyTorch's fused `scaled_dot_product_attention`.

**Training** (`gptlab/train.py`)
- AdamW, warmup plus cosine decay, gradient clipping, gradient accumulation,
  fp16 autocast with a loss scaler (T4 GPUs have no bfloat16), optional
  `torch.compile`, and DistributedDataParallel over 2 GPUs.
- Exact resume: model, optimizer, loss scaler, data position and random states
  are all saved. A test checks that N steps straight give bit-identical losses
  to N/2 steps, a stop, a resume in fresh objects, and N/2 more.
- Logs one JSON line per step: loss, learning rate, gradient norm, tokens/s,
  MFU, and memory. Every 100 steps it also logs per-layer stats: activation
  size, attention entropy, max attention logit, and gradient and weight norms.
- MFU (model FLOPs utilization) uses the 6N rule plus the attention term, with
  real peak numbers per GPU. See `gptlab/flops.py`. The formula is checked
  against PyTorch's own FLOP counter in the tests.

**Evaluation** (`gptlab/evaluate.py`, `gptlab/hellaswag.py`): validation loss,
perplexity, bits per byte (comparable across tokenizers), zero-shot HellaSwag
with our own scoring code (random chance is 25%), and sample generations.

**Kaggle runner** (`gptlab/runner/`, `kaggle/runner.ipynb`): one notebook that
works through `runs/queue.yaml`. It resumes jobs across sessions, uploads
checkpoints to the Hugging Face Hub, pushes logs to the `results` branch, and
stops cleanly before Kaggle's time limit. See [RUNNING.md](RUNNING.md).

## Experiments

| Experiment | Status |
| --- | --- |
| Smoke test and speed benchmark | done: whole pipeline passed on 2x T4; speeds in `EXPERIMENTS.md` |
| A. Scaling (5 to 6 sizes, power-law fit, comparison with Chinchilla) | not run yet |
| B. Ablations (RoPE, RMSNorm, pre/post norm, SwiGLU, warmup; 2 seeds) | not run yet |
| C. Stability (too-high learning rate, diagnosis, fix) | not run yet |
| D. Efficiency (fp16, compile, DDP, batch size) | not run yet |
| E. Evaluation (loss, perplexity, HellaSwag, samples) | not run yet |

The plan, the compute budget (about 46 GPU hours) and the measured speeds are in
[EXPERIMENTS.md](EXPERIMENTS.md).

## Repo layout

| Path | What it holds |
| --- | --- |
| `gptlab/` | The Python package: model, training loop, evaluation, FLOPs, checkpoints |
| `gptlab/data/` | Download, cleaning, tokenizer, token shards, data loader |
| `gptlab/runner/` | The job runner that the Kaggle notebook calls |
| `configs/` | One YAML file per run. Any result can be re-run from its file |
| `scripts/make_configs.py` | Writes the experiment configs in `configs/exp/` from one table |
| `runs/queue.yaml` | The list of jobs Kaggle works through |
| `kaggle/runner.ipynb` | The notebook you paste into Kaggle once |
| `tests/` | pytest tests. They run on a CPU in under a minute |

## Reproduce

```bash
pip install -r requirements.txt
python -m pytest                      # all tests, CPU only

# Build a dataset, train, evaluate (on a GPU machine):
python -m gptlab.data.prepare --config configs/data/fineweb_edu_16k.yaml --out data/fineweb-edu-16k
torchrun --standalone --nproc_per_node=2 -m gptlab.train --config configs/smoke/train.yaml --set data.name=fineweb-edu-16k
python -m gptlab.evaluate --run-dir out/smoke-train
```

Each run saves its full resolved config and the git commit, so it can be re-run exactly.
On Kaggle, all of this is driven by the queue: see [RUNNING.md](RUNNING.md).
