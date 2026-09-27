# gptlab: a small GPT trained from scratch

This project trains small GPT-style language models (about 1M to 100M parameters)
from scratch and runs real experiments on them: scaling laws, architecture
ablations, training stability, speed, and evaluation.

Everything runs on free hardware: CPU tests in the cloud, and GPU training on
Kaggle (2x T4). The model, training loop and evaluation code are written from
scratch in PyTorch.

> Status: work in progress. Results will appear here once runs have finished.
> Until then every result section says "not run yet".

## Summary

Not run yet.

## Repo layout

| Path | What it holds |
| --- | --- |
| `gptlab/` | The Python package: model, training loop, data pipeline, evaluation |
| `gptlab/data/` | Download, cleaning, tokenizer, token shards, data loader |
| `gptlab/runner/` | The job runner that Kaggle calls |
| `configs/` | One YAML file per run. A result can be re-run from its file |
| `runs/queue.yaml` | The list of jobs Kaggle works through |
| `kaggle/` | The notebook you paste into Kaggle once |
| `tests/` | pytest tests. They run on a CPU in seconds |

## Quick start (CPU)

```bash
pip install -r requirements.txt
python -m pytest
```

## Running on Kaggle

See [RUNNING.md](RUNNING.md).
