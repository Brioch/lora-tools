# LoRA tools

[![CI](https://github.com/Brioch/lora-tools/actions/workflows/ci.yml/badge.svg)](https://github.com/Brioch/lora-tools/actions/workflows/ci.yml)
[![Coverage](https://img.shields.io/endpoint?url=https://gist.githubusercontent.com/Brioch/ab179799bf4df9083c5a72975b6e6c14/raw/lora-tools-coverage.json)](https://github.com/Brioch/lora-tools/actions/workflows/ci.yml)
[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENCE)

A small collection of command-line tools for a Krea 2 / ComfyUI LoKr workflow:
inspect and edit LoRA `.safetensors` metadata, embed or strip ComfyUI workflow
metadata in images, prepare and lint training datasets, and plan multi-resolution
training runs. Most of it is format-agnostic and pure stdlib; only the
image/dataset tools need Pillow, and the weight-health tool needs numpy.

## Quick start

Everything runs through [uv](https://docs.astral.sh/uv/) — no manual setup. Each
script declares its own dependencies inline (PEP 723), so `uv run` fetches what it
needs on first use:

```bash
uv run tools/inspect_lora.py path/to/lora.safetensors
```

## What's inside

| Tool | Purpose |
|------|---------|
| `inspect_lora.py` | Identify a LoRA's convention / math format and dump its metadata |
| `lora_health.py` | Reconstruct each module's ΔW and flag dead / fried / over-cooked layers |
| `health_sweep.py` | Run `lora_health` across many checkpoints and show the trajectory |
| `edit_metadata.py` | Edit `__metadata__` — ModelSpec fields and arbitrary keys |
| `metadata_ui.py` | Local web UI for viewing and editing metadata |
| `embed_workflow.py` | Embed a ComfyUI workflow into an image |
| `strip_metadata.py` | Strip all metadata from an image |
| `calc_training.py` | Turn an image count + step budget into a training schedule |
| `compare_datasets.py` | Flag validation images that duplicate training images |
| `lint_dataset.py` | Validate a dataset's image + caption pairs before training |
| `prepare_images.py` | Normalize images: orientation, mode, format, longest edge |
| `dedupe_images.py` | Remove exact and near-duplicate images (and their captions) |
| `edit_captions.py` | Batch-edit caption tag lists (replace, remove, add, dedupe, sort) |
| `balance_regularization.py` | Match a regularization set's image count to the training set |

Full usage, examples, and flags for each: **[docs/tools.md](docs/tools.md)**.

## Documentation

Everything lives in [`docs/`](docs/):

- **[Tools reference](docs/tools.md)** — usage and flags for every script
- **[Training budget and batching](docs/training-budget-and-batching.md)** — sizing a
  run: steps, epochs, batch, effective steps, LR scaling, gradient accumulation
- **[Monitoring training](docs/monitoring-training.md)** — reading the loss,
  deterministic validation, and building a held-out set
- **[LoKr training: dim, alpha, and factor](docs/lokr-dim-and-alpha.md)** — what
  the LoKr knobs do, the `alpha/dim` scale, and setting them for a character

## Requirements

- Python ≥ 3.13 and [uv](https://docs.astral.sh/uv/)
- Pillow (image/dataset tools) and numpy (`lora_health.py`) — fetched
  automatically by `uv run`

## License

MIT — see [LICENCE](LICENCE).
