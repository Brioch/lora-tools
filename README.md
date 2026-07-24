# LoRA tools

[![CI](https://github.com/Brioch/lora-tools/actions/workflows/ci.yml/badge.svg)](https://github.com/Brioch/lora-tools/actions/workflows/ci.yml)
[![Python 3.13+](https://img.shields.io/badge/python-3.13%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENCE)

A small collection of command-line tools for a Krea 2 / ComfyUI LoKr workflow:
inspect and edit LoRA `.safetensors` metadata, embed or strip ComfyUI workflow
metadata in images, and plan multi-resolution training runs. Most of it is
format-agnostic and pure stdlib; only the image/dataset tools need Pillow.

## Quick start

Everything runs through [uv](https://docs.astral.sh/uv/) — no manual setup. Each
script is self-contained (PEP 723 inline dependencies), so `uv run` fetches what
it needs on first use:

```bash
uv run tools/inspect_lora.py path/to/lora.safetensors
```

## What's inside

| Tool | Purpose |
|------|---------|
| `inspect_lora.py` | Identify a LoRA's convention / math format and dump its metadata |
| `edit_metadata.py` | Edit `__metadata__` — ModelSpec fields and arbitrary keys |
| `metadata_ui.py` | Local web UI for viewing and editing metadata |
| `embed_workflow.py` | Embed a ComfyUI workflow into an image |
| `strip_metadata.py` | Strip all metadata from an image |
| `calc_training.py` | Turn an image count + step budget into a training schedule |
| `compare_datasets.py` | Flag validation images that duplicate training images |

Full usage, examples, and flags for each: **[docs/tools.md](docs/tools.md)**.

## Documentation

Everything lives in [`docs/`](docs/):

- **[Tools reference](docs/tools.md)** — usage and flags for every script
- **[Training budget and batching](docs/training-budget-and-batching.md)** — sizing a
  run: steps, epochs, batch, effective steps, LR scaling, gradient accumulation
- **[Monitoring training](docs/monitoring-training.md)** — reading the loss,
  deterministic validation, and building a held-out set

## Requirements

- Python ≥ 3.13 and [uv](https://docs.astral.sh/uv/)
- Pillow — only for the image/dataset tools, fetched automatically by `uv run`

## License

MIT — see [LICENCE](LICENCE).
