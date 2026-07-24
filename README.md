# LoRA tools

Concepts and training notes live in [`docs/`](docs/) — start with
[docs/README.md](docs/README.md).

## Inspect LoRA metadata

Use `tools/inspect_lora.py` to see which convention (and math format — LoRA /
LoKr / LoHa) a file uses. It also prints a `format` block (prefix convention ×
math suffix × trainer/base-model hints) and dumps the full `__metadata__`:

```bash
uv run tools/inspect_lora.py path/to/lora.safetensors
uv run tools/inspect_lora.py path/to/lora.safetensors --grep txtfusion  # filter keys
uv run tools/inspect_lora.py path/to/lora.safetensors --raw             # every key, untruncated meta
```

## Editing metadata (`tools/edit_metadata.py`)

Edit a LoRA's `__metadata__` — the [SAI ModelSpec](https://github.com/Stability-AI/ModelSpec)
fields (`modelspec.title`, `modelspec.author`, …) and any other key. Only the
safetensors header is rewritten; the tensor buffer is copied verbatim, so it's
instant and never loads a tensor. Pure stdlib (no `safetensors` package needed).

```bash
# Set ModelSpec fields; writes <name>.edited.safetensors next to the input.
uv run tools/edit_metadata.py lora.safetensors \
    --title "SteepSlope v2" --author yourname \
    --usage-hint "trigger: sslope" --tags "character,style"

# Preview the before -> after diff without writing.
uv run tools/edit_metadata.py lora.safetensors --title "X" --dry-run

# Overwrite the original in place (atomic temp + rename).
uv run tools/edit_metadata.py lora.safetensors --architecture krea2/lora --in-place

# Arbitrary keys and deletions.
uv run tools/edit_metadata.py lora.safetensors \
    --set modelspec.description="Epoch 2" --del notes --del ss_optimizer
```

| Flag | Description |
|------|-------------|
| `input` | Input `.safetensors` LoRA (positional). |
| `-o`, `--output` | Output path (default `<input>.edited.safetensors`). |
| `--in-place` | Overwrite the input (mutually exclusive with `-o`). |
| `--dry-run` | Show the before→after diff, write nothing. |
| `--set KEY=VALUE` | Set an arbitrary metadata key (repeatable). |
| `--del KEY` | Delete a metadata key (repeatable). |

ModelSpec named flags (each maps to the matching `modelspec.` key):
`--sai-model-spec`, `--architecture`, `--implementation`, `--title`,
`--description`, `--author`, `--date`, `--license`, `--usage-hint`, `--tags`,
`--merged-from`, `--hash-sha256`, `--resolution`, `--prediction-type`,
`--trigger-phrase`. Constrained fields (`prediction_type`, `resolution`, `date`,
`hash_sha256`) emit a non-fatal warning if the value looks off-spec.

## Metadata UI

For a point-and-click alternative, run the metadata editor UI:

```bash
uv run tools/metadata_ui.py            # opens http://127.0.0.1:8760 in your browser
uv run tools/metadata_ui.py --dir /path/to/loras --port 8080
```

It lists the `.safetensors` files in a directory, shows their metadata, and lets
you edit the ModelSpec fields (and add/remove arbitrary keys) with a Save button.

A **Browse…** button opens your machine's native file-open dialog, so you can edit any `.safetensors` on disk, not just those in
`--dir`. That needs a `tkinter`-capable Python and a display; when the dialog
isn't available the `--dir` listing is the fallback.


## Embedding a ComfyUI workflow (`tools/embed_workflow.py`)

Embed a ComfyUI workflow JSON into an image so it can be dragged onto the
ComfyUI canvas to load the graph. Output can be **WebP** (default), **PNG**, or
**JPEG** — each written the way ComfyUI reads it back (WebP and JPEG use EXIF
`Make`/`Model` tags; PNG uses `tEXt` chunks). Any Pillow-readable image works as
input. Requires Pillow (`pip install pillow`).

JPEG note: its EXIF segment is capped at ~64 KB, so large workflows must use PNG
or WebP, and JPEG re-encodes (lossy). PNG/WebP-lossless keep pixels intact.

```bash
# Embed a workflow into an image, writing <image>.embed.webp
uv run tools/embed_workflow.py screenshot.png workflow.json

# Write a PNG or JPEG instead (format inferred from the -o extension).
uv run tools/embed_workflow.py screenshot.png workflow.json -o out.png
uv run tools/embed_workflow.py screenshot.png workflow.json -o out.jpg

# Or force the format explicitly.
uv run tools/embed_workflow.py screenshot.png workflow.json --format jpg

# Also embed an API-format prompt (optional).
uv run tools/embed_workflow.py shot.webp workflow.json --prompt prompt.json

# Inspect what an image already has embedded.
uv run tools/embed_workflow.py preview.webp --read
```

| Flag | Description |
|------|-------------|
| `image` | Input image, any Pillow-readable format (positional). |
| `workflow` | Workflow `.json` to embed (positional; omit with `--read`). |
| `-o`, `--output` | Output path (default `<image>.embed.<format>`). |
| `--format {webp,png,jpg}` | Output format (default: inferred from `-o` extension, else webp). |
| `--prompt JSON` | Also embed an API-format prompt `.json`. |
| `--lossless` / `--lossy` | Lossless WebP (default) or lossy (WebP only). |
| `--quality N` | Quality 0–100 (default 90; WebP effort when lossless; also JPEG quality). |
| `--read` | Print the metadata already embedded in `<image>` and exit. |


## Stripping image metadata (`tools/strip_metadata.py`)

Remove **all** embedded metadata from an image — EXIF blocks and PNG/WebP text
chunks alike — writing a clean copy that carries only pixels. Handy for dropping
a ComfyUI workflow/prompt (the reverse of `embed_workflow.py`) or stray camera
EXIF before sharing an image. Verify with `uv run tools/embed_workflow.py
<image> --read`. Requires Pillow (`pip install pillow`).

```bash
# Strip metadata, writing <image>.stripped.<ext> next to the input.
uv run tools/strip_metadata.py preview.webp

# Overwrite the original in place.
uv run tools/strip_metadata.py preview.webp --in-place

# Choose an output path (format inferred from its extension).
uv run tools/strip_metadata.py shot.png -o clean.webp
```

| Flag | Description |
|------|-------------|
| `image` | Input image, any Pillow-readable format (positional). |
| `-o`, `--output` | Output path (default `<image>.stripped.<ext>`). |
| `--in-place` | Overwrite the input (mutually exclusive with `-o`). |
| `--quality N` | Quality 0–100 for lossy output like JPEG (default 90). |


## Training settings calculator (`tools/calc_training.py`)

Turn a dataset image count and a total **effective-step** budget into a
multi-resolution training schedule (steps/epoch, epochs, steps per pass). The
default recipe is a Krea 2 character LoKr: a 512 bulk pass at batch 2 and a 1024
refinement pass at batch 1, with 3/4 of the budget on 512 and 1/4 on 1024.

Effective steps are batch-1-equivalent: a pass of `S` steps at batch `B` counts
as `S × B`, so passes with different batch sizes share one comparable scale. Pure
stdlib (no dependencies).

> For the reasoning behind the numbers — effective steps vs gradient updates,
> learning-rate scaling, and gradient accumulation — see
> [docs/training-budget-and-batching.md](docs/training-budget-and-batching.md).
> For how to tell a run is working (loss, validation, held-out sets), see
> [docs/monitoring-training.md](docs/monitoring-training.md).

```bash
# Interactive mode — prompts for each input (also the default when run with no
# arguments). Press Enter to accept the shown [default].
uv run tools/calc_training.py -i

# Default recipe (512 @ batch 2, 1024 @ batch 1), 3/4 - 1/4 split.
uv run tools/calc_training.py 31 1250

# Rebalance the budget (weights are normalized, so 60/40, 3/1, 75/25 all work).
uv run tools/calc_training.py 31 1250 --split 60/40

# A custom three-pass schedule.
uv run tools/calc_training.py 20 800 --pass 512:2 --pass 768:2 --pass 1024:1 --split 2/1/1

# Gradient accumulation: 1024 at batch 1 but accum 2 (effective batch 2) when VRAM
# caps the real batch. Overriding --pass resets the split, so restate it.
uv run tools/calc_training.py 31 1250 --pass 512:2 --pass 1024:1:2 --split 3/1
```

Steps/epoch is `floor(images / (batch × accum))` by default, matching a trainer
that drops the trailing partial batch (`drop_last`); pass `--keep-last` for one
that pads it. Epochs are derived from the image count
(`round(fraction × total / images)`), so they're independent of the batching mode.

| Flag | Description |
|------|-------------|
| `images` | Number of images in the dataset (positional; omit to enter interactive mode). |
| `total_steps` | Total effective-step budget, batch-1-equivalent (positional). |
| `-i`, `--interactive` | Prompt for each input (also the default when run with no arguments). |
| `--pass RES:BATCH[:ACCUM]` | A training pass; `ACCUM` is gradient-accumulation steps (default 1). Repeatable; default `512:2` and `1024:1`. |
| `--split WEIGHTS` | `/`-separated budget weights, one per pass, normalized by their sum (default `3/1`). |
| `--keep-last` | Count the trailing partial batch (`ceil` steps/epoch); default drops it. |

> **Note:** overriding `--pass` resets `--split` to equal weights, so pass `--split`
> too if you want to keep the default `3/1` budget. Gradient accumulation
> (`ACCUM`) models an effective batch of `batch × accum`; see
> [docs/training-budget-and-batching.md](docs/training-budget-and-batching.md#gradient-accumulation).


## Checking a validation set for duplicates (`tools/compare_datasets.py`)

Compare a training folder and a validation folder and flag validation images that
duplicate — exactly or near-exactly — a training image. A near-duplicate shared
between the two sets quietly breaks validation (the val loss tracks the memorized
training frame instead of measuring generalization; see
[docs/monitoring-training.md](docs/monitoring-training.md#how-close-to-the-training-data-is-too-close)).

Matching uses a perceptual hash (dHash) compared by Hamming distance, so resizes,
re-compression and minor edits still match. It exits non-zero when any clash is
found, so it doubles as a pre-training gate. Requires Pillow (`pip install pillow`).

```bash
# Flag validation images too close to any training image.
uv run tools/compare_datasets.py train/ val/

# Stricter or looser matching (lower = stricter; 0 = only identical).
uv run tools/compare_datasets.py train/ val/ --threshold 5

# Show the closest training match for every validation image, not just clashes.
uv run tools/compare_datasets.py train/ val/ --show-all
```

| Flag | Description |
|------|-------------|
| `train_dir` | Folder of training images (positional). |
| `val_dir` | Folder of validation images to check (positional). |
| `--threshold N` | Max Hamming distance to flag as a clash (default 10; lower = stricter, 0 = only identical). |
| `--hash {dhash,ahash}` | Perceptual hash algorithm (default `dhash`). |
| `--size N` | Hash size; `N×N` bits (default 8 → 64-bit). |
| `-r`, `--recursive` | Recurse into subdirectories. |
| `--show-all` | Print the closest training match for every validation image. |


## License

MIT — see [LICENSE](LICENSE).
