# Tools reference

Usage and flags for every script in [`tools/`](../tools/). Each one runs directly
with `uv run tools/<script>.py` — the PEP 723 header declares its dependencies, so
`uv` fetches what it needs on first use (only the image/dataset tools pull in
Pillow). Every script also has a `--help`.

- [Inspect LoRA metadata](#inspect-lora-metadata) — `inspect_lora.py`
- [LoRA weight health](#lora-weight-health-toolslora_healthpy) — `lora_health.py`
- [Checkpoint health sweep](#checkpoint-health-sweep-toolshealth_sweeppy) — `health_sweep.py`
- [Editing metadata](#editing-metadata-toolsedit_metadatapy) — `edit_metadata.py`
- [Metadata UI](#metadata-ui) — `metadata_ui.py`
- [Embedding a ComfyUI workflow](#embedding-a-comfyui-workflow-toolsembed_workflowpy) — `embed_workflow.py`
- [Stripping image metadata](#stripping-image-metadata-toolsstrip_metadatapy) — `strip_metadata.py`
- [Training settings calculator](#training-settings-calculator-toolscalc_trainingpy) — `calc_training.py`
- [Checking a validation set for duplicates](#checking-a-validation-set-for-duplicates-toolscompare_datasetspy) — `compare_datasets.py`
- [Linting a dataset](#linting-a-dataset-toolslint_datasetpy) — `lint_dataset.py`
- [Normalizing dataset images](#normalizing-dataset-images-toolsprepare_imagespy) — `prepare_images.py`
- [Removing duplicate images](#removing-duplicate-images-toolsdedupe_imagespy) — `dedupe_images.py`
- [Batch-editing captions](#batch-editing-captions-toolsedit_captionspy) — `edit_captions.py`
- [Balancing a regularization set](#balancing-a-regularization-set-toolsbalance_regularizationpy) — `balance_regularization.py`

The five dataset tools share a small helper module, `tools/dsutils.py` (extension
normalization, image/caption pairing, dHash). It is imported rather than run, so it
has no CLI of its own.

## Inspect LoRA metadata

Use `tools/inspect_lora.py` to see which convention (and math format — LoRA /
LoKr / LoHa) a file uses. It also prints a `format` block (prefix convention ×
math suffix × trainer/base-model hints) and dumps the full `__metadata__`:

```bash
uv run tools/inspect_lora.py path/to/lora.safetensors
uv run tools/inspect_lora.py path/to/lora.safetensors --grep txtfusion  # filter keys
uv run tools/inspect_lora.py path/to/lora.safetensors --raw             # every key, untruncated meta
```

## LoRA weight health (`tools/lora_health.py`)

Where `inspect_lora.py` reads only the header, `tools/lora_health.py` decodes the
tensor **data**, reconstructs each module's effective delta-weight (ΔW) — folding
the low-rank factors and the `alpha/rank` scale back together — and reports the
numbers that actually track training quality:

- **‖ΔW‖_F** — the effective magnitude of a module's change to the base weights
- **σ_max** — its largest singular value (peak per-direction strength)
- **stable rank** — `‖ΔW‖_F² / σ_max²` ([Rudelson & Vershynin,
  2007](https://doi.org/10.1145/1255443.1255449)); ≈1 means the update collapsed
  onto a single direction, high means it stays spread across many directions

From those it flags the failure modes a metadata dump can't see:

- **non-finite** — any `NaN`/`Inf` value, i.e. a diverged / "fried" run
- **dead** — effective `‖ΔW‖_F` at or below `--dead-threshold` (never trained)
- **suspect** — large `‖ΔW‖_F` **and** low stable rank (`--outlier-factor`,
  `--collapse-rank`): the update is both strong and collapsed, a candidate
  over-cooked layer

```bash
uv run tools/lora_health.py path/to/lora.safetensors
uv run tools/lora_health.py path/to/lora.safetensors --all     # every module
uv run tools/lora_health.py path/to/lora.safetensors --json    # machine-readable
uv run tools/lora_health.py path/to/lora.safetensors --grep attn
```

Reconstruction is exact for classic LoRA (`down`/`up`, `A`/`B`), LoKr, and LoHa;
the `alpha/rank` scale is applied when both are recoverable (shown per module).
`F32`/`F16`/`BF16`/`F64` are decoded; float8 and other exotic dtypes are skipped.
It reads the whole file (slower than `inspect_lora.py`, fine for the tens-of-MB
files here) and needs **numpy** for the SVD — `uv run` fetches it automatically.
**Exit status is `1` when any module is non-finite**, so it doubles as a CI gate.

> **Basis and caveat.** The stable-rank metric is standard linear algebra
> ([Rudelson & Vershynin, 2007](https://doi.org/10.1145/1255443.1255449)), but
> reading a "strong + collapsed" module as *over-cooked* is a **heuristic, not a
> validated result** — no paper establishes it for LoRAs. Spectral shape is only
> *indirectly* tied to training quality in the literature (e.g. Martin &
> Mahoney's heavy-tailed self-regularisation / `WeightWatcher`), and the flag has
> both false positives (a legitimately focused, low-rank adaptation) and false
> negatives (a LoRA fried diffusely across many directions). Weight statistics
> can *suggest* over-cooking but can't prove it — the ground truth is behavioural
> (a strength sweep, prompt-adherence, and training-set memorisation checks; see
> [monitoring training](monitoring-training.md)). Treat **suspect** as "look
> here," not a verdict.

## Checkpoint health sweep (`tools/health_sweep.py`)

Where `lora_health.py` reports one file, `tools/health_sweep.py` runs the **same
analysis across many checkpoints** and lays the results out one row per epoch, so
you can see the *trajectory* — where the adapter saturates and whether it's
collapsing. It reuses `lora_health`'s own reconstruction and flag logic (imports
`collect` and `_flags`), so every number matches `lora_health.py` exactly.

Point it at a directory, a glob, or a list of files:

```bash
uv run tools/health_sweep.py path/to/checkpoints/           # recurse a directory
uv run tools/health_sweep.py 'runs/**/*.safetensors' --grep attn
uv run tools/health_sweep.py ckpts/ --csv > trajectory.csv  # for plotting
```

Each row summarises the spread across modules — `‖ΔW‖_F` (min/median/max), stable
rank (min/median/max) — plus the per-checkpoint **non-finite / dead / suspect**
counts, followed by a short structural *read*:

- **`‖ΔW‖_F` median climbing then flattening** → the adapter saturated; later
  epochs add magnitude but little new signal.
- **stable-rank median falling** across epochs → updates collapsing onto fewer
  directions, the structural fingerprint of over-cooking.

Epoch labels are parsed from the filename (`epoch40`, `-e12`, `step_1200`,
`-000060.safetensors`, …). If parsing can't find a distinct number per file — e.g.
timestamped names like `2026-07-24_19-19-03-save-760-40-0.safetensors`, where the
trailing field is constant — it **warns, falls back to filename order, and shows a
filename column** so the ordering stays verifiable. Point it at the right field
with `--epoch-regex` (one capture group; use `=` so a leading `-` isn't read as a
flag):

```bash
uv run tools/health_sweep.py ckpts/ --epoch-regex='-(\d+)-\d+\.safetensors$'  # the 40 in -save-760-40-0
uv run tools/health_sweep.py ckpts/ --files                                   # always show filenames
```

`--dead-threshold`, `--outlier-factor`, and `--collapse-rank` pass straight
through to the same flag logic as `lora_health.py`. Same **structural-only**
caveat applies: this shows where the adapter saturates and whether it's
collapsing, **not** whether a checkpoint nails the concept — pair it with the
generation/validation checks in [monitoring training](monitoring-training.md), and
see [LoKr training: dim, alpha, and factor](lokr-dim-and-alpha.md) for what a
healthy damped-LoKr trajectory looks like.

## Editing metadata (`tools/edit_metadata.py`)

Edit a LoRA's `__metadata__` — the [SAI ModelSpec](https://github.com/Stability-AI/ModelSpec)
fields (`modelspec.title`, `modelspec.author`, …) and any other key. Only the
safetensors header is rewritten; the tensor buffer is copied verbatim, so it's
instant and never loads a tensor. Pure stdlib (no `safetensors` package needed).

```bash
# Set ModelSpec fields; writes <name>.edited.safetensors next to the input.
uv run tools/edit_metadata.py lora.safetensors \
    --title "My Character v2" --author yourname \
    --usage-hint "trigger: mychar" --tags "character,style"

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
> [training-budget-and-batching.md](training-budget-and-batching.md).
> For how to tell a run is working (loss, validation, held-out sets), see
> [monitoring-training.md](monitoring-training.md).

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
> [training-budget-and-batching.md](training-budget-and-batching.md#gradient-accumulation).


## Checking a validation set for duplicates (`tools/compare_datasets.py`)

Compare a training folder and a validation folder and flag validation images that
duplicate — exactly or near-exactly — a training image. A near-duplicate shared
between the two sets quietly breaks validation (the val loss tracks the memorized
training frame instead of measuring generalization; see
[monitoring-training.md](monitoring-training.md#how-close-to-the-training-data-is-too-close)).

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


## Linting a dataset (`tools/lint_dataset.py`)

Check a directory of image + caption pairs for the problems that most often waste
GPU time: an image the loader cannot decode, a caption that was never written, a
stray caption left behind after its image was deleted. Findings are split by how
much they matter, and the exit code follows that split, so it doubles as a
pre-training gate:

| Severity | Exit | Checks |
|----------|------|--------|
| `ERRORS` | 1 | corrupt/unreadable image, image with no caption, empty caption |
| `WARNINGS` | 0 | orphan caption (no image), non-RGB mode, image below `--min-size`, extension not matching the real format |

Pair it with [`dedupe_images.py`](#removing-duplicate-images-toolsdedupe_imagespy)
for duplicate frames and
[`prepare_images.py`](#normalizing-dataset-images-toolsprepare_imagespy) to fix most
of what it warns about. Requires Pillow.

```bash
# Gate a dataset: exits 1 if anything is actually broken.
uv run tools/lint_dataset.py --dir ./train

# Also warn about small images, and accept images without captions.
uv run tools/lint_dataset.py --dir ./train --min-size 1024 --no-require-caption
```

| Flag | Description |
|------|-------------|
| `--dir DIR` | Dataset directory (required). |
| `--image-exts EXT ...` | Image extensions to consider, case-insensitive (default `.png .jpg .jpeg .webp .bmp`). |
| `--caption-ext EXT` | Caption extension paired with each image (default `.txt`). |
| `--min-size N` | Warn if an image's shortest edge is below this, in px (default 0 = off). |
| `--no-require-caption` | Treat a missing caption as a warning instead of an error. |


## Normalizing dataset images (`tools/prepare_images.py`)

Trainers choke on, or silently mangle, the odds and ends a scraped dataset carries:
sideways phone photos (EXIF rotation), transparent PNGs that composite to black,
palette or grayscale modes, and 6000px originals that cost VRAM without adding
detail. For every image this can apply EXIF orientation and strip metadata, flatten
transparency to RGB over white, convert to one target format, and downscale so the
longest edge fits `--max-edge` (it never upscales).

It never touches the input unless you ask it to: by default the results go to a
sibling `<dir>.prepared/` directory with paired captions copied along, and `--out-dir`
picks that destination explicitly. Requires Pillow; exits 1 if any image failed to
process.

```bash
# Normalize ./raw into ./raw.prepared/, leaving ./raw untouched.
uv run tools/prepare_images.py --dir ./raw --format jpg --max-edge 1536

# Same, but choose the destination.
uv run tools/prepare_images.py --dir ./raw --out-dir ./train --format png

# Rewrite the dataset in place (destructive; preview it with --dry-run first).
uv run tools/prepare_images.py --dir ./train --max-edge 1536 --in-place
```

> **`--in-place` is irreversible.** It rewrites each image over itself, and a format
> change (`--format`) also deletes the old-extension original. There is no backup and
> no undo — keep a copy of the dataset, or run `--dry-run` first. Every other mode
> leaves the input directory exactly as it was.

| Flag | Description |
|------|-------------|
| `--dir DIR` | Input directory (required). |
| `--out-dir DIR` | Destination directory (default: a sibling `<dir>.prepared`). Paired captions are copied along. Mutually exclusive with `--in-place`. |
| `--in-place` | Rewrite the input directory instead. **Destructive** — a format change deletes the old-extension original. |
| `--format {png,jpg}` | Convert every image to this format (default: keep each as-is). |
| `--max-edge N` | Downscale so the longest edge is ≤ this, in px (default 0 = off, never upscales). |
| `--no-flatten` | Keep the original mode instead of forcing RGB. |
| `--quality N` | JPEG quality (default 95). |
| `--image-exts EXT ...` | Image extensions to consider, case-insensitive (default `.png .jpg .jpeg .webp .bmp`). |
| `--caption-ext EXT` | Caption extension paired with each image (default `.txt`). |
| `--dry-run` | Print what would be written without touching the filesystem. |


## Removing duplicate images (`tools/dedupe_images.py`)

Duplicates in a training set silently reweight it: the same frame seen twice pulls
the model twice as hard toward it, which is how a character LoRA ends up locked to
one pose. This finds them in two passes — byte-identical files (SHA-256), then
perceptual near-duplicates (dHash within `--threshold` Hamming distance, which
catches crops, re-encodes and resizes).

It uses the same 64-bit dHash as
[`compare_datasets.py`](#checking-a-validation-set-for-duplicates-toolscompare_datasetspy),
so distances from the two tools are comparable: 0 is identical at hash resolution,
1–10 near-duplicate, higher genuinely distinct. Within each group one image is kept
(highest resolution by default) and the rest are deleted with their captions.
Deletions prompt for confirmation unless `-y` or `--dry-run`. Requires Pillow.

```bash
# Report and remove near-duplicates (prompts before deleting).
uv run tools/dedupe_images.py --dir ./train --threshold 5

# Byte-identical files only, and show what would go without touching anything.
uv run tools/dedupe_images.py --dir ./train --exact-only --dry-run
```

| Flag | Description |
|------|-------------|
| `--dir DIR` | Dataset directory (required). |
| `--threshold N` | Max dHash Hamming distance for near-duplicates (default 5; 0 = identical only). |
| `--exact-only` | Only remove byte-identical files, skipping the perceptual pass. |
| `--keep {largest,first}` | Which image to keep in each group (default `largest`, tie-broken by name). |
| `--image-exts EXT ...` | Image extensions to consider, case-insensitive (default `.png .jpg .jpeg .webp .bmp`). |
| `--caption-ext EXT` | Caption extension deleted alongside each image (default `.txt`). |
| `--dry-run` | Report the groups without deleting anything. |
| `-y`, `--yes` | Skip the confirmation prompt before deleting. |


## Batch-editing captions (`tools/edit_captions.py`)

Edit every caption `.txt` in a dataset at once. Captions are treated as
comma-separated tag lists (the common diffusion format), so the edits are tag-aware
rather than plain text substitution: replacing `man` will not mangle `woman`, and
adding a trigger word twice is a no-op — which makes the tool safe to re-run.

Operations are applied in a fixed order regardless of flag order: `--replace`,
`--remove`, `--prepend`, `--add`, `--dedupe`, `--sort`. Matching is
case-insensitive; the original casing of kept tags is preserved. Every run prints a
before/after diff of the files it touches. Pure stdlib.

```bash
# Add a trigger word to the front of every caption and drop duplicate tags.
uv run tools/edit_captions.py --dir ./train --prepend "mytoken" --dedupe

# Clean up unwanted tags and rename one.
uv run tools/edit_captions.py --dir ./train --remove "blurry" --replace "man" "person"
```

| Flag | Description |
|------|-------------|
| `--dir DIR` | Dataset directory (required). |
| `--caption-ext EXT` | Caption extension to edit (default `.txt`). |
| `--replace OLD NEW` | Rename a whole tag `OLD` to `NEW`, case-insensitive. Repeatable. |
| `--remove TAG` | Remove a tag, case-insensitive. Repeatable. |
| `--prepend TAG` | Prepend a tag if absent, e.g. a trigger word. Repeatable. |
| `--add TAG` | Append a tag if absent. Repeatable. |
| `--dedupe` | Drop duplicate tags, keeping the first occurrence. |
| `--sort` | Sort tags alphabetically. |
| `--dry-run` | Show the before/after diff without writing. |

At least one operation is required; with none the tool exits 2 rather than
rewriting every file to no effect.


## Balancing a regularization set (`tools/balance_regularization.py`)

A Dreambooth/LoRA dataset wants the regularization (class) image count to match the
training image count so neither set overpowers the other: too few class images and
the class prior collapses into your subject, too many and the subject never lands.
This makes the regularization directory hold *exactly* as many image+caption pairs
as the training directory — drawing new ones at random from a separate pool, or
trimming the excess.

It is idempotent: run it twice with the same `--seed` and the second run has nothing
to do, so it is safe to re-run after adding training images. Images already present
in the regularization directory (by stem) are never drawn twice from the pool.
Deletions prompt for confirmation unless `-y` or `--dry-run`. Pure stdlib.

```bash
# Bring ./reg to the same image count as ./train, drawing from ./pool.
uv run tools/balance_regularization.py \
    --train-dir ./train --pool-dir ./pool --reg-dir ./reg --seed 0

# Preview the plan without touching the filesystem.
uv run tools/balance_regularization.py \
    --train-dir ./train --pool-dir ./pool --reg-dir ./reg --dry-run
```

| Flag | Description |
|------|-------------|
| `--train-dir DIR` | Training dir; its image count sets the target n (required). |
| `--pool-dir DIR` | Source pool of candidate regularization images (required). |
| `--reg-dir DIR` | Destination regularization dir to balance; created if absent (required). |
| `--seed N` | Seed for reproducible random selection (default: unseeded). |
| `--image-exts EXT ...` | Image extensions to consider, case-insensitive (default `.png .jpg .jpeg .webp .bmp`). |
| `--caption-ext EXT` | Caption extension paired with each image (default `.txt`). |
| `--move` | Move pool images instead of copying them. |
| `--dry-run` | Print planned actions without changing the filesystem. |
| `-y`, `--yes` | Skip the confirmation prompt before deletions. |

> **Note:** a pool smaller than the shortfall is not an error — the tool takes what
> it can, warns on stderr how many the set is short by, and still exits 0.
