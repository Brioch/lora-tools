# LoRA tools

## Inspect LoRA metadata

Use `tools/inspect_lora.py` to see which convention (and math format — LoRA /
LoKr / LoHa) a file uses. It also prints a `format` block (prefix convention ×
math suffix × trainer/base-model hints) and dumps the full `__metadata__`:

```bash
python tools/inspect_lora.py path/to/lora.safetensors
python tools/inspect_lora.py path/to/lora.safetensors --grep txtfusion  # filter keys
python tools/inspect_lora.py path/to/lora.safetensors --raw             # every key, untruncated meta
```

## Editing metadata (`tools/edit_metadata.py`)

Edit a LoRA's `__metadata__` — the [SAI ModelSpec](https://github.com/Stability-AI/ModelSpec)
fields (`modelspec.title`, `modelspec.author`, …) and any other key. Only the
safetensors header is rewritten; the tensor buffer is copied verbatim, so it's
instant and never loads a tensor. Pure stdlib (no `safetensors` package needed).

```bash
# Set ModelSpec fields; writes <name>.edited.safetensors next to the input.
python tools/edit_metadata.py lora.safetensors \
    --title "SteepSlope v2" --author yourname \
    --usage-hint "trigger: sslope" --tags "character,style"

# Preview the before -> after diff without writing.
python tools/edit_metadata.py lora.safetensors --title "X" --dry-run

# Overwrite the original in place (atomic temp + rename).
python tools/edit_metadata.py lora.safetensors --architecture krea2/lora --in-place

# Arbitrary keys and deletions.
python tools/edit_metadata.py lora.safetensors \
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
python tools/metadata_ui.py            # opens http://127.0.0.1:8760 in your browser
python tools/metadata_ui.py --dir /path/to/loras --port 8080
```

It lists the `.safetensors` files in a directory, shows their metadata, and lets
you edit the ModelSpec fields (and add/remove arbitrary keys) with a Save button.

A **Browse…** button opens your machine's native file-open dialog, so you can edit any `.safetensors` on disk, not just those in
`--dir`. That needs a `tkinter`-capable Python and a display; when the dialog
isn't available the `--dir` listing is the fallback.


## License

MIT — see [LICENSE](LICENSE).
