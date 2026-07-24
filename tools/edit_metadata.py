#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Edit the __metadata__ of a .safetensors LoRA (ModelSpec fields and more).

Only the safetensors JSON header is rewritten; the tensor byte-buffer is
stream-copied verbatim, so this is instant even on multi-GB files and never
loads a tensor. Tensor data_offsets are relative to the buffer, so changing the
header length keeps every tensor valid.

The named flags follow the Stability-AI Model Metadata Standard (SAI ModelSpec):
    https://github.com/Stability-AI/ModelSpec
All ModelSpec keys carry the "modelspec." prefix in the file; the flags add it
for you (--title -> modelspec.title). Use --set / --del for arbitrary keys
(e.g. ss_* training fields, notes).

Examples:
    # Set a few ModelSpec fields, write lora.edited.safetensors next to it.
    python edit_metadata.py lora.safetensors \
        --title "SteepSlope v2" --author yourname \
        --usage-hint "trigger: sslope" --tags "character,style"

    # Preview the change without writing anything.
    python edit_metadata.py lora.safetensors --title "X" --dry-run

    # Overwrite the original in place (atomic replace).
    python edit_metadata.py lora.safetensors --architecture krea2/lora --in-place

    # Arbitrary keys + deletions.
    python edit_metadata.py lora.safetensors \
        --set modelspec.description="Epoch 2" --del notes --del ss_optimizer

Requires no third-party packages (pure stdlib).
"""

import argparse
import json
import os
import re
import shutil
import struct
import sys
from typing import Any

MODELSPEC = "modelspec."

# Named flags -> full metadata key. Grouped by ModelSpec section; each entry is
# (flag_suffix, metadata_key, help). flag_suffix uses hyphens for the CLI.
FIELDS: list[tuple[str, str, str]] = [
    # --- core (all models) ---
    ("sai-model-spec", MODELSPEC + "sai_model_spec", "spec version string, e.g. 1.0.0"),
    ("architecture", MODELSPEC + "architecture", "architecture id, e.g. krea2/lora"),
    (
        "implementation",
        MODELSPEC + "implementation",
        "implementation id, e.g. diffusers / sgm / comfy",
    ),
    ("title", MODELSPEC + "title", "user-friendly model name"),
    (
        "description",
        MODELSPEC + "description",
        "description / capabilities (markdown ok)",
    ),
    ("author", MODELSPEC + "author", "creator name / company / username"),
    ("date", MODELSPEC + "date", "ISO-8601 date, e.g. 2026-07-14"),
    ("license", MODELSPEC + "license", "license name / URL / terms"),
    (
        "usage-hint",
        MODELSPEC + "usage_hint",
        "brief operational guidance / trigger words",
    ),
    ("tags", MODELSPEC + "tags", "comma-separated discovery tags"),
    ("merged-from", MODELSPEC + "merged_from", "comma-separated source model names"),
    ("hash-sha256", MODELSPEC + "hash_sha256", "tensor hash, 0x-prefixed hex"),
    # --- image-generation models ---
    (
        "resolution",
        MODELSPEC + "resolution",
        "base resolution WIDTHxHEIGHT, e.g. 1024x1024",
    ),
    ("prediction-type", MODELSPEC + "prediction_type", "v or epsilon"),
    (
        "trigger-phrase",
        MODELSPEC + "trigger_phrase",
        "required phrase for an adapter/LoRA",
    ),
]

# dest (argparse turns "-" into "_") -> metadata key
FLAG_TO_KEY = {suffix.replace("-", "_"): key for suffix, key, _ in FIELDS}


def read_header(path: str) -> tuple[int, dict[str, Any]]:
    """Return (header_len, header_dict). Leaves nothing open."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return n, header


def validate(key: str, value: str) -> str | None:
    """Return a warning string for a non-ModelSpec-conformant value, else None."""
    if key == MODELSPEC + "prediction_type" and value not in ("v", "epsilon"):
        return f"prediction_type should be 'v' or 'epsilon', got {value!r}"
    if key == MODELSPEC + "resolution" and not re.fullmatch(r"\d+x\d+", value):
        return f"resolution should be WIDTHxHEIGHT (e.g. 1024x1024), got {value!r}"
    if key == MODELSPEC + "date" and not re.match(r"^\d{4}-\d{2}-\d{2}", value):
        return f"date should be ISO-8601 (e.g. 2026-07-14), got {value!r}"
    if key == MODELSPEC + "hash_sha256" and not value.startswith("0x"):
        return f"hash_sha256 should be 0x-prefixed hex, got {value!r}"
    return None


def parse_set(item: str) -> tuple[str, str]:
    """Parse a --set KEY=VALUE item into (key, value)."""
    if "=" not in item:
        raise argparse.ArgumentTypeError(f"--set expects KEY=VALUE, got {item!r}")
    key, value = item.split("=", 1)
    key = key.strip()
    if not key:
        raise argparse.ArgumentTypeError(f"--set has an empty key: {item!r}")
    return key, value


def rewrite(in_path: str, out_path: str, new_meta: dict[str, str]) -> None:
    """Write out_path = new header (with new_meta) + original tensor buffer."""
    with open(in_path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))  # f now sits at the tensor buffer
        if new_meta:
            header["__metadata__"] = new_meta
        else:
            header.pop("__metadata__", None)
        new_header = json.dumps(header, ensure_ascii=False).encode("utf-8")
        # Pad with spaces so the tensor buffer stays 8-byte aligned (matches the
        # safetensors reference writer; trailing whitespace is valid JSON).
        pad = (8 - (len(new_header) % 8)) % 8
        new_header += b" " * pad
        with open(out_path, "wb") as g:
            g.write(struct.pack("<Q", len(new_header)))
            g.write(new_header)
            shutil.copyfileobj(f, g)  # verbatim copy of the tensor bytes


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("input", help="path to the input .safetensors LoRA")
    ap.add_argument(
        "-o", "--output", help="output path (default: <input>.edited.safetensors)"
    )
    ap.add_argument(
        "--in-place",
        action="store_true",
        help="overwrite the input file (atomic temp+rename)",
    )
    ap.add_argument(
        "--dry-run", action="store_true", help="show the diff, write nothing"
    )
    ap.add_argument(
        "--set",
        dest="sets",
        metavar="KEY=VALUE",
        type=parse_set,
        action="append",
        default=[],
        help="set an arbitrary metadata key (repeatable)",
    )
    ap.add_argument(
        "--del",
        dest="dels",
        metavar="KEY",
        action="append",
        default=[],
        help="delete a metadata key (repeatable)",
    )

    grp = ap.add_argument_group(
        "ModelSpec fields (https://github.com/Stability-AI/ModelSpec)"
    )
    for suffix, key, help_text in FIELDS:
        grp.add_argument(
            f"--{suffix}",
            dest=suffix.replace("-", "_"),
            default=None,
            metavar="VALUE",
            help=help_text,
        )

    args = ap.parse_args()

    if args.in_place and args.output:
        sys.exit("error: --in-place and -o/--output are mutually exclusive")
    if not os.path.isfile(args.input):
        sys.exit(f"error: no such file: {args.input}")

    try:
        _, header = read_header(args.input)
    except Exception as e:
        sys.exit(f"error reading {args.input}: {e}")
    old_meta = header.get("__metadata__", {}) or {}

    # Collect the requested changes (named flags first, then generic --set).
    sets: dict[str, str] = {}
    for dest, key in FLAG_TO_KEY.items():
        val = getattr(args, dest)
        if val is not None:
            sets[key] = val
    for key, val in args.sets:
        sets[key] = val

    dels = list(dict.fromkeys(args.dels))  # de-dup, keep order

    if not sets and not dels:
        sys.exit("error: nothing to do — pass a ModelSpec flag, --set, or --del")

    # Validate (non-fatal warnings).
    for key, val in sets.items():
        warn = validate(key, val)
        if warn:
            print(f"warning: {warn}", file=sys.stderr)

    # Build the new metadata dict (safetensors requires string values).
    new_meta = {str(k): str(v) for k, v in old_meta.items()}
    for key, val in sets.items():
        new_meta[key] = str(val)
    missing = [k for k in dels if k not in new_meta]
    for key in dels:
        new_meta.pop(key, None)
    if missing:
        print(f"warning: --del key(s) not present, ignored: {missing}", file=sys.stderr)

    # Report the diff.
    print(f"input   : {args.input}")
    changed = sorted(set(sets) | set(k for k in dels if k in old_meta))
    print(f"changes : {len(changed)}")
    for key in changed:
        before = repr(old_meta[key]) if key in old_meta else "(absent)"
        after = repr(new_meta[key]) if key in new_meta else "(deleted)"
        print(f"  {key}\n      {before}  ->  {after}")

    if args.dry_run:
        print("dry-run : nothing written.")
        return

    if args.in_place:
        out = args.input
        tmp = out + ".tmp"
        rewrite(args.input, tmp, new_meta)
        os.replace(tmp, out)
    else:
        out = (
            args.output
            or re.sub(r"\.safetensors$", "", args.input) + ".edited.safetensors"
        )
        rewrite(args.input, out, new_meta)
    print(f"output  : {out}")


if __name__ == "__main__":
    main()
