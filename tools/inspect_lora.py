#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Inspect the keys of a .safetensors LoRA without loading any tensors.

Usage:
    python inspect_lora.py path/to/lora.safetensors
    python inspect_lora.py path/to/lora.safetensors --raw       # list every key
    python inspect_lora.py path/to/lora.safetensors --grep attn # filter keys

Reads only the safetensors JSON header, so it's instant even on multi-GB files.
"""

import argparse
import json
import re
import struct
import sys
from collections import Counter
from collections.abc import Iterable
from typing import Any

# Suffixes that identify the LoRA/LyCORIS math format of a module.
FORMAT_SUFFIXES: list[str] = [
    "lora_down",
    "lora_up",  # kohya / classic LoRA
    "lora_A",
    "lora_B",  # PEFT / diffusers LoRA
    "lokr_w1",
    "lokr_w2",  # LoKr (LyCORIS Kronecker)
    "lokr_w1_a",
    "lokr_w1_b",  # decomposed LoKr
    "lokr_w2_a",
    "lokr_w2_b",
    "hada_w1_a",
    "hada_w1_b",  # LoHa (Hadamard)
    "hada_w2_a",
    "hada_w2_b",
    "diff",
    "diff_b",  # full-diff
    "alpha",  # scaling
]


def read_header(path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    meta = header.pop("__metadata__", {})
    return header, meta


def detect_convention(keys: Iterable[str]) -> tuple[set[str], str]:
    """Classify keys by their prefix/naming convention.

    Returns (conventions, primary) where conventions is the set of matched
    labels (a file can mix dialects) and primary is the most common one. See
    strip_lora.py for the comfy/diffusers key layouts this repo already knows.
    """
    counts: Counter[str] = Counter()
    for k in keys:
        if k.startswith(("diffusion_model.", "text_encoders.")):
            counts["comfy"] += 1
        elif k.startswith(
            ("transformer.", "unet.", "text_encoder", "base_model.model.")
        ):
            counts["diffusers"] += 1
        elif k.startswith(("lora_unet_", "lora_te")):
            counts["kohya"] += 1
        else:
            counts["original"] += 1
    conventions = set(counts)
    primary = counts.most_common(1)[0][0] if counts else "original"
    return conventions, primary


# math suffix -> human label
MATH_LABELS: list[tuple[set[str], str]] = [
    ({"lora_A", "lora_B"}, "PEFT / diffusers (lora_A/lora_B)"),
    ({"lora_down", "lora_up"}, "kohya LoRA (lora_down/lora_up)"),
    (
        {"lokr_w1", "lokr_w2", "lokr_w1_a", "lokr_w1_b", "lokr_w2_a", "lokr_w2_b"},
        "LoKr (LyCORIS)",
    ),
    ({"hada_w1_a", "hada_w1_b", "hada_w2_a", "hada_w2_b"}, "LoHa (LyCORIS)"),
    ({"diff", "diff_b"}, "full-diff"),
]


def detect_math(fmts: Iterable[str]) -> list[str]:
    """Map the detected format suffixes to math-format label(s)."""
    present = set(fmts)
    labels = [label for suffixes, label in MATH_LABELS if suffixes & present]
    return labels or ["(unknown)"]


def _json_or_none(value: Any) -> Any:
    try:
        return json.loads(value)
    except (ValueError, TypeError):
        return None


def describe_metadata(meta: dict[str, Any]) -> dict[str, str]:
    """Pull trainer / base-model hints out of __metadata__ (may be empty)."""
    hints: dict[str, str] = {}

    # Trainer: the `software` field is a JSON blob {name, repo, version}.
    software = _json_or_none(meta.get("software"))
    if isinstance(software, dict) and software.get("name"):
        ver = software.get("version")
        hints["trainer"] = f"{software['name']}{' ' + ver if ver else ''}"

    ss_keys = [k for k in meta if k.startswith("ss_")]
    if ss_keys:
        note = "ss_* present"
        module = meta.get("ss_network_module")
        if module:
            note += f", ss_network_module={module}"
        hints["ss"] = note

    impl = meta.get("modelspec.implementation")
    if impl:
        hints["implementation"] = impl

    base = (
        meta.get("modelspec.architecture")
        or meta.get("ss_base_model_version")
        or meta.get("ss_sd_model_name")
    )
    if base:
        dim, alpha = meta.get("ss_network_dim"), meta.get("ss_network_alpha")
        if dim or alpha:
            base += f"  (dim {dim} / alpha {alpha})"
        hints["base model"] = base

    return hints


def render_meta_value(value: object, truncate: bool) -> str:
    """Render a metadata value; decode JSON-string blobs compactly."""
    decoded = _json_or_none(value) if isinstance(value, str) else None
    if isinstance(decoded, (dict, list)):
        value = json.dumps(decoded, separators=(",", ":"))
    text = str(value)
    if truncate and len(text) > 160:
        text = text[:157] + "..."
    return text


def shape_of(key: str) -> str:
    """Collapse a key to a structural shape: numeric indices -> N, format suffix dropped."""
    k = key
    for suf in FORMAT_SUFFIXES:
        k = re.sub(r"\." + re.escape(suf) + r"(\..*)?$", "", k)
    k = re.sub(r"\.weight$|\.bias$", "", k)
    k = re.sub(r"\.(\d+)(\.|$)", r".N\2", k)  # .12. -> .N.
    k = re.sub(r"_(\d+)_", "_N_", k)  # _12_ -> _N_
    return k


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("path", help="path to the .safetensors file")
    ap.add_argument("--raw", action="store_true", help="print every tensor key")
    ap.add_argument("--grep", metavar="SUBSTR", help="only show keys containing SUBSTR")
    args = ap.parse_args()

    try:
        header, meta = read_header(args.path)
    except Exception as e:
        sys.exit(f"error reading {args.path}: {e}")

    keys = sorted(header.keys())
    if args.grep:
        keys = [k for k in keys if args.grep in k]

    print(f"file            : {args.path}")
    print(f"total tensors   : {len(keys)}")

    # Detected math format(s).
    fmts = sorted(
        {
            suf
            for k in keys
            for suf in FORMAT_SUFFIXES
            if re.search(r"\." + re.escape(suf) + r"(\.|$)", k)
        }
    )
    print(f"format suffixes : {fmts or ['(none detected)']}")

    # Top-level namespaces.
    tops = Counter(k.split(".")[0] for k in keys)
    print(f"top-level       : {dict(tops)}")

    # --- format: prefix convention x math suffix x metadata hints ---------
    conventions, primary = detect_convention(keys)
    math_labels = detect_math(fmts)
    hints = describe_metadata(meta)

    others = sorted(conventions - {primary})
    conv_line = primary
    if others:
        conv_line += f"  (+ also: {', '.join(others)})"
    print("\nformat")
    print(f"  convention : {conv_line}")
    print(f"  math       : {', '.join(math_labels)}")
    if hints.get("trainer") or hints.get("ss"):
        parts = [p for p in (hints.get("trainer"), hints.get("ss")) if p]
        print(f"  trainer    : {' — '.join(parts)}")
    if hints.get("implementation"):
        print(f"  implement. : {hints['implementation']}")
    if hints.get("base model"):
        print(f"  base model : {hints['base model']}")

    # Block indices for the common Krea2 groups.
    def indices(pattern):
        return sorted(
            {int(m.group(1)) for k in keys for m in [re.search(pattern, k)] if m}
        )

    main_blocks = indices(r"(?:^|\.)blocks\.(\d+)\.")  # diffusion_model.blocks.N
    lw = indices(r"layerwise_blocks\.(\d+)\.")
    rf = indices(r"refiner_blocks\.(\d+)\.")

    def fmt_idx(lst):
        return f"{lst[0]}..{lst[-1]} (count {len(lst)})" if lst else "none"

    print(f"main blocks.N   : {fmt_idx(main_blocks)}")
    print(f"layerwise_blocks: {fmt_idx(lw)}")
    print(f"refiner_blocks  : {fmt_idx(rf)}")

    # Structural key shapes (indices collapsed) with counts.
    shapes = Counter(shape_of(k) for k in keys)
    print(f"\nstructural key shapes ({len(shapes)} distinct):")
    for shape in sorted(shapes):
        print(f"  {shapes[shape]:5d}  {shape}")

    if meta:
        print(f"\nmetadata ({len(meta)} entries):")
        width = max(len(k) for k in meta)
        for k in sorted(meta):
            print(
                f"  {k:<{width}} : {render_meta_value(meta[k], truncate=not args.raw)}"
            )
    else:
        print("\nmetadata        : (none)")

    if args.raw:
        print(f"\nall {len(keys)} keys:")
        for k in keys:
            print(f"  {k}")


if __name__ == "__main__":
    main()
