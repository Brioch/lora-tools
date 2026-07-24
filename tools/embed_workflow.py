#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = ["pillow>=12.3.0"]
# ///
"""Embed a ComfyUI workflow JSON into an image so it can be dragged onto the
ComfyUI canvas to load the graph. Output can be WebP, PNG, or JPEG.

ComfyUI stores this metadata differently per format, and this script writes
exactly what its front-end reads back:
  * WebP/JPEG: EXIF string entries split on the first ':', so a "workflow" key
    lives in the Make tag as "workflow:<json>" (and "prompt" in Model).
  * PNG: plain tEXt chunks keyed "workflow" and "prompt", holding the raw JSON.
JPEG uses the same EXIF scheme as WebP, but its EXIF segment is capped at ~64KB,
so large workflows need PNG or WebP; JPEG is also lossy (re-encodes the pixels).

Examples:
    # Embed a workflow into an image, writing <image>.embed.webp
    python embed_workflow.py screenshot.png workflow.json

    # Write a PNG instead (format inferred from the -o extension).
    python embed_workflow.py screenshot.png workflow.json -o out.png

    # Or force the format explicitly.
    python embed_workflow.py screenshot.png workflow.json --format png

    # Overwrite the source image in place.
    python embed_workflow.py preview.webp workflow.json -o preview.webp

    # Also embed an API-format prompt (optional).
    python embed_workflow.py shot.webp workflow.json --prompt prompt.json

    # Check what an image already has embedded.
    python embed_workflow.py preview.webp --read

Any Pillow-readable image works as input (PNG, JPG, WebP, ...). Requires Pillow
(present in a ComfyUI environment):
    pip install pillow
"""

import argparse
import json
import os
import sys
from typing import Any

# EXIF tag IDs ComfyUI uses for WebP metadata.
TAG_MAKE = 0x010F  # ComfyUI stores extra_pnginfo entries here, e.g. "workflow:..."
TAG_MODEL = 0x0110  # ComfyUI stores the API prompt here, e.g. "prompt:..."


def read_embedded(path: str) -> dict[str, str]:
    """Return {key: value_str} of the ComfyUI metadata in an image (WebP or PNG)."""
    from PIL import Image

    img = Image.open(path)
    found: dict[str, str] = {}
    # PNG (and other) tEXt chunks store the raw JSON under 'workflow'/'prompt'.
    info = getattr(img, "text", None) or img.info
    for key in ("workflow", "prompt"):
        val = info.get(key)
        if isinstance(val, str):
            found[key] = val
    # WebP/JPEG EXIF stores 'workflow:<json>' / 'prompt:<json>' in Make/Model.
    exif = img.getexif()
    for tag in (TAG_MAKE, TAG_MODEL):
        val = exif.get(tag)
        if isinstance(val, str) and ":" in val:
            key, _, payload = val.partition(":")
            found.setdefault(key, payload)
    return found


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("image", help="input image (any Pillow-readable format)")
    ap.add_argument(
        "workflow", nargs="?", help="workflow .json to embed (omit with --read)"
    )
    ap.add_argument(
        "-o", "--output", help="output path (default: <image>.embed.<format>)"
    )
    ap.add_argument(
        "--format",
        choices=["webp", "png", "jpg"],
        help="output format (default: inferred from -o extension, else webp)",
    )
    ap.add_argument(
        "--prompt",
        metavar="JSON",
        help="optional API-format prompt .json to also embed",
    )
    ap.add_argument(
        "--lossless",
        dest="lossless",
        action="store_true",
        default=True,
        help="save lossless WebP (default: on; keeps screenshots crisp; WebP only)",
    )
    ap.add_argument(
        "--lossy",
        dest="lossless",
        action="store_false",
        help="save lossy WebP instead (WebP only)",
    )
    ap.add_argument(
        "--quality",
        type=int,
        default=90,
        help="WebP quality 0-100 (default: 90; effort level when lossless; WebP only)",
    )
    ap.add_argument(
        "--read",
        action="store_true",
        help="print the full metadata already embedded in <image> and exit",
    )
    args = ap.parse_args()

    try:
        from PIL import Image
    except ImportError:
        sys.exit("error: this script needs Pillow (pip install pillow)")

    if not os.path.isfile(args.image):
        sys.exit(f"error: no such file: {args.image}")

    if args.read:
        found = read_embedded(args.image)
        if not found:
            print(f"{args.image}: no ComfyUI workflow/prompt metadata found.")
            return
        for key, payload in found.items():
            try:
                pretty = json.dumps(json.loads(payload), indent=2, ensure_ascii=False)
            except ValueError:
                pretty = payload  # not JSON; print the raw string
            print(f"# {args.image}: '{key}' ({len(payload)} chars)")
            print(pretty)
        return

    if not args.workflow:
        ap.error("workflow .json is required unless --read is given")
    if not os.path.isfile(args.workflow):
        sys.exit(f"error: no such file: {args.workflow}")

    with open(args.workflow) as f:
        workflow = json.load(f)
    if not (isinstance(workflow, dict) and "nodes" in workflow):
        print(
            "warning: this JSON has no top-level 'nodes' key — it may be an API "
            "prompt rather than a UI workflow; ComfyUI won't load it by drag-drop.",
            file=sys.stderr,
        )

    prompt = None
    if args.prompt:
        if not os.path.isfile(args.prompt):
            sys.exit(f"error: no such file: {args.prompt}")
        with open(args.prompt) as f:
            prompt = json.load(f)

    # Resolve output format: explicit --format, else the -o extension, else webp.
    fmt = args.format
    if not fmt and args.output:
        fmt = {".png": "png", ".webp": "webp", ".jpg": "jpg", ".jpeg": "jpg"}.get(
            os.path.splitext(args.output)[1].lower()
        )
    fmt = fmt or "webp"

    img: Image.Image = Image.open(args.image)
    img.load()

    out = args.output or os.path.splitext(args.image)[0] + ".embed." + fmt
    if fmt == "png":
        from PIL.PngImagePlugin import PngInfo

        meta = PngInfo()
        meta.add_text("workflow", json.dumps(workflow))
        if prompt is not None:
            meta.add_text("prompt", json.dumps(prompt))
        img.save(out, format="PNG", pnginfo=meta)
    else:
        # WebP and JPEG both carry the metadata in EXIF, the way ComfyUI reads it.
        exif = img.getexif()
        exif[TAG_MAKE] = "workflow:" + json.dumps(workflow)
        if prompt is not None:
            exif[TAG_MODEL] = "prompt:" + json.dumps(prompt)
        exif_bytes = exif.tobytes()
        save_kwargs: dict[str, Any] = {"exif": exif_bytes}
        if fmt == "jpg":
            # EXIF lives in a single APP1 segment capped at ~64KB.
            if len(exif_bytes) > 65533:
                sys.exit(
                    f"error: workflow is too large for JPEG EXIF "
                    f"({len(exif_bytes)} bytes > 64KB limit); use PNG or WebP."
                )
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")  # JPEG can't store alpha/palette
            save_kwargs.update(quality=args.quality)
            img.save(out, format="JPEG", **save_kwargs)
        else:
            if args.lossless:
                save_kwargs.update(
                    lossless=True, quality=args.quality
                )  # quality = effort when lossless
            else:
                save_kwargs.update(quality=args.quality)
            img.save(out, format="WEBP", **save_kwargs)

    n = (
        len(workflow["nodes"])
        if isinstance(workflow, dict) and "nodes" in workflow
        else "?"
    )
    print(
        f"embedded workflow ({n} nodes){' + prompt' if prompt is not None else ''} -> {out}"
    )
    print(
        f"  size: {img.size[0]}x{img.size[1]}, {os.path.getsize(out)} bytes. "
        f"Drag it onto the ComfyUI canvas to load."
    )


if __name__ == "__main__":
    main()
