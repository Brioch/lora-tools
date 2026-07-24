#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = ["pillow>=12.3.0"]
# ///
"""Strip all embedded metadata from an image — EXIF blocks and PNG/WebP text
chunks alike — writing a clean copy that carries only pixels.

Handy for removing a ComfyUI workflow/prompt (see embed_workflow.py) or any
stray EXIF (camera info, GPS, ...) before sharing an image. The pixel data is
re-saved into a fresh image with an empty info dict, so nothing metadata-shaped
survives. Verify with `python embed_workflow.py <image> --read`.

Examples:
    # Strip metadata, writing <image>.stripped.<ext> next to the input.
    python strip_metadata.py preview.webp

    # Overwrite the original in place.
    python strip_metadata.py preview.webp --in-place

    # Choose an output path (format inferred from its extension).
    python strip_metadata.py shot.png -o clean.webp

Any Pillow-readable image works as input. Requires Pillow (present in a ComfyUI
environment):
    pip install pillow
"""

import argparse
import os
import sys
from typing import Any

# Formats we can name via an extension; anything else falls back to the input's.
EXT_FORMAT: dict[str, str] = {
    ".png": "PNG",
    ".webp": "WEBP",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
}


def strip(img: Any) -> Any:
    """Return a copy of img holding only its pixels — no info/EXIF/text chunks."""
    from PIL import Image

    clean = Image.new(img.mode, img.size)
    clean.paste(img)
    if img.mode == "P":
        clean.putpalette(img.getpalette())
    return clean


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("image", help="input image (any Pillow-readable format)")
    ap.add_argument(
        "-o", "--output", help="output path (default: <image>.stripped.<ext>)"
    )
    ap.add_argument(
        "--in-place",
        action="store_true",
        help="overwrite the input (mutually exclusive with -o)",
    )
    ap.add_argument(
        "--quality",
        type=int,
        default=90,
        help="quality 0-100 for lossy output like JPEG (default: 90)",
    )
    args = ap.parse_args()

    if args.output and args.in_place:
        ap.error("--in-place and -o/--output are mutually exclusive")

    try:
        from PIL import Image
    except ImportError:
        sys.exit("error: this script needs Pillow (pip install pillow)")

    if not os.path.isfile(args.image):
        sys.exit(f"error: no such file: {args.image}")

    src = Image.open(args.image)
    src.load()
    in_fmt = src.format  # e.g. "PNG", "WEBP" — before we discard the original

    if args.in_place:
        out = args.image
        out_fmt = in_fmt
    else:
        out = args.output or (
            os.path.splitext(args.image)[0]
            + ".stripped"
            + (os.path.splitext(args.image)[1] or ".png")
        )
        out_fmt = EXT_FORMAT.get(os.path.splitext(out)[1].lower()) or in_fmt or "PNG"

    clean = strip(src)

    save_kwargs: dict[str, Any] = {}
    if out_fmt == "WEBP":
        save_kwargs.update(lossless=True, quality=args.quality)  # avoid re-encode loss
    elif out_fmt == "JPEG":
        save_kwargs.update(quality=args.quality)
    clean.save(out, format=out_fmt, **save_kwargs)

    before = os.path.getsize(args.image)
    after = os.path.getsize(out)
    print(f"stripped metadata -> {out}")
    print(f"  {clean.size[0]}x{clean.size[1]} {out_fmt}, {before} -> {after} bytes.")


if __name__ == "__main__":
    main()
