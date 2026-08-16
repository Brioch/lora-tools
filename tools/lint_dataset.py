#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = ["pillow>=12.3.0"]
# ///
"""Validate a diffusion-training dataset directory before training.

Most wasted training runs are traceable to something in the dataset that a five
second check would have caught: an image the loader cannot decode, a caption file
that was never written, a stray caption left behind after its image was deleted.
This walks a directory of image + caption pairs and reports them, split by how much
they matter:

  ERRORS   (exit 1)   corrupt/unreadable image, image with no caption, empty caption
  WARNINGS (exit 0)   orphan caption (no image), non-RGB mode, image below --min-size,
                      extension not matching the real image format, near-blank image,
                      caption missing the --trigger word

Two checks are worth calling out:

  * --trigger WORD catches the mistake that silently wastes a whole run: training with
    a trigger word that some captions spell differently, or omit. The word is matched
    as a whole tag, case-insensitively, using the same tokenizer as edit_captions.py.
  * near-blank images are caught by grayscale dynamic range (--min-contrast). A frame
    that is flat — a blown-out white background, a solid fill, a fully transparent PNG
    flattened to one colour — contributes nothing but still costs a training slot.

Because it exits non-zero on errors it doubles as a pre-training gate. Pair it with
`dedupe_images.py` (duplicate frames), `caption_stats.py` (tag distribution) and
`prepare_images.py` (fixing what it flags).

Usage:
    # Gate a dataset: exits 1 if anything is actually broken.
    uv run tools/lint_dataset.py --dir ./train

    # Also warn about small images, and accept images without captions.
    uv run tools/lint_dataset.py --dir ./train --min-size 1024 --no-require-caption

    # Check every caption carries the trigger word you plan to train with.
    uv run tools/lint_dataset.py --dir ./train --trigger mychar

Requires Pillow. Exits 0 when clean or warnings-only, 1 on errors, 2 on a bad --dir.
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path

from dsutils import (
    DEFAULT_CAPTION_EXT,
    DEFAULT_IMAGE_EXTS,
    HelpFormatter,
    list_images,
    normalize_ext,
    normalize_exts,
    paired_caption,
    split_tags,
)
from PIL import Image

# Suffix -> Pillow format name, to detect extension/format mismatches.
_EXT_FORMAT: dict[str, str] = {
    ".png": "PNG",
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".webp": "WEBP",
    ".bmp": "BMP",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=HelpFormatter)
    p.add_argument("--dir", required=True, type=Path, help="Dataset directory.")
    p.add_argument(
        "--image-exts",
        nargs="+",
        default=DEFAULT_IMAGE_EXTS,
        help="Image extensions to consider (case-insensitive).",
    )
    p.add_argument(
        "--caption-ext",
        default=DEFAULT_CAPTION_EXT,
        help="Caption file extension paired with each image.",
    )
    p.add_argument(
        "--min-size",
        type=int,
        default=0,
        help="Warn if an image's shortest edge is below this (px). 0 = off.",
    )
    p.add_argument(
        "--no-require-caption",
        action="store_true",
        help="Treat a missing caption as a warning instead of an error.",
    )
    p.add_argument(
        "--trigger",
        metavar="WORD",
        help="Warn about captions missing this trigger word (matched as a whole tag, "
        "case-insensitively).",
    )
    p.add_argument(
        "--min-contrast",
        type=int,
        default=8,
        help="Warn if an image's grayscale range is below this (0-255). 0 = off.",
    )
    return p.parse_args(argv)


def grayscale_range(im: Image.Image) -> int:
    """Span between the darkest and lightest grey level present in *im*, 0-255.

    0 means every pixel is the same shade — a solid fill, a blown-out background, or a
    transparent PNG flattened to one colour. Read off the histogram rather than
    getextrema() so the result is a plain int for any input mode.
    """
    used = [level for level, count in enumerate(im.convert("L").histogram()) if count]
    return used[-1] - used[0] if used else 0


def report(title: str, groups: dict[str, list[str]]) -> int:
    """Print *groups* under *title* and return the total number of entries."""
    total = sum(len(v) for v in groups.values())
    if not total:
        return 0
    print(f"\n{title} ({total}):")
    for cat, files in sorted(groups.items()):
        print(f"  {cat}: {len(files)}")
        for f in files:
            print(f"    - {f}")
    return total


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.dir.is_dir():
        print(f"error: --dir is not a directory: {args.dir}", file=sys.stderr)
        return 2

    exts = normalize_exts(args.image_exts)
    caption_ext = normalize_ext(args.caption_ext)
    trigger = args.trigger.strip().lower() if args.trigger else None
    images = list_images(args.dir, exts)

    errors: defaultdict[str, list[str]] = defaultdict(list)  # category -> [file, ...]
    warnings: defaultdict[str, list[str]] = defaultdict(list)

    image_stems: set[str] = set()
    for img in images:
        image_stems.add(img.stem)

        # Caption presence / emptiness / trigger word.
        cap = paired_caption(img, caption_ext)
        if cap is None:
            bucket = warnings if args.no_require_caption else errors
            bucket["missing caption"].append(img.name)
        else:
            text = cap.read_text(encoding="utf-8", errors="replace")
            if not text.strip():
                errors["empty caption"].append(img.name)
            elif trigger and trigger not in {t.lower() for t in split_tags(text)}:
                warnings["trigger missing"].append(img.name)

        # Decode the image to catch corruption; then inspect mode/size/format.
        try:
            with Image.open(img) as im:
                im.load()
                mode, (w, h) = im.mode, im.size
                fmt = im.format
                contrast = grayscale_range(im)
        except Exception as e:  # any decode failure means the image is unusable
            errors["corrupt image"].append(f"{img.name} ({e})")
            continue

        if mode not in ("RGB", "L"):
            warnings["non-RGB mode"].append(f"{img.name} ({mode})")
        if args.min_size and min(w, h) < args.min_size:
            warnings["below min-size"].append(f"{img.name} ({w}x{h})")
        if args.min_contrast and contrast < args.min_contrast:
            warnings["near-blank image"].append(f"{img.name} (range {contrast})")
        expected = _EXT_FORMAT.get(img.suffix.lower())
        if expected and fmt and fmt != expected:
            warnings["extension mismatch"].append(f"{img.name} (is {fmt})")

    # Orphan captions: caption files whose stem has no image.
    for cap_file in sorted(args.dir.glob(f"*{caption_ext}")):
        if cap_file.stem not in image_stems:
            warnings["orphan caption"].append(cap_file.name)

    print(f"scanned {len(images)} image(s) in {args.dir}")
    n_err = report("ERRORS", errors)
    n_warn = report("WARNINGS", warnings)
    if not n_err and not n_warn:
        print("clean: no issues found.")
    print(f"\nsummary: {n_err} error(s), {n_warn} warning(s)")
    return 1 if n_err else 0


if __name__ == "__main__":
    raise SystemExit(main())
