#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = ["pillow>=12.3.0"]
# ///
"""Normalize dataset images for diffusion training.

Trainers choke on, or silently mangle, the odds and ends a scraped dataset carries:
sideways phone photos (EXIF rotation), transparent PNGs that composite to black,
palette or grayscale modes, and 6000px originals that cost VRAM without adding
detail. For every image this can:

  - apply EXIF orientation then strip metadata (fixes sideways photos),
  - flatten transparency / palette / grayscale to RGB (over a white background),
  - convert to a single target format (png or jpg),
  - downscale so the longest edge is <= --max-edge (never upscales).

It never touches the input unless you ask it to. By default the results go to a
sibling `<dir>.prepared/` directory with paired captions copied along; --out-dir
picks that destination explicitly. Only --in-place rewrites the input directory, and
because a format change there also deletes the old-extension original, that mode is
irreversible — keep a backup, or preview it with --dry-run first.

Usage:
    # Normalize ./raw into ./raw.prepared/, leaving ./raw untouched.
    uv run tools/prepare_images.py --dir ./raw --format jpg --max-edge 1536

    # Same, but choose the destination.
    uv run tools/prepare_images.py --dir ./raw --out-dir ./train --format png

    # Rewrite the dataset in place (destructive; --dry-run to preview).
    uv run tools/prepare_images.py --dir ./train --max-edge 1536 --in-place

Requires Pillow. Exits 1 if any image failed to process, 2 on a bad --dir or
conflicting destination flags.
"""

import argparse
import shutil
import sys
from pathlib import Path

from dsutils import (
    DEFAULT_CAPTION_EXT,
    DEFAULT_IMAGE_EXTS,
    HelpFormatter,
    list_images,
    normalize_ext,
    normalize_exts,
    paired_caption,
)
from PIL import Image, ImageOps

_FORMAT_EXT: dict[str, str] = {"png": ".png", "jpg": ".jpg"}
_FORMAT_SAVE: dict[str, str] = {"png": "PNG", "jpg": "JPEG"}
# Appended to the input directory's name when neither --out-dir nor --in-place is given.
_PREPARED_SUFFIX = ".prepared"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=HelpFormatter)
    p.add_argument("--dir", required=True, type=Path, help="Input directory.")
    p.add_argument(
        "--out-dir",
        type=Path,
        help="Destination directory (default: a sibling '<dir>.prepared'). "
        "Paired captions are copied along.",
    )
    p.add_argument(
        "--in-place",
        action="store_true",
        help="Rewrite the input directory instead of writing to a new one. "
        "Destructive: a format change deletes the old-extension original.",
    )
    p.add_argument(
        "--format",
        choices=["png", "jpg"],
        help="Convert every image to this format (default: keep each as-is).",
    )
    p.add_argument(
        "--max-edge",
        type=int,
        default=0,
        help="Downscale so the longest edge is <= this (px). 0 = off, never upscales.",
    )
    p.add_argument(
        "--no-flatten",
        action="store_true",
        help="Keep the original mode instead of forcing RGB.",
    )
    p.add_argument("--quality", type=int, default=95, help="JPEG quality.")
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
        "--dry-run",
        action="store_true",
        help="Print what would be written without touching the filesystem.",
    )
    return p.parse_args(argv)


def flatten_rgb(im: Image.Image) -> Image.Image:
    """Composite any transparency over white and return an RGB image."""
    if im.mode == "RGB":
        return im
    if im.mode in ("RGBA", "LA") or (im.mode == "P" and "transparency" in im.info):
        rgba = im.convert("RGBA")
        bg = Image.new("RGB", rgba.size, (255, 255, 255))
        bg.paste(rgba, mask=rgba.split()[-1])
        return bg
    return im.convert("RGB")


def destination_dir(source: Path, out_dir: Path | None, in_place: bool) -> Path:
    """Return the directory to write into — *source* itself only when in_place."""
    if in_place:
        return source
    if out_dir is not None:
        return out_dir
    return source.parent / f"{source.name}{_PREPARED_SUFFIX}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.dir.is_dir():
        print(f"error: --dir is not a directory: {args.dir}", file=sys.stderr)
        return 2
    if args.max_edge < 0:
        print("error: --max-edge must be >= 0", file=sys.stderr)
        return 2
    if args.in_place and args.out_dir:
        print("error: --in-place and --out-dir are mutually exclusive", file=sys.stderr)
        return 2

    source = args.dir.resolve()
    out_dir = destination_dir(source, args.out_dir, args.in_place)
    # Writing into the input directory is in-place work; make the caller say so.
    if not args.in_place and out_dir.resolve() == source:
        print(
            "error: --out-dir is the input directory; pass --in-place to rewrite it",
            file=sys.stderr,
        )
        return 2
    if not args.in_place:
        print(f"writing to: {out_dir}")
        if not args.dry_run:
            out_dir.mkdir(parents=True, exist_ok=True)

    exts = normalize_exts(args.image_exts)
    caption_ext = normalize_ext(args.caption_ext)
    dest_ext = _FORMAT_EXT[args.format] if args.format else None

    processed = failed = 0
    for img in list_images(args.dir, exts):
        try:
            with Image.open(img) as src:
                # exif_transpose returns None only for a None input; `or src` keeps
                # mypy happy and leaves images without EXIF orientation untouched.
                im: Image.Image = ImageOps.exif_transpose(src) or src
                if not args.no_flatten:
                    im = flatten_rgb(im)
                if args.max_edge and max(im.size) > args.max_edge:
                    im.thumbnail(
                        (args.max_edge, args.max_edge), Image.Resampling.LANCZOS
                    )

                dest = out_dir / (img.stem + (dest_ext or img.suffix))
                save_fmt = _FORMAT_SAVE.get(args.format) if args.format else None
                if save_fmt == "JPEG" and im.mode != "RGB":
                    im = im.convert("RGB")

                print(
                    f"{'[dry-run] ' if args.dry_run else ''}{img.name} -> {dest.name}"
                    f" ({im.mode}, {im.size[0]}x{im.size[1]})"
                )

                if not args.dry_run:
                    if save_fmt == "JPEG":
                        im.save(dest, format=save_fmt, quality=args.quality)
                    else:
                        im.save(dest, format=save_fmt)
                    # In-place format change: remove the old-extension original.
                    if args.in_place and dest != img:
                        img.unlink()
                    # Copy the caption along when writing to a separate directory.
                    if not args.in_place:
                        cap = paired_caption(img, caption_ext)
                        if cap:
                            shutil.copy2(cap, out_dir / cap.name)
            processed += 1
        except Exception as e:  # corrupt/unsupported — report and keep going
            failed += 1
            print(f"failed: {img.name} ({e})", file=sys.stderr)

    print(
        f"\n{'would process' if args.dry_run else 'processed'} {processed} image(s)"
        + (f", {failed} failed" if failed else "")
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
