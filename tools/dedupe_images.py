#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = ["pillow>=12.3.0"]
# ///
"""Remove duplicate and near-duplicate images (and their captions) from a directory.

Duplicates in a training set silently reweight it: the same frame seen twice pulls
the model twice as hard toward it, which is how a character LoRA ends up locked to
one pose. This finds them in two passes:

  1. Exact duplicates - identical file bytes (SHA-256).
  2. Near duplicates  - perceptual hash (dHash) within --threshold Hamming distance.
                        Set --threshold 0 to catch only images that are identical at
                        hash resolution; higher values catch crops, re-encodes and
                        resizes. Same 64-bit dHash as compare_datasets.py, so the
                        distances from the two tools are comparable.

Within each duplicate group one image is kept (default: highest resolution, then
first name) and the rest are deleted together with their paired captions. Deletions
prompt for confirmation unless -y or --dry-run.

Usage:
    # Report and remove near-duplicates (prompts before deleting).
    uv run tools/dedupe_images.py --dir ./train --threshold 5

    # Byte-identical files only, and show what would go without touching anything.
    uv run tools/dedupe_images.py --dir ./train --exact-only --dry-run

Requires Pillow. Exits 0 on success, 1 if the confirmation was declined, 2 on a bad
--dir.
"""

import argparse
import hashlib
import sys
from pathlib import Path

from dsutils import (
    DEFAULT_CAPTION_EXT,
    DEFAULT_IMAGE_EXTS,
    HelpFormatter,
    dhash,
    group_similar,
    list_images,
    normalize_ext,
    normalize_exts,
    paired_caption,
)
from PIL import Image


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=HelpFormatter)
    p.add_argument("--dir", required=True, type=Path, help="Dataset directory.")
    p.add_argument(
        "--threshold",
        type=int,
        default=5,
        help="Max dHash Hamming distance for near-duplicates (0 = identical only).",
    )
    p.add_argument(
        "--exact-only",
        action="store_true",
        help="Only remove byte-identical files (skip the perceptual pass).",
    )
    p.add_argument(
        "--keep",
        choices=["largest", "first"],
        default="largest",
        help="Which image to keep in each duplicate group.",
    )
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
        help="Report the groups without deleting anything.",
    )
    p.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt before deleting.",
    )
    return p.parse_args(argv)


def sha256(path: Path) -> str:
    """SHA-256 of the file bytes, to spot byte-identical copies."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def pixels(path: Path) -> int:
    """Pixel count of *path*, used to pick the best image in a group."""
    with Image.open(path) as im:
        return im.size[0] * im.size[1]


def choose_keeper(group: list[Path], keep: str) -> Path:
    """Return the one image in *group* to keep."""
    if keep == "first":
        return sorted(group, key=lambda p: p.name)[0]
    # Largest resolution, tie-break by name.
    return sorted(group, key=lambda p: (-pixels(p), p.name))[0]


def group_exact(images: list[Path]) -> list[list[Path]]:
    """Group images by identical file bytes; only groups of 2+ are returned."""
    by_hash: dict[str, list[Path]] = {}
    for img in images:
        by_hash.setdefault(sha256(img), []).append(img)
    return [g for g in by_hash.values() if len(g) > 1]


def group_near(images: list[Path], threshold: int) -> list[list[Path]]:
    """Group images whose dHash is within *threshold*; only groups of 2+ come back."""
    hashes: dict[Path, int] = {}
    for img in images:
        try:
            with Image.open(img) as im:
                hashes[img] = dhash(im)
        except Exception as e:  # corrupt/unsupported — skip, don't abort the run
            print(f"skip (unreadable): {img.name} ({e})", file=sys.stderr)
    return [g for g in group_similar(hashes, threshold) if len(g) > 1]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.dir.is_dir():
        print(f"error: --dir is not a directory: {args.dir}", file=sys.stderr)
        return 2
    if args.threshold < 0:
        print("error: --threshold must be >= 0", file=sys.stderr)
        return 2

    exts = normalize_exts(args.image_exts)
    caption_ext = normalize_ext(args.caption_ext)
    images = list_images(args.dir, exts)
    print(f"scanned {len(images)} image(s) in {args.dir}")

    if args.exact_only:
        groups = group_exact(images)
        label = "exact"
    else:
        groups = group_near(images, args.threshold)
        label = f"near (threshold {args.threshold})"

    to_delete: list[Path] = []
    print(f"\n{label} duplicate group(s): {len(groups)}")
    for group in groups:
        keeper = choose_keeper(group, args.keep)
        losers = [p for p in sorted(group, key=lambda p: p.name) if p != keeper]
        print(f"  keep {keeper.name}  |  remove: {', '.join(p.name for p in losers)}")
        to_delete.extend(losers)

    if not to_delete:
        print("\nno duplicates to remove.")
        return 0

    print(f"\n{len(to_delete)} image(s) marked for removal.")
    if args.dry_run:
        print("(dry-run; nothing deleted)")
        return 0
    if not args.yes:
        resp = input(f"Delete {len(to_delete)} image(s) (and captions)? [y/N] ")
        if resp.strip().lower() not in ("y", "yes"):
            print("aborted; nothing deleted.")
            return 1

    removed = 0
    for img in to_delete:
        cap = paired_caption(img, caption_ext)
        for target in filter(None, (img, cap)):
            target.unlink()
        removed += 1
    print(f"removed {removed} image(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
