#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = ["pillow>=12.3.0"]
# ///
"""Split a dataset into training and validation sets without leaking duplicates.

A near-duplicate shared between the two sets quietly breaks validation: as the model
memorizes the training frame its loss on the near-clone falls too, so the validation
curve keeps improving while the model overfits (see docs/monitoring-training.md).
`compare_datasets.py` detects that after the fact — this prevents it.

The trick is to split by *cluster*, not by image. Every image is perceptually hashed
(the same dHash as compare_datasets.py and dedupe_images.py) and images within
--threshold Hamming distance are grouped, transitively. Whole clusters then move to
the validation set, so a frame and its near-twins always land on the same side of the
split and no pair can straddle it. That granularity means the requested count is a
target rather than a guarantee: the tool reports what it actually achieved.

Images move out of the training directory by default, because a held-out image that
is still in the training set is not held out. Use --copy if your master copy lives
elsewhere. Moves prompt for confirmation unless -y or --dry-run.

Usage:
    # Hold out 10% of ./train into ./train.val, reproducibly.
    uv run tools/split_dataset.py --dir ./train --fraction 0.1 --seed 0

    # An exact count, to a directory you name, previewing first.
    uv run tools/split_dataset.py --dir ./train --count 8 --val-dir ./val --dry-run

Requires Pillow. Exits 0 on success, 1 if the confirmation was declined, 2 on bad
arguments.
"""

import argparse
import random
import shutil
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

# Appended to the input directory's name when --val-dir is not given.
_VAL_SUFFIX = ".val"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=HelpFormatter)
    p.add_argument(
        "--dir", required=True, type=Path, help="Training dataset directory."
    )
    p.add_argument(
        "--val-dir",
        type=Path,
        help="Destination for the validation set (default: a sibling '<dir>.val').",
    )
    p.add_argument(
        "--fraction",
        type=float,
        default=0.1,
        help="Share of images to hold out, 0-1. Ignored when --count is given.",
    )
    p.add_argument(
        "--count",
        type=int,
        help="Exact number of images to hold out (overrides --fraction).",
    )
    p.add_argument(
        "--threshold",
        type=int,
        default=5,
        help="Max dHash Hamming distance for two images to count as near-duplicates.",
    )
    p.add_argument(
        "--seed", type=int, default=None, help="Seed for a reproducible split."
    )
    p.add_argument(
        "--copy",
        action="store_true",
        help="Copy into the validation set instead of moving (leaves --dir intact).",
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
        help="Caption file extension moved alongside each image.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Report the planned split without touching the filesystem.",
    )
    p.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt before moving files.",
    )
    return p.parse_args(argv)


def cluster_images(images: list[Path], threshold: int) -> list[list[Path]]:
    """Group *images* into near-duplicate clusters, skipping unreadable files."""
    hashes: dict[Path, int] = {}
    for img in images:
        try:
            with Image.open(img) as im:
                hashes[img] = dhash(im)
        except Exception as e:  # corrupt/unsupported — skip, don't abort the run
            print(f"skip (unreadable): {img.name} ({e})", file=sys.stderr)
    return group_similar(hashes, threshold)


def choose_clusters(
    clusters: list[list[Path]], target: int, rng: random.Random
) -> list[list[Path]]:
    """Pick whole clusters at random until *target* images are held out.

    Clusters that fit inside the remaining budget are taken smallest-first so the
    target is met precisely; the shuffle randomizes the choice among clusters of equal
    size (the sort is stable). If not even the smallest cluster fits, one oversized
    cluster is taken anyway — holding out something beats holding out nothing — but
    never one so large that no training image would remain. Returns [] when the whole
    dataset is a single near-duplicate cluster, i.e. when no leak-free split exists.
    """
    total = sum(len(c) for c in clusters)
    order = list(clusters)
    rng.shuffle(order)
    chosen: list[list[Path]] = []
    held = 0
    for cluster in sorted(order, key=len):
        if held >= target:
            break
        if held + len(cluster) <= target:
            chosen.append(cluster)
            held += len(cluster)
        elif not chosen and len(cluster) < total:
            chosen.append(cluster)
            held += len(cluster)
    return chosen


def transfer(
    image: Path, dest_dir: Path, caption_ext: str, copy: bool, dry: bool
) -> bool:
    """Move or copy *image* and its paired caption into *dest_dir*.

    Returns True if the image had a paired caption.
    """
    caption = paired_caption(image, caption_ext)
    action = "COPY" if copy else "MOVE"
    op = shutil.copy2 if copy else shutil.move
    for src in filter(None, (image, caption)):
        print(f"  {action} {src.name}")
        if not dry:
            op(str(src), str(dest_dir / src.name))
    return caption is not None


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.dir.is_dir():
        print(f"error: --dir is not a directory: {args.dir}", file=sys.stderr)
        return 2
    if args.threshold < 0:
        print("error: --threshold must be >= 0", file=sys.stderr)
        return 2
    if args.count is not None and args.count < 1:
        print("error: --count must be >= 1", file=sys.stderr)
        return 2
    if args.count is None and not 0 < args.fraction < 1:
        print("error: --fraction must be between 0 and 1 (exclusive)", file=sys.stderr)
        return 2

    source = args.dir.resolve()
    val_dir = args.val_dir or source.parent / f"{source.name}{_VAL_SUFFIX}"
    if val_dir.resolve() == source:
        print("error: --val-dir must differ from --dir", file=sys.stderr)
        return 2

    exts = normalize_exts(args.image_exts)
    caption_ext = normalize_ext(args.caption_ext)
    images = list_images(args.dir, exts)
    if len(images) < 2:
        print(
            f"error: need at least 2 images to split, found {len(images)} in {args.dir}",
            file=sys.stderr,
        )
        return 2

    target = (
        args.count if args.count is not None else round(len(images) * args.fraction)
    )
    target = max(1, min(target, len(images) - 1))  # never hold out everything

    clusters = cluster_images(images, args.threshold)
    dupe_clusters = [c for c in clusters if len(c) > 1]
    rng = random.Random(args.seed)
    chosen = choose_clusters(clusters, target, rng)
    held = [img for cluster in chosen for img in sorted(cluster)]
    if not held:
        print(
            "error: every image is a near-duplicate of every other, so no leak-free "
            "split exists; lower --threshold or add more varied images.",
            file=sys.stderr,
        )
        return 2

    print(f"scanned {len(images)} image(s) in {args.dir}")
    print(f"clusters: {len(clusters)} ({len(dupe_clusters)} with near-duplicates)")
    if dupe_clusters:
        print("  near-duplicate cluster(s) kept together:")
        for cluster in dupe_clusters:
            print(f"    {', '.join(p.name for p in cluster)}")
    print(f"target: {target} image(s)  ->  holding out {len(held)}")
    print(f"validation dir: {val_dir}")

    if len(held) != target:
        print(
            f"warning: cluster granularity gave {len(held)} instead of {target}; "
            "lower --threshold to split more finely.",
            file=sys.stderr,
        )

    if not args.dry_run and not args.copy and not args.yes:
        resp = input(f"Move {len(held)} image(s) out of {args.dir}? [y/N] ")
        if resp.strip().lower() not in ("y", "yes"):
            print("aborted; nothing moved.")
            return 1

    if not args.dry_run:
        val_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'[dry-run] ' if args.dry_run else ''}validation set:")
    missing_captions = 0
    for img in held:
        if not transfer(img, val_dir, caption_ext, args.copy, args.dry_run):
            missing_captions += 1

    remaining = len(images) - (0 if args.copy else len(held))
    print("---")
    print(f"train: {remaining}  val: {len(held)}")
    if args.dry_run:
        print("(dry-run; nothing written)")
    if missing_captions:
        print(
            f"warning: {missing_captions} held-out image(s) had no paired "
            f"'{caption_ext}' caption.",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
