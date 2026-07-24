#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = ["pillow>=12.3.0"]
# ///
"""Compare two image folders and flag validation images that duplicate — exactly
or near-exactly — an image in the training set.

A near-duplicate shared between your training and validation sets quietly breaks
validation: as the model memorizes the training frame, its loss on the near-clone
drops too, so the validation curve keeps falling even while it overfits (see
docs/monitoring-training.md). This finds those clashes so you can drop them from
the validation set.

Matching uses a perceptual hash (dHash by default): each image is reduced to an
NxN-bit fingerprint and compared by Hamming distance, so resizes, re-compression
and minor edits still match. Distance 0 means visually identical at hash
resolution; a byte-identical file is labelled IDENTICAL. Lower threshold = stricter
(fewer flagged); a rough guide for the default 64-bit hash: 0 identical, 1-10
near-duplicate, higher genuinely distinct.

Examples:
    # Flag validation images too close to any training image.
    python compare_datasets.py train/ val/

    # Stricter or looser matching (lower = stricter).
    python compare_datasets.py train/ val/ --threshold 5

    # Show the closest training match for every validation image, not just clashes.
    python compare_datasets.py train/ val/ --show-all

Requires Pillow (pip install pillow). Exits non-zero if any clash is found, so it
doubles as a pre-training gate.
"""

import argparse
import hashlib
import os
import sys
from typing import Any

IMAGE_EXTS: set[str] = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tif",
    ".tiff",
    ".gif",
}


def list_images(folder: str, recursive: bool = False) -> list[str]:
    """Return sorted image paths in folder (optionally recursing)."""
    if not os.path.isdir(folder):
        sys.exit(f"error: not a directory: {folder}")
    paths: list[str] = []
    if recursive:
        for root, _, files in os.walk(folder):
            paths += [
                os.path.join(root, f)
                for f in files
                if os.path.splitext(f)[1].lower() in IMAGE_EXTS
            ]
    else:
        paths = [
            os.path.join(folder, f)
            for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTS
            and os.path.isfile(os.path.join(folder, f))
        ]
    return sorted(paths)


def dhash(image: Any, size: int = 8) -> int:
    """Difference hash: compare each pixel to its right neighbour. size*size bits."""
    from PIL import Image

    small = image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    px = small.tobytes()  # one byte per pixel in "L" mode
    bits = 0
    for row in range(size):
        base = row * (size + 1)
        for col in range(size):
            bits = (bits << 1) | (1 if px[base + col] > px[base + col + 1] else 0)
    return bits


def ahash(image: Any, size: int = 8) -> int:
    """Average hash: compare each pixel to the mean. size*size bits."""
    small = image.convert("L").resize((size, size))
    px = small.tobytes()  # one byte per pixel in "L" mode
    avg = sum(px) / len(px)
    bits = 0
    for p in px:
        bits = (bits << 1) | (1 if p >= avg else 0)
    return bits


def hamming(a: int, b: int) -> int:
    """Number of differing bits between two integer fingerprints."""
    return bin(a ^ b).count("1")


def file_sha256(path: str) -> str:
    """SHA-256 of the file bytes, to spot byte-identical copies."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_fingerprints(
    paths: list[str], algo: str, size: int
) -> list[tuple[str, int, str]]:
    """Return [(path, perceptual_hash, sha256)], skipping unreadable images."""
    from PIL import Image

    out: list[tuple[str, int, str]] = []
    for p in paths:
        try:
            with Image.open(p) as img:
                img.load()
                phash = dhash(img, size) if algo == "dhash" else ahash(img, size)
        except Exception as e:  # corrupt/unsupported — skip, don't abort the run
            print(f"warning: skipping {p}: {e}", file=sys.stderr)
            continue
        out.append((p, phash, file_sha256(p)))
    return out


def classify(distance: int, identical_file: bool, threshold: int) -> str:
    if identical_file:
        return "IDENTICAL"
    if distance == 0:
        return "identical-image"
    if distance <= threshold:
        return "near-dupe"
    return "distinct"


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("train_dir", help="folder of training images")
    ap.add_argument("val_dir", help="folder of validation images to check")
    ap.add_argument(
        "--threshold",
        type=int,
        default=10,
        help="max Hamming distance to flag as a clash (default 10; "
        "lower = stricter, 0 = only identical)",
    )
    ap.add_argument(
        "--hash",
        dest="algo",
        choices=["dhash", "ahash"],
        default="dhash",
        help="perceptual hash algorithm (default dhash)",
    )
    ap.add_argument(
        "--size", type=int, default=8, help="hash size; NxN bits (default 8 -> 64-bit)"
    )
    ap.add_argument(
        "-r", "--recursive", action="store_true", help="recurse into subdirectories"
    )
    ap.add_argument(
        "--show-all",
        action="store_true",
        help="print the closest training match for every validation image",
    )
    args = ap.parse_args()

    try:
        from PIL import Image  # noqa: F401 — imported for the friendly error only
    except ImportError:
        sys.exit("error: this script needs Pillow (pip install pillow)")

    if args.threshold < 0:
        ap.error("--threshold must be >= 0")
    if args.size < 2:
        ap.error("--size must be >= 2")

    train = list_images(args.train_dir, args.recursive)
    val = list_images(args.val_dir, args.recursive)
    if not train:
        sys.exit(f"error: no images found in {args.train_dir}")
    if not val:
        sys.exit(f"error: no images found in {args.val_dir}")

    train_fp = load_fingerprints(train, args.algo, args.size)
    val_fp = load_fingerprints(val, args.algo, args.size)
    if not train_fp or not val_fp:
        sys.exit("error: no readable images to compare")

    bits = args.size * args.size
    rows: list[tuple[str, str, int, bool]] = []  # (val, train, distance, identical)
    for vp, vh, vsha in val_fp:
        tp, th, tsha = min(train_fp, key=lambda t: hamming(vh, t[1]))
        rows.append((vp, tp, hamming(vh, th), vsha == tsha))
    rows.sort(key=lambda r: r[2])

    print(
        f"Comparing {len(train_fp)} training vs {len(val_fp)} validation images "
        f"({args.algo}, {bits}-bit, threshold {args.threshold})\n"
    )

    shown = rows if args.show_all else [r for r in rows if r[2] <= args.threshold]
    for vp, tp, d, idf in shown:
        vrel = os.path.relpath(vp, args.val_dir)
        trel = os.path.relpath(tp, args.train_dir)
        print(
            f"  [{classify(d, idf, args.threshold):15}] {vrel}  <->  {trel}  (distance {d})"
        )

    clashes = [r for r in rows if r[2] <= args.threshold]
    print()
    if clashes:
        print(
            f"{len(clashes)} of {len(val_fp)} validation images clash with training "
            f"data (distance <= {args.threshold}). Remove them from the validation set."
        )
        sys.exit(1)
    print(
        f"No clashes: all {len(val_fp)} validation images are distinct from the "
        f"{len(train_fp)} training images (closest distance {rows[0][2]}, "
        f"threshold {args.threshold})."
    )


if __name__ == "__main__":
    main()
