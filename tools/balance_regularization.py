#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Balance a regularization directory against a training set for diffusion training.

A diffusion (Dreambooth/LoRA) dataset wants the regularization (class) image count to
match the training image count so neither set overpowers the other: too few class
images and the class prior collapses into your subject, too many and the subject never
lands. This tool makes the regularization directory hold *exactly* as many
image+caption pairs as the training directory, drawing new images randomly from a
separate source pool and trimming any excess.

It is idempotent — running it twice with the same --seed leaves the second run with
nothing to do — so it is safe to re-run after adding training images. Pass --seed for
a reproducible selection; deletions prompt for confirmation unless -y or --dry-run.

Usage:
    # Bring ./reg to the same image count as ./train, drawing from ./pool.
    uv run tools/balance_regularization.py \\
        --train-dir ./train --pool-dir ./pool --reg-dir ./reg --seed 0

    # Preview the plan without touching the filesystem.
    uv run tools/balance_regularization.py \\
        --train-dir ./train --pool-dir ./pool --reg-dir ./reg --dry-run

Pure stdlib. Exits 0 on success, 1 if the deletion prompt was declined, 2 on a bad
directory argument.
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
    list_images,
    normalize_ext,
    normalize_exts,
    paired_caption,
)


def transfer(
    image: Path, dest_dir: Path, caption_ext: str, move: bool, dry_run: bool
) -> bool:
    """Copy or move *image* (and its paired caption) into *dest_dir*.

    Returns True if the image had a paired caption, False otherwise.
    """
    caption = paired_caption(image, caption_ext)
    action = "MOVE" if move else "COPY"
    op = shutil.move if move else shutil.copy2
    for src in filter(None, (image, caption)):
        print(f"  {action} {src.name} -> {dest_dir}/")
        if not dry_run:
            op(str(src), str(dest_dir / src.name))
    return caption is not None


def remove(image: Path, caption_ext: str, dry_run: bool) -> bool:
    """Delete *image* and its paired caption. Returns True if a caption existed."""
    caption = paired_caption(image, caption_ext)
    for target in filter(None, (image, caption)):
        print(f"  DELETE {target.name}")
        if not dry_run:
            target.unlink()
    return caption is not None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=HelpFormatter)
    p.add_argument(
        "--train-dir",
        required=True,
        type=Path,
        help="Training dir; its image count sets the target n.",
    )
    p.add_argument(
        "--pool-dir",
        required=True,
        type=Path,
        help="Source pool of candidate regularization images.",
    )
    p.add_argument(
        "--reg-dir",
        required=True,
        type=Path,
        help="Destination regularization dir to balance.",
    )
    p.add_argument(
        "--seed", type=int, default=None, help="Seed for reproducible random selection."
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
        "--move",
        action="store_true",
        help="Move pool images instead of copying them.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned actions without changing the filesystem.",
    )
    p.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip the confirmation prompt before deletions.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    exts = normalize_exts(args.image_exts)
    caption_ext = normalize_ext(args.caption_ext)

    for label, d in (("train-dir", args.train_dir), ("pool-dir", args.pool_dir)):
        if not d.is_dir():
            print(f"error: --{label} is not a directory: {d}", file=sys.stderr)
            return 2

    if args.seed is not None:
        random.seed(args.seed)

    n = len(list_images(args.train_dir, exts))
    if n == 0:
        print(
            f"error: no images found in --train-dir: {args.train_dir}", file=sys.stderr
        )
        return 2

    if not args.reg_dir.exists():
        print(f"reg-dir does not exist, creating: {args.reg_dir}")
        if not args.dry_run:
            args.reg_dir.mkdir(parents=True, exist_ok=True)
    elif not args.reg_dir.is_dir():
        print(f"error: --reg-dir is not a directory: {args.reg_dir}", file=sys.stderr)
        return 2

    reg_images = list_images(args.reg_dir, exts) if args.reg_dir.is_dir() else []
    current = len(reg_images)

    print(f"target n (train images): {n}")
    print(f"regularization images before: {current}")

    added = removed = 0
    warnings: list[str] = []
    missing_captions = 0

    if current == n:
        print("already balanced; nothing to do.")

    elif current < n:
        need = n - current
        existing_stems = {p.stem for p in reg_images}
        candidates = [
            p for p in list_images(args.pool_dir, exts) if p.stem not in existing_stems
        ]
        if len(candidates) < need:
            warnings.append(
                f"pool has only {len(candidates)} usable image(s) but {need} "
                f"needed; reg-dir will be short by {need - len(candidates)}."
            )
            chosen = candidates
        else:
            chosen = random.sample(candidates, need)

        print(
            f"adding {len(chosen)} image(s) from pool "
            f"({'move' if args.move else 'copy'}):"
        )
        for img in chosen:
            has_caption = transfer(
                img, args.reg_dir, caption_ext, args.move, args.dry_run
            )
            missing_captions += 0 if has_caption else 1
            added += 1

    else:  # current > n
        excess = current - n
        to_remove = random.sample(reg_images, excess)
        print(f"removing {excess} excess image(s) from reg-dir.")
        if not args.dry_run and not args.yes:
            resp = (
                input(
                    f"Delete {excess} image(s) (and captions) from "
                    f"{args.reg_dir}? [y/N] "
                )
                .strip()
                .lower()
            )
            if resp not in ("y", "yes"):
                print("aborted; no files deleted.")
                return 1
        for img in to_remove:
            has_caption = remove(img, caption_ext, args.dry_run)
            missing_captions += 0 if has_caption else 1
            removed += 1

    if missing_captions:
        warnings.append(
            f"{missing_captions} affected image(s) had no paired "
            f"'{caption_ext}' caption."
        )

    final = current + added - removed
    print("---")
    print(f"added: {added}  removed: {removed}")
    print(
        f"regularization images after: {final}"
        + ("  (dry-run; not written)" if args.dry_run else "")
    )
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
