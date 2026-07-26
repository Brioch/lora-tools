#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Batch-edit the caption `.txt` files in a dataset directory.

Captions are treated as comma-separated tag lists (the common diffusion format), so
the edits are tag-aware rather than plain text substitution: replacing `man` will not
mangle `woman`, and adding a trigger word twice is a no-op. Operations are applied in
this order to every caption file:

    1. --replace OLD NEW   (rename a whole tag OLD to NEW, case-insensitive; repeatable)
    2. --remove TAG        (drop matching tag, case-insensitive; repeatable)
    3. --prepend TAG       (insert at the front if absent, e.g. a trigger word; repeatable)
    4. --add TAG           (append at the end if absent; repeatable)
    5. --dedupe            (remove duplicate tags, keeping first occurrence)
    6. --sort              (sort tags alphabetically)

Tag matching for --remove/--add/--prepend/--dedupe is case-insensitive; the original
casing of kept tags is preserved. Every run prints a before/after diff of the files it
touches, and --dry-run prints that diff without writing.

Usage:
    # Add a trigger word to the front of every caption and drop duplicate tags.
    uv run tools/edit_captions.py --dir ./train --prepend "mytoken" --dedupe

    # Clean up unwanted tags and rename one.
    uv run tools/edit_captions.py --dir ./train --remove "blurry" --replace "man" "person"

Pure stdlib. Exits 0 on success, 2 on a bad --dir or when no operation was given.
"""

import argparse
import sys
from pathlib import Path

from dsutils import DEFAULT_CAPTION_EXT, HelpFormatter, normalize_ext


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=HelpFormatter)
    p.add_argument("--dir", required=True, type=Path, help="Dataset directory.")
    p.add_argument(
        "--caption-ext",
        default=DEFAULT_CAPTION_EXT,
        help="Caption file extension to edit.",
    )
    p.add_argument(
        "--replace",
        nargs=2,
        action="append",
        metavar=("OLD", "NEW"),
        default=[],
        help="Rename a whole tag OLD to NEW (case-insensitive). Repeatable.",
    )
    p.add_argument(
        "--remove",
        action="append",
        default=[],
        metavar="TAG",
        help="Remove a tag (case-insensitive). Repeatable.",
    )
    p.add_argument(
        "--prepend",
        action="append",
        default=[],
        metavar="TAG",
        help="Prepend a tag if absent (e.g. a trigger word). Repeatable.",
    )
    p.add_argument(
        "--add",
        action="append",
        default=[],
        metavar="TAG",
        help="Append a tag if absent. Repeatable.",
    )
    p.add_argument("--dedupe", action="store_true", help="Drop duplicate tags.")
    p.add_argument("--sort", action="store_true", help="Sort tags alphabetically.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the before/after diff without writing.",
    )
    return p.parse_args(argv)


def split_tags(text: str) -> list[str]:
    """Split a caption into its non-empty, stripped comma-separated tags."""
    return [t.strip() for t in text.split(",") if t.strip()]


def transform(text: str, args: argparse.Namespace) -> str:
    """Apply every requested tag operation to *text* and return the new caption."""
    tags = split_tags(text)

    if args.replace:
        rename = {old.lower(): new for old, new in args.replace}
        tags = [rename.get(t.lower(), t) for t in tags]

    if args.remove:
        drop = {t.lower() for t in args.remove}
        tags = [t for t in tags if t.lower() not in drop]

    for tag in reversed(args.prepend):  # reversed so listed order is kept at the front
        if tag.lower() not in {t.lower() for t in tags}:
            tags.insert(0, tag)

    for tag in args.add:
        if tag.lower() not in {t.lower() for t in tags}:
            tags.append(tag)

    if args.dedupe:
        seen: set[str] = set()
        out: list[str] = []
        for t in tags:
            if t.lower() not in seen:
                seen.add(t.lower())
                out.append(t)
        tags = out

    if args.sort:
        tags = sorted(tags, key=str.lower)

    return ", ".join(tags)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.dir.is_dir():
        print(f"error: --dir is not a directory: {args.dir}", file=sys.stderr)
        return 2

    if not any(
        [args.replace, args.remove, args.prepend, args.add, args.dedupe, args.sort]
    ):
        print("error: no edit operation specified (nothing to do).", file=sys.stderr)
        return 2

    caption_ext = normalize_ext(args.caption_ext)
    files = sorted(args.dir.glob(f"*{caption_ext}"))

    changed = 0
    for f in files:
        original = f.read_text(encoding="utf-8")
        updated = transform(original, args)
        if updated != original.strip():
            changed += 1
            print(f"{'[dry-run] ' if args.dry_run else ''}{f.name}")
            print(f"  - {original.strip()}")
            print(f"  + {updated}")
            if not args.dry_run:
                f.write_text(updated + "\n", encoding="utf-8")

    print(
        f"\n{'would change' if args.dry_run else 'changed'} {changed} "
        f"of {len(files)} caption file(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
