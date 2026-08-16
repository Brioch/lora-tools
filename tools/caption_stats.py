#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Report the tag distribution across a dataset's caption files.

The read-only counterpart to `edit_captions.py`: that tool changes tags, this one
shows you which ones are worth changing. Captions are read as comma-separated tag
lists (the same tokenizer `edit_captions` uses, so the counts match what an edit
would act on) and the report calls out the two ends of the distribution that
actually drive a decision:

  * ubiquitous tags — present in nearly every caption. They carry no discriminative
    signal and compete with your trigger word for the concept, so they are usually
    worth dropping — unless one of them *is* the trigger word, which is exactly where
    you want to see 100%. --ubiquitous-pct sets the cutoff.
  * rare tags — one or two occurrences. Usually tagger noise or typos ("bluu eyes"),
    which bloat the vocabulary without ever being learned; --rare sets the cutoff.

Both lists are the input to a follow-up `edit_captions --remove` or `--replace` run.
Tags are counted case-insensitively (matching how `edit_captions` matches), and each
is reported under its most common spelling.

Usage:
    # What is in these captions?
    uv run tools/caption_stats.py --dir ./train

    # Widen the report, and treat a tag with 3 or fewer occurrences as rare.
    uv run tools/caption_stats.py --dir ./train --top 50 --rare 3

    # Machine-readable, for diffing two datasets or scripting an edit.
    uv run tools/caption_stats.py --dir ./train --json

Pure stdlib. Exits 0 on success, 2 on a bad --dir.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import TypedDict

from dsutils import DEFAULT_CAPTION_EXT, HelpFormatter, normalize_ext, split_tags


class TagStat(TypedDict):
    """One tag's frequency, by captions containing it and by total occurrences."""

    tag: str
    captions: int
    occurrences: int
    pct: float


class Spread(TypedDict):
    min: int
    median: int
    max: int


class Stats(TypedDict):
    captions: int
    empty: int
    instances: int
    vocabulary: int
    tags_per_caption: Spread
    tags: list[TagStat]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=HelpFormatter)
    p.add_argument("--dir", required=True, type=Path, help="Dataset directory.")
    p.add_argument(
        "--caption-ext",
        default=DEFAULT_CAPTION_EXT,
        help="Caption file extension to read.",
    )
    p.add_argument(
        "--top",
        type=int,
        default=25,
        help="How many of the most frequent tags to list.",
    )
    p.add_argument(
        "--rare",
        type=int,
        default=2,
        help="Report tags occurring this many times or fewer. 0 = skip.",
    )
    p.add_argument(
        "--ubiquitous-pct",
        type=float,
        default=90.0,
        help="Report tags present in at least this %% of captions. 0 = skip.",
    )
    p.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    return p.parse_args(argv)


def summarise(captions: list[list[str]]) -> Stats:
    """Fold tokenized *captions* into the counts and spreads worth reporting."""
    instances: Counter[str] = Counter()  # tag -> total occurrences
    documents: Counter[str] = Counter()  # tag -> captions containing it
    spellings: dict[str, Counter[str]] = {}

    for tags in captions:
        for tag in tags:
            key = tag.lower()
            instances[key] += 1
            spellings.setdefault(key, Counter())[tag] += 1
        for key in {t.lower() for t in tags}:
            documents[key] += 1

    lengths = sorted(len(t) for t in captions)
    # Ranked by document frequency: "in how many captions" is what decides whether a
    # tag is doing work, not how often it repeats inside one caption.
    ranked = sorted(documents.items(), key=lambda kv: (-kv[1], kv[0]))
    return Stats(
        captions=len(captions),
        empty=sum(1 for t in captions if not t),
        instances=sum(instances.values()),
        vocabulary=len(instances),
        tags_per_caption=Spread(
            min=lengths[0] if lengths else 0,
            median=lengths[len(lengths) // 2] if lengths else 0,
            max=lengths[-1] if lengths else 0,
        ),
        tags=[
            TagStat(
                tag=spellings[key].most_common(1)[0][0],
                captions=count,
                occurrences=instances[key],
                pct=100.0 * count / len(captions) if captions else 0.0,
            )
            for key, count in ranked
        ],
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.dir.is_dir():
        print(f"error: --dir is not a directory: {args.dir}", file=sys.stderr)
        return 2

    caption_ext = normalize_ext(args.caption_ext)
    files = sorted(args.dir.glob(f"*{caption_ext}"))
    captions = [
        split_tags(f.read_text(encoding="utf-8", errors="replace")) for f in files
    ]
    stats = summarise(captions)
    tags = stats["tags"]

    ubiquitous = [
        t for t in tags if args.ubiquitous_pct > 0 and t["pct"] >= args.ubiquitous_pct
    ]
    # The two lists are kept disjoint: on a small dataset a trigger word can sit under
    # the absolute --rare cutoff while still being in every caption, and calling that
    # "tagger noise" would be actively misleading.
    everywhere = {t["tag"] for t in ubiquitous}
    rare = [
        t
        for t in tags
        if args.rare > 0
        and t["occurrences"] <= args.rare
        and t["tag"] not in everywhere
    ]

    if args.json:
        print(
            json.dumps(
                {"dir": str(args.dir), **stats, "ubiquitous": ubiquitous, "rare": rare},
                indent=2,
            )
        )
        return 0

    print(f"scanned {len(files)} caption file(s) in {args.dir}")
    if not files:
        print(f"no '{caption_ext}' files found.")
        return 0

    per = stats["tags_per_caption"]
    print(f"\ncaptions      : {stats['captions']} ({stats['empty']} empty)")
    print(f"tag instances : {stats['instances']}")
    print(f"vocabulary    : {stats['vocabulary']} unique tag(s)")
    print(f"tags/caption  : min {per['min']}  median {per['median']}  max {per['max']}")

    if tags:
        shown = tags[: args.top]
        print(f"\ntop {len(shown)} tag(s) by caption count:")
        print(f"  {'captions':>8} {'%':>5}  {'total':>5}  tag")
        for t in shown:
            print(
                f"  {t['captions']:>8} {t['pct']:>5.1f}  "
                f"{t['occurrences']:>5}  {t['tag']}"
            )
        if len(tags) > len(shown):
            print(f"  … (+{len(tags) - len(shown)} more; --top to widen)")

    if ubiquitous:
        print(
            f"\nubiquitous (in >= {args.ubiquitous_pct:g}% of captions) — no "
            "discriminative signal; drop them unless one is your trigger word:"
        )
        for t in ubiquitous:
            print(f"  {t['tag']}  ({t['captions']} captions, {t['pct']:.1f}%)")

    if rare:
        print(f"\nrare (<= {args.rare} occurrence(s)) — often tagger noise or typos:")
        for t in rare:
            print(f"  {t['tag']}  ({t['occurrences']})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
