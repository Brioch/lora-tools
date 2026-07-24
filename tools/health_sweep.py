#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy>=2.1"]
# ///
"""Run lora_health across many checkpoints and print the trajectory.

Requires `lora_health.py` alongside this script (it imports its reconstruction
and flag logic), so run it from within tools/ rather than in isolation.

Reuses lora_health's own reconstruction (collect) and flag logic (_flags),
so the per-checkpoint numbers match `lora_health.py` exactly. For each file it
summarises the effective ‖ΔW‖_F spread and stable-rank spread, plus the
non-finite / dead / suspect counts, then lays them out one row per epoch so you
can see where the adapter saturates and whether stable rank is collapsing.

What to look for down the columns:
  * ‖ΔW‖_F median climbing then flattening  -> adapter saturated; later epochs
    add magnitude but little new signal.
  * stable-rank median falling over epochs   -> updates collapsing onto fewer
    directions: the structural fingerprint of over-cooking.

This is a structural view only — it does NOT measure overfitting/likeness.
Pair it with the out-of-distribution generation test to pick a checkpoint.

Usage (from the repo root):
    uv run tools/health_sweep.py /path/to/checkpoints/
    uv run tools/health_sweep.py 'runs/**/*.safetensors' --grep attn
    uv run tools/health_sweep.py ckpts/ --csv > trajectory.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from statistics import median
from typing import TypedDict


class Summary(TypedDict):
    epoch: int
    label: str
    file: str
    modules: int
    fro_min: float
    fro_med: float
    fro_max: float
    sr_min: float
    sr_med: float
    sr_max: float
    non_finite: int
    dead: int
    suspect: int


# Import the tool that lives next to this script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lora_health import _flags, collect  # noqa: E402

# Epoch/step extractors, tried in order; first match wins. Each has one group.
_EPOCH_PATTERNS = [
    r"epoch[_-]?(\d+)",
    r"[_-]e(\d+)",
    r"[_-]ep(\d+)",
    r"step[_-]?(\d+)",
    r"[_-](\d+)\.safetensors$",
    r"(\d+)",  # last-ditch: any run of digits (rightmost wins below)
]


def epoch_of(path: str, override: str | None) -> tuple[int, str]:
    """Best-effort (epoch_number, raw_label) from a filename."""
    name = os.path.basename(path)
    patterns = [override] if override else _EPOCH_PATTERNS
    for pat in patterns:
        matches = list(re.finditer(pat, name, re.IGNORECASE))
        if matches:
            m = matches[-1]  # rightmost, so trailing counters beat run-name digits
            return int(m.group(1)), m.group(1)
    return -1, name  # unknown -> sorts first, labelled by filename


def expand(inputs: list[str]) -> list[str]:
    """Turn dirs / globs / files into a flat list of .safetensors paths."""
    out: list[str] = []
    for item in inputs:
        if os.path.isdir(item):
            out += glob.glob(os.path.join(item, "**", "*.safetensors"), recursive=True)
        elif any(c in item for c in "*?["):
            out += glob.glob(item, recursive=True)
        else:
            out.append(item)
    return sorted(set(out))


def summarise(path: str, args: argparse.Namespace) -> Summary | None:
    """One checkpoint -> summary row, or None if it has no usable modules."""
    try:
        rows, _ = collect(path, args.grep)
    except Exception as e:  # unreadable/corrupt file: note and skip, don't abort
        print(f"  ! {os.path.basename(path)}: {e}", file=sys.stderr)
        return None
    if not rows:
        return None

    non_finite, dead, suspect = _flags(
        rows, args.dead_threshold, args.outlier_factor, args.collapse_rank
    )
    fros = sorted(r["fro"] for r in rows if r["fro"] > args.dead_threshold)
    srs = sorted(r["stable_rank"] for r in rows if r["stable_rank"] is not None)
    epoch, label = epoch_of(path, args.epoch_regex)
    return {
        "epoch": epoch,
        "label": label,
        "file": os.path.basename(path),
        "modules": len(rows),
        "fro_min": fros[0] if fros else 0.0,
        "fro_med": median(fros) if fros else 0.0,
        "fro_max": fros[-1] if fros else 0.0,
        "sr_min": srs[0] if srs else 0.0,
        "sr_med": median(srs) if srs else 0.0,
        "sr_max": srs[-1] if srs else 0.0,
        "non_finite": len(non_finite),
        "dead": len(dead),
        "suspect": len(suspect),
    }


def _tag(s: Summary) -> str:
    return str(s["epoch"]) if s["epoch"] >= 0 else s["label"]


def print_table(summ: list[Summary], show_files: bool) -> None:
    fcol = f"  {'file':<32}" if show_files else ""
    hdr = (
        f"{'epoch':>6}  {'mods':>5}  {'‖ΔW‖_F (min/med/max)':>26}  "
        f"{'stable rank (min/med/max)':>27}  {'nan':>3} {'dead':>4} {'susp':>4}{fcol}"
    )
    print(hdr)
    print("-" * len(hdr))
    for s in summ:
        fro = f"{s['fro_min']:>7.3g}/{s['fro_med']:>7.3g}/{s['fro_max']:>7.3g}"
        sr = f"{s['sr_min']:>7.3g}/{s['sr_med']:>7.3g}/{s['sr_max']:>7.3g}"
        fcell = f"  {s['file'][-32:]:<32}" if show_files else ""
        print(
            f"{_tag(s):>6}  {s['modules']:>5}  {fro:>26}  {sr:>27}  "
            f"{s['non_finite']:>3} {s['dead']:>4} {s['suspect']:>4}{fcell}"
        )


def print_read(summ: list[Summary]) -> None:
    """A short, mechanical trajectory read — signals, not a verdict."""
    if len(summ) < 2:
        return
    peak = max(summ, key=lambda s: s["fro_med"])
    first, last = summ[0], summ[-1]
    print("\nread (structural signals only — confirm with generations):")
    print(f"  ‖ΔW‖_F median peaks at {_tag(peak)} ({peak['fro_med']:.3g})")
    if peak is not last:
        print("    -> magnitude already past its peak by the last checkpoint")
    sr0, sr1 = first["sr_med"], last["sr_med"]
    if sr0 > 0:
        drop = (sr0 - sr1) / sr0 * 100
        arrow = "falling" if drop > 5 else ("rising" if drop < -5 else "flat")
        print(
            f"  stable-rank median {arrow}: {sr0:.3g} -> {sr1:.3g} "
            f"({drop:+.0f}% across the sweep)"
        )
        if drop > 15:
            print(
                "    -> updates collapsing onto fewer directions; favour an "
                "earlier checkpoint"
            )
    if any(s["suspect"] for s in summ):
        first_susp = next(s for s in summ if s["suspect"])
        print(
            f"  first 'suspect' (strong+collapsed) layer appears at {_tag(first_susp)}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("inputs", nargs="+", help="checkpoint files, dirs, or globs")
    ap.add_argument("--grep", metavar="SUBSTR", help="only modules whose keys match")
    ap.add_argument(
        "--epoch-regex", metavar="RE", help="override epoch extractor (1 group)"
    )
    ap.add_argument("--dead-threshold", type=float, default=1e-6)
    ap.add_argument("--outlier-factor", type=float, default=4.0)
    ap.add_argument("--collapse-rank", type=float, default=2.0)
    ap.add_argument("--csv", action="store_true", help="emit CSV instead of a table")
    ap.add_argument(
        "--files", action="store_true", help="always show a filename column"
    )
    args = ap.parse_args()

    paths = expand(args.inputs)
    if not paths:
        sys.exit("error: no .safetensors files matched")

    summ = [s for s in (summarise(p, args) for p in paths) if s is not None]
    if not summ:
        sys.exit("error: no checkpoints yielded usable modules")

    # When epoch parsing fails, every file collapses to the same tag; fall back
    # to file-order (the sort is already lexicographic) and surface filenames so
    # the ordering is verifiable rather than silently wrong.
    degenerate = len({s["epoch"] for s in summ}) == 1 and len(summ) > 1
    summ.sort(key=lambda s: (s["epoch"], s["file"]))

    if args.csv:
        w = csv.DictWriter(sys.stdout, fieldnames=list(summ[0].keys()))
        w.writeheader()
        w.writerows(summ)
        return

    if degenerate:
        print(
            "warning: could not parse distinct epochs from filenames — rows are in "
            "filename order. Pass --epoch-regex '<pattern with one (group)>' to label them.",
            file=sys.stderr,
        )
    print(f"swept {len(summ)} checkpoint(s)\n")
    print_table(summ, show_files=args.files or degenerate)
    print_read(summ)


if __name__ == "__main__":
    main()
