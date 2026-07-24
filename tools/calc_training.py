#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Calculate LoKr/LoRA training settings for a multi-resolution schedule from a
dataset image count and a total effective-step budget.

The default recipe is a Krea 2 character LoKr: a 512 bulk pass at batch 2 and a
1024 refinement pass at batch 1, with 3/4 of the budget on 512 and 1/4 on 1024.

"Effective steps" are batch-1-equivalent steps: a pass of S steps at batch B
counts as S x B effective steps. The budget you pass (e.g. 1250) is in these
units, so passes with different batch sizes sit on one comparable scale. Because
steps/epoch ~= images/batch, a pass's effective steps work out to ~= images x
epochs regardless of batch — so batch size changes the reported step count, not
the budget accounting.

What that count is (and isn't): one step at batch B shows the model B images and
does ONE gradient update. So `steps x batch` = images seen = epochs x dataset
size — a sound measure of data exposure, which is what this tool budgets against.
Gradient UPDATES, by contrast, are `steps` alone: a bigger batch gives fewer but
lower-variance updates for the same exposure. So treat batch-as-multiplier as an
image count, not a convergence law — equating batch-2-for-450 with batch-1-for-900
in training progress is a heuristic that leans on learning-rate scaling.

Examples:
    # Interactive mode — prompts for each input (also the default with no args).
    python calc_training.py -i

    # Default recipe (512 @ batch 2, 1024 @ batch 1), 3/4 - 1/4 split.
    python calc_training.py 31 1250

    # Rebalance the budget between passes (weights are normalized by their sum,
    # so 60/40, 3/1 and 75/25 all work).
    python calc_training.py 31 1250 --split 60/40

    # A custom three-pass schedule.
    python calc_training.py 20 800 --pass 512:2 --pass 768:2 --pass 1024:1 --split 2/1/1

    # Gradient accumulation: run 1024 at batch 1 but accum 2 (effective batch 2,
    # matching the 512 pass) when VRAM caps the real batch at 1. Overriding --pass
    # resets the split to equal, so restate --split to keep the 3/4 - 1/4 budget.
    python calc_training.py 31 1250 --pass 512:2 --pass 1024:1:2 --split 3/1

Steps/epoch is floor(images / (batch*accum)) by default, matching a trainer that
drops the trailing partial batch (drop_last); pass --keep-last for one that pads it.

Gradient accumulation (RES:BATCH:ACCUM): a step runs ACCUM micro-batches, sums
their gradients, then does ONE optimizer update — so the effective batch is
batch*accum, at batch-1 VRAM cost but ~ACCUM x slower. It doesn't change images
seen (the budget) or epochs; it makes fewer, lower-variance optimizer steps (the
reported "steps" count drops) and behaves like a larger batch for LR-scaling — so
scale the learning rate by batch*accum, not batch alone.

Pure stdlib — no dependencies.
"""

import argparse
import math
import sys


def parse_pass(spec: str) -> tuple[int, int, int]:
    """Parse a RES:BATCH[:ACCUM] pass spec into (res, batch, accum) positive ints.

    ACCUM is gradient-accumulation steps (default 1); the effective batch a step
    trains on is batch * accum.
    """
    parts = spec.split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"expected RES:BATCH[:ACCUM], got {spec!r}")
    try:
        nums = [int(p) for p in parts]
    except ValueError:
        raise ValueError(f"RES, BATCH and ACCUM must be integers, got {spec!r}")
    if any(n <= 0 for n in nums):
        raise ValueError(f"RES, BATCH and ACCUM must be positive, got {spec!r}")
    res, batch = nums[0], nums[1]
    accum = nums[2] if len(nums) == 3 else 1
    return res, batch, accum


def parse_split(spec: str, n: int) -> list[float]:
    """Parse a `/`-separated weight list into n fractions summing to 1.0."""
    parts = spec.split("/")
    if len(parts) != n:
        raise ValueError(f"expected {n} weight(s) for {n} pass(es), got {len(parts)}")
    try:
        weights = [float(p) for p in parts]
    except ValueError:
        raise ValueError(f"weights must be numbers, got {spec!r}")
    if any(w <= 0 for w in weights):
        raise ValueError(f"weights must be positive, got {spec!r}")
    total = sum(weights)
    return [w / total for w in weights]


def compute(
    images: int,
    total_steps: int,
    passes: list[tuple[int, int, int]],
    fractions: list[float],
    drop_last: bool = True,
) -> list[dict[str, float]]:
    """Return a list of per-pass dicts with the derived schedule.

    epochs is derived from the image count (frac * T / images) — how epochs are
    normally reasoned about — so it doesn't depend on the batching mode. Only
    steps/epoch depends on drop_last: floor(images/batch) when a trainer drops the
    trailing partial batch (the common default), else ceil (pads/keeps it).
    """
    round_batch = math.floor if drop_last else math.ceil
    rows: list[dict[str, float]] = []
    for (res, batch, accum), frac in zip(passes, fractions):
        eff_batch = batch * accum  # one optimizer step trains on this many images
        steps_per_epoch = max(1, round_batch(images / eff_batch))
        epochs = max(1, round(frac * total_steps / images))
        steps = steps_per_epoch * epochs
        rows.append(
            {
                "res": res,
                "batch": batch,
                "accum": accum,
                "frac": frac,
                "steps_per_epoch": steps_per_epoch,
                "epochs": epochs,
                "steps": steps,
                "effective": steps * eff_batch,
            }
        )
    return rows


def format_report(
    images: int,
    total_steps: int,
    split_spec: str,
    rows: list[dict[str, float]],
    drop_last: bool = True,
) -> str:
    """Render the schedule as a human-readable table."""
    shares = " / ".join(f"{r['frac'] * 100:.0f}%" for r in rows)
    mode = "drop-last" if drop_last else "keep-last"
    lines = [
        f"Dataset: {images} images | target: {total_steps} effective steps "
        f"| split {split_spec} ({shares}) | {mode}",
        "",
    ]

    # Only show the accum column when some pass actually accumulates.
    show_accum = any(r["accum"] > 1 for r in rows)
    header = (
        ["res", "batch"]
        + (["accum"] if show_accum else [])
        + ["steps/epoch", "epochs", "steps", "effective"]
    )
    keys = (
        ["res", "batch"]
        + (["accum"] if show_accum else [])
        + ["steps_per_epoch", "epochs", "steps", "effective"]
    )
    table = [header] + [[str(r[k]) for k in keys] for r in rows]
    widths = [max(len(row[i]) for row in table) for i in range(len(header))]
    for row in table:
        lines.append(
            "  " + "  ".join(cell.rjust(widths[i]) for i, cell in enumerate(row))
        )

    total_actual = sum(r["steps"] for r in rows)
    total_eff = sum(r["effective"] for r in rows)
    delta = (total_eff - total_steps) / total_steps * 100
    lines.append("")
    lines.append(
        f"Total: {total_actual} steps, {total_eff} effective steps "
        f"(target {total_steps}, {delta:+.1f}%)"
    )
    return "\n".join(lines)


def _ask(label: str, default: str | int | None = None) -> str:
    """Prompt once, returning the entered string or the default on empty input."""
    suffix = f" [{default}]" if default is not None else ""
    raw = input(f"{label}{suffix}: ").strip()
    return raw if raw else ("" if default is None else str(default))


def _ask_int(label: str, default: int | None = None) -> int:
    """Prompt for a positive int, re-asking until valid."""
    while True:
        raw = _ask(label, default)
        if not raw:
            print("  a value is required")
            continue
        try:
            value = int(raw)
        except ValueError:
            print(f"  not an integer: {raw!r}")
            continue
        if value <= 0:
            print("  must be a positive integer")
            continue
        return value


def _ask_passes(default: str) -> list[tuple[int, int, int]]:
    """Prompt for a space/comma-separated list of RES:BATCH[:ACCUM] specs."""
    while True:
        specs = (
            _ask("Passes (RES:BATCH[:ACCUM], space-separated)", default)
            .replace(",", " ")
            .split()
        )
        if not specs:
            print("  at least one pass is required")
            continue
        try:
            passes = [parse_pass(s) for s in specs]
        except ValueError as e:
            print(f"  {e}")
            continue
        return passes


def _ask_split(default: str, n: int) -> str:
    """Prompt for a `/`-separated weight list valid for n passes."""
    while True:
        spec = _ask("Budget split (weights per pass)", default)
        try:
            parse_split(spec, n)
        except ValueError as e:
            print(f"  {e}")
            continue
        return spec


def _ask_bool(label: str, default: bool = True) -> bool:
    """Prompt for yes/no, returning a bool."""
    hint = "Y/n" if default else "y/N"
    while True:
        raw = input(f"{label} [{hint}]: ").strip().lower()
        if not raw:
            return default
        if raw in ("y", "yes"):
            return True
        if raw in ("n", "no"):
            return False
        print("  please answer y or n")


def run_interactive(
    default_passes: str = "512:2 1024:1", default_split: str | None = None
) -> None:
    """Prompt for inputs, then print the schedule."""
    print("Training-settings calculator — interactive mode.")
    print("Press Enter to accept the [default] shown; Ctrl-C to quit.\n")
    images = _ask_int("Number of images in the dataset")
    total_steps = _ask_int("Total effective-step budget (batch-1-equivalent)", 1250)
    passes = _ask_passes(default_passes)
    if default_split is None:
        default_split = "3/1" if len(passes) == 2 else "/".join(["1"] * len(passes))
    split_spec = _ask_split(default_split, len(passes))
    drop_last = _ask_bool("Drop the trailing partial batch (drop_last)?", True)

    fractions = parse_split(split_spec, len(passes))
    rows = compute(images, total_steps, passes, fractions, drop_last)
    print()
    print(format_report(images, total_steps, split_spec, rows, drop_last))


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "images", type=int, nargs="?", help="number of images in the dataset"
    )
    ap.add_argument(
        "total_steps",
        type=int,
        nargs="?",
        help="total effective-step budget (batch-1-equivalent, e.g. 1250)",
    )
    ap.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="prompt for inputs interactively (also the default when run "
        "with no arguments)",
    )
    ap.add_argument(
        "--pass",
        dest="passes",
        action="append",
        metavar="RES:BATCH[:ACCUM]",
        help="a training pass as RES:BATCH[:ACCUM], where ACCUM is "
        "gradient-accumulation steps (default 1; effective batch = "
        "batch*accum). Repeatable; default: --pass 512:2 --pass 1024:1",
    )
    ap.add_argument(
        "--split",
        metavar="WEIGHTS",
        help="`/`-separated budget weights, one per pass, normalized by "
        "their sum (default: 3/1)",
    )
    ap.add_argument(
        "--keep-last",
        action="store_true",
        help="count the trailing partial batch (ceil steps/epoch), for "
        "trainers that pad it; default drops it (floor)",
    )
    args = ap.parse_args()

    if args.interactive or (args.images is None and args.total_steps is None):
        default_passes = " ".join(args.passes) if args.passes else "512:2 1024:1"
        try:
            run_interactive(default_passes, args.split)
        except (EOFError, KeyboardInterrupt):
            sys.exit("\naborted.")
        return

    if args.images is None or args.total_steps is None:
        ap.error("images and total_steps are required (or use -i for interactive mode)")
    if args.images <= 0:
        ap.error("images must be a positive integer")
    if args.total_steps <= 0:
        ap.error("total_steps must be a positive integer")

    pass_specs = args.passes if args.passes else ["512:2", "1024:1"]
    try:
        passes = [parse_pass(s) for s in pass_specs]
    except ValueError as e:
        ap.error(str(e))

    split_spec = (
        args.split
        if args.split
        else "/".join(["3", "1"] if not args.passes else ["1"] * len(passes))
    )
    try:
        fractions = parse_split(split_spec, len(passes))
    except ValueError as e:
        ap.error(str(e))

    drop_last = not args.keep_last
    rows = compute(args.images, args.total_steps, passes, fractions, drop_last)
    print(format_report(args.images, args.total_steps, split_spec, rows, drop_last))


if __name__ == "__main__":
    main()
