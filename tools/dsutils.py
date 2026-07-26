# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""Shared helpers for the dataset tools.

Imported by `lint_dataset.py`, `edit_captions.py`, `prepare_images.py`,
`dedupe_images.py` and `balance_regularization.py`. This is a module rather than a
CLI, so it has no `main()`; the PEP 723 header above is there so the tools that
import it stay runnable with `uv run tools/<script>.py`.

Nothing here imports Pillow at module level — `dhash()` pulls it in on first call —
so the caption-only tools stay pure stdlib.

Convention: an image `foo.png` is paired with a caption `foo.txt` (same stem) in the
same directory. Captions are optional unless a tool requires them.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

DEFAULT_IMAGE_EXTS: list[str] = [".png", ".jpg", ".jpeg", ".webp", ".bmp"]
DEFAULT_CAPTION_EXT: str = ".txt"


class HelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter, argparse.RawDescriptionHelpFormatter
):
    """Print the module docstring verbatim, then append each flag's default."""


def normalize_ext(ext: str) -> str:
    """Return *ext* guaranteed to start with a dot (lower-casing not applied)."""
    return ext if ext.startswith(".") else f".{ext}"


def normalize_exts(exts: list[str]) -> set[str]:
    """Return a lower-cased set of dotted extensions."""
    return {normalize_ext(e).lower() for e in exts}


def list_images(directory: Path, exts: set[str]) -> list[Path]:
    """Return image files in *directory* (non-recursive), sorted by name."""
    directory = Path(directory)
    return sorted(
        p for p in directory.iterdir() if p.is_file() and p.suffix.lower() in exts
    )


def paired_caption(image: Path, caption_ext: str) -> Path | None:
    """Return the caption path paired with *image* if it exists, else None."""
    caption = Path(image).with_suffix(caption_ext)
    return caption if caption.exists() else None


def dhash(image: PILImage, size: int = 8) -> int:
    """Difference hash: compare each pixel to its right neighbour. size*size bits.

    Deliberately mirrors `compare_datasets.dhash` (that script stays standalone and
    self-contained), so distances reported by the two tools are comparable. Needs
    Pillow, imported lazily so importing this module never requires it.
    """
    from PIL import Image

    small = image.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS)
    px = small.tobytes()  # one byte per pixel in "L" mode
    bits = 0
    for row in range(size):
        base = row * (size + 1)
        for col in range(size):
            bits = (bits << 1) | (1 if px[base + col] > px[base + col + 1] else 0)
    return bits


def hamming(a: int, b: int) -> int:
    """Number of differing bits between two integer fingerprints."""
    return bin(a ^ b).count("1")
