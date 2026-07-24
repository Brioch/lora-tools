#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy>=2.1"]
# ///
"""Report weight-health statistics for a .safetensors LoRA.

Where inspect_lora.py reads only the header, this decodes the tensor *data*,
reconstructs each module's effective delta-weight (ΔW) — folding the low-rank
factors and the alpha/rank scale back together — and reports the numbers that
actually track training quality:

  * ‖ΔW‖_F     — effective magnitude of a module's change to the base weights
  * σ_max      — its largest singular value (peak per-direction strength)
  * stable rank — ‖ΔW‖_F² / σ_max² (Rudelson & Vershynin, 2007); ~1 means the
                  update has collapsed onto a single direction, high means it
                  stays spread across many directions

It flags three failure modes a metadata dump can't see:

  * non-finite — any NaN/Inf value (a diverged / "fried" run)
  * dead       — effective ‖ΔW‖_F at or below --dead-threshold (never trained)
  * suspect    — large ‖ΔW‖_F *and* low stable rank (--collapse-rank): the update
                 is both strong and collapsed — a candidate over-cooked layer

Reconstruction is exact for classic LoRA (down/up, A/B), LoKr, and LoHa; the
alpha/rank scale is applied when both are recoverable (shown per module).

NOTE on the "suspect" flag: the stable-rank metric is standard linear algebra,
but reading "strong + collapsed" as *over-cooking* is a heuristic, not a
validated result — no paper establishes it for LoRAs, and it has false positives
(a legitimately focused, low-rank adaptation) and false negatives (a LoRA fried
diffusely across many directions). Spectral shape is only *indirectly* linked to
training quality in the literature (e.g. Martin & Mahoney's heavy-tailed
self-regularisation). Weight statistics can *suggest* over-cooking but can't
prove it — the ground truth is behavioural (a strength sweep, prompt-adherence,
memorisation checks). Treat "suspect" as "look here," not a verdict.

Usage:
    python lora_health.py path/to/lora.safetensors
    python lora_health.py path/to/lora.safetensors --all     # every module
    python lora_health.py path/to/lora.safetensors --json     # machine-readable
    python lora_health.py path/to/lora.safetensors --grep attn

Exit status is 1 when any module is non-finite, so it doubles as a CI gate.
F32/F16/BF16/F64 are decoded; float8 and other exotic dtypes are skipped.
"""

import argparse
import json
import re
import struct
import sys
from collections import Counter
from typing import Any

import numpy as np

# safetensors dtype -> little-endian numpy dtype. BF16 has no numpy type and is
# widened to float32 by hand; integer / float8 dtypes are decoded only enough to
# be counted as skipped (a norm over quantised or index tensors is meaningless).
_NP_DTYPE: dict[str, str] = {"F64": "<f8", "F32": "<f4", "F16": "<f2"}

# Math-format suffixes, longest-first so "lokr_w1_a" wins over "lokr_w1".
_SUFFIXES = (
    "lora_down",
    "lora_up",
    "lora_A",
    "lora_B",
    "lokr_w1_a",
    "lokr_w1_b",
    "lokr_w2_a",
    "lokr_w2_b",
    "lokr_w1",
    "lokr_w2",
    "hada_w1_a",
    "hada_w1_b",
    "hada_w2_a",
    "hada_w2_b",
    "alpha",
)
_SUFFIX_RE = re.compile(r"\.(" + "|".join(_SUFFIXES) + r")(?:\.weight)?$")


def read_header(path: str) -> tuple[dict[str, Any], dict[str, Any], int]:
    """Return (tensor header, __metadata__, data_start) for a safetensors file."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    meta = header.pop("__metadata__", {})
    return header, meta, 8 + n


def to_array(raw: bytes, dtype: str, shape: list[int]) -> Any:
    """Decode a tensor's bytes into a float64 ndarray, or None if unsupported."""
    if dtype == "BF16":
        # bfloat16 is the high 16 bits of a float32; widen by left-shifting.
        u32 = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16
        flat = u32.view(np.float32)
    else:
        np_dtype = _NP_DTYPE.get(dtype)
        if np_dtype is None:
            return None
        flat = np.frombuffer(raw, dtype=np_dtype)
    return flat.astype(np.float64).reshape(shape) if shape else flat.astype(np.float64)


def as_2d(a: Any) -> Any:
    """Fold a tensor to 2-D (rows = first axis) for matrix maths; conv -> matrix."""
    if a.ndim <= 1:
        return a.reshape(1, -1)
    return a.reshape(a.shape[0], -1)


def product_svdvals(down: Any, up: Any) -> Any:
    """Singular values of ``up @ down`` without forming the full out×in product.

    down is (r, in·…), up is (out, r). Both factors are low-rank (r ≪ out, in),
    so a QR reduction gives the same singular values from an r×r core.
    """
    d = as_2d(down)  # (r, in)
    u = as_2d(up)  # (out, r)
    _, r_u = np.linalg.qr(u)  # u = Q_u R_u,      R_u is (r, r)
    _, r_d = np.linalg.qr(d.T)  # dᵀ = Q_d R_d,   R_d is (r, r)
    core = r_u @ r_d.T  # same singular values as up @ down
    return np.linalg.svd(core, compute_uv=False)


def _scalar(a: Any) -> float:
    return float(np.asarray(a).reshape(-1)[0])


def _lokr_factor(w: dict[str, Any], base: str) -> Any:
    """Rebuild a LoKr/LoHa factor: the tensor itself, or its a·b decomposition."""
    if base in w:
        return as_2d(w[base])
    a, b = w.get(f"{base}_a"), w.get(f"{base}_b")
    if a is not None and b is not None:
        return as_2d(a) @ as_2d(b)
    return None


def _spectrum(w: dict[str, Any]) -> tuple[str, Any, int | None] | None:
    """Map a module's weight tensors to (format, ΔW singular values, rank).

    Returns None when the tensors don't form a reconstructable module.
    """
    keys = set(w)
    if {"lora_down", "lora_up"} <= keys:
        return (
            "lora",
            product_svdvals(w["lora_down"], w["lora_up"]),
            w["lora_down"].shape[0],
        )
    if {"lora_A", "lora_B"} <= keys:
        return "lora", product_svdvals(w["lora_A"], w["lora_B"]), w["lora_A"].shape[0]
    if any(k.startswith("lokr_") for k in keys):
        w1, w2 = _lokr_factor(w, "lokr_w1"), _lokr_factor(w, "lokr_w2")
        if w1 is None or w2 is None:
            return None
        # σ(w1 ⊗ w2) = outer(σ(w1), σ(w2)); no need to build the Kronecker product.
        s = np.outer(
            np.linalg.svd(w1, compute_uv=False), np.linalg.svd(w2, compute_uv=False)
        )
        rank = w["lokr_w1_b"].shape[0] if "lokr_w1_b" in w else None
        return "lokr", s.ravel(), rank
    if any(k.startswith("hada_") for k in keys):
        m1, m2 = _lokr_factor(w, "hada_w1"), _lokr_factor(w, "hada_w2")
        if m1 is None or m2 is None:
            return None
        s = np.linalg.svd(m1 * m2, compute_uv=False)  # LoHa is a Hadamard product
        rank = w["hada_w1_b"].shape[0] if "hada_w1_b" in w else None
        return "loha", s, rank
    return None


def analyse_module(name: str, w: dict[str, Any], alpha: float | None) -> dict[str, Any]:
    """Per-module health: effective ΔW magnitude, stable rank, and flags."""
    non_finite = any(not np.all(np.isfinite(t)) for t in w.values())
    params = int(sum(t.size for t in w.values()))
    spec = None if non_finite else _spectrum(w)

    row: dict[str, Any] = {
        "name": name,
        "params": params,
        "non_finite": non_finite,
        "alpha": alpha,
    }
    if spec is None:
        # Unreconstructable (or non-finite): fall back to a plain Frobenius norm.
        energy = sum(float(np.sum(np.square(t[np.isfinite(t)]))) for t in w.values())
        row.update(
            format="raw",
            rank=None,
            scale=1.0,
            fro=float(np.sqrt(energy)),
            sigma_max=None,
            stable_rank=None,
        )
        return row

    fmt, s, rank = spec
    s = np.asarray(s, dtype=np.float64)
    energy = float(np.sum(np.square(s)))
    sigma_max = float(s.max()) if s.size else 0.0
    scale = alpha / rank if (alpha is not None and rank) else 1.0
    row.update(
        format=fmt,
        rank=rank,
        scale=scale,
        fro=scale * float(np.sqrt(energy)),
        sigma_max=scale * sigma_max,
        # Stable rank is scale-invariant, so alpha uncertainty never distorts it.
        stable_rank=(energy / sigma_max**2) if sigma_max > 0 else 0.0,
    )
    return row


def collect(
    path: str, grep: str | None = None
) -> tuple[list[dict[str, Any]], list[str]]:
    """Group tensors into modules and analyse each. Returns (rows, skipped-dtypes)."""
    header, _, base = read_header(path)
    modules: dict[str, dict[str, Any]] = {}
    alphas: dict[str, float] = {}
    skipped: list[str] = []

    with open(path, "rb") as f:
        for key in sorted(header):
            if grep and grep not in key:
                continue
            m = _SUFFIX_RE.search(key)
            mod, suffix = (key[: m.start()], m.group(1)) if m else (key, "weight")
            info = header[key]
            dtype = info.get("dtype", "?")
            begin, end = info.get("data_offsets", (0, 0))
            f.seek(base + begin)
            arr = to_array(f.read(end - begin), dtype, info.get("shape", []))
            if arr is None:
                skipped.append(dtype)
                continue
            if suffix == "alpha":
                alphas[mod] = _scalar(arr)
            else:
                modules.setdefault(mod, {})[suffix] = arr

    rows = [analyse_module(name, w, alphas.get(name)) for name, w in modules.items()]
    rows.sort(key=lambda r: r["fro"], reverse=True)
    return rows, skipped


def _flags(
    rows: list[dict[str, Any]], dead: float, factor: float, collapse: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    non_finite = [r for r in rows if r["non_finite"]]
    live = [r for r in rows if not r["non_finite"]]
    dead_mods = [r for r in live if r["fro"] <= dead]
    fros = sorted(r["fro"] for r in live if r["fro"] > dead)
    median = fros[len(fros) // 2] if fros else 0.0
    # "Suspect": strong (norm well above the median) AND collapsed (low stable rank).
    suspect = [
        r
        for r in live
        if r["fro"] > max(median * factor, dead)
        and r["stable_rank"] is not None
        and r["stable_rank"] <= collapse
    ]
    return non_finite, dead_mods, suspect


def _fmt_names(rs: list[dict[str, Any]], limit: int = 6) -> str:
    if not rs:
        return "none"
    shown = ", ".join(r["name"] for r in rs[:limit])
    return shown + (f", … (+{len(rs) - limit} more)" if len(rs) > limit else "")


def _print_report(
    path: str, rows: list[dict[str, Any]], skipped: list[str], args: argparse.Namespace
) -> list[dict[str, Any]]:
    non_finite, dead, suspect = _flags(
        rows, args.dead_threshold, args.outlier_factor, args.collapse_rank
    )
    params = sum(r["params"] for r in rows)
    fmts = Counter(r["format"] for r in rows)

    print(f"file            : {path}")
    print(f"modules         : {len(rows)}   (params {params:,})")
    print(f"formats         : {dict(fmts)}")
    if skipped:
        print(f"skipped (dtype) : {len(skipped)}: {', '.join(sorted(set(skipped)))}")

    print("\nhealth")
    print(f"  non-finite : {_fmt_names(non_finite)}")
    print(f"  dead (‖ΔW‖≤{args.dead_threshold:g}) : {_fmt_names(dead)}")
    print(f"  suspect (strong+collapsed) : {_fmt_names(suspect)}")

    spectral = [r for r in rows if r["stable_rank"] is not None]
    if spectral:
        sr = sorted(r["stable_rank"] for r in spectral)
        fro = sorted(r["fro"] for r in spectral)
        print(
            f"  ‖ΔW‖_F     : min {fro[0]:.3g}  median {fro[len(fro) // 2]:.3g}  "
            f"max {fro[-1]:.3g}"
        )
        print(
            f"  stable rank: min {sr[0]:.3g}  median {sr[len(sr) // 2]:.3g}  "
            f"max {sr[-1]:.3g}"
        )

    shown = rows if args.all else rows[:12]
    if shown:
        head = (
            "" if args.all else f" (top {len(shown)} by ‖ΔW‖_F; --all for every module)"
        )
        print(f"\nmodules{head}:")
        width = min(max((len(r["name"]) for r in shown), default=6), 70)
        print(
            f"  {'‖ΔW‖_F':>9}  {'σ_max':>8}  {'s.rank':>7}  {'rank':>4}  "
            f"{'fmt':>4}  module"
        )
        for r in shown:
            srk = (
                f"{r['stable_rank']:7.2f}" if r["stable_rank"] is not None else " " * 7
            )
            sm = f"{r['sigma_max']:8.3g}" if r["sigma_max"] is not None else " " * 8
            rk = f"{r['rank']:4d}" if r["rank"] is not None else " " * 4
            flag = " !" if r["non_finite"] else ""
            print(
                f"  {r['fro']:9.3g}  {sm}  {srk}  {rk}  {r['format']:>4}  "
                f"{r['name'][:width]}{flag}"
            )
    return non_finite


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("path", help="path to the .safetensors file")
    ap.add_argument(
        "--all", action="store_true", help="print every module, not the top 12"
    )
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument(
        "--grep", metavar="SUBSTR", help="only modules whose keys contain SUBSTR"
    )
    ap.add_argument(
        "--dead-threshold",
        type=float,
        default=1e-6,
        help="effective ‖ΔW‖_F at or below which a module is dead (default: 1e-6)",
    )
    ap.add_argument(
        "--outlier-factor",
        type=float,
        default=4.0,
        help="a suspect's ‖ΔW‖_F must exceed this × the median (default: 4.0)",
    )
    ap.add_argument(
        "--collapse-rank",
        type=float,
        default=2.0,
        help="a suspect's stable rank must be at or below this (default: 2.0)",
    )
    args = ap.parse_args()

    try:
        rows, skipped = collect(args.path, args.grep)
    except Exception as e:
        sys.exit(f"error reading {args.path}: {e}")

    if not rows:
        sys.exit(f"error: no reconstructable modules found in {args.path}")

    if args.json:
        print(json.dumps({"file": args.path, "modules": rows}, indent=2))
        non_finite = [r for r in rows if r["non_finite"]]
    else:
        non_finite = _print_report(args.path, rows, skipped, args)

    if non_finite:
        sys.exit(1)  # a fried checkpoint is a hard failure — usable as a CI gate


if __name__ == "__main__":
    main()
