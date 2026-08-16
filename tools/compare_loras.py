#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy>=2.1"]
# ///
"""Compare two LoRA checkpoints module by module: same direction, or just louder?

`lora_health.py` reads one file and `health_sweep.py` tracks magnitude across many,
but neither answers the question you actually have when comparing two runs: did this
adapter learn something *different*, or the same thing at a different strength? A
magnitude number alone cannot tell those apart — halving alpha and retraining from
scratch can both change ‖ΔW‖_F by the same amount.

For each module present in both files it reports:

  * cos    — cosine similarity between the two effective ΔW matrices. 1.00 means the
             update points the same way and only its scale differs; toward 0 means the
             two adapters learned genuinely different directions; negative means they
             pull against each other.
  * ratio  — ‖ΔW_b‖_F / ‖ΔW_a‖_F, i.e. how much louder b is than a.
  * rel Δ  — ‖ΔW_b − ΔW_a‖_F / ‖ΔW_a‖_F, the total relative change, folding direction
             and magnitude into one number.

Reading it: high cos with ratio away from 1 is a pure strength change (the same
adaptation, rescaled — often what an alpha edit does). Low cos is a different
adaptation. cos ~1 with ratio ~1 across the board means the two checkpoints have
converged and the later one is not learning anything new.

Nothing is materialized at full out×in size. For LoRA the Frobenius inner product
reduces to a trace over two small r×r cores; for LoKr it factorizes across the
Kronecker product. So this is as cheap as lora_health.py on the same file.

Usage:
    # Two epochs of the same run — has it converged?
    uv run tools/compare_loras.py epoch30.safetensors epoch40.safetensors

    # Two settings — did lowering alpha change what was learned, or only how loud?
    uv run tools/compare_loras.py alpha16.safetensors alpha8.safetensors --all

    # Only the attention modules, machine-readable.
    uv run tools/compare_loras.py a.safetensors b.safetensors --grep attn --json

Needs numpy. Exits 1 when the two files share no comparable module.
"""

import argparse
import json
import os
import sys
from typing import Any

import numpy as np

# Import the tool that lives next to this script.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lora_health import (  # noqa: E402
    _SUFFIX_RE,
    _lokr_factor,
    _scalar,
    as_2d,
    read_header,
    to_array,
)


def load_modules(
    path: str, grep: str | None = None
) -> tuple[dict[str, dict[str, Any]], dict[str, float]]:
    """Group a file's tensors into {module: {suffix: array}} plus {module: alpha}.

    Mirrors lora_health.collect's grouping so module names line up between the tools.
    """
    header, _, base = read_header(path)
    modules: dict[str, dict[str, Any]] = {}
    alphas: dict[str, float] = {}

    with open(path, "rb") as f:
        for key in sorted(header):
            if grep and grep not in key:
                continue
            m = _SUFFIX_RE.search(key)
            mod, suffix = (key[: m.start()], m.group(1)) if m else (key, "weight")
            info = header[key]
            begin, end = info.get("data_offsets", (0, 0))
            f.seek(base + begin)
            arr = to_array(
                f.read(end - begin), info.get("dtype", "?"), info.get("shape", [])
            )
            if arr is None:
                continue  # float8 / integer dtypes: a norm over them is meaningless
            if suffix == "alpha":
                alphas[mod] = _scalar(arr)
            else:
                modules.setdefault(mod, {})[suffix] = arr
    return modules, alphas


def _kind(w: dict[str, Any]) -> str | None:
    """Classify a module's math format, or None if it isn't reconstructable."""
    keys = set(w)
    if {"lora_down", "lora_up"} <= keys or {"lora_A", "lora_B"} <= keys:
        return "lora"
    if any(k.startswith("lokr_") for k in keys):
        return "lokr"
    if any(k.startswith("hada_") for k in keys):
        return "loha"
    return None


def _lora_factors(w: dict[str, Any]) -> tuple[Any, Any]:
    """Return (down, up) as 2-D arrays, whichever naming convention was used."""
    if "lora_down" in w:
        return as_2d(w["lora_down"]), as_2d(w["lora_up"])
    return as_2d(w["lora_A"]), as_2d(w["lora_B"])


def _rank(w: dict[str, Any], kind: str) -> int | None:
    """The module's rank, for the alpha/rank scale — same rule as lora_health."""
    if kind == "lora":
        return int(_lora_factors(w)[0].shape[0])
    key = "lokr_w1_b" if kind == "lokr" else "hada_w1_b"
    return int(w[key].shape[0]) if key in w else None


def inner_product(a: dict[str, Any], b: dict[str, Any], kind: str) -> float:
    """Frobenius inner product <ΔW_a, ΔW_b> without building either ΔW.

    LoRA: ΔW = U @ D, so <U₁D₁, U₂D₂> = trace(D₁ᵀ U₁ᵀ U₂ D₂) = trace((U₁ᵀU₂)(D₂D₁ᵀ)),
    and both factors are r×r. Ranks may differ between the two files; the trace is
    still well defined as long as the out/in dimensions match.

    LoKr: <A₁⊗B₁, A₂⊗B₂> = <A₁,A₂>·<B₁,B₂>, so the Kronecker product is never formed.

    LoHa: the Hadamard factors are already full-size matrices, so multiply and sum.
    """
    if kind == "lora":
        d_a, u_a = _lora_factors(a)
        d_b, u_b = _lora_factors(b)
        return float(np.trace((u_a.T @ u_b) @ (d_b @ d_a.T)))
    if kind == "lokr":
        w1_a, w2_a = _lokr_factor(a, "lokr_w1"), _lokr_factor(a, "lokr_w2")
        w1_b, w2_b = _lokr_factor(b, "lokr_w1"), _lokr_factor(b, "lokr_w2")
        return float(np.sum(w1_a * w1_b)) * float(np.sum(w2_a * w2_b))
    m1_a, m2_a = _lokr_factor(a, "hada_w1"), _lokr_factor(a, "hada_w2")
    m1_b, m2_b = _lokr_factor(b, "hada_w1"), _lokr_factor(b, "hada_w2")
    return float(np.sum((m1_a * m2_a) * (m1_b * m2_b)))


def compare_module(
    name: str,
    a: dict[str, Any],
    b: dict[str, Any],
    alpha_a: float | None,
    alpha_b: float | None,
) -> dict[str, Any] | None:
    """Compare one module across two files. None if it isn't comparable."""
    kind = _kind(a)
    if kind is None or kind != _kind(b):
        return None
    if any(not np.all(np.isfinite(t)) for t in (*a.values(), *b.values())):
        return {"name": name, "format": kind, "non_finite": True}

    try:
        raw_aa = inner_product(a, a, kind)
        raw_bb = inner_product(b, b, kind)
        raw_ab = inner_product(a, b, kind)
    except (ValueError, TypeError):
        return None  # shapes don't line up: different base model or architecture

    rank_a, rank_b = _rank(a, kind), _rank(b, kind)
    scale_a = alpha_a / rank_a if (alpha_a is not None and rank_a) else 1.0
    scale_b = alpha_b / rank_b if (alpha_b is not None and rank_b) else 1.0

    # Norms carry the alpha/rank scale, matching lora_health's ‖ΔW‖_F. Cosine is
    # scale-invariant for positive scales, so it comes from the raw products.
    norm_a = scale_a * float(np.sqrt(max(raw_aa, 0.0)))
    norm_b = scale_b * float(np.sqrt(max(raw_bb, 0.0)))
    denom = np.sqrt(max(raw_aa, 0.0)) * np.sqrt(max(raw_bb, 0.0))
    cos = float(raw_ab / denom) if denom > 0 else 0.0
    # ‖b−a‖² = ‖a‖² + ‖b‖² − 2<a,b>, with each term carrying its own scale.
    gap = norm_a**2 + norm_b**2 - 2 * scale_a * scale_b * raw_ab
    return {
        "name": name,
        "format": kind,
        "non_finite": False,
        "cos": cos,
        "fro_a": norm_a,
        "fro_b": norm_b,
        "ratio": (norm_b / norm_a) if norm_a > 0 else float("inf"),
        "rel_delta": (float(np.sqrt(max(gap, 0.0))) / norm_a) if norm_a > 0 else 0.0,
        "rank_a": rank_a,
        "rank_b": rank_b,
    }


def compare(
    path_a: str, path_b: str, grep: str | None = None
) -> tuple[list[dict[str, Any]], list[str], list[str], list[str]]:
    """Compare two files. Returns (rows, only-in-a, only-in-b, incomparable)."""
    mods_a, alphas_a = load_modules(path_a, grep)
    mods_b, alphas_b = load_modules(path_b, grep)

    shared = sorted(set(mods_a) & set(mods_b))
    rows: list[dict[str, Any]] = []
    incomparable: list[str] = []
    for name in shared:
        row = compare_module(
            name, mods_a[name], mods_b[name], alphas_a.get(name), alphas_b.get(name)
        )
        if row is None:
            incomparable.append(name)
        else:
            rows.append(row)

    rows.sort(key=lambda r: r.get("cos", -2.0))  # most-diverged first
    return (
        rows,
        sorted(set(mods_a) - set(mods_b)),
        sorted(set(mods_b) - set(mods_a)),
        incomparable,
    )


def _median(values: list[float]) -> float:
    return sorted(values)[len(values) // 2] if values else 0.0


def _verdict(cos: float, ratio: float) -> str:
    """One line naming what the medians mean, so the table isn't read cold."""
    if cos >= 0.99 and 0.95 <= ratio <= 1.05:
        return "effectively identical — the later checkpoint learned nothing new."
    if cos >= 0.95:
        return (
            "same direction, different strength — a rescale rather than a different "
            "adaptation."
        )
    if cos >= 0.7:
        return "largely the same adaptation, with real drift in some modules."
    if cos >= 0.0:
        return "materially different adaptations — these learned different things."
    return (
        "opposed — the updates largely cancel; check you are comparing the right files."
    )


def _print_report(
    path_a: str,
    path_b: str,
    rows: list[dict[str, Any]],
    only_a: list[str],
    only_b: list[str],
    incomparable: list[str],
    args: argparse.Namespace,
) -> None:
    live = [r for r in rows if not r["non_finite"]]
    fried = [r for r in rows if r["non_finite"]]

    print(f"a : {path_a}")
    print(f"b : {path_b}")
    print(f"shared modules : {len(rows)}")
    if only_a:
        print(f"only in a      : {len(only_a)} ({', '.join(only_a[:4])}…)")
    if only_b:
        print(f"only in b      : {len(only_b)} ({', '.join(only_b[:4])}…)")
    if incomparable:
        print(f"incomparable   : {len(incomparable)} (shape or format mismatch)")
    if fried:
        print(
            f"non-finite     : {len(fried)} ({', '.join(r['name'] for r in fried[:4])})"
        )

    if live:
        cos = [r["cos"] for r in live]
        ratios = [r["ratio"] for r in live if r["ratio"] != float("inf")]
        med_cos, med_ratio = _median(cos), _median(ratios)
        print("\nsummary")
        print(
            f"  cos    : min {min(cos):.4f}  median {med_cos:.4f}  max {max(cos):.4f}"
        )
        print(f"  ratio  : median {med_ratio:.3f}")
        print(f"  rel Δ  : median {_median([r['rel_delta'] for r in live]):.3f}")
        print(f"  read   : {_verdict(med_cos, med_ratio)}")

        shown = live if args.all else live[: args.top]
        head = "" if args.all else f" (lowest {len(shown)} cos; --all for every module)"
        print(f"\nmodules{head}:")
        width = min(max((len(r["name"]) for r in shown), default=6), 60)
        print(
            f"  {'cos':>7}  {'ratio':>7}  {'rel Δ':>7}  {'‖ΔW‖_a':>9}  "
            f"{'‖ΔW‖_b':>9}  module"
        )
        for r in shown:
            print(
                f"  {r['cos']:7.4f}  {r['ratio']:7.3f}  {r['rel_delta']:7.3f}  "
                f"{r['fro_a']:9.3g}  {r['fro_b']:9.3g}  {r['name'][:width]}"
            )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("a", help="first .safetensors LoRA (the baseline)")
    ap.add_argument("b", help="second .safetensors LoRA (compared against a)")
    ap.add_argument(
        "--all", action="store_true", help="print every module, not just the top N"
    )
    ap.add_argument(
        "--top",
        type=int,
        default=12,
        help="how many of the most-diverged modules to show (default: 12)",
    )
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    ap.add_argument(
        "--grep", metavar="SUBSTR", help="only modules whose keys contain SUBSTR"
    )
    args = ap.parse_args()

    try:
        rows, only_a, only_b, incomparable = compare(args.a, args.b, args.grep)
    except Exception as e:
        sys.exit(f"error reading inputs: {e}")

    if not rows:
        sys.exit(
            f"error: no comparable modules shared by {args.a} and {args.b} "
            "(different architectures, or --grep matched nothing)"
        )

    if args.json:
        print(
            json.dumps(
                {
                    "a": args.a,
                    "b": args.b,
                    "modules": rows,
                    "only_in_a": only_a,
                    "only_in_b": only_b,
                    "incomparable": incomparable,
                },
                indent=2,
            )
        )
        return
    _print_report(args.a, args.b, rows, only_a, only_b, incomparable, args)


if __name__ == "__main__":
    main()
