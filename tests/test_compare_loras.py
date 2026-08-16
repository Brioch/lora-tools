"""Tests for tools/compare_loras.py — the ΔW inner-product shortcuts and the CLI."""

import json
import struct
import sys

import compare_loras
import lora_health
import numpy as np
import pytest
from compare_loras import _kind, compare, compare_module, inner_product

_ST_DTYPE = {"float64": "F64", "float32": "F32", "float16": "F16"}


def write_st(path, arrays):
    """Build a minimal .safetensors file from {name: ndarray}."""
    header, blob = {}, b""
    for name, a in arrays.items():
        a = np.ascontiguousarray(a)
        begin = len(blob)
        blob += a.tobytes()
        header[name] = {
            "dtype": _ST_DTYPE[str(a.dtype)],
            "shape": list(a.shape),
            "data_offsets": [begin, len(blob)],
        }
    hb = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(blob)
    return str(path)


def lora_module(rng, rank=4, out=12, inp=16, alpha=None):
    """One LoRA module's tensors, optionally with an alpha scalar."""
    t = {
        "lora_down.weight": rng.standard_normal((rank, inp)).astype("float32"),
        "lora_up.weight": rng.standard_normal((out, rank)).astype("float32"),
    }
    if alpha is not None:
        t["alpha"] = np.array(alpha, "float32")
    return t


def make_ckpt(path, seed, n_modules=3, rank=4, alpha=None, scale=1.0):
    """Write a checkpoint of LoRA modules; *scale* multiplies every up-projection."""
    rng = np.random.default_rng(seed)
    arrays = {}
    for i in range(n_modules):
        for suffix, val in lora_module(rng, rank=rank, alpha=alpha).items():
            arrays[f"blocks.{i}.{suffix}"] = (
                val * scale if suffix == "lora_up.weight" else val
            )
    return write_st(path, arrays)


def f64(x):
    return np.asarray(x, dtype=np.float64)


class TestKind:
    def test_lora_down_up(self):
        assert _kind({"lora_down": f64([[1.0]]), "lora_up": f64([[1.0]])}) == "lora"

    def test_lora_a_b(self):
        assert _kind({"lora_A": f64([[1.0]]), "lora_B": f64([[1.0]])}) == "lora"

    def test_lokr(self):
        assert _kind({"lokr_w1": f64([[1.0]]), "lokr_w2": f64([[1.0]])}) == "lokr"

    def test_loha(self):
        assert _kind({"hada_w1_a": f64([[1.0]])}) == "loha"

    def test_unrecognized(self):
        assert _kind({"weight": f64([[1.0]])}) is None


class TestInnerProduct:
    """The shortcuts must equal the brute-force Frobenius product of the full ΔW."""

    def test_lora_matches_brute_force_across_differing_ranks(self):
        rng = np.random.default_rng(0)
        a = {
            "lora_down": f64(rng.standard_normal((4, 16))),
            "lora_up": f64(rng.standard_normal((12, 4))),
        }
        b = {
            "lora_down": f64(rng.standard_normal((6, 16))),
            "lora_up": f64(rng.standard_normal((12, 6))),
        }
        brute = np.sum(
            (a["lora_up"] @ a["lora_down"]) * (b["lora_up"] @ b["lora_down"])
        )
        assert inner_product(a, b, "lora") == pytest.approx(brute)

    def test_lokr_matches_the_kronecker_product(self):
        rng = np.random.default_rng(1)
        a = {
            "lokr_w1": f64(rng.standard_normal((3, 4))),
            "lokr_w2": f64(rng.standard_normal((5, 6))),
        }
        b = {
            "lokr_w1": f64(rng.standard_normal((3, 4))),
            "lokr_w2": f64(rng.standard_normal((5, 6))),
        }
        brute = np.sum(
            np.kron(a["lokr_w1"], a["lokr_w2"]) * np.kron(b["lokr_w1"], b["lokr_w2"])
        )
        assert inner_product(a, b, "lokr") == pytest.approx(brute)

    def test_lokr_from_factored_form(self):
        rng = np.random.default_rng(2)
        w = {
            "lokr_w1_a": f64(rng.standard_normal((3, 2))),
            "lokr_w1_b": f64(rng.standard_normal((2, 4))),
            "lokr_w2_a": f64(rng.standard_normal((5, 2))),
            "lokr_w2_b": f64(rng.standard_normal((2, 6))),
        }
        full = np.kron(w["lokr_w1_a"] @ w["lokr_w1_b"], w["lokr_w2_a"] @ w["lokr_w2_b"])
        assert inner_product(w, w, "lokr") == pytest.approx(np.sum(full * full))

    def test_loha_matches_the_hadamard_product(self):
        rng = np.random.default_rng(3)
        w = {
            "hada_w1_a": f64(rng.standard_normal((8, 3))),
            "hada_w1_b": f64(rng.standard_normal((3, 8))),
            "hada_w2_a": f64(rng.standard_normal((8, 3))),
            "hada_w2_b": f64(rng.standard_normal((3, 8))),
        }
        full = (w["hada_w1_a"] @ w["hada_w1_b"]) * (w["hada_w2_a"] @ w["hada_w2_b"])
        assert inner_product(w, w, "loha") == pytest.approx(np.sum(full * full))

    def test_is_symmetric(self):
        rng = np.random.default_rng(4)
        a = {
            "lora_down": f64(rng.standard_normal((3, 8))),
            "lora_up": f64(rng.standard_normal((6, 3))),
        }
        b = {
            "lora_down": f64(rng.standard_normal((3, 8))),
            "lora_up": f64(rng.standard_normal((6, 3))),
        }
        assert inner_product(a, b, "lora") == pytest.approx(inner_product(b, a, "lora"))


class TestCompareModule:
    def _pair(self, seed=0, rank=4):
        rng = np.random.default_rng(seed)
        return {
            "lora_down": f64(rng.standard_normal((rank, 16))),
            "lora_up": f64(rng.standard_normal((12, rank))),
        }

    def test_identical_gives_cos_one_and_no_change(self):
        w = self._pair()
        row = compare_module("m", w, w, None, None)
        assert row["cos"] == pytest.approx(1.0)
        assert row["ratio"] == pytest.approx(1.0)
        assert row["rel_delta"] == pytest.approx(0.0, abs=1e-9)

    def test_pure_rescale_keeps_cos_one_and_moves_ratio(self):
        a = self._pair()
        b = {"lora_down": a["lora_down"], "lora_up": a["lora_up"] * 3.0}
        row = compare_module("m", a, b, None, None)
        assert row["cos"] == pytest.approx(1.0)
        assert row["ratio"] == pytest.approx(3.0)
        assert row["rel_delta"] == pytest.approx(2.0)  # ‖3a − a‖/‖a‖

    def test_sign_flip_is_opposed(self):
        a = self._pair()
        b = {"lora_down": a["lora_down"], "lora_up": -a["lora_up"]}
        assert compare_module("m", a, b, None, None)["cos"] == pytest.approx(-1.0)

    def test_independent_modules_are_far_from_one(self):
        row = compare_module("m", self._pair(1), self._pair(2), None, None)
        assert abs(row["cos"]) < 0.5

    def test_alpha_scales_the_norms_but_not_cos(self):
        w = self._pair()
        plain = compare_module("m", w, w, None, None)
        scaled = compare_module("m", w, w, 8.0, 4.0)  # rank 4 -> scales 2.0 and 1.0
        assert scaled["cos"] == pytest.approx(plain["cos"])
        assert scaled["fro_a"] == pytest.approx(plain["fro_a"] * 2.0)
        assert scaled["ratio"] == pytest.approx(0.5)

    def test_differing_ranks_still_compare(self):
        row = compare_module("m", self._pair(rank=4), self._pair(rank=8), None, None)
        assert row is not None
        assert row["rank_a"] == 4 and row["rank_b"] == 8

    def test_non_finite_is_flagged_not_scored(self):
        a = self._pair()
        b = self._pair()
        b["lora_up"][0, 0] = np.nan
        row = compare_module("m", a, b, None, None)
        assert row["non_finite"] is True
        assert "cos" not in row

    def test_shape_mismatch_is_incomparable(self):
        a = {
            "lora_down": f64(np.ones((2, 8))),
            "lora_up": f64(np.ones((6, 2))),
        }
        b = {
            "lora_down": f64(np.ones((2, 9))),  # different in-dimension
            "lora_up": f64(np.ones((6, 2))),
        }
        assert compare_module("m", a, b, None, None) is None

    def test_format_mismatch_is_incomparable(self):
        a = self._pair()
        b = {"lokr_w1": f64(np.ones((3, 4))), "lokr_w2": f64(np.ones((5, 6)))}
        assert compare_module("m", a, b, None, None) is None

    def test_unrecognized_module_is_incomparable(self):
        w = {"weight": f64(np.ones((4, 4)))}
        assert compare_module("m", w, w, None, None) is None

    def test_zero_module_does_not_divide_by_zero(self):
        z = {"lora_down": f64(np.zeros((2, 8))), "lora_up": f64(np.zeros((6, 2)))}
        row = compare_module("m", z, z, None, None)
        assert row["cos"] == 0.0
        assert row["rel_delta"] == 0.0


class TestLoraFactorsAndRank:
    def test_lora_a_b_naming_is_handled(self):
        rng = np.random.default_rng(0)
        w = {
            "lora_A": f64(rng.standard_normal((3, 8))),
            "lora_B": f64(rng.standard_normal((6, 3))),
        }
        brute = np.sum((w["lora_B"] @ w["lora_A"]) ** 2)
        assert inner_product(w, w, "lora") == pytest.approx(brute)

    def test_lokr_rank_comes_from_w1_b(self):
        w = {
            "lokr_w1_a": f64(np.ones((3, 2))),
            "lokr_w1_b": f64(np.ones((2, 4))),
            "lokr_w2_a": f64(np.ones((5, 2))),
            "lokr_w2_b": f64(np.ones((2, 6))),
        }
        assert compare_module("m", w, w, 4.0, 4.0)["rank_a"] == 2

    def test_unfactored_lokr_has_no_rank(self):
        w = {"lokr_w1": f64(np.ones((3, 4))), "lokr_w2": f64(np.ones((5, 6)))}
        row = compare_module("m", w, w, 8.0, 8.0)
        assert row["rank_a"] is None
        assert row["cos"] == pytest.approx(1.0)

    def test_loha_rank_comes_from_hada_w1_b(self):
        rng = np.random.default_rng(1)
        w = {
            "hada_w1_a": f64(rng.standard_normal((8, 3))),
            "hada_w1_b": f64(rng.standard_normal((3, 8))),
            "hada_w2_a": f64(rng.standard_normal((8, 3))),
            "hada_w2_b": f64(rng.standard_normal((3, 8))),
        }
        assert compare_module("m", w, w, 6.0, 6.0)["rank_a"] == 3


class TestVerdicts:
    @pytest.mark.parametrize(
        "cos,ratio,expected",
        [
            (1.0, 1.0, "effectively identical"),
            (0.97, 3.0, "same direction, different strength"),
            (0.80, 1.0, "largely the same adaptation"),
            (0.20, 1.0, "materially different adaptations"),
            (-0.9, 1.0, "opposed"),
        ],
    )
    def test_reading_matches_the_medians(self, cos, ratio, expected):
        assert expected in compare_loras._verdict(cos, ratio)


class TestSkippedDtypes:
    def test_unsupported_dtype_is_ignored(self, tmp_path):
        """A float8/int tensor has no meaningful norm, so it must not form a module."""
        path = tmp_path / "odd.safetensors"
        arrays = {
            "blocks.0.lora_down.weight": np.ones((2, 8), "float32"),
            "blocks.0.lora_up.weight": np.ones((6, 2), "float32"),
        }
        write_st(path, arrays)
        # Rewrite the header so one tensor claims an unsupported dtype.
        raw = path.read_bytes()
        n = struct.unpack("<Q", raw[:8])[0]
        header = json.loads(raw[8 : 8 + n])
        header["blocks.1.lora_down.weight"] = {
            "dtype": "F8_E4M3",
            "shape": [2, 2],
            "data_offsets": [0, 4],
        }
        blob = json.dumps(header).encode()
        path.write_bytes(struct.pack("<Q", len(blob)) + blob + raw[8 + n :])

        mods, _ = compare_loras.load_modules(str(path))
        assert "blocks.1" not in mods
        assert "blocks.0" in mods


class TestAgreementWithLoraHealth:
    def test_fro_matches_lora_health_for_the_same_file(self, tmp_path):
        """The two tools must report the same ‖ΔW‖_F, alpha scale included."""
        path = make_ckpt(tmp_path / "a.safetensors", seed=5, alpha=8.0)
        rows, _ = lora_health.collect(path)
        health = {r["name"]: r["fro"] for r in rows}
        mine, _, _, _ = compare(path, path)
        assert mine, "expected comparable modules"
        for row in mine:
            assert row["fro_a"] == pytest.approx(health[row["name"]], rel=1e-9)


class TestCompare:
    def test_identical_files_are_identical(self, tmp_path):
        path = make_ckpt(tmp_path / "a.safetensors", seed=0)
        rows, only_a, only_b, bad = compare(path, path)
        assert len(rows) == 3
        assert (only_a, only_b, bad) == ([], [], [])
        assert all(r["cos"] == pytest.approx(1.0) for r in rows)

    def test_rescaled_file_keeps_cos_one(self, tmp_path):
        a = make_ckpt(tmp_path / "a.safetensors", seed=0)
        b = make_ckpt(tmp_path / "b.safetensors", seed=0, scale=2.0)
        rows, _, _, _ = compare(a, b)
        assert all(r["cos"] == pytest.approx(1.0, abs=1e-6) for r in rows)
        assert all(r["ratio"] == pytest.approx(2.0, rel=1e-5) for r in rows)

    def test_unrelated_files_diverge(self, tmp_path):
        a = make_ckpt(tmp_path / "a.safetensors", seed=1)
        b = make_ckpt(tmp_path / "b.safetensors", seed=2)
        rows, _, _, _ = compare(a, b)
        assert all(abs(r["cos"]) < 0.5 for r in rows)

    def test_reports_modules_unique_to_each_side(self, tmp_path):
        a = make_ckpt(tmp_path / "a.safetensors", seed=0, n_modules=4)
        b = make_ckpt(tmp_path / "b.safetensors", seed=0, n_modules=2)
        rows, only_a, only_b, _ = compare(a, b)
        assert len(rows) == 2
        assert only_a == ["blocks.2", "blocks.3"]
        assert only_b == []

    def test_grep_filters_modules(self, tmp_path):
        path = make_ckpt(tmp_path / "a.safetensors", seed=0, n_modules=4)
        rows, _, _, _ = compare(path, path, grep="blocks.1")
        assert [r["name"] for r in rows] == ["blocks.1"]

    def test_rows_are_sorted_most_diverged_first(self, tmp_path):
        a = make_ckpt(tmp_path / "a.safetensors", seed=1)
        b = make_ckpt(tmp_path / "b.safetensors", seed=2)
        rows, _, _, _ = compare(a, b)
        assert [r["cos"] for r in rows] == sorted(r["cos"] for r in rows)


class TestMainCLI:
    def test_prints_a_report(self, tmp_path, monkeypatch, capsys):
        a = make_ckpt(tmp_path / "a.safetensors", seed=0)
        monkeypatch.setattr(sys, "argv", ["compare_loras.py", a, a])
        compare_loras.main()
        out = capsys.readouterr().out
        assert "shared modules : 3" in out
        assert "cos" in out and "ratio" in out
        assert "effectively identical" in out

    def test_verdict_names_a_pure_rescale(self, tmp_path, monkeypatch, capsys):
        a = make_ckpt(tmp_path / "a.safetensors", seed=0)
        b = make_ckpt(tmp_path / "b.safetensors", seed=0, scale=4.0)
        monkeypatch.setattr(sys, "argv", ["compare_loras.py", a, b])
        compare_loras.main()
        assert "same direction, different strength" in capsys.readouterr().out

    def test_verdict_names_a_different_adaptation(self, tmp_path, monkeypatch, capsys):
        a = make_ckpt(tmp_path / "a.safetensors", seed=1)
        b = make_ckpt(tmp_path / "b.safetensors", seed=2)
        monkeypatch.setattr(sys, "argv", ["compare_loras.py", a, b])
        compare_loras.main()
        assert "different" in capsys.readouterr().out

    def test_all_and_top(self, tmp_path, monkeypatch, capsys):
        a = make_ckpt(tmp_path / "a.safetensors", seed=0, n_modules=5)
        monkeypatch.setattr(sys, "argv", ["compare_loras.py", a, a, "--all"])
        compare_loras.main()
        assert "lowest" not in capsys.readouterr().out

        monkeypatch.setattr(sys, "argv", ["compare_loras.py", a, a, "--top", "2"])
        compare_loras.main()
        assert "lowest 2 cos" in capsys.readouterr().out

    def test_json_output(self, tmp_path, monkeypatch, capsys):
        a = make_ckpt(tmp_path / "a.safetensors", seed=0)
        monkeypatch.setattr(sys, "argv", ["compare_loras.py", a, a, "--json"])
        compare_loras.main()
        payload = json.loads(capsys.readouterr().out)
        assert len(payload["modules"]) == 3
        assert payload["only_in_a"] == [] and payload["incomparable"] == []

    def test_reports_non_finite_modules(self, tmp_path, monkeypatch, capsys):
        a = make_ckpt(tmp_path / "a.safetensors", seed=0, n_modules=1)
        rng = np.random.default_rng(0)
        bad = rng.standard_normal((12, 4)).astype("float32")
        bad[0, 0] = np.inf
        b = write_st(
            tmp_path / "b.safetensors",
            {
                "blocks.0.lora_down.weight": rng.standard_normal((4, 16)).astype(
                    "float32"
                ),
                "blocks.0.lora_up.weight": bad,
            },
        )
        monkeypatch.setattr(sys, "argv", ["compare_loras.py", a, b])
        compare_loras.main()
        assert "non-finite     : 1" in capsys.readouterr().out

    def test_no_shared_modules_exits(self, tmp_path, monkeypatch):
        a = write_st(
            tmp_path / "a.safetensors",
            {
                "x.lora_down.weight": np.ones((2, 4), "float32"),
                "x.lora_up.weight": np.ones((6, 2), "float32"),
            },
        )
        b = write_st(
            tmp_path / "b.safetensors",
            {
                "y.lora_down.weight": np.ones((2, 4), "float32"),
                "y.lora_up.weight": np.ones((6, 2), "float32"),
            },
        )
        monkeypatch.setattr(sys, "argv", ["compare_loras.py", a, b])
        with pytest.raises(SystemExit) as e:
            compare_loras.main()
        assert "no comparable modules" in str(e.value.code)

    def test_unreadable_input_exits(self, tmp_path, monkeypatch):
        bad = tmp_path / "bad.safetensors"
        bad.write_bytes(b"not a safetensors file")
        monkeypatch.setattr(sys, "argv", ["compare_loras.py", str(bad), str(bad)])
        with pytest.raises(SystemExit) as e:
            compare_loras.main()
        assert "error reading inputs" in str(e.value.code)

    def test_reports_unique_and_incomparable_modules(
        self, tmp_path, monkeypatch, capsys
    ):
        a = write_st(
            tmp_path / "a.safetensors",
            {
                "shared.lora_down.weight": np.ones((2, 8), "float32"),
                "shared.lora_up.weight": np.ones((6, 2), "float32"),
                "mismatch.lora_down.weight": np.ones((2, 8), "float32"),
                "mismatch.lora_up.weight": np.ones((6, 2), "float32"),
                "onlya.lora_down.weight": np.ones((2, 8), "float32"),
                "onlya.lora_up.weight": np.ones((6, 2), "float32"),
            },
        )
        b = write_st(
            tmp_path / "b.safetensors",
            {
                "shared.lora_down.weight": np.ones((2, 8), "float32"),
                "shared.lora_up.weight": np.ones((6, 2), "float32"),
                # Same module name, incompatible in-dimension.
                "mismatch.lora_down.weight": np.ones((2, 9), "float32"),
                "mismatch.lora_up.weight": np.ones((6, 2), "float32"),
                "onlyb.lora_down.weight": np.ones((2, 8), "float32"),
                "onlyb.lora_up.weight": np.ones((6, 2), "float32"),
            },
        )
        monkeypatch.setattr(sys, "argv", ["compare_loras.py", a, b])
        compare_loras.main()
        out = capsys.readouterr().out
        assert "only in a      : 1 (onlya…)" in out
        assert "only in b      : 1 (onlyb…)" in out
        assert "incomparable   : 1 (shape or format mismatch)" in out

    def test_grep_matching_nothing_exits(self, tmp_path, monkeypatch):
        a = make_ckpt(tmp_path / "a.safetensors", seed=0)
        monkeypatch.setattr(sys, "argv", ["compare_loras.py", a, a, "--grep", "zzz"])
        with pytest.raises(SystemExit):
            compare_loras.main()
