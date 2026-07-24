"""Tests for tools/lora_health.py — decode, ΔW reconstruction, flags, and CLI."""

import json
import struct
import sys

import lora_health
import numpy as np
import pytest
from lora_health import analyse_module, product_svdvals, to_array

_ST_DTYPE = {"float64": "F64", "float32": "F32", "float16": "F16"}


def health_line(out, label):
    """Return the text after ':' on the health-report line starting with `label`."""
    line = next(ln for ln in out.splitlines() if ln.strip().startswith(label))
    return line.split(":", 1)[1].strip()


def write_st(path, arrays):
    """Build a .safetensors file from {name: ndarray | ("BF16", raw_bytes, shape)}."""
    header, blob = {}, b""
    for name, val in arrays.items():
        if isinstance(val, tuple):  # pre-encoded (e.g. BF16, or an exotic dtype)
            dtype, raw, shape = val
        else:
            a = np.ascontiguousarray(val)
            dtype, raw, shape = _ST_DTYPE[str(a.dtype)], a.tobytes(), list(a.shape)
        begin = len(blob)
        blob += raw
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [begin, len(blob)],
        }
    hb = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        f.write(blob)
    return str(path)


class TestDecode:
    def test_f32_round_trip(self):
        a = to_array(struct.pack("<3f", 1.0, -2.0, 3.5), "F32", [3])
        assert list(a) == [1.0, -2.0, 3.5]

    def test_bf16_widens_correctly(self):
        raw = b"".join(struct.pack("<f", v)[2:4] for v in (1.0, -2.0, 0.5))
        assert list(to_array(raw, "BF16", [3])) == [1.0, -2.0, 0.5]

    def test_unsupported_dtype_is_none(self):
        assert to_array(b"\x00\x00", "F8_E4M3", [1]) is None


class TestSpectrum:
    def test_product_svdvals_matches_full_product(self):
        rng = np.random.default_rng(0)
        down = rng.standard_normal((8, 32))  # (r, in)
        up = rng.standard_normal((16, 8))  # (out, r)
        got = np.sort(product_svdvals(down, up))[::-1]
        # np.linalg.svd returns min(out,in) values (rank r nonzero + zeros); the
        # QR path returns just the r nonzero ones, so compare against the top r.
        want = np.sort(np.linalg.svd(up @ down, compute_uv=False))[::-1][: got.size]
        assert np.allclose(got, want)

    def test_lora_fro_and_stable_rank(self):
        rng = np.random.default_rng(1)
        down, up = rng.standard_normal((4, 20)), rng.standard_normal((12, 4))
        row = analyse_module("m", {"lora_down": down, "lora_up": up}, None)
        dw = up @ down
        assert row["fro"] == pytest.approx(np.linalg.norm(dw))
        s = np.linalg.svd(dw, compute_uv=False)
        assert row["stable_rank"] == pytest.approx(np.sum(s**2) / s.max() ** 2)
        assert row["rank"] == 4

    def test_rank_one_module_has_stable_rank_one(self):
        up = np.ones((10, 1))
        down = np.arange(1.0, 6.0).reshape(1, 5)
        row = analyse_module("m", {"lora_down": down, "lora_up": up}, None)
        assert row["stable_rank"] == pytest.approx(1.0)

    def test_alpha_scales_magnitude(self):
        down, up = np.ones((2, 3)), np.ones((4, 2))
        base = analyse_module("m", {"lora_down": down, "lora_up": up}, None)
        scaled = analyse_module("m", {"lora_down": down, "lora_up": up}, alpha=1.0)
        # scale = alpha / rank = 1/2, applied to the norm but not the stable rank.
        assert scaled["fro"] == pytest.approx(base["fro"] * 0.5)
        assert scaled["stable_rank"] == pytest.approx(base["stable_rank"])

    def test_peft_ab_format(self):
        rng = np.random.default_rng(5)
        a, b = rng.standard_normal((6, 24)), rng.standard_normal((18, 6))
        row = analyse_module("m", {"lora_A": a, "lora_B": b}, None)
        assert row["format"] == "lora" and row["rank"] == 6
        assert row["fro"] == pytest.approx(np.linalg.norm(b @ a))

    def test_lokr_fro_is_product_of_factor_norms(self):
        rng = np.random.default_rng(2)
        w1, w2 = rng.standard_normal((3, 4)), rng.standard_normal((5, 6))
        row = analyse_module("m", {"lokr_w1": w1, "lokr_w2": w2}, None)
        assert row["fro"] == pytest.approx(np.linalg.norm(w1) * np.linalg.norm(w2))
        assert row["format"] == "lokr"

    def test_lokr_decomposed_factor(self):
        rng = np.random.default_rng(6)
        w1a, w1b = rng.standard_normal((3, 2)), rng.standard_normal((2, 4))  # w1 = a@b
        w2 = rng.standard_normal((5, 6))
        row = analyse_module(
            "m", {"lokr_w1_a": w1a, "lokr_w1_b": w1b, "lokr_w2": w2}, None
        )
        want = np.linalg.norm(w1a @ w1b) * np.linalg.norm(w2)
        assert row["fro"] == pytest.approx(want)
        assert row["rank"] == 2  # inner dim of the w1 decomposition

    def test_loha_hadamard_product(self):
        rng = np.random.default_rng(7)
        w = {
            "hada_w1_a": rng.standard_normal((4, 2)),
            "hada_w1_b": rng.standard_normal((2, 5)),
            "hada_w2_a": rng.standard_normal((4, 2)),
            "hada_w2_b": rng.standard_normal((2, 5)),
        }
        row = analyse_module("m", w, None)
        dw = (w["hada_w1_a"] @ w["hada_w1_b"]) * (w["hada_w2_a"] @ w["hada_w2_b"])
        assert row["format"] == "loha"
        assert row["fro"] == pytest.approx(np.linalg.norm(dw))

    def test_one_dim_weight_uses_as_2d(self):
        # A 1-D lokr factor still reconstructs (as_2d treats it as a row vector).
        row = analyse_module(
            "m", {"lokr_w1": np.array([3.0, 4.0]), "lokr_w2": np.ones((1, 1))}, None
        )
        assert row["fro"] == pytest.approx(5.0)

    def test_non_finite_module_flagged(self):
        up = np.array([[1.0], [np.inf]])
        row = analyse_module("m", {"lora_down": np.ones((1, 2)), "lora_up": up}, None)
        assert row["non_finite"] is True

    def test_incomplete_lokr_falls_back_to_raw(self):
        # Only w1 present (no w2, no w2_a/_b) -> not reconstructable -> raw norm.
        row = analyse_module("m", {"lokr_w1": np.ones((2, 2))}, None)
        assert row["format"] == "raw" and row["stable_rank"] is None

    def test_incomplete_loha_falls_back_to_raw(self):
        row = analyse_module("m", {"hada_w1_a": np.ones((2, 2))}, None)
        assert row["format"] == "raw"


@pytest.fixture
def healthy(tmp_path):
    rng = np.random.default_rng(3)
    arrays = {}
    for i in range(3):
        n = f"diffusion_model.blocks.{i}.mlp"
        arrays[f"{n}.lora_down.weight"] = rng.standard_normal((4, 16)).astype("float32")
        arrays[f"{n}.lora_up.weight"] = rng.standard_normal((16, 4)).astype("float32")
    return write_st(tmp_path / "healthy.safetensors", arrays)


class TestCLI:
    def test_summary(self, healthy, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["lora_health.py", healthy])
        lora_health.main()
        out = capsys.readouterr().out
        assert "health" in out and "stable rank" in out and "modules" in out

    def test_all_and_json(self, healthy, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["lora_health.py", healthy, "--json"])
        lora_health.main()
        data = json.loads(capsys.readouterr().out)
        assert len(data["modules"]) == 3
        assert all(m["format"] == "lora" and m["rank"] == 4 for m in data["modules"])

    def test_grep(self, healthy, monkeypatch, capsys):
        monkeypatch.setattr(
            sys, "argv", ["lora_health.py", healthy, "--grep", "blocks.0", "--json"]
        )
        lora_health.main()
        assert len(json.loads(capsys.readouterr().out)["modules"]) == 1

    def test_all_flag_lists_modules(self, healthy, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["lora_health.py", healthy, "--all"])
        lora_health.main()
        assert "modules:" in capsys.readouterr().out  # no "top N" suffix

    def test_non_finite_exits_nonzero(self, tmp_path, monkeypatch, capsys):
        path = write_st(
            tmp_path / "fried.safetensors",
            {
                "m.lora_down.weight": np.ones((1, 2), "float32"),
                "m.lora_up.weight": np.array([[1.0], [np.inf]], "float32"),
            },
        )
        monkeypatch.setattr(sys, "argv", ["lora_health.py", path])
        with pytest.raises(SystemExit) as e:
            lora_health.main()
        assert e.value.code == 1
        assert "non-finite" in capsys.readouterr().out

    def test_suspect_flag_for_strong_collapsed_module(
        self, tmp_path, monkeypatch, capsys
    ):
        rng = np.random.default_rng(4)
        arrays = {}
        for i in range(5):  # ordinary spread-out modules
            n = f"blocks.{i}"
            arrays[f"{n}.lora_down.weight"] = rng.standard_normal((4, 16)).astype(
                "float32"
            )
            arrays[f"{n}.lora_up.weight"] = rng.standard_normal((16, 4)).astype(
                "float32"
            )
        # A rank-1, high-magnitude module: strong AND collapsed -> suspect.
        arrays["blocks.9.lora_down.weight"] = (np.ones((1, 16)) * 30).astype("float32")
        arrays["blocks.9.lora_up.weight"] = (np.ones((16, 1)) * 30).astype("float32")
        path = write_st(tmp_path / "cooked.safetensors", arrays)
        monkeypatch.setattr(sys, "argv", ["lora_health.py", path])
        lora_health.main()
        assert "blocks.9" in health_line(capsys.readouterr().out, "suspect")

    def test_dead_module_flagged(self, tmp_path, monkeypatch, capsys):
        arrays = {
            "a.lora_down.weight": np.ones((2, 4), "float32"),
            "a.lora_up.weight": np.ones((4, 2), "float32"),
            "b.lora_down.weight": np.zeros((2, 4), "float32"),  # dead
            "b.lora_up.weight": np.zeros((4, 2), "float32"),
        }
        path = write_st(tmp_path / "d.safetensors", arrays)
        monkeypatch.setattr(sys, "argv", ["lora_health.py", path])
        lora_health.main()
        assert health_line(capsys.readouterr().out, "dead") == "b"

    def test_skipped_dtype_noted(self, tmp_path, monkeypatch, capsys):
        arrays = {
            "a.lora_down.weight": np.ones((2, 4), "float32"),
            "a.lora_up.weight": np.ones((4, 2), "float32"),
            "a.quant": ("F8_E4M3", b"\x00\x00", [2]),
        }
        path = write_st(tmp_path / "q.safetensors", arrays)
        monkeypatch.setattr(sys, "argv", ["lora_health.py", path])
        lora_health.main()
        assert "skipped (dtype)" in capsys.readouterr().out

    def test_raw_fallback_for_unknown_module(self, tmp_path, monkeypatch, capsys):
        # A lone tensor with no recognised math suffix -> raw Frobenius fallback.
        path = write_st(
            tmp_path / "r.safetensors", {"lonely.diff": np.ones((3, 3), "float32")}
        )
        monkeypatch.setattr(sys, "argv", ["lora_health.py", path, "--json"])
        lora_health.main()
        (mod,) = json.loads(capsys.readouterr().out)["modules"]
        assert mod["format"] == "raw" and mod["stable_rank"] is None
        assert mod["fro"] == pytest.approx(3.0)  # ‖ones(3,3)‖_F = 3

    def test_alpha_tensor_applied_via_collect(self, tmp_path, monkeypatch, capsys):
        arrays = {
            "m.lora_down.weight": np.ones((2, 4), "float32"),
            "m.lora_up.weight": np.ones((4, 2), "float32"),
            "m.alpha": np.array(1.0, "float32"),  # scale = alpha/rank = 1/2
        }
        path = write_st(tmp_path / "al.safetensors", arrays)
        monkeypatch.setattr(sys, "argv", ["lora_health.py", path, "--json"])
        lora_health.main()
        (mod,) = json.loads(capsys.readouterr().out)["modules"]
        assert mod["alpha"] == 1.0 and mod["scale"] == pytest.approx(0.5)

    def test_missing_file_exits(self, tmp_path, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["lora_health.py", str(tmp_path / "nope.st")])
        with pytest.raises(SystemExit):
            lora_health.main()

    def test_no_modules_exits(self, tmp_path, monkeypatch):
        path = write_st(tmp_path / "empty.safetensors", {})
        monkeypatch.setattr(sys, "argv", ["lora_health.py", path])
        with pytest.raises(SystemExit) as e:
            lora_health.main()
        assert "no reconstructable" in str(e.value.code)
