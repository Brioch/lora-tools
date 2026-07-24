"""Tests for tools/health_sweep.py — epoch parsing, expansion, summary, CLI."""

import csv
import io
import json
import struct
import sys

import health_sweep
import numpy as np
import pytest
from health_sweep import Summary, epoch_of, expand, summarise

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


def lora_module(rng, rank=4, out=16, inp=16, alpha=None):
    """One LoRA module's tensors, optionally with an alpha scalar."""
    t = {
        "lora_down.weight": rng.standard_normal((rank, inp)).astype("float32"),
        "lora_up.weight": rng.standard_normal((out, rank)).astype("float32"),
    }
    if alpha is not None:
        t["alpha"] = np.array(alpha, "float32")
    return t


def make_ckpt(path, seed, n_modules=3, alpha=None):
    """Write a checkpoint with `n_modules` LoRA modules."""
    rng = np.random.default_rng(seed)
    arrays = {}
    for i in range(n_modules):
        for suffix, val in lora_module(rng, alpha=alpha).items():
            arrays[f"blocks.{i}.{suffix}"] = val
    return write_st(path, arrays)


def ns(**kw):
    """Args namespace with the defaults summarise/main read."""
    base = dict(
        grep=None,
        epoch_regex=None,
        dead_threshold=1e-6,
        outlier_factor=4.0,
        collapse_rank=2.0,
    )
    base.update(kw)
    return type("Args", (), base)()


class TestEpochOf:
    @pytest.mark.parametrize(
        "name, want",
        [
            ("run-epoch40.safetensors", 40),
            ("run_epoch_7.safetensors", 7),
            ("model-e12.safetensors", 12),
            ("ckpt-ep3.safetensors", 3),
            ("train-step_1200.safetensors", 1200),
            ("char-000060.safetensors", 60),
            ("plain-99.safetensors", 99),  # last-ditch digit run
        ],
    )
    def test_builtin_patterns(self, name, want):
        assert epoch_of(name, None)[0] == want

    def test_no_digits_is_minus_one(self):
        assert epoch_of("final.safetensors", None) == (-1, "final.safetensors")

    def test_override_regex(self):
        # The real-world case: pick the middle field of -save-<step>-<epoch>-<idx>.
        name = "2026-07-24_19-19-03-save-760-40-0.safetensors"
        assert epoch_of(name, r"-(\d+)-\d+\.safetensors$")[0] == 40

    def test_rightmost_match_wins(self):
        # Two 'step' tokens: the later one is the counter that varies.
        assert epoch_of("step_1-run-step_500.safetensors", None)[0] == 500


class TestExpand:
    def test_directory_recurses(self, tmp_path):
        (tmp_path / "sub").mkdir()
        a = write_st(tmp_path / "a.safetensors", {})
        b = write_st(tmp_path / "sub" / "b.safetensors", {})
        assert expand([str(tmp_path)]) == sorted([a, b])

    def test_glob_and_explicit_are_deduped(self, tmp_path):
        a = write_st(tmp_path / "a.safetensors", {})
        got = expand([str(tmp_path / "*.safetensors"), a])
        assert got == [a]

    def test_explicit_file_passthrough(self, tmp_path):
        a = write_st(tmp_path / "a.safetensors", {})
        assert expand([a]) == [a]


class TestSummarise:
    def test_metrics_match_underlying_tool(self, tmp_path):
        path = make_ckpt(tmp_path / "run-epoch5.safetensors", seed=1, n_modules=3)
        s = summarise(path, ns())
        assert s is not None
        assert s["epoch"] == 5 and s["modules"] == 3
        assert s["fro_min"] <= s["fro_med"] <= s["fro_max"]
        assert s["sr_min"] <= s["sr_med"] <= s["sr_max"]
        assert s["non_finite"] == 0 and s["dead"] == 0 and s["suspect"] == 0

    def test_grep_filters_modules(self, tmp_path):
        path = make_ckpt(tmp_path / "run-epoch1.safetensors", seed=2, n_modules=3)
        assert summarise(path, ns(grep="blocks.0"))["modules"] == 1

    def test_unreadable_file_returns_none(self, tmp_path, capsys):
        bad = tmp_path / "broken.safetensors"
        bad.write_bytes(b"not a safetensors header")
        assert summarise(str(bad), ns()) is None
        assert "broken.safetensors" in capsys.readouterr().err

    def test_no_modules_returns_none(self, tmp_path):
        empty = write_st(tmp_path / "empty.safetensors", {})
        assert summarise(empty, ns()) is None


def summ_row(epoch, fro_med, sr_med, suspect=0, label=""):
    """A minimal Summary for exercising the print helpers directly."""
    return Summary(
        epoch=epoch,
        label=label or str(epoch),
        file=f"e{epoch}.safetensors",
        modules=1,
        fro_min=fro_med,
        fro_med=fro_med,
        fro_max=fro_med,
        sr_min=sr_med,
        sr_med=sr_med,
        sr_max=sr_med,
        non_finite=0,
        dead=0,
        suspect=suspect,
    )


class TestPrintRead:
    def test_single_row_prints_nothing(self, capsys):
        health_sweep.print_read([summ_row(1, 10.0, 20.0)])
        assert capsys.readouterr().out == ""

    def test_peak_and_falling_rank(self, capsys):
        rows = [
            summ_row(10, 100.0, 30.0),
            summ_row(20, 200.0, 25.0),  # peak magnitude
            summ_row(30, 150.0, 18.0),  # rank down 40% -> collapse warning
        ]
        health_sweep.print_read(rows)
        out = capsys.readouterr().out
        assert "peaks at 20" in out
        assert "past its peak" in out
        assert "falling" in out and "collapsing" in out

    def test_flat_rank_no_collapse_note(self, capsys):
        rows = [summ_row(10, 100.0, 30.0), summ_row(20, 200.0, 30.0)]
        health_sweep.print_read(rows)
        out = capsys.readouterr().out
        assert "flat" in out and "collapsing" not in out

    def test_suspect_reported(self, capsys):
        rows = [summ_row(10, 100.0, 30.0), summ_row(20, 200.0, 30.0, suspect=1)]
        health_sweep.print_read(rows)
        assert "first 'suspect'" in capsys.readouterr().out


@pytest.fixture
def ckpt_dir(tmp_path):
    for ep in (10, 20, 30):
        make_ckpt(tmp_path / f"run-epoch{ep:03d}.safetensors", seed=ep)
    return str(tmp_path)


class TestCLI:
    def test_table_output(self, ckpt_dir, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["health_sweep.py", ckpt_dir])
        health_sweep.main()
        out = capsys.readouterr().out
        assert "swept 3 checkpoint(s)" in out
        assert "‖ΔW‖_F" in out and "stable rank" in out

    def test_csv_output(self, ckpt_dir, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["health_sweep.py", ckpt_dir, "--csv"])
        health_sweep.main()
        rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
        assert [r["epoch"] for r in rows] == ["10", "20", "30"]

    def test_files_flag_adds_column(self, ckpt_dir, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["health_sweep.py", ckpt_dir, "--files"])
        health_sweep.main()
        out = capsys.readouterr().out
        assert "file" in out and "run-epoch010.safetensors" in out

    def test_degenerate_epochs_warn_and_show_files(self, tmp_path, monkeypatch, capsys):
        # Names whose only digit is a constant -> all parse to the same epoch.
        for i in range(1, 4):
            make_ckpt(tmp_path / f"{i:03d}-krea-v0.safetensors", seed=i)
        monkeypatch.setattr(sys, "argv", ["health_sweep.py", str(tmp_path)])
        health_sweep.main()
        cap = capsys.readouterr()
        assert "could not parse distinct epochs" in cap.err
        assert "krea-v0.safetensors" in cap.out  # file column auto-shown

    def test_epoch_regex_recovers_labels(self, tmp_path, monkeypatch, capsys):
        for i in range(1, 4):
            make_ckpt(tmp_path / f"{i:03d}-krea-v0.safetensors", seed=i)
        monkeypatch.setattr(
            sys, "argv", ["health_sweep.py", str(tmp_path), "--epoch-regex", r"(\d+)-"]
        )
        health_sweep.main()
        cap = capsys.readouterr()
        assert "could not parse" not in cap.err  # distinct epochs now

    def test_no_matches_exits(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            sys, "argv", ["health_sweep.py", str(tmp_path / "none-*.safetensors")]
        )
        with pytest.raises(SystemExit) as e:
            health_sweep.main()
        assert "no .safetensors" in str(e.value.code)

    def test_no_usable_modules_exits(self, tmp_path, monkeypatch):
        write_st(tmp_path / "empty.safetensors", {})
        monkeypatch.setattr(sys, "argv", ["health_sweep.py", str(tmp_path)])
        with pytest.raises(SystemExit) as e:
            health_sweep.main()
        assert "no checkpoints yielded" in str(e.value.code)
