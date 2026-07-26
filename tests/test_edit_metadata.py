"""Tests for tools/edit_metadata.py — header round-trip, helpers, and CLI."""

import argparse
import json
import struct
import sys

import edit_metadata
import pytest
from edit_metadata import parse_set, read_header, rewrite, validate


def write_safetensors(path, metadata, buffer=b"\x00\x01\x02\x03"):
    """Write a minimal valid .safetensors file: 8-byte len + JSON header + buffer."""
    header = {
        "__metadata__": metadata,
        "weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, len(buffer)]},
    }
    blob = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(buffer)
    return buffer


def tensor_bytes(path):
    """Return the raw tensor buffer (everything after the JSON header)."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        f.seek(8 + n)
        return f.read()


class TestHeaderRoundTrip:
    def test_updates_metadata_and_preserves_tensor_bytes(self, tmp_path):
        src = tmp_path / "in.safetensors"
        dst = tmp_path / "out.safetensors"
        buffer = write_safetensors(src, {"a": "1"})

        rewrite(str(src), str(dst), {"a": "2", "b": "x"})

        _, header = read_header(str(dst))
        assert header["__metadata__"] == {"a": "2", "b": "x"}
        assert tensor_bytes(dst) == buffer

    def test_header_length_is_8_byte_aligned(self, tmp_path):
        src = tmp_path / "in.safetensors"
        dst = tmp_path / "out.safetensors"
        write_safetensors(src, {"a": "1"})

        rewrite(str(src), str(dst), {"title": "My Character v2"})

        n, _ = read_header(str(dst))
        assert n % 8 == 0

    def test_empty_meta_drops_metadata_key(self, tmp_path):
        src = tmp_path / "in.safetensors"
        dst = tmp_path / "out.safetensors"
        write_safetensors(src, {"a": "1"})

        rewrite(str(src), str(dst), {})

        _, header = read_header(str(dst))
        assert "__metadata__" not in header


class TestParseSet:
    def test_basic(self):
        assert parse_set("title=hello") == ("title", "hello")

    def test_value_may_contain_equals(self):
        assert parse_set("k=a=b") == ("k", "a=b")

    def test_strips_key_whitespace(self):
        assert parse_set("  title =v") == ("title", "v")

    def test_missing_equals(self):
        with pytest.raises(argparse.ArgumentTypeError, match="KEY=VALUE"):
            parse_set("novalue")

    def test_empty_key(self):
        with pytest.raises(argparse.ArgumentTypeError, match="empty key"):
            parse_set("=v")


class TestValidate:
    def test_prediction_type_ok(self):
        assert validate("modelspec.prediction_type", "v") is None
        assert validate("modelspec.prediction_type", "epsilon") is None

    def test_prediction_type_bad(self):
        assert "prediction_type" in validate("modelspec.prediction_type", "nope")

    def test_resolution(self):
        assert validate("modelspec.resolution", "1024x1024") is None
        assert "resolution" in validate("modelspec.resolution", "1024")

    def test_date(self):
        assert validate("modelspec.date", "2026-07-14") is None
        assert "date" in validate("modelspec.date", "July 2026")

    def test_hash(self):
        assert validate("modelspec.hash_sha256", "0xdeadbeef") is None
        assert "hash_sha256" in validate("modelspec.hash_sha256", "deadbeef")

    def test_unknown_key_passes(self):
        assert validate("modelspec.title", "anything") is None


class TestMainCLI:
    def test_set_named_field_writes_edited_file(self, tmp_path, monkeypatch, capsys):
        src = tmp_path / "lora.safetensors"
        write_safetensors(src, {})
        monkeypatch.setattr(sys, "argv", ["edit_metadata.py", str(src), "--title", "X"])
        edit_metadata.main()
        out = capsys.readouterr().out
        edited = tmp_path / "lora.edited.safetensors"
        assert edited.exists()
        _, header = read_header(str(edited))
        assert header["__metadata__"]["modelspec.title"] == "X"
        assert "output" in out

    def test_dry_run_writes_nothing(self, tmp_path, monkeypatch, capsys):
        src = tmp_path / "lora.safetensors"
        write_safetensors(src, {})
        monkeypatch.setattr(
            sys, "argv", ["edit_metadata.py", str(src), "--title", "X", "--dry-run"]
        )
        edit_metadata.main()
        assert "dry-run" in capsys.readouterr().out
        assert not (tmp_path / "lora.edited.safetensors").exists()

    def test_in_place_overwrites(self, tmp_path, monkeypatch, capsys):
        src = tmp_path / "lora.safetensors"
        write_safetensors(src, {"old": "1"})
        monkeypatch.setattr(
            sys,
            "argv",
            ["edit_metadata.py", str(src), "--set", "note=hi", "--in-place"],
        )
        edit_metadata.main()
        _, header = read_header(str(src))
        assert header["__metadata__"]["note"] == "hi"

    def test_custom_output_path(self, tmp_path, monkeypatch):
        src = tmp_path / "lora.safetensors"
        out_path = tmp_path / "custom.safetensors"
        write_safetensors(src, {})
        monkeypatch.setattr(
            sys,
            "argv",
            ["edit_metadata.py", str(src), "--title", "Y", "-o", str(out_path)],
        )
        edit_metadata.main()
        assert out_path.exists()

    def test_del_present_and_absent_warns(self, tmp_path, monkeypatch, capsys):
        src = tmp_path / "lora.safetensors"
        write_safetensors(src, {"keep": "1", "drop": "2"})
        monkeypatch.setattr(
            sys,
            "argv",
            ["edit_metadata.py", str(src), "--del", "drop", "--del", "ghost"],
        )
        edit_metadata.main()
        err = capsys.readouterr().err
        assert "not present" in err
        _, header = read_header(str(tmp_path / "lora.edited.safetensors"))
        assert "drop" not in header["__metadata__"]

    def test_offspec_value_warns(self, tmp_path, monkeypatch, capsys):
        src = tmp_path / "lora.safetensors"
        write_safetensors(src, {})
        monkeypatch.setattr(
            sys,
            "argv",
            ["edit_metadata.py", str(src), "--prediction-type", "bogus"],
        )
        edit_metadata.main()
        assert "warning" in capsys.readouterr().err

    def test_missing_file_exits(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            sys,
            "argv",
            ["edit_metadata.py", str(tmp_path / "nope.safetensors"), "--title", "X"],
        )
        with pytest.raises(SystemExit):
            edit_metadata.main()

    def test_in_place_and_output_mutually_exclusive(self, tmp_path, monkeypatch):
        src = tmp_path / "lora.safetensors"
        write_safetensors(src, {})
        monkeypatch.setattr(
            sys,
            "argv",
            ["edit_metadata.py", str(src), "--title", "X", "--in-place", "-o", "z.st"],
        )
        with pytest.raises(SystemExit):
            edit_metadata.main()

    def test_nothing_to_do_exits(self, tmp_path, monkeypatch):
        src = tmp_path / "lora.safetensors"
        write_safetensors(src, {})
        monkeypatch.setattr(sys, "argv", ["edit_metadata.py", str(src)])
        with pytest.raises(SystemExit):
            edit_metadata.main()

    def test_corrupt_file_exits(self, tmp_path, monkeypatch):
        bad = tmp_path / "bad.safetensors"
        bad.write_bytes(b"not a real safetensors header at all")
        monkeypatch.setattr(sys, "argv", ["edit_metadata.py", str(bad), "--title", "X"])
        with pytest.raises(SystemExit):
            edit_metadata.main()
