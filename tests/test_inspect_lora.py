"""Tests for tools/inspect_lora.py — pure helpers and the CLI."""

import json
import sys

import inspect_lora
import pytest
from inspect_lora import (
    _json_or_none,
    describe_metadata,
    detect_convention,
    detect_math,
    render_meta_value,
    shape_of,
)


class TestDetectConvention:
    def test_comfy(self):
        conventions, primary = detect_convention(["diffusion_model.blocks.0.lora_A"])
        assert primary == "comfy"
        assert conventions == {"comfy"}

    def test_diffusers(self):
        _, primary = detect_convention(["transformer.blocks.0.lora_A"])
        assert primary == "diffusers"

    def test_kohya(self):
        _, primary = detect_convention(["lora_unet_down_blocks_0"])
        assert primary == "kohya"

    def test_unknown_is_original(self):
        _, primary = detect_convention(["some.random.key"])
        assert primary == "original"

    def test_mixed_reports_all_and_picks_majority(self):
        keys = ["diffusion_model.a", "diffusion_model.b", "transformer.c"]
        conventions, primary = detect_convention(keys)
        assert conventions == {"comfy", "diffusers"}
        assert primary == "comfy"


class TestDetectMath:
    def test_peft(self):
        assert detect_math({"lora_A", "lora_B"}) == ["PEFT / diffusers (lora_A/lora_B)"]

    def test_kohya(self):
        assert detect_math({"lora_down", "lora_up"}) == [
            "kohya LoRA (lora_down/lora_up)"
        ]

    def test_none_is_unknown(self):
        assert detect_math(set()) == ["(unknown)"]

    def test_multiple_labels(self):
        labels = detect_math({"lora_A", "hada_w1_a"})
        assert "PEFT / diffusers (lora_A/lora_B)" in labels
        assert "LoHa (LyCORIS)" in labels


class TestShapeOf:
    def test_collapses_indices_and_drops_suffix(self):
        assert shape_of("diffusion_model.blocks.12.lora_A.weight") == (
            "diffusion_model.blocks.N"
        )

    def test_underscore_index(self):
        assert shape_of("lora_unet_blocks_7_attn") == "lora_unet_blocks_N_attn"


class TestJsonOrNone:
    def test_valid_json(self):
        assert _json_or_none('{"a": 1}') == {"a": 1}

    def test_invalid_json(self):
        assert _json_or_none("not json") is None

    def test_non_string(self):
        assert _json_or_none(None) is None


class TestRenderMetaValue:
    def test_compacts_json_blob(self):
        assert (
            render_meta_value('{"name": "x", "v": 2}', truncate=True)
            == '{"name":"x","v":2}'
        )

    def test_truncates_long_values(self):
        rendered = render_meta_value("a" * 200, truncate=True)
        assert len(rendered) == 160 and rendered.endswith("...")

    def test_no_truncate(self):
        assert render_meta_value("a" * 200, truncate=False) == "a" * 200


class TestDescribeMetadata:
    def test_empty(self):
        assert describe_metadata({}) == {}

    def test_pulls_all_hints(self):
        meta = {
            "software": json.dumps({"name": "OneTrainer", "version": "1.2"}),
            "ss_network_module": "networks.lora",
            "ss_network_dim": "16",
            "ss_network_alpha": "8",
            "modelspec.implementation": "comfy",
            "modelspec.architecture": "krea2/lora",
        }
        hints = describe_metadata(meta)
        assert hints["trainer"] == "OneTrainer 1.2"
        assert "ss_network_module" in hints["ss"]
        assert hints["implementation"] == "comfy"
        assert "krea2/lora" in hints["base model"] and "dim 16" in hints["base model"]


TENSOR = {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}


class TestMainCLI:
    def _crafted(self, make_safetensors, tmp_path):
        meta = {
            "software": json.dumps({"name": "OneTrainer", "version": "1.2"}),
            "ss_network_module": "networks.lora",
            "modelspec.architecture": "krea2/lora",
        }
        extra = {
            "diffusion_model.blocks.0.lora_A.weight": TENSOR,
            "diffusion_model.blocks.1.lora_B.weight": TENSOR,
            "diffusion_model.layerwise_blocks.0.lora_A.weight": TENSOR,
            "diffusion_model.refiner_blocks.0.lora_A.weight": TENSOR,
        }
        return make_safetensors(
            tmp_path / "lora.safetensors", metadata=meta, header_extra=extra
        )

    def test_full_report(self, make_safetensors, tmp_path, monkeypatch, capsys):
        path = self._crafted(make_safetensors, tmp_path)
        monkeypatch.setattr(sys, "argv", ["inspect_lora.py", path])
        inspect_lora.main()
        out = capsys.readouterr().out
        assert "convention" in out and "comfy" in out
        assert "PEFT" in out
        assert "OneTrainer" in out
        assert "structural key shapes" in out
        assert "main blocks.N" in out

    def test_grep_and_raw(self, make_safetensors, tmp_path, monkeypatch, capsys):
        path = self._crafted(make_safetensors, tmp_path)
        monkeypatch.setattr(
            sys, "argv", ["inspect_lora.py", path, "--grep", "refiner", "--raw"]
        )
        inspect_lora.main()
        assert "refiner_blocks" in capsys.readouterr().out

    def test_missing_file_exits(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            sys, "argv", ["inspect_lora.py", str(tmp_path / "nope.safetensors")]
        )
        with pytest.raises(SystemExit):
            inspect_lora.main()
