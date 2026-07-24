"""Tests for the pure classification helpers in tools/inspect_lora.py."""

from inspect_lora import (
    _json_or_none,
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
        keys = [
            "diffusion_model.a",
            "diffusion_model.b",
            "transformer.c",
        ]
        conventions, primary = detect_convention(keys)
        assert conventions == {"comfy", "diffusers"}
        assert primary == "comfy"  # two comfy vs one diffusers


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
