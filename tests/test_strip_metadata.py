"""Tests for tools/strip_metadata.py — metadata removal and CLI."""

import sys

import pytest
import strip_metadata
from strip_metadata import strip


def png_with_text(path, key="workflow", value='{"nodes": []}'):
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    meta = PngInfo()
    meta.add_text(key, value)
    Image.new("RGB", (16, 16), (1, 2, 3)).save(path, pnginfo=meta)
    return str(path)


def has_text(path, key="workflow"):
    from PIL import Image

    with Image.open(path) as img:
        info = getattr(img, "text", None) or img.info
        return key in info


class TestStripFunction:
    def test_returns_pixels_without_info(self):
        from PIL import Image

        img = Image.new("RGB", (8, 8), (5, 6, 7))
        img.info["workflow"] = "x"
        clean = strip(img)
        assert clean.size == img.size
        assert "workflow" not in clean.info

    def test_preserves_palette_for_p_mode(self):
        from PIL import Image

        p = Image.new("P", (8, 8))
        p.putpalette([0, 0, 0, 255, 255, 255])
        clean = strip(p)
        assert clean.mode == "P"
        assert clean.getpalette() is not None


class TestMainCLI:
    def test_strips_to_new_file(self, tmp_path, monkeypatch, capsys):
        img = png_with_text(tmp_path / "preview.png")
        assert has_text(img)
        monkeypatch.setattr(sys, "argv", ["strip_metadata.py", img])
        strip_metadata.main()
        out = tmp_path / "preview.stripped.png"
        assert out.exists() and not has_text(str(out))
        assert "stripped metadata" in capsys.readouterr().out

    def test_in_place(self, tmp_path, monkeypatch):
        img = png_with_text(tmp_path / "preview.png")
        monkeypatch.setattr(sys, "argv", ["strip_metadata.py", img, "--in-place"])
        strip_metadata.main()
        assert not has_text(img)

    def test_output_path_changes_format(self, tmp_path, monkeypatch):
        img = png_with_text(tmp_path / "shot.png")
        out = tmp_path / "clean.webp"
        monkeypatch.setattr(sys, "argv", ["strip_metadata.py", img, "-o", str(out)])
        strip_metadata.main()
        assert out.exists() and not has_text(str(out))

    def test_jpeg_output_uses_quality(self, tmp_path, monkeypatch):
        img = png_with_text(tmp_path / "shot.png")
        out = tmp_path / "clean.jpg"
        # JPEG output takes the quality save path (RGB input, no metadata carried).
        monkeypatch.setattr(
            sys, "argv", ["strip_metadata.py", img, "-o", str(out), "--quality", "70"]
        )
        strip_metadata.main()
        assert out.exists() and not has_text(str(out))

    def test_missing_file_exits(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            sys, "argv", ["strip_metadata.py", str(tmp_path / "nope.png")]
        )
        with pytest.raises(SystemExit):
            strip_metadata.main()

    def test_in_place_and_output_mutually_exclusive(self, tmp_path, monkeypatch):
        img = png_with_text(tmp_path / "preview.png")
        monkeypatch.setattr(
            sys, "argv", ["strip_metadata.py", img, "--in-place", "-o", "x.png"]
        )
        with pytest.raises(SystemExit):
            strip_metadata.main()
