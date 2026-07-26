"""Tests for tools/prepare_images.py — flattening, conversion, resizing and the CLI."""

import sys

import prepare_images
import pytest
from prepare_images import destination_dir, flatten_rgb


def pngs(directory):
    return sorted(p.name for p in directory.glob("*.png"))


def jpgs(directory):
    return sorted(p.name for p in directory.glob("*.jpg"))


class TestFlattenRgb:
    def test_rgb_is_returned_untouched(self):
        from PIL import Image

        im = Image.new("RGB", (4, 4), (1, 2, 3))
        assert flatten_rgb(im) is im

    def test_rgba_composites_over_white(self):
        from PIL import Image

        im = Image.new("RGBA", (4, 4), (0, 0, 0, 0))  # fully transparent
        out = flatten_rgb(im)
        assert out.mode == "RGB"
        assert out.getpixel((0, 0)) == (255, 255, 255)

    def test_la_is_flattened(self):
        from PIL import Image

        out = flatten_rgb(Image.new("LA", (4, 4), (0, 0)))
        assert out.mode == "RGB"
        assert out.getpixel((0, 0)) == (255, 255, 255)

    def test_palette_with_transparency_is_flattened(self):
        from PIL import Image

        im = Image.new("P", (4, 4))
        im.info["transparency"] = 0
        assert flatten_rgb(im).mode == "RGB"

    @pytest.mark.parametrize("mode", ["L", "P", "CMYK"])
    def test_opaque_modes_are_converted(self, mode):
        from PIL import Image

        assert flatten_rgb(Image.new(mode, (4, 4))).mode == "RGB"


class TestConversion:
    def test_converts_and_resizes_in_place(self, tmp_path, make_image):
        make_image(tmp_path / "big.png", size=(2000, 1000))
        from PIL import Image

        Image.new("RGBA", (100, 100), (10, 20, 30, 128)).save(tmp_path / "rgba.png")
        rc = prepare_images.main(
            [
                "--dir",
                str(tmp_path),
                "--format",
                "jpg",
                "--max-edge",
                "512",
                "--in-place",
            ]
        )
        assert rc == 0
        assert jpgs(tmp_path) == ["big.jpg", "rgba.jpg"]
        assert pngs(tmp_path) == []  # originals removed after the format change
        with Image.open(tmp_path / "big.jpg") as im:
            assert max(im.size) <= 512
            assert im.mode == "RGB"

    def test_never_upscales(self, tmp_path, make_image):
        from PIL import Image

        make_image(tmp_path / "small.png", size=(64, 32))
        prepare_images.main(
            ["--dir", str(tmp_path), "--max-edge", "2048", "--in-place"]
        )
        with Image.open(tmp_path / "small.png") as im:
            assert im.size == (64, 32)

    def test_no_flatten_keeps_the_mode(self, tmp_path):
        from PIL import Image

        Image.new("RGBA", (32, 32), (0, 0, 0, 0)).save(tmp_path / "a.png")
        prepare_images.main(["--dir", str(tmp_path), "--no-flatten", "--in-place"])
        with Image.open(tmp_path / "a.png") as im:
            assert im.mode == "RGBA"

    def test_flattens_without_a_format_change(self, tmp_path):
        from PIL import Image

        Image.new("RGBA", (32, 32), (0, 0, 0, 0)).save(tmp_path / "a.png")
        prepare_images.main(["--dir", str(tmp_path), "--in-place"])
        with Image.open(tmp_path / "a.png") as im:
            assert im.mode == "RGB"

    def test_jpeg_forces_rgb_even_with_no_flatten(self, tmp_path):
        from PIL import Image

        Image.new("RGBA", (32, 32), (10, 20, 30, 128)).save(tmp_path / "a.png")
        prepare_images.main(
            ["--dir", str(tmp_path), "--format", "jpg", "--no-flatten", "--in-place"]
        )
        with Image.open(tmp_path / "a.jpg") as im:
            assert im.mode == "RGB"  # JPEG cannot hold an alpha channel

    def test_quality_is_honoured(self, tmp_path, make_image):
        make_image(tmp_path / "a.png", size=(256, 256))
        prepare_images.main(
            ["--dir", str(tmp_path), "--format", "jpg", "--quality", "20", "--in-place"]
        )
        small = (tmp_path / "a.jpg").stat().st_size
        make_image(tmp_path / "b.png", size=(256, 256))
        prepare_images.main(
            ["--dir", str(tmp_path), "--format", "jpg", "--quality", "95", "--in-place"]
        )
        assert small < (tmp_path / "b.jpg").stat().st_size

    def test_exif_orientation_is_applied(self, tmp_path):
        from PIL import Image

        im = Image.new("RGB", (40, 20), (5, 5, 5))
        exif = im.getexif()
        exif[0x0112] = 6  # rotate 90° CW on display
        im.save(tmp_path / "rot.jpg", exif=exif)
        prepare_images.main(["--dir", str(tmp_path), "--in-place"])
        with Image.open(tmp_path / "rot.jpg") as out:
            assert out.size == (20, 40)  # transposed, and the EXIF tag is gone
            assert out.getexif().get(0x0112) is None


class TestDestinationDir:
    def test_defaults_to_a_prepared_sibling(self, tmp_path):
        assert destination_dir(tmp_path / "raw", None, False) == (
            tmp_path / "raw.prepared"
        )

    def test_out_dir_wins_when_not_in_place(self, tmp_path):
        out = tmp_path / "chosen"
        assert destination_dir(tmp_path / "raw", out, False) == out

    @pytest.mark.parametrize("out", [None, "chosen"])
    def test_in_place_always_returns_the_source(self, tmp_path, out):
        src = tmp_path / "raw"
        explicit = tmp_path / out if out else None
        assert destination_dir(src, explicit, True) == src


class TestNonDestructiveDefault:
    def test_writes_to_a_sibling_and_leaves_the_input_alone(
        self, tmp_path, make_image, capsys
    ):
        src = tmp_path / "raw"
        src.mkdir()
        make_image(src / "a.png", size=(64, 64))
        (src / "a.txt").write_text("cap")
        assert prepare_images.main(["--dir", str(src), "--format", "jpg"]) == 0
        out = tmp_path / "raw.prepared"
        assert (out / "a.jpg").exists() and (out / "a.txt").exists()
        assert pngs(src) == ["a.png"]  # original neither rewritten nor deleted
        assert (src / "a.txt").exists()
        assert "writing to:" in capsys.readouterr().out

    def test_dry_run_does_not_create_the_sibling(self, tmp_path, make_image):
        src = tmp_path / "raw"
        src.mkdir()
        make_image(src / "a.png")
        prepare_images.main(["--dir", str(src), "--dry-run"])
        assert not (tmp_path / "raw.prepared").exists()

    def test_same_suffix_does_not_overwrite_the_input(self, tmp_path, make_image):
        src = tmp_path / "raw"
        src.mkdir()
        make_image(src / "a.png", size=(2000, 2000))
        before = (src / "a.png").read_bytes()
        prepare_images.main(["--dir", str(src), "--max-edge", "64"])
        assert (src / "a.png").read_bytes() == before
        assert (tmp_path / "raw.prepared" / "a.png").exists()

    def test_in_place_with_out_dir_exits_2(self, tmp_path, capsys):
        rc = prepare_images.main(
            ["--dir", str(tmp_path), "--out-dir", str(tmp_path / "o"), "--in-place"]
        )
        assert rc == 2
        assert "mutually exclusive" in capsys.readouterr().err

    def test_out_dir_equal_to_the_input_exits_2(self, tmp_path, capsys):
        rc = prepare_images.main(["--dir", str(tmp_path), "--out-dir", str(tmp_path)])
        assert rc == 2
        assert "pass --in-place" in capsys.readouterr().err


class TestOutDir:
    def test_copies_the_caption_and_leaves_input_alone(self, tmp_path, make_image):
        src, out = tmp_path / "src", tmp_path / "out"
        src.mkdir()
        make_image(src / "z.png", size=(64, 64))
        (src / "z.txt").write_text("hi")
        assert (
            prepare_images.main(
                ["--dir", str(src), "--out-dir", str(out), "--format", "png"]
            )
            == 0
        )
        assert (out / "z.png").exists() and (out / "z.txt").exists()
        assert (src / "z.png").exists()

    def test_missing_caption_is_not_fatal(self, tmp_path, make_image):
        src, out = tmp_path / "src", tmp_path / "out"
        src.mkdir()
        make_image(src / "z.png")
        assert prepare_images.main(["--dir", str(src), "--out-dir", str(out)]) == 0
        assert (out / "z.png").exists()
        assert list(out.glob("*.txt")) == []


class TestFailuresAndDryRun:
    def test_corrupt_image_is_counted_and_reported(self, tmp_path, make_image, capsys):
        make_image(tmp_path / "ok.png")
        (tmp_path / "broken.png").write_bytes(b"not a png")
        rc = prepare_images.main(
            ["--dir", str(tmp_path), "--format", "jpg", "--in-place"]
        )
        captured = capsys.readouterr()
        assert rc == 1
        assert "failed: broken.png" in captured.err
        assert "1 failed" in captured.out
        assert (tmp_path / "ok.jpg").exists()  # the good image still went through

    def test_dry_run_writes_nothing(self, tmp_path, make_image, capsys):
        make_image(tmp_path / "a.png", size=(2000, 2000))
        prepare_images.main(
            [
                "--dir",
                str(tmp_path),
                "--format",
                "jpg",
                "--max-edge",
                "64",
                "--dry-run",
                "--in-place",
            ]
        )
        assert jpgs(tmp_path) == []
        assert pngs(tmp_path) == ["a.png"]
        assert "[dry-run]" in capsys.readouterr().out

    def test_dry_run_with_out_dir_does_not_create_it(self, tmp_path, make_image):
        src, out = tmp_path / "src", tmp_path / "out"
        src.mkdir()
        make_image(src / "a.png")
        prepare_images.main(["--dir", str(src), "--out-dir", str(out), "--dry-run"])
        assert not out.exists()


class TestMainCLI:
    def test_reads_sys_argv_by_default(self, tmp_path, make_image, monkeypatch):
        make_image(tmp_path / "a.png")
        monkeypatch.setattr(
            sys, "argv", ["prepare_images.py", "--dir", str(tmp_path), "--in-place"]
        )
        assert prepare_images.main() == 0

    def test_missing_dir_exits_2(self, tmp_path, capsys):
        assert prepare_images.main(["--dir", str(tmp_path / "nope")]) == 2
        assert "not a directory" in capsys.readouterr().err

    def test_negative_max_edge_exits_2(self, tmp_path, capsys):
        rc = prepare_images.main(["--dir", str(tmp_path), "--max-edge", "-1"])
        assert rc == 2
        assert "--max-edge must be >= 0" in capsys.readouterr().err

    def test_empty_dir_is_fine(self, tmp_path, capsys):
        assert prepare_images.main(["--dir", str(tmp_path), "--in-place"]) == 0
        assert "processed 0 image(s)" in capsys.readouterr().out
