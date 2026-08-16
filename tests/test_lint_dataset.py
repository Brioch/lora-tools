"""Tests for tools/lint_dataset.py — the error/warning split and the exit-code gate."""

import random
import sys

import lint_dataset
import pytest
from lint_dataset import grayscale_range, report


def detailed_image(path, size=(256, 256), seed=0):
    """A noise image, which unlike a solid fill has real dynamic range.

    The `make_image` fixture writes a solid colour, and a solid image legitimately
    trips the near-blank check — so tests asserting a clean dataset need this instead.
    """
    from PIL import Image

    rng = random.Random(seed)
    im = Image.new("L", size)
    im.putdata([rng.randrange(256) for _ in range(size[0] * size[1])])
    im.convert("RGB").save(path)
    return path


class TestReport:
    def test_empty_groups_report_nothing(self, capsys):
        assert report("ERRORS", {}) == 0
        assert capsys.readouterr().out == ""

    def test_counts_every_entry(self, capsys):
        total = report("ERRORS", {"a": ["x", "y"], "b": ["z"]})
        out = capsys.readouterr().out
        assert total == 3
        assert "ERRORS (3)" in out
        assert "- x" in out and "- z" in out


class TestGrayscaleRange:
    def test_solid_fill_has_no_range(self):
        from PIL import Image

        assert grayscale_range(Image.new("RGB", (8, 8), (40, 40, 40))) == 0

    def test_black_to_white_spans_everything(self):
        from PIL import Image

        im = Image.new("L", (2, 1))
        im.putdata([0, 255])
        assert grayscale_range(im) == 255

    def test_measures_the_span_of_levels_present(self):
        from PIL import Image

        im = Image.new("L", (3, 1))
        im.putdata([100, 110, 105])
        assert grayscale_range(im) == 10

    def test_transparent_rgba_flattens_to_one_level(self):
        from PIL import Image

        assert grayscale_range(Image.new("RGBA", (8, 8), (0, 0, 0, 0))) == 0


class TestErrors:
    def test_missing_caption_is_an_error(self, tmp_path, make_image, capsys):
        make_image(tmp_path / "a.png")
        assert lint_dataset.main(["--dir", str(tmp_path)]) == 1
        assert "missing caption" in capsys.readouterr().out

    def test_empty_caption_is_an_error(self, tmp_path, make_image, capsys):
        make_image(tmp_path / "a.png")
        (tmp_path / "a.txt").write_text("   \n")
        assert lint_dataset.main(["--dir", str(tmp_path)]) == 1
        assert "empty caption" in capsys.readouterr().out

    def test_corrupt_image_is_an_error(self, tmp_path, capsys):
        (tmp_path / "broken.png").write_bytes(b"not a png")
        (tmp_path / "broken.txt").write_text("cap")
        assert lint_dataset.main(["--dir", str(tmp_path)]) == 1
        assert "corrupt image" in capsys.readouterr().out

    def test_reports_every_category_at_once(self, tmp_path, make_image, capsys):
        make_image(tmp_path / "a.png", size=(256, 256))
        (tmp_path / "a.txt").write_text("ok")
        make_image(tmp_path / "b.png", size=(256, 256))
        (tmp_path / "b.txt").write_text("  ")
        make_image(tmp_path / "c.png", size=(256, 256))  # no caption
        (tmp_path / "orphan.txt").write_text("no image")
        (tmp_path / "broken.png").write_bytes(b"not a png")
        rc = lint_dataset.main(["--dir", str(tmp_path), "--min-size", "512"])
        out = capsys.readouterr().out
        assert rc == 1
        for token in (
            "empty caption",
            "missing caption",
            "corrupt image",
            "orphan caption",
            "below min-size",
        ):
            assert token in out


class TestWarnings:
    def test_clean_dir_passes(self, tmp_path, capsys):
        detailed_image(tmp_path / "x.png")
        (tmp_path / "x.txt").write_text("ok")
        assert lint_dataset.main(["--dir", str(tmp_path)]) == 0
        assert "clean: no issues found" in capsys.readouterr().out

    def test_non_rgb_mode_only_warns(self, tmp_path, capsys):
        from PIL import Image

        Image.new("RGBA", (64, 64)).save(tmp_path / "a.png")
        (tmp_path / "a.txt").write_text("ok")
        assert lint_dataset.main(["--dir", str(tmp_path)]) == 0
        assert "non-RGB mode" in capsys.readouterr().out

    def test_below_min_size_only_warns(self, tmp_path, make_image, capsys):
        make_image(tmp_path / "a.png", size=(64, 64))
        (tmp_path / "a.txt").write_text("ok")
        assert lint_dataset.main(["--dir", str(tmp_path), "--min-size", "512"]) == 0
        assert "below min-size" in capsys.readouterr().out

    def test_extension_mismatch_only_warns(self, tmp_path, make_image, capsys):
        # JPEG bytes behind a .png suffix.
        make_image(tmp_path / "a.png", fmt="JPEG")
        (tmp_path / "a.txt").write_text("ok")
        assert lint_dataset.main(["--dir", str(tmp_path)]) == 0
        assert "extension mismatch" in capsys.readouterr().out

    def test_orphan_caption_only_warns(self, tmp_path, capsys):
        (tmp_path / "ghost.txt").write_text("no image")
        assert lint_dataset.main(["--dir", str(tmp_path)]) == 0
        assert "orphan caption" in capsys.readouterr().out

    def test_no_require_caption_downgrades_to_warning(
        self, tmp_path, make_image, capsys
    ):
        make_image(tmp_path / "a.png")
        rc = lint_dataset.main(["--dir", str(tmp_path), "--no-require-caption"])
        out = capsys.readouterr().out
        assert rc == 0
        assert "WARNINGS" in out and "missing caption" in out
        assert "ERRORS" not in out


class TestTriggerAudit:
    def _dataset(self, tmp_path, make_image, captions):
        for i, text in enumerate(captions):
            make_image(
                tmp_path / f"{i}.png", size=(32, 32), color=(i * 40 + 10, 90, 200)
            )
            (tmp_path / f"{i}.txt").write_text(text)
        return tmp_path

    def test_missing_trigger_warns(self, tmp_path, make_image, capsys):
        self._dataset(tmp_path, make_image, ["mychar, smiling", "smiling"])
        rc = lint_dataset.main(["--dir", str(tmp_path), "--trigger", "mychar"])
        out = capsys.readouterr().out
        assert rc == 0  # a warning, not an error
        assert "trigger missing" in out
        assert "1.png" in out

    def test_present_trigger_is_clean(self, tmp_path, make_image, capsys):
        self._dataset(tmp_path, make_image, ["mychar, a", "mychar, b"])
        lint_dataset.main(["--dir", str(tmp_path), "--trigger", "mychar"])
        assert "trigger missing" not in capsys.readouterr().out

    def test_matching_is_case_insensitive(self, tmp_path, make_image, capsys):
        self._dataset(tmp_path, make_image, ["MyChar, a"])
        lint_dataset.main(["--dir", str(tmp_path), "--trigger", "MYCHAR"])
        assert "trigger missing" not in capsys.readouterr().out

    def test_matches_whole_tags_only(self, tmp_path, make_image, capsys):
        """'mychar' inside 'mycharacter' is a different tag and must not count."""
        self._dataset(tmp_path, make_image, ["mycharacter, a"])
        lint_dataset.main(["--dir", str(tmp_path), "--trigger", "mychar"])
        assert "trigger missing" in capsys.readouterr().out

    def test_no_trigger_flag_skips_the_check(self, tmp_path, make_image, capsys):
        self._dataset(tmp_path, make_image, ["nothing here"])
        lint_dataset.main(["--dir", str(tmp_path)])
        assert "trigger missing" not in capsys.readouterr().out

    def test_empty_caption_reports_only_the_error(self, tmp_path, make_image, capsys):
        self._dataset(tmp_path, make_image, ["  "])
        rc = lint_dataset.main(["--dir", str(tmp_path), "--trigger", "mychar"])
        out = capsys.readouterr().out
        assert rc == 1
        assert "empty caption" in out
        assert "trigger missing" not in out  # not double-reported


class TestNearBlankImages:
    def test_solid_image_warns(self, tmp_path, make_image, capsys):
        make_image(tmp_path / "flat.png", color=(128, 128, 128))
        (tmp_path / "flat.txt").write_text("ok")
        rc = lint_dataset.main(["--dir", str(tmp_path)])
        out = capsys.readouterr().out
        assert rc == 0
        assert "near-blank image" in out
        assert "range 0" in out

    def test_detailed_image_is_clean(self, tmp_path, capsys):
        import random

        from PIL import Image

        rng = random.Random(0)
        im = Image.new("L", (32, 32))
        im.putdata([rng.randrange(256) for _ in range(32 * 32)])
        im.convert("RGB").save(tmp_path / "noise.png")
        (tmp_path / "noise.txt").write_text("ok")
        assert lint_dataset.main(["--dir", str(tmp_path)]) == 0
        assert "near-blank" not in capsys.readouterr().out

    def test_check_can_be_disabled(self, tmp_path, make_image, capsys):
        make_image(tmp_path / "flat.png", color=(10, 10, 10))
        (tmp_path / "flat.txt").write_text("ok")
        lint_dataset.main(["--dir", str(tmp_path), "--min-contrast", "0"])
        assert "near-blank" not in capsys.readouterr().out

    def test_threshold_is_tunable(self, tmp_path, capsys):
        from PIL import Image

        im = Image.new("L", (4, 1))
        im.putdata([100, 100, 100, 120])  # range 20
        im.convert("RGB").save(tmp_path / "low.png")
        (tmp_path / "low.txt").write_text("ok")
        lint_dataset.main(["--dir", str(tmp_path), "--min-contrast", "8"])
        assert "near-blank" not in capsys.readouterr().out
        lint_dataset.main(["--dir", str(tmp_path), "--min-contrast", "30"])
        assert "near-blank" in capsys.readouterr().out


class TestMainCLI:
    def test_reads_sys_argv_by_default(self, tmp_path, make_image, monkeypatch):
        make_image(tmp_path / "a.png")
        (tmp_path / "a.txt").write_text("ok")
        monkeypatch.setattr(sys, "argv", ["lint_dataset.py", "--dir", str(tmp_path)])
        assert lint_dataset.main() == 0

    def test_missing_dir_exits_2(self, tmp_path, capsys):
        assert lint_dataset.main(["--dir", str(tmp_path / "nope")]) == 2
        assert "not a directory" in capsys.readouterr().err

    def test_custom_caption_ext(self, tmp_path, capsys):
        detailed_image(tmp_path / "a.png")
        (tmp_path / "a.caption").write_text("ok")
        rc = lint_dataset.main(["--dir", str(tmp_path), "--caption-ext", "caption"])
        assert rc == 0
        assert "clean" in capsys.readouterr().out

    @pytest.mark.parametrize("exts", [["png"], [".PNG"]])
    def test_image_exts_are_normalized(self, tmp_path, make_image, exts):
        make_image(tmp_path / "a.png")
        (tmp_path / "a.txt").write_text("ok")
        assert lint_dataset.main(["--dir", str(tmp_path), "--image-exts", *exts]) == 0
