"""Tests for tools/lint_dataset.py — the error/warning split and the exit-code gate."""

import sys

import lint_dataset
import pytest
from lint_dataset import report


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
    def test_clean_dir_passes(self, tmp_path, make_image, capsys):
        make_image(tmp_path / "x.png")
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


class TestMainCLI:
    def test_reads_sys_argv_by_default(self, tmp_path, make_image, monkeypatch):
        make_image(tmp_path / "a.png")
        (tmp_path / "a.txt").write_text("ok")
        monkeypatch.setattr(sys, "argv", ["lint_dataset.py", "--dir", str(tmp_path)])
        assert lint_dataset.main() == 0

    def test_missing_dir_exits_2(self, tmp_path, capsys):
        assert lint_dataset.main(["--dir", str(tmp_path / "nope")]) == 2
        assert "not a directory" in capsys.readouterr().err

    def test_custom_caption_ext(self, tmp_path, make_image, capsys):
        make_image(tmp_path / "a.png")
        (tmp_path / "a.caption").write_text("ok")
        rc = lint_dataset.main(["--dir", str(tmp_path), "--caption-ext", "caption"])
        assert rc == 0
        assert "clean" in capsys.readouterr().out

    @pytest.mark.parametrize("exts", [["png"], [".PNG"]])
    def test_image_exts_are_normalized(self, tmp_path, make_image, exts):
        make_image(tmp_path / "a.png")
        (tmp_path / "a.txt").write_text("ok")
        assert lint_dataset.main(["--dir", str(tmp_path), "--image-exts", *exts]) == 0
