"""Tests for tools/dedupe_images.py — exact/near grouping and the delete gate."""

import random
import shutil
import sys

import dedupe_images
import pytest
from dedupe_images import choose_keeper, group_exact, group_near, pixels, sha256


def noise_image(path, seed, size=(32, 32)):
    """Deterministic random-noise image, so distinct seeds get far-apart hashes."""
    from PIL import Image

    rng = random.Random(seed)
    img = Image.new("L", size)
    img.putdata([rng.randrange(256) for _ in range(size[0] * size[1])])
    img.convert("RGB").save(path)
    return path


def gradient_image(path, size=(64, 64)):
    """A smooth horizontal ramp — its dHash survives a resize, so copies group."""
    from PIL import Image

    w, h = size
    img = Image.new("L", size)
    img.putdata([x * 255 // w for _ in range(h) for x in range(w)])
    img.convert("RGB").save(path)
    return path


def pngs(directory):
    return sorted(p.name for p in directory.glob("*.png"))


class TestSha256:
    def test_identical_bytes_match(self, tmp_path):
        a = tmp_path / "a.bin"
        b = tmp_path / "b.bin"
        a.write_bytes(b"hello")
        b.write_bytes(b"hello")
        assert sha256(a) == sha256(b)

    def test_different_bytes_differ(self, tmp_path):
        a = tmp_path / "a.bin"
        b = tmp_path / "b.bin"
        a.write_bytes(b"hello")
        b.write_bytes(b"world")
        assert sha256(a) != sha256(b)


class TestPixels:
    def test_counts_width_times_height(self, tmp_path, make_image):
        make_image(tmp_path / "a.png", size=(20, 10))
        assert pixels(tmp_path / "a.png") == 200


class TestChooseKeeper:
    def test_largest_wins_by_default(self, tmp_path, make_image):
        big = tmp_path / "big.png"
        small = tmp_path / "small.png"
        make_image(big, size=(64, 64))
        make_image(small, size=(16, 16))
        assert choose_keeper([small, big], "largest") == big

    def test_first_picks_the_lowest_name(self, tmp_path, make_image):
        a = tmp_path / "a.png"
        z = tmp_path / "z.png"
        make_image(a, size=(16, 16))
        make_image(z, size=(64, 64))
        assert choose_keeper([z, a], "first") == a

    def test_largest_ties_break_by_name(self, tmp_path, make_image):
        a = tmp_path / "a.png"
        b = tmp_path / "b.png"
        make_image(a, size=(32, 32))
        make_image(b, size=(32, 32))
        assert choose_keeper([b, a], "largest") == a


class TestGrouping:
    def test_group_exact_finds_byte_identical_files(self, tmp_path):
        a = noise_image(tmp_path / "a.png", seed=1)
        shutil.copyfile(a, tmp_path / "copy.png")
        noise_image(tmp_path / "other.png", seed=999)
        groups = group_exact(sorted(tmp_path.glob("*.png")))
        assert len(groups) == 1
        assert sorted(p.name for p in groups[0]) == ["a.png", "copy.png"]

    def test_group_exact_ignores_singletons(self, tmp_path):
        noise_image(tmp_path / "a.png", seed=1)
        noise_image(tmp_path / "b.png", seed=2)
        assert group_exact(sorted(tmp_path.glob("*.png"))) == []

    def test_group_near_matches_a_resized_copy(self, tmp_path):
        gradient_image(tmp_path / "big.png", size=(64, 64))
        gradient_image(tmp_path / "small.png", size=(32, 32))
        noise_image(tmp_path / "other.png", seed=7)
        groups = group_near(sorted(tmp_path.glob("*.png")), threshold=5)
        assert len(groups) == 1
        assert sorted(p.name for p in groups[0]) == ["big.png", "small.png"]

    def test_group_near_threshold_zero_is_strict(self, tmp_path):
        noise_image(tmp_path / "a.png", seed=1)
        noise_image(tmp_path / "b.png", seed=2)
        assert group_near(sorted(tmp_path.glob("*.png")), threshold=0) == []

    def test_group_near_skips_unreadable(self, tmp_path, capsys):
        noise_image(tmp_path / "a.png", seed=1)
        (tmp_path / "broken.png").write_bytes(b"not a png")
        group_near(sorted(tmp_path.glob("*.png")), threshold=5)
        assert "skip (unreadable): broken.png" in capsys.readouterr().err


class TestMainCLI:
    def test_removes_near_duplicate_and_its_caption(self, tmp_path):
        gradient_image(tmp_path / "orig.png", size=(64, 64))
        gradient_image(tmp_path / "orig_copy.png", size=(32, 32))
        (tmp_path / "orig.txt").write_text("c")
        (tmp_path / "orig_copy.txt").write_text("c")
        noise_image(tmp_path / "unique.png", seed=999)
        rc = dedupe_images.main(["--dir", str(tmp_path), "--threshold", "5", "-y"])
        remaining = pngs(tmp_path)
        assert rc == 0
        assert remaining == ["orig.png", "unique.png"]  # the 64px original is kept
        assert [p.name for p in tmp_path.glob("*.txt")] == ["orig.txt"]

    def test_exact_only_skips_the_perceptual_pass(self, tmp_path):
        # Two gradients that are near-duplicates but not byte-identical.
        gradient_image(tmp_path / "a.png", size=(64, 64))
        gradient_image(tmp_path / "b.png", size=(32, 32))
        rc = dedupe_images.main(["--dir", str(tmp_path), "--exact-only", "-y"])
        assert rc == 0
        assert pngs(tmp_path) == ["a.png", "b.png"]

    def test_exact_only_removes_byte_identical(self, tmp_path):
        a = noise_image(tmp_path / "a.png", seed=1)
        shutil.copyfile(a, tmp_path / "z.png")
        dedupe_images.main(["--dir", str(tmp_path), "--exact-only", "-y"])
        assert pngs(tmp_path) == ["a.png"]

    def test_keep_first_overrides_resolution(self, tmp_path):
        gradient_image(tmp_path / "aaa.png", size=(32, 32))
        gradient_image(tmp_path / "zzz.png", size=(64, 64))
        dedupe_images.main(["--dir", str(tmp_path), "--keep", "first", "-y"])
        assert pngs(tmp_path) == ["aaa.png"]

    def test_dry_run_keeps_everything(self, tmp_path, capsys):
        a = noise_image(tmp_path / "a.png", seed=5)
        shutil.copyfile(a, tmp_path / "b.png")
        rc = dedupe_images.main(["--dir", str(tmp_path), "--exact-only", "--dry-run"])
        assert rc == 0
        assert len(pngs(tmp_path)) == 2
        assert "(dry-run; nothing deleted)" in capsys.readouterr().out

    def test_prompt_accepted_deletes(self, tmp_path, monkeypatch):
        a = noise_image(tmp_path / "a.png", seed=5)
        shutil.copyfile(a, tmp_path / "z.png")
        monkeypatch.setattr("builtins.input", lambda _: "yes")
        assert dedupe_images.main(["--dir", str(tmp_path), "--exact-only"]) == 0
        assert pngs(tmp_path) == ["a.png"]

    def test_prompt_declined_aborts(self, tmp_path, monkeypatch, capsys):
        a = noise_image(tmp_path / "a.png", seed=5)
        shutil.copyfile(a, tmp_path / "z.png")
        monkeypatch.setattr("builtins.input", lambda _: "")
        rc = dedupe_images.main(["--dir", str(tmp_path), "--exact-only"])
        assert rc == 1
        assert "aborted; nothing deleted" in capsys.readouterr().out
        assert len(pngs(tmp_path)) == 2

    def test_no_duplicates_reports_cleanly(self, tmp_path, capsys):
        noise_image(tmp_path / "a.png", seed=1)
        noise_image(tmp_path / "b.png", seed=2)
        assert dedupe_images.main(["--dir", str(tmp_path), "--threshold", "0"]) == 0
        assert "no duplicates to remove" in capsys.readouterr().out

    def test_reads_sys_argv_by_default(self, tmp_path, monkeypatch):
        noise_image(tmp_path / "a.png", seed=1)
        monkeypatch.setattr(
            sys,
            "argv",
            ["dedupe_images.py", "--dir", str(tmp_path), "--threshold", "0"],
        )
        assert dedupe_images.main() == 0

    def test_missing_dir_exits_2(self, tmp_path, capsys):
        assert dedupe_images.main(["--dir", str(tmp_path / "nope")]) == 2
        assert "not a directory" in capsys.readouterr().err

    @pytest.mark.parametrize("threshold", ["-1", "-99"])
    def test_negative_threshold_exits_2(self, tmp_path, threshold, capsys):
        rc = dedupe_images.main(["--dir", str(tmp_path), "--threshold", threshold])
        assert rc == 2
        assert "--threshold must be >= 0" in capsys.readouterr().err
