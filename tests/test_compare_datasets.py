"""Tests for tools/compare_datasets.py — pure helpers and the CLI clash gate."""

import random
import shutil
import sys

import compare_datasets
import pytest
from compare_datasets import classify, hamming


def noise_image(path, seed):
    """Deterministic random-noise image, so distinct seeds get far-apart hashes."""
    from PIL import Image

    rng = random.Random(seed)
    img = Image.new("L", (32, 32))
    img.putdata([rng.randrange(256) for _ in range(32 * 32)])
    img.convert("RGB").save(path)
    return str(path)


class TestHamming:
    def test_identical_is_zero(self):
        assert hamming(0b1010, 0b1010) == 0

    def test_counts_differing_bits(self):
        assert hamming(0b1010, 0b0011) == 2

    def test_all_bits_differ(self):
        assert hamming(0b1111, 0b0000) == 4


class TestClassify:
    def test_identical_file_wins(self):
        assert classify(0, identical_file=True, threshold=10) == "IDENTICAL"

    def test_distance_zero_is_identical_image(self):
        assert classify(0, identical_file=False, threshold=10) == "identical-image"

    def test_within_threshold_is_near_dupe(self):
        assert classify(5, identical_file=False, threshold=10) == "near-dupe"

    def test_at_threshold_is_near_dupe(self):
        assert classify(10, identical_file=False, threshold=10) == "near-dupe"

    def test_beyond_threshold_is_distinct(self):
        assert classify(11, identical_file=False, threshold=10) == "distinct"

    @pytest.mark.parametrize(
        "distance,expected",
        [(0, "identical-image"), (1, "near-dupe"), (99, "distinct")],
    )
    def test_spectrum(self, distance, expected):
        assert classify(distance, identical_file=False, threshold=10) == expected


@pytest.fixture
def datasets(tmp_path):
    """A train/ and val/ dir; val holds a byte-identical copy plus a distinct image."""
    train = tmp_path / "train"
    val = tmp_path / "val"
    train.mkdir()
    val.mkdir()
    a = noise_image(train / "a.png", seed=1)
    noise_image(train / "b.png", seed=2)
    shutil.copyfile(a, val / "dup.png")  # byte-identical -> clash
    noise_image(val / "unique.png", seed=999)  # far from all train -> distinct
    return train, val


class TestMainCLI:
    def test_clash_exits_nonzero(self, datasets, monkeypatch, capsys):
        train, val = datasets
        monkeypatch.setattr(sys, "argv", ["compare_datasets.py", str(train), str(val)])
        with pytest.raises(SystemExit) as e:
            compare_datasets.main()
        assert e.value.code == 1
        assert "clash" in capsys.readouterr().out

    def test_show_all_and_ahash(self, datasets, monkeypatch, capsys):
        train, val = datasets
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "compare_datasets.py",
                str(train),
                str(val),
                "--show-all",
                "--hash",
                "ahash",
            ],
        )
        with pytest.raises(SystemExit):
            compare_datasets.main()
        assert "IDENTICAL" in capsys.readouterr().out

    def test_no_clash_returns_cleanly(self, tmp_path, monkeypatch, capsys):
        train = tmp_path / "t"
        val = tmp_path / "v"
        train.mkdir()
        val.mkdir()
        noise_image(train / "a.png", seed=1)
        noise_image(val / "b.png", seed=999)
        monkeypatch.setattr(
            sys,
            "argv",
            ["compare_datasets.py", str(train), str(val), "--threshold", "0"],
        )
        compare_datasets.main()  # no clash -> no SystemExit
        assert "No clashes" in capsys.readouterr().out

    def test_recursive(self, tmp_path, monkeypatch):
        train = tmp_path / "t"
        val = tmp_path / "v"
        (train / "sub").mkdir(parents=True)
        val.mkdir()
        a = noise_image(train / "sub" / "a.png", seed=1)
        shutil.copyfile(a, val / "dup.png")
        monkeypatch.setattr(
            sys, "argv", ["compare_datasets.py", str(train), str(val), "-r"]
        )
        with pytest.raises(SystemExit):
            compare_datasets.main()

    def test_skips_unreadable_image(self, tmp_path, monkeypatch, capsys):
        train = tmp_path / "t"
        val = tmp_path / "v"
        train.mkdir()
        val.mkdir()
        a = noise_image(train / "a.png", seed=1)
        shutil.copyfile(a, val / "dup.png")
        (val / "corrupt.png").write_bytes(b"not an image")
        monkeypatch.setattr(sys, "argv", ["compare_datasets.py", str(train), str(val)])
        with pytest.raises(SystemExit):
            compare_datasets.main()
        assert "skipping" in capsys.readouterr().err

    @pytest.mark.parametrize(
        "extra",
        [[], ["--threshold", "-1"], ["--size", "1"]],
    )
    def test_bad_dirs_and_args_exit(self, tmp_path, monkeypatch, extra):
        # train_dir is a file, not a dir -> error (also covers arg validation).
        f = tmp_path / "notdir.png"
        noise_image(f, seed=1)
        val = tmp_path / "v"
        val.mkdir()
        monkeypatch.setattr(
            sys, "argv", ["compare_datasets.py", str(f), str(val), *extra]
        )
        with pytest.raises(SystemExit):
            compare_datasets.main()

    def test_empty_dir_exits(self, tmp_path, monkeypatch):
        train = tmp_path / "t"
        val = tmp_path / "v"
        train.mkdir()
        val.mkdir()
        noise_image(train / "a.png", seed=1)  # train non-empty, val empty
        monkeypatch.setattr(sys, "argv", ["compare_datasets.py", str(train), str(val)])
        with pytest.raises(SystemExit):
            compare_datasets.main()

    def test_empty_train_dir_exits(self, tmp_path, monkeypatch):
        train = tmp_path / "t"
        val = tmp_path / "v"
        train.mkdir()
        val.mkdir()
        noise_image(val / "a.png", seed=1)  # train empty, val non-empty
        monkeypatch.setattr(sys, "argv", ["compare_datasets.py", str(train), str(val)])
        with pytest.raises(SystemExit) as e:
            compare_datasets.main()
        assert "no images found" in str(e.value.code)

    def test_all_unreadable_exits(self, tmp_path, monkeypatch):
        train = tmp_path / "t"
        val = tmp_path / "v"
        train.mkdir()
        val.mkdir()
        # Files match an image extension but none decode -> nothing to compare.
        (train / "corrupt.png").write_bytes(b"not an image")
        noise_image(val / "a.png", seed=1)
        monkeypatch.setattr(sys, "argv", ["compare_datasets.py", str(train), str(val)])
        with pytest.raises(SystemExit) as e:
            compare_datasets.main()
        assert "no readable images" in str(e.value.code)
