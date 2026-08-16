"""Tests for tools/split_dataset.py — cluster-aware holdout and the leak guarantee."""

import random
import sys

import pytest
import split_dataset
from split_dataset import choose_clusters


def noise_image(path, seed, size=(32, 32)):
    """Deterministic random-noise image, so distinct seeds get far-apart hashes."""
    from PIL import Image

    rng = random.Random(seed)
    img = Image.new("L", size)
    img.putdata([rng.randrange(256) for _ in range(size[0] * size[1])])
    img.convert("RGB").save(path)
    return path


def gradient_image(path, size=(64, 64)):
    """A smooth horizontal ramp — its dHash survives a resize, so copies cluster."""
    from PIL import Image

    w, h = size
    img = Image.new("L", size)
    img.putdata([x * 255 // w for _ in range(h) for x in range(w)])
    img.convert("RGB").save(path)
    return path


def make_dataset(directory, n, captions=True):
    """*n* mutually distinct noise images, optionally with paired captions."""
    directory.mkdir(parents=True, exist_ok=True)
    for i in range(n):
        noise_image(directory / f"img{i:02d}.png", seed=100 + i)
        if captions:
            (directory / f"img{i:02d}.txt").write_text(f"caption {i}")
    return directory


def pngs(directory):
    return sorted(p.name for p in directory.glob("*.png"))


class TestChooseClusters:
    def test_picks_singletons_up_to_the_target(self):
        clusters = [[f"a{i}"] for i in range(10)]
        chosen = choose_clusters(clusters, 3, random.Random(0))
        assert sum(len(c) for c in chosen) == 3

    def test_prefers_clusters_that_fit_the_budget(self):
        clusters = [["big1", "big2", "big3"], ["s1"], ["s2"]]
        chosen = choose_clusters(clusters, 2, random.Random(0))
        assert sorted(c[0] for c in chosen) == ["s1", "s2"]

    def test_takes_one_oversized_cluster_rather_than_nothing(self):
        clusters = [["a", "b", "c"], ["d", "e", "f"]]
        chosen = choose_clusters(clusters, 2, random.Random(0))
        assert sum(len(c) for c in chosen) == 3  # overshoots rather than holding out 0

    def test_refuses_when_everything_is_one_cluster(self):
        assert choose_clusters([["a", "b", "c"]], 1, random.Random(0)) == []

    def test_is_reproducible_for_a_seed(self):
        clusters = [[f"a{i}"] for i in range(10)]
        first = choose_clusters(clusters, 4, random.Random(7))
        second = choose_clusters(clusters, 4, random.Random(7))
        assert first == second

    def test_different_seeds_can_differ(self):
        clusters = [[f"a{i}"] for i in range(20)]
        picks = {
            tuple(sorted(c[0] for c in choose_clusters(clusters, 5, random.Random(s))))
            for s in range(5)
        }
        assert len(picks) > 1


class TestSplitting:
    def test_holds_out_the_requested_fraction(self, tmp_path):
        train = make_dataset(tmp_path / "train", 10)
        rc = split_dataset.main(
            ["--dir", str(train), "--fraction", "0.2", "--seed", "0", "-y"]
        )
        val = tmp_path / "train.val"
        assert rc == 0
        assert len(pngs(val)) == 2
        assert len(pngs(train)) == 8  # moved out, not copied

    def test_count_overrides_fraction(self, tmp_path):
        train = make_dataset(tmp_path / "train", 10)
        split_dataset.main(["--dir", str(train), "--count", "3", "--seed", "0", "-y"])
        assert len(pngs(tmp_path / "train.val")) == 3

    def test_captions_follow_their_images(self, tmp_path):
        train = make_dataset(tmp_path / "train", 6)
        split_dataset.main(["--dir", str(train), "--count", "2", "--seed", "0", "-y"])
        val = tmp_path / "train.val"
        assert [p[:-4] for p in pngs(val)] == sorted(p.stem for p in val.glob("*.txt"))
        assert len(list(train.glob("*.txt"))) == 4

    def test_custom_val_dir(self, tmp_path):
        train = make_dataset(tmp_path / "train", 6)
        out = tmp_path / "elsewhere"
        split_dataset.main(
            ["--dir", str(train), "--val-dir", str(out), "--count", "2", "-y"]
        )
        assert len(pngs(out)) == 2

    def test_copy_leaves_the_training_set_intact(self, tmp_path):
        train = make_dataset(tmp_path / "train", 6)
        rc = split_dataset.main(["--dir", str(train), "--count", "2", "--copy"])
        assert rc == 0
        assert len(pngs(train)) == 6
        assert len(pngs(tmp_path / "train.val")) == 2

    def test_is_reproducible_for_a_seed(self, tmp_path):
        picks = []
        for run in range(2):
            train = make_dataset(tmp_path / f"t{run}", 10)
            split_dataset.main(
                ["--dir", str(train), "--count", "3", "--seed", "42", "--copy"]
            )
            picks.append(pngs(tmp_path / f"t{run}.val"))
        assert picks[0] == picks[1]

    def test_missing_captions_warn(self, tmp_path, capsys):
        train = make_dataset(tmp_path / "train", 4, captions=False)
        split_dataset.main(["--dir", str(train), "--count", "1", "-y"])
        assert "no paired" in capsys.readouterr().err


class TestNoLeakage:
    def test_near_duplicates_stay_on_one_side(self, tmp_path, capsys):
        """The whole point: a frame and its twin must never straddle the split."""
        train = tmp_path / "train"
        train.mkdir()
        gradient_image(train / "twin_a.png", size=(64, 64))
        gradient_image(train / "twin_b.png", size=(32, 32))  # near-duplicate of twin_a
        for i in range(6):
            noise_image(train / f"solo{i}.png", seed=200 + i)

        split_dataset.main(["--dir", str(train), "--count", "2", "--seed", "0", "-y"])
        val_names = set(pngs(tmp_path / "train.val"))
        train_names = set(pngs(train))
        twins = {"twin_a.png", "twin_b.png"}
        # Either both twins moved or neither did — never one of each.
        assert twins <= val_names or twins <= train_names
        assert "near-duplicate cluster(s) kept together" in capsys.readouterr().out

    def test_cluster_granularity_is_reported(self, tmp_path, capsys):
        train = tmp_path / "train"
        train.mkdir()
        # Two clusters of three: no combination adds up to the requested 2.
        gradient_image(train / "a1.png", size=(64, 64))
        gradient_image(train / "a2.png", size=(48, 48))
        gradient_image(train / "a3.png", size=(32, 32))
        for i in range(3):
            noise_image(train / f"n{i}.png", seed=300)  # identical noise -> one cluster
        split_dataset.main(["--dir", str(train), "--count", "2", "--seed", "0", "-y"])
        assert "cluster granularity gave" in capsys.readouterr().err

    def test_refuses_when_no_leak_free_split_exists(self, tmp_path, capsys):
        train = tmp_path / "train"
        train.mkdir()
        for i in range(4):
            gradient_image(train / f"same{i}.png", size=(64 - i * 8, 64 - i * 8))
        rc = split_dataset.main(["--dir", str(train), "--count", "1", "-y"])
        assert rc == 2
        assert "no leak-free split exists" in capsys.readouterr().err

    def test_threshold_zero_splits_finely(self, tmp_path):
        train = make_dataset(tmp_path / "train", 8)
        rc = split_dataset.main(
            ["--dir", str(train), "--count", "2", "--threshold", "0", "-y"]
        )
        assert rc == 0
        assert len(pngs(tmp_path / "train.val")) == 2

    def test_unreadable_images_are_skipped(self, tmp_path, capsys):
        train = make_dataset(tmp_path / "train", 4)
        (train / "broken.png").write_bytes(b"not a png")
        split_dataset.main(["--dir", str(train), "--count", "1", "-y"])
        assert "skip (unreadable): broken.png" in capsys.readouterr().err


class TestPromptAndDryRun:
    def test_dry_run_moves_nothing(self, tmp_path, capsys):
        train = make_dataset(tmp_path / "train", 6)
        rc = split_dataset.main(["--dir", str(train), "--count", "2", "--dry-run"])
        assert rc == 0
        assert len(pngs(train)) == 6
        assert not (tmp_path / "train.val").exists()
        assert "(dry-run; nothing written)" in capsys.readouterr().out

    def test_prompt_accepted_moves(self, tmp_path, monkeypatch):
        train = make_dataset(tmp_path / "train", 6)
        monkeypatch.setattr("builtins.input", lambda _: "y")
        assert split_dataset.main(["--dir", str(train), "--count", "2"]) == 0
        assert len(pngs(train)) == 4

    def test_prompt_declined_aborts(self, tmp_path, monkeypatch, capsys):
        train = make_dataset(tmp_path / "train", 6)
        monkeypatch.setattr("builtins.input", lambda _: "n")
        rc = split_dataset.main(["--dir", str(train), "--count", "2"])
        assert rc == 1
        assert "aborted; nothing moved" in capsys.readouterr().out
        assert len(pngs(train)) == 6

    def test_copy_does_not_prompt(self, tmp_path, monkeypatch):
        train = make_dataset(tmp_path / "train", 4)

        def explode(_):
            raise AssertionError("copy must not prompt")

        monkeypatch.setattr("builtins.input", explode)
        assert split_dataset.main(["--dir", str(train), "--count", "1", "--copy"]) == 0


class TestMainCLI:
    def test_reads_sys_argv_by_default(self, tmp_path, monkeypatch):
        train = make_dataset(tmp_path / "train", 4)
        monkeypatch.setattr(
            sys, "argv", ["split_dataset.py", "--dir", str(train), "--count", "1", "-y"]
        )
        assert split_dataset.main() == 0

    def test_missing_dir_exits_2(self, tmp_path, capsys):
        assert split_dataset.main(["--dir", str(tmp_path / "nope")]) == 2
        assert "not a directory" in capsys.readouterr().err

    def test_too_few_images_exits_2(self, tmp_path, capsys):
        train = make_dataset(tmp_path / "train", 1)
        assert split_dataset.main(["--dir", str(train)]) == 2
        assert "need at least 2 images" in capsys.readouterr().err

    def test_val_dir_same_as_dir_exits_2(self, tmp_path, capsys):
        train = make_dataset(tmp_path / "train", 4)
        rc = split_dataset.main(["--dir", str(train), "--val-dir", str(train)])
        assert rc == 2
        assert "must differ" in capsys.readouterr().err

    @pytest.mark.parametrize(
        "extra,message",
        [
            (["--threshold", "-1"], "--threshold must be >= 0"),
            (["--count", "0"], "--count must be >= 1"),
            (["--fraction", "0"], "--fraction must be between 0 and 1"),
            (["--fraction", "1"], "--fraction must be between 0 and 1"),
        ],
    )
    def test_bad_arguments_exit_2(self, tmp_path, extra, message, capsys):
        train = make_dataset(tmp_path / "train", 4)
        assert split_dataset.main(["--dir", str(train), *extra]) == 2
        assert message in capsys.readouterr().err
