"""Tests for tools/balance_regularization.py — filling, trimming and the delete gate."""

import sys

import balance_regularization
import pytest
from balance_regularization import remove, transfer


def make_pairs(directory, stems, ext=".png", caption=True):
    """Create image (+ caption) pairs named after *stems* in *directory*."""
    directory.mkdir(parents=True, exist_ok=True)
    for stem in stems:
        (directory / f"{stem}{ext}").write_bytes(b"x")
        if caption:
            (directory / f"{stem}.txt").write_text("cap")
    return directory


def pngs(directory):
    return sorted(p.name for p in directory.glob("*.png"))


def txts(directory):
    return sorted(p.name for p in directory.glob("*.txt"))


@pytest.fixture
def dirs(tmp_path):
    """A train/pool/reg trio: 5 training images, a 20-image pool, empty reg."""
    make_pairs(tmp_path / "train", [f"t{i}" for i in range(5)])
    make_pairs(tmp_path / "pool", [f"p{i}" for i in range(20)])
    (tmp_path / "reg").mkdir()
    return tmp_path / "train", tmp_path / "pool", tmp_path / "reg"


def argv_for(train, pool, reg, *extra):
    return [
        "--train-dir",
        str(train),
        "--pool-dir",
        str(pool),
        "--reg-dir",
        str(reg),
        *extra,
    ]


class TestTransfer:
    def test_copies_image_and_caption(self, tmp_path):
        src = make_pairs(tmp_path / "src", ["a"])
        dest = tmp_path / "dest"
        dest.mkdir()
        assert transfer(src / "a.png", dest, ".txt", move=False, dry_run=False) is True
        assert (dest / "a.png").exists() and (dest / "a.txt").exists()
        assert (src / "a.png").exists()  # copy leaves the source in place

    def test_move_removes_the_source(self, tmp_path):
        src = make_pairs(tmp_path / "src", ["a"])
        dest = tmp_path / "dest"
        dest.mkdir()
        transfer(src / "a.png", dest, ".txt", move=True, dry_run=False)
        assert not (src / "a.png").exists()
        assert (dest / "a.png").exists()

    def test_reports_missing_caption(self, tmp_path):
        src = make_pairs(tmp_path / "src", ["a"], caption=False)
        dest = tmp_path / "dest"
        dest.mkdir()
        assert transfer(src / "a.png", dest, ".txt", move=False, dry_run=False) is False

    def test_dry_run_writes_nothing(self, tmp_path):
        src = make_pairs(tmp_path / "src", ["a"])
        dest = tmp_path / "dest"
        dest.mkdir()
        transfer(src / "a.png", dest, ".txt", move=False, dry_run=True)
        assert pngs(dest) == []


class TestRemove:
    def test_deletes_image_and_caption(self, tmp_path):
        d = make_pairs(tmp_path / "reg", ["a"])
        assert remove(d / "a.png", ".txt", dry_run=False) is True
        assert pngs(d) == [] and txts(d) == []

    def test_dry_run_keeps_files(self, tmp_path):
        d = make_pairs(tmp_path / "reg", ["a"])
        remove(d / "a.png", ".txt", dry_run=True)
        assert pngs(d) == ["a.png"]


class TestFilling:
    def test_fills_empty_reg_to_n(self, dirs):
        train, pool, reg = dirs
        assert (
            balance_regularization.main(argv_for(train, pool, reg, "--seed", "0")) == 0
        )
        assert len(pngs(reg)) == 5
        # Captions stay paired with their images.
        assert [p[:-4] for p in pngs(reg)] == [p[:-4] for p in txts(reg)]

    def test_is_idempotent(self, dirs):
        train, pool, reg = dirs
        argv = argv_for(train, pool, reg, "--seed", "0")
        balance_regularization.main(argv)
        balance_regularization.main(argv)
        assert len(pngs(reg)) == 5

    def test_already_balanced_is_a_no_op(self, dirs, capsys):
        train, pool, reg = dirs
        make_pairs(reg, [f"r{i}" for i in range(5)])
        balance_regularization.main(argv_for(train, pool, reg))
        assert "already balanced" in capsys.readouterr().out
        assert len(pngs(reg)) == 5

    def test_move_empties_the_pool(self, dirs):
        train, pool, reg = dirs
        balance_regularization.main(argv_for(train, pool, reg, "--seed", "0", "--move"))
        assert len(pngs(pool)) == 15
        assert len(pngs(reg)) == 5

    def test_skips_stems_already_in_reg(self, dirs):
        train, pool, reg = dirs
        make_pairs(reg, ["p0", "p1"], caption=False)
        balance_regularization.main(argv_for(train, pool, reg, "--seed", "3"))
        names = pngs(reg)
        assert len(names) == 5 and len(set(names)) == 5

    def test_pool_shortfall_warns(self, tmp_path, capsys):
        train = make_pairs(tmp_path / "train", [f"t{i}" for i in range(10)])
        pool = make_pairs(tmp_path / "pool", [f"p{i}" for i in range(3)])
        reg = tmp_path / "reg"
        reg.mkdir()
        balance_regularization.main(argv_for(train, pool, reg, "--seed", "0"))
        assert len(pngs(reg)) == 3
        assert "short" in capsys.readouterr().err.lower()

    def test_missing_captions_warn(self, tmp_path, capsys):
        train = make_pairs(tmp_path / "train", ["t0"])
        pool = make_pairs(tmp_path / "pool", ["p0"], caption=False)
        reg = tmp_path / "reg"
        reg.mkdir()
        balance_regularization.main(argv_for(train, pool, reg, "--seed", "0"))
        assert "no paired" in capsys.readouterr().err

    def test_creates_a_missing_reg_dir(self, tmp_path, capsys):
        train = make_pairs(tmp_path / "train", ["t0"])
        pool = make_pairs(tmp_path / "pool", ["p0"])
        reg = tmp_path / "nested" / "reg"  # does not exist yet
        assert balance_regularization.main(argv_for(train, pool, reg)) == 0
        assert "creating" in capsys.readouterr().out
        assert pngs(reg) == ["p0.png"]


class TestTrimming:
    def test_trims_excess_with_yes(self, tmp_path):
        train = make_pairs(tmp_path / "train", [f"t{i}" for i in range(3)])
        pool = make_pairs(tmp_path / "pool", [f"p{i}" for i in range(5)])
        reg = make_pairs(tmp_path / "reg", [f"p{i}" for i in range(8)])
        rc = balance_regularization.main(
            argv_for(train, pool, reg, "--seed", "1", "--yes")
        )
        assert rc == 0
        assert len(pngs(reg)) == 3 and len(txts(reg)) == 3

    def test_prompt_accepted_deletes(self, tmp_path, monkeypatch):
        train = make_pairs(tmp_path / "train", ["t0"])
        pool = make_pairs(tmp_path / "pool", ["p0"])
        reg = make_pairs(tmp_path / "reg", ["r0", "r1", "r2"])
        monkeypatch.setattr("builtins.input", lambda _: "y")
        assert (
            balance_regularization.main(argv_for(train, pool, reg, "--seed", "1")) == 0
        )
        assert len(pngs(reg)) == 1

    def test_prompt_declined_aborts(self, tmp_path, monkeypatch, capsys):
        train = make_pairs(tmp_path / "train", ["t0"])
        pool = make_pairs(tmp_path / "pool", ["p0"])
        reg = make_pairs(tmp_path / "reg", ["r0", "r1", "r2"])
        monkeypatch.setattr("builtins.input", lambda _: "n")
        rc = balance_regularization.main(argv_for(train, pool, reg, "--seed", "1"))
        assert rc == 1
        assert "aborted" in capsys.readouterr().out
        assert len(pngs(reg)) == 3  # nothing deleted


class TestDryRun:
    def test_fill_changes_nothing(self, dirs):
        train, pool, reg = dirs
        balance_regularization.main(
            argv_for(train, pool, reg, "--seed", "0", "--dry-run")
        )
        assert pngs(reg) == []

    def test_trim_changes_nothing(self, tmp_path, capsys):
        train = make_pairs(tmp_path / "train", ["t0"])
        pool = make_pairs(tmp_path / "pool", ["p0"])
        reg = make_pairs(tmp_path / "reg", ["r0", "r1", "r2"])
        balance_regularization.main(argv_for(train, pool, reg, "--dry-run"))
        assert len(pngs(reg)) == 3
        assert "not written" in capsys.readouterr().out


class TestMainCLI:
    def test_reads_sys_argv_by_default(self, dirs, monkeypatch):
        train, pool, reg = dirs
        monkeypatch.setattr(
            sys,
            "argv",
            ["balance_regularization.py", *argv_for(train, pool, reg, "--seed", "0")],
        )
        assert balance_regularization.main() == 0
        assert len(pngs(reg)) == 5

    @pytest.mark.parametrize("bad", ["train", "pool"])
    def test_missing_source_dir_exits_2(self, tmp_path, bad, capsys):
        train = make_pairs(tmp_path / "train", ["t0"])
        pool = make_pairs(tmp_path / "pool", ["p0"])
        reg = tmp_path / "reg"
        reg.mkdir()
        paths = {"train": train, "pool": pool}
        paths[bad] = tmp_path / "nope"
        rc = balance_regularization.main(argv_for(paths["train"], paths["pool"], reg))
        assert rc == 2
        assert f"--{bad}-dir is not a directory" in capsys.readouterr().err

    def test_empty_train_dir_exits_2(self, tmp_path, capsys):
        train = tmp_path / "train"
        train.mkdir()
        pool = make_pairs(tmp_path / "pool", ["p0"])
        reg = tmp_path / "reg"
        reg.mkdir()
        assert balance_regularization.main(argv_for(train, pool, reg)) == 2
        assert "no images found" in capsys.readouterr().err

    def test_reg_dir_that_is_a_file_exits_2(self, tmp_path, capsys):
        train = make_pairs(tmp_path / "train", ["t0"])
        pool = make_pairs(tmp_path / "pool", ["p0"])
        reg = tmp_path / "reg.png"
        reg.write_bytes(b"x")
        assert balance_regularization.main(argv_for(train, pool, reg)) == 2
        assert "--reg-dir is not a directory" in capsys.readouterr().err
