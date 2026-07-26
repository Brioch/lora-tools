"""Tests for tools/edit_captions.py — tag-aware transforms and the CLI."""

import argparse
import sys

import edit_captions
import pytest
from edit_captions import split_tags, transform


def args_ns(**kw):
    """Namespace with every operation off unless overridden."""
    base = {
        "replace": [],
        "remove": [],
        "prepend": [],
        "add": [],
        "dedupe": False,
        "sort": False,
    }
    base.update(kw)
    return argparse.Namespace(**base)


class TestSplitTags:
    def test_strips_and_drops_empties(self):
        assert split_tags(" a ,, b ,") == ["a", "b"]

    def test_empty_caption(self):
        assert split_tags("") == []


class TestTransform:
    def test_replace_matches_whole_tags_only(self):
        out = transform("man, woman", args_ns(replace=[["man", "person"]]))
        assert out == "person, woman"

    def test_replace_is_case_insensitive(self):
        assert transform("MAN", args_ns(replace=[["man", "person"]])) == "person"

    def test_remove_drops_matching_tags(self):
        assert transform("a, blurry, b", args_ns(remove=["BLURRY"])) == "a, b"

    def test_prepend_keeps_listed_order(self):
        out = transform("c", args_ns(prepend=["a", "b"]))
        assert out == "a, b, c"

    def test_prepend_is_idempotent(self):
        assert transform("tok, a", args_ns(prepend=["TOK"])) == "tok, a"

    def test_add_appends_when_absent(self):
        assert transform("a", args_ns(add=["b"])) == "a, b"

    def test_add_is_idempotent(self):
        assert transform("a, b", args_ns(add=["B"])) == "a, b"

    def test_dedupe_keeps_first_casing(self):
        assert transform("Tree, tree, a", args_ns(dedupe=True)) == "Tree, a"

    def test_sort_is_case_insensitive(self):
        assert transform("b, A, c", args_ns(sort=True)) == "A, b, c"

    def test_operations_apply_in_documented_order(self):
        # replace -> remove -> prepend -> add -> dedupe -> sort
        out = transform(
            "man, blurry, tree, tree",
            args_ns(
                replace=[["man", "person"]],
                remove=["blurry"],
                prepend=["tok"],
                dedupe=True,
            ),
        )
        assert out == "tok, person, tree"


class TestMainCLI:
    def test_applies_every_operation(self, tmp_path):
        (tmp_path / "1.txt").write_text("man, blurry, tree, tree")
        (tmp_path / "2.txt").write_text("woman, blurry")
        rc = edit_captions.main(
            [
                "--dir",
                str(tmp_path),
                "--prepend",
                "tok",
                "--remove",
                "blurry",
                "--replace",
                "man",
                "person",
                "--dedupe",
            ]
        )
        first = (tmp_path / "1.txt").read_text().strip()
        second = (tmp_path / "2.txt").read_text().strip()
        assert rc == 0
        assert first.startswith("tok,")
        assert "blurry" not in first and "blurry" not in second
        assert "person" in [t.strip() for t in first.split(",")]
        assert "woman" in [t.strip() for t in second.split(",")]  # not a substring edit
        assert first.count("tree") == 1

    def test_prepend_twice_is_idempotent(self, tmp_path):
        (tmp_path / "1.txt").write_text("a, b")
        edit_captions.main(["--dir", str(tmp_path), "--prepend", "tok"])
        edit_captions.main(["--dir", str(tmp_path), "--prepend", "tok"])
        assert (tmp_path / "1.txt").read_text().count("tok") == 1

    def test_prints_a_diff_for_changed_files(self, tmp_path, capsys):
        (tmp_path / "1.txt").write_text("a, b")
        edit_captions.main(["--dir", str(tmp_path), "--add", "c"])
        out = capsys.readouterr().out
        assert "  - a, b" in out and "  + a, b, c" in out
        assert "changed 1 of 1" in out

    def test_dry_run_writes_nothing(self, tmp_path, capsys):
        (tmp_path / "1.txt").write_text("a, b")
        edit_captions.main(["--dir", str(tmp_path), "--add", "c", "--dry-run"])
        assert (tmp_path / "1.txt").read_text() == "a, b"
        assert "would change 1" in capsys.readouterr().out

    def test_no_change_is_reported(self, tmp_path, capsys):
        (tmp_path / "1.txt").write_text("a, b")
        assert edit_captions.main(["--dir", str(tmp_path), "--add", "b"]) == 0
        assert "changed 0 of 1" in capsys.readouterr().out

    def test_custom_caption_ext(self, tmp_path):
        (tmp_path / "1.caption").write_text("a")
        edit_captions.main(
            ["--dir", str(tmp_path), "--caption-ext", "caption", "--add", "b"]
        )
        assert (tmp_path / "1.caption").read_text().strip() == "a, b"

    def test_reads_sys_argv_by_default(self, tmp_path, monkeypatch):
        (tmp_path / "1.txt").write_text("a")
        monkeypatch.setattr(
            sys, "argv", ["edit_captions.py", "--dir", str(tmp_path), "--add", "b"]
        )
        assert edit_captions.main() == 0
        assert (tmp_path / "1.txt").read_text().strip() == "a, b"

    def test_no_operation_exits_2(self, tmp_path, capsys):
        assert edit_captions.main(["--dir", str(tmp_path)]) == 2
        assert "no edit operation" in capsys.readouterr().err

    def test_missing_dir_exits_2(self, tmp_path, capsys):
        rc = edit_captions.main(["--dir", str(tmp_path / "nope"), "--add", "x"])
        assert rc == 2
        assert "not a directory" in capsys.readouterr().err

    @pytest.mark.parametrize("flag", ["--dedupe", "--sort"])
    def test_flag_only_runs_count_as_operations(self, tmp_path, flag):
        (tmp_path / "1.txt").write_text("b, a, a")
        assert edit_captions.main(["--dir", str(tmp_path), flag]) == 0
