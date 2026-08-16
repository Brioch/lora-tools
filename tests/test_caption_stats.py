"""Tests for tools/caption_stats.py — tag counting, the two cutoffs, and the CLI."""

import json
import sys

import caption_stats
import pytest
from caption_stats import summarise


def write_captions(directory, texts, ext=".txt"):
    """Write one caption file per entry in *texts*, named 1..n."""
    directory.mkdir(parents=True, exist_ok=True)
    for i, text in enumerate(texts, start=1):
        (directory / f"{i}{ext}").write_text(text, encoding="utf-8")
    return directory


class TestSummarise:
    def test_counts_captions_and_instances(self):
        stats = summarise([["a", "b"], ["a"]])
        assert stats["captions"] == 2
        assert stats["instances"] == 3
        assert stats["vocabulary"] == 2

    def test_document_frequency_ignores_repeats_within_a_caption(self):
        stats = summarise([["tree", "tree", "tree"], ["sky"]])
        tree = next(t for t in stats["tags"] if t["tag"] == "tree")
        assert tree["captions"] == 1  # one caption contains it…
        assert tree["occurrences"] == 3  # …three times

    def test_tags_are_folded_case_insensitively(self):
        stats = summarise([["Blue Eyes"], ["blue eyes"], ["BLUE EYES"]])
        assert stats["vocabulary"] == 1
        assert stats["tags"][0]["captions"] == 3

    def test_most_common_spelling_is_reported(self):
        stats = summarise([["cat"], ["cat"], ["Cat"]])
        assert stats["tags"][0]["tag"] == "cat"

    def test_ranked_by_caption_count_then_name(self):
        stats = summarise([["b", "a"], ["a"], ["a"]])
        assert [t["tag"] for t in stats["tags"]] == ["a", "b"]

    def test_pct_is_share_of_captions(self):
        stats = summarise([["a"], ["a"], ["b"], ["b"]])
        assert {t["tag"]: t["pct"] for t in stats["tags"]} == {"a": 50.0, "b": 50.0}

    def test_tags_per_caption_spread(self):
        per = summarise([["a"], ["a", "b", "c"], ["a", "b"]])["tags_per_caption"]
        assert (per["min"], per["median"], per["max"]) == (1, 2, 3)

    def test_empty_captions_are_counted(self):
        stats = summarise([[], ["a"]])
        assert stats["empty"] == 1

    def test_no_captions_at_all(self):
        stats = summarise([])
        assert stats["captions"] == 0
        assert stats["tags"] == []
        assert stats["tags_per_caption"]["median"] == 0


class TestMainCLI:
    def test_reports_the_distribution(self, tmp_path, capsys):
        write_captions(tmp_path, ["mytoken, smiling", "mytoken, frowning"])
        assert caption_stats.main(["--dir", str(tmp_path)]) == 0
        out = capsys.readouterr().out
        assert "scanned 2 caption file(s)" in out
        assert "vocabulary    : 3 unique tag(s)" in out
        assert "mytoken" in out

    def test_flags_ubiquitous_tags(self, tmp_path, capsys):
        write_captions(tmp_path, ["mytoken, a", "mytoken, b", "mytoken, c"])
        caption_stats.main(["--dir", str(tmp_path)])
        out = capsys.readouterr().out
        assert "ubiquitous" in out
        assert "mytoken  (3 captions, 100.0%)" in out

    def test_flags_rare_tags(self, tmp_path, capsys):
        write_captions(tmp_path, ["common, bluu eyes", "common", "common"])
        caption_stats.main(["--dir", str(tmp_path)])
        out = capsys.readouterr().out
        assert "rare (<= 2 occurrence(s))" in out
        assert "bluu eyes" in out

    def test_cutoffs_can_be_disabled(self, tmp_path, capsys):
        write_captions(tmp_path, ["only, rare"])
        caption_stats.main(
            ["--dir", str(tmp_path), "--rare", "0", "--ubiquitous-pct", "0"]
        )
        out = capsys.readouterr().out
        assert "ubiquitous" not in out
        assert "rare (" not in out

    def test_top_limits_the_table(self, tmp_path, capsys):
        write_captions(tmp_path, ["a, b, c, d, e"])
        caption_stats.main(["--dir", str(tmp_path), "--top", "2"])
        out = capsys.readouterr().out
        assert "top 2 tag(s)" in out
        assert "(+3 more" in out

    def test_json_output(self, tmp_path, capsys):
        write_captions(tmp_path, ["mytoken, a", "mytoken, b"])
        caption_stats.main(["--dir", str(tmp_path), "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["captions"] == 2
        assert payload["vocabulary"] == 3
        assert [t["tag"] for t in payload["ubiquitous"]] == ["mytoken"]
        # mytoken is under the absolute --rare cutoff but is in every caption, so it
        # must appear only as ubiquitous — the two lists are disjoint.
        assert {t["tag"] for t in payload["rare"]} == {"a", "b"}

    def test_ubiquitous_tags_are_never_also_called_rare(self, tmp_path, capsys):
        write_captions(tmp_path, ["mytoken", "mytoken"])
        caption_stats.main(["--dir", str(tmp_path), "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert [t["tag"] for t in payload["ubiquitous"]] == ["mytoken"]
        assert payload["rare"] == []

    def test_custom_caption_ext(self, tmp_path, capsys):
        write_captions(tmp_path, ["a, b"], ext=".caption")
        caption_stats.main(["--dir", str(tmp_path), "--caption-ext", "caption"])
        assert "scanned 1 caption file(s)" in capsys.readouterr().out

    def test_empty_dir_is_reported_not_crashed(self, tmp_path, capsys):
        assert caption_stats.main(["--dir", str(tmp_path)]) == 0
        assert "no '.txt' files found" in capsys.readouterr().out

    def test_reads_sys_argv_by_default(self, tmp_path, monkeypatch):
        write_captions(tmp_path, ["a"])
        monkeypatch.setattr(sys, "argv", ["caption_stats.py", "--dir", str(tmp_path)])
        assert caption_stats.main() == 0

    def test_missing_dir_exits_2(self, tmp_path, capsys):
        assert caption_stats.main(["--dir", str(tmp_path / "nope")]) == 2
        assert "not a directory" in capsys.readouterr().err

    @pytest.mark.parametrize("text", ["", "   ", ",,,"])
    def test_blank_captions_do_not_crash(self, tmp_path, text):
        write_captions(tmp_path, [text])
        assert caption_stats.main(["--dir", str(tmp_path)]) == 0
