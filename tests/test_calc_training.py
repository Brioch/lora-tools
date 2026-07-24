"""Tests for tools/calc_training.py — pure logic, CLI, and interactive mode."""

import sys

import calc_training
import pytest
from calc_training import (
    _ask_bool,
    _ask_int,
    _ask_passes,
    _ask_split,
    compute,
    format_report,
    parse_pass,
    parse_split,
)


class TestParsePass:
    def test_res_batch(self):
        assert parse_pass("512:2") == (512, 2, 1)

    def test_res_batch_accum(self):
        assert parse_pass("1024:1:2") == (1024, 1, 2)

    @pytest.mark.parametrize("spec", ["512", "512:2:1:1", ""])
    def test_wrong_arity(self, spec):
        with pytest.raises(ValueError, match="RES:BATCH"):
            parse_pass(spec)

    @pytest.mark.parametrize("spec", ["a:2", "512:x", "512:2:z"])
    def test_non_integer(self, spec):
        with pytest.raises(ValueError, match="integer"):
            parse_pass(spec)

    @pytest.mark.parametrize("spec", ["512:0", "0:2", "512:2:0"])
    def test_non_positive(self, spec):
        with pytest.raises(ValueError, match="positive"):
            parse_pass(spec)


class TestParseSplit:
    def test_normalizes_to_one(self):
        assert parse_split("3/1", 2) == [0.75, 0.25]

    def test_percentages_equivalent(self):
        assert parse_split("60/40", 2) == pytest.approx([0.6, 0.4])

    def test_equal_weights_sum_to_one(self):
        assert sum(parse_split("1/1/1", 3)) == pytest.approx(1.0)

    def test_wrong_count(self):
        with pytest.raises(ValueError, match="expected 2 weight"):
            parse_split("1/1/1", 2)

    @pytest.mark.parametrize("spec", ["a/b", "1/x"])
    def test_non_numeric(self, spec):
        with pytest.raises(ValueError, match="numbers"):
            parse_split(spec, 2)

    @pytest.mark.parametrize("spec", ["0/1", "-1/2"])
    def test_non_positive(self, spec):
        with pytest.raises(ValueError, match="positive"):
            parse_split(spec, 2)


class TestCompute:
    def test_default_krea2_recipe(self):
        passes = [(512, 2, 1), (1024, 1, 1)]
        fractions = parse_split("3/1", 2)
        rows = compute(31, 1250, passes, fractions, drop_last=True)

        bulk, refine = rows
        assert (bulk["steps_per_epoch"], bulk["epochs"], bulk["steps"]) == (15, 30, 450)
        assert bulk["effective"] == 900
        assert (refine["steps_per_epoch"], refine["epochs"], refine["steps"]) == (
            31,
            10,
            310,
        )
        assert refine["effective"] == 310

    def test_keep_last_ceils_steps_per_epoch(self):
        drop = compute(31, 1250, [(512, 2, 1)], [1.0], drop_last=True)
        keep = compute(31, 1250, [(512, 2, 1)], [1.0], drop_last=False)
        assert drop[0]["steps_per_epoch"] == 15
        assert keep[0]["steps_per_epoch"] == 16

    def test_accum_multiplies_effective_batch(self):
        rows = compute(31, 1250, [(1024, 1, 2)], [1.0], drop_last=True)
        assert rows[0]["steps_per_epoch"] == 15

    def test_effective_is_steps_times_effective_batch(self):
        rows = compute(20, 800, [(512, 2, 1), (768, 2, 1)], [0.5, 0.5])
        for r in rows:
            assert r["effective"] == r["steps"] * r["batch"] * r["accum"]


class TestFormatReport:
    def test_shows_accum_column_only_when_used(self):
        rows = compute(31, 1250, [(512, 2, 1), (1024, 1, 1)], [0.75, 0.25])
        assert "accum" not in format_report(31, 1250, "3/1", rows)
        rows_accum = compute(31, 1250, [(1024, 1, 2)], [1.0])
        assert "accum" in format_report(31, 1250, "1", rows_accum)

    def test_reports_totals_and_target(self):
        rows = compute(31, 1250, [(512, 2, 1), (1024, 1, 1)], [0.75, 0.25])
        out = format_report(31, 1250, "3/1", rows)
        assert "31 images" in out
        assert "Total:" in out and "target 1250" in out


def _feed(monkeypatch, answers):
    it = iter(answers)
    monkeypatch.setattr("builtins.input", lambda prompt="": next(it))


class TestInteractiveHelpers:
    def test_ask_int_reasks_until_valid(self, monkeypatch):
        _feed(monkeypatch, ["", "abc", "-1", "7"])
        assert _ask_int("n") == 7

    def test_ask_int_accepts_default_on_blank(self, monkeypatch):
        _feed(monkeypatch, [""])
        assert _ask_int("n", 1250) == 1250

    def test_ask_bool_variants(self, monkeypatch):
        _feed(monkeypatch, ["maybe", "y"])
        assert _ask_bool("q") is True
        _feed(monkeypatch, ["n"])
        assert _ask_bool("q") is False
        _feed(monkeypatch, [""])
        assert _ask_bool("q", default=True) is True

    def test_ask_passes_reasks_on_bad_spec(self, monkeypatch):
        _feed(monkeypatch, ["nope", "512:2 1024:1"])
        assert _ask_passes("512:2") == [(512, 2, 1), (1024, 1, 1)]

    def test_ask_passes_accepts_default(self, monkeypatch):
        _feed(monkeypatch, [""])
        assert _ask_passes("512:2") == [(512, 2, 1)]

    def test_ask_split_reasks_on_bad_spec(self, monkeypatch):
        _feed(monkeypatch, ["9/9/9", "3/1"])
        assert _ask_split("3/1", 2) == "3/1"


class TestMainCLI:
    def test_positional_default_recipe(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["calc_training.py", "31", "1250"])
        calc_training.main()
        out = capsys.readouterr().out
        assert "Total:" in out and "512" in out and "1024" in out

    def test_custom_passes_and_split_and_keep_last(self, monkeypatch, capsys):
        monkeypatch.setattr(
            sys,
            "argv",
            [
                "calc_training.py",
                "20",
                "800",
                "--pass",
                "512:2",
                "--pass",
                "1024:1",
                "--split",
                "1/1",
                "--keep-last",
            ],
        )
        calc_training.main()
        assert "Total:" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "argv",
        [
            ["calc_training.py", "0", "1250"],  # images <= 0
            ["calc_training.py", "31", "0"],  # total_steps <= 0
            ["calc_training.py", "31"],  # only one positional
            ["calc_training.py", "31", "1250", "--pass", "bad"],  # bad pass
            ["calc_training.py", "31", "1250", "--split", "1/2/3"],  # bad split
        ],
    )
    def test_argument_errors_exit(self, monkeypatch, argv):
        monkeypatch.setattr(sys, "argv", argv)
        with pytest.raises(SystemExit):
            calc_training.main()

    def test_interactive_default_flow(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["calc_training.py", "-i"])
        _feed(monkeypatch, ["31", "1250", "", "", ""])
        calc_training.main()
        assert "Total:" in capsys.readouterr().out

    def test_no_args_runs_interactive(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["calc_training.py"])
        _feed(monkeypatch, ["31", "1250", "", "", "y"])
        calc_training.main()
        assert "Total:" in capsys.readouterr().out

    def test_interactive_abort_on_eof(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["calc_training.py", "-i"])

        def raise_eof(prompt=""):
            raise EOFError

        monkeypatch.setattr("builtins.input", raise_eof)
        with pytest.raises(SystemExit):
            calc_training.main()
