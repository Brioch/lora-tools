"""Tests for the pure scheduling logic in tools/calc_training.py."""

import pytest
from calc_training import compute, parse_pass, parse_split


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
        # 31 images, 1250-effective-step budget, 3/4 - 1/4 split.
        passes = [(512, 2, 1), (1024, 1, 1)]
        fractions = parse_split("3/1", 2)
        rows = compute(31, 1250, passes, fractions, drop_last=True)

        bulk, refine = rows
        # 512 @ batch 2: floor(31/2)=15 steps/epoch, round(0.75*1250/31)=30 epochs.
        assert (bulk["steps_per_epoch"], bulk["epochs"], bulk["steps"]) == (15, 30, 450)
        assert bulk["effective"] == 900  # 450 steps * batch 2
        # 1024 @ batch 1: floor(31/1)=31 steps/epoch, round(0.25*1250/31)=10 epochs.
        assert (refine["steps_per_epoch"], refine["epochs"], refine["steps"]) == (
            31,
            10,
            310,
        )
        assert refine["effective"] == 310

    def test_keep_last_ceils_steps_per_epoch(self):
        drop = compute(31, 1250, [(512, 2, 1)], [1.0], drop_last=True)
        keep = compute(31, 1250, [(512, 2, 1)], [1.0], drop_last=False)
        assert drop[0]["steps_per_epoch"] == 15  # floor(31/2)
        assert keep[0]["steps_per_epoch"] == 16  # ceil(31/2)

    def test_accum_multiplies_effective_batch(self):
        # batch 1 with accum 2 == effective batch 2.
        rows = compute(31, 1250, [(1024, 1, 2)], [1.0], drop_last=True)
        assert rows[0]["steps_per_epoch"] == 15  # floor(31 / (1*2))

    def test_effective_is_steps_times_effective_batch(self):
        rows = compute(20, 800, [(512, 2, 1), (768, 2, 1)], [0.5, 0.5])
        for r in rows:
            assert r["effective"] == r["steps"] * r["batch"] * r["accum"]
