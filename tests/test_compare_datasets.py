"""Tests for the pure helpers in tools/compare_datasets.py (no Pillow needed)."""

import pytest
from compare_datasets import classify, hamming


class TestHamming:
    def test_identical_is_zero(self):
        assert hamming(0b1010, 0b1010) == 0

    def test_counts_differing_bits(self):
        # 0b1010 ^ 0b0011 == 0b1001 -> 2 bits differ.
        assert hamming(0b1010, 0b0011) == 2

    def test_all_bits_differ(self):
        assert hamming(0b1111, 0b0000) == 4


class TestClassify:
    def test_identical_file_wins(self):
        # A byte-identical file is labelled IDENTICAL even at distance 0.
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
