"""Tests for tools/dsutils.py — the extension, pairing and perceptual-hash helpers."""

import argparse
import random

import pytest
from dsutils import (
    HelpFormatter,
    dhash,
    group_similar,
    hamming,
    list_images,
    normalize_ext,
    normalize_exts,
    paired_caption,
    split_tags,
)


def noise_image(path, seed):
    """Deterministic random-noise image, so distinct seeds get far-apart hashes."""
    from PIL import Image

    rng = random.Random(seed)
    img = Image.new("L", (32, 32))
    img.putdata([rng.randrange(256) for _ in range(32 * 32)])
    img.convert("RGB").save(path)
    return str(path)


class TestNormalizeExt:
    def test_adds_missing_dot(self):
        assert normalize_ext("png") == ".png"

    def test_keeps_existing_dot(self):
        assert normalize_ext(".png") == ".png"

    def test_does_not_change_case(self):
        assert normalize_ext("PNG") == ".PNG"


class TestNormalizeExts:
    def test_lowercases_and_dots(self):
        assert normalize_exts(["PNG", ".JpG"]) == {".png", ".jpg"}

    def test_collapses_duplicates(self):
        assert normalize_exts([".png", "png", "PNG"]) == {".png"}

    def test_empty_input(self):
        assert normalize_exts([]) == set()


class TestListImages:
    def test_filters_by_extension(self, tmp_path, make_image):
        make_image(tmp_path / "a.png")
        make_image(tmp_path / "b.jpg", fmt="JPEG")
        (tmp_path / "notes.txt").write_text("x")
        names = [p.name for p in list_images(tmp_path, {".png"})]
        assert names == ["a.png"]

    def test_sorted_by_name(self, tmp_path, make_image):
        for stem in ("c", "a", "b"):
            make_image(tmp_path / f"{stem}.png")
        assert [p.name for p in list_images(tmp_path, {".png"})] == [
            "a.png",
            "b.png",
            "c.png",
        ]

    def test_suffix_match_is_case_insensitive(self, tmp_path, make_image):
        make_image(tmp_path / "a.PNG", fmt="PNG")
        assert len(list_images(tmp_path, {".png"})) == 1

    def test_skips_subdirectories(self, tmp_path, make_image):
        make_image(tmp_path / "a.png")
        (tmp_path / "sub.png").mkdir()  # a directory that looks like an image
        assert [p.name for p in list_images(tmp_path, {".png"})] == ["a.png"]

    def test_empty_directory(self, tmp_path):
        assert list_images(tmp_path, {".png"}) == []


class TestPairedCaption:
    def test_returns_existing_caption(self, tmp_path, make_image):
        make_image(tmp_path / "a.png")
        (tmp_path / "a.txt").write_text("cap")
        assert paired_caption(tmp_path / "a.png", ".txt") == tmp_path / "a.txt"

    def test_returns_none_when_absent(self, tmp_path, make_image):
        make_image(tmp_path / "a.png")
        assert paired_caption(tmp_path / "a.png", ".txt") is None


class TestSplitTags:
    def test_strips_and_drops_empties(self):
        assert split_tags(" a ,, b ,") == ["a", "b"]

    def test_preserves_order_and_duplicates(self):
        assert split_tags("b, a, b") == ["b", "a", "b"]

    def test_keeps_internal_spaces(self):
        assert split_tags("blue eyes, long hair") == ["blue eyes", "long hair"]

    @pytest.mark.parametrize("text", ["", "   ", ",", ",,,", "\n"])
    def test_blank_input_yields_no_tags(self, text):
        assert split_tags(text) == []


class TestGroupSimilar:
    def test_identical_hashes_group(self, tmp_path):
        a, b = tmp_path / "a.png", tmp_path / "b.png"
        assert group_similar({a: 0b1010, b: 0b1010}, threshold=0) == [[a, b]]

    def test_distant_hashes_stay_apart(self, tmp_path):
        a, b = tmp_path / "a.png", tmp_path / "b.png"
        assert group_similar({a: 0b0000, b: 0b1111}, threshold=2) == [[a], [b]]

    def test_grouping_is_transitive(self, tmp_path):
        """a~b and b~c must put all three together even when a and c are 2 apart."""
        a, b, c = (tmp_path / f"{n}.png" for n in "abc")
        groups = group_similar({a: 0b000, b: 0b001, c: 0b011}, threshold=1)
        assert groups == [[a, b, c]]

    def test_singletons_are_returned(self, tmp_path):
        a, b, c = (tmp_path / f"{n}.png" for n in "abc")
        groups = group_similar({a: 0b00, b: 0b00, c: 0b11}, threshold=0)
        assert groups == [[a, b], [c]]

    def test_output_is_deterministic_regardless_of_insertion_order(self, tmp_path):
        a, b, c = (tmp_path / f"{n}.png" for n in "abc")
        forward = group_similar({a: 0, b: 0, c: 7}, threshold=1)
        backward = group_similar({c: 7, b: 0, a: 0}, threshold=1)
        assert forward == backward == [[a, b], [c]]

    def test_empty_input(self):
        assert group_similar({}, threshold=5) == []


class TestHamming:
    def test_identical_is_zero(self):
        assert hamming(0b1010, 0b1010) == 0

    def test_counts_differing_bits(self):
        assert hamming(0b1010, 0b0011) == 2

    def test_all_bits_differ(self):
        assert hamming(0b1111, 0b0000) == 4


class TestDhash:
    def test_same_pixels_same_hash(self, tmp_path):
        from PIL import Image

        noise_image(tmp_path / "a.png", seed=1)
        with Image.open(tmp_path / "a.png") as im:
            first = dhash(im)
        with Image.open(tmp_path / "a.png") as im:
            assert dhash(im) == first

    def test_different_images_differ(self, tmp_path):
        from PIL import Image

        noise_image(tmp_path / "a.png", seed=1)
        noise_image(tmp_path / "b.png", seed=999)
        with Image.open(tmp_path / "a.png") as a, Image.open(tmp_path / "b.png") as b:
            assert hamming(dhash(a), dhash(b)) > 5

    def test_horizontal_gradient_is_resize_stable(self, tmp_path):
        """A smooth gradient hashes the same at two resolutions — the near-dupe case."""
        from PIL import Image

        big = Image.new("L", (64, 64))
        big.putdata([x * 4 for _ in range(64) for x in range(64)])
        small = big.resize((32, 32), Image.Resampling.LANCZOS)
        assert dhash(big) == dhash(small)

    def test_size_sets_the_bit_count(self, tmp_path):
        from PIL import Image

        noise_image(tmp_path / "a.png", seed=2)
        with Image.open(tmp_path / "a.png") as im:
            assert dhash(im, size=4).bit_length() <= 16
            assert dhash(im, size=8).bit_length() <= 64


class TestHelpFormatter:
    @pytest.fixture
    def parser(self):
        p = argparse.ArgumentParser(
            prog="demo",
            description="Line one.\n\n  Indented line.",
            formatter_class=HelpFormatter,
        )
        p.add_argument("--n", type=int, default=7, help="How many.")
        return p

    def test_keeps_description_line_breaks(self, parser):
        assert "  Indented line." in parser.format_help()

    def test_appends_defaults(self, parser):
        assert "default: 7" in parser.format_help()
