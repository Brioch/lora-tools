"""Shared fixtures: build minimal .safetensors files and images on disk."""

import json
import struct

import pytest


@pytest.fixture
def make_safetensors():
    """Factory: write a minimal valid .safetensors file and return its path.

    header_extra lets a test add tensor keys (e.g. a realistic LoRA layout).
    """

    def _make(path, metadata=None, buffer=b"\x00\x01\x02\x03", header_extra=None):
        header = {
            "weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, len(buffer)]}
        }
        if header_extra:
            header.update(header_extra)
        if metadata is not None:
            header["__metadata__"] = metadata
        blob = json.dumps(header).encode("utf-8")
        with open(path, "wb") as f:
            f.write(struct.pack("<Q", len(blob)))
            f.write(blob)
            f.write(buffer)
        return str(path)

    return _make


@pytest.fixture
def make_image():
    """Factory: write a small solid-colour image and return its path."""

    def _make(path, color=(200, 30, 30), size=(16, 16), fmt=None):
        from PIL import Image

        Image.new("RGB", size, color).save(path, format=fmt)
        return str(path)

    return _make
