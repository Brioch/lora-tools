"""Tests for tools/embed_workflow.py — embed/read round-trips across formats."""

import json
import sys

import embed_workflow
import pytest
from embed_workflow import read_embedded


def write_json(path, obj):
    path.write_text(json.dumps(obj))
    return str(path)


WORKFLOW = {"nodes": [{"id": 1, "type": "KSampler"}, {"id": 2, "type": "VAEDecode"}]}


class TestRoundTrip:
    def test_png(self, make_image, tmp_path, monkeypatch, capsys):
        img = make_image(tmp_path / "shot.png")
        wf = write_json(tmp_path / "wf.json", WORKFLOW)
        out = tmp_path / "out.png"
        monkeypatch.setattr(sys, "argv", ["embed_workflow.py", img, wf, "-o", str(out)])
        embed_workflow.main()
        found = read_embedded(str(out))
        assert json.loads(found["workflow"]) == WORKFLOW

    def test_webp_default_with_prompt(self, make_image, tmp_path, monkeypatch):
        img = make_image(tmp_path / "shot.png")
        wf = write_json(tmp_path / "wf.json", WORKFLOW)
        pr = write_json(tmp_path / "prompt.json", {"1": {"class_type": "X"}})
        # No -o: writes <image>.embed.webp next to the input.
        monkeypatch.setattr(sys, "argv", ["embed_workflow.py", img, wf, "--prompt", pr])
        embed_workflow.main()
        found = read_embedded(str(tmp_path / "shot.embed.webp"))
        assert "workflow" in found and "prompt" in found

    def test_jpeg_converts_non_rgb(self, tmp_path, monkeypatch):
        from PIL import Image

        img = tmp_path / "shot.png"
        Image.new("RGBA", (16, 16), (10, 20, 30, 255)).save(img)
        wf = write_json(tmp_path / "wf.json", WORKFLOW)
        out = tmp_path / "out.jpg"
        monkeypatch.setattr(
            sys, "argv", ["embed_workflow.py", str(img), wf, "-o", str(out)]
        )
        embed_workflow.main()
        assert json.loads(read_embedded(str(out))["workflow"]) == WORKFLOW

    def test_read_reports_embedded(self, make_image, tmp_path, monkeypatch, capsys):
        img = make_image(tmp_path / "shot.png")
        wf = write_json(tmp_path / "wf.json", WORKFLOW)
        out = tmp_path / "out.png"
        monkeypatch.setattr(sys, "argv", ["embed_workflow.py", img, wf, "-o", str(out)])
        embed_workflow.main()
        capsys.readouterr()
        monkeypatch.setattr(sys, "argv", ["embed_workflow.py", str(out), "--read"])
        embed_workflow.main()
        assert "workflow" in capsys.readouterr().out

    def test_read_when_nothing_embedded(
        self, make_image, tmp_path, monkeypatch, capsys
    ):
        img = make_image(tmp_path / "plain.png")
        monkeypatch.setattr(sys, "argv", ["embed_workflow.py", img, "--read"])
        embed_workflow.main()
        assert "no ComfyUI" in capsys.readouterr().out

    def test_webp_lossy_uses_quality(self, make_image, tmp_path, monkeypatch):
        img = make_image(tmp_path / "shot.png")
        wf = write_json(tmp_path / "wf.json", WORKFLOW)
        out = tmp_path / "out.webp"
        # --lossy takes the non-lossless save path (quality, not effort).
        monkeypatch.setattr(
            sys, "argv", ["embed_workflow.py", img, wf, "-o", str(out), "--lossy"]
        )
        embed_workflow.main()
        assert json.loads(read_embedded(str(out))["workflow"]) == WORKFLOW


class TestErrors:
    def test_missing_image(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            sys, "argv", ["embed_workflow.py", str(tmp_path / "nope.png"), "--read"]
        )
        with pytest.raises(SystemExit):
            embed_workflow.main()

    def test_workflow_required_without_read(self, make_image, tmp_path, monkeypatch):
        img = make_image(tmp_path / "shot.png")
        monkeypatch.setattr(sys, "argv", ["embed_workflow.py", img])
        with pytest.raises(SystemExit):
            embed_workflow.main()

    def test_missing_workflow_file(self, make_image, tmp_path, monkeypatch):
        img = make_image(tmp_path / "shot.png")
        monkeypatch.setattr(
            sys, "argv", ["embed_workflow.py", img, str(tmp_path / "nope.json")]
        )
        with pytest.raises(SystemExit):
            embed_workflow.main()

    def test_non_workflow_json_warns(self, make_image, tmp_path, monkeypatch, capsys):
        img = make_image(tmp_path / "shot.png")
        wf = write_json(tmp_path / "wf.json", {"not": "a workflow"})
        out = tmp_path / "out.png"
        monkeypatch.setattr(sys, "argv", ["embed_workflow.py", img, wf, "-o", str(out)])
        embed_workflow.main()
        assert "nodes" in capsys.readouterr().err

    def test_missing_prompt_file(self, make_image, tmp_path, monkeypatch):
        img = make_image(tmp_path / "shot.png")
        wf = write_json(tmp_path / "wf.json", WORKFLOW)
        monkeypatch.setattr(
            sys,
            "argv",
            ["embed_workflow.py", img, wf, "--prompt", str(tmp_path / "nope.json")],
        )
        with pytest.raises(SystemExit):
            embed_workflow.main()

    def test_jpeg_workflow_too_large_exits(self, make_image, tmp_path, monkeypatch):
        img = make_image(tmp_path / "shot.png")
        # A payload past JPEG's ~64KB EXIF APP1 limit must be rejected, not saved.
        wf = write_json(tmp_path / "wf.json", {"nodes": [], "blob": "x" * 70000})
        out = tmp_path / "out.jpg"
        monkeypatch.setattr(sys, "argv", ["embed_workflow.py", img, wf, "-o", str(out)])
        with pytest.raises(SystemExit) as e:
            embed_workflow.main()
        assert "too large" in str(e.value.code)
