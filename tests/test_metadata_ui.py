"""Tests for tools/metadata_ui.py — helpers, the HTTP handler, and main()."""

import json
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import metadata_ui
import pytest


class TestSafePath:
    def test_relative_name(self, tmp_path, make_safetensors, monkeypatch):
        make_safetensors(tmp_path / "lora.safetensors", metadata={})
        monkeypatch.setattr(metadata_ui, "ROOT", str(tmp_path))
        assert metadata_ui.safe_path("lora.safetensors") == str(
            (tmp_path / "lora.safetensors").resolve()
        )

    def test_absolute_path(self, tmp_path, make_safetensors, monkeypatch):
        p = make_safetensors(tmp_path / "lora.safetensors", metadata={})
        monkeypatch.setattr(metadata_ui, "ROOT", str(tmp_path))
        assert metadata_ui.safe_path(p) == str(
            (tmp_path / "lora.safetensors").resolve()
        )

    @pytest.mark.parametrize("name", [None, "", "notlora.txt", "missing.safetensors"])
    def test_rejects(self, tmp_path, monkeypatch, name):
        monkeypatch.setattr(metadata_ui, "ROOT", str(tmp_path))
        assert metadata_ui.safe_path(name) is None


class TestListFilesAndReadMeta:
    def test_list_files(self, tmp_path, make_safetensors, monkeypatch):
        make_safetensors(tmp_path / "a.safetensors", metadata={})
        make_safetensors(tmp_path / "b.safetensors", metadata={})
        (tmp_path / "note.txt").write_text("x")
        monkeypatch.setattr(metadata_ui, "ROOT", str(tmp_path))
        names = {f["name"] for f in metadata_ui.list_files()}
        assert names == {"a.safetensors", "b.safetensors"}

    def test_read_meta(self, tmp_path, make_safetensors):
        p = make_safetensors(tmp_path / "a.safetensors", metadata={"k": "v"})
        assert metadata_ui.read_meta(p) == {"k": "v"}


class TestPickFile:
    def test_returns_chosen_path(self, monkeypatch):
        class P:
            returncode = 0
            stdout = "/some/lora.safetensors\n"
            stderr = ""

        monkeypatch.setattr(metadata_ui.subprocess, "run", lambda *a, **k: P())
        assert metadata_ui.pick_file() == "/some/lora.safetensors"

    def test_raises_on_failure(self, monkeypatch):
        class P:
            returncode = 1
            stdout = ""
            stderr = "no display\n"

        monkeypatch.setattr(metadata_ui.subprocess, "run", lambda *a, **k: P())
        with pytest.raises(RuntimeError, match="no display"):
            metadata_ui.pick_file()


def _req(url, method="GET", payload=None, raw=None):
    if raw is not None:
        data = raw
    elif payload is not None:
        data = json.dumps(payload).encode()
    else:
        data = None
    headers = {"Content-Type": "application/json"} if data is not None else {}
    r = urllib.request.Request(url, data=data, method=method, headers=headers)
    try:
        with urllib.request.urlopen(r, timeout=5) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


@pytest.fixture
def server(tmp_path, make_safetensors, monkeypatch):
    make_safetensors(
        tmp_path / "lora.safetensors", metadata={"modelspec.title": "Hi", "x": "1"}
    )
    monkeypatch.setattr(metadata_ui, "ROOT", str(tmp_path))
    srv = ThreadingHTTPServer(("127.0.0.1", 0), metadata_ui.Handler)
    port = srv.server_address[1]
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        yield f"http://127.0.0.1:{port}", tmp_path, monkeypatch
    finally:
        srv.shutdown()
        th.join()


class TestHTTPHandler:
    def test_index_serves_html(self, server):
        base, *_ = server
        status, body = _req(base + "/")
        assert status == 200 and b"<!doctype html" in body

    def test_api_list(self, server):
        base, *_ = server
        _, body = _req(base + "/api/list")
        data = json.loads(body)
        assert any(f["name"] == "lora.safetensors" for f in data["files"])

    def test_api_meta(self, server):
        base, *_ = server
        _, body = _req(base + "/api/meta?file=lora.safetensors")
        data = json.loads(body)
        titles = {f["key"]: f["value"] for f in data["known"]}
        assert titles["modelspec.title"] == "Hi"
        assert data["other"] == {"x": "1"}

    def test_api_meta_missing(self, server):
        base, *_ = server
        status, _ = _req(base + "/api/meta?file=nope.safetensors")
        assert status == 404

    def test_api_browse_cancel(self, server):
        base, _, monkeypatch = server
        monkeypatch.setattr(metadata_ui, "pick_file", lambda: "")
        _, body = _req(base + "/api/browse")
        assert json.loads(body) == {"path": None}

    def test_api_browse_picks(self, server):
        base, tmp_path, monkeypatch = server
        chosen = str(tmp_path / "lora.safetensors")
        monkeypatch.setattr(metadata_ui, "pick_file", lambda: chosen)
        _, body = _req(base + "/api/browse")
        assert json.loads(body)["path"].endswith("lora.safetensors")

    def test_api_browse_error(self, server):
        base, _, monkeypatch = server

        def boom():
            raise RuntimeError("no tk")

        monkeypatch.setattr(metadata_ui, "pick_file", boom)
        _, body = _req(base + "/api/browse")
        assert "error" in json.loads(body)

    def test_unknown_get_404(self, server):
        base, *_ = server
        status, _ = _req(base + "/nope")
        assert status == 404

    def test_save_dry_run(self, server):
        base, *_ = server
        payload = {
            "file": "lora.safetensors",
            "metadata": {"modelspec.title": "New"},
            "dry_run": True,
        }
        _, body = _req(base + "/api/save", "POST", payload)
        data = json.loads(body)
        assert (
            data["ok"]
            and data["dry_run"]
            and data["metadata"]["modelspec.title"] == "New"
        )

    def test_save_writes_edited(self, server):
        base, tmp_path, _ = server
        payload = {"file": "lora.safetensors", "metadata": {"modelspec.title": "New"}}
        _, body = _req(base + "/api/save", "POST", payload)
        assert json.loads(body)["output"] == "lora.edited.safetensors"
        assert (tmp_path / "lora.edited.safetensors").exists()

    def test_save_in_place_with_warning(self, server):
        base, tmp_path, _ = server
        payload = {
            "file": "lora.safetensors",
            "metadata": {"modelspec.prediction_type": "bogus"},  # off-spec -> warning
            "in_place": True,
        }
        _, body = _req(base + "/api/save", "POST", payload)
        assert json.loads(body)["warnings"]

    def test_save_missing_file(self, server):
        base, *_ = server
        status, _ = _req(
            base + "/api/save", "POST", {"file": "nope.safetensors", "metadata": {}}
        )
        assert status == 404

    def test_save_metadata_not_object(self, server):
        base, *_ = server
        status, _ = _req(
            base + "/api/save", "POST", {"file": "lora.safetensors", "metadata": "nope"}
        )
        assert status == 400

    def test_save_bad_json(self, server):
        base, *_ = server
        status, _ = _req(base + "/api/save", "POST", raw=b"{not json")
        assert status == 400

    def test_post_unknown_path_404(self, server):
        base, *_ = server
        status, _ = _req(base + "/api/other", "POST", {})
        assert status == 404


class TestMain:
    def test_serves_then_stops(self, tmp_path, monkeypatch, capsys):
        class FakeServer:
            def __init__(self, addr, handler):
                pass

            def serve_forever(self):
                raise KeyboardInterrupt

        monkeypatch.setattr(metadata_ui, "ThreadingHTTPServer", FakeServer)
        monkeypatch.setattr(metadata_ui.webbrowser, "open", lambda url: None)
        monkeypatch.setattr(sys, "argv", ["metadata_ui.py", "--dir", str(tmp_path)])
        metadata_ui.main()
        assert "stopped" in capsys.readouterr().out

    def test_bad_dir_exits(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            sys, "argv", ["metadata_ui.py", "--dir", str(tmp_path / "nope")]
        )
        with pytest.raises(SystemExit):
            metadata_ui.main()
