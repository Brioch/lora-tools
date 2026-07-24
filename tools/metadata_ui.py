#!/usr/bin/env python3
# /// script
# requires-python = ">=3.13"
# dependencies = []
# ///
"""A tiny local web UI to view and edit a LoRA's safetensors metadata.

Runs a local-only HTTP server (stdlib, no dependencies) that lists the
.safetensors files in a directory, shows their __metadata__, and lets you edit
the ModelSpec fields (and add/remove arbitrary keys) with a Save button. It
drives the same header-rewrite as tools/edit_metadata.py, so the tensor buffer
is copied verbatim and nothing is ever loaded into memory.

A "Browse…" button opens your machine's native file-open dialog (via stdlib
tkinter) so you can edit any .safetensors on disk, not just those in --dir.
That needs a tkinter-capable Python and a display; the --dir listing is the
fallback when the dialog isn't available.

Usage:
    python tools/metadata_ui.py                     # serve the current dir
    python tools/metadata_ui.py --dir /path/to/loras --port 8080
    python tools/metadata_ui.py --no-browser        # don't auto-open a browser

The server binds to 127.0.0.1 only and refuses any path outside --dir.
"""

import argparse
import json
import os
import re
import subprocess
import sys
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Reuse the CLI's field table, validators and header-rewrite (same directory).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edit_metadata import FIELDS, read_header, rewrite, validate  # noqa: E402

KNOWN_KEYS = {key for _, key, _ in FIELDS}

# Serving directory, resolved absolute; set in main().
ROOT = os.getcwd()


def list_files():
    out = []
    for name in sorted(os.listdir(ROOT)):
        path = os.path.join(ROOT, name)
        if name.endswith(".safetensors") and os.path.isfile(path):
            out.append(
                {
                    "name": name,
                    "path": os.path.realpath(path),
                    "size": os.path.getsize(path),
                }
            )
    return out


def safe_path(name):
    """Resolve a user-supplied path to an existing .safetensors file.

    Accepts an absolute path (as returned by the native Browse dialog) or a
    name relative to ROOT (from the default listing). The server binds to
    127.0.0.1 only and the picker is user-driven, so any local file the user
    points at is fair game — the same trust model as the sibling CLIs.
    """
    if not name or not name.endswith(".safetensors"):
        return None
    base = name if os.path.isabs(name) else os.path.join(ROOT, name)
    path = os.path.realpath(base)
    if not os.path.isfile(path):
        return None
    return path


# Runs in a throwaway subprocess so Tk never touches an HTTP worker thread.
# argv[1] is the initial directory; prints the chosen path (empty on cancel).
DIALOG_SCRIPT = r"""
import sys
import tkinter
from tkinter import filedialog
root = tkinter.Tk()
root.withdraw()
root.attributes("-topmost", True)
path = filedialog.askopenfilename(
    initialdir=sys.argv[1] if len(sys.argv) > 1 else ".",
    title="Select a LoRA",
    filetypes=[("Safetensors", "*.safetensors"), ("All files", "*")])
root.destroy()
sys.stdout.write(path or "")
"""


def pick_file():
    """Open a native file-open dialog; return the chosen path, "" on cancel.

    Raises on failure (no tkinter, no display) so the caller can report it.
    """
    proc = subprocess.run(
        [sys.executable, "-c", DIALOG_SCRIPT, ROOT], capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise RuntimeError((proc.stderr or "dialog failed").strip().splitlines()[-1])
    return proc.stdout.strip()


def read_meta(path):
    _, header = read_header(path)
    return header.get("__metadata__", {}) or {}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # keep the console quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/?"):
            return self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        if self.path == "/api/list":
            return self._send(200, {"dir": ROOT, "files": list_files()})
        if self.path == "/api/browse":
            try:
                path = pick_file()
            except Exception as e:
                return self._send(200, {"error": f"file dialog unavailable: {e}"})
            if not path:
                return self._send(200, {"path": None})
            if not path.endswith(".safetensors"):
                return self._send(200, {"error": "not a .safetensors file"})
            return self._send(200, {"path": os.path.realpath(path)})
        if self.path.startswith("/api/meta"):
            name = self._query().get("file")
            path = safe_path(name)
            if not path:
                return self._send(404, {"error": "file not found"})
            meta = read_meta(path)
            known = [
                {"key": k, "flag": f, "help": h, "value": meta.get(k, "")}
                for f, k, h in FIELDS
            ]
            other = {k: v for k, v in meta.items() if k not in KNOWN_KEYS}
            return self._send(200, {"file": name, "known": known, "other": other})
        return self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/api/save":
            return self._send(404, {"error": "not found"})
        try:
            length = int(self.headers.get("Content-Length", 0))
            payload = json.loads(self.rfile.read(length) or b"{}")
        except Exception as e:
            return self._send(400, {"error": f"bad request: {e}"})

        path = safe_path(payload.get("file"))
        if not path:
            return self._send(404, {"error": "file not found"})

        meta = payload.get("metadata")
        if not isinstance(meta, dict):
            return self._send(400, {"error": "metadata must be an object"})
        # Drop empty values (an empty field means "unset"); stringify the rest.
        new_meta = {str(k): str(v) for k, v in meta.items() if str(v) != ""}

        warnings = [w for w in (validate(k, v) for k, v in new_meta.items()) if w]

        if payload.get("dry_run"):
            return self._send(
                200,
                {
                    "ok": True,
                    "dry_run": True,
                    "metadata": new_meta,
                    "warnings": warnings,
                },
            )

        if payload.get("in_place"):
            out = path
            tmp = path + ".tmp"
            rewrite(path, tmp, new_meta)
            os.replace(tmp, out)
        else:
            out = re.sub(r"\.safetensors$", "", path) + ".edited.safetensors"
            rewrite(path, out, new_meta)
        return self._send(
            200, {"ok": True, "output": os.path.basename(out), "warnings": warnings}
        )

    def _query(self):
        from urllib.parse import parse_qs, urlparse

        q = parse_qs(urlparse(self.path).query)
        return {k: v[0] for k, v in q.items()}


PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LoRA metadata editor</title>
<style>
  :root { color-scheme: light dark; --bg:#fff; --fg:#1a1a1a; --mut:#666;
    --line:#ddd; --card:#f7f7f8; --accent:#4f46e5; --accent-fg:#fff; --danger:#c0392b; }
  @media (prefers-color-scheme: dark) { :root { --bg:#161618; --fg:#e8e8ea;
    --mut:#9a9aa2; --line:#33333a; --card:#202024; --accent:#7c74ff; } }
  * { box-sizing: border-box; }
  body { margin:0; font:14px/1.5 system-ui, sans-serif; color:var(--fg); background:var(--bg);
    display:grid; grid-template-columns:260px 1fr; height:100vh; }
  aside { border-right:1px solid var(--line); overflow:auto; padding:12px; }
  aside h1 { font-size:13px; text-transform:uppercase; letter-spacing:.05em; color:var(--mut); margin:4px 4px 10px; }
  .browse { width:100%; background:var(--accent); color:var(--accent-fg); border:none;
    border-radius:8px; padding:9px 12px; font:inherit; font-weight:600; cursor:pointer; margin-bottom:12px; }
  .file { padding:8px 10px; border-radius:8px; cursor:pointer; }
  .file:hover { background:var(--card); }
  .file.active { background:var(--accent); color:var(--accent-fg); }
  .file small { display:block; opacity:.7; }
  main { overflow:auto; padding:20px 24px; }
  .empty { color:var(--mut); margin-top:40px; text-align:center; }
  h2 { margin:0 0 4px; font-size:18px; word-break:break-all; }
  .sub { color:var(--mut); margin-bottom:18px; }
  .field { margin-bottom:12px; }
  label { display:block; font-weight:600; margin-bottom:3px; }
  label .k { font-weight:400; color:var(--mut); margin-left:6px; font-size:12px; }
  input, textarea { width:100%; padding:7px 9px; border:1px solid var(--line);
    border-radius:7px; background:var(--bg); color:var(--fg); font:inherit; }
  textarea { resize:vertical; min-height:52px; }
  .row { display:flex; gap:8px; align-items:center; margin-bottom:8px; }
  .row input.k { flex:0 0 40%; } .row input.v { flex:1; }
  .row button, .del { background:none; border:1px solid var(--line); color:var(--danger);
    border-radius:7px; padding:6px 10px; cursor:pointer; }
  h3 { margin:24px 0 10px; font-size:13px; text-transform:uppercase; letter-spacing:.05em; color:var(--mut); }
  .bar { position:sticky; bottom:0; background:var(--bg); border-top:1px solid var(--line);
    padding:14px 0 4px; margin-top:20px; display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
  button.primary { background:var(--accent); color:var(--accent-fg); border:none;
    padding:9px 16px; border-radius:8px; font-weight:600; cursor:pointer; }
  button.ghost { background:none; border:1px solid var(--line); color:var(--fg);
    padding:9px 14px; border-radius:8px; cursor:pointer; }
  label.chk { display:flex; align-items:center; gap:6px; font-weight:400; }
  label.chk input { width:auto; }
  .msg { margin-left:auto; color:var(--mut); }
  .msg.ok { color:#2e7d32; } .msg.err { color:var(--danger); }
  .warn { color:#b8860b; font-size:13px; margin-top:8px; white-space:pre-wrap; }
  .addbtn { background:none; border:1px dashed var(--line); color:var(--fg);
    border-radius:7px; padding:7px 12px; cursor:pointer; }
</style></head>
<body>
<aside><button class="browse" id="browse">Browse…</button><h1>Files</h1><div id="files"></div></aside>
<main id="main"><div class="empty">Select a file on the left.</div></main>
<script>
let current = null;      // absolute path of the selected file
let picked = [];         // files chosen via Browse, kept across refreshes
const fmtSize = b => b > 1e6 ? (b/1048576).toFixed(1)+' MB' : (b/1024).toFixed(0)+' KB';
const base = p => p.replace(/\/+$/,'').split(/[\\/]/).pop();
const dir  = p => p.slice(0, p.length - base(p).length).replace(/[\\/]+$/,'') || '/';
const el = (t, props={}, ...kids) => { const e=document.createElement(t);
  Object.assign(e, props); for (const k of kids) e.append(k); return e; };

async function loadList() {
  const r = await (await fetch('/api/list')).json();
  document.title = 'LoRA metadata — ' + r.dir;
  const seen = new Set();
  // ROOT listing first, then Browse-picked files not already in the listing.
  const files = [];
  for (const f of [...r.files, ...picked]) {
    if (seen.has(f.path)) continue;
    seen.add(f.path); files.push(f);
  }
  const box = document.getElementById('files'); box.innerHTML='';
  if (!files.length) box.append(el('div',{className:'empty',textContent:'No .safetensors here — use Browse…'}));
  for (const f of files) {
    const sub = f.size != null ? fmtSize(f.size) : dir(f.path);
    const d = el('div',{className:'file'}, el('span',{textContent:base(f.path)}), el('small',{textContent:sub}));
    d.title = f.path;
    d.onclick = () => selectFile(f.path, d);
    if (f.path === current) d.classList.add('active');
    box.append(d);
  }
}

async function selectFile(path, node) {
  document.querySelectorAll('.file').forEach(n=>n.classList.remove('active'));
  node.classList.add('active');
  current = path;
  const data = await (await fetch('/api/meta?file='+encodeURIComponent(path))).json();
  render(data);
}

async function browse() {
  const btn = document.getElementById('browse');
  btn.disabled = true; const label = btn.textContent; btn.textContent = 'Opening…';
  try {
    const r = await (await fetch('/api/browse')).json();
    if (r.error) { alert(r.error); return; }
    if (!r.path) return;              // dialog cancelled
    if (!picked.some(f => f.path === r.path)) picked.push({name: base(r.path), path: r.path});
    current = r.path;
    await loadList();
    const node = [...document.querySelectorAll('.file')].find(n => n.title === r.path);
    if (node) selectFile(r.path, node);
  } finally { btn.disabled = false; btn.textContent = label; }
}
document.getElementById('browse').onclick = browse;

function render(data) {
  const m = document.getElementById('main'); m.innerHTML='';
  m.append(el('h2',{textContent:base(data.file)}));
  m.append(el('div',{className:'sub',textContent:data.file}));
  m.append(el('div',{className:'sub',textContent:'ModelSpec fields — https://github.com/Stability-AI/ModelSpec'}));

  for (const f of data.known) {
    const wrap = el('div',{className:'field'});
    const lab = el('label',{textContent:f.flag}); lab.append(el('span',{className:'k',textContent:f.key}));
    lab.title = f.help;
    const big = /description|usage_hint|license/.test(f.key);
    const inp = el(big?'textarea':'input',{value:f.value}); inp.dataset.key=f.key; inp.className='known';
    inp.placeholder = f.help;
    wrap.append(lab, inp); m.append(wrap);
  }

  m.append(el('h3',{textContent:'Other metadata'}));
  const other = el('div',{id:'other'});
  for (const [k,v] of Object.entries(data.other)) other.append(otherRow(k,v));
  m.append(other);
  const add = el('button',{className:'addbtn',textContent:'+ add key'});
  add.onclick = () => other.append(otherRow('',''));
  m.append(add);

  const bar = el('div',{className:'bar'});
  const inPlace = el('input',{type:'checkbox',id:'inplace'});
  const chk = el('label',{className:'chk'}, inPlace, document.createTextNode('overwrite in place'));
  const save = el('button',{className:'primary',textContent:'Save'});
  const dry  = el('button',{className:'ghost',textContent:'Preview (dry-run)'});
  const msg  = el('span',{className:'msg',id:'msg'});
  save.onclick = () => doSave(false);
  dry.onclick  = () => doSave(true);
  bar.append(save, dry, chk, msg); m.append(bar);
}

function otherRow(k,v) {
  const row = el('div',{className:'row'});
  const ki = el('input',{className:'k',value:k,placeholder:'key'});
  const vi = el('input',{className:'v',value:v,placeholder:'value'});
  const del = el('button',{textContent:'✕',title:'remove'});
  del.onclick = () => row.remove();
  row.append(ki,vi,del); return row;
}

function collect() {
  const meta = {};
  document.querySelectorAll('.known').forEach(i => { if (i.value !== '') meta[i.dataset.key]=i.value; });
  document.querySelectorAll('#other .row').forEach(r => {
    const k = r.querySelector('.k').value.trim(), v = r.querySelector('.v').value;
    if (k) meta[k]=v;
  });
  return meta;
}

async function doSave(dry) {
  const msg = document.getElementById('msg'); msg.className='msg'; msg.textContent='working…';
  const body = { file: current, metadata: collect(),
    in_place: document.getElementById('inplace').checked, dry_run: dry };
  const r = await (await fetch('/api/save',{method:'POST',headers:{'Content-Type':'application/json'},
    body: JSON.stringify(body)})).json();
  if (r.error) { msg.className='msg err'; msg.textContent='error: '+r.error; return; }
  if (dry) { msg.className='msg ok'; msg.textContent='dry-run OK — '+Object.keys(r.metadata).length+' keys'; }
  else { msg.className='msg ok'; msg.textContent='saved → '+r.output; loadList(); }
  if (r.warnings && r.warnings.length) {
    let w = document.getElementById('warnbox');
    if (!w) { w = el('div',{className:'warn',id:'warnbox'}); document.querySelector('.bar').after(w); }
    w.textContent = '⚠ ' + r.warnings.join('\n⚠ ');
  }
}

loadList();
</script>
</body></html>
"""


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--dir", default=".", help="directory of .safetensors files (default: cwd)"
    )
    ap.add_argument(
        "--host", default="127.0.0.1", help="bind host (default: 127.0.0.1)"
    )
    ap.add_argument("--port", type=int, default=8760, help="bind port (default: 8760)")
    ap.add_argument(
        "--no-browser", action="store_true", help="don't auto-open a browser"
    )
    args = ap.parse_args()

    global ROOT
    ROOT = os.path.realpath(args.dir)
    if not os.path.isdir(ROOT):
        sys.exit(f"error: not a directory: {args.dir}")

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    url = f"http://{args.host}:{args.port}"
    print(f"serving {ROOT}\nopen {url}  (Ctrl-C to stop)")
    if not args.no_browser:
        try:
            webbrowser.open(url)
        except Exception:
            pass
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
