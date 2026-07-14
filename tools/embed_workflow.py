#!/usr/bin/env python3
"""Embed a ComfyUI workflow JSON into a WebP image so it can be dragged onto the
ComfyUI canvas to load the graph.

ComfyUI's front-end reads WebP metadata from the EXIF block: it walks the EXIF
string entries and splits each on the first ':', looking for a "workflow" (and
optionally "prompt") key. This script writes exactly that, using the same tags
ComfyUI itself uses when saving WebP (Make = workflow, Model = prompt).

Examples:
    # Embed a workflow into an image, writing <image>.embed.webp
    python embed_workflow.py screenshot.png workflow.json

    # Overwrite the source image in place.
    python embed_workflow.py preview.webp workflow.json -o preview.webp

    # Also embed an API-format prompt (optional).
    python embed_workflow.py shot.webp workflow.json --prompt prompt.json

    # Check what a WebP already has embedded.
    python embed_workflow.py preview.webp --read

Any Pillow-readable image works as input (PNG, JPG, WebP, ...); the output is
always WebP. Requires Pillow (present in a ComfyUI environment):
    pip install pillow
"""
import argparse
import json
import os
import sys

# EXIF tag IDs ComfyUI uses for WebP metadata.
TAG_MAKE = 0x010F   # ComfyUI stores extra_pnginfo entries here, e.g. "workflow:..."
TAG_MODEL = 0x0110  # ComfyUI stores the API prompt here, e.g. "prompt:..."


def read_embedded(path):
    """Return {key: value_str} of the ComfyUI-style metadata in a WebP's EXIF."""
    from PIL import Image
    exif = Image.open(path).getexif()
    found = {}
    for tag in (TAG_MAKE, TAG_MODEL):
        val = exif.get(tag)
        if isinstance(val, str) and ":" in val:
            key, _, payload = val.partition(":")
            found[key] = payload
    return found


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("image", help="input image (any Pillow-readable format)")
    ap.add_argument("workflow", nargs="?", help="workflow .json to embed (omit with --read)")
    ap.add_argument("-o", "--output", help="output .webp path (default: <image>.embed.webp)")
    ap.add_argument("--prompt", metavar="JSON", help="optional API-format prompt .json to also embed")
    ap.add_argument("--lossless", dest="lossless", action="store_true", default=True,
                    help="save lossless WebP (default: on; keeps screenshots crisp)")
    ap.add_argument("--lossy", dest="lossless", action="store_false",
                    help="save lossy WebP instead")
    ap.add_argument("--quality", type=int, default=90,
                    help="WebP quality 0-100 (default: 90; effort level when lossless)")
    ap.add_argument("--read", action="store_true",
                    help="just print the metadata already embedded in <image> and exit")
    args = ap.parse_args()

    try:
        from PIL import Image
    except ImportError:
        sys.exit("error: this script needs Pillow (pip install pillow)")

    if not os.path.isfile(args.image):
        sys.exit(f"error: no such file: {args.image}")

    if args.read:
        found = read_embedded(args.image)
        if not found:
            print(f"{args.image}: no ComfyUI workflow/prompt metadata found.")
            return
        for key, payload in found.items():
            try:
                n = len(json.loads(payload).get("nodes", [])) if key == "workflow" else None
            except (ValueError, AttributeError):
                n = None
            extra = f" ({n} nodes)" if n is not None else ""
            print(f"{args.image}: '{key}' present, {len(payload)} chars{extra}")
        return

    if not args.workflow:
        ap.error("workflow .json is required unless --read is given")
    if not os.path.isfile(args.workflow):
        sys.exit(f"error: no such file: {args.workflow}")

    with open(args.workflow) as f:
        workflow = json.load(f)
    if not (isinstance(workflow, dict) and "nodes" in workflow):
        print("warning: this JSON has no top-level 'nodes' key — it may be an API "
              "prompt rather than a UI workflow; ComfyUI won't load it by drag-drop.",
              file=sys.stderr)

    img = Image.open(args.image)
    img.load()

    exif = img.getexif()
    exif[TAG_MAKE] = "workflow:" + json.dumps(workflow)
    if args.prompt:
        if not os.path.isfile(args.prompt):
            sys.exit(f"error: no such file: {args.prompt}")
        with open(args.prompt) as f:
            prompt = json.load(f)
        exif[TAG_MODEL] = "prompt:" + json.dumps(prompt)

    out = args.output or os.path.splitext(args.image)[0] + ".embed.webp"
    save_kwargs = {"exif": exif.tobytes()}
    if args.lossless:
        save_kwargs.update(lossless=True, quality=args.quality)  # quality = effort when lossless
    else:
        save_kwargs.update(quality=args.quality)
    img.save(out, format="WEBP", **save_kwargs)

    n = len(workflow["nodes"]) if isinstance(workflow, dict) and "nodes" in workflow else "?"
    print(f"embedded workflow ({n} nodes){' + prompt' if args.prompt else ''} -> {out}")
    print(f"  size: {img.size[0]}x{img.size[1]}, {os.path.getsize(out)} bytes. "
          f"Drag it onto the ComfyUI canvas to load.")


if __name__ == "__main__":
    main()
