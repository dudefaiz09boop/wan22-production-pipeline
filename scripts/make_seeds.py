#!/usr/bin/env python3
"""Render seed start frames for chain=false shots, from character refs.

Usage:
    uv run python scripts/make_seeds.py                  # c01 c21 c30 c47
    uv run python scripts/make_seeds.py c30 c47          # named shots only
    uv run python scripts/make_seeds.py --server http://10.0.0.5:8188
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from pathlib import Path

import yaml

from comfy_api import upload
from env_utils import resolve_server

SPEC = Path("docs/beats.yaml")
SEED_WORKFLOW = Path("workflows/seed_t2i.json")
DEFAULT_SHOTS = ["c01", "c21", "c30", "c47"]


def post(server: str, wf: dict) -> str:
    req = urllib.request.Request(
        f"{server}/prompt",
        data=json.dumps({"prompt": wf}).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())["prompt_id"]
    except urllib.error.HTTPError as exc:
        raise SystemExit(f"ComfyUI rejected the workflow: {exc.read().decode()}")


def wait(server: str, pid: str, timeout: int = 900) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with urllib.request.urlopen(f"{server}/history/{pid}") as r:
            hist = json.loads(r.read())
        if pid in hist:
            return hist[pid]
        time.sleep(5)
    raise SystemExit(f"{pid} did not finish in {timeout}s")


def refs_for(shot: dict, refs: dict[str, list[str]]) -> list[str]:
    """Refs ordered by who is visible in cam[0], then by speaker."""
    names = [n for n in refs if n in shot["cam"][0]]
    speaker = shot.get("speaker")
    if speaker and speaker not in names:
        names.append(speaker)
    return [p for n in names for p in refs[n]]


def fill(base: dict, shot: dict, meta: dict, negative: str, pool: list[str]) -> dict:
    wf = json.loads(json.dumps(base))
    for node in wf.values():
        if not isinstance(node, dict):
            continue
        ct = node.get("class_type", "")
        inputs = node.setdefault("inputs", {})
        metadata = node.get("_meta", {})
        title = metadata.get("title") if isinstance(metadata, dict) else None
        todo = inputs.pop("__todo__", None)
        if ct == "CLIPTextEncode" and (todo == "positive" or title == "__todo__"):
            inputs[metadata.get("text_input", "text")] = (
                shot["cam"][0].split(":", 1)[-1].strip()
            )
        elif ct == "CLIPTextEncode" and (
            todo == "negative" or title == "__negative__"
        ):
            inputs[metadata.get("text_input", "text")] = negative
        elif ct == "LoadImage" and (
            inputs.pop("__ref__", None) is not None or title == "__ref__"
        ):
            inputs[metadata.get("ref_input", "image")] = pool.pop(0) if pool else ""
        elif ct == "EmptyLatentImage":
            inputs[metadata.get("width_input", "width")], inputs[
                metadata.get("height_input", "height")
            ] = meta["res_base"]
        elif ct == "KSampler":
            inputs[metadata.get("seed_input", "seed")] = (
                zlib.crc32(shot["id"].encode()) & 0x7FFFFFFF
            )
    return wf


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("shots", nargs="*", default=DEFAULT_SHOTS)
    ap.add_argument("--server", default=None)
    args = ap.parse_args()
    args.server = resolve_server(args.server)

    spec = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    meta = spec["meta"]
    negative = Path(meta["negative"]).read_text(encoding="utf-8").strip()
    refs = {c["name"]: c.get("refs", []) for c in spec["story"]["characters"]}
    shots = {s["id"]: s for s in spec["shots"]}
    base = json.loads(SEED_WORKFLOW.read_text(encoding="utf-8"))

    unknown = [s for s in args.shots if s not in shots]
    if unknown:
        raise SystemExit(f"no such shot: {', '.join(unknown)}")

    Path("startframes").mkdir(exist_ok=True)
    uploads: dict[Path, str] = {}

    for sid in args.shots:
        shot = shots[sid]
        pool = refs_for(shot, refs)
        if not pool:
            raise SystemExit(f"{sid}: no refs resolved from cam text or speaker")
        count = len(pool)
        uploaded_refs = [upload(args.server, Path(ref), uploads) for ref in pool]
        wf = fill(base, shot, meta, negative, uploaded_refs)

        pid = post(args.server, wf)
        entry = wait(args.server, pid)
        images = [i for n in entry["outputs"].values() for i in (n.get("images") or [])]
        if not images:
            raise SystemExit(f"{sid}: render produced no image")
        item = images[-1]
        query = urllib.parse.urlencode(
            {
                "filename": item["filename"],
                "subfolder": item.get("subfolder", ""),
                "type": item.get("type", "output"),
            }
        )
        with urllib.request.urlopen(f"{args.server}/view?{query}") as r:
            Path("startframes", f"{sid}.png").write_bytes(r.read())
        print(f"{sid} seeded from {count} ref(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())