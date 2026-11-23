#!/usr/bin/env python3
"""Post prompts/*.json to a ComfyUI /prompt endpoint, in order.

Chaining: when a shot's cfg has chain=true and a previous clip exists, its
LoadImage is filled with the last frame of that clip instead of a file on disk.
Only chain=false shots and a fresh first shot read startframes/.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

from comfy_api import upload
from env_utils import resolve_server

CHAIN_DIR = Path("chained")
LORA_REGISTRY_PATH = Path("workflows") / "loras.json"


def load_lora_registry(path: Path = LORA_REGISTRY_PATH) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _sampler_branches(wf: dict) -> dict[str, dict]:
    """Map 'high'/'low' to {sampler_id, model_ref} for the two KSamplerAdvanced passes."""
    branches: dict[str, dict] = {}
    for node_id, node in wf.items():
        if not isinstance(node, dict) or node.get("class_type") != "KSamplerAdvanced":
            continue
        inputs = node.get("inputs", {})
        ref = inputs.get("model")
        title = ""
        if isinstance(ref, list) and len(ref) == 2:
            loader = wf.get(str(ref[0]))
            if isinstance(loader, dict):
                title = str(loader.get("_meta", {}).get("title", ""))
        if "high" in title.lower():
            branch = "high"
        elif "low" in title.lower():
            branch = "low"
        else:
            branch = "high" if inputs.get("add_noise") == "enable" else "low"
        branches[branch] = {"sampler_id": node_id, "model_ref": ref}
    return branches


def inject_loras(wf: dict, cfg: dict, registry: dict) -> dict:
    """Chain LoraLoaderModelOnly nodes into the high/low sampler branches.

    Registry null slots are honoured (that pass is skipped for that LoRA).
    Missing local lora files warn loudly but are never silently dropped.
    """
    loras = cfg.get("loras") or []
    if not loras:
        return wf
    branches = _sampler_branches(wf)
    next_id = max((int(k) for k in wf if str(k).isdigit()), default=0) + 1
    for key in loras:
        entry = registry.get(key)
        if not isinstance(entry, dict):
            print(
                f"WARNING: lora key {key!r} is not in workflows/loras.json; skipped",
                file=sys.stderr,
            )
            continue
        for stage in ("high", "low"):
            target = entry.get(stage)
            if target is None:
                continue
            if not Path(target).is_file():
                print(
                    f"WARNING [{key}] {stage}: {target} missing from the repo; "
                    "ensure it is staged in the ComfyUI models/loras directory",
                    file=sys.stderr,
                )
            branch = branches.get(stage)
            if branch is None:
                print(
                    f"WARNING [{key}] {stage}: no {stage}-noise KSamplerAdvanced in workflow; skipped",
                    file=sys.stderr,
                )
                continue
            node_id = str(next_id)
            next_id += 1
            wf[node_id] = {
                "class_type": "LoraLoaderModelOnly",
                "_meta": {"title": f"{key} {stage}"},
                "inputs": {
                    "model": branch["model_ref"],
                    "lora_name": str(target).split("/")[-1],
                    "strength_model": float(entry.get("w", 1.0)),
                },
            }
            branch["model_ref"] = [node_id, 0]
    for branch in branches.values():
        wf[branch["sampler_id"]].setdefault("inputs", {})["model"] = branch["model_ref"]
    return wf


def load_workflow(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def inject(
    wf: dict,
    prompt: str,
    negative: str,
    cfg: dict,
    image: str,
    registry: dict | None = None,
) -> dict:
    """Fill the TODO-marked nodes. Keys are stable by contract."""
    for node in wf.values():
        if not isinstance(node, dict):
            continue
        ct = node.get("class_type", "")
        inputs = node.setdefault("inputs", {})
        if ct == "CLIPTextEncode":
            marker = inputs.pop("__todo__", None)
            if marker == "positive":
                inputs["text"] = prompt
            elif marker == "negative":
                inputs["text"] = negative
        elif ct == "LoadImage":
            inputs["image"] = image
        elif ct in ("Wan22ImageToVideo", "WanImageToVideo"):
            inputs["width"] = cfg["width"]
            inputs["height"] = cfg["height"]
            # Project contract: 8 fps, 41 frames for a 5s clip and 33 for a
            # 4s clip. WanImageToVideo requires (length - 1) % 4 == 0, which
            # both values satisfy. Rendering at 16 fps doubled the frame count
            # and therefore the sampling cost for no authored benefit.
            inputs["length"] = 41 if cfg["duration"] >= 5 else 33
    if registry is None:
        registry = load_lora_registry()
    return inject_loras(wf, cfg, registry)


def post(server: str, wf: dict) -> str:
    req = urllib.request.Request(
        f"{server}/prompt",
        data=json.dumps({"prompt": wf}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())["prompt_id"]


def wait_for(server: str, pid: str, timeout: int = 3600) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        with urllib.request.urlopen(f"{server}/history/{pid}") as r:
            hist = json.loads(r.read())
        if pid in hist:
            return hist[pid]
        time.sleep(5)
    raise TimeoutError(f"{pid} did not finish in {timeout}s")


def download_output(server: str, entry: dict, dest: Path) -> Path:
    """Download the rendered video for a finished prompt to `dest`."""
    items = [
        item
        for node in entry.get("outputs", {}).values()
        for key in ("videos", "gifs")
        for item in (node.get(key) or [])
    ]
    if not items:
        raise RuntimeError("history entry carried no video output")
    item = items[-1]
    query = urllib.parse.urlencode(
        {
            "filename": item["filename"],
            "subfolder": item.get("subfolder", ""),
            "type": item.get("type", "output"),
        }
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"{server}/view?{query}") as r:
        dest.write_bytes(r.read())
    return dest


def last_frame(server: str, entry: dict, dest: Path) -> Path:
    items = [
        item
        for node in entry.get("outputs", {}).values()
        for key in ("videos", "gifs", "images")
        for item in (node.get(key) or [])
    ]
    if not items:
        raise RuntimeError("history entry carried no output files")
    item = items[-1]
    query = urllib.parse.urlencode(
        {
            "filename": item["filename"],
            "subfolder": item.get("subfolder", ""),
            "type": item.get("type", "output"),
        }
    )
    with urllib.request.urlopen(f"{server}/view?{query}") as r:
        raw = r.read()
    src = dest.with_suffix(".src" + Path(item["filename"]).suffix)
    src.write_bytes(raw)
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-sseof", "-0.1", "-i", str(src), "-frames:v", "1", str(dest)],
        check=True,
        capture_output=True,
    )
    src.unlink(missing_ok=True)
    return dest


def preceding_chained_frame(prompt: Path, cfg: dict) -> Path | None:
    if not cfg.get("chain"):
        return None
    number = _shot_number(prompt)
    if number <= 1:
        return None
    previous = CHAIN_DIR / f"c{number - 1:02d}.png"
    return previous if previous.is_file() else None


def _shot_number(prompt: Path) -> int:
    shot_number = prompt.stem[1:]
    if not prompt.stem.startswith("c") or not shot_number.isdigit():
        raise ValueError(f"invalid shot prompt id: {prompt.stem}")
    return int(shot_number)


def select_start_image(
    prompt: Path,
    cfg: dict,
    previous: Path | None,
    failed_shot: str | None,
    dry_run: bool = False,
) -> tuple[Path, bool]:
    chain_requested = bool(cfg.get("chain"))
    if chain_requested and failed_shot is not None:
        raise RuntimeError(
            f"{prompt.stem}: cannot chain because previous shot {failed_shot} failed"
        )

    chained = (
        chain_requested
        and previous is not None
        and (dry_run or previous.is_file())
    )
    if chained:
        return previous, True

    if chain_requested and _shot_number(prompt) > 1:
        print(f"{prompt.stem}: no prev frame, reseeded from start_frame")
    return Path(cfg["start_frame"]), False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=None)
    ap.add_argument("--workflow", default="workflows/base_i2v.json")
    ap.add_argument("--spec", default="docs/beats.yaml")
    ap.add_argument("--negative", default=None, help="override meta.negative")
    ap.add_argument("--from", dest="start", default="c01")
    ap.add_argument("--to", dest="end", default="c70")
    ap.add_argument(
        "--no-resume",
        action="store_true",
        help="do not start from the preceding clip's saved chained frame",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    args.server = resolve_server(args.server)

    if not args.dry_run and not shutil.which("ffmpeg"):
        print("ffmpeg not on PATH; needed to chain clips", file=sys.stderr)
        return 1

    meta = yaml.safe_load(Path(args.spec).read_text(encoding="utf-8"))["meta"]
    neg = Path(args.negative or meta["negative"]).read_text(encoding="utf-8")
    base = load_workflow(Path(args.workflow))
    todo = [
        p
        for p in sorted(Path("prompts").glob("c*.json"))
        if args.start <= p.stem <= args.end
    ]
    if not todo:
        print("no prompts in range", file=sys.stderr)
        return 1

    prev: Path | None = None
    if not args.no_resume:
        first_cfg = json.loads(todo[0].read_text(encoding="utf-8"))
        try:
            prev = preceding_chained_frame(todo[0], first_cfg)
        except ValueError as error:
            print(str(error), file=sys.stderr)
            return 1
    uploads: dict[Path, str] = {}
    failed_shot: str | None = None
    had_failures = False
    for p in todo:
        cfg = json.loads(p.read_text(encoding="utf-8"))
        try:
            image_path, chained = select_start_image(
                p,
                cfg,
                prev,
                failed_shot,
                dry_run=args.dry_run,
            )
        except (RuntimeError, ValueError) as error:
            print(str(error), file=sys.stderr)
            return 1
        if not image_path.is_file() and not args.dry_run:
            print(f"{p.stem}: no start frame at {image_path}; seed it first", file=sys.stderr)
            return 1

        body = Path("prompts", p.stem + ".txt").read_text(encoding="utf-8")

        if args.dry_run:
            print(f"{p.stem}: ok ({len(body)} chars, {'chain' if chained else 'seed'})")
            prev = CHAIN_DIR / f"{p.stem}.png"
            continue

        try:
            image = upload(args.server, image_path, uploads)
            wf = inject(json.loads(json.dumps(base)), body, neg, cfg, image)
            pid = post(args.server, wf)
            print(f"{p.stem} queued {pid}")
            entry = wait_for(args.server, pid)
            prev = last_frame(args.server, entry, CHAIN_DIR / f"{p.stem}.png")
            print(f"{p.stem} done -> {prev}")
            failed_shot = None
        except (
            OSError,
            RuntimeError,
            ValueError,
            KeyError,
            subprocess.SubprocessError,
        ) as error:
            print(f"{p.stem}: render failed: {error}", file=sys.stderr)
            prev = None
            failed_shot = p.stem
            had_failures = True
    return 1 if had_failures else 0


if __name__ == "__main__":
    sys.exit(main())