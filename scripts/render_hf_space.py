#!/usr/bin/env python3
"""Render the remaining beats through a hosted Wan 2.2 I2V Gradio Space.

The Kaggle account used by scripts/render_continuity.py is unavailable, so this
drives a public ZeroGPU Space instead. It keeps the same production contract:

- the compiled prompt from prompts/<shot>.txt is sent verbatim;
- the keyframe is the shot's start frame, or the previous clip's final frame
  when the shot is chained and a handoff exists;
- output is normalized to the native 832x480 / 16 fps / 81 frame target;
- renders/segments/<shot>.mp4 and renders/manifest.json are written in the same
  shape render_continuity.py uses, so stitching and resume keep working.

ZeroGPU enforces a per-call quota, so the script stops cleanly when the quota is
exhausted and can be re-run later to continue from the first missing segment.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import render_continuity as rc

SPACE_ID = "Saravutw/WAN2.2_I2V_LIGHTNING_4-8step_custom"
SPACE_HOST = "saravutw-wan2-2-i2v-lightning-4-8step-custom.hf.space"
GENERATE_URL = f"https://{SPACE_HOST}/gradio_api/call/generate_video"
SEGMENTS = ROOT / "renders" / "segments"
HANDOFFS = ROOT / "renders" / "handoffs"
STATE_PATH = ROOT / "renders" / "manifest.json"
NEGATIVE = (ROOT / "negatives" / "artifact_neg.txt").read_text(encoding="utf-8").strip()


def file_data(path: Path) -> dict:
    data = path.read_bytes()
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return {
        "path": None,
        "url": f"data:{mime};base64," + base64.b64encode(data).decode("ascii"),
        "size": len(data),
        "orig_name": path.name,
        "mime_type": mime,
        "is_stream": False,
        "meta": {"_type": "gradio.FileData"},
    }


def generate(image: Path, prompt: str, seed: int, steps: int, duration: float) -> str:
    payload = {
        "data": [
            file_data(image),
            None,
            prompt,
            steps,
            NEGATIVE[:900],
            duration,
            1.0,
            1.0,
            seed,
            False,
            8,
            "UniPCMultistep",
            3.0,
            16,
            False,
            True,
        ],
    }
    request = urllib.request.Request(
        GENERATE_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        event_id = json.loads(response.read()).get("event_id")
    if not event_id:
        raise RuntimeError("The Space did not return an event id")
    stream = urllib.request.Request(f"{GENERATE_URL}/{event_id}")
    with urllib.request.urlopen(stream, timeout=1800) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body in ("", "null"):
                continue
            parsed = json.loads(body)
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
                url = parsed[0].get("url")
                if url:
                    return url
    raise RuntimeError("The Space finished without returning a video")


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".part")
    with urllib.request.urlopen(url, timeout=600) as response:
        temporary.write_bytes(response.read())
    os.replace(temporary, destination)


def keyframe_for(shot_id: str, config: dict) -> tuple[Path, str]:
    override = config.get("start_frame_override")
    if isinstance(override, str) and override:
        path = (ROOT / override).resolve()
        if not path.is_file():
            raise RuntimeError(f"{shot_id}: start-frame override is missing at {override}")
        return path, override
    handoff = HANDOFFS / f"{shot_id}.png"
    if config.get("chain") and handoff.is_file():
        return handoff, handoff.relative_to(ROOT).as_posix()
    return (ROOT / config["start_frame"]).resolve(), config["start_frame"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_shot", default="c01")
    parser.add_argument("--to", dest="to_shot", default="c70")
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--pause", type=float, default=3.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    selected = rc.select_shots(args.from_shot, args.to_shot)
    durations = rc.load_shot_durations(selected)
    state = json.loads(STATE_PATH.read_text(encoding="utf-8")) if STATE_PATH.is_file() else {
        "schema_version": 1,
        "video_worker_id": f"hf-space:{SPACE_ID}",
        "video_profile": "hf-space-wan22-i2v-lightning",
        "jobs": {},
    }
    state.setdefault("jobs", {})

    for index, shot_id in enumerate(selected, 1):
        segment = SEGMENTS / f"{shot_id}.mp4"
        if segment.is_file() and rc.normalized_segment_matches(
            "ffprobe", segment, durations[shot_id],
        ):
            print(f"[{index}/{len(selected)}] {shot_id}: already rendered; skipping", flush=True)
            continue
        config = json.loads((ROOT / "prompts" / f"{shot_id}.json").read_text(encoding="utf-8"))
        prompt = rc.read_prompt(shot_id, json.loads(
            (ROOT / "startframes" / "manifest.json").read_text(encoding="utf-8"),
        )["jobs"][shot_id]["prompt_sha256"])
        image, source = keyframe_for(shot_id, config)
        seed = int(config.get("seed") or 0) or (abs(hash(shot_id)) % 2_000_000_000)
        print(
            f"[{index}/{len(selected)}] {shot_id} | ref={source} | "
            f"seed={seed} | {durations[shot_id]}s",
            flush=True,
        )
        if args.dry_run:
            continue
        started = time.monotonic()
        try:
            url = generate(image, prompt, seed, args.steps, float(durations[shot_id]))
        except urllib.error.HTTPError as error:
            detail = error.read()[:300].decode("utf-8", "replace")
            if error.code in (429, 402) or "quota" in detail.lower():
                print(f"{shot_id}: ZeroGPU quota exhausted; stopping. {detail}", flush=True)
                return 2
            raise
        raw = ROOT / "renders" / "raw" / f"{shot_id}.mp4"
        download(url, raw)
        rc.normalize_video("ffmpeg", raw, segment, durations[shot_id])
        probe = rc.probe_video("ffprobe", segment)
        stream = next(
            (item for item in probe.get("streams", []) if item.get("codec_type") == "video"),
            None,
        )
        actual = float(probe.get("format", {}).get("duration", 0))
        if not stream or (stream.get("width"), stream.get("height")) != (
            rc.RENDER_WIDTH, rc.RENDER_HEIGHT,
        ):
            raise RuntimeError(f"{shot_id}: normalized clip failed its dimension check")
        state["jobs"][shot_id] = {
            "shot_id": shot_id,
            "scene": config["scene"],
            "build": config["build"],
            "chain": config["chain"],
            "prompt_sha256": json.loads(
                (ROOT / "startframes" / "manifest.json").read_text(encoding="utf-8"),
            )["jobs"][shot_id]["prompt_sha256"],
            "start_frame": config["start_frame"],
            "keyframe_source": source,
            "video_status": "completed",
            "video_job_id": f"hf:{shot_id}:{seed}",
            "segment_sha256": rc.sha256(segment.read_bytes()),
            "segment_bytes": segment.stat().st_size,
            "duration_seconds": actual,
            "video_dimensions": [stream["width"], stream["height"]],
        }
        rc.save_json(STATE_PATH, state)
        rc.stage_next_handoff("ffmpeg", shot_id, segment)
        print(
            f"{shot_id}: rendered in {time.monotonic() - started:.1f}s "
            f"({actual:.2f}s, {stream['width']}x{stream['height']})",
            flush=True,
        )
        time.sleep(args.pause)

    print("Done. Stitch with scripts/render_continuity.py once every segment exists.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
