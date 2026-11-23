#!/usr/bin/env python3
"""Stage or submit post-hoc still-frame inpaint passes for rendered beats.

The named targets are the c26 tail and the c28 thighs, but beat ids are plain
arguments; nothing about specific beats is hardcoded. Frames are extracted from
renders/segments/{beat}.mp4 with ffmpeg. With --stage-only (or no gateway) the
extracted frames and a job manifest are written under renders/inpaint/ for
manual passes. With a gateway base URL (--gateway or STUDIO_API_URL) the
staged frames are submitted as still-image Krea-2 inpaint jobs to the local
Studio; video is never submitted. Authored prompt text is never modified.

Examples:
  uv run python scripts/inpaint_pass.py c26 c28 --stage-only
  uv run python scripts/inpaint_pass.py c28 --gateway http://127.0.0.1:3000 \
      --prompt "repair the thigh region, preserve identity and lighting" \
      --region "thigh:900,430,340,260"
"""
from __future__ import annotations

import argparse
import base64
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

import render_continuity

ROOT = Path(__file__).resolve().parents[1]


def parse_regions(values: list[str]) -> list[dict]:
    """Parse --region entries shaped like label:x,y,w,h into dicts."""
    regions = []
    for value in values:
        label, separator, coordinates = value.partition(":")
        if not separator:
            raise ValueError(f"Region {value!r} must look like label:x,y,w,h")
        parts = [part.strip() for part in coordinates.split(",")]
        if len(parts) != 4 or any(not part.lstrip("-").isdigit() for part in parts):
            raise ValueError(f"Region {value!r} must have integer x,y,w,h")
        regions.append({
            "label": label.strip() or "region",
            "x": int(parts[0]),
            "y": int(parts[1]),
            "width": int(parts[2]),
            "height": int(parts[3]),
        })
    return regions


def frame_name(beat_id: str, at: float | None) -> str:
    stamp = "final" if at is None else f"t{str(at).replace('.', 'p')}s"
    return f"{beat_id}_{stamp}.png"


def extract_frames(
    ffmpeg: str,
    segment: Path,
    times: list[float],
    out_dir: Path,
    beat_id: str,
) -> list[dict]:
    """Extract the requested frames; an empty times list takes the final frame."""
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    requests = [(time, out_dir / frame_name(beat_id, time)) for time in times]
    if not times:
        requests = [(None, out_dir / frame_name(beat_id, None))]
    for at, destination in requests:
        destination.unlink(missing_ok=True)
        command = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
        if at is None:
            command.extend(["-sseof", "-0.05"])
        else:
            command.extend(["-ss", str(at)])
        command.extend(["-i", str(segment), "-frames:v", "1", str(destination)])
        render_continuity.run_media(command, timeout=300)
        if not destination.is_file() or destination.stat().st_size == 0:
            raise RuntimeError(f"{beat_id}: ffmpeg produced no frame at {at or 'end'}")
        width, height = render_continuity.png_dimensions(destination.read_bytes())
        frames.append({
            "path": destination.relative_to(ROOT).as_posix(),
            "time_seconds": at,
            "width": width,
            "height": height,
        })
    return frames


def build_manifest(
    beat_id: str,
    segment: Path,
    frames: list[dict],
    regions: list[dict],
    status: str,
) -> dict:
    return {
        "schema_version": 1,
        "beat_id": beat_id,
        "segment": segment.relative_to(ROOT).as_posix(),
        "frames": frames,
        "regions": regions,
        "status": status,
        "jobs": [],
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }


def save_manifest(out_dir: Path, manifest: dict) -> Path:
    path = out_dir / "manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path


def resolve_gateway(args) -> str | None:
    candidate = args.gateway or os.environ.get("STUDIO_API_URL", "").strip()
    if not candidate:
        return None
    return render_continuity.validate_api_origin(candidate)


def resolve_api_key() -> str:
    values = dict(render_continuity.read_env_file())
    values.update({key: value for key, value in os.environ.items() if value})
    if len(values.get("API_KEY", "")) < 32:
        raise RuntimeError("API_KEY is missing or invalid; gateway submission needs it")
    return values["API_KEY"]


def pick_image_worker(api: str, api_key: str) -> dict:
    response = render_continuity.api_json(api, api_key, "/v1/workers/active")
    matches = [
        worker for worker in response.get("data", [])
        if isinstance(worker, dict)
        and worker.get("kind") == "image"
        and worker.get("status") == "idle"
    ]
    if not matches:
        raise RuntimeError("No idle image worker is available for inpaint jobs")
    return matches[0]


def submit_frames(
    api: str,
    api_key: str,
    worker: dict,
    manifest: dict,
    prompt: str,
    negative_prompt: str,
    reference_strength: float | None,
) -> list[dict]:
    """Submit each staged frame as a still-image inpaint job; returns job rows."""
    jobs = []
    for frame in manifest["frames"]:
        frame_path = ROOT / frame["path"]
        data = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": frame["width"],
            "height": frame["height"],
            "inpaint": True,
            "reference_base64": base64.b64encode(frame_path.read_bytes()).decode("ascii"),
            "reference_filename": Path(frame["path"]).name,
        }
        if reference_strength is not None:
            data["reference_strength"] = reference_strength
        response = render_continuity.api_json(
            api,
            api_key,
            f"/v1/workers/{worker['id']}/jobs",
            data=data,
        )
        jobs.append({
            "frame": frame["path"],
            "task_id": response["id"],
            "poll_url": response.get("poll_url"),
            "width": frame["width"],
            "height": frame["height"],
        })
        print(f"{manifest['beat_id']}: inpaint job {response['id']} queued", flush=True)
    return jobs


def main(argv: list[str] | None = None) -> int:
    global ROOT
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("beats", nargs="+", help="beat ids, e.g. c26 c28")
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="project root (defaults to the repository)",
    )
    parser.add_argument(
        "--segments-dir",
        type=Path,
        default=None,
        help="rendered segments directory (default ROOT/renders/segments)",
    )
    parser.add_argument(
        "--at",
        default="",
        help="comma-separated seconds to extract, e.g. 3.2,4.8; default: final frame",
    )
    parser.add_argument(
        "--region",
        action="append",
        default=[],
        metavar="LABEL:X,Y,W,H",
        help="region hint recorded in the manifest, repeatable",
    )
    parser.add_argument(
        "--gateway",
        default=None,
        help="loopback Studio base URL for still inpaint jobs (env STUDIO_API_URL)",
    )
    parser.add_argument(
        "--stage-only",
        action="store_true",
        help="write frames + manifest only, even if a gateway is configured",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="inpaint instruction required for submission; authored text is untouched",
    )
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument(
        "--reference-strength",
        type=float,
        default=None,
        help="Krea-2 reference denoise override (0.2-0.9; default gateway-side 0.25)",
    )
    parser.add_argument("--ffmpeg", default=os.environ.get("FFMPEG_PATH", "ffmpeg"))
    args = parser.parse_args(argv)
    if args.reference_strength is not None and not 0.2 <= args.reference_strength <= 0.9:
        parser.error("--reference-strength must be between 0.2 and 0.9")

    ROOT = args.root.resolve()
    render_continuity.ROOT = ROOT

    try:
        beat_ids = [render_continuity.normalize_shot_id(beat) for beat in args.beats]
        regions = parse_regions(args.region)
    except ValueError as error:
        parser.error(str(error))

    times: list[float] = []
    if args.at:
        try:
            times = [float(part) for part in args.at.split(",") if part.strip()]
        except ValueError:
            parser.error("--at must be comma-separated numbers")
        if any(time < 0 for time in times):
            parser.error("--at seconds must be non-negative")

    segments_dir = (args.segments_dir or (ROOT / "renders" / "segments")).resolve()
    gateway = resolve_gateway(args)
    if gateway is None and not args.stage_only:
        # No gateway configured: staging is the safe default.
        args.stage_only = True
    if gateway is not None and not args.stage_only and not args.prompt:
        parser.error("submitting to the gateway requires --prompt")

    failures = 0
    for beat_id in beat_ids:
        segment = segments_dir / f"{beat_id}.mp4"
        if not segment.is_file():
            print(f"{beat_id}: no rendered segment at {segment.name}; skipped", flush=True)
            failures += 1
            continue
        out_dir = ROOT / "renders" / "inpaint" / beat_id
        frames = extract_frames(args.ffmpeg, segment, times, out_dir, beat_id)
        manifest = build_manifest(
            beat_id,
            segment,
            frames,
            regions,
            status="staged",
        )
        if gateway is not None and not args.stage_only:
            api_key = resolve_api_key()
            worker = pick_image_worker(gateway, api_key)
            manifest["jobs"] = submit_frames(
                gateway,
                api_key,
                worker,
                manifest,
                args.prompt,
                args.negative_prompt,
                args.reference_strength,
            )
            manifest["status"] = "submitted"
        save_manifest(out_dir, manifest)
        print(
            f"{beat_id}: {manifest['status']} with {len(frames)} frame(s) under "
            f"{out_dir.relative_to(ROOT).as_posix()}",
            flush=True,
        )
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
