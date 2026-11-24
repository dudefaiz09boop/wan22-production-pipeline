#!/usr/bin/env python3
"""Render the tagged shot batch in order with low-denoise frame handoffs."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
from urllib.parse import urlsplit
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SHOT_IDS = [f"c{number:02d}" for number in range(1, 71)]
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
VIDEO_KEYFRAME_MAX_BYTES = 5 * 1024 * 1024
# Wan 2.2 is trained at 16 fps with 81 frames (wan/configs/shared_config.py:
# sample_fps=16, frame_num=81). The rest of the pipeline now matches.
VIDEO_FPS = 16
# Native render resolution from docs/beats.yaml meta.res_base. Segments are kept
# at this size instead of being upscaled to 1280x720, which added no detail.
RENDER_WIDTH = 832
RENDER_HEIGHT = 480
MAX_CLIP_DURATION_SECONDS = 5
POLL_FLOOR_SECONDS = 2


def poll_delay(started_at: float, cap_seconds: int) -> float:
    """Ramp a status-poll interval from 2s up to the configured cap.

    Status reads are cheap loopback calls, so a fast render is noticed within
    a couple of seconds instead of waiting a full slow interval; long renders
    back off to the cap so the gateway is not polled continuously.
    """
    cap = max(1, int(cap_seconds))
    elapsed = max(0.0, time.monotonic() - started_at)
    return max(min(POLL_FLOOR_SECONDS, cap), min(cap, elapsed / 4))


def validate_api_origin(api: str) -> str:
    parsed = urlsplit(api)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeError("The Studio API must be a loopback HTTP origin; credentials are never sent remotely")
    return api.rstrip("/")


def read_env_file() -> dict[str, str]:
    env_file = ROOT / ".env"
    if not env_file.is_file():
        return {}
    values = {}
    for line in env_file.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", line)
        if match:
            values[match.group(1)] = match.group(2).strip("\"'")
    return values


def load_env() -> dict[str, str]:
    result = read_env_file()
    result.update({key: value for key, value in os.environ.items() if value})
    if len(result.get("API_KEY", "")) < 32:
        raise RuntimeError("API_KEY is missing or invalid in .env")
    return result


def resolve_video_profile() -> tuple[str, str]:
    process_value = os.environ.get("KAGGLE_VIDEO_PROFILE", "").strip()
    if process_value:
        return process_value, "$env:KAGGLE_VIDEO_PROFILE"
    dotenv_value = read_env_file().get("KAGGLE_VIDEO_PROFILE", "").strip()
    if dotenv_value:
        return dotenv_value, ".env"
    raise RuntimeError(
        "KAGGLE_VIDEO_PROFILE not found: checked "
        "$env:KAGGLE_VIDEO_PROFILE and .env"
    )


def api_request(api: str, api_key: str, route: str, *, data: dict | None = None) -> bytes:
    headers = {"Authorization": f"Bearer {api_key}"}
    encoded = None
    if data is not None:
        encoded = json.dumps(data, separators=(",", ":")).encode("utf-8")
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(f"{api}{route}", data=encoded, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        raise RuntimeError(f"Studio API returned HTTP {error.code} for {route}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Studio API request failed for {route}: {error.reason}") from error


def api_json(api: str, api_key: str, route: str, *, data: dict | None = None) -> dict:
    try:
        return json.loads(api_request(api, api_key, route, data=data))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"Studio API returned invalid JSON for {route}") from error


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def png_dimensions(data: bytes) -> tuple[int, int]:
    if len(data) < 24 or data[:8] != PNG_SIGNATURE or data[12:16] != b"IHDR":
        raise RuntimeError("Generated start frame is not a valid PNG")
    return int.from_bytes(data[16:20], "big"), int.from_bytes(data[20:24], "big")


def run_media(command: list[str], timeout: int = 600) -> None:
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode:
        detail = (result.stderr or result.stdout)[-5000:]
        raise RuntimeError(f"Media command failed ({result.returncode}): {detail}")


def encode_jpeg(
    ffmpeg: str,
    source: Path,
    destination: Path,
    width: int,
    height: int,
    max_bytes: int,
    *,
    from_video_end: bool = False,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    for quality in (2, 4, 6, 8, 10, 12, 15, 18, 22, 26):
        destination.unlink(missing_ok=True)
        command = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
        if from_video_end:
            command.extend(["-sseof", "-0.05"])
        command.extend([
            "-i", str(source),
            "-frames:v", "1",
            "-vf", (
                f"scale={width}:{height}:force_original_aspect_ratio=increase,"
                f"crop={width}:{height},setsar=1"
            ),
            "-q:v", str(quality),
            str(destination),
        ])
        run_media(command, timeout=180)
        if destination.is_file() and destination.stat().st_size <= max_bytes:
            return
    raise RuntimeError(
        f"Could not encode {source.name} below the {max_bytes}-byte keyframe limit",
    )


def read_prompt(shot_id: str, expected_hash: str) -> str:
    prompt_path = ROOT / "prompts" / f"{shot_id}.txt"
    # Hash the raw bytes: the manifest records the on-disk digest, and text-mode
    # reads normalize CRLF to LF, which would never match on Windows.
    raw = prompt_path.read_bytes()
    if sha256(raw) != expected_hash:
        raise RuntimeError(f"{shot_id}: prompt does not match the queued image manifest")
    prompt = raw.decode("utf-8")
    if len(prompt) > 2000:
        raise RuntimeError(f"{shot_id}: compiled prompt exceeds the worker API limit")
    return prompt


def wait_for_images(
    api: str,
    api_key: str,
    image_worker_id: str | None,
    manifest_path: Path,
    selected: list[str],
    poll_seconds: int,
    deadline: float,
) -> dict:
    previous_ready = None
    started_at = time.monotonic()
    verified: dict[str, tuple[tuple[int, int], str]] = {}
    while True:
        if image_worker_id is not None:
            worker = api_json(api, api_key, f"/v1/workers/{image_worker_id}")
            if worker.get("stopRequested") or worker.get("status") in {"stopped", "stopping", "failed"}:
                raise RuntimeError(
                    f"Image worker became {worker.get('status')}; no worker was stopped by this script",
                )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("batch_status") == "images_failed":
            raise RuntimeError("One or more reference-conditioned image jobs failed")
        jobs = manifest.get("jobs", {})
        if not isinstance(jobs, dict) or not set(selected).issubset(jobs):
            raise RuntimeError("Start-frame manifest is missing selected shot jobs")
        ready = 0
        for shot_id in selected:
            entry = jobs.get(shot_id, {})
            start_frame = entry.get("start_frame")
            if not isinstance(start_frame, str) or not start_frame:
                continue
            path = (ROOT / start_frame).resolve()
            if not path.is_relative_to(ROOT):
                raise RuntimeError(f"{shot_id}: unsafe start-frame path in manifest")
            if entry.get("status") != "completed" or not path.is_file():
                continue
            stat = path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            cached = verified.get(shot_id)
            if cached is None or cached[0] != signature:
                image = path.read_bytes()
                width, height = png_dimensions(image)
                config = json.loads((ROOT / "prompts" / f"{shot_id}.json").read_text(encoding="utf-8"))
                # The keyframe is resampled to the render size by encode_jpeg, so a
                # start frame only has to meet the documented 832x480 minimum. It
                # does not have to equal the render size exactly.
                if width < 832 or height < 480:
                    raise RuntimeError(
                        f"{shot_id}: start frame is {width}x{height}, below the 832x480 minimum",
                    )
                if (width, height) != (int(config["width"]), int(config["height"])):
                    print(
                        f"{shot_id}: start frame is {width}x{height}; "
                        f"scaling to {config['width']}x{config['height']} for the render",
                        flush=True,
                    )
                digest = sha256(image)
                if entry.get("output_sha256") != digest:
                    raise RuntimeError(f"{shot_id}: start-frame checksum does not match the image manifest")
                verified[shot_id] = (signature, digest)
            elif entry.get("output_sha256") != cached[1]:
                raise RuntimeError(f"{shot_id}: start-frame checksum does not match the image manifest")
            ready += 1
        if ready != previous_ready:
            print(f"Start frames: {ready}/{len(selected)} ready", flush=True)
            previous_ready = ready
        if ready == len(selected):
            return manifest
        if time.monotonic() >= deadline:
            raise RuntimeError("Timed out waiting for start frames; workers were left untouched")
        time.sleep(poll_delay(started_at, poll_seconds))


def missing_start_frames(selected: list[str]) -> list[str]:
    root = ROOT.resolve()
    missing = []
    for shot_id in selected:
        config_path = root / "prompts" / f"{shot_id}.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        start_frame = (root / config["start_frame"]).resolve()
        if not start_frame.is_relative_to(root):
            raise RuntimeError(f"{shot_id}: unsafe start-frame path in prompt config")
        if not start_frame.is_file():
            missing.append(shot_id)
    return missing


def discover_image_worker(api: str, api_key: str, selected: list[str]) -> dict | None:
    response = api_json(api, api_key, "/v1/workers/active")
    workers = response.get("data", [])
    if not isinstance(workers, list):
        workers = []
    matches = [
        worker for worker in workers
        if isinstance(worker, dict)
        and worker.get("kind") == "image"
        and worker.get("status") == "idle"
    ]
    if matches:
        return matches[0]
    for worker in workers:
        if isinstance(worker, dict):
            print("|".join(str(worker.get(key, "")) for key in ("kind", "model", "status", "id")))
    missing = missing_start_frames(selected)
    if not missing:
        print("no idle image worker; refs present, continuing", flush=True)
        return None
    print(f"Missing required start frames: {', '.join(missing)}", file=sys.stderr)
    raise SystemExit(2)


def discover_video_worker(api: str, api_key: str, video_profile: str) -> dict:
    response = api_json(api, api_key, "/v1/workers")
    matches = [
        worker for worker in response.get("data", [])
        if worker.get("kind") == "video"
        and worker.get("model") == video_profile
        and worker.get("status") in {"starting", "idle", "busy"}
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one existing {video_profile} video worker; found {len(matches)}. "
            "No worker was started or stopped.",
        )
    worker = matches[0]
    if worker.get("stopRequested") or worker.get("status") in {"stopped", "stopping", "failed"}:
        raise RuntimeError(f"Consistent Face worker is {worker.get('status')}; no worker was changed")
    return worker


def normalize_shot_id(value: str) -> str:
    match = re.fullmatch(r"c?(\d+)", value.strip(), re.IGNORECASE)
    if not match:
        raise ValueError(f"Invalid shot ID {value!r}; use c01 through c70")
    shot_id = f"c{int(match.group(1)):02d}"
    if shot_id not in SHOT_IDS:
        raise ValueError(f"Shot ID {shot_id} is not in the ordered shot list")
    return shot_id


def select_shots(start: str, end: str) -> list[str]:
    first = normalize_shot_id(start)
    last = normalize_shot_id(end)
    first_index = SHOT_IDS.index(first)
    last_index = SHOT_IDS.index(last)
    if first_index > last_index:
        raise ValueError(f"Range start {first} comes after range end {last}")
    return SHOT_IDS[first_index : last_index + 1]


def extract_last_frame(ffmpeg: str, segment: Path, destination: Path) -> None:
    """Extract the segment's final frame at its native size, atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".tmp.png")
    temporary.unlink(missing_ok=True)
    # Seek back a full second and let -update 1 overwrite the output for every
    # decoded frame, so the file left behind is the last frame. A short -sseof
    # window is unreliable: at 16 fps a 0.05s seek lands past the final frame.
    run_media([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-sseof", "-1", "-i", str(segment),
        "-update", "1", "-q:v", "1", str(temporary),
    ], timeout=180)
    if not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg produced no final frame for {segment.name}")
    os.replace(temporary, destination)


def stage_next_handoff(ffmpeg: str, shot_id: str, source: Path) -> Path | None:
    """Stage the next chained shot's handoff from this shot's finished clip.

    `source` is the downloaded raw clip during a normal run (so the handoff is
    ready before post-processing) or the normalized segment on resume. The next
    shot's chain flag is read from its generated config; shot ids are never
    hardcoded.
    """
    index = SHOT_IDS.index(shot_id)
    if index + 1 >= len(SHOT_IDS):
        return None
    next_id = SHOT_IDS[index + 1]
    config_path = ROOT / "prompts" / f"{next_id}.json"
    if not config_path.is_file():
        return None
    next_config = json.loads(config_path.read_text(encoding="utf-8"))
    if not next_config.get("chain"):
        return None
    handoff = ROOT / "renders" / "handoffs" / f"{next_id}.png"
    extract_last_frame(ffmpeg, source, handoff)
    return handoff


def ensure_handoff(
    shot_id: str,
    config: dict,
    handoff_path: Path,
    require: bool,
) -> None:
    """With --require-handoffs, refuse a chained shot whose handoff is missing."""
    if not require or shot_id == SHOT_IDS[0] or not config["chain"]:
        return
    if not handoff_path.is_file():
        raise RuntimeError(
            f"{shot_id}: --require-handoffs is set and the handoff frame is "
            f"missing at {handoff_path}; refusing to silently reseed this "
            "chained shot from its start frame",
        )


def keyframe_choice(
    shot_id: str,
    config: dict,
    handoff_available: bool,
) -> tuple[Path, bool, bool]:
    start_frame = (ROOT / config["start_frame"]).resolve()
    if not start_frame.is_relative_to(ROOT):
        raise RuntimeError(f"{shot_id}: unsafe start-frame path in prompt config")
    # An explicit reference override replaces the generated start frame for this
    # shot. It is still a real image on disk, so the render contract is unchanged.
    override = config.get("start_frame_override")
    if isinstance(override, str) and override:
        override_path = (ROOT / override).resolve()
        if not override_path.is_relative_to(ROOT):
            raise RuntimeError(f"{shot_id}: unsafe start-frame override path")
        if not override_path.is_file():
            raise RuntimeError(f"{shot_id}: start-frame override is missing at {override}")
        return override_path, False, False
    if shot_id == SHOT_IDS[0] or not config["chain"]:
        return start_frame, False, False
    if handoff_available:
        return ROOT / "renders" / "handoffs" / f"{shot_id}.png", True, False
    return start_frame, False, True


def shot_line(
    index: int,
    total: int,
    shot_id: str,
    reference: Path,
    chained: bool,
    reseeded: bool,
) -> None:
    try:
        ref_path = reference.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        ref_path = str(reference)
    message = (
        f"[{index}/{total}] {shot_id} | ref={ref_path} | "
        f"chained={str(chained).lower()}"
    )
    if reseeded:
        message += " | reseeded"
    print(message, flush=True)


def shot_duration(config: dict, shot_id: str) -> int:
    duration = config.get("duration")
    if (
        isinstance(duration, bool)
        or not isinstance(duration, int)
        or not 1 <= duration <= MAX_CLIP_DURATION_SECONDS
    ):
        raise RuntimeError(
            f"{shot_id}: compiled duration must be an integer from 1 to "
            f"{MAX_CLIP_DURATION_SECONDS} seconds"
        )
    if "beat_duration" in config and config["beat_duration"] != duration:
        raise RuntimeError(f"{shot_id}: compiled duration fields disagree")
    return duration


def load_shot_durations(selected: list[str]) -> dict[str, int]:
    durations = {}
    for shot_id in selected:
        config = json.loads((ROOT / "prompts" / f"{shot_id}.json").read_text(encoding="utf-8"))
        durations[shot_id] = shot_duration(config, shot_id)
    return durations


def video_frame_count(duration_seconds: int, fps: int = VIDEO_FPS) -> int:
    if (
        isinstance(duration_seconds, bool)
        or not isinstance(duration_seconds, int)
        or duration_seconds < 1
        or isinstance(fps, bool)
        or not isinstance(fps, int)
        or fps < 1
    ):
        raise ValueError("Video duration and frame rate must be positive integers")
    return round(duration_seconds * fps) + 1


def dry_run(selected: list[str], video_profile: str, profile_source: str) -> int:
    print(f"Video profile: {video_profile} (from {profile_source})", flush=True)
    planned: set[str] = set()
    seeds = []
    durations = load_shot_durations(selected)
    for index, shot_id in enumerate(selected, 1):
        config = json.loads((ROOT / "prompts" / f"{shot_id}.json").read_text(encoding="utf-8"))
        shot_index = SHOT_IDS.index(shot_id)
        handoff_available = False
        if shot_index:
            previous_id = SHOT_IDS[shot_index - 1]
            handoff_available = (
                previous_id in planned
                or (ROOT / "renders" / "handoffs" / f"{shot_id}.png").is_file()
            )
        reference, chained, reseeded = keyframe_choice(
            shot_id,
            config,
            handoff_available,
        )
        shot_line(index, len(selected), shot_id, reference, chained, reseeded)
        if not chained:
            seeds.append(shot_id)
        planned.add(shot_id)
    expected_duration = sum(durations.values())
    print(
        f"Dry run: {len(selected)} shots; expected duration "
        f"{expected_duration}s (+/-2s); seeds {' '.join(seeds)}",
        flush=True,
    )
    return 0


def wait_for_task(
    api: str,
    api_key: str,
    job_id: str,
    poll_seconds: int,
    deadline: float,
) -> dict:
    previous_status = None
    started_at = time.monotonic()
    while True:
        task = api_json(api, api_key, f"/v1/tasks/{job_id}")
        status = task.get("status")
        if status != previous_status:
            print(f"Job {job_id}: {status}", flush=True)
            previous_status = status
        if status == "completed":
            return task
        if status in {"failed", "cancelled", "stopped", "remote_completed"}:
            raise RuntimeError(f"Job {job_id} ended as {status}: {task.get('error') or task.get('statusMessage')}")
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Timed out waiting for job {job_id}; workers were left untouched")
        time.sleep(poll_delay(started_at, poll_seconds))


def download_video(api: str, api_key: str, job_id: str, destination: Path) -> None:
    data = api_request(api, api_key, f"/v1/tasks/{job_id}/video")
    if not data:
        raise RuntimeError(f"Video task {job_id} returned an empty file")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(data)
    os.replace(temporary, destination)


def normalize_video(
    ffmpeg: str,
    source: Path,
    destination: Path,
    duration_seconds: int,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Keep the model's native resolution and frame rate. Upscaling 832x480 to
    # 1280x720 and resampling 8 fps to 24 fps adds no detail: it only inflates
    # the file and duplicates frames, which reads as softness and judder.
    run_media([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(source),
        "-vf", f"scale={RENDER_WIDTH}:{RENDER_HEIGHT}:flags=lanczos,setsar=1,format=yuv420p",
        "-an", "-t", str(duration_seconds),
        "-r", str(VIDEO_FPS),
        "-c:v", "libx264", "-preset", "slow", "-crf", "16",
        "-pix_fmt", "yuv420p", "-movflags", "+faststart",
        str(destination),
    ], timeout=900)


def finalize_segment(
    ffmpeg: str,
    ffprobe: str,
    state: dict,
    state_path: Path,
    shot_id: str,
    raw_video: Path,
    duration_seconds: int,
    task_status: str,
) -> None:
    """Normalize a downloaded clip and record it.

    Called immediately after the *next* job is submitted, so the remote GPU
    renders while this machine runs ffmpeg and rewrites the render manifest.
    """
    segment = ROOT / "renders" / "segments" / f"{shot_id}.mp4"
    normalize_video(ffmpeg, raw_video, segment, duration_seconds)
    probe = probe_video(ffprobe, segment)
    stream = next((item for item in probe.get("streams", []) if item.get("codec_type") == "video"), None)
    actual_duration = float(probe.get("format", {}).get("duration", 0))
    if (
        not stream
        or (stream.get("width"), stream.get("height")) != (RENDER_WIDTH, RENDER_HEIGHT)
        or abs(actual_duration - duration_seconds) > 0.1
    ):
        raise RuntimeError(
            f"{shot_id}: normalized video failed its dimensions/duration check "
            f"(expected {duration_seconds}s, got {actual_duration:.3f}s)"
        )
    record = state["jobs"][shot_id]
    record["video_status"] = task_status
    record["segment_sha256"] = sha256(segment.read_bytes())
    record["segment_bytes"] = segment.stat().st_size
    record["duration_seconds"] = actual_duration
    record["video_dimensions"] = [stream["width"], stream["height"]]
    save_json(state_path, state)
    print(
        f"{shot_id}: rendered and tagged ({actual_duration:.2f}s, {RENDER_WIDTH}x{RENDER_HEIGHT})",
        flush=True,
    )


def probe_video(ffprobe: str, path: Path) -> dict:
    command = [
        ffprobe, "-v", "error", "-show_entries",
        "format=duration:stream=codec_type,codec_name,width,height,avg_frame_rate",
        "-of", "json", str(path),
    ]
    result = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    if result.returncode:
        raise RuntimeError(f"ffprobe failed for {path.name}: {result.stderr[-2000:]}")
    return json.loads(result.stdout)


def normalized_segment_matches(
    ffprobe: str,
    path: Path,
    duration_seconds: int,
) -> bool:
    metadata = probe_video(ffprobe, path)
    stream = next(
        (item for item in metadata.get("streams", []) if item.get("codec_type") == "video"),
        None,
    )
    duration = float(metadata.get("format", {}).get("duration", 0))
    return bool(
        stream
        and (stream.get("width"), stream.get("height")) == (RENDER_WIDTH, RENDER_HEIGHT)
        and abs(duration - duration_seconds) <= 0.1
    )


def stitch(
    ffmpeg: str,
    ffprobe: str,
    state: dict,
    state_path: Path,
    selected: list[str],
    expected_duration_seconds: int,
) -> Path:
    segments = [
        ROOT / "renders" / "segments" / f"{shot_id}.mp4"
        for shot_id in selected
    ]
    if any(not item.is_file() for item in segments):
        missing = [item.name for item in segments if not item.is_file()]
        raise RuntimeError(f"Cannot stitch; normalized segments are missing: {', '.join(missing)}")
    listing = ROOT / "renders" / "segments.txt"
    listing.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in segments:
        escaped = str(item.resolve()).replace("\\", "/").replace("'", "'\\''")
        rows.append(f"file '{escaped}'")
    listing.write_text("\n".join(rows) + "\n", encoding="utf-8")
    shot_count = len(selected)
    output = ROOT / "renders" / f"midnight-heat-{shot_count}-shot.mp4"
    run_media([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-f", "concat", "-safe", "0", "-i", str(listing),
        "-c", "copy", "-movflags", "+faststart", str(output),
    ], timeout=1800)
    metadata = probe_video(ffprobe, output)
    stream = next((item for item in metadata.get("streams", []) if item.get("codec_type") == "video"), None)
    duration = float(metadata.get("format", {}).get("duration", 0))
    if not stream or (stream.get("width"), stream.get("height")) != (RENDER_WIDTH, RENDER_HEIGHT):
        raise RuntimeError(
            f"Final film failed the {RENDER_WIDTH}x{RENDER_HEIGHT} video-stream check"
        )
    if not expected_duration_seconds - 2 <= duration <= expected_duration_seconds + 2:
        raise RuntimeError(
            f"Final film duration is {duration:.3f}s; "
            f"expected about {expected_duration_seconds}s"
        )
    state["film"] = {
        "path": str(output.relative_to(ROOT)),
        "sha256": sha256(output.read_bytes()),
        "bytes": output.stat().st_size,
        "duration_seconds": duration,
        "width": stream["width"],
        "height": stream["height"],
        "status": "completed",
    }
    save_json(state_path, state)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:3000")
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument(
        "--image-worker-id",
        default=None,
        help="existing image worker to use when reference frames are still being "
        "generated; without this flag an idle image worker is discovered, so a "
        "busy start-frame worker is never mistaken for an idle one",
    )
    parser.add_argument("--timeout-hours", type=float, default=168)
    parser.add_argument("--from", dest="from_shot", default="c01")
    parser.add_argument("--to", dest="to_shot", default="c70")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--require-handoffs",
        action="store_true",
        help="refuse to submit a chained shot whose handoff frame is missing "
        "instead of silently reseeding from its start frame",
    )
    args = parser.parse_args()
    if args.poll_seconds < 1 or args.timeout_hours <= 0:
        parser.error("poll interval and timeout must be positive")
    try:
        selected = select_shots(args.from_shot, args.to_shot)
    except ValueError as error:
        parser.error(str(error))

    video_profile, profile_source = resolve_video_profile()
    if args.dry_run:
        return dry_run(selected, video_profile, profile_source)

    shot_durations = load_shot_durations(selected)
    env = load_env()
    api_key = env["API_KEY"]
    api = validate_api_origin(args.api)
    ffmpeg = env.get("FFMPEG_PATH", "ffmpeg")
    ffprobe = env.get("FFPROBE_PATH", "ffprobe")
    if args.image_worker_id:
        image_worker = api_json(api, api_key, f"/v1/workers/{args.image_worker_id}")
        if image_worker.get("kind") != "image":
            raise RuntimeError("The worker named by --image-worker-id is not an image worker")
        if image_worker.get("stopRequested") or image_worker.get("status") in {
            "stopped", "stopping", "failed",
        }:
            raise RuntimeError(f"Image worker {args.image_worker_id} is {image_worker.get('status')}")
    else:
        image_worker = discover_image_worker(api, api_key, selected)
    video_worker = discover_video_worker(api, api_key, video_profile)
    print(f"Video profile: {video_profile} (from {profile_source})", flush=True)
    print(
        f"Using existing workers only: image={image_worker['id'] if image_worker else 'none'}, "
        f"video={video_worker['id']} ({video_worker['model']})",
        flush=True,
    )
    image_manifest_path = ROOT / "startframes" / "manifest.json"
    state_path = ROOT / "renders" / "manifest.json"
    # Do not gate the whole film on every start frame: each shot waits for its
    # own verified frame inside the loop, so a chained render can run while a
    # start-frame worker finishes later beats.
    image_manifest = json.loads(image_manifest_path.read_text(encoding="utf-8"))
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {
            "schema_version": 1,
            "video_worker_id": video_worker["id"],
            "video_profile": video_profile,
            "image_manifest": "startframes/manifest.json",
            "image_reference_sha256": image_manifest.get("reference_sha256"),
            "jobs": {},
        }
    if state.get("video_worker_id") != video_worker["id"]:
        raise RuntimeError("Render manifest belongs to a different video worker; refusing to mix outputs")
    save_json(state_path, state)
    negative_prompt = (ROOT / "negatives" / "artifact_neg.txt").read_text(encoding="utf-8").strip()
    if len(negative_prompt) > 4096:
        raise RuntimeError("Negative prompt exceeds the video worker API limit")
    deadline = time.monotonic() + args.timeout_hours * 3600
    pending_finalize: dict | None = None

    for index, shot_id in enumerate(selected):
        image_manifest = wait_for_images(
            api,
            api_key,
            image_worker["id"] if image_worker else None,
            image_manifest_path,
            [shot_id],
            args.poll_seconds,
            deadline,
        )
        clip_duration = shot_durations[shot_id]
        image_entry = image_manifest["jobs"][shot_id]
        config = json.loads((ROOT / "prompts" / f"{shot_id}.json").read_text(encoding="utf-8"))
        start_frame = (ROOT / config["start_frame"]).resolve()
        if not start_frame.is_relative_to(ROOT):
            raise RuntimeError(f"{shot_id}: unsafe start-frame path in prompt config")
        prompt = read_prompt(shot_id, image_entry["prompt_sha256"])
        record = state["jobs"].setdefault(shot_id, {
            "shot_id": shot_id,
            "scene": config["scene"],
            "build": config["build"],
            "chain": config["chain"],
            "prompt_sha256": image_entry["prompt_sha256"],
            "start_frame": config["start_frame"],
            "start_frame_sha256": image_entry["output_sha256"],
            "reference_contact_sha256": image_entry.get("reference_contact_sha256"),
            "image_job_id": image_entry.get("job_id"),
        })
        segment = ROOT / "renders" / "segments" / f"{shot_id}.mp4"
        raw_video = ROOT / "renders" / "raw" / f"{shot_id}.mp4"

        if record.get("video_job_id") and segment.is_file():
            if (
                record.get("segment_sha256") == sha256(segment.read_bytes())
                and normalized_segment_matches(ffprobe, segment, clip_duration)
            ):
                if pending_finalize is not None:
                    finalize_segment(ffmpeg, ffprobe, state, state_path, **pending_finalize)
                    pending_finalize = None
                keyframe_source = ROOT / record.get(
                    "keyframe_source",
                    config["start_frame"],
                )
                shot_line(
                    index + 1,
                    len(selected),
                    shot_id,
                    keyframe_source,
                    bool(config["chain"] and keyframe_source.parent.name == "handoffs"),
                    False,
                )
                staged = stage_next_handoff(ffmpeg, shot_id, segment)
                if staged is not None:
                    next_id = SHOT_IDS[SHOT_IDS.index(shot_id) + 1]
                    print(f"{shot_id}: staged handoff for {next_id}", flush=True)
                print(f"{shot_id}: already rendered; skipping", flush=True)
                continue

        handoff_path = ROOT / "renders" / "handoffs" / f"{shot_id}.png"
        ensure_handoff(shot_id, config, handoff_path, args.require_handoffs)
        keyframe_source, chained, reseeded = keyframe_choice(
            shot_id,
            config,
            handoff_path.is_file(),
        )
        if not keyframe_source.is_file():
            raise RuntimeError(
                f"{shot_id}: reference frame is missing at {keyframe_source}"
            )

        keyframe = ROOT / "renders" / "keyframes" / f"{shot_id}.jpg"
        encode_jpeg(
            ffmpeg,
            keyframe_source,
            keyframe,
            int(config["width"]),
            int(config["height"]),
            VIDEO_KEYFRAME_MAX_BYTES,
        )
        keyframe_bytes = keyframe.read_bytes()
        shot_line(
            index + 1,
            len(selected),
            shot_id,
            keyframe_source,
            chained,
            reseeded,
        )
        if record.get("video_job_id"):
            task = api_json(api, api_key, f"/v1/tasks/{record['video_job_id']}")
            if task.get("status") in {"failed", "cancelled", "stopped"}:
                raise RuntimeError(f"{shot_id}: video job failed: {task.get('error')}")
        else:
            response = api_json(api, api_key, f"/v1/workers/{video_worker['id']}/jobs/video", data={
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "width": int(config["width"]),
                "height": int(config["height"]),
                "frames": video_frame_count(clip_duration),
                "fps": VIDEO_FPS,
                "steps": 5,
                "cfg": 1,
                "loras": config["loras"],
                "keyframe_base64": base64.b64encode(keyframe_bytes).decode("ascii"),
                "keyframe_filename": keyframe.name,
                "story": {
                    "shot_id": shot_id,
                    "sequence": index + 1,
                    "scene": config["scene"],
                    "build": config["build"],
                    "chain": bool(config["chain"]),
                    "duration": clip_duration,
                    "video_profile": video_profile,
                    "image_job_id": image_entry.get("job_id"),
                    "handoff_job_id": record.get("handoff_job_id"),
                    "prompt_sha256": image_entry["prompt_sha256"],
                    "start_frame_sha256": image_entry["output_sha256"],
                    "reference_contact_sha256": image_entry.get("reference_contact_sha256"),
                },
            })
            record["video_job_id"] = response["id"]
            record["video_status"] = response.get("status", "queued")
            record["keyframe_sha256"] = sha256(keyframe_bytes)
            record["keyframe_source"] = keyframe_source.resolve().relative_to(ROOT).as_posix()
            save_json(state_path, state)

        if pending_finalize is not None:
            finalize_segment(ffmpeg, ffprobe, state, state_path, **pending_finalize)
            pending_finalize = None

        task = wait_for_task(api, api_key, record["video_job_id"], args.poll_seconds, deadline)
        download_video(api, api_key, record["video_job_id"], raw_video)
        try:
            staged = stage_next_handoff(ffmpeg, shot_id, raw_video)
        except RuntimeError:
            finalize_segment(
                ffmpeg,
                ffprobe,
                state,
                state_path,
                shot_id=shot_id,
                raw_video=raw_video,
                duration_seconds=clip_duration,
                task_status=task["status"],
            )
            staged = stage_next_handoff(ffmpeg, shot_id, segment)
        else:
            pending_finalize = {
                "shot_id": shot_id,
                "raw_video": raw_video,
                "duration_seconds": clip_duration,
                "task_status": task["status"],
            }
        if staged is not None:
            next_id = SHOT_IDS[SHOT_IDS.index(shot_id) + 1]
            print(f"{shot_id}: staged handoff for {next_id}", flush=True)

    if pending_finalize is not None:
        finalize_segment(ffmpeg, ffprobe, state, state_path, **pending_finalize)
        pending_finalize = None

    film = stitch(
        ffmpeg,
        ffprobe,
        state,
        state_path,
        selected,
        sum(shot_durations.values()),
    )
    print(f"Complete: {film} ({film.stat().st_size} bytes)", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
