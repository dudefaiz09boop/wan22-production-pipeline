#!/usr/bin/env python3
"""Collect tagged Qwen jobs, download completed PNGs, and safely replace manifested frames."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import sys
import time
import urllib.error
import urllib.request
import zlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
SHOT_IDS = [f"c{number:02d}" for number in range(1, 71)]


def select_shot_ids(first_id: str, last_id: str) -> list[str]:
    if first_id not in SHOT_IDS or last_id not in SHOT_IDS:
        raise ValueError("Shot range must use ids c01 through c70")
    first = SHOT_IDS.index(first_id)
    last = SHOT_IDS.index(last_id)
    if first > last:
        raise ValueError("Shot range start must not follow its end")
    return SHOT_IDS[first:last + 1]


def parse_shot_ids(value: str) -> list[str]:
    shot_ids = [shot_id.strip() for shot_id in value.split(",")]
    if not shot_ids or any(shot_id not in SHOT_IDS for shot_id in shot_ids):
        raise ValueError("Shot ids must use ids c01 through c70")
    if len(set(shot_ids)) != len(shot_ids):
        raise ValueError("Shot ids must not contain duplicates")
    return shot_ids


def api_request(url: str, api_key: str, *, data: bytes | None = None) -> bytes:
    headers = {"Authorization": f"Bearer {api_key}"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            return response.read()
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")
        raise RuntimeError(f"Studio API returned HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"Studio API request failed: {error.reason}") from error


def read_api_key() -> str:
    env_file = ROOT / ".env"
    if not env_file.is_file():
        raise RuntimeError(f"Missing API configuration: {env_file}")
    for line in env_file.read_text(encoding="utf-8").splitlines():
        match = re.match(r"^\s*API_KEY\s*=\s*(.*?)\s*$", line)
        if match:
            key = match.group(1).strip("\"'")
            if len(key) >= 32:
                return key
    raise RuntimeError("API_KEY is missing or invalid in .env")


def png_size(data: bytes) -> tuple[int, int]:
    if len(data) < 8 or data[:8] != PNG_SIGNATURE:
        raise RuntimeError("Generated output is not a valid PNG")
    offset = 8
    width = height = bit_depth = color_type = interlace = None
    palette_seen = False
    idat_parts = []
    seen_idat = False
    seen_iend = False
    while offset < len(data):
        if len(data) - offset < 12:
            raise RuntimeError("Generated PNG has a truncated chunk")
        length = struct.unpack(">I", data[offset:offset + 4])[0]
        chunk_type = data[offset + 4:offset + 8]
        chunk_end = offset + 12 + length
        if chunk_end > len(data):
            raise RuntimeError("Generated PNG has a truncated chunk")
        chunk_data = data[offset + 8:offset + 8 + length]
        expected_crc = struct.unpack(">I", data[offset + 8 + length:chunk_end])[0]
        if zlib.crc32(chunk_type + chunk_data) & 0xFFFFFFFF != expected_crc:
            raise RuntimeError("Generated PNG has an invalid chunk checksum")
        if offset == 8:
            if chunk_type != b"IHDR" or length != 13:
                raise RuntimeError("Generated PNG is missing a valid IHDR chunk")
            width, height, bit_depth, color_type, compression, filtering, interlace = (
                struct.unpack(">IIBBBBB", chunk_data)
            )
            if (
                width == 0
                or height == 0
                or compression != 0
                or filtering != 0
                or interlace not in {0, 1}
            ):
                raise RuntimeError("Generated PNG has invalid image metadata")
            allowed_depths = {
                0: {1, 2, 4, 8, 16},
                2: {8, 16},
                3: {1, 2, 4, 8},
                4: {8, 16},
                6: {8, 16},
            }
            if bit_depth not in allowed_depths.get(color_type, set()):
                raise RuntimeError("Generated PNG has an invalid bit depth or color type")
        elif chunk_type == b"PLTE":
            palette_seen = True
        elif chunk_type == b"IDAT":
            if seen_iend:
                raise RuntimeError("Generated PNG contains image data after IEND")
            seen_idat = True
            idat_parts.append(chunk_data)
        elif chunk_type == b"IEND":
            if length != 0 or seen_iend or chunk_end != len(data):
                raise RuntimeError("Generated PNG has an invalid IEND chunk")
            seen_iend = True
        elif not (chunk_type[0] & 0x20):
            raise RuntimeError("Generated PNG contains an unknown critical chunk")
        offset = chunk_end

    if (
        width is None
        or height is None
        or bit_depth is None
        or color_type is None
        or interlace is None
        or not seen_idat
        or not seen_iend
        or (color_type == 3 and not palette_seen)
    ):
        raise RuntimeError("Generated PNG is missing required chunks")
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[color_type]
    passes = (
        [(0, 0, 1, 1)]
        if interlace == 0
        else [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4),
              (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)]
    )
    expected_size = 0
    for x_start, y_start, x_step, y_step in passes:
        pass_width = max(0, (width - x_start + x_step - 1) // x_step)
        pass_height = max(0, (height - y_start + y_step - 1) // y_step)
        row_size = (pass_width * channels * bit_depth + 7) // 8
        expected_size += pass_height * (row_size + 1)
    try:
        decompressor = zlib.decompressobj()
        pixels = decompressor.decompress(b"".join(idat_parts)) + decompressor.flush()
    except zlib.error as error:
        raise RuntimeError("Generated PNG has invalid compressed image data") from error
    if not decompressor.eof or decompressor.unused_data or len(pixels) != expected_size:
        raise RuntimeError("Generated PNG image data is incomplete or malformed")
    row_offset = 0
    for x_start, y_start, x_step, y_step in passes:
        pass_width = max(0, (width - x_start + x_step - 1) // x_step)
        pass_height = max(0, (height - y_start + y_step - 1) // y_step)
        row_size = (pass_width * channels * bit_depth + 7) // 8
        for _ in range(pass_height):
            if pixels[row_offset] > 4:
                raise RuntimeError("Generated PNG uses an invalid row filter")
            row_offset += row_size + 1
    return width, height


def is_placeholder(path: Path) -> bool:
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return False
    return data[:8] == PNG_SIGNATURE and png_size(data) == (1, 1)


def may_replace_frame(path: Path) -> bool:
    return not path.exists() or is_placeholder(path)


def may_replace_manifest_frame(path: Path, entry: dict, replace_existing: bool) -> bool:
    if may_replace_frame(path):
        return True
    previous_hash = entry.get("previous_output_sha256")
    return (
        replace_existing
        and isinstance(previous_hash, str)
        and hashlib.sha256(path.read_bytes()).hexdigest() == previous_hash
    )


def write_manifest(path: Path, manifest: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for attempt in range(6):
        try:
            os.replace(temporary, path)
            return
        except PermissionError:
            if attempt == 5:
                raise
            time.sleep(0.2)


def collect_completed_frame(
    api: str,
    api_key: str,
    shot_id: str,
    entry: dict,
    config: dict,
    *,
    replace_existing: bool,
) -> tuple[int, int]:
    frame_path = (ROOT / entry["start_frame"]).resolve()
    if not frame_path.is_relative_to(ROOT.resolve()) or frame_path.suffix.lower() != ".png":
        raise RuntimeError(f"{shot_id}: unsafe start-frame path in manifest")
    if frame_path.is_file() and entry.get("output_sha256"):
        existing = frame_path.read_bytes()
        if hashlib.sha256(existing).hexdigest() == entry["output_sha256"]:
            return png_size(existing)
    if not may_replace_manifest_frame(frame_path, entry, replace_existing):
        raise RuntimeError(
            f"{shot_id}: refusing to overwrite an existing start frame "
            "that does not match the manifest's recorded previous hash",
        )

    image = api_request(f"{api}/v1/tasks/{entry['job_id']}/image?index=0", api_key)
    width, height = png_size(image)
    if (width, height) != (int(config["width"]), int(config["height"])):
        raise RuntimeError(
            f"{shot_id}: expected {config['width']}x{config['height']}, got {width}x{height}",
        )
    frame_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = frame_path.with_suffix(frame_path.suffix + ".tmp")
    temporary.write_bytes(image)
    os.replace(temporary, frame_path)
    entry["output_sha256"] = hashlib.sha256(image).hexdigest()
    entry["output_dimensions"] = [width, height]
    entry["status"] = "completed"
    entry["terminal_status"] = "succeeded"
    entry.pop("error", None)
    return width, height


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", default="http://127.0.0.1:3000")
    parser.add_argument("--manifest", default="prompts/startframe_batch_manifest.json")
    parser.add_argument("--from", dest="first_id", default="c01")
    parser.add_argument("--to", dest="last_id", default="c70")
    parser.add_argument("--ids", help="comma-separated beat ids for a non-contiguous batch")
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--timeout-hours", type=float, default=72)
    parser.add_argument("--once", action="store_true", help="poll once and exit")
    parser.add_argument(
        "--replace-existing",
        action="store_true",
        help="replace frames only when their hashes match previous_output_sha256 in the manifest",
    )
    args = parser.parse_args()
    if args.poll_seconds < 1 or args.timeout_hours <= 0:
        parser.error("poll interval and timeout must be positive")
    try:
        if args.ids:
            if args.first_id != "c01" or args.last_id != "c70":
                parser.error("--ids cannot be combined with --from or --to")
            selected_ids = parse_shot_ids(args.ids)
        else:
            selected_ids = select_shot_ids(args.first_id, args.last_id)
    except ValueError as error:
        parser.error(str(error))

    manifest_path = (ROOT / args.manifest).resolve()
    if not manifest_path.is_relative_to(ROOT) or not manifest_path.is_file():
        raise RuntimeError("Start-frame manifest is missing or outside the workspace")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = manifest.get("jobs")
    if not isinstance(jobs, dict) or set(jobs) != set(SHOT_IDS):
        raise RuntimeError("Expected tagged start-frame jobs c01 through c70")
    if any(not isinstance(jobs[shot_id], dict) for shot_id in SHOT_IDS):
        raise RuntimeError("Start-frame batch contains an invalid job record")
    if any(
        not isinstance(jobs[shot_id].get("job_id"), str)
        and jobs[shot_id].get("terminal_status") not in {"failed", "succeeded"}
        for shot_id in selected_ids
    ):
        raise RuntimeError(
            f"Start-frame batch is incomplete; submit {selected_ids[0]} through "
            f"{selected_ids[-1]} first"
        )
    api_key = read_api_key()
    api = args.api.rstrip("/")
    deadline = time.monotonic() + args.timeout_hours * 3600
    previous_summary = None

    while True:
        worker = json.loads(api_request(
            f"{api}/v1/workers/{manifest['worker_id']}", api_key,
        ))
        if worker.get("stopRequested") or worker.get("status") in {"stopped", "stopping", "failed"}:
            raise RuntimeError(
                f"Image worker became {worker.get('status')}; no worker was stopped by this script",
            )

        pending = 0
        failures = 0
        completed = 0
        for shot_id in selected_ids:
            entry = jobs[shot_id]
            if not isinstance(entry.get("job_id"), str):
                if entry.get("terminal_status") == "failed":
                    failures += 1
                    continue
                if entry.get("terminal_status") == "succeeded" and entry.get("output_sha256"):
                    completed += 1
                    continue
                raise RuntimeError(f"{shot_id}: no task id or terminal status in batch manifest")
            task = json.loads(api_request(f"{api}/v1/tasks/{entry['job_id']}", api_key))
            status = task.get("status")
            entry["status"] = status
            if status in {"queued", "running", "submitting", "submitted", "processing"}:
                entry["terminal_status"] = None
                pending += 1
                continue
            if status in {"failed", "cancelled", "stopped"}:
                entry["error"] = task.get("error") or status
                entry["terminal_status"] = "failed"
                failures += 1
                continue
            if status != "completed":
                entry["terminal_status"] = None
                pending += 1
                continue

            entry["terminal_status"] = "succeeded"
            config_path = ROOT / "prompts" / f"{shot_id}.json"
            config = json.loads(config_path.read_text(encoding="utf-8"))
            width, height = collect_completed_frame(
                api,
                api_key,
                shot_id,
                entry,
                config,
                replace_existing=args.replace_existing,
            )
            completed += 1
            print(f"{shot_id}: completed and tagged ({width}x{height})", flush=True)

        all_succeeded = all(
            jobs[shot_id].get("terminal_status") == "succeeded"
            and jobs[shot_id].get("output_sha256")
            for shot_id in SHOT_IDS
        )
        all_terminal = all(
            jobs[shot_id].get("terminal_status") in {"succeeded", "failed"}
            for shot_id in SHOT_IDS
        )
        manifest["batch_status"] = "generated" if all_succeeded else (
            "images_failed" if all_terminal else "generating"
        )
        manifest["completed_count"] = sum(
            jobs[shot_id].get("terminal_status") == "succeeded"
            and bool(jobs[shot_id].get("output_sha256"))
            for shot_id in SHOT_IDS
        )
        manifest["failed_count"] = sum(
            jobs[shot_id].get("terminal_status") == "failed"
            for shot_id in SHOT_IDS
        )
        manifest["pending_count"] = len(SHOT_IDS) - (
            manifest["completed_count"] + manifest["failed_count"]
        )
        write_manifest(manifest_path, manifest)
        summary = (completed, failures, pending, worker.get("status"))
        if summary != previous_summary:
            print(
                f"Image batch: {completed}/{len(selected_ids)} ready, {failures} failed, "
                f"{pending} pending; worker={worker.get('status')}/{worker.get('remoteStatus')}",
                flush=True,
            )
            previous_summary = summary
        if pending == 0:
            return 1 if failures else 0
        if args.once:
            return 0
        if time.monotonic() >= deadline:
            raise RuntimeError("Image batch monitoring timed out; worker remains untouched")
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
