#!/usr/bin/env python3
"""Build a local start-frame manifest from the existing prompt and image files."""
from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

import yaml

import collect_startframes


ROOT = Path(__file__).resolve().parents[1]
SHOT_IDS = [f"c{number:02d}" for number in range(1, 71)]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def png_dimensions(data: bytes, shot_id: str) -> tuple[int, int]:
    try:
        return collect_startframes.png_size(data)
    except RuntimeError as error:
        raise RuntimeError(f"{shot_id}: {error}") from error


def validate_distinct_hashes(hashes: list[str]) -> None:
    if len(hashes) != len(SHOT_IDS) or len(set(hashes)) != len(SHOT_IDS):
        raise RuntimeError("Start frames must have 70 distinct SHA-256 hashes")


def build_manifest(root: Path = ROOT) -> dict:
    spec_path = root / "docs" / "beats.yaml"
    data = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    shots = data.get("shots") if isinstance(data, dict) else None
    if not isinstance(shots, list):
        raise RuntimeError("docs/beats.yaml must contain a shots list")
    shot_ids = [shot.get("id") for shot in shots if isinstance(shot, dict)]
    if shot_ids != SHOT_IDS:
        raise RuntimeError("docs/beats.yaml must contain ordered shots c01 through c70")

    frames: dict[str, tuple[Path, dict]] = {}
    missing = []
    for shot in shots:
        shot_id = shot["id"]
        expected = f"startframes/{shot_id}.png"
        if shot.get("start_frame") != expected:
            raise RuntimeError(f"{shot_id}: unexpected start_frame path in docs/beats.yaml")
        config_path = root / "prompts" / f"{shot_id}.json"
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("id") != shot_id or config.get("start_frame") != expected:
            raise RuntimeError(f"{shot_id}: prompt config does not match docs/beats.yaml")
        frame_path = (root / expected).resolve()
        try:
            frame_path.relative_to(root.resolve())
        except ValueError as error:
            raise RuntimeError(f"{shot_id}: start-frame path escapes the workspace") from error
        frames[shot_id] = (frame_path, config)
        if not frame_path.is_file():
            missing.append(shot_id)
    if missing:
        raise RuntimeError(f"Missing start frames: {', '.join(missing)}")

    jobs = {}
    reference_hashes = []
    for shot_id in SHOT_IDS:
        frame_path, config = frames[shot_id]
        frame = frame_path.read_bytes()
        width, height = png_dimensions(frame, shot_id)
        expected_dimensions = (int(config["width"]), int(config["height"]))
        if (width, height) != expected_dimensions:
            raise RuntimeError(
                f"{shot_id}: expected {expected_dimensions[0]}x{expected_dimensions[1]}, "
                f"got {width}x{height}"
            )
        if width < 832 or height < 480:
            raise RuntimeError(f"{shot_id}: start frame is below the 832x480 minimum")
        prompt = (root / "prompts" / f"{shot_id}.txt").read_bytes()
        if not prompt:
            raise RuntimeError(f"{shot_id}: compiled prompt is empty")
        output_hash = sha256(frame)
        prompt_hash = sha256(prompt)
        reference_hashes.append(f"{shot_id}:{output_hash}\n")
        jobs[shot_id] = {
            "job_id": None,
            "status": "completed",
            "start_frame": f"startframes/{shot_id}.png",
            "prompt_sha256": prompt_hash,
            "output_sha256": output_hash,
            "output_dimensions": [width, height],
        }

    hashes = [jobs[shot_id]["output_sha256"] for shot_id in SHOT_IDS]
    validate_distinct_hashes(hashes)
    provenance = {}
    batch_path = root / "prompts" / "startframe_batch_manifest.json"
    if batch_path.is_file():
        batch = json.loads(batch_path.read_text(encoding="utf-8"))
        if batch.get("reference_free") is True:
            batch_jobs = batch.get("jobs")
            if not isinstance(batch_jobs, dict) or set(batch_jobs) != set(SHOT_IDS):
                raise RuntimeError("Reference-free batch state must contain all 70 beats")
            if any(
                not isinstance(batch_jobs[shot_id], dict)
                or batch_jobs[shot_id].get("terminal_status") != "succeeded"
                or batch_jobs[shot_id].get("output_sha256") != jobs[shot_id]["output_sha256"]
                for shot_id in SHOT_IDS
            ):
                raise RuntimeError("Reference-free batch hashes do not match all 70 frame files")
            provenance = {
                "reference_free": True,
                "provisional": True,
                "regenerate_after": "Wave 1a identity LoRA",
                "redo_candidates": ["c25", "c33", "c44", "c61", "c68"],
            }

    return {
        "schema_version": 1,
        "worker_id": None,
        "batch_status": "images_ready",
        "completed_count": len(SHOT_IDS),
        "failed_count": 0,
        "pending_count": 0,
        "reference_sha256": sha256("".join(reference_hashes).encode("ascii")),
        **provenance,
        "jobs": jobs,
    }


def write_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> int:
    try:
        manifest = build_manifest()
        write_manifest(ROOT / "startframes" / "manifest.json", manifest)
    except (OSError, ValueError, KeyError, TypeError, yaml.YAMLError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print(
        f"Start-frame manifest: {manifest['completed_count']}/70 ready; "
        f"batch_status={manifest['batch_status']}; "
        "written=startframes/manifest.json",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
