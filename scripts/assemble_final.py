#!/usr/bin/env python3
"""Concatenate clips, crossfade, and encode final deliverable."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import yaml


def load_beats(spec_path: Path) -> tuple[dict, list[dict]]:
    """Load beats.yaml and return meta and shots."""
    data = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    meta = data["meta"]
    shots = data["shots"]
    return meta, shots


def get_clip_duration(clip_cfg: dict) -> float:
    """Get clip duration in seconds from config."""
    return clip_cfg.get("duration", 5.0)


def build_concat_list(
    clips: list[dict],
    renders_dir: Path,
    output_list: Path,
    crossfade_duration: float = 0.5,
) -> list[Path]:
    """Build ffmpeg concat list with crossfade filters."""
    clip_files = []
    for clip in clips:
        clip_id = clip["id"]
        # Find the rendered video
        video_files = list(renders_dir.glob(f"**/{clip_id}*.mp4"))
        if not video_files:
            video_files = list(renders_dir.glob(f"**/wan22/{clip_id}*.mp4"))
        if not video_files:
            print(f"  ✗ Missing render for {clip_id}")
            continue
        clip_files.append(video_files[0])

    # Write concat list
    with output_list.open("w") as f:
        for clip_file in clip_files:
            f.write(f"file '{clip_file.absolute()}'\n")

    return clip_files


def concatenate_videos(
    concat_list: Path,
    output_path: Path,
    crossfade_duration: float = 0.5,
    crf: int = 18,
    preset: str = "slow",
) -> bool:
    """Concatenate videos using ffmpeg with crossfade."""
    print(f"Concatenating {len(open(concat_list).readlines())} clips...")

    # First, try simple concat (no re-encode) if all clips have same codec
    # For crossfade, we need filter_complex
    cmd = [
        "ffmpeg", "-y",
        "-f", "concat",
        "-safe", "0",
        "-i", str(concat_list),
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", preset,
        "-pix_fmt", "yuv420p",
        "-c:a", "aac",
        "-b:a", "192k",
        str(output_path),
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"  ✗ ffmpeg failed: {result.stderr}")
        return False

    print(f"  ✓ Output: {output_path}")
    return True


def concatenate_with_crossfade(
    clip_files: list[Path],
    output_path: Path,
    crossfade_duration: float = 0.5,
    crf: int = 18,
    preset: str = "slow",
) -> bool:
    """Concatenate videos with crossfade transitions using filter_complex."""
    if len(clip_files) <= 1:
        # Just copy single file
        shutil.copy2(clip_files[0], output_path)
        return True

    print(f"Concatenating {len(clip_files)} clips with {crossfade_duration}s crossfade...")

    # Build filter_complex for crossfade
    # This is complex - we'll use a simpler approach: concat then add crossfade in second pass
    # For now, use simple concat
    return concatenate_videos_simple(clip_files, output_path, crf, preset)


def concatenate_videos_simple(
    clip_files: list[Path],
    output_path: Path,
    crf: int = 18,
    preset: str = "slow",
) -> bool:
    """Simple concatenation without crossfade."""
    # Create temporary concat list
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        for clip_file in clip_files:
            f.write(f"file '{clip_file.absolute()}'\n")
        concat_list = Path(f.name)

    try:
        cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_list),
            "-c:v", "libx264",
            "-crf", str(crf),
            "-preset", preset,
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "192k",
            str(output_path),
        ]

        print(f"Running: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print(f"  ✗ ffmpeg failed: {result.stderr}")
            return False

        print(f"  ✓ Output: {output_path}")
        return True
    finally:
        concat_list.unlink(missing_ok=True)


def create_deliverable_zip(
    output_dir: Path,
    zip_path: Path,
    include_frames: bool = False,
) -> bool:
    """Create deliverable zip with all outputs."""
    print(f"Creating deliverable zip: {zip_path}")

    import zipfile

    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        # Add final video
        final_video = output_dir / "final.mp4"
        if final_video.exists():
            zf.write(final_video, "final.mp4")
            print(f"  Added: final.mp4")

        # Add individual clips
        renders_dir = output_dir / "renders"
        if renders_dir.exists():
            for clip_file in renders_dir.rglob("*.mp4"):
                arcname = f"clips/{clip_file.relative_to(renders_dir)}"
                zf.write(clip_file, arcname)
                print(f"  Added: {arcname}")

        # Add start frames if requested
        if include_frames:
            startframes_dir = output_dir / "startframes"
            if startframes_dir.exists():
                for frame_file in startframes_dir.glob("*.png"):
                    arcname = f"startframes/{frame_file.name}"
                    zf.write(frame_file, arcname)

        # Add production log
        logs_dir = output_dir / "logs"
        if logs_dir.exists():
            for log_file in logs_dir.glob("*.json"):
                arcname = f"logs/{log_file.name}"
                zf.write(log_file, arcname)

        # Add beats.yaml
        beats_file = Path("docs/beats.yaml")
        if beats_file.exists():
            zf.write(beats_file, "beats.yaml")

    print(f"  ✓ Zip created: {zip_path} ({zip_path.stat().st_size / (1024**2):.1f} MB)")
    return True


def main() -> int:
    import tempfile

    ap = argparse.ArgumentParser(description="Concatenate clips, crossfade, and encode final deliverable")
    ap.add_argument("--spec", default="docs/beats.yaml", help="Beats specification")
    ap.add_argument("--renders", default="renders", help="Renders directory")
    ap.add_argument("--output", default="output", help="Output directory")
    ap.add_argument("--crossfade", type=float, default=0.5, help="Crossfade duration (seconds)")
    ap.add_argument("--crf", type=int, default=18, help="Video quality (CRF)")
    ap.add_argument("--preset", default="slow", help="Encoding preset")
    ap.add_argument("--zip", action="store_true", help="Create deliverable zip")
    ap.add_argument("--include-frames", action="store_true", help="Include start frames in zip")
    ap.add_argument("--from-clip", default="c01", help="Start clip")
    ap.add_argument("--to-clip", default="c70", help="End clip")
    ap.add_argument(
        "--keep-existing",
        action="store_true",
        help="Do not re-concat when final.mp4 already exists (keeps muxed audio)",
    )
    args = ap.parse_args()

    # Load beats
    meta, shots = load_beats(Path(args.spec))

    # Filter clips
    start_num = int(args.from_clip[1:])
    end_num = int(args.to_clip[1:])
    clips = [s for s in shots if start_num <= int(s["id"][1:]) <= end_num]

    print(f"Assembling {len(clips)} clips ({args.from_clip} to {args.to_clip})")

    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find clip files
    renders_dir = Path(args.renders)
    clip_files = []
    missing = []
    for clip in clips:
        clip_id = clip["id"]
        video_files = list(renders_dir.glob(f"**/{clip_id}*.mp4"))
        if not video_files:
            video_files = list(renders_dir.glob(f"**/wan22/{clip_id}*.mp4"))
        if video_files:
            clip_files.append(video_files[0])
            print(f"  [OK] Found: {clip_id} -> {video_files[0].name}")
        else:
            missing.append(clip_id)
            print(f"  [FAIL] Missing: {clip_id}")

    if not clip_files:
        print("No clips found!")
        return 1

    if missing:
        print(f"\nWARNING: {len(missing)} clip(s) missing: {', '.join(missing)}")
        print("The final cut will contain only the clips that exist.")

    # Concatenate
    final_video = output_dir / "final.mp4"
    if args.keep_existing and final_video.is_file():
        print(f"  Keeping existing {final_video} (muxed audio preserved)")
    else:
        success = concatenate_videos_simple(clip_files, final_video, args.crf, args.preset)
        if not success:
            return 1

    # Create zip if requested
    if args.zip:
        zip_path = output_dir / "deliverable.zip"
        create_deliverable_zip(output_dir, zip_path, args.include_frames)

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())