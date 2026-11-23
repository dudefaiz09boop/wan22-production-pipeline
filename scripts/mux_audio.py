#!/usr/bin/env python3
"""Mux per-clip narration into the rendered clips, then build the final cut.

Two jobs:

1. `--clips`: for each rendered clip, attach its narration WAV (from
   audio/index.json) so the clip carries its own audio track. Clips without
   narration get a silent track of the right length, so the concat stays
   sample-accurate.

2. `--final`: concatenate the muxed clips into one MP4 with a single
   continuous audio stream, then optionally add a music bed.

This is the stage that makes the deliverable a real video with sound rather
than a silent picture.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml


def load_durations(spec: Path) -> dict[str, float]:
    data = yaml.safe_load(spec.read_text(encoding="utf-8"))
    out = {}
    for shot in (data or {}).get("shots") or []:
        sid = shot.get("id")
        if isinstance(sid, str):
            out[sid] = float(shot.get("duration") or 5)
    return out


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def has_audio(path: Path) -> bool:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "a",
            "-show_entries", "stream=index", "-of", "csv=p=0", str(path),
        ],
        capture_output=True, text=True,
    )
    return bool(result.stdout.strip())


def mux_clip(video: Path, audio: Path | None, dest: Path, duration: float) -> bool:
    """Attach audio to a clip, generating silence when narration is absent."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if audio is not None and audio.is_file():
        cmd = [
            "ffmpeg", "-y", "-i", str(video), "-i", str(audio),
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-shortest", str(dest),
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-i", str(video),
            "-f", "lavfi", "-i", f"anullsrc=r=48000:cl=stereo",
            "-map", "0:v:0", "-map", "1:a:0",
            "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
            "-t", f"{duration:.3f}", str(dest),
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  mux failed for {video.name}: {result.stderr[-400:]}", file=sys.stderr)
        return False
    return True


def concat_with_audio(clips: list[Path], dest: Path, crf: int, preset: str) -> bool:
    """Concatenate clips that already carry audio into one MP4."""
    listing = dest.with_suffix(".txt")
    listing.write_text(
        "\n".join(f"file '{c.resolve().as_posix()}'" for c in clips),
        encoding="utf-8",
    )
    cmd = [
        "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", str(listing),
        "-c:v", "libx264", "-crf", str(crf), "-preset", preset,
        "-pix_fmt", "yuv420p",
        "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2",
        "-movflags", "+faststart", str(dest),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    listing.unlink(missing_ok=True)
    if result.returncode != 0:
        print(f"  concat failed: {result.stderr[-600:]}", file=sys.stderr)
        return False
    return True


def add_music_bed(video: Path, music: Path, dest: Path, music_db: float) -> bool:
    """Mix a music bed under the existing narration track."""
    cmd = [
        "ffmpeg", "-y", "-i", str(video), "-stream_loop", "-1", "-i", str(music),
        "-filter_complex",
        f"[1:a]volume={music_db}dB[m];[0:a][m]amix=inputs=2:duration=first:dropout_transition=0[a]",
        "-map", "0:v:0", "-map", "[a]",
        "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-shortest", str(dest),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  music mix failed: {result.stderr[-400:]}", file=sys.stderr)
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Mux narration into clips and build the final cut")
    ap.add_argument("--spec", default="docs/beats.yaml")
    ap.add_argument("--renders", default="renders")
    ap.add_argument("--audio", default="audio")
    ap.add_argument("--muxed", default="renders/muxed")
    ap.add_argument("--output", default="output/final.mp4")
    ap.add_argument("--music", default=None, help="Optional music bed file")
    ap.add_argument("--music-db", type=float, default=-22.0)
    ap.add_argument("--crf", type=int, default=18)
    ap.add_argument("--preset", default="slow")
    ap.add_argument("--from-clip", default="c01")
    ap.add_argument("--to-clip", default="c70")
    args = ap.parse_args()

    durations = load_durations(Path(args.spec))
    audio_index_path = Path(args.audio) / "index.json"
    audio_index = {}
    if audio_index_path.is_file():
        audio_index = json.loads(audio_index_path.read_text(encoding="utf-8")).get("beats", {})

    start = int(args.from_clip[1:])
    end = int(args.to_clip[1:])
    clip_ids = [f"c{n:02d}" for n in range(start, end + 1)]

    renders = Path(args.renders)
    muxed_dir = Path(args.muxed)
    muxed_dir.mkdir(parents=True, exist_ok=True)

    muxed: list[Path] = []
    missing_video = []
    with_narration = 0

    for clip_id in clip_ids:
        candidates = sorted(renders.glob(f"**/{clip_id}.mp4"))
        if not candidates:
            missing_video.append(clip_id)
            continue
        video = candidates[0]
        duration = durations.get(clip_id, 5.0)

        audio_name = audio_index.get(clip_id)
        audio = Path(args.audio) / audio_name if audio_name else None
        if audio is not None and audio.is_file():
            with_narration += 1

        dest = muxed_dir / f"{clip_id}.mp4"
        if mux_clip(video, audio, dest, duration):
            muxed.append(dest)

    print(f"muxed {len(muxed)} clip(s), {with_narration} with narration")
    if missing_video:
        print(f"missing renders ({len(missing_video)}): {', '.join(missing_video)}", file=sys.stderr)

    if not muxed:
        print("nothing to concatenate", file=sys.stderr)
        return 1

    final = Path(args.output)
    final.parent.mkdir(parents=True, exist_ok=True)

    if args.music:
        temp = final.with_name(final.stem + ".nomusic.mp4")
        if not concat_with_audio(muxed, temp, args.crf, args.preset):
            return 1
        if not add_music_bed(temp, Path(args.music), final, args.music_db):
            return 1
        temp.unlink(missing_ok=True)
    else:
        if not concat_with_audio(muxed, final, args.crf, args.preset):
            return 1

    total = probe_duration(final)
    print(f"final: {final}  {total:.1f}s  audio={has_audio(final)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
