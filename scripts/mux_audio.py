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


def load_meta(spec: Path) -> dict:
    data = yaml.safe_load(spec.read_text(encoding="utf-8")) or {}
    meta = data.get("meta")
    return meta if isinstance(meta, dict) else {}


def resolve_target_size(
    meta: dict, width: int | None, height: int | None
) -> tuple[int, int] | None:
    """Pick the frame size every clip is normalised to before concat.

    The spec renders `build: cf` shots at res_face and everything else at
    res_base, so the clips arrive at two different sizes. The concat demuxer
    requires one size, so everything is scaled to res_base unless overridden.
    """
    if width and height:
        return (width, height)
    base = meta.get("res_base")
    if isinstance(base, list) and len(base) == 2:
        return (int(base[0]), int(base[1]))
    return None


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


def realism_filters(width: int, height: int, fps: int, grade: bool) -> str:
    """Normalise geometry and, when grading, add camera-footage texture.

    The scale/pad/setsar/fps part is mandatory: the concat demuxer requires
    one resolution and one frame rate, and the spec renders `build: cf` at
    res_face while everything else is res_base.

    The grade part is what stops the output reading as AI video. Grain is
    applied *after* scaling so it stays a constant size instead of being
    resampled, and it is luma-weighted so shadows stay clean. Chromatic
    aberration is a 1px channel shift, not a visible fringe.
    """
    chain = [
        f"scale={width}:{height}:force_original_aspect_ratio=decrease",
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2",
        "setsar=1",
        f"fps={fps}",
    ]
    if grade:
        chain += [
            # 1px lateral CA: red left, blue right.
            "rgbashift=rh=-1:bh=1",
            # Gentle S-curve, warm highlights, cool shadows.
            "eq=contrast=1.06:saturation=1.04:gamma=0.99",
            "colorbalance=rs=0.02:bs=-0.02:rh=0.03:bh=-0.03",
            # Halation: bloom the highlights only, then recombine.
            "split=2[base][hi]",
            "[hi]lutyuv=y='if(gt(val,190),val,0)':u=128:v=128,"
            "gblur=sigma=6[bloom]",
            "[base][bloom]blend=all_mode=screen:all_opacity=0.18",
            # Grain last, luma-weighted so shadows do not turn to noise.
            "noise=alls=7:allf=t+u",
            # Recover a little bite lost to the bloom.
            "unsharp=5:5:0.5:5:5:0.0",
        ]
    return ",".join(chain)


def mux_clip(
    video: Path,
    audio: Path | None,
    dest: Path,
    duration: float,
    target: tuple[int, int] | None = None,
    fps: int = 16,
    grade: bool = False,
) -> bool:
    """Attach audio to a clip, generating silence when narration is absent.

    ``target`` normalises the frame size. The concat demuxer used by
    ``concat_with_audio`` requires every input to share a resolution and a
    frame rate, and the spec renders `build: cf` shots at res_face (1280x720)
    and everything else at res_base (832x480). Without this the final concat
    fails.

    ``grade`` adds the realism chain (grain, halation, CA, grade). It forces a
    re-encode, so it is off by default and the plain mux keeps `-c:v copy`.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Only re-encode when the frame size, frame rate, or grade actually has to
    # change. A plain mux keeps the original stream with `-c:v copy`.
    vf: list[str] = []
    video_codec: list[str] = ["-c:v", "copy"]
    if target is not None:
        width, height = target
        vf = ["-vf", realism_filters(width, height, fps, grade)]
        video_codec = [
            "-c:v", "libx264", "-crf", "18", "-preset", "medium",
            "-pix_fmt", "yuv420p",
        ]

    if audio is not None and audio.is_file():
        cmd = [
            "ffmpeg", "-y", "-i", str(video), "-i", str(audio),
            *vf,
            "-map", "0:v:0", "-map", "1:a:0",
            *video_codec,
            "-c:a", "aac", "-b:a", "192k",
            "-shortest", str(dest),
        ]
    else:
        cmd = [
            "ffmpeg", "-y", "-i", str(video),
            "-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo",
            *vf,
            "-map", "0:v:0", "-map", "1:a:0",
            *video_codec,
            "-c:a", "aac", "-b:a", "192k",
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
    ap.add_argument(
        "--width",
        type=int,
        default=None,
        help="Normalise every clip to this width before concat (default: meta.res_base)",
    )
    ap.add_argument("--height", type=int, default=None, help="Normalise to this height")
    ap.add_argument(
        "--fps",
        type=int,
        default=16,
        help="Normalise every clip to this frame rate before concat (default 16)",
    )
    ap.add_argument(
        "--grade",
        action="store_true",
        help="Add the realism grade: grain, halation, chromatic aberration, colour",
    )
    args = ap.parse_args()

    durations = load_durations(Path(args.spec))
    meta = load_meta(Path(args.spec))
    target = resolve_target_size(meta, args.width, args.height)
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
        # Prefer the lipsynced clip when the render dir contains both. A plain
        # sorted() puts renders/c01.mp4 before renders/lipsync/c01.mp4, so the
        # lipsync pass would be silently discarded.
        lipsynced = [c for c in candidates if "lipsync" in c.parts]
        video = lipsynced[0] if lipsynced else candidates[0]
        duration = durations.get(clip_id, 5.0)

        audio_name = audio_index.get(clip_id)
        audio = Path(args.audio) / audio_name if audio_name else None
        if audio is not None and audio.is_file():
            with_narration += 1

        dest = muxed_dir / f"{clip_id}.mp4"
        if mux_clip(
            video, audio, dest, duration,
            target=target, fps=args.fps, grade=args.grade,
        ):
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
