#!/usr/bin/env python3
"""Measure video-container, timing, decode, black-frame, and freeze metrics."""
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from fractions import Fraction
from pathlib import Path
from typing import Any


BLACK_SEGMENT = re.compile(
    r"black_start:(?P<start>-?\d+(?:\.\d+)?)\s+"
    r"black_end:(?P<end>-?\d+(?:\.\d+)?)\s+"
    r"black_duration:(?P<duration>-?\d+(?:\.\d+)?)"
)
FREEZE_START = re.compile(r"freeze_start:\s*(?P<start>-?\d+(?:\.\d+)?)")
FREEZE_DURATION = re.compile(r"freeze_duration:\s*(?P<duration>-?\d+(?:\.\d+)?)")
FACE_METRIC_RANGES = {
    "face_detection_coverage": (0.0, 1.0),
    "identity_similarity_min": (-1.0, 1.0),
    "landmark_jitter_px": (0.0, math.inf),
    "expression_jump_p95": (0.0, math.inf),
    "head_motion_jitter": (0.0, math.inf),
    "lip_sync_offset_ms": (-math.inf, math.inf),
    "lip_sync_confidence": (0.0, 1.0),
}


class EvaluationError(RuntimeError):
    pass


def run(command: list[str], *, capture: bool = True) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            check=False,
            capture_output=capture,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=30 * 60,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise EvaluationError(f"Could not run {command[0]}: {error}") from error


def parse_rate(value: str | None) -> float | None:
    if not value or value in {"0/0", "N/A"}:
        return None
    try:
        rate = float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return None
    return rate if math.isfinite(rate) and rate > 0 else None


def probe(path: Path, ffprobe: str) -> dict[str, Any]:
    result = run([
        ffprobe,
        "-v", "error",
        "-show_entries",
        "format=duration:stream=index,codec_type,codec_name,width,height,avg_frame_rate,duration,start_time,nb_frames",
        "-of", "json",
        str(path),
    ])
    if result.returncode:
        raise EvaluationError(f"ffprobe failed: {result.stderr.strip()}")
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as error:
        raise EvaluationError(f"ffprobe returned invalid JSON: {error}") from error


def parse_detections(stderr: str) -> tuple[list[dict[str, float]], list[float]]:
    black_segments = [
        {
            "startSeconds": float(match.group("start")),
            "endSeconds": float(match.group("end")),
            "durationSeconds": float(match.group("duration")),
        }
        for match in BLACK_SEGMENT.finditer(stderr)
    ]
    freeze_durations = [
        float(match.group("duration"))
        for match in FREEZE_DURATION.finditer(stderr)
    ]
    if not freeze_durations:
        starts = [float(match.group("start")) for match in FREEZE_START.finditer(stderr)]
        ends = [float(match.group("start")) for match in re.finditer(r"freeze_end:\s*(-?\d+(?:\.\d+)?)", stderr)]
        freeze_durations = [max(0.0, end - start) for start, end in zip(starts, ends)]
    return black_segments, freeze_durations


def load_face_metrics(path: Path | None) -> dict[str, float] | None:
    if path is None:
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvaluationError(f"Could not read face-metrics report {path}: {error}") from error
    metrics = data.get("metrics") if isinstance(data, dict) else None
    if not isinstance(metrics, dict):
        raise EvaluationError("Face-metrics report must contain a 'metrics' object.")
    normalized: dict[str, float] = {}
    for name, bounds in FACE_METRIC_RANGES.items():
        value = metrics.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise EvaluationError(f"Face-metrics report is missing numeric metric {name!r}.")
        number = float(value)
        if not math.isfinite(number) or not bounds[0] <= number <= bounds[1]:
            raise EvaluationError(f"Face-metrics metric {name!r} is outside its valid range.")
        normalized[name] = number
    blink_count = metrics.get("blink_count")
    if isinstance(blink_count, bool) or not isinstance(blink_count, int) or blink_count < 0:
        raise EvaluationError("Face-metrics report must contain a non-negative integer blink_count.")
    normalized["blink_count"] = blink_count
    return normalized


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    video_path = args.video.resolve()
    if not video_path.is_file():
        raise EvaluationError(f"Video file does not exist: {video_path}")
    ffmpeg = args.ffmpeg or shutil.which("ffmpeg")
    ffprobe = args.ffprobe or shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise EvaluationError("Both ffmpeg and ffprobe must be installed or selected with --ffmpeg/--ffprobe.")

    media = probe(video_path, ffprobe)
    streams = media.get("streams")
    if not isinstance(streams, list):
        raise EvaluationError("ffprobe did not report media streams.")
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    if not video_streams:
        raise EvaluationError("The file contains no video stream.")
    video_stream = video_streams[0]
    audio_stream = audio_streams[0] if audio_streams else None
    video_duration = float(video_stream.get("duration") or media.get("format", {}).get("duration") or 0)
    if video_duration <= 0:
        raise EvaluationError("Could not determine a positive video duration.")
    fps = parse_rate(video_stream.get("avg_frame_rate")) or parse_rate(video_stream.get("r_frame_rate"))
    frame_count = int(video_stream["nb_frames"]) if str(video_stream.get("nb_frames", "")).isdigit() else None
    if frame_count is None and fps:
        frame_count = round(video_duration * fps)
    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)
    audio_duration = float(audio_stream.get("duration") or 0) if audio_stream else None
    video_start = float(video_stream.get("start_time") or 0)
    audio_start = float(audio_stream.get("start_time") or 0) if audio_stream else None
    black_filter = f"blackdetect=d=0.1:pix_th={args.black_pixel_threshold}"
    freeze_filter = f"freezedetect=n=-50dB:d={args.freeze_min_seconds}"
    analysis = run([
        ffmpeg,
        "-hide_banner", "-nostats", "-i", str(video_path),
        "-vf", f"{black_filter},{freeze_filter}",
        "-an", "-f", "null", "-",
    ])
    black_segments, freeze_durations = parse_detections(analysis.stderr)
    if analysis.returncode:
        raise EvaluationError(f"ffmpeg artifact analysis failed: {analysis.stderr[-2000:].strip()}")
    decode = run([
        ffmpeg,
        "-hide_banner", "-v", "error", "-i", str(video_path),
        "-map", "0:v:0", "-f", "null", "-",
    ])
    decode_errors = [line.strip() for line in decode.stderr.splitlines() if line.strip()]
    if decode.returncode and not decode_errors:
        decode_errors.append(f"ffmpeg exited with code {decode.returncode}")

    black_seconds = sum(segment["durationSeconds"] for segment in black_segments)
    freeze_seconds = sum(freeze_durations)
    audio_offset_ms = (audio_start - video_start) * 1000 if audio_start is not None else None
    duration_skew_ms = (audio_duration - video_duration) * 1000 if audio_duration is not None else None
    render_seconds = args.render_seconds
    realtime_factor = render_seconds / video_duration if render_seconds is not None else None

    checks = [
        {"name": "video_stream_decodes", "passed": not decode_errors, "measured": len(decode_errors)},
        {"name": "expected_audio_present", "passed": not args.expect_audio or audio_stream is not None,
         "measured": audio_stream is not None},
        {"name": "resolution", "passed": (
            (args.expected_width is None or width == args.expected_width)
            and (args.expected_height is None or height == args.expected_height)
        ), "measured": {"width": width, "height": height}},
        {"name": "frame_rate", "passed": (
            args.expected_fps is None or (fps is not None and abs(fps - args.expected_fps) <= args.fps_tolerance)
        ), "measured": fps},
        {"name": "black_frames", "passed": black_seconds <= args.max_black_seconds,
         "measured": black_seconds},
        {"name": "frozen_frames", "passed": freeze_seconds <= args.max_freeze_seconds,
         "measured": freeze_seconds},
    ]
    if audio_stream:
        checks.extend([
            {"name": "audio_video_start_offset", "passed": (
                audio_offset_ms is not None and abs(audio_offset_ms) <= args.max_av_offset_ms
            ), "measured": audio_offset_ms},
            {"name": "audio_video_duration_skew", "passed": (
                duration_skew_ms is not None and abs(duration_skew_ms) <= args.max_av_duration_skew_ms
            ), "measured": duration_skew_ms},
        ])

    face_metrics = load_face_metrics(args.face_metrics)
    face_checks = []
    if face_metrics is not None:
        thresholds = {
            "face_detection_coverage": args.min_face_coverage,
            "identity_similarity_min": args.min_identity_similarity,
            "landmark_jitter_px": args.max_landmark_jitter_px,
            "lip_sync_offset_ms": args.max_lip_sync_offset_ms,
            "lip_sync_confidence": args.min_lip_sync_confidence,
        }
        for metric, threshold in thresholds.items():
            value = face_metrics[metric]
            passed = value >= threshold if metric in {
                "face_detection_coverage", "identity_similarity_min", "lip_sync_confidence",
            } else abs(value) <= threshold if metric == "lip_sync_offset_ms" else value <= threshold
            face_checks.append({
                "name": metric,
                "passed": passed,
                "measured": value,
                "threshold": threshold,
            })
        for metric, threshold in (
            ("expression_jump_p95", args.max_expression_jump_p95),
            ("head_motion_jitter", args.max_head_motion_jitter),
        ):
            if threshold is not None:
                face_checks.append({
                    "name": metric,
                    "passed": face_metrics[metric] <= threshold,
                    "measured": face_metrics[metric],
                    "threshold": threshold,
                })
        if args.min_blinks_per_minute is not None:
            blink_rate = face_metrics["blink_count"] / video_duration * 60
            face_checks.append({
                "name": "blink_rate_per_minute",
                "passed": blink_rate >= args.min_blinks_per_minute,
                "measured": blink_rate,
                "threshold": args.min_blinks_per_minute,
            })

    all_checks = checks + face_checks
    technical_passed = all(check["passed"] for check in checks)
    face_passed = bool(face_checks) and all(check["passed"] for check in face_checks)
    return {
        "schemaVersion": 1,
        "video": {
            "path": str(video_path),
            "width": width,
            "height": height,
            "fps": fps,
            "frameCount": frame_count,
            "durationSeconds": video_duration,
            "videoCodec": video_stream.get("codec_name"),
            "audioPresent": audio_stream is not None,
            "audioDurationSeconds": audio_duration,
            "audioVideoStartOffsetMs": audio_offset_ms,
            "audioVideoDurationSkewMs": duration_skew_ms,
        },
        "artifacts": {
            "decodeErrors": decode_errors,
            "blackSegments": black_segments,
            "blackSeconds": black_seconds,
            "freezeDurationsSeconds": freeze_durations,
            "freezeSeconds": freeze_seconds,
        },
        "performance": {
            "renderSeconds": render_seconds,
            "realTimeFactor": realtime_factor,
        },
        "faceMetrics": {
            "status": "measured" if face_metrics is not None else "not_measured",
            "source": str(args.face_metrics.resolve()) if args.face_metrics else None,
            "values": face_metrics,
            "limitations": [
                "Identity similarity requires a calibrated face-embedding model and reference image.",
                "Lip-sync offset/confidence must come from a mouth/audio synchrony evaluator; stream timing is not lip-sync accuracy.",
            ],
        },
        "checks": all_checks,
        "technicalPassed": technical_passed,
        "faceMetricsPassed": face_passed,
        "productionReady": technical_passed and face_passed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--expect-audio", action="store_true")
    parser.add_argument("--expected-width", type=int)
    parser.add_argument("--expected-height", type=int)
    parser.add_argument("--expected-fps", type=float)
    parser.add_argument("--fps-tolerance", type=float, default=0.05)
    parser.add_argument("--max-black-seconds", type=float, default=0.0)
    parser.add_argument("--black-pixel-threshold", type=float, default=0.1)
    parser.add_argument("--freeze-min-seconds", type=float, default=0.25)
    parser.add_argument("--max-freeze-seconds", type=float, default=0.25)
    parser.add_argument("--max-av-offset-ms", type=float, default=100.0)
    parser.add_argument("--max-av-duration-skew-ms", type=float, default=150.0)
    parser.add_argument("--render-seconds", type=float)
    parser.add_argument("--face-metrics", type=Path)
    parser.add_argument("--min-face-coverage", type=float, default=0.95)
    parser.add_argument("--min-identity-similarity", type=float, default=0.35)
    parser.add_argument("--max-landmark-jitter-px", type=float, default=12.0)
    parser.add_argument("--max-expression-jump-p95", type=float)
    parser.add_argument("--max-head-motion-jitter", type=float)
    parser.add_argument("--max-lip-sync-offset-ms", type=float, default=100.0)
    parser.add_argument("--min-lip-sync-confidence", type=float, default=0.2)
    parser.add_argument("--min-blinks-per-minute", type=float)
    parser.add_argument("--ffmpeg")
    parser.add_argument("--ffprobe")
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--require-production-ready", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        report = evaluate(args)
    except EvaluationError as error:
        print(f"FAIL {error}", file=sys.stderr)
        return 2
    output = json.dumps(report, indent=2)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(output + "\n", encoding="utf-8")
    print(output)
    if not report["technicalPassed"]:
        return 1
    if args.require_production_ready and not report["productionReady"]:
        print("FAIL production-ready gate requires passing technical and face metrics.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
