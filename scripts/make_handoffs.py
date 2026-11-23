#!/usr/bin/env python3
"""Backfill renders/handoffs/{shot_id}.png from already-rendered segments.

For every chained shot in docs/beats.yaml whose previous shot has a rendered
segment under renders/segments/, extract that segment's FINAL frame (ffmpeg
-sseof, mirroring scripts/render.py) at the segment's native size. Idempotent:
existing handoff images are left alone unless --force is passed.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def load_ordered_shots(beats_path: Path) -> list[tuple[str, bool]]:
    """Ordered (shot_id, chain) pairs read per shot; ids are never hardcoded."""
    data = yaml.safe_load(beats_path.read_text(encoding="utf-8"))
    shots = data.get("shots") or []
    return [(str(shot["id"]), bool(shot.get("chain"))) for shot in shots]


def plan_handoffs(
    ordered: list[tuple[str, bool]],
    segments_dir: Path,
    handoffs_dir: Path,
    *,
    force: bool = False,
) -> list[dict]:
    """Handoff jobs for chained shots whose previous segment exists.

    Entries with segment_ready=False are reported but never extracted.
    """
    plans: list[dict] = []
    for index, (shot_id, chain) in enumerate(ordered):
        if index == 0 or not chain:
            continue
        previous_id = ordered[index - 1][0]
        destination = handoffs_dir / f"{shot_id}.png"
        if destination.is_file() and not force:
            continue
        segment = segments_dir / f"{previous_id}.mp4"
        plans.append({
            "shot_id": shot_id,
            "previous_id": previous_id,
            "segment": segment,
            "destination": destination,
            "segment_ready": segment.is_file(),
        })
    return plans


def run_ffmpeg(command: list[str], timeout: int = 300) -> None:
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
        raise RuntimeError(f"ffmpeg failed ({result.returncode}): {detail}")


def extract_last_frame(ffmpeg: str, segment: Path, destination: Path) -> None:
    """Final frame at the segment's native size; no scaling filters."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".tmp.png")
    temporary.unlink(missing_ok=True)
    run_ffmpeg([
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-sseof", "-0.05", "-i", str(segment),
        "-frames:v", "1", str(temporary),
    ])
    if not temporary.is_file() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg produced no final frame for {segment.name}")
    data = temporary.read_bytes()
    if data[:8] != PNG_SIGNATURE:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg did not produce a PNG handoff for {segment.name}")
    os.replace(temporary, destination)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beats", type=Path, default=ROOT / "docs" / "beats.yaml")
    parser.add_argument("--segments-dir", type=Path, default=ROOT / "renders" / "segments")
    parser.add_argument("--handoffs-dir", type=Path, default=ROOT / "renders" / "handoffs")
    parser.add_argument("--ffmpeg", default=os.environ.get("FFMPEG_PATH", "ffmpeg"))
    parser.add_argument("--force", action="store_true", help="re-extract existing handoffs")
    parser.add_argument("--dry-run", action="store_true", help="list planned handoffs only")
    args = parser.parse_args(argv)

    ordered = load_ordered_shots(args.beats)
    plans = plan_handoffs(ordered, args.segments_dir, args.handoffs_dir, force=args.force)
    if not plans:
        print("All chained handoffs are present; nothing to do", flush=True)
        return 0

    created = 0
    failed = 0
    for plan in plans:
        if args.dry_run:
            state = "ready" if plan["segment_ready"] else "segment-missing"
            print(
                f"would extract {plan['destination'].name} from {plan['previous_id']} "
                f"({state})",
                flush=True,
            )
            continue
        if not plan["segment_ready"]:
            print(
                f"{plan['shot_id']}: previous segment {plan['segment'].name} is "
                "missing; skipped",
                flush=True,
            )
            failed += 1
            continue
        try:
            extract_last_frame(args.ffmpeg, plan["segment"], plan["destination"])
        except (RuntimeError, subprocess.TimeoutExpired) as error:
            print(f"{plan['shot_id']}: {error}", flush=True)
            failed += 1
            continue
        created += 1
        print(
            f"{plan['shot_id']}: handoff extracted from {plan['previous_id']} "
            f"-> {plan['destination'].name}",
            flush=True,
        )
    if args.dry_run:
        print(f"Dry run: {len(plans)} handoff(s) planned", flush=True)
        return 0
    print(f"Created {created} handoff(s); {failed} skipped/failed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.TimeoutExpired) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
