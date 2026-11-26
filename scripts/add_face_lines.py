#!/usr/bin/env python3
"""Put a face in frame on the beats whose cam lines never show one.

The startframe seeds the video, so it has to establish who is in the shot. 14
beats are written entirely in hands, wrists, props and anatomy -- c03 is a hand
snapping around a wrist, c10 is a close-up of a vulva -- and SDXL dutifully
produced no face at all. The clip then opens on a faceless frame.

A prompt-level anchor was tried and is not enough: the concrete nouns in those
cam lines outvote it. This rewrites one cam line per affected beat so the face
is part of the action rather than an instruction bolted on.

Idempotent: a line that already carries its replacement is left alone.

Usage:
    python scripts/add_face_lines.py --check
    python scripts/add_face_lines.py --write
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SPEC = Path("docs/beats.yaml")

# (beat id, cam index) -> replacement line. Each keeps the original camera verb
# and the original action; the face is folded into it.
REPLACEMENTS: dict[tuple[str, int], str] = {
    ("c03", 1): (
        "track side: yanking her, her face twisted, heels skidding on mosaic"
    ),
    ("c10", 3): (
        "static medium: Faiz bracing the desk edge, his jaw tight, hips settling"
    ),
    ("c13", 3): (
        "pan up: her face, mouth open, framed art rattling on the nail"
    ),
    ("c22", 1): (
        "tilt down: her face in the dark, the glass falling, bouncing, rolling"
    ),
    ("c27", 1): (
        "tilt up: her face slack, the desk creaking, her leg sliding to the floor"
    ),
    ("c39", 5): (
        "pan out: her face, unimpressed, straightening the desk, pausing at an ink stain"
    ),
    ("c43", 3): (
        "track: her face, shaking her head, dropping them in the bin by the almirah"
    ),
    ("c46", 1): (
        "zoom in slow: her face, the varnish, a faint smear under the shine"
    ),
    ("c47", 2): (
        "zoom in: her face at the peephole, the corridor dark, one bulb out"
    ),
    ("c51", 1): (
        "tilt down: Juveriya's face, a tomato rolling off, stopping at Faiz's foot"
    ),
    ("c52", 1): (
        "pan slow: Ammi's face, the squared desk, the bin with the shredded cotton"
    ),
    ("c58", 1): (
        "zoom in: his face, the shirt parting, her knuckles against his chest"
    ),
    ("c63", 1): (
        "zoom in: his face, his hand slamming the desk beside Juveriya's hip"
    ),
    ("c70", 1): (
        "pan: their faces, the swept floor, the shoes kicked under the rack"
    ),
}

SHOT_RE = re.compile(r"^\s*-\s+id:\s*(\S+)\s*$")
CAM_RE = re.compile(r'^(\s*)-\s+"(.*)"\s*$')


def apply(text: str) -> tuple[str, list[str], list[str]]:
    """Return (new_text, applied, missing)."""
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    current: str | None = None
    cam_index = -1
    in_cam = False
    applied: list[str] = []
    seen: set[tuple[str, int]] = set()

    for line in lines:
        shot = SHOT_RE.match(line)
        if shot:
            current = shot.group(1)
            cam_index = -1
            in_cam = False
            out.append(line)
            continue

        if re.match(r"^\s+cam:\s*$", line):
            in_cam = True
            out.append(line)
            continue

        if in_cam:
            cam = CAM_RE.match(line)
            if cam:
                cam_index += 1
                key = (current, cam_index) if current else None
                if key and key in REPLACEMENTS:
                    indent = cam.group(1)
                    replacement = REPLACEMENTS[key]
                    seen.add(key)
                    if cam.group(2) != replacement:
                        out.append(f'{indent}- "{replacement}"\n')
                        applied.append(f"{current}[{cam_index}]")
                        continue
            elif line.strip() and not line.startswith((" ", "\t")):
                in_cam = False

        out.append(line)

    missing = [f"{b}[{i}]" for (b, i) in REPLACEMENTS if (b, i) not in seen]
    return "".join(out), applied, missing


def main() -> int:
    ap = argparse.ArgumentParser(description="Add a face line to faceless beats")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if not SPEC.is_file():
        print(f"missing {SPEC}", file=sys.stderr)
        return 1

    original = SPEC.read_text(encoding="utf-8")
    updated, applied, missing = apply(original)

    print(f"cam lines replaced: {len(applied)}")
    if applied:
        print("  " + ", ".join(applied))
    if missing:
        print(f"not found: {', '.join(missing)}", file=sys.stderr)

    if args.check or not args.write:
        return 0

    SPEC.write_text(updated, encoding="utf-8")
    print(f"wrote {SPEC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
