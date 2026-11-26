#!/usr/bin/env python3
"""Raise the act coverage on the beats that are supposed to show it.

The story is explicit penetrative sex between the two leads, but the cam lines
only named penetration on 9 of 70 beats and oral on 3. Scenes 5 and 6 -- the
middle of the act -- had almost none, so the video model had nothing to render
for the part of the film that matters most.

This rewrites specific cam lines to name the act, keeping the camera verb and
the original blocking. Idempotent.

Usage:
    python scripts/amplify_acts.py --check
    python scripts/amplify_acts.py --write
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SPEC = Path("docs/beats.yaml")

# (beat id, cam index) -> replacement line.
REPLACEMENTS: dict[tuple[str, int], str] = {
    # Scene 2 -- first penetration, standing
    ("c05", 2): (
        "static close-up: fingers pushing back in, her vulva stretching glossy, "
        "his cock sliding in beside them"
    ),
    ("c09", 3): (
        "static close-up: his shaft buried to the base, her pussy stretched "
        "around it, wet ring at the root"
    ),
    # Scene 3 -- onto the desk
    ("c10", 3): (
        "static medium: Faiz bracing the desk edge, his jaw tight, cock pushing "
        "into her, hips settling"
    ),
    ("c12", 2): (
        "arc profile: her palm shoving his chest, arm shaking, his shaft driving "
        "in and out of her"
    ),
    # Scene 4 -- face down, spanked, held
    ("c15", 2): (
        "handheld: her body jerking, gasp, his cock slamming into her from "
        "behind, her hand slapping his arm"
    ),
    ("c16", 2): (
        "tilt up: her jaw slack, hair tangled across her wet mouth, his shaft "
        "pounding into her"
    ),
    ("c18", 3): (
        "static: third slap, the mirror ringing on the wall, his cock still "
        "buried in her arse"
    ),
    ("c19", 1): (
        "tilt up: twisting them, her hips shoving back onto his hand, his cock "
        "thrusting into her arse"
    ),
    # Scene 5 -- blackout, the middle of the act
    ("c20", 0): (
        "static wide: Faiz driving deep, his cock buried in her, Juveriya's "
        "spine dipping into the desk"
    ),
    ("c21", 5): (
        "static: moonlight stripes across Juveriya's back, Faiz's shaft sliding "
        "in and out of her, wet shine at the join"
    ),
    ("c22", 3): (
        "static: Faiz pushing deep, slow grind, his cock buried in her, her "
        "outline flattening"
    ),
    ("c23", 3): (
        "static: fingers pressing, her hips bucking back onto his shaft, taking "
        "him to the base"
    ),
    # Scene 6 -- the climax
    ("c25", 1): (
        "dolly in: his fingers denting her hip, the marks holding, his cock "
        "pounding into her"
    ),
    ("c27", 0): (
        "static: Faiz's weight down on Juveriya, ribs compressing, both gasping, "
        "his cock still buried, his cum sliding out of her"
    ),
    # Scene 7 -- after
    ("c28", 0): (
        "static medium: Faiz pulls his cock out of Juveriya, her face turned on "
        "the desk, hips lifting after him, shaft wet"
    ),
    # Oral -- the story calls for it and only 3 beats showed it
    ("c29", 4): (
        "static: a slow kiss on her mouth, her hand climbing to his neck, then "
        "down to his cock"
    ),
    ("c39", 0): (
        "static close-up: the wet patch on the desk varnish, Juveriya's fingers "
        "hovering over it, her mouth still wet from him"
    ),
}

SHOT_RE = re.compile(r"^\s*-\s+id:\s*(\S+)\s*$")
CAM_RE = re.compile(r'^(\s*)-\s+"(.*)"\s*$')


def apply(text: str) -> tuple[str, list[str], list[str]]:
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
    ap = argparse.ArgumentParser(description="Raise act coverage on the cam lines")
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
