#!/usr/bin/env python3
"""Raise the explicit coverage of the cam lines.

The beat sheet was written with the action implied rather than shown: POV
appeared in 1 of 70 beats, breasts in 9, pussy in 6, cum in 6. Wan 2.2 renders
what the prompt names, so an implied shot comes back as a generic one.

This replaces specific cam lines with versions that name the anatomy and the
act, keeping the camera verb and staying under the 220-char limit validate.py
enforces. It is idempotent: a line that already carries its replacement is left
alone.

Usage:
    python scripts/amplify_beats.py --check
    python scripts/amplify_beats.py --write
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SPEC = Path("docs/beats.yaml")

# (beat id, cam index) -> replacement line.
REPLACEMENTS: dict[tuple[str, int], str] = {
    # --- POV: the audience is Faiz ---
    ("c01", 2): (
        "static medium: POV, Faiz crowding her, forearm braced by her head, "
        "her breasts pressed flat against his chest"
    ),
    ("c25", 0): (
        "static medium: POV over Faiz's shoulder, frantic rhythm, his hips "
        "snapping, Juveriya's hips meeting his"
    ),
    ("c55", 3): "zoom in: POV, Faiz at the bookshelf, half turned, reading her",
    ("c60", 0): (
        "static wide: POV, Juveriya seated on the desk, driving up into Faiz, "
        "heels hooked on the drawer"
    ),
    # --- breasts ---
    ("c01", 3): (
        "tilt down: his hand flat on her waist, thumb pressing the kurti into "
        "her hip, the cloth pulling tight over her breasts"
    ),
    ("c03", 4): (
        "static medium: his forearm braced by her head, Juveriya's breasts "
        "heaving under the kurti, nipples showing through the cotton"
    ),
    ("c12", 0): (
        "static close-up: sweat rolling down Juveriya's stomach, skirt rucked, "
        "her breasts swinging loose above it"
    ),
    ("c14", 0): (
        "static wide: desk shuddering, Juveriya's thighs slapping Faiz's hips, "
        "her breasts bouncing with every thrust"
    ),
    ("c15", 4): (
        "arc: Faiz's grin, laughing against her skin, her breasts crushed "
        "against his chest"
    ),
    ("c21", 4): (
        "handheld: his hand gripping her hip in the dark, fingers digging, "
        "her breasts swaying"
    ),
    ("c32", 1): (
        "static: she steps into them, pulling up under the skirt, her breasts "
        "bare under his open shirt"
    ),
    ("c33", 4): (
        "static: palms flat, the welt showing under Faiz's open shirt, her "
        "breasts pressed to the prayer mat"
    ),
    ("c56", 4): (
        "static: Juveriya's hand going to his collar, straightening it, then "
        "not, her breasts lifting with her breath"
    ),
    ("c58", 3): (
        "static: Faiz reaching for her kurti hem, Juveriya's hand stopping "
        "him, her breasts moving under the cloth"
    ),
    ("c62", 1): (
        "handheld shake: Juveriya folding forward onto Faiz's chest, her "
        "breasts crushed against him"
    ),
    # --- pussy ---
    ("c04", 3): (
        "tilt down: panties dragged down her calves, elastic biting, her "
        "pussy bare and wet in the gap of the torn skirt"
    ),
    ("c08", 3): (
        "tilt down: wet shine on both thighs, her pussy stretched around his "
        "shaft, fluid at the crease"
    ),
    ("c11", 5): (
        "static close-up: her heel bouncing off his back, his shaft wet and "
        "glistening, her pussy lips dragging with each stroke"
    ),
    ("c13", 5): (
        "static hold: Juveriya's legs shaking, toes curling, her pussy taking "
        "him to the base, wet sound continuous"
    ),
    ("c20", 2): (
        "handheld: Faiz grinding, his hips rolling, hers answering, her pussy "
        "gripping him wet"
    ),
    ("c29", 5): (
        "zoom out slow: the kiss holding, Juveriya's leg hanging off the "
        "desk, her pussy wet and open"
    ),
    ("c31", 0): (
        "static close-up: shredded cotton on the mosaic, elastic snapped, her "
        "bare foot beside it, her pussy still wet between her thighs"
    ),
    ("c57", 3): (
        "tilt down: her leg coming up around his hip, skirt riding, her pussy "
        "bare under it"
    ),
    # --- ass ---
    ("c16", 3): (
        "static: her cheek pushed into the wood grain, one eye open, her arse "
        "raised and taking him"
    ),
    ("c17", 0): (
        "static top-down: Juveriya pinned face down, spine curved, skirt at "
        "her waist, her bare arse up and split by his shaft"
    ),
    # --- cum ---
    ("c24", 3): (
        "static: her walls clamping, Faiz's hips stalling mid-thrust, his cum "
        "flooding her"
    ),
    ("c26", 4): (
        "zoom in: joined hips, wet friction, his cum leaking out around his "
        "shaft, fluid at the base"
    ),
    ("c27", 0): (
        "static: Faiz's weight down on Juveriya, ribs compressing, both "
        "gasping, his cum sliding out of her"
    ),
    ("c63", 4): (
        "tilt down: his cum spilling at the join, her thigh wet, dripping "
        "onto the desk"
    ),
    ("c64", 0): (
        "static medium: Juveriya flat on her back on the desk, Faiz's head on "
        "her stomach, his cum drying on her thigh"
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
    ap = argparse.ArgumentParser(description="Raise explicit coverage of cam lines")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if not SPEC.is_file():
        print(f"missing {SPEC}", file=sys.stderr)
        return 1

    original = SPEC.read_text(encoding="utf-8")
    updated, applied, missing = apply(original)

    print(f"cam lines replaced: {len(applied)}")
    if missing:
        print(f"not found (already applied or moved): {', '.join(missing)}")

    if args.check or not args.write:
        return 0

    SPEC.write_text(updated, encoding="utf-8")
    print(f"wrote {SPEC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
