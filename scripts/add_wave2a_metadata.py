#!/usr/bin/env python3
"""Add locked Wave 2a per-shot metadata to the production beat sheet."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BEATS = ROOT / "docs" / "beats.yaml"

NOISE_3 = {"c12", "c18", "c22", "c35", "c36", "c37"}
HEROES = {"c25", "c33", "c44", "c61", "c68"}
THREADS = {"c09", "c30", "c66", "c69", "c70"}

GEOGRAPHY = {
    1: "narrow hallway",
    2: "study room and desk",
    3: "study desk",
    4: "study desk",
    5: "study room in blackout",
    6: "study desk",
    7: "study aftermath area",
    8: "study and flat phone position",
    9: "study and flat passage",
    10: "hallway, kitchen and front door",
    11: "study desk",
    12: "hallway and front door",
}


def scene_number(block: str) -> int:
    match = re.search(r"^    scene: (\d+)$", block, re.M)
    return int(match.group(1)) if match else 1


def continuity(block: str) -> str:
    match = re.search(r'^    continuity: (.+)$', block, re.M)
    if not match:
        return "tracked continuity"
    value = match.group(1).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    return value.replace('"', '\\"')


def light_for(shot_number: int) -> str:
    if shot_number <= 4:
        return "tube_flicker"
    if 21 <= shot_number <= 23:
        return "blackout_moonlight"
    if 47 <= shot_number <= 54:
        return "daylight"
    if 55 <= shot_number <= 64:
        return "evening_tube"
    if shot_number >= 65:
        return "daylight"
    return "night_tube"


def mark_for(shot_number: int) -> str:
    if shot_number <= 14:
        return "none"
    if shot_number <= 26:
        return "fresh"
    if shot_number <= 46:
        return "tender"
    if shot_number <= 54:
        return "fading"
    return "yellow"


def main() -> int:
    text = BEATS.read_text(encoding="utf-8")
    text = re.sub(
        r"\n    noise: .*?\n    (continuity|cam):",
        r"\n    \1:",
        text,
        flags=re.M | re.S,
    )
    parts = re.split(r"(?=^  - id: c\d{2}$)", text, flags=re.M)
    head = parts[0]
    blocks = parts[1:]
    output = [head]

    for block in blocks:
        match = re.search(r"^  - id: (c\d{2})$", block, re.M)
        if not match:
            output.append(block)
            continue
        shot_id = match.group(1)
        shot_number = int(shot_id[1:])
        scene = scene_number(block)
        light = light_for(shot_number)
        mark = mark_for(shot_number)
        noise = 3 if shot_id in NOISE_3 else 2
        duration = 4 if shot_id in NOISE_3 else 5
        thread = "secrecy" if shot_id in THREADS else ""
        profile = "hero" if shot_id in HEROES else "draft"
        face = "Faiz and Juveriya faces as authored"
        prop = continuity(block)

        metadata = (
            f"    noise: {noise}\n"
            f"    duration: {duration}\n"
            f"    light: {light}\n"
            f"    mark_state: {mark}\n"
            f'    thread: "{thread}"\n'
            f"    profile: {profile}\n"
            f"    audio_mode: none\n"
            "    anchors:\n"
            f'      face: "{face}"\n'
            '      body: "adult posture as established by the authored cam line"\n'
            '      wardrobe: "tracked wardrobe and undressing state"\n'
            f'      prop: "{prop}"\n'
            f'      light: "{light}"\n'
            f'      geography: "{GEOGRAPHY[scene]}"\n'
        )

        if re.search(r"^    continuity:", block, re.M):
            block = re.sub(
                r"^    continuity:",
                metadata + "    continuity:",
                block,
                count=1,
                flags=re.M,
            )
        else:
            block = re.sub(
                r"^    cam:",
                metadata + "    cam:",
                block,
                count=1,
                flags=re.M,
            )
        output.append(block)

    BEATS.write_text("".join(output), encoding="utf-8")
    print(f"Added Wave 2a metadata to {len(blocks)} shots")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(1)
