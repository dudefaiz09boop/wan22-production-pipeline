#!/usr/bin/env python3
"""Add the general NSFW quality LoRA to explicit shots that declare none.

Only touches shots whose cam text contains explicit anatomy markers and whose
`loras` list is empty. Never edits cam lines, never touches shots that already
declare a LoRA (those are already tuned), and never exceeds the validator's
three-LoRA budget.

Run with --apply to write; without it, prints the proposed change.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SPEC = Path("docs/beats.yaml")

# Markers that indicate explicit anatomy is on screen.
EXPLICIT = re.compile(
    r"\b(cock|shaft|penis|vulva|pussy|clit|labia|anus|rim|nipple|breast|"
    r"cum|semen|fluid|slick|wet|thrust|penetrat|insert|fuck|sex)\b",
    re.IGNORECASE,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--lora", default="nsfw_general")
    args = ap.parse_args()

    text = SPEC.read_text(encoding="utf-8")
    lines = text.splitlines()

    # Locate each shot block and its loras line.
    shot_starts = [
        (i, m.group(1))
        for i, line in enumerate(lines)
        if (m := re.match(r"^  - id: (c\d+)$", line))
    ]

    proposals = []
    for idx, (start, sid) in enumerate(shot_starts):
        end = shot_starts[idx + 1][0] if idx + 1 < len(shot_starts) else len(lines)
        block = lines[start:end]
        cam_text = " ".join(
            line.strip().lstrip("- ").strip('"')
            for line in block
            if line.strip().startswith("- ")
        )
        loras_line = next(
            (j for j, line in enumerate(block) if line.strip().startswith("loras:")),
            None,
        )
        if loras_line is None:
            continue
        current = block[loras_line].split(":", 1)[1].strip()
        if current != "[]":
            continue
        if not EXPLICIT.search(cam_text):
            continue
        proposals.append((start + loras_line, sid))

    if not proposals:
        print("no shots to update")
        return 0

    print(f"{len(proposals)} explicit shot(s) with no LoRA:")
    for _, sid in proposals:
        print(f"  {sid}")

    if not args.apply:
        print("\n(dry run; pass --apply to write)")
        return 0

    for line_no, _ in sorted(proposals, reverse=True):
        indent = lines[line_no][: len(lines[line_no]) - len(lines[line_no].lstrip())]
        lines[line_no] = f"{indent}loras: [{args.lora}]"

    SPEC.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nupdated {len(proposals)} shot(s) in {SPEC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
