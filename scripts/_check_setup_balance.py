#!/usr/bin/env python3
"""Sanity-check the setup shell script's block nesting.

Handles `done < file` redirection and skips Python heredoc bodies.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

SCRIPT = Path("scripts/setup_aic_fast.sh")

OPENERS = re.compile(r"^(for|while|if|case)\s")
CLOSERS = {"done", "fi", "esac"}


def main() -> int:
    lines = SCRIPT.read_text(encoding="utf-8").splitlines()
    stack: list[tuple[int, str]] = []
    in_heredoc = False
    problems: list[str] = []

    for number, line in enumerate(lines, 1):
        stripped = line.strip()

        if in_heredoc:
            if stripped == "PY":
                in_heredoc = False
            continue
        if "<<'PY'" in line:
            in_heredoc = True
            continue

        if OPENERS.match(stripped):
            stack.append((number, stripped[:44]))
            continue

        first = stripped.split()[0] if stripped else ""
        if first in CLOSERS:
            if not stack:
                problems.append(f"line {number}: unmatched {first}")
                continue
            stack.pop()

    for number, text in stack:
        problems.append(f"line {number}: unclosed {text}")

    if problems:
        for problem in problems:
            print(f"  BAD {problem}")
        return 1

    print(f"  OK  block nesting balanced ({len(lines)} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
