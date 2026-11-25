#!/usr/bin/env python3
"""Sanity-check the setup shell script's block nesting.

Handles `done < file` redirection, skips Python heredoc bodies, and copes with
single-line constructs such as `if [[ ... ]]; then X=1; fi`. The previous
version only looked at the first token of each line, so any one-line `if`
was reported as unclosed even though bash accepts it.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

SCRIPT = Path("scripts/setup_aic_fast.sh")

OPENERS = {"if", "for", "while", "until", "case", "select"}
CLOSERS = {"fi", "done", "esac"}

# Split on shell metacharacters and whitespace so keywords are isolated even
# when several statements share a line.
TOKEN_SPLIT = re.compile(r"[;&|(){}<>\s]+")

LOOP_OPENERS = {"for", "while", "until", "select"}


def strip_comment(line: str) -> str:
    """Drop a trailing `#` comment that is not inside quotes."""
    out: list[str] = []
    quote: str | None = None
    for index, char in enumerate(line):
        if quote:
            out.append(char)
            if char == quote:
                quote = None
        elif char in "'\"":
            quote = char
            out.append(char)
        elif char == "#" and (index == 0 or line[index - 1] in " \t;"):
            break
        else:
            out.append(char)
    return "".join(out)


def strip_quoted(line: str) -> str:
    """Remove quoted spans so keywords inside strings are not counted."""
    return re.sub(r"'[^']*'|\"[^\"]*\"", " ", line)


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

        cleaned = strip_quoted(strip_comment(line))
        for token in TOKEN_SPLIT.split(cleaned):
            if token in OPENERS:
                stack.append((number, token))
            elif token in CLOSERS:
                if not stack:
                    problems.append(f"line {number}: unmatched {token}")
                    continue
                opener_line, opener = stack.pop()
                if token == "fi" and opener != "if":
                    problems.append(
                        f"line {number}: fi closes {opener} from line {opener_line}"
                    )
                elif token == "esac" and opener != "case":
                    problems.append(
                        f"line {number}: esac closes {opener} from line {opener_line}"
                    )
                elif token == "done" and opener not in LOOP_OPENERS:
                    problems.append(
                        f"line {number}: done closes {opener} from line {opener_line}"
                    )

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
