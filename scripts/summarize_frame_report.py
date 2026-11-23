#!/usr/bin/env python3
"""Summarise a start-frame quality report."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPORT = Path("startframes/quality_report.json")


def main() -> int:
    if not REPORT.is_file():
        print(f"missing {REPORT}; run scripts/audit_startframes.py first", file=sys.stderr)
        return 1
    data = json.loads(REPORT.read_text(encoding="utf-8"))
    bad = [d for d in data["details"] if not d["ok"]]

    print(f"total {data['total']}  ok {data['ok']}  problems {data['problems']}")
    print()

    print("worst sharpness")
    for item in sorted(bad, key=lambda d: d["sharpness"])[:10]:
        print(
            f"  {item['id']}: sharp={item['sharpness']:<7} "
            f"luma={item['mean_luma']:<6} faces={len(item['faces'])}"
        )
    print()

    no_face = [d["id"] for d in bad if not d["faces"]]
    small = [d["id"] for d in bad if d["faces"] and "face too small" in " ".join(d["warnings"])]
    dark = [d["id"] for d in bad if "too dark" in " ".join(d["warnings"])]
    blurry = [d["id"] for d in bad if "blurry" in " ".join(d["warnings"])]

    print(f"no face detected ({len(no_face)}): {', '.join(no_face)}")
    print(f"face too small ({len(small)}): {', '.join(small)}")
    print(f"too dark ({len(dark)}): {', '.join(dark)}")
    print(f"blurry ({len(blurry)}): {', '.join(blurry)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
