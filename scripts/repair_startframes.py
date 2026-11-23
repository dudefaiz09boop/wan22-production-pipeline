#!/usr/bin/env python3
"""Regenerate only the start frames that failed the quality audit.

Reads startframes/quality_report.json, then re-runs generate_stills.py for
just the failing beats with more candidates and a higher identity gate. This
is the repair path for disfigured or faceless frames without redoing all 70.

Usage:
    python scripts/repair_startframes.py --dry-run
    python scripts/repair_startframes.py
    python scripts/repair_startframes.py --beats c31 c39 c45
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPORT = Path("startframes/quality_report.json")


def failing_beats(report: Path) -> list[str]:
    if not report.is_file():
        raise SystemExit(f"missing {report}; run scripts/audit_startframes.py first")
    data = json.loads(report.read_text(encoding="utf-8"))
    return [d["id"] for d in data["details"] if not d["ok"]]


def main() -> int:
    ap = argparse.ArgumentParser(description="Regenerate failing start frames")
    ap.add_argument("--report", default=str(REPORT))
    ap.add_argument("--beats", nargs="+", help="Override the failing list")
    ap.add_argument("--candidates", type=int, default=4, help="Candidates per beat")
    ap.add_argument("--gate", type=float, default=0.45, help="Identity gate")
    ap.add_argument(
        "--sdxl-model",
        default="models/Juggernaut-XL-v9/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
        help="SDXL base model path",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    beats = args.beats or failing_beats(Path(args.report))
    if not beats:
        print("no failing frames")
        return 0

    print(f"{len(beats)} frame(s) to regenerate: {', '.join(beats)}")
    if args.dry_run:
        print("(dry run)")
        return 0

    cmd = [
        sys.executable, "scripts/generate_stills.py",
        "--prompts", "prompts",
        "--output", "startframes",
        "--references", "references",
        "--loras", "models/loras/trained",
        "--sdxl-model", args.sdxl_model,
        "--spec", "docs/beats.yaml",
        "--swapper-model", "models/inswapper_128.onnx",
        "--candidates", str(args.candidates),
        "--gate", str(args.gate),
        "--beats", *beats,
    ]
    print("running:", " ".join(cmd[:6]), "...")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("regeneration failed", file=sys.stderr)
        return result.returncode

    print("\nre-auditing...")
    subprocess.run(
        [sys.executable, "scripts/audit_startframes.py", "--dir", "startframes"]
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
