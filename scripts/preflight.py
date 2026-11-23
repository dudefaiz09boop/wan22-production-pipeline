#!/usr/bin/env python3
"""Time two single-clip submissions through render.py."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from env_utils import resolve_server


ROOT = Path(__file__).resolve().parent.parent
RENDER = ROOT / "scripts" / "render.py"
OUTPUT = ROOT / "preflight.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--server",
        default=None,
        help="ComfyUI base URL (defaults to COMFY_URL or localhost)",
    )
    ap.add_argument("clips", nargs=2, metavar="CLIP_ID")
    args = ap.parse_args()
    args.server = resolve_server(args.server)

    results = []
    for clip in args.clips:
        started = time.monotonic()
        try:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(RENDER),
                    "--server",
                    args.server,
                    "--no-resume",
                    "--from",
                    clip,
                    "--to",
                    clip,
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
            )
            log = completed.stdout
            if completed.stderr:
                log += completed.stderr
            ok = completed.returncode == 0
        except OSError as exc:
            log = f"{type(exc).__name__}: {exc}"
            ok = False

        result = {
            "clip": clip,
            "seconds": time.monotonic() - started,
            "ok": ok,
            "log_tail": "\n".join(log.splitlines()[-20:]),
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False))

    OUTPUT.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if all(result["ok"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
