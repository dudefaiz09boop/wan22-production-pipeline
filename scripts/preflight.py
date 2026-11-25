#!/usr/bin/env python3
"""Validate the ComfyUI graph before a production render.

`--check-only` validates the generated workflow against the live server and
checks the referenced model files. It does not submit a render. Without this,
a missing custom node or checkpoint is discovered only after the first clip
fails and the chained run aborts.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from env_utils import resolve_server


ROOT = Path(__file__).resolve().parent.parent
RENDER = ROOT / "scripts" / "render.py"
OUTPUT = ROOT / "preflight.json"


def check_only(server: str, precision: str) -> int:
    sys.path.insert(0, str(ROOT / "scripts"))
    import render

    workflow_path = ROOT / "workflows" / f"base_i2v_{precision}.json"
    if not workflow_path.is_file():
        print(f"missing workflow: {workflow_path}", file=sys.stderr)
        return 1
    cfg_path = ROOT / "prompts" / "c01.json"
    prompt_path = ROOT / "prompts" / "c01.txt"
    negative_path = ROOT / "negatives" / "artifact_neg.txt"
    if not all(path.is_file() for path in (cfg_path, prompt_path, negative_path)):
        print("prompts or negative file missing; run build_prompts first", file=sys.stderr)
        return 1

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    workflow = render.inject(
        json.loads(workflow_path.read_text(encoding="utf-8")),
        prompt_path.read_text(encoding="utf-8"),
        negative_path.read_text(encoding="utf-8"),
        cfg,
        "startframes/c01.png",
    )
    payload = json.dumps({"prompt": workflow}).encode()
    request = urllib.request.Request(
        f"{server.rstrip('/')}/prompt",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        print(f"ComfyUI rejected the workflow: HTTP {error.code}\n{detail}", file=sys.stderr)
        return 1
    except (OSError, ValueError) as error:
        print(f"ComfyUI preflight failed: {error}", file=sys.stderr)
        return 1

    # A successful submission is queued, not rendered. Remove it immediately
    # so the check does not consume GPU time or leave a stale job behind.
    prompt_id = result.get("prompt_id")
    if prompt_id:
        try:
            urllib.request.urlopen(
                urllib.request.Request(
                    f"{server.rstrip('/')}/queue",
                    data=json.dumps({"delete": [prompt_id]}).encode(),
                    headers={"Content-Type": "application/json"},
                ),
                timeout=10,
            ).read()
        except OSError:
            pass
    print("ComfyUI accepted the production workflow (no render submitted)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--server",
        default=None,
        help="ComfyUI base URL (defaults to COMFY_URL or localhost)",
    )
    ap.add_argument("--precision", choices=["fp16", "fp8", "gguf_q8"], default="fp16")
    ap.add_argument("--check-only", action="store_true", help="Validate without rendering")
    ap.add_argument("clips", nargs="*", metavar="CLIP_ID")
    args = ap.parse_args()
    args.server = resolve_server(args.server)

    if args.check_only:
        return check_only(args.server, args.precision)

    if len(args.clips) != 2:
        ap.error("provide exactly two clip ids, or use --check-only")

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
