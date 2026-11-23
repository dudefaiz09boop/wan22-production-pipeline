#!/usr/bin/env python3
"""Pull rendered results from an AIC Cloud GPU instance to this machine.

AIC Cloud GPU instances are created in the dashboard and expose no GPU API,
so the instance cannot push to your laptop on its own. This script runs
locally and pulls the deliverable over SSH or HTTP.

AIC GPU instances use a generated .pem key and a non-standard SSH port, so
both are first-class options here.

Usage:
    python scripts/pull_results.py --host root@gpu-868186.aiccloud.one \
        --port 40581 --key ~/.ssh/aic-gpu-gpu-wan22.pem
    python scripts/pull_results.py --http http://38.29.145.28:8000
    python scripts/pull_results.py --host root@1.2.3.4 --dest D:/renders
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from pathlib import Path

DEFAULT_REMOTE_DIR = "/workspace/output"
ARTIFACTS = ("deliverable.zip", "final.mp4")


def pull_ssh(host: str, remote_dir: str, dest: Path, port: int | None, key: str | None) -> int:
    if not shutil.which("scp"):
        print("scp not found. Install OpenSSH client, or use --http.", file=sys.stderr)
        return 1
    dest.mkdir(parents=True, exist_ok=True)

    base = ["scp", "-o", "StrictHostKeyChecking=accept-new"]
    if port:
        base += ["-P", str(port)]
    if key:
        base += ["-i", str(Path(key).expanduser())]

    failures = 0
    for name in ARTIFACTS:
        remote = f"{host}:{remote_dir}/{name}"
        print(f"scp {remote} -> {dest}")
        result = subprocess.run(base + [remote, str(dest)])
        if result.returncode != 0:
            print(f"  failed: {name}", file=sys.stderr)
            failures += 1
    return 1 if failures else 0


def pull_http(base_url: str, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    base = base_url.rstrip("/")
    failures = 0
    for name in ARTIFACTS:
        url = f"{base}/{name}"
        target = dest / name
        print(f"GET {url} -> {target}")
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                target.write_bytes(response.read())
            size_mb = target.stat().st_size / 1024**2
            print(f"  saved {size_mb:.1f} MB")
        except Exception as error:  # noqa: BLE001 - report and continue
            print(f"  failed: {error}", file=sys.stderr)
            failures += 1
    return 1 if failures else 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Pull rendered results from AIC Cloud")
    ap.add_argument("--host", help="SSH target, e.g. root@gpu-868186.aiccloud.one")
    ap.add_argument("--http", help="HTTP base URL, e.g. http://38.29.145.28:8000")
    ap.add_argument("--port", type=int, help="SSH port (AIC GPU instances are not 22)")
    ap.add_argument("--key", help="Path to the .pem private key")
    ap.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR, help="Remote output directory")
    ap.add_argument("--dest", default="renders_local", help="Local destination directory")
    args = ap.parse_args()

    if not args.host and not args.http:
        ap.error("provide --host or --http")

    dest = Path(args.dest)
    if args.host:
        return pull_ssh(args.host, args.remote_dir, dest, args.port, args.key)
    return pull_http(args.http, dest)


if __name__ == "__main__":
    sys.exit(main())
