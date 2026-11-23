#!/usr/bin/env python3
"""Prepare an AIC Cloud .pem key for SSH on Windows.

Windows OpenSSH refuses a private key that other users can read, and `chmod`
does not exist there. This applies the equivalent ACL with icacls, then
prints the exact ssh and scp commands for the instance.

Usage:
    python scripts/setup_ssh_key.py --key "%USERPROFILE%\\.ssh\\aic-gpu-gpu-wan22.pem" \
        --host gpu-868186.aiccloud.one --port 40581
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def lock_down_windows(path: Path) -> bool:
    """Remove inherited ACLs and grant read only to the current user."""
    user = os.environ.get("USERNAME")
    if not user:
        print("USERNAME is not set; cannot build the icacls grant", file=sys.stderr)
        return False
    result = subprocess.run(
        [
            "icacls", str(path),
            "/inheritance:r",
            "/grant:r", f"{user}:(R)",
        ],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"icacls failed: {result.stderr.strip() or result.stdout.strip()}", file=sys.stderr)
        return False
    return True


def lock_down_posix(path: Path) -> bool:
    result = subprocess.run(["chmod", "600", str(path)], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"chmod failed: {result.stderr.strip()}", file=sys.stderr)
        return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare an AIC Cloud .pem key for SSH")
    ap.add_argument("--key", required=True, help="Path to the .pem private key")
    ap.add_argument("--host", help="Instance hostname or IP")
    ap.add_argument("--port", type=int, default=22, help="SSH port")
    ap.add_argument("--user", default="root", help="SSH user")
    args = ap.parse_args()

    key = Path(os.path.expandvars(args.key)).expanduser()
    if not key.is_file():
        print(f"key not found: {key}", file=sys.stderr)
        print("Download it from the AIC Cloud dashboard (SSH KEY -> Download).", file=sys.stderr)
        return 1

    print(f"key: {key}  ({key.stat().st_size} bytes)")

    if os.name == "nt":
        if not lock_down_windows(key):
            return 1
        print("permissions: locked to the current user (icacls)")
    else:
        if not lock_down_posix(key):
            return 1
        print("permissions: 600 (chmod)")

    if args.host:
        print()
        print("ssh:")
        print(f'  ssh -i "{key}" -p {args.port} {args.user}@{args.host}')
        print()
        print("scp (note capital -P for the port):")
        print(
            f'  scp -i "{key}" -P {args.port} '
            f'"<local-file>" {args.user}@{args.host}:/workspace/'
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
