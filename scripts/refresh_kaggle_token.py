#!/usr/bin/env python3
"""Refresh the Kaggle CLI OAuth access token.

Kaggle's CLI access token expires roughly every 3 hours. When it does, every
`kaggle` command fails with "Authentication required" or a misleading
"Permission denied" on private kernels. This refreshes it in place using the
stored refresh token, without a browser round trip.

Usage:
    python scripts/refresh_kaggle_token.py
    python scripts/refresh_kaggle_token.py --check
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path

CREDENTIALS = Path.home() / ".kaggle" / "credentials.json"
TOKEN_URL = "https://www.kaggle.com/api/v1/oauth2/token"


def load(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"missing {path}; run `kaggle auth login` first")
    return json.loads(path.read_text(encoding="utf-8"))


def expires_at(creds: dict) -> datetime | None:
    raw = creds.get("access_token_expiration")
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def refresh(creds: dict) -> dict:
    body = urllib.parse.urlencode({
        "grant_type": "refresh_token",
        "refresh_token": creds["refresh_token"],
        "client_id": "kaggle-cli",
    }).encode()
    request = urllib.request.Request(
        TOKEN_URL, data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        data = json.loads(response.read())

    expiry = datetime.now(timezone.utc) + timedelta(seconds=int(data["expires_in"]))
    creds["access_token"] = data["access_token"]
    creds["access_token_expiration"] = expiry.isoformat()
    return creds


def main() -> int:
    ap = argparse.ArgumentParser(description="Refresh the Kaggle CLI access token")
    ap.add_argument("--check", action="store_true", help="Report expiry without refreshing")
    args = ap.parse_args()

    creds = load(CREDENTIALS)
    expiry = expires_at(creds)
    now = datetime.now(timezone.utc)

    if expiry:
        remaining = (expiry - now).total_seconds()
        print(f"token expires {expiry.isoformat()} ({remaining / 60:.0f} min left)")
        if args.check:
            return 0 if remaining > 60 else 1
        if remaining > 300:
            print("still valid; nothing to do")
            return 0
    elif args.check:
        print("no expiry recorded")
        return 1

    if "refresh_token" not in creds:
        print("no refresh_token; run `kaggle auth login`", file=sys.stderr)
        return 1

    creds = refresh(creds)
    CREDENTIALS.write_text(json.dumps(creds, indent=2), encoding="utf-8")
    print(f"refreshed; new expiry {creds['access_token_expiration']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
