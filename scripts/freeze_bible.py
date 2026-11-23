#!/usr/bin/env python3
"""Freeze generated bible JSON files into a SHA-256 manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIBLE = ROOT / "bible"
MANIFEST = BIBLE / "manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest() -> dict:
    files = sorted(
        path for path in BIBLE.rglob("*")
        if path.is_file() and path != MANIFEST
        and path.suffix.lower() == ".json"
    )
    return {
        "bible_version": 1,
        "algorithm": "sha256",
        "files": [
            {"path": path.relative_to(BIBLE).as_posix(), "sha256": sha256(path)}
            for path in files
        ],
    }


def freeze(*, check: bool = False) -> list[str]:
    payload = build_manifest()
    expected = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if check:
        if not MANIFEST.is_file():
            return ["missing bible/manifest.json"]
        if MANIFEST.read_text(encoding="utf-8") != expected:
            return ["bible manifest is stale"]
        mismatches = []
        for entry in payload["files"]:
            path = BIBLE / entry["path"]
            if not path.is_file() or sha256(path) != entry["sha256"]:
                mismatches.append(entry["path"])
        return [f"bible checksum mismatch: {path}" for path in mismatches]
    BIBLE.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(expected, encoding="utf-8")
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    errors = freeze(check=args.check)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"Frozen {len(build_manifest()['files'])} bible JSON file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
