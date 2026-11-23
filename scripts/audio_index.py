#!/usr/bin/env python3
"""Resolve a beat's staged audio sidecar.

Ingested sidecars are staged with their upload UUID as the filename, so the
historical `refs/audio/<beat>.wav` path rarely exists. Resolution order:

1. `refs/audio/<beat>.wav` (hand-placed or renamed file) wins;
2. otherwise a `refs/audio/index.json` mapping of `{"beats": {"c01": "<file>"}}`
   is consulted, and the named file must exist next to the index.

`audio_mode: s2v` is only valid when a resolution succeeds.
"""
from __future__ import annotations

import json
from pathlib import Path


def index_path(root: Path) -> Path:
    return root / "refs" / "audio" / "index.json"


def resolve_audio(root: Path, beat_id: str) -> Path | None:
    direct = root / "refs" / "audio" / f"{beat_id}.wav"
    if direct.is_file():
        return direct
    index = index_path(root)
    if not index.is_file():
        return None
    try:
        data = json.loads(index.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    beats = data.get("beats")
    if not isinstance(beats, dict):
        return None
    name = beats.get(beat_id)
    if not isinstance(name, str) or not name:
        return None
    if "/" in name or "\\" in name:
        return None
    candidate = index.parent / name
    if candidate.is_file():
        return candidate
    return None
