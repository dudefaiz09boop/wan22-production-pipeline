#!/usr/bin/env python3
"""Download the anatomy LoRAs declared in workflows/loras.json from Civitai.

The beats require `penis_insert` and `vag_detail` on specific shots. Those
weights are not in the repo, so a GPU run fails at ComfyUI execution unless
they are staged first. This script resolves each registry entry's Civitai
model/version, downloads the file, and verifies it is a real safetensors
file before placing it where ComfyUI expects it.

Usage:
    python scripts/download_anatomy_loras.py
    python scripts/download_anatomy_loras.py --dest /workspace/models/loras
    python scripts/download_anatomy_loras.py --only penis_insert vag_detail
    python scripts/download_anatomy_loras.py --check
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

REGISTRY_PATH = Path("workflows") / "loras.json"
CIVITAI_API = "https://civitai.com/api/v1"
CHUNK = 8 * 1024 * 1024
TIMEOUT = 300
RETRIES = 4


def load_registry(path: Path) -> dict:
    if not path.is_file():
        raise SystemExit(f"registry not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def civitai_entries(registry: dict) -> dict[str, dict]:
    """Return registry entries that carry a Civitai source."""
    out = {}
    for key, entry in registry.items():
        if not isinstance(entry, dict):
            continue
        source = entry.get("source")
        if not isinstance(source, dict):
            continue
        # Source may be flat or per-stage (high/low).
        if "civitai_model" in source:
            out[key] = {"model": source.get("civitai_model"),
                        "version": source.get("civitai_version")}
            continue
        stages = {}
        for stage in ("high", "low"):
            stage_source = source.get(stage)
            if isinstance(stage_source, dict) and stage_source.get("civitai_model"):
                stages[stage] = {
                    "model": stage_source.get("civitai_model"),
                    "version": stage_source.get("civitai_version"),
                }
        if stages:
            out[key] = stages
    return out


def api_get(path: str, token: str | None) -> dict:
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    last_error = None
    for attempt in range(RETRIES):
        try:
            response = requests.get(f"{CIVITAI_API}{path}", headers=headers, timeout=60)
            if response.status_code == 404:
                raise SystemExit(f"Civitai 404 for {path} - model or version removed")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            last_error = error
            time.sleep(2 ** attempt)
    raise SystemExit(f"Civitai request failed for {path}: {last_error}")


def resolve_download(model_id: int, version_id: int | None, token: str | None) -> tuple[str, str]:
    """Return (download_url, filename) for a Civitai model version."""
    if version_id:
        data = api_get(f"/model-versions/{version_id}", token)
    else:
        model = api_get(f"/models/{model_id}", token)
        versions = model.get("modelVersions") or []
        if not versions:
            raise SystemExit(f"model {model_id} has no versions")
        data = versions[0]

    files = data.get("files") or []
    # Prefer a safetensors primary file.
    safetensors = [f for f in files if str(f.get("name", "")).endswith(".safetensors")]
    chosen = None
    for candidate in safetensors:
        if candidate.get("primary"):
            chosen = candidate
            break
    if chosen is None and safetensors:
        chosen = safetensors[0]
    if chosen is None:
        raise SystemExit(f"no safetensors file for model {model_id} version {version_id}")

    url = chosen.get("downloadUrl")
    name = chosen.get("name")
    if not url or not name:
        raise SystemExit(f"incomplete file metadata for model {model_id}")
    return url, name


def looks_like_safetensors(path: Path) -> bool:
    """A safetensors file starts with a little-endian u64 header length."""
    try:
        with path.open("rb") as handle:
            head = handle.read(8)
        if len(head) != 8:
            return False
        header_len = int.from_bytes(head, "little")
        size = path.stat().st_size
        return 0 < header_len < size
    except OSError:
        return False


def download(url: str, dest: Path, token: str | None) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    headers = {"Authorization": f"Bearer {token}"} if token else {}

    resume = part.stat().st_size if part.exists() else 0
    if resume:
        headers["Range"] = f"bytes={resume}-"

    try:
        with requests.get(url, headers=headers, stream=True, timeout=TIMEOUT) as response:
            if resume and response.status_code != 206:
                part.unlink(missing_ok=True)
                resume = 0
                headers.pop("Range", None)
                response.close()
                with requests.get(url, headers=headers, stream=True, timeout=TIMEOUT) as retry:
                    retry.raise_for_status()
                    _write(retry, part, "wb")
            else:
                response.raise_for_status()
                _write(response, part, "ab" if resume else "wb")
    except requests.RequestException as error:
        print(f"  download failed: {error}", file=sys.stderr)
        return False

    if not looks_like_safetensors(part):
        print("  downloaded file is not valid safetensors", file=sys.stderr)
        part.unlink(missing_ok=True)
        return False

    part.replace(dest)
    return True


def _write(response, part: Path, mode: str) -> None:
    total = 0
    with part.open(mode) as handle:
        for chunk in response.iter_content(chunk_size=CHUNK):
            if chunk:
                handle.write(chunk)
                total += len(chunk)
    print(f"  wrote {total / 1024**2:.1f} MB")


def main() -> int:
    ap = argparse.ArgumentParser(description="Download anatomy LoRAs from Civitai")
    ap.add_argument("--registry", default=str(REGISTRY_PATH))
    ap.add_argument("--dest", default="loras", help="Destination directory")
    ap.add_argument("--only", nargs="+", help="Only these registry keys")
    ap.add_argument("--check", action="store_true", help="Report status without downloading")
    ap.add_argument("--force", action="store_true", help="Re-download existing files")
    args = ap.parse_args()

    token = os.environ.get("CIVITAI_API_KEY")
    if not token and not args.check:
        print("WARNING: CIVITAI_API_KEY is not set; downloads may be rate limited", file=sys.stderr)

    registry = load_registry(Path(args.registry))
    entries = civitai_entries(registry)
    if args.only:
        entries = {k: v for k, v in entries.items() if k in set(args.only)}

    if not entries:
        print("no Civitai-backed LoRA entries found", file=sys.stderr)
        return 1

    dest_dir = Path(args.dest)
    failures = []

    for key, spec in entries.items():
        entry = registry[key]
        print(f"[{key}]")
        stages = spec if "model" not in spec else {"single": spec}

        for stage, ids in stages.items():
            target_rel = entry.get(stage) if stage != "single" else (entry.get("high") or entry.get("low"))
            if not target_rel:
                print(f"  {stage}: no target path in registry, skipped")
                continue
            target = dest_dir / Path(target_rel).name

            if target.is_file() and not args.force:
                if looks_like_safetensors(target):
                    print(f"  {stage}: OK {target.name} ({target.stat().st_size / 1024**2:.1f} MB)")
                    continue
                print(f"  {stage}: existing file is invalid, re-downloading")

            if args.check:
                print(f"  {stage}: MISSING {target.name}")
                failures.append(f"{key}/{stage}")
                continue

            try:
                url, name = resolve_download(ids["model"], ids.get("version"), token)
            except SystemExit as error:
                print(f"  {stage}: {error}", file=sys.stderr)
                failures.append(f"{key}/{stage}")
                continue

            print(f"  {stage}: {name} -> {target}")
            if download(url, target, token):
                print(f"  {stage}: OK ({target.stat().st_size / 1024**2:.1f} MB)")
            else:
                failures.append(f"{key}/{stage}")

    print("-" * 60)
    if failures:
        print(f"FAILED: {', '.join(failures)}", file=sys.stderr)
        return 1
    print("all anatomy LoRAs present")
    return 0


if __name__ == "__main__":
    sys.exit(main())
