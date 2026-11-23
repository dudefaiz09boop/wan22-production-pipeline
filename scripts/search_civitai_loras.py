#!/usr/bin/env python3
"""Search Civitai for LoRAs usable with the WAN 2.2 I2V pipeline.

Queries the Civitai API for LoRAs whose base model is WAN 2.2 / WAN Video,
sorted by rating, and prints name, id, version, trigger words and file size.
Read-only: nothing is downloaded.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from pathlib import Path

BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
API = "https://civitai.com/api/v1/models"


def load_env(path: Path = Path(".env")) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def search(query: str, base_models: list[str], limit: int, token: str | None) -> list[dict]:
    params = {
        "query": query,
        "types": "LORA",
        "limit": str(limit),
        "nsfw": "true",
    }
    url = f"{API}?{urllib.parse.urlencode(params)}"
    headers = {"User-Agent": BROWSER_UA, "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=60) as response:
        data = json.loads(response.read())

    wanted = {b.lower() for b in base_models}
    out = []
    for item in data.get("items", []):
        versions = item.get("modelVersions") or []
        if not versions:
            continue
        version = versions[0]
        base = (version.get("baseModel") or "").lower()
        if wanted and not any(w in base for w in wanted):
            continue
        files = [
            f for f in version.get("files", [])
            if str(f.get("name", "")).endswith(".safetensors")
        ]
        if not files:
            continue
        primary = next((f for f in files if f.get("primary")), files[0])
        out.append({
            "name": item.get("name"),
            "model_id": item.get("id"),
            "version_id": version.get("id"),
            "base_model": version.get("baseModel"),
            "rating": round(item.get("stats", {}).get("rating", 0), 2),
            "downloads": item.get("stats", {}).get("downloadCount", 0),
            "trigger": (version.get("trainedWords") or [None])[0],
            "file": primary.get("name"),
            "size_mb": round((primary.get("sizeKB") or 0) / 1024, 1),
            "nsfw": item.get("nsfw"),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Search Civitai for WAN 2.2 LoRAs")
    ap.add_argument("--query", default="wan 2.2")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument(
        "--base-models",
        nargs="+",
        default=["Wan Video", "WAN 2.2", "Wan2.2", "Wan 2.1"],
    )
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    load_env()
    token = os.environ.get("CIVITAI_API_KEY")
    results = search(args.query, args.base_models, args.limit, token)

    if args.json:
        print(json.dumps(results, indent=2))
        return 0

    print(f"{len(results)} WAN LoRA(s) for query {args.query!r}")
    print("-" * 100)
    for item in results:
        print(
            f"{item['rating']:>5}  {item['downloads']:>8}dl  "
            f"{item['size_mb']:>7}MB  {item['base_model']:<14} "
            f"{item['name'][:44]:<44} v{item['version_id']}"
        )
        if item["trigger"]:
            print(f"        trigger: {item['trigger']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
