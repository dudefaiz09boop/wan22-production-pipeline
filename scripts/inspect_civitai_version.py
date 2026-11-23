#!/usr/bin/env python3
"""Inspect specific Civitai model versions: base model, trigger, files."""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from pathlib import Path

BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"


def load_env(path: Path = Path(".env")) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def main() -> int:
    load_env()
    token = os.environ.get("CIVITAI_API_KEY")
    headers = {"User-Agent": BROWSER_UA, "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    for version_id in sys.argv[1:]:
        url = f"https://civitai.com/api/v1/model-versions/{version_id}"
        try:
            with urllib.request.urlopen(
                urllib.request.Request(url, headers=headers), timeout=40
            ) as response:
                data = json.loads(response.read())
        except Exception as error:  # noqa: BLE001
            print(f"v{version_id}: FAILED {error}")
            continue

        print(f"v{version_id}  base={data.get('baseModel')}")
        print(f"  trigger: {data.get('trainedWords')}")
        for item in data.get("files", []):
            if item["name"].endswith(".safetensors"):
                size = round((item.get("sizeKB") or 0) / 1024, 1)
                print(f"  file: {item['name']}  {size} MB")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
