"""ComfyUI endpoint discovery helpers."""
from __future__ import annotations

import http.client
import json
import os
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SERVER = "http://127.0.0.1:8188"
PROBE_TIMEOUT = 6


def _dotenv_server() -> str | None:
    path = ROOT / ".env"
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return None

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        key, separator, value = stripped.partition("=")
        if not separator:
            continue
        key = key.strip()
        if key.startswith("export "):
            key = key[7:].strip()
        if key != "COMFY_URL":
            continue

        value = value.strip()
        if value.startswith(("'", '"')):
            quote = value[0]
            closing_quote = value.find(quote, 1)
            if closing_quote == -1:
                return None
            value = value[1:closing_quote]
        else:
            value = value.split("#", 1)[0].strip()
        return value or None
    return None


def probe(server: str) -> str | None:
    """Return a named ComfyUI device when its system_stats endpoint is healthy."""
    url = f"{server.rstrip('/')}/system_stats"
    try:
        with urllib.request.urlopen(url, timeout=PROBE_TIMEOUT) as response:
            if response.getcode() != 200:
                return None
            data = json.loads(response.read())
    except (
        OSError,
        ValueError,
        urllib.error.URLError,
        http.client.HTTPException,
    ):
        return None

    if not isinstance(data, dict):
        return None
    devices = data.get("devices")
    if not isinstance(devices, list):
        return None
    for device in devices:
        if isinstance(device, dict):
            name = device.get("name")
            if isinstance(name, str) and name.strip():
                return name.strip()
    return None


def resolve_server(explicit: str | None) -> str:
    """Select the first healthy ComfyUI endpoint in configured priority order."""
    candidates = (
        (explicit, "--server"),
        (os.environ.get("COMFY_URL"), "$env:COMFY_URL"),
        (_dotenv_server(), ".env COMFY_URL"),
        (DEFAULT_SERVER, "default"),
    )
    tried: set[str] = set()
    for candidate, source in candidates:
        if not isinstance(candidate, str) or not candidate.strip():
            continue
        server = candidate.strip().rstrip("/")
        if server in tried:
            continue
        tried.add(server)
        device_name = probe(server)
        if device_name:
            print(f"ComfyUI server {server} ({source}; device: {device_name})")
            return server

    raise SystemExit(
        "No ComfyUI server returned HTTP 200 with a named device from "
        f"/system_stats (timeout {PROBE_TIMEOUT}s per candidate). Start the "
        "Kaggle ComfyUI notebook, copy its current tunnel base URL, then set "
        "$env:COMFY_URL='https://<current-tunnel>' or pass --server and retry. "
        f"Local fallback: {DEFAULT_SERVER}."
    )
