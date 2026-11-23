"""Stdlib helpers for the ComfyUI HTTP API."""
from __future__ import annotations

import hashlib
import json
import mimetypes
import os
import urllib.request
import uuid
from pathlib import Path

DEFAULT_SERVER = "http://127.0.0.1:8188"


def server_default() -> str:
    configured = os.environ.get("COMFY_URL", "").strip()
    return configured.rstrip("/") or DEFAULT_SERVER


def upload(server: str, path: Path, cache: dict[Path, str]) -> str:
    """Upload a local image once per run and return its ComfyUI input path."""
    local_path = path.resolve(strict=True)
    if not local_path.is_file():
        raise IsADirectoryError(f"ComfyUI upload is not a file: {local_path}")
    if local_path in cache:
        return cache[local_path]

    boundary = f"----ComfyUpload{uuid.uuid4().hex}"
    subfolder = (
        "pipeline/"
        + hashlib.sha256(str(local_path).encode("utf-8")).hexdigest()[:16]
    )
    filename = (
        local_path.name.replace("\\", "_")
        .replace('"', "_")
        .replace("\r", "_")
        .replace("\n", "_")
    )
    content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    parts = [
        (
            f"--{boundary}\r\n"
            'Content-Disposition: form-data; name="image"; '
            f'filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n\r\n"
        ).encode("utf-8"),
        local_path.read_bytes(),
        b"\r\n",
    ]
    for name, value in (
        ("type", "input"),
        ("overwrite", "true"),
        ("subfolder", subfolder),
    ):
        parts.append(
            (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
                f"{value}\r\n"
            ).encode("utf-8")
        )
    parts.append(f"--{boundary}--\r\n".encode("ascii"))

    request = urllib.request.Request(
        f"{server.rstrip('/')}/upload/image",
        data=b"".join(parts),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        result = json.loads(response.read())
    if not isinstance(result, dict) or not isinstance(result.get("name"), str):
        raise ValueError("ComfyUI upload response is missing an image name")
    remote_subfolder = result.get("subfolder", "")
    if not isinstance(remote_subfolder, str):
        raise ValueError("ComfyUI upload response has an invalid subfolder")

    remote_path = "/".join(
        part.strip("/")
        for part in (remote_subfolder, result["name"])
        if part.strip("/")
    )
    cache[local_path] = remote_path
    return remote_path
