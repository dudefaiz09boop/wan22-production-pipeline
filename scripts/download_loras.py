#!/usr/bin/env python3
"""Download trained character LoRAs from Kaggle (sofiya009/lora-nuke-faiz-juv)."""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB chunks
MAX_RETRIES = 5
RETRY_BACKOFF = 2
TIMEOUT = 300  # 5 minutes per request


class DownloadError(Exception):
    """Custom exception for download errors."""
    pass


def create_session() -> requests.Session:
    """Create a requests session with retry strategy."""
    session = requests.Session()
    retry_strategy = Retry(
        total=MAX_RETRIES,
        backoff_factor=RETRY_BACKOFF,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["HEAD", "GET", "OPTIONS"],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def get_file_size(url: str, session: requests.Session, token: Optional[str] = None) -> int:
    """Get the remote file size via HEAD request."""
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = session.head(url, headers=headers, timeout=30, allow_redirects=True)
    response.raise_for_status()
    content_length = response.headers.get("Content-Length")
    if content_length is None:
        raise DownloadError(f"Server did not return Content-Length for {url}")
    return int(content_length)


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    sha256 = hashlib.sha256()
    with filepath.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            sha256.update(chunk)
    return sha256.hexdigest().upper()


def verify_file(filepath: Path, expected_size: int, expected_sha256: Optional[str] = None) -> bool:
    """Verify file size and optionally SHA-256."""
    if not filepath.exists():
        return False
    stat = filepath.stat()
    if stat.st_size != expected_size:
        print(f"  Size mismatch: {stat.st_size} != {expected_size}")
        return False
    if expected_sha256:
        print(f"  Verifying SHA-256...")
        actual_sha256 = compute_sha256(filepath)
        if actual_sha256 != expected_sha256.upper():
            print(f"  SHA-256 mismatch: {actual_sha256} != {expected_sha256}")
            return False
    return True


def download_file(
    url: str,
    dest: Path,
    expected_size: int,
    expected_sha256: Optional[str] = None,
    token: Optional[str] = None,
) -> bool:
    """Download a file with resume support and progress tracking."""
    session = create_session()
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    dest.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest.with_suffix(dest.suffix + ".part")
    resume_byte = 0
    if part_path.exists():
        resume_byte = part_path.stat().st_size
        if resume_byte >= expected_size:
            if verify_file(part_path, expected_size, expected_sha256):
                part_path.rename(dest)
                return True
            else:
                part_path.unlink()
                resume_byte = 0

    if resume_byte > 0:
        headers["Range"] = f"bytes={resume_byte}-"
        print(f"  Resuming from {resume_byte / (1024**2):.1f} MB")

    try:
        response = session.get(url, headers=headers, stream=True, timeout=TIMEOUT)
        if resume_byte > 0 and response.status_code != 206:
            print(f"  Server doesn't support resume (status {response.status_code}), restarting")
            part_path.unlink(missing_ok=True)
            resume_byte = 0
            headers.pop("Range", None)
            response = session.get(url, headers=headers, stream=True, timeout=TIMEOUT)

        response.raise_for_status()

        total_size = expected_size
        downloaded = resume_byte
        start_time = time.time()
        last_progress_time = start_time

        mode = "ab" if resume_byte > 0 else "wb"
        with part_path.open(mode) as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)

                    now = time.time()
                    if now - last_progress_time >= 2.0:
                        pct = (downloaded / total_size) * 100
                        speed = (downloaded - resume_byte) / (now - start_time) / (1024**2)
                        eta = (total_size - downloaded) / (speed * 1024**2) if speed > 0 else 0
                        print(
                            f"  {downloaded / (1024**2):.1f} / {total_size / (1024**2):.1f} MB "
                            f"({pct:.1f}%) @ {speed:.1f} MB/s, ETA: {eta:.0f}s"
                        )
                        last_progress_time = now

        if verify_file(part_path, expected_size, expected_sha256):
            part_path.rename(dest)
            print(f"  ✓ Downloaded and verified: {dest.name}")
            return True
        else:
            raise DownloadError("File verification failed after download")

    except Exception as e:
        print(f"  ✗ Download failed: {e}")
        return False


# Trained LoRA configurations from sofiya009/lora-nuke-faiz-juv
TRAINED_LORAS = {
    "faiz": {
        "filename": "faiz_lora.safetensors",
        "kaggle_path": "faiz_lora.safetensors",
        "description": "Faiz character LoRA",
        "trigger": "faiz",
        "strength": 0.85,
    },
    "juv": {
        "filename": "juv_lora.safetensors",
        "kaggle_path": "juv_lora.safetensors",
        "description": "Juveriya character LoRA",
        "trigger": "juv",
        "strength": 0.85,
    },
    "both": {
        "filename": "faiz_juv_lora.safetensors",
        "kaggle_path": "faiz_juv_lora.safetensors",
        "description": "Combined Faiz + Juveriya LoRA",
        "trigger": "faiz juv",
        "strength": 0.8,
    },
}


def download_trained_loras(
    dest_dir: Path,
    kaggle_username: str = "sofiya009",
    kaggle_dataset: str = "lora-nuke-faiz-juv",
    token: Optional[str] = None,
    force: bool = False,
) -> dict[str, bool]:
    """Download trained LoRAs from Kaggle dataset."""
    results = {}
    base_url = f"https://www.kaggle.com/api/v1/datasets/download/{kaggle_username}/{kaggle_dataset}"

    print(f"Downloading trained LoRAs from Kaggle: {kaggle_username}/{kaggle_dataset}")
    print(f"Destination: {dest_dir}")
    print("-" * 60)

    dest_dir.mkdir(parents=True, exist_ok=True)

    # A LoRA already staged locally (for example one recovered from a Kaggle
    # run) is authoritative. Never re-download or retrain it. The extra roots
    # cover the training output directories, which is where a locally trained
    # file actually lands; without them a 404 from Kaggle was fatal even though
    # the weights were sitting in the repo.
    #
    # Order matters: a freshly trained file must beat an older copy. Picking the
    # first match meant a stale loras/faiz_lora.safetensors silently shadowed a
    # new training run, so retraining appeared to change nothing.
    local_sources = [
        Path("models/loras/trained"),
        Path("lora_output"),
        Path("lora_out3"),
        Path("juv_out"),
        Path("recover"),
        Path("loras"),
        Path("training_data"),
    ]

    for lora_key, lora_info in TRAINED_LORAS.items():
        filename = lora_info["filename"]
        dest = dest_dir / filename

        print(f"[{lora_key}] {filename} - {lora_info['description']}")

        candidates = [
            src / filename for src in local_sources if (src / filename).is_file()
        ]
        if dest.exists():
            candidates.append(dest)

        if candidates and not force:
            newest = max(candidates, key=lambda p: p.stat().st_mtime)
            if newest != dest:
                shutil.copy2(newest, dest)
                print(f"  [OK] Copied newest local copy: {newest}")
            else:
                print("  [OK] Already exists (use --force to re-download)")
            results[lora_key] = True
            continue

        # Kaggle download URL for specific file
        file_url = f"{base_url}/{lora_info['kaggle_path']}"

        session = create_session()
        try:
            expected_size = get_file_size(file_url, session, token)
        except Exception as e:
            print(f"  [FAIL] Failed to get file size: {e}")
            results[lora_key] = False
            continue

        success = download_file(file_url, dest, expected_size, None, token)
        results[lora_key] = success

    return results


def update_lora_registry(lora_dir: Path, registry_path: Path) -> None:
    """Update workflows/loras.json with trained LoRA entries."""
    registry = {}
    if registry_path.exists():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))

    # Add trained LoRA entries
    trained_entries = {
        "faiz_trained": {
            "high": str(lora_dir / "faiz_lora.safetensors"),
            "low": str(lora_dir / "faiz_lora.safetensors"),
            "w": 0.85,
            "trigger": "faiz",
            "trigger_scope": "prompt_prefix",
            "source": {
                "kaggle_dataset": "sofiya009/lora-nuke-faiz-juv",
                "kaggle_file": "faiz_lora.safetensors",
            },
        },
        "juv_trained": {
            "high": str(lora_dir / "juv_lora.safetensors"),
            "low": str(lora_dir / "juv_lora.safetensors"),
            "w": 0.85,
            "trigger": "juv",
            "trigger_scope": "prompt_prefix",
            "source": {
                "kaggle_dataset": "sofiya009/lora-nuke-faiz-juv",
                "kaggle_file": "juv_lora.safetensors",
            },
        },
        "both_trained": {
            "high": str(lora_dir / "faiz_juv_lora.safetensors"),
            "low": str(lora_dir / "faiz_juv_lora.safetensors"),
            "w": 0.8,
            "trigger": "faiz juv",
            "trigger_scope": "prompt_prefix",
            "source": {
                "kaggle_dataset": "sofiya009/lora-nuke-faiz-juv",
                "kaggle_file": "faiz_juv_lora.safetensors",
            },
        },
    }

    registry.update(trained_entries)

    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")
    print(f"Updated LoRA registry: {registry_path}")


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Download trained character LoRAs from Kaggle")
    ap.add_argument("--dest", default="models/loras/trained", help="Destination directory")
    ap.add_argument("--kaggle-username", default="sofiya009", help="Kaggle username")
    ap.add_argument("--kaggle-dataset", default="lora-nuke-faiz-juv", help="Kaggle dataset name")
    ap.add_argument("--token", help="Kaggle/HF token (or set KAGGLE_KEY/HF_TOKEN env var)")
    ap.add_argument("--force", action="store_true", help="Force re-download")
    ap.add_argument("--update-registry", action="store_true", help="Update workflows/loras.json")
    ap.add_argument("--registry-path", default="workflows/loras.json", help="LoRA registry path")
    args = ap.parse_args()

    token = args.token or os.environ.get("KAGGLE_KEY") or os.environ.get("HF_TOKEN")
    dest_dir = Path(args.dest)

    results = download_trained_loras(
        dest_dir,
        args.kaggle_username,
        args.kaggle_dataset,
        token,
        args.force,
    )

    print("-" * 60)
    success_count = sum(1 for v in results.values() if v)
    print(f"Completed: {success_count}/{len(results)} LoRAs successful")

    if success_count != len(results):
        failed = [k for k, v in results.items() if not v]
        print(f"Failed: {', '.join(failed)}")
        return 1

    if args.update_registry:
        update_lora_registry(dest_dir, Path(args.registry_path))

    return 0


if __name__ == "__main__":
    sys.exit(main())