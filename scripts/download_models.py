#!/usr/bin/env python3
"""Resumable model downloads from Hugging Face with progress tracking."""
from __future__ import annotations

import hashlib
import os
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
    progress_callback=None,
) -> bool:
    """Download a file with resume support and progress tracking."""
    session = create_session()
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    # Check if partial file exists
    dest.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest.with_suffix(dest.suffix + ".part")
    resume_byte = 0
    if part_path.exists():
        resume_byte = part_path.stat().st_size
        if resume_byte >= expected_size:
            # Part file is complete or oversized, verify and rename
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

                    # Progress callback
                    if progress_callback:
                        progress_callback(downloaded, total_size)

                    # Print progress every 2 seconds
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

        # Verify and finalize
        if verify_file(part_path, expected_size, expected_sha256):
            part_path.rename(dest)
            print(f"  ✓ Downloaded and verified: {dest.name}")
            return True
        else:
            raise DownloadError("File verification failed after download")

    except Exception as e:
        print(f"  [FAIL] Download failed: {e}")
        return False


def download_models(
    repo_id: str,
    files: list[str],
    dest_dir: Path,
    token: Optional[str] = None,
    sha256_map: Optional[dict[str, str]] = None,
) -> dict[str, bool]:
    """Download multiple files from a Hugging Face repo."""
    results = {}
    base_url = f"https://huggingface.co/{repo_id}/resolve/main"

    print(f"Downloading from {repo_id}")
    print(f"Destination: {dest_dir}")
    print(f"Files: {len(files)}")
    print("-" * 60)

    for i, file_path in enumerate(files, 1):
        filename = Path(file_path).name
        dest = dest_dir / filename
        url = f"{base_url}/{file_path}"

        print(f"[{i}/{len(files)}] {filename}")

        # Check if already exists and verified
        expected_sha256 = sha256_map.get(filename) if sha256_map else None
        if dest.exists():
            # We need to get expected size from server
            session = create_session()
            try:
                expected_size = get_file_size(url, session, token)
                if verify_file(dest, expected_size, expected_sha256):
                    print(f"  [OK] Already exists and verified")
                    results[filename] = True
                    continue
            except Exception:
                pass  # Will re-download

        # Get expected size
        session = create_session()
        try:
            expected_size = get_file_size(url, session, token)
        except Exception as e:
            print(f"  [FAIL] Failed to get file size: {e}")
            results[filename] = False
            continue

        success = download_file(
            url,
            dest,
            expected_size,
            expected_sha256,
            token,
        )
        results[filename] = success

    return results


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Download WAN 2.2 models from Hugging Face")
    ap.add_argument("--repo", required=True, help="Hugging Face repo ID")
    ap.add_argument("--files", nargs="+", required=True, help="Files to download (relative paths in repo)")
    ap.add_argument("--dest", required=True, help="Destination directory")
    ap.add_argument("--token", help="Hugging Face token (or set HF_TOKEN env var)")
    ap.add_argument("--sha256-file", help="JSON file with SHA-256 hashes")
    args = ap.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    dest_dir = Path(args.dest)
    dest_dir.mkdir(parents=True, exist_ok=True)

    sha256_map = {}
    if args.sha256_file:
        import json
        with open(args.sha256_file) as f:
            sha256_map = json.load(f)

    results = download_models(args.repo, args.files, dest_dir, token, sha256_map)

    print("-" * 60)
    success_count = sum(1 for v in results.values() if v)
    print(f"Completed: {success_count}/{len(results)} files successful")

    if success_count != len(results):
        failed = [k for k, v in results.items() if not v]
        print(f"Failed: {', '.join(failed)}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())