#!/usr/bin/env python3
"""Download the Wan 2.2 GGUF checkpoints from Civitai.

`auto_model_selector.py` only knows about Hugging Face, and the Comfy-Org repo
ships no GGUF files at all. The workflows load the Civitai 2053259 pair, so
without this script a fresh instance renders nothing: ComfyUI reports the
model as missing on the first clip.

Two pairs are staged:

  fm  Fast Move        Lightning distillation baked in  -> no LightX2V LoRA
  cf  Consistent Face  no distillation                  -> needs LightX2V LoRA

Usage:
    python scripts/download_wan22_gguf.py
    python scripts/download_wan22_gguf.py --dest /workspace/models
    python scripts/download_wan22_gguf.py --only fm
    python scripts/download_wan22_gguf.py --check
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import requests

CIVITAI_API = "https://civitai.com/api/v1"
CHUNK = 8 * 1024 * 1024
TIMEOUT = 600
RETRIES = 4

# Civitai sits behind Cloudflare, which rejects the default Python user agent
# with "error code: 1010". A browser user agent is required.
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

MODEL_ID = 2053259

# version_id -> (build, stage, filename, sha256 or None)
#
# The LOW Fast Move filename really is missing its opening paren. That is the
# author's typo and the hash lookup in run_production.py is keyed on the
# literal string, so it must not be "fixed" here.
GGUF_VERSIONS: dict[int, tuple[str, str, str, str | None]] = {
    2540892: (
        "fm",
        "high",
        "Wan2_2-I2V-A14B-NSFW_fast_move-HIGH_Q8(lightning_edition)V2.gguf",
        "DEB6DCF39FEFEF7850D1DD49A52B70A64C65B8D8E7B5F03BC1B6A541CF58AAE9",
    ),
    2540896: (
        "fm",
        "low",
        "Wan2_2-I2V-A14B-NSFW_fast_move-LOW_Q8lightning_edition)V2.gguf",
        "B5722E8E01A21F89A1C0B5E9DE9A3B907CE92324D1951D5CE70307956790DAAE",
    ),
    2668710: (
        "cf",
        "high",
        "Wan2_2-I2V-A14B-HIGH_consistent_face_Q8.gguf",
        None,
    ),
    2668712: (
        "cf",
        "low",
        "Wan2_2-I2V-A14B-LOW_consistent_face_Q8.gguf",
        None,
    ),
}

# The LightX2V LoRAs the CF pair needs. Sourced from Comfy-Org, not Civitai.
LIGHTX2V_REPO = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
LIGHTX2V_FILES = (
    "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
    "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
)
LIGHTX2V_REPO_DIR = "split_files/loras"


def load_env(path: Path = Path(".env")) -> None:
    """Load .env into the environment without overriding existing values."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def api_get(path: str, token: str | None) -> dict:
    headers = {"User-Agent": BROWSER_UA, "Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    last_error: Exception | None = None
    for attempt in range(RETRIES):
        try:
            response = requests.get(f"{CIVITAI_API}{path}", headers=headers, timeout=60)
            if response.status_code == 404:
                raise SystemExit(f"Civitai 404 for {path} - model or version removed")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as error:
            last_error = error
            time.sleep(2**attempt)
    raise SystemExit(f"Civitai request failed for {path}: {last_error}")


def resolve_download(version_id: int, token: str | None) -> tuple[str, str]:
    """Return (download_url, filename) for a Civitai model version."""
    data = api_get(f"/model-versions/{version_id}", token)
    files = data.get("files") or []
    gguf = [f for f in files if str(f.get("name", "")).endswith(".gguf")]
    chosen = None
    for candidate in gguf:
        if candidate.get("primary"):
            chosen = candidate
            break
    if chosen is None and gguf:
        chosen = gguf[0]
    if chosen is None:
        raise SystemExit(f"no GGUF file for version {version_id}")

    url = chosen.get("downloadUrl")
    name = chosen.get("name")
    if not url or not name:
        raise SystemExit(f"incomplete file metadata for version {version_id}")
    # Civitai requires the API token on the download URL itself.
    if token and "token=" not in url:
        separator = "&" if "?" in url else "?"
        url = f"{url}{separator}token={token}"
    return url, name


def looks_like_gguf(path: Path) -> bool:
    """A GGUF file starts with the magic bytes 'GGUF'."""
    try:
        with path.open("rb") as handle:
            return handle.read(4) == b"GGUF"
    except OSError:
        return False


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(CHUNK), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def download(url: str, dest: Path, token: str | None) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    headers = {"User-Agent": BROWSER_UA}
    if token:
        headers["Authorization"] = f"Bearer {token}"

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

    if not looks_like_gguf(part):
        print("  downloaded file is not a valid GGUF", file=sys.stderr)
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


def stage_one(
    version_id: int,
    build: str,
    stage: str,
    filename: str,
    expected_sha: str | None,
    dest_dir: Path,
    token: str | None,
    check_only: bool,
) -> bool:
    dest = dest_dir / filename
    label = f"{build}/{stage}"

    if dest.is_file() and looks_like_gguf(dest):
        if expected_sha:
            actual = sha256_of(dest)
            if actual != expected_sha:
                print(f"  [{label}] SHA-256 mismatch, re-downloading")
                print(f"    expected {expected_sha}")
                print(f"    actual   {actual}")
                dest.unlink()
            else:
                print(f"  [{label}] OK (sha256 verified)")
                return True
        else:
            print(f"  [{label}] OK ({dest.stat().st_size / 1024**3:.2f} GB)")
            return True

    if check_only:
        print(f"  [{label}] MISSING: {filename}")
        return False

    print(f"  [{label}] resolving version {version_id}")
    url, remote_name = resolve_download(version_id, token)
    if remote_name != filename:
        print(
            f"  [{label}] WARNING: Civitai reports {remote_name!r}, "
            f"workflow expects {filename!r}",
            file=sys.stderr,
        )

    if not download(url, dest, token):
        return False

    if expected_sha:
        actual = sha256_of(dest)
        if actual != expected_sha:
            print(f"  [{label}] SHA-256 mismatch after download", file=sys.stderr)
            print(f"    expected {expected_sha}", file=sys.stderr)
            print(f"    actual   {actual}", file=sys.stderr)
            return False
        print(f"  [{label}] sha256 verified")

    print(f"  [{label}] {dest.stat().st_size / 1024**3:.2f} GB")
    return True


def stage_lightx2v(dest_dir: Path, token: str | None, check_only: bool) -> bool:
    """Fetch the LightX2V 4-step LoRAs the CF pair needs.

    These live in the Comfy-Org repackaged repo, not Civitai. Without them the
    CF pair under-samples into flat gray at 5 steps.
    """
    lora_dir = dest_dir / "loras"
    lora_dir.mkdir(parents=True, exist_ok=True)
    all_ok = True
    for name in LIGHTX2V_FILES:
        dest = lora_dir / name
        if dest.is_file() and dest.stat().st_size > 1024:
            print(f"  [lightx2v] OK {name}")
            continue
        if check_only:
            print(f"  [lightx2v] MISSING: {name}")
            all_ok = False
            continue

        url = (
            f"https://huggingface.co/{LIGHTX2V_REPO}/resolve/main/"
            f"{LIGHTX2V_REPO_DIR}/{name}"
        )
        headers = {"User-Agent": BROWSER_UA}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        print(f"  [lightx2v] downloading {name}")
        part = dest.with_suffix(dest.suffix + ".part")
        try:
            with requests.get(url, headers=headers, stream=True, timeout=TIMEOUT) as response:
                response.raise_for_status()
                _write(response, part, "wb")
        except requests.RequestException as error:
            print(f"  [lightx2v] failed: {error}", file=sys.stderr)
            part.unlink(missing_ok=True)
            all_ok = False
            continue
        part.replace(dest)
        print(f"  [lightx2v] {dest.stat().st_size / 1024**2:.0f} MB")
    return all_ok


def main() -> int:
    ap = argparse.ArgumentParser(description="Download Wan 2.2 GGUF checkpoints from Civitai")
    ap.add_argument("--dest", default=None, help="Model directory (default: $MODEL_DIR or models)")
    ap.add_argument("--only", nargs="+", choices=["fm", "cf"], help="Restrict to one build")
    ap.add_argument("--check", action="store_true", help="Report status without downloading")
    args = ap.parse_args()

    load_env()
    token = os.environ.get("CIVITAI_API_KEY")
    if not token and not args.check:
        print(
            "WARNING: CIVITAI_API_KEY is not set; downloads may be rate limited",
            file=sys.stderr,
        )

    dest_dir = Path(args.dest or os.environ.get("MODEL_DIR", "models"))
    dest_dir.mkdir(parents=True, exist_ok=True)

    wanted = set(args.only) if args.only else {"fm", "cf"}
    print(f"Civitai model {MODEL_ID} -> {dest_dir}")
    print(f"Builds: {', '.join(sorted(wanted))}")

    results: dict[str, bool] = {}
    for version_id, (build, stage, filename, sha) in sorted(GGUF_VERSIONS.items()):
        if build not in wanted:
            continue
        results[f"{build}/{stage}"] = stage_one(
            version_id, build, stage, filename, sha, dest_dir, token, args.check
        )

    ok = sum(1 for v in results.values() if v)
    print(f"\n{ok}/{len(results)} checkpoint(s) ready")

    loras_ok = True
    if "cf" in wanted:
        print("\nCF pair needs the LightX2V LoRAs (render.py chains them automatically):")
        loras_ok = stage_lightx2v(dest_dir, token, args.check)

    return 0 if ok == len(results) and loras_ok else 1


if __name__ == "__main__":
    sys.exit(main())
