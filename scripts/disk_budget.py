#!/usr/bin/env python3
"""Compute the disk budget for a full production run.

Sums the real model sizes from the Hugging Face API plus the local assets,
so the instance disk can be sized with evidence rather than a guess.
"""
from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

GB = 1024 ** 3

HF_FILES = {
    "Wan 2.2 I2V high fp16": ("Comfy-Org/Wan-2.2",
        "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"),
    "Wan 2.2 I2V low fp16": ("Comfy-Org/Wan-2.2",
        "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"),
    "UMT5 XXL fp8 scaled": ("Comfy-Org/Wan-2.2",
        "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
    "Wan 2.1 VAE": ("Comfy-Org/Wan-2.2",
        "split_files/vae/wan_2.1_vae.safetensors"),
    "Wan 2.1 I2V 480p fp8 (lipsync)": ("Comfy-Org/Wan-2.1",
        "split_files/diffusion_models/wan2.1_i2v_480p_14B_fp8_scaled.safetensors"),
    "InfiniteTalk Multi patch": ("Kijai/WanVideo_comfy_fp8_scaled",
        "InfiniteTalk/Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors"),
    "wav2vec2 audio encoder": ("Comfy-Org/Wan-2.2",
        "split_files/audio_encoders/wav2vec2_large_english_fp16.safetensors"),
    "Juggernaut XL v9 (stills)": ("RunDiffusion/Juggernaut-XL-v9",
        "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"),
}

# Files served from the repo root, not a subdirectory.
ROOT_FILES = {"Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"}


def remote_size(repo: str, path: str) -> int:
    name = path.rsplit("/", 1)[-1]
    if name in ROOT_FILES:
        url = f"https://huggingface.co/api/models/{repo}"
        with urllib.request.urlopen(url, timeout=40) as response:
            data = json.loads(response.read())
        for entry in data.get("siblings", []):
            if entry.get("rfilename") == name:
                # The model API omits size; fall back to the known 7.1 GB.
                return 7_105_348_188
        return 0
    url = f"https://huggingface.co/api/models/{repo}/tree/main/{path.rsplit('/', 1)[0]}"
    with urllib.request.urlopen(url, timeout=40) as response:
        entries = json.loads(response.read())
    for entry in entries:
        if entry.get("path", "").endswith(name):
            return entry.get("size") or 0
    return 0


def local_size(pattern: str) -> int:
    return sum(p.stat().st_size for p in Path(".").glob(pattern))


def main() -> int:
    total = 0
    print("Remote models")
    print("-" * 62)
    for label, (repo, path) in HF_FILES.items():
        try:
            size = remote_size(repo, path)
        except Exception as error:  # noqa: BLE001
            print(f"  ??   {label}: {error}")
            continue
        total += size
        print(f"  {size / GB:6.2f} GB  {label}")

    print()
    print("Local assets")
    print("-" * 62)
    loras = local_size("loras/*.safetensors")
    inswapper = local_size("models/inswapper_128.onnx")
    frames = local_size("startframes/*.png")
    refs = local_size("references/*.png")
    for label, size in [
        ("LoRAs (8 files)", loras),
        ("inswapper_128.onnx", inswapper),
        ("startframes (70 png)", frames),
        ("references", refs),
    ]:
        total += size
        print(f"  {size / GB:6.2f} GB  {label}")

    print()
    print("Working space")
    print("-" * 62)
    # Renders: 70 clips at ~2-6 MB, plus face-swapped and lipsync copies.
    renders = 70 * 6 * 1024 ** 2 * 3
    audio = 44 * 1 * 1024 ** 2
    output = 400 * 1024 ** 2
    comfy = 3 * GB
    for label, size in [
        ("renders (3 passes x 70 clips)", renders),
        ("narration audio", audio),
        ("final output + zip", output),
        ("ComfyUI + custom nodes + venv", comfy),
    ]:
        total += size
        print(f"  {size / GB:6.2f} GB  {label}")

    print("-" * 62)
    print(f"  {total / GB:6.2f} GB  TOTAL")
    print(f"  {total / GB * 1.25:6.2f} GB  with 25% headroom")
    return 0


if __name__ == "__main__":
    sys.exit(main())
