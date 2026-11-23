#!/usr/bin/env python3
"""Auto-detect GPU VRAM and select the best available WAN 2.2 model precision."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

Precision = Literal["fp16", "fp8", "gguf_q8"]


@dataclass
class ModelConfig:
    precision: Precision
    high_noise_model: str
    low_noise_model: str
    loader_class: str
    description: str
    vram_required_gb: int


# Model configurations for each precision tier
MODEL_CONFIGS: dict[Precision, ModelConfig] = {
    "fp16": ModelConfig(
        precision="fp16",
        high_noise_model="wan2.2_i2v_high_noise_14B_fp16.safetensors",
        low_noise_model="wan2.2_i2v_low_noise_14B_fp16.safetensors",
        loader_class="UnetLoader",
        description="FP16 (full precision) - best quality, requires 48GB+ VRAM",
        vram_required_gb=48,
    ),
    "fp8": ModelConfig(
        precision="fp8",
        high_noise_model="wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
        low_noise_model="wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
        loader_class="UnetLoader",
        description="FP8 scaled - good quality, requires 24-47GB VRAM",
        vram_required_gb=24,
    ),
    "gguf_q8": ModelConfig(
        precision="gguf_q8",
        high_noise_model="wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
        low_noise_model="wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
        loader_class="UnetLoader",
        description="FP8 scaled fallback for <24GB VRAM (no GGUF in the Comfy-Org repo)",
        vram_required_gb=16,
    ),
}

# Hugging Face repo IDs for each precision.
# The old `Comfy-Org/Wan_2.2_ComfyUI_Repackaged` id redirects to
# `Comfy-Org/Wan-2.2`; use the canonical id directly.
HF_REPOS = {
    "fp16": "Comfy-Org/Wan-2.2",
    "fp8": "Comfy-Org/Wan-2.2",
    "gguf_q8": "Comfy-Org/Wan-2.2",
}

# File patterns in the HF repo.
# Verified against https://huggingface.co/api/models/Comfy-Org/Wan-2.2:
# the text encoder is `umt5_xxl_fp8_e4m3fn_scaled.safetensors` (with the
# `_scaled` suffix) and the repo ships no GGUF files at all.
HF_FILES = {
    "fp16": [
        "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors",
        "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors",
    ],
    "fp8": [
        "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
        "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
    ],
    # GGUF is not published in the Comfy-Org repo. The GGUF tier uses the
    # fp8 pair, which fits the same <24GB budget and needs no extra loader.
    "gguf_q8": [
        "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
        "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
    ],
}

# Shared files needed for all precisions
SHARED_FILES = [
    "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "split_files/vae/wan_2.1_vae.safetensors",
]


def get_gpu_vram_gb() -> int | None:
    """Detect GPU VRAM in GB using nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        vram_mb = [int(line.strip()) for line in result.stdout.strip().split("\n") if line.strip()]
        if vram_mb:
            # Return the maximum VRAM across all GPUs
            return max(vram_mb) // 1024
    except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
        pass
    return None


def select_precision(vram_gb: int | None, force: Precision | None = None) -> Precision:
    """Select the best precision based on available VRAM.

    Thresholds are set from measured model sizes, not round numbers:

    FP16 needs one 26.62 GB UNET resident at a time plus the 6.27 GB text
    encoder, the VAE and activations (~37 GB peak). ComfyUI swaps the HIGH
    and LOW UNETs, so both never need to be resident together. 44 GB is the
    real floor, which is what an RTX A6000 reports (46068 MiB -> 44 GB).

    FP8 needs 14.3 GB per UNET, so 22 GB is the floor.
    """
    if force:
        return force

    if vram_gb is None:
        print("WARNING: Could not detect GPU VRAM, defaulting to FP8", file=sys.stderr)
        return "fp8"

    if vram_gb >= 44:
        return "fp16"
    elif vram_gb >= 22:
        return "fp8"
    else:
        return "gguf_q8"


def get_model_config(precision: Precision) -> ModelConfig:
    """Get the model configuration for a given precision."""
    return MODEL_CONFIGS[precision]


def get_hf_repo(precision: Precision) -> str:
    """Get the Hugging Face repo for a given precision."""
    return HF_REPOS[precision]


def get_hf_files(precision: Precision) -> list[str]:
    """Get the list of HF files to download for a given precision."""
    return HF_FILES[precision] + SHARED_FILES


def get_local_model_dir(precision: Precision) -> Path:
    """Get the local directory where models should be stored."""
    base = Path(os.environ.get("MODEL_DIR", "models"))
    return base / f"wan22_{precision}"


def print_selection(vram_gb: int | None, precision: Precision, config: ModelConfig) -> None:
    """Print the model selection summary."""
    print("=" * 60)
    print("WAN 2.2 Model Selection")
    print("=" * 60)
    if vram_gb:
        print(f"Detected GPU VRAM: {vram_gb} GB")
    else:
        print("Detected GPU VRAM: Unknown (no nvidia-smi)")
    print(f"Selected Precision: {precision.upper()}")
    print(f"Description: {config.description}")
    print(f"High-noise model: {config.high_noise_model}")
    print(f"Low-noise model:  {config.low_noise_model}")
    print(f"Loader class:     {config.loader_class}")
    print(f"Local model dir:  {get_local_model_dir(precision)}")
    print(f"HF repo:          {get_hf_repo(precision)}")
    print("=" * 60)


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Auto-select WAN 2.2 model based on GPU VRAM")
    ap.add_argument("--force", choices=["fp16", "fp8", "gguf_q8"], help="Force a specific precision")
    ap.add_argument("--output", choices=["json", "env", "summary"], default="summary", help="Output format")
    ap.add_argument("--model-dir", help="Override model directory")
    args = ap.parse_args()

    if args.model_dir:
        os.environ["MODEL_DIR"] = args.model_dir

    vram_gb = get_gpu_vram_gb()
    precision = select_precision(vram_gb, args.force)
    config = get_model_config(precision)

    if args.output == "json":
        result = {
            "precision": precision,
            "vram_gb": vram_gb,
            "high_noise_model": config.high_noise_model,
            "low_noise_model": config.low_noise_model,
            "loader_class": config.loader_class,
            "model_dir": str(get_local_model_dir(precision)),
            "hf_repo": get_hf_repo(precision),
            "hf_files": get_hf_files(precision),
            "description": config.description,
        }
        print(json.dumps(result, indent=2))
    elif args.output == "env":
        print(f"WAN_PRECISION={precision}")
        print(f"WAN_HIGH_NOISE_MODEL={config.high_noise_model}")
        print(f"WAN_LOW_NOISE_MODEL={config.low_noise_model}")
        print(f"WAN_LOADER_CLASS={config.loader_class}")
        print(f"WAN_MODEL_DIR={get_local_model_dir(precision)}")
        print(f"WAN_HF_REPO={get_hf_repo(precision)}")
        if vram_gb:
            print(f"WAN_VRAM_GB={vram_gb}")
    else:
        print_selection(vram_gb, precision, config)

    return 0


if __name__ == "__main__":
    sys.exit(main())