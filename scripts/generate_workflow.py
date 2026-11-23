#!/usr/bin/env python3
"""Generate ComfyUI workflow JSON for the selected model precision."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

Precision = Literal["fp16", "fp8", "gguf_q8"]


def get_loader_class(precision: Precision) -> str:
    """Get the UnetLoader class for the precision."""
    if precision == "gguf_q8":
        return "UnetLoaderGGUF"
    return "UnetLoader"


def get_model_filenames(precision: Precision) -> tuple[str, str]:
    """Get high/low noise model filenames for the precision."""
    if precision == "fp16":
        return (
            "wan2.2_i2v_high_noise_14B_fp16.safetensors",
            "wan2.2_i2v_low_noise_14B_fp16.safetensors",
        )
    elif precision == "fp8":
        return (
            "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
            "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
        )
    else:  # gguf_q8
        return (
            "wan2.2_i2v_high_noise_14B_q8.gguf",
            "wan2.2_i2v_low_noise_14B_q8.gguf",
        )


def generate_workflow(precision: Precision, model_dir: str = "models") -> dict:
    """Generate a complete ComfyUI workflow for the given precision."""
    loader_class = get_loader_class(precision)
    high_model, low_model = get_model_filenames(precision)

    # Base workflow structure matching the existing base_i2v.json
    workflow = {
        "1": {
            "class_type": loader_class,
            "_meta": {"title": f"HIGH - {precision.upper()}"},
            "inputs": {"unet_name": high_model},
        },
        "2": {
            "class_type": loader_class,
            "_meta": {"title": f"LOW - {precision.upper()}"},
            "inputs": {"unet_name": low_model},
        },
        "3": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn.safetensors", "type": "wan"},
        },
        "4": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": "wan_2.1_vae.safetensors"},
        },
        "5": {
            "class_type": "LoadImage",
            "inputs": {"image": "startframes/c01.png", "upload": "image"},
        },
        "6": {
            "class_type": "CLIPTextEncode",
            "inputs": {"__todo__": "positive", "text": "", "clip": ["3", 0]},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"__todo__": "negative", "text": "", "clip": ["3", 0]},
        },
        "8": {
            "class_type": "Wan22ImageToVideo",
            "inputs": {
                "width": 832,
                "height": 480,
                "length": 80,
                "batch_size": 1,
                "positive": ["6", 0],
                "negative": ["7", 0],
                "vae": ["4", 0],
                "start_image": ["5", 0],
            },
        },
        "9": {
            "class_type": "KSamplerAdvanced",
            "_meta": {"title": f"HIGH {2 if precision != 'gguf_q8' else 2} steps"},
            "inputs": {
                "add_noise": "enable",
                "noise_seed": 0,
                "steps": 2,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "start_at_step": 0,
                "end_at_step": 2,
                "return_with_leftover_noise": "enable",
                "model": ["1", 0],
                "positive": ["8", 0],
                "negative": ["8", 1],
                "latent_image": ["8", 2],
            },
        },
        "10": {
            "class_type": "KSamplerAdvanced",
            "_meta": {"title": f"LOW {3 if precision != 'gguf_q8' else 3} steps"},
            "inputs": {
                "add_noise": "disable",
                "noise_seed": 0,
                "steps": 5,
                "cfg": 1.0,
                "sampler_name": "euler",
                "scheduler": "simple",
                "start_at_step": 2,
                "end_at_step": 5,
                "return_with_leftover_noise": "disable",
                "model": ["2", 0],
                "positive": ["8", 0],
                "negative": ["8", 1],
                "latent_image": ["9", 0],
            },
        },
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["4", 0]}},
        "12": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "frame_rate": 16,
                "loop_count": 0,
                "filename_prefix": "wan22/",
                "format": "video/h264-mp4",
                "pingpong": False,
                "save_output": True,
                "images": ["11", 0],
            },
        },
    }

    return workflow


def save_workflow(workflow: dict, output_path: Path) -> None:
    """Save workflow to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(workflow, f, indent=2)
    print(f"Workflow saved to {output_path}")


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Generate ComfyUI workflow for WAN 2.2")
    ap.add_argument(
        "--precision",
        choices=["fp16", "fp8", "gguf_q8"],
        required=True,
        help="Model precision",
    )
    ap.add_argument("--output", required=True, help="Output JSON file path")
    ap.add_argument("--model-dir", default="models", help="Model directory (for reference)")
    args = ap.parse_args()

    workflow = generate_workflow(args.precision, args.model_dir)
    save_workflow(workflow, Path(args.output))

    return 0


if __name__ == "__main__":
    sys.exit(main())