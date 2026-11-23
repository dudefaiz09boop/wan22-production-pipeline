#!/usr/bin/env python3
"""Generate ComfyUI workflow JSON for the selected model precision."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

Precision = Literal["fp16", "fp8", "gguf_q8"]


def get_loader_class(precision: Precision) -> str:
    """Get the UnetLoader class for the precision.

    ComfyUI 0.46 renamed the core loader to UNETLoader and made
    weight_dtype a required input.
    """
    if precision == "gguf_q8":
        return "UnetLoaderGGUF"
    return "UNETLoader"


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


def generate_workflow(
    precision: Precision,
    model_dir: str = "models",
    steps: int = 4,
    cfg: float = 1.0,
    shift: float = 5.0,
) -> dict:
    """Generate a complete ComfyUI workflow for the given precision.

    Mirrors the official ComfyUI Wan 2.2 14B I2V template: the base fp16
    experts are accelerated with the LightX2V 4-step LoRAs, sampled at
    CFG 1.0 with an SD3 shift of 5.0. The HIGH expert takes the first two
    steps and the LOW expert the remaining two.
    """
    loader_class = get_loader_class(precision)
    high_model, low_model = get_model_filenames(precision)

    # Official template split: HIGH 0->2, LOW 2->4 of a 4-step schedule.
    split = max(1, steps // 2)

    # Base workflow structure matching the existing base_i2v.json
    workflow = {
        "1": {
            "class_type": loader_class,
            "_meta": {"title": f"HIGH - {precision.upper()}"},
            "inputs": {"unet_name": high_model, "weight_dtype": "default"},
        },
        "2": {
            "class_type": loader_class,
            "_meta": {"title": f"LOW - {precision.upper()}"},
            "inputs": {"unet_name": low_model, "weight_dtype": "default"},
        },
        # LightX2V 4-step distillation LoRAs. Without these the base model
        # needs 20+ steps and still washes out at CFG 1.0.
        "15": {
            "class_type": "LoraLoaderModelOnly",
            "_meta": {"title": "HIGH - LightX2V 4step"},
            "inputs": {
                "model": ["1", 0],
                "lora_name": "lx_high.safetensors",
                "strength_model": 1.0,
            },
        },
        "16": {
            "class_type": "LoraLoaderModelOnly",
            "_meta": {"title": "LOW - LightX2V 4step"},
            "inputs": {
                "model": ["2", 0],
                "lora_name": "lx_low.safetensors",
                "strength_model": 1.0,
            },
        },
        # Wan 2.2 is a flow-matching model and needs the SD3 sampling shift.
        # The official template uses 5.0.
        "13": {
            "class_type": "ModelSamplingSD3",
            "_meta": {"title": f"HIGH - {precision.upper()}"},
            "inputs": {"model": ["15", 0], "shift": shift},
        },
        "14": {
            "class_type": "ModelSamplingSD3",
            "_meta": {"title": f"LOW - {precision.upper()}"},
            "inputs": {"model": ["16", 0], "shift": shift},
        },
        "3": {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan"},
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
            "class_type": "WanImageToVideo",
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
            "_meta": {"title": f"HIGH {split} steps"},
            "inputs": {
                "add_noise": "enable",
                "noise_seed": 0,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "simple",
                "start_at_step": 0,
                "end_at_step": split,
                "return_with_leftover_noise": "enable",
                "model": ["13", 0],
                "positive": ["8", 0],
                "negative": ["8", 1],
                "latent_image": ["8", 2],
            },
        },
        "10": {
            "class_type": "KSamplerAdvanced",
            "_meta": {"title": f"LOW {steps - split} steps"},
            "inputs": {
                "add_noise": "disable",
                "noise_seed": 0,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": "euler",
                "scheduler": "simple",
                "start_at_step": split,
                "end_at_step": steps,
                "return_with_leftover_noise": "disable",
                "model": ["14", 0],
                "positive": ["8", 0],
                "negative": ["8", 1],
                "latent_image": ["9", 0],
            },
        },
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["4", 0]}},
        "12": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "frame_rate": 8,
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
    ap.add_argument("--steps", type=int, default=4, help="Total sampling steps")
    ap.add_argument("--cfg", type=float, default=1.0, help="CFG scale")
    ap.add_argument("--shift", type=float, default=5.0, help="SD3 flow shift")
    args = ap.parse_args()

    workflow = generate_workflow(
        args.precision, args.model_dir, args.steps, args.cfg, args.shift
    )
    save_workflow(workflow, Path(args.output))

    return 0


if __name__ == "__main__":
    sys.exit(main())