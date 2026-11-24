#!/usr/bin/env python3
"""Generate ComfyUI workflow JSON for the selected model precision."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Literal

Precision = Literal["fp16", "fp8", "gguf_q8"]

# The GGUF tier ships Lightning-baked weights: the step distillation is already
# merged into the checkpoint. Stacking LightX2V LoRAs on top degrades quality,
# so this tier must never receive them.
LIGHTNING_BAKED: frozenset[str] = frozenset({"gguf_q8"})

# Civitai model 2053259, versions 2540892 / 2540896. SHA-256 pinned in
# run_production.py. The LOW filename really is missing its opening paren --
# that is the author's typo and the hash lookup is keyed on the literal string.
GGUF_Q8_HIGH = "Wan2_2-I2V-A14B-NSFW_fast_move-HIGH_Q8(lightning_edition)V2.gguf"
GGUF_Q8_LOW = "Wan2_2-I2V-A14B-NSFW_fast_move-LOW_Q8lightning_edition)V2.gguf"

# Comfy-Org/Wan_2.2_ComfyUI_Repackaged -> split_files/loras/
LIGHTX2V_HIGH = "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors"
LIGHTX2V_LOW = "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors"

TEXT_ENCODER = "umt5_xxl_fp8_e4m3fn_scaled.safetensors"
VAE = "wan_2.1_vae.safetensors"

# Project contract: 16 fps, the rate Wan 2.2 is trained at. 8 fps read as
# stop-motion and also mismatched the lipsync pass, which has always rendered
# at 16. WanImageToVideo requires (length - 1) % 4 == 0, so frames are
# duration * fps + 1 and the fps is chosen to keep that true.
FPS = 16
FRAMES_5S = 81
FRAMES_4S = 65


def frames_for(duration: float, fps: int = FPS) -> int:
    """Frame count for a clip, snapped to the (length - 1) % 4 == 0 contract."""
    length = int(round(duration * fps)) + 1
    while (length - 1) % 4:
        length += 1
    return length

# Author-recommended schedule for the Lightning-baked pair: HIGH 0->2,
# LOW 2->5. "Jellai recommends 2+3 for even better results, and I agree."
DEFAULT_STEPS = 5


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
    if precision == "fp8":
        return (
            "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
            "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
        )
    return (GGUF_Q8_HIGH, GGUF_Q8_LOW)


def generate_workflow(
    precision: Precision,
    model_dir: str = "models",
    steps: int = DEFAULT_STEPS,
    cfg: float = 1.0,
    shift: float = 5.0,
    nag: bool = False,
    nag_scale: float = 11.0,
    nag_alpha: float = 0.25,
    nag_tau: float = 2.5,
) -> dict:
    """Generate a complete ComfyUI workflow for the given precision.

    Two recipes, selected by whether the checkpoint already carries the
    Lightning distillation:

    * ``gguf_q8`` -- Lightning-baked. No LoRAs, 2+3 steps, CFG 1.0.
    * ``fp16`` / ``fp8`` -- base weights. LightX2V 4-step LoRAs are required;
      without them the model needs 20+ steps and still washes out at CFG 1.0.

    ``nag`` inserts a KJNodes WanVideoNAG node so the negative prompt is
    actually evaluated. At CFG 1.0 the negative branch is otherwise dead
    code. NAG roughly doubles generation time, so it is opt-in.
    """
    loader_class = get_loader_class(precision)
    high_model, low_model = get_model_filenames(precision)
    baked = precision in LIGHTNING_BAKED

    # HIGH takes the first `split` steps, LOW the remainder.
    split = max(1, steps // 2)

    workflow: dict = {
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
            "inputs": {"clip_name": TEXT_ENCODER, "type": "wan"},
        },
        "4": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": VAE},
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
                "length": FRAMES_5S,
                "batch_size": 1,
                "positive": ["6", 0],
                "negative": ["7", 0],
                "vae": ["4", 0],
                "start_image": ["5", 0],
            },
        },
        "11": {"class_type": "VAEDecode", "inputs": {"samples": ["10", 0], "vae": ["4", 0]}},
        "12": {
            "class_type": "VHS_VideoCombine",
            "inputs": {
                "frame_rate": FPS,
                "loop_count": 0,
                "filename_prefix": "wan22/",
                "format": "video/h264-mp4",
                "pingpong": False,
                "save_output": True,
                "images": ["11", 0],
            },
        },
    }

    # UNETLoader requires weight_dtype; UnetLoaderGGUF does not accept it.
    if loader_class == "UNETLoader":
        for node_id in ("1", "2"):
            workflow[node_id]["inputs"]["weight_dtype"] = "default"

    # Model chain: loader -> [LightX2V LoRA] -> ModelSamplingSD3 -> [NAG] -> sampler
    high_ref: list = ["1", 0]
    low_ref: list = ["2", 0]

    if not baked:
        workflow["15"] = {
            "class_type": "LoraLoaderModelOnly",
            "_meta": {"title": "HIGH - LightX2V 4step"},
            "inputs": {
                "model": high_ref,
                "lora_name": LIGHTX2V_HIGH,
                "strength_model": 1.0,
            },
        }
        workflow["16"] = {
            "class_type": "LoraLoaderModelOnly",
            "_meta": {"title": "LOW - LightX2V 4step"},
            "inputs": {
                "model": low_ref,
                "lora_name": LIGHTX2V_LOW,
                "strength_model": 1.0,
            },
        }
        high_ref = ["15", 0]
        low_ref = ["16", 0]

    # Wan 2.2 is a flow-matching model and needs the SD3 sampling shift.
    workflow["13"] = {
        "class_type": "ModelSamplingSD3",
        "_meta": {"title": f"HIGH - {precision.upper()}"},
        "inputs": {"model": high_ref, "shift": shift},
    }
    workflow["14"] = {
        "class_type": "ModelSamplingSD3",
        "_meta": {"title": f"LOW - {precision.upper()}"},
        "inputs": {"model": low_ref, "shift": shift},
    }
    high_ref = ["13", 0]
    low_ref = ["14", 0]

    if nag:
        # WanVideoNAG patches cross-attention with the negative conditioning,
        # which is the only way to make negatives bite at CFG 1.0.
        for node_id, ref, label in (("17", high_ref, "HIGH"), ("18", low_ref, "LOW")):
            workflow[node_id] = {
                "class_type": "WanVideoNAG",
                "_meta": {"title": f"{label} - NAG"},
                "inputs": {
                    "model": ref,
                    "conditioning": ["7", 0],
                    "nag_scale": nag_scale,
                    "nag_alpha": nag_alpha,
                    "nag_tau": nag_tau,
                },
            }
        high_ref = ["17", 0]
        low_ref = ["18", 0]

    workflow["9"] = {
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
            "model": high_ref,
            "positive": ["8", 0],
            "negative": ["8", 1],
            "latent_image": ["8", 2],
        },
    }
    workflow["10"] = {
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
            "model": low_ref,
            "positive": ["8", 0],
            "negative": ["8", 1],
            "latent_image": ["9", 0],
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
    ap.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_STEPS,
        help=f"Total sampling steps (default {DEFAULT_STEPS} = 2+3)",
    )
    ap.add_argument("--cfg", type=float, default=1.0, help="CFG scale")
    ap.add_argument("--shift", type=float, default=5.0, help="SD3 flow shift")
    ap.add_argument(
        "--nag",
        action="store_true",
        help="Insert WanVideoNAG so the negative prompt is evaluated (doubles render time)",
    )
    ap.add_argument("--nag-scale", type=float, default=11.0, help="NAG guidance strength")
    ap.add_argument("--nag-alpha", type=float, default=0.25, help="NAG mixing coefficient")
    ap.add_argument("--nag-tau", type=float, default=2.5, help="NAG clipping threshold")
    args = ap.parse_args()

    workflow = generate_workflow(
        args.precision,
        args.model_dir,
        args.steps,
        args.cfg,
        args.shift,
        nag=args.nag,
        nag_scale=args.nag_scale,
        nag_alpha=args.nag_alpha,
        nag_tau=args.nag_tau,
    )
    save_workflow(workflow, Path(args.output))

    return 0


if __name__ == "__main__":
    sys.exit(main())