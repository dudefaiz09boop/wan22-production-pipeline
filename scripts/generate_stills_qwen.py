#!/usr/bin/env python3
"""Generate startframes with Qwen-Image 2.1, using photo references.

Why this exists
---------------
The SDXL path could not hold identity. SDXL encodes text with CLIP at 77
tokens, so the prompt had to be trimmed to fragments, and identity came from a
LoRA plus an InsightFace paste -- which swapped the characters when the wrong
face was picked.

Qwen-Image 2.1 does not work that way. It encodes with an 8B Qwen2.5-VL vision
model and accepts reference images directly:

    TextEncodeQwenImage21(clip, images.image_1, images.image_2, vae)

So the prompt says "use <image1> for her body, <image2> for her exact face" and
the model reads the face from the photo. No LoRA, no swap, no token budget.

This is the approach the attached ComfyUI template uses, and it is the only one
that has produced a correct face in this project.

Usage
-----
    python scripts/generate_stills_qwen.py --beats c01 c02
    python scripts/generate_stills_qwen.py            # all 70

Requires ComfyUI running with:
    models/diffusion_models/qwen_image-Q4_K_M.gguf
    models/text_encoders/Qwen2.5-VL-7B-Instruct-Q4_NL.gguf
    models/vae/qwen_image_vae.safetensors
    custom_nodes/ComfyUI-GGUF
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from comfy_api import DEFAULT_SERVER, upload  # noqa: E402
from render import post_prompt, wait_for, download_output  # noqa: E402

SPEC = Path("docs/beats.yaml")
PROMPTS = Path("prompts")
OUTPUT = Path("startframes")

# Reference plates. One body shot and one face shot per character, which is
# what the template uses: <image1> for body and posture, <image2> for the face.
REFS = {
    "faiz": {
        "body": "refs/Faiz/01_front_standing_straight_00001.png",
        "face": "refs/Faiz/02_front_relaxed_00001.png",
    },
    "juv": {
        "body": "refs/Juvi/01_front_standing_00001.png",
        "face": "refs/Juvi/02_front_relaxed_00001.png",
    },
}

# Qwen-Image 2.1 at the template's portrait resolution.
WIDTH = 896
HEIGHT = 1344
STEPS = 25
CFG = 1.0
SAMPLER = "euler"
SCHEDULER = "simple"

NEGATIVE = (
    "cropped body, cropped feet, selfie, close-up portrait, multiple people, "
    "extra limbs, deformed hands, blurry, low quality, watermark, text, "
    "cartoon, anime, 3d render, cgi, plastic skin"
)


def build_prompt(beat_id: str, cam_lines: list[str], cast: list[str]) -> str:
    """Compose a Qwen-Image prompt from the beat's cam lines and its cast.

    Qwen takes a long natural-language prompt, so the cam lines go in whole --
    no trimming, no fragments. The reference instructions come first because
    that is what binds identity.
    """
    parts: list[str] = []

    if len(cast) == 2:
        parts.append(
            "Create a photorealistic photograph of two people, a brother and "
            "sister, both adults, in a modest Indian Muslim flat. "
            "Use <image1> as the reference for the man's body, build and "
            "posture, and <image2> for his exact face, beard, skin tone and "
            "hair. Use <image3> as the reference for the woman's body, build "
            "and posture, and <image4> for her exact face, skin tone and long "
            "straight black hair. Keep both faces identical to the references."
        )
    else:
        char = cast[0]
        parts.append(
            f"Create a photorealistic photograph of one person, an adult "
            f"Indian Muslim {'man' if char == 'faiz' else 'woman'}, in a modest "
            f"Indian Muslim flat. Use <image1> as the reference for the body, "
            f"build and posture, and <image2> for the exact face, skin tone and "
            f"hair. Keep the face identical to the reference."
        )

    parts.append("The shot: " + " ".join(cam_lines))
    parts.append(
        "Photorealistic DSLR photograph, 35mm lens, natural skin texture, "
        "sharp focus, film grain, available light, both faces clearly visible "
        "and in frame."
    )
    return " ".join(parts)


def build_workflow(
    prompt: str,
    negative: str,
    ref_images: list[str],
    width: int,
    height: int,
    seed: int,
) -> dict:
    """Qwen-Image 2.1 text-to-image with reference images.

    Mirrors the node graph in the attached template: UNETLoader(GGUF) ->
    QwenImage21Cache -> KSampler, with TextEncodeQwenImage21 taking the
    reference images and the VAE.
    """
    wf: dict = {}

    wf["1"] = {
        "class_type": "UnetLoaderGGUF",
        "inputs": {"unet_name": "qwen_image-Q4_K_M.gguf"},
    }
    wf["2"] = {
        "class_type": "CLIPLoaderGGUF",
        "inputs": {
            "clip_name": "Qwen2.5-VL-7B-Instruct-Q4_NL.gguf",
            "type": "qwen_image",
        },
    }
    wf["3"] = {
        "class_type": "VAELoader",
        "inputs": {"vae_name": "qwen_image_vae.safetensors"},
    }
    wf["4"] = {
        "class_type": "QwenImage21Cache",
        "inputs": {"model": ["1", 0]},
    }

    # Reference images, one LoadImage per plate.
    image_links = []
    for index, name in enumerate(ref_images, start=10):
        wf[str(index)] = {
            "class_type": "LoadImage",
            "inputs": {"image": name},
        }
        image_links.append([str(index), 0])

    encode_inputs: dict = {
        "clip": ["2", 0],
        "vae": ["3", 0],
        "prompt": prompt,
        "negative_prompt": negative,
    }
    for slot, link in enumerate(image_links, start=1):
        encode_inputs[f"images.image_{slot}"] = link

    wf["20"] = {"class_type": "TextEncodeQwenImage21", "inputs": encode_inputs}

    wf["21"] = {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": width, "height": height, "batch_size": 1},
    }
    wf["22"] = {
        "class_type": "KSampler",
        "inputs": {
            "model": ["4", 0],
            "positive": ["20", 0],
            "negative": ["20", 1],
            "latent_image": ["21", 0],
            "seed": seed,
            "steps": STEPS,
            "cfg": CFG,
            "sampler_name": SAMPLER,
            "scheduler": SCHEDULER,
            "denoise": 1.0,
        },
    }
    wf["23"] = {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["22", 0], "vae": ["3", 0]},
    }
    wf["24"] = {
        "class_type": "SaveImage",
        "inputs": {"images": ["23", 0], "filename_prefix": "startframe"},
    }
    return wf


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate startframes with Qwen-Image 2.1")
    ap.add_argument("--spec", default=str(SPEC))
    ap.add_argument("--output", default=str(OUTPUT))
    ap.add_argument("--server", default=DEFAULT_SERVER)
    ap.add_argument("--beats", nargs="+")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    spec = yaml.safe_load(Path(args.spec).read_text(encoding="utf-8"))
    shots = spec.get("shots") or []
    if args.beats:
        wanted = set(args.beats)
        shots = [s for s in shots if s.get("id") in wanted]

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    missing = [
        p for c in REFS.values() for p in c.values() if not Path(p).is_file()
    ]
    if missing:
        print(f"missing reference plates: {missing}", file=sys.stderr)
        return 1

    upload_cache: dict = {}
    made = 0
    for shot in shots:
        beat_id = shot.get("id")
        cam = [line for line in (shot.get("cam") or []) if isinstance(line, str)]
        cast = ["faiz", "juv"]

        ref_images = [
            upload(args.server, Path(REFS["faiz"]["body"]), upload_cache),
            upload(args.server, Path(REFS["faiz"]["face"]), upload_cache),
            upload(args.server, Path(REFS["juv"]["body"]), upload_cache),
            upload(args.server, Path(REFS["juv"]["face"]), upload_cache),
        ]
        prompt = build_prompt(beat_id, cam, cast)
        wf = build_workflow(prompt, NEGATIVE, ref_images, WIDTH, HEIGHT, args.seed)

        if args.dry_run:
            print(f"--- {beat_id} ---")
            print(prompt[:300])
            continue

        try:
            prompt_id = post_prompt(wf, args.server)
            history = wait_for(args.server, prompt_id)
            entry = (history.get("outputs") or {}).get("24") or {}
            images = entry.get("images") or []
            if not images:
                print(f"  {beat_id}: no image returned", file=sys.stderr)
                continue
            dest = out / f"{beat_id}.png"
            download_output(args.server, images[0], dest)
            print(f"  {beat_id}: saved {dest}")
            made += 1
        except (urllib.error.URLError, TimeoutError) as error:
            print(f"  {beat_id}: {error}", file=sys.stderr)

    print(f"generated {made} startframe(s)")
    return 0 if made else 1


if __name__ == "__main__":
    sys.exit(main())
