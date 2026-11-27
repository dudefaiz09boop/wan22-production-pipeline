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
    models/text_encoders/Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf
    models/vae/qwen_image_vae.safetensors
    custom_nodes/ComfyUI-GGUF
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from comfy_api import DEFAULT_SERVER, upload  # noqa: E402
from render import post, wait_for, download_output  # noqa: E402

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
    """Compose a Qwen-Image prompt in the template's style.

    The template writes a natural-language instruction, not a tag list: it names
    the reference images, states what each one is for, then describes the shot.
    Qwen-Image is an instruction-following model, so that structure is what it
    was trained to read.

    The cam lines go in whole. There is no token budget to fight -- the encoder
    is an 8B vision-language model.
    """
    shot = " ".join(cam_lines)

    if len(cast) == 2:
        return (
            "Create a new photograph of the same two people, a brother and "
            "sister, both adults, in a modest Indian Muslim flat. "
            "Use <image1> as the primary reference for the man's body "
            "proportions, build, posture and overall identity, and <image2> for "
            "his exact face, short boxed beard, connected moustache, black hair, "
            "warm brown skin and dark brown eyes. "
            "Use <image3> as the primary reference for the woman's body "
            "proportions, build, posture and overall identity, and <image4> for "
            "her exact face, very long straight black hair, dark kajal, full "
            "lips, warm light-brown skin and dark brown eyes. "
            "Keep both faces identical to the references, at the same age, with "
            "the same skin tone and texture. "
            f"The shot: {shot}. "
            "Both faces clearly visible and in frame, natural human anatomy, "
            "photorealistic DSLR photograph, 35mm lens, sharp focus, high detail "
            "skin texture, natural available light, film grain, ultra realistic."
        )

    char = cast[0]
    who = "man" if char == "faiz" else "woman"
    return (
        f"Create a new photograph of the same person, an adult Indian Muslim "
        f"{who}, in a modest Indian Muslim flat. "
        f"Use <image1> as the primary reference for the body proportions, "
        f"build, posture and overall identity, and <image2> for the exact face, "
        f"skin tone and texture, and hairstyle. "
        f"Keep the face identical to the reference, at the same age. "
        f"The shot: {shot}. "
        "Face clearly visible and in frame, natural human anatomy, "
        "photorealistic DSLR photograph, 35mm lens, sharp focus, high detail "
        "skin texture, natural available light, film grain, ultra realistic."
    )


def build_workflow(
    prompt: str,
    negative: str,
    ref_images: list[str],
    width: int,
    height: int,
    seed: int,
) -> dict:
    """Qwen-Image 2.1 text-to-image with reference images.

    Node graph and every input name taken from the working template
    (Image generation.json), not guessed:

        UnetLoaderGGUF        unet_name
        CLIPLoaderGGUF        clip_name, type
        VAELoader             vae_name
        QwenImage21Cache      model, device, dtype   (widgets: auto, default)
        LoadImage             image
        TextEncodeQwenImage21 clip, vae, prompt, negative_prompt, resolution,
                              images.image_1..N
        EmptyLatentImage      width, height, batch_size
        KSampler              model, positive, negative, latent_image,
                              seed, steps, cfg, sampler_name, scheduler, denoise
        VAEDecode             samples, vae
        SaveImage             images, filename_prefix
    """
    wf: dict = {}

    wf["1"] = {
        "class_type": "UnetLoaderGGUF",
        "inputs": {"unet_name": "qwen-image-Q4_K_M.gguf"},
    }
    wf["2"] = {
        "class_type": "CLIPLoaderGGUF",
        "inputs": {
            "clip_name": "Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf",
            "type": "qwen_image",
        },
    }
    wf["3"] = {
        "class_type": "VAELoader",
        "inputs": {"vae_name": "qwen_image_vae.safetensors"},
    }
    wf["4"] = {
        "class_type": "QwenImage21Cache",
        "inputs": {
            "model": ["1", 0],
            "device": "auto",
            "dtype": "default",
        },
    }

    # Reference images. The template links images.image_N to LoadImage slot 0.
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
        "resolution": 1024,
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


def _fetch_image(server: str, item: dict, dest: Path) -> Path:
    """Download one SaveImage output from ComfyUI's /view endpoint."""
    query = urllib.parse.urlencode({
        "filename": item.get("filename", ""),
        "subfolder": item.get("subfolder", ""),
        "type": item.get("type", "output"),
    })
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"{server}/view?{query}") as r:
        dest.write_bytes(r.read())
    return dest


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
            prompt_id = post(args.server, wf)
            history = wait_for(args.server, prompt_id)

            # Surface an execution error instead of silently returning nothing.
            status = history.get("status") or {}
            if status.get("status_str") == "error":
                for msg in status.get("messages") or []:
                    if msg and msg[0] == "execution_error":
                        print(f"  {beat_id}: {msg[1].get('exception_message')}",
                              file=sys.stderr)
                continue

            entry = (history.get("outputs") or {}).get("24") or {}
            images = entry.get("images") or []
            if not images:
                print(f"  {beat_id}: no image in outputs {list((history.get('outputs') or {}).keys())}",
                      file=sys.stderr)
                continue

            dest = out / f"{beat_id}.png"
            _fetch_image(args.server, images[0], dest)
            print(f"  {beat_id}: saved {dest}")
            made += 1
        except (urllib.error.URLError, TimeoutError) as error:
            print(f"  {beat_id}: {error}", file=sys.stderr)

    print(f"generated {made} startframe(s)")
    return 0 if made else 1


if __name__ == "__main__":
    sys.exit(main())
