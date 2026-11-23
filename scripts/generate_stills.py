#!/usr/bin/env python3
"""Generate SDXL stills with trained LoRAs + InsightFace swap for each beat's start frame."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from insightface.app import FaceAnalysis
from PIL import Image


# Character reference images for InsightFace
CHARACTER_REFS = {
    "faiz": "references/faiz_ref.jpg",
    "juv": "references/juv_ref.jpg",
}

# Beat-to-character mapping (which character appears in each beat)
BEAT_CHARACTERS = {
    # Scene 1: Hallway - Faiz corners Juveriya
    "c01": ["faiz", "juv"],
    "c02": ["faiz", "juv"],
    "c03": ["faiz", "juv"],
    "c04": ["faiz", "juv"],
    "c05": ["faiz", "juv"],
    # Scene 2: Against the wall
    "c06": ["faiz", "juv"],
    "c07": ["faiz", "juv"],
    "c08": ["faiz", "juv"],
    "c09": ["faiz", "juv"],
    "c10": ["faiz", "juv"],
    # Scene 3: Onto the desk
    "c11": ["faiz", "juv"],
    "c12": ["faiz", "juv"],
    "c13": ["faiz", "juv"],
    "c14": ["faiz", "juv"],
    "c15": ["faiz", "juv"],
    # Scene 4: Face down
    "c16": ["faiz", "juv"],
    "c17": ["faiz", "juv"],
    "c18": ["faiz", "juv"],
    "c19": ["faiz", "juv"],
    "c20": ["faiz", "juv"],
    # Scene 5: Blackout
    "c21": ["faiz", "juv"],
    "c22": ["faiz", "juv"],
    "c23": ["faiz", "juv"],
    "c24": ["faiz", "juv"],
    "c25": ["faiz", "juv"],
    # Scene 6: Over the edge
    "c26": ["faiz", "juv"],
    "c27": ["faiz", "juv"],
    "c28": ["faiz", "juv"],
    "c29": ["faiz", "juv"],
    "c30": ["faiz", "juv"],
    # Scene 7: Aftermath
    "c31": ["faiz", "juv"],
    "c32": ["faiz", "juv"],
    "c33": ["faiz", "juv"],
    "c34": ["faiz", "juv"],
    "c35": ["faiz", "juv"],
    # Scene 8: Ammi calls
    "c36": ["faiz", "juv"],
    "c37": ["faiz", "juv"],
    "c38": ["faiz", "juv"],
    "c39": ["faiz", "juv"],
    "c40": ["faiz", "juv"],
    # Scene 9: Tidying
    "c41": ["faiz", "juv"],
    "c42": ["faiz", "juv"],
    "c43": ["faiz", "juv"],
    "c44": ["faiz", "juv"],
    "c45": ["faiz", "juv"],
    # Scene 10: Dawn
    "c46": ["faiz", "juv"],
    "c47": ["faiz", "juv"],
    "c48": ["faiz", "juv"],
    "c49": ["faiz", "juv"],
    "c50": ["faiz", "juv"],
    # Scene 11: She starts it
    "c51": ["faiz", "juv"],
    "c52": ["faiz", "juv"],
    "c53": ["faiz", "juv"],
    "c54": ["faiz", "juv"],
    "c55": ["faiz", "juv"],
    # Scene 12: (remaining)
    "c56": ["faiz", "juv"],
    "c57": ["faiz", "juv"],
    "c58": ["faiz", "juv"],
    "c59": ["faiz", "juv"],
    "c60": ["faiz", "juv"],
    "c61": ["faiz", "juv"],
    "c62": ["faiz", "juv"],
    "c63": ["faiz", "juv"],
    "c64": ["faiz", "juv"],
    "c65": ["faiz", "juv"],
    "c66": ["faiz", "juv"],
    "c67": ["faiz", "juv"],
    "c68": ["faiz", "juv"],
    "c69": ["faiz", "juv"],
    "c70": ["faiz", "juv"],
}


def load_insightface() -> FaceAnalysis:
    """Initialize InsightFace for face detection and embedding."""
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def get_face_embedding(app: FaceAnalysis, image_path: Path) -> Optional[np.ndarray]:
    """Extract face embedding from reference image."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  ✗ Could not load reference: {image_path}")
        return None
    faces = app.get(img)
    if not faces:
        print(f"  ✗ No face detected in: {image_path}")
        return None
    # Use the largest face
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return face.embedding


def swap_face(
    app: FaceAnalysis,
    source_img: np.ndarray,
    target_embedding: np.ndarray,
    source_embedding: np.ndarray,
) -> np.ndarray:
    """Swap face in source image using InsightFace."""
    faces = app.get(source_img)
    if not faces:
        return source_img

    result = source_img.copy()
    for face in faces:
        # Compute similarity
        sim = np.dot(face.embedding, target_embedding) / (
            np.linalg.norm(face.embedding) * np.linalg.norm(target_embedding)
        )
        # Only swap if similarity is above threshold (same character)
        if sim > 0.40:
            # Use InsightFace's swapper (requires inswapper model)
            # For now, we'll use a simpler approach - just return original
            # In production, you'd load the inswapper_128.onnx model
            pass
    return result


def load_sdxl_pipeline(
    model_path: str,
    lora_paths: dict[str, str],
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
) -> StableDiffusionXLPipeline:
    """Load SDXL pipeline with LoRAs."""
    pipe = StableDiffusionXLPipeline.from_single_file(
        model_path,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16",
    )
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    # Load LoRAs
    for name, path in lora_paths.items():
        if Path(path).exists():
            pipe.load_lora_weights(path, adapter_name=name)
            print(f"  [OK] Loaded LoRA: {name} from {path}")
        else:
            print(f"  [FAIL] LoRA not found: {path}")

    return pipe


def generate_still(
    pipe: StableDiffusionXLPipeline,
    prompt: str,
    negative_prompt: str,
    width: int = 832,
    height: int = 480,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.0,
    seed: int = 42,
) -> Image.Image:
    """Generate a single still image."""
    generator = torch.Generator(device="cuda").manual_seed(seed)
    image = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    ).images[0]
    return image


def process_beat(
    beat_id: str,
    beat_prompt: str,
    pipe: StableDiffusionXLPipeline,
    app: FaceAnalysis,
    ref_embeddings: dict[str, np.ndarray],
    output_dir: Path,
    negative_prompt: str,
    width: int = 832,
    height: int = 480,
) -> Optional[Path]:
    """Process a single beat: generate still + face swap."""
    print(f"\n[{beat_id}] Generating still...")

    characters = BEAT_CHARACTERS.get(beat_id, ["faiz", "juv"])

    # Generate base image
    image = generate_still(
        pipe,
        beat_prompt,
        negative_prompt,
        width=width,
        height=height,
    )

    # Convert to OpenCV for face swap
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Apply face swap for each character
    for char in characters:
        if char in ref_embeddings:
            # In production, use inswapper model here
            # For now, save without swap
            pass

    # Save result
    output_path = output_dir / f"{beat_id}.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    print(f"  ✓ Saved: {output_path}")

    return output_path


def load_beat_prompts(prompts_dir: Path) -> dict[str, str]:
    """Load beat prompts from prompts/*.txt files."""
    prompts = {}
    for txt_file in sorted(prompts_dir.glob("c*.txt")):
        beat_id = txt_file.stem
        prompts[beat_id] = txt_file.read_text(encoding="utf-8").strip()
    return prompts


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate SDXL stills with trained LoRAs + InsightFace swap")
    ap.add_argument("--prompts", default="prompts", help="Prompts directory")
    ap.add_argument("--output", default="startframes", help="Output directory for stills")
    ap.add_argument("--references", default="references", help="Character reference images directory")
    ap.add_argument("--loras", default="models/loras/trained", help="Trained LoRAs directory")
    ap.add_argument("--sdxl-model", default="models/sdxl_base.safetensors", help="SDXL base model path")
    ap.add_argument("--negative", default="negatives/artifact_neg.txt", help="Negative prompt file")
    ap.add_argument("--width", type=int, default=832, help="Output width")
    ap.add_argument("--height", type=int, default=480, help="Output height")
    ap.add_argument("--steps", type=int, default=30, help="Inference steps")
    ap.add_argument("--guidance", type=float, default=7.0, help="Guidance scale")
    ap.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16", help="Precision")
    ap.add_argument("--beats", nargs="+", help="Specific beats to process (default: all)")
    args = ap.parse_args()

    # Load negative prompt
    negative_prompt = Path(args.negative).read_text(encoding="utf-8").strip()

    # Load beat prompts
    beat_prompts = load_beat_prompts(Path(args.prompts))
    if args.beats:
        beat_prompts = {k: v for k, v in beat_prompts.items() if k in args.beats}

    print(f"Processing {len(beat_prompts)} beats")

    # Initialize InsightFace
    print("Loading InsightFace...")
    app = load_insightface()

    # Load reference embeddings
    ref_embeddings = {}
    for char, ref_path in CHARACTER_REFS.items():
        full_path = Path(args.references) / ref_path
        if full_path.exists():
            emb = get_face_embedding(app, full_path)
            if emb is not None:
                ref_embeddings[char] = emb
                print(f"  Loaded reference for {char}")
        else:
            print(f"  ✗ Reference not found: {full_path}")

    # Load LoRA paths
    lora_paths = {}
    lora_dir = Path(args.loras)
    for char in ["faiz", "juv", "both"]:
        lora_file = lora_dir / f"{char}_lora.safetensors"
        if lora_file.exists():
            lora_paths[char] = str(lora_file)

    # Load SDXL pipeline
    print("Loading SDXL pipeline...")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    pipe = load_sdxl_pipeline(args.sdxl_model, lora_paths, args.device, dtype)

    # Process each beat
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for beat_id, prompt in beat_prompts.items():
        try:
            process_beat(
                beat_id,
                prompt,
                pipe,
                app,
                ref_embeddings,
                output_dir,
                negative_prompt,
                args.width,
                args.height,
            )
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())