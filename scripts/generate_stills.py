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

from face_swap_video import swap_faces_in_frame
from lora_adapters import select_adapters


# Character reference images for InsightFace
CHARACTER_REFS = {
    "faiz": "references/faiz_ref",
    "juv": "references/juv_ref",
}

REF_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def resolve_ref(references_dir: Path, stem: str) -> Path | None:
    """Find a reference image by stem, accepting any supported extension."""
    for ext in REF_EXTENSIONS:
        candidate = references_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None

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
    """Load SDXL and register every available LoRA adapter.

    Adapters are registered but deliberately NOT activated here: which
    character is in frame changes per beat, so activation happens in
    activate_adapters() once the beat's cast is known.
    """
    pipe = StableDiffusionXLPipeline.from_single_file(
        model_path,
        torch_dtype=dtype,
        use_safetensors=True,
        variant="fp16",
    )
    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(device)

    # Loading alone is not enough: without set_adapters the adapters stay
    # inactive and the stills are generated from the base model.
    loaded = []
    for name, path in lora_paths.items():
        if Path(path).exists():
            pipe.load_lora_weights(path, adapter_name=name)
            loaded.append(name)
            print(f"  [OK] Loaded LoRA: {name} from {path}")
        else:
            print(f"  [FAIL] LoRA not found: {path}")

    pipe._loaded_lora_names = loaded  # type: ignore[attr-defined]
    return pipe


def activate_adapters(
    pipe: StableDiffusionXLPipeline,
    characters: list[str],
    weight: float = 0.85,
) -> list[str]:
    """Activate the adapters for the characters actually in this beat."""
    loaded = getattr(pipe, "_loaded_lora_names", [])
    wanted = select_adapters(loaded, characters)
    if not wanted:
        return []

    pipe.set_adapters(wanted, adapter_weights=[weight] * len(wanted))
    print(f"  [OK] Active adapters: {', '.join(wanted)} @ {weight}")
    return wanted


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


def face_similarity(app: FaceAnalysis, frame: np.ndarray, ref_face) -> float:
    """Best cosine similarity between any face in the frame and a reference."""
    faces = app.get(frame)
    if not faces:
        return 0.0
    best = 0.0
    for face in faces:
        score = float(
            np.dot(face.normed_embedding, ref_face.normed_embedding)
            / (
                np.linalg.norm(face.normed_embedding)
                * np.linalg.norm(ref_face.normed_embedding)
                + 1e-9
            )
        )
        best = max(best, score)
    return best


def process_beat(
    beat_id: str,
    beat_prompt: str,
    pipe: StableDiffusionXLPipeline,
    app: FaceAnalysis,
    swapper,
    ref_faces: dict[str, object],
    output_dir: Path,
    negative_prompt: str,
    width: int = 832,
    height: int = 480,
    seed: int = 42,
    gate: float = 0.40,
    candidates: int = 2,
) -> Optional[Path]:
    """Generate a still, swap both faces, and keep the best-scoring candidate."""
    print(f"\n[{beat_id}] Generating still at {width}x{height}...")

    characters = BEAT_CHARACTERS.get(beat_id, ["faiz", "juv"])
    activate_adapters(pipe, characters)
    best = None

    for attempt in range(max(1, candidates)):
        image = generate_still(
            pipe,
            beat_prompt,
            negative_prompt,
            width=width,
            height=height,
            seed=seed + attempt,
        )
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        if swapper is not None and ref_faces:
            frame, _ = swap_faces_in_frame(
                app, swapper, frame, ref_faces, characters, gate
            )

        # Score the character that is actually in frame.
        #
        # This used to be min() over the shot's cast, which reports the ABSENT
        # character's score. A single-person close-up then caps at ~0.2 no
        # matter how good the swap is, because the missing face scores ~0.1
        # against its reference. That is why c01 read 0.817 on one run and
        # 0.152 on the next with the same weights.
        #
        # max() answers the question the gate is asking: did the character who
        # is in this frame come out looking like themselves?
        per_character = {
            char: face_similarity(app, frame, ref_faces[char])
            for char in characters
            if char in ref_faces
        }
        score = max(per_character.values(), default=0.0)
        if not app.get(frame):
            print(f"  candidate {attempt + 1}: no face in frame (ECU shot)")
        else:
            detail = " ".join(f"{c}={s:.2f}" for c, s in per_character.items())
            print(f"  candidate {attempt + 1}: identity score {score:.3f} ({detail})")

        if best is None or score > best[0]:
            best = (score, frame)
        if score >= gate:
            break

    score, frame = best
    output_path = output_dir / f"{beat_id}.png"
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), frame)
    print(f"  ✓ Saved: {output_path} (score {score:.3f})")

    return output_path


def load_beat_prompts(prompts_dir: Path) -> dict[str, str]:
    """Load beat prompts from prompts/*.txt files.

    Prefers `<id>.still.txt` when present. The plain `<id>.txt` is the Wan 2.2
    video prompt, encoded with umt5-xxl at 512 tokens; SDXL encodes with CLIP,
    which truncates at 77, so the video prompt loses its tail -- and the tail is
    where the explicit content and the LoRA triggers sit. build_still_prompts.py
    writes the CLIP-sized variant.
    """
    prompts = {}
    for txt_file in sorted(prompts_dir.glob("c*.txt")):
        if txt_file.name.endswith(".still.txt"):
            continue
        beat_id = txt_file.stem
        still = txt_file.with_name(f"{beat_id}.still.txt")
        source = still if still.is_file() else txt_file
        prompts[beat_id] = source.read_text(encoding="utf-8").strip()
    return prompts


def load_shot_resolutions(spec_path: Path) -> dict[str, tuple[int, int]]:
    """Map shot id -> (width, height) from meta.res_base / meta.res_face.

    Face-closeup shots (build: cf) use res_face; everything else uses res_base.
    """
    try:
        import yaml
        data = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - fall back to CLI defaults
        return {}
    meta = (data or {}).get("meta") or {}
    base = meta.get("res_base")
    face = meta.get("res_face")
    out: dict[str, tuple[int, int]] = {}
    for shot in (data or {}).get("shots") or []:
        sid = shot.get("id")
        if not isinstance(sid, str):
            continue
        res = face if shot.get("build") == "cf" else base
        if isinstance(res, list) and len(res) == 2:
            out[sid] = (int(res[0]), int(res[1]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate SDXL stills with trained LoRAs + InsightFace swap")
    ap.add_argument("--prompts", default="prompts", help="Prompts directory")
    ap.add_argument("--output", default="startframes", help="Output directory for stills")
    ap.add_argument("--references", default="references", help="Character reference images directory")
    ap.add_argument("--loras", default="models/loras/trained", help="Trained LoRAs directory")
    ap.add_argument("--sdxl-model", default="models/sdxl_base.safetensors", help="SDXL base model path")
    ap.add_argument(
        "--negative",
        default="negatives/still_neg.txt",
        help=(
            "Negative prompt file. Defaults to the CLIP-sized stills list: "
            "artifact_neg.txt is 268 tokens and CLIP truncates at 77, so most "
            "of it never reached the model."
        ),
    )
    ap.add_argument("--spec", default="docs/beats.yaml", help="Spec for per-shot resolution")
    ap.add_argument("--swapper-model", default="models/inswapper_128.onnx", help="InsightFace swapper")
    ap.add_argument("--width", type=int, default=832, help="Fallback width")
    ap.add_argument("--height", type=int, default=480, help="Fallback height")
    ap.add_argument("--steps", type=int, default=30, help="Inference steps")
    ap.add_argument("--guidance", type=float, default=7.0, help="Guidance scale")
    ap.add_argument("--seed", type=int, default=1000, help="Base seed")
    ap.add_argument("--gate", type=float, default=0.40, help="Identity accept threshold")
    ap.add_argument("--candidates", type=int, default=2, help="Candidates per beat")
    ap.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    ap.add_argument("--dtype", choices=["fp16", "fp32"], default="fp16", help="Precision")
    ap.add_argument("--beats", nargs="+", help="Specific beats to process (default: all)")
    args = ap.parse_args()

    negative_prompt = Path(args.negative).read_text(encoding="utf-8").strip()
    neg_tokens = None
    try:
        from transformers import CLIPTokenizer

        _tok = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        neg_tokens = len(_tok(negative_prompt, truncation=False)["input_ids"])
    except Exception:  # noqa: BLE001 - offline
        pass
    if neg_tokens is not None and neg_tokens > 77:
        print(
            f"  WARNING: negative prompt is {neg_tokens} CLIP tokens; only the "
            f"first 77 reach the model. Use negatives/still_neg.txt."
        )
    beat_prompts = load_beat_prompts(Path(args.prompts))
    if args.beats:
        beat_prompts = {k: v for k, v in beat_prompts.items() if k in args.beats}
    resolutions = load_shot_resolutions(Path(args.spec))

    print(f"Processing {len(beat_prompts)} beats")

    print("Loading InsightFace...")
    app = load_insightface()

    # Reference faces (the swapper needs the Face object, not just embeddings).
    ref_faces = {}
    for char, ref_stem in CHARACTER_REFS.items():
        full_path = resolve_ref(Path(args.references), Path(ref_stem).name)
        if full_path is None:
            print(f"  ✗ Reference not found for {char} in {args.references}")
            continue
        img = cv2.imread(str(full_path))
        faces = app.get(img) if img is not None else []
        if not faces:
            print(f"  ✗ No face in reference for {char}")
            continue
        ref_faces[char] = max(
            faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )
        print(f"  Loaded reference for {char}: {full_path.name}")

    swapper = None
    swapper_path = Path(args.swapper_model)
    if swapper_path.is_file():
        from insightface.model_zoo import get_model
        swapper = get_model(str(swapper_path), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        print("  Loaded inswapper")
    else:
        print(f"  ✗ Swapper model missing: {swapper_path} - stills will not be identity-swapped")

    lora_paths = {}
    lora_dir = Path(args.loras)
    for char in ["faiz", "juv", "both"]:
        lora_file = lora_dir / f"{char}_lora.safetensors"
        if lora_file.exists():
            lora_paths[char] = str(lora_file)

    print("Loading SDXL pipeline...")
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    pipe = load_sdxl_pipeline(args.sdxl_model, lora_paths, args.device, dtype)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for beat_id, prompt in beat_prompts.items():
        width, height = resolutions.get(beat_id, (args.width, args.height))
        try:
            process_beat(
                beat_id,
                prompt,
                pipe,
                app,
                swapper,
                ref_faces,
                output_dir,
                negative_prompt,
                width,
                height,
                args.seed + int(beat_id[1:]) * 10,
                args.gate,
                args.candidates,
            )
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())