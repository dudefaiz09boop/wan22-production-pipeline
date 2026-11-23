#!/usr/bin/env python3
"""Build and deploy a Kaggle kernel that verifies the trained LoRAs.

Training loss only proves the trainer ran. This kernel proves the LoRA
actually reproduces each character:

1. Load the SDXL base model plus one trained LoRA.
2. Generate test images from prompts that use the LoRA trigger word.
3. Detect faces with InsightFace and compare each face embedding against
   reference embeddings taken from that character's own training crops.
4. Report min/mean cosine similarity and pass/fail against a gate.

A LoRA that does not move the face toward its reference set scores near the
baseline (no-LoRA) similarity and fails the gate.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from pathlib import Path

HERE = Path(__file__).parent
DEPLOY_DIR = HERE / "deploy" / "verify"
DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

KERNEL_ID = "sofiya009/lora-verify-faiz-juveriya"
KERNEL_TITLE = "LoRA Verify Faiz Juveriya"

BASE_MODEL_REPO = "RunDiffusion/Juggernaut-XL-v9"
BASE_MODEL_FILE = "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"

# Cosine gate. The fzjv pipeline used 0.40 for an accepted identity match.
GATE = 0.40

# Test prompts per character. Deliberately varied pose/lighting so the score
# reflects identity rather than one memorised composition.
TEST_PROMPTS = {
    "faiz": [
        "faiz, south asian man, short black hair, stubble, portrait, neutral expression, studio lighting",
        "faiz, south asian man, short black hair, stubble, three quarter view, window light",
        "faiz, south asian man, short black hair, stubble, looking at camera, warm lamp light",
        "faiz, south asian man, short black hair, stubble, slight smile, daylight",
    ],
    "juv": [
        "juv, south asian woman, long dark hair, portrait, neutral expression, studio lighting",
        "juv, south asian woman, long dark hair, three quarter view, window light",
        "juv, south asian woman, long dark hair, looking at camera, warm lamp light",
        "juv, south asian woman, long dark hair, slight smile, daylight",
    ],
}

NEGATIVE = (
    "blurry, deformed, extra fingers, plastic skin, cartoon, "
    "generic european face, watermark, text, low quality"
)


def _code_cell(text: str) -> dict:
    lines = text.split("\n")
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    return {
        "cell_type": "code",
        "id": uuid.uuid4().hex[:8],
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source,
    }


def build_notebook() -> dict:
    cells: list[dict] = []

    cells.append(_code_cell("\n".join([
        "# 1. Environment",
        "import json, os, sys, glob, shutil",
        "import torch",
        "print('torch', torch.__version__)",
        "print('gpus', torch.cuda.device_count())",
        "print('free GB', round(shutil.disk_usage('/kaggle/working').free / 1024**3, 1))",
    ])))

    cells.append(_code_cell("\n".join([
        "# 2. Dependencies",
        "!pip install -q --upgrade pip",
        "!pip install -q 'numpy>=2.0' 'ftfy>=6.1' 'diffusers>=0.30' \\",
        "    'transformers>=4.41' 'accelerate>=0.30' 'safetensors>=0.4' \\",
        "    'peft>=0.11' 'insightface>=0.7' 'onnxruntime-gpu>=1.18' \\",
        "    'opencv-python-headless>=4.10' 'tqdm>=4.66'",
        "!apt-get -y -qq install git ffmpeg libgl1 libglib2.0-0 2>/dev/null | tail -2",
        "import numpy, cv2, insightface",
        "print('numpy', numpy.__version__)",
        "print('insightface', insightface.__version__)",
    ])))

    cells.append(_code_cell("\n".join([
        "# 3. Locate the trained LoRAs and the reference crops",
        "import os",
        "",
        "WORK = '/kaggle/working'",
        "LORA_DIR = '/kaggle/input/lora-nuke-faiz-juv'",
        "",
        "def find_lora(name):",
        "    for base in (LORA_DIR, WORK, '/kaggle/input'):",
        "        for path in glob.glob(f'{base}/**/{name}', recursive=True):",
        "            if os.path.getsize(path) > 1_000_000:",
        "                return path",
        "    return None",
        "",
        "LORAS = {",
        "    'faiz': find_lora('faiz_lora.safetensors'),",
        "    'juv': find_lora('juv_lora.safetensors'),",
        "    'faiz_juv': find_lora('faiz_juv_lora.safetensors'),",
        "}",
        "for key, path in LORAS.items():",
        "    print(key, '->', path, os.path.getsize(path) if path else 'MISSING')",
        "",
        "assert LORAS['faiz'], 'faiz LoRA not found'",
        "assert LORAS['juv'], 'juv LoRA not found'",
    ])))

    cells.append(_code_cell("\n".join([
        "# 4. Build reference embeddings from each character's training crops",
        "import cv2, numpy as np",
        "from insightface.app import FaceAnalysis",
        "",
        "app = FaceAnalysis(name='buffalo_l',",
        "                   providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])",
        "app.prepare(ctx_id=0, det_size=(640, 640))",
        "",
        "def dataset_dir(slug):",
        "    for candidate in (f'/kaggle/input/{slug}', f'/kaggle/input/datasets/sofiya009/{slug}'):",
        "        if os.path.isdir(candidate):",
        "            return candidate",
        "    return None",
        "",
        "def build_refs(slug, limit=40):",
        "    root = dataset_dir(slug)",
        "    if not root:",
        "        print(slug, 'NOT MOUNTED')",
        "        return []",
        "    paths = []",
        "    for dirpath, _, names in os.walk(root):",
        "        for name in names:",
        "            if name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):",
        "                paths.append(os.path.join(dirpath, name))",
        "    paths.sort()",
        "    refs = []",
        "    for path in paths[:limit]:",
        "        img = cv2.imread(path)",
        "        if img is None:",
        "            continue",
        "        faces = app.get(img)",
        "        if not faces:",
        "            continue",
        "        face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0])*(f.bbox[3]-f.bbox[1]))",
        "        refs.append(face.normed_embedding)",
        "    print(slug, 'reference embeddings:', len(refs))",
        "    return refs",
        "",
        "REFS = {",
        "    'faiz': build_refs('lora-train-faiz'),",
        "    'juv': build_refs('lora-train-juveriya'),",
        "}",
        "assert REFS['faiz'], 'no faiz reference embeddings'",
        "assert REFS['juv'], 'no juv reference embeddings'",
    ])))

    cells.append(_code_cell("\n".join([
        "# 5. Load the SDXL base model",
        "from huggingface_hub import hf_hub_download",
        "from diffusers import StableDiffusionXLPipeline, EulerDiscreteScheduler",
        "",
        "MODEL_DIR = '/kaggle/working/base'",
        "os.makedirs(MODEL_DIR, exist_ok=True)",
        "base_path = hf_hub_download(",
        "    repo_id='RunDiffusion/Juggernaut-XL-v9',",
        "    filename='Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors',",
        "    local_dir=MODEL_DIR,",
        ")",
        "print('base:', base_path, os.path.getsize(base_path), 'bytes')",
        "",
        "def load_pipe():",
        "    pipe = StableDiffusionXLPipeline.from_single_file(",
        "        base_path, torch_dtype=torch.float16, use_safetensors=True,",
        "    )",
        "    pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)",
        "    pipe.to('cuda')",
        "    pipe.set_progress_bar_config(disable=True)",
        "    pipe.enable_vae_tiling()",
        "    return pipe",
        "",
        "pipe = load_pipe()",
        "print('pipeline ready')",
    ])))

    cells.append(_code_cell("\n".join([
        "# 6. Scoring helpers",
        "import numpy as np",
        "",
        "def cos(a, b):",
        "    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))",
        "",
        "def score_image(img_bgr, refs):",
        "    faces = app.get(img_bgr)",
        "    if not faces:",
        "        return None",
        "    best = 0.0",
        "    for face in faces:",
        "        for ref in refs:",
        "            best = max(best, cos(face.normed_embedding, ref))",
        "    return best",
        "",
        "def generate(prompt, seed):",
        "    g = torch.Generator('cuda').manual_seed(seed)",
        "    return pipe(",
        "        prompt=prompt,",
        "        negative_prompt=NEGATIVE,",
        "        width=832, height=1216,",
        "        num_inference_steps=28,",
        "        guidance_scale=5.5,",
        "        generator=g,",
        "    ).images[0]",
        "",
        "print('helpers ready')",
    ])))

    cells.append(_code_cell("\n".join([
        "# 7. Baseline: score the base model with NO LoRA",
        "# This is the control. A LoRA must beat it to be doing anything.",
        "import cv2, numpy as np",
        "",
        "BASELINE = {}",
        "for character, prompts in TEST_PROMPTS.items():",
        "    scores = []",
        "    for i, prompt in enumerate(prompts):",
        "        img = generate(prompt, seed=9000 + i)",
        "        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)",
        "        s = score_image(bgr, REFS[character])",
        "        if s is not None:",
        "            scores.append(s)",
        "    BASELINE[character] = round(float(np.mean(scores)), 4) if scores else 0.0",
        "    print(f'baseline {character}: mean={BASELINE[character]} n={len(scores)}')",
    ])))

    cells.append(_code_cell("\n".join([
        "# 8. Score each LoRA against its own reference set",
        "import cv2, numpy as np",
        "",
        "RESULTS = {}",
        "",
        "def evaluate(lora_key, character, lora_path, weight=0.85):",
        "    pipe.load_lora_weights(lora_path, adapter_name='test')",
        "    pipe.set_adapters(['test'], adapter_weights=[weight])",
        "    scores = []",
        "    for i, prompt in enumerate(TEST_PROMPTS[character]):",
        "        img = generate(prompt, seed=9000 + i)",
        "        bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)",
        "        s = score_image(bgr, REFS[character])",
        "        if s is not None:",
        "            scores.append(s)",
        "    pipe.unload_lora_weights()",
        "    mean = float(np.mean(scores)) if scores else 0.0",
        "    low = float(np.min(scores)) if scores else 0.0",
        "    base = BASELINE[character]",
        "    delta = mean - base",
        "    passed = mean >= GATE and delta > 0.02",
        "    RESULTS[lora_key] = {",
        "        'character': character,",
        "        'mean': round(mean, 4),",
        "        'min': round(low, 4),",
        "        'baseline': base,",
        "        'delta': round(delta, 4),",
        "        'n': len(scores),",
        "        'pass': passed,",
        "    }",
        "    print(f'{lora_key}: mean={mean:.4f} min={low:.4f} baseline={base:.4f} delta={delta:+.4f} pass={passed}')",
        "",
        "evaluate('faiz', 'faiz', LORAS['faiz'])",
        "evaluate('juv', 'juv', LORAS['juv'])",
        "if LORAS['faiz_juv']:",
        "    evaluate('faiz_juv', 'faiz', LORAS['faiz_juv'])",
    ])))

    cells.append(_code_cell("\n".join([
        "# 9. Verdict",
        "import json",
        "",
        "print(json.dumps(RESULTS, indent=2))",
        "",
        "failed = [k for k, v in RESULTS.items() if not v['pass']]",
        "if failed:",
        "    print('FAILED:', failed)",
        "    print('A failing LoRA does not reproduce its character. Retrain with more')",
        "    print('steps, higher repeats, or cleaner training crops.')",
        "else:",
        "    print('ALL LORAS PASSED the identity gate')",
        "",
        "with open('/kaggle/working/verify_report.json', 'w') as f:",
        "    json.dump({'gate': GATE, 'baseline': BASELINE, 'results': RESULTS}, f, indent=2)",
        "print('wrote /kaggle/working/verify_report.json')",
    ])))

    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_metadata() -> dict:
    return {
        "id": KERNEL_ID,
        "title": KERNEL_TITLE,
        "code_file": "verify_loras.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [
            "sofiya009/lora-train-faiz",
            "sofiya009/lora-train-juveriya",
        ],
        "competition_sources": [],
        "kernel_sources": [],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build/deploy the LoRA verification kernel")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--output", action="store_true")
    args = ap.parse_args()

    nb_path = DEPLOY_DIR / "verify_loras.ipynb"
    nb_path.write_text(json.dumps(build_notebook(), indent=1), encoding="utf-8")
    print(f"Notebook saved: {nb_path}")

    meta_path = DEPLOY_DIR / "kernel-metadata.json"
    meta_path.write_text(json.dumps(build_metadata(), indent=2), encoding="utf-8")
    print(f"Metadata saved: {meta_path}")

    if args.push:
        result = subprocess.run(
            ["kaggle", "kernels", "push", "-p", str(DEPLOY_DIR)],
            capture_output=True, text=True,
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        if result.returncode != 0:
            return 1

    if args.status:
        subprocess.run(["kaggle", "kernels", "status", KERNEL_ID])

    if args.output:
        subprocess.run(["kaggle", "kernels", "output", KERNEL_ID, "-p", "/tmp/verify_output"])

    return 0


if __name__ == "__main__":
    sys.exit(main())
