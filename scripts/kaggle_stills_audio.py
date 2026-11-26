#!/usr/bin/env python3
"""Build and deploy a Kaggle kernel that produces startframes and narration.

The paid GPU is only needed for video rendering. Stills (SDXL + trained LoRAs +
InsightFace swap) and narration (Chatterbox Hindi) both fit on a free Kaggle T4,
so running them there removes them from the AIC Cloud bill entirely.

Outputs, downloaded with `--output`:
    startframes/c01.png ... c70.png
    audio/c01.wav ... c70.wav
    audio/index.json
    stills_audio.zip

Usage:
    python scripts/kaggle_stills_audio.py --push
    python scripts/kaggle_stills_audio.py --status
    python scripts/kaggle_stills_audio.py --output
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
DEPLOY_DIR = HERE / "deploy" / "stills_audio"
DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

KERNEL_ID = "sofiya009/wan22-stills-audio"
KERNEL_TITLE = "Wan22 Stills Audio"

# Datasets the kernel mounts. The repo itself is cloned at runtime so the
# scripts stay in one place.
DATASET_SOURCES = [
    "sofiya009/lora-train-faiz",
    "sofiya009/lora-train-juveriya",
    "sofiya009/lora-weights-faiz-juv-v2",
    "sofiya009/inswapper-128",
]

REPO_URL = "https://github.com/dudefaiz09boop/wan22-production-pipeline.git"


def _code_cell(text: str) -> dict:
    lines = text.split("\n")
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    cell_id = hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]
    return {
        "cell_type": "code",
        "id": cell_id,
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source,
    }


def build_notebook() -> dict:
    cells: list[dict] = []

    cells.append(_code_cell("\n".join([
        "# 1. Environment",
        "import json, os, shutil, subprocess, sys",
        "import torch",
        "print('torch', torch.__version__)",
        "print('gpus', torch.cuda.device_count())",
        "print('gpu0', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')",
        "print('free GB', round(shutil.disk_usage('/kaggle/working').free / 1024**3, 1))",
    ])))

    cells.append(_code_cell("\n".join([
        "# 2. Dependencies",
        "# torchao is removed: Kaggle ships 0.10.0 and peft refuses to import below",
        "# 0.16.0. Nothing here uses torchao quantisation.",
        "!pip install -q --upgrade pip",
        "!pip install -q 'numpy>=2.0' 'ftfy>=6.1' 'diffusers>=0.30' \\",
        "    'transformers>=4.41' 'accelerate>=0.30' 'safetensors>=0.4' \\",
        "    'peft>=0.11' 'insightface>=0.7' 'onnxruntime-gpu>=1.18' \\",
        "    'opencv-python-headless>=4.10' 'tqdm>=4.66' 'pyyaml>=6.0' \\",
        "    'edge-tts>=6.1'",
        "!pip uninstall -y -q torchao 2>/dev/null | tail -1",
        "!apt-get -y -qq install git ffmpeg libgl1 libglib2.0-0 2>/dev/null | tail -2",
        "",
        "# chatterbox-tts pins numpy<1.26 and diffusers==0.29.0, which cannot",
        "# coexist with the numpy 2 / diffusers 0.30 the stills need. Install it",
        "# without its pins; the versions it actually needs are already present.",
        "# This mirrors scripts/setup_aic_fast.sh.",
        "!pip install -q --no-deps chatterbox-tts==0.1.4 \\",
        "    resemble-perth 'conformer==0.3.2' s3tokenizer 'pykakasi==2.3.0' \\",
        "    pyloudnorm omegaconf spacy-pkuseg 'librosa==0.11.0'",
        "",
        "import numpy, cv2, insightface, yaml",
        "print('numpy', numpy.__version__)",
        "print('insightface', insightface.__version__)",
        "import chatterbox",
        "print('chatterbox ok')",
    ])))

    cells.append(_code_cell("\n".join([
        "# 3. Clone the pipeline",
        "import os, subprocess",
        "",
        "REPO = '/kaggle/working/repo'",
        "if not os.path.isdir(REPO):",
        "    subprocess.run(['git', 'clone', '--depth', '1',",
        f"                    '{REPO_URL}', REPO], check=True)",
        "os.chdir(REPO)",
        "print('cwd', os.getcwd())",
        "print(subprocess.run(['git', 'log', '--oneline', '-1'],",
        "                     capture_output=True, text=True).stdout)",
    ])))

    cells.append(_code_cell("\n".join([
        "# 4. Stage the trained LoRAs and the SDXL base model",
        "import glob, os, shutil",
        "",
        "LORA_DIR = 'models/loras/trained'",
        "os.makedirs(LORA_DIR, exist_ok=True)",
        "",
        "def find(name):",
        "    for path in glob.glob(f'/kaggle/input/**/{name}', recursive=True):",
        "        if os.path.getsize(path) > 1_000_000:",
        "            return path",
        "    return None",
        "",
        "for name in ('faiz_lora.safetensors', 'juv_lora.safetensors'):",
        "    src = find(name)",
        "    if not src:",
        "        raise SystemExit(f'{name} not found in /kaggle/input')",
        "    shutil.copy2(src, os.path.join(LORA_DIR, name))",
        "    print('staged', name, os.path.getsize(src), 'bytes')",
        "",
        "from huggingface_hub import hf_hub_download",
        "os.makedirs('models', exist_ok=True)",
        "base = hf_hub_download(",
        "    repo_id='RunDiffusion/Juggernaut-XL-v9',",
        "    filename='Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors',",
        "    local_dir='models',",
        ")",
        "shutil.copy2(base, 'models/sdxl_base.safetensors')",
        "print('base', os.path.getsize('models/sdxl_base.safetensors'), 'bytes')",
    ])))

    cells.append(_code_cell("\n".join([
        "# 5. InsightFace swapper model",
        "# Mounted from sofiya009/inswapper-128. The Hugging Face repo is gated",
        "# (401) and the facefusion release path 404s, so neither can be fetched",
        "# anonymously at runtime.",
        "import glob, os, shutil",
        "",
        "SWAPPER = 'models/inswapper_128.onnx'",
        "if not os.path.isfile(SWAPPER):",
        "    found = glob.glob('/kaggle/input/**/inswapper_128.onnx', recursive=True)",
        "    if not found:",
        "        raise SystemExit('inswapper_128.onnx not found; attach sofiya009/inswapper-128')",
        "    shutil.copy2(found[0], SWAPPER)",
        "print('swapper', os.path.getsize(SWAPPER), 'bytes')",
    ])))

    cells.append(_code_cell("\n".join([
        "# 6. Build the prompts",
        "# prompts/ is gitignored, so a fresh clone has none. build_prompts compiles",
        "# docs/beats.yaml into prompts/c*.json. --lang hi keeps the Devanagari",
        "# dialogue; the default used to be en, which selected the romanized text.",
        "import subprocess, sys",
        "",
        "cmd = [sys.executable, 'scripts/build_prompts.py', '--no-frames', '--lang', 'hi']",
        "print('RUN:', ' '.join(cmd), flush=True)",
        "rc = subprocess.call(cmd, stdout=sys.stdout, stderr=sys.stderr)",
        "print('build_prompts rc:', rc, flush=True)",
        "import glob",
        "prompts = sorted(glob.glob('prompts/c*.json'))",
        "print('prompts built:', len(prompts))",
        "assert prompts, 'no prompts built'",
    ])))

    cells.append(_code_cell("\n".join([
        "# 7. Generate the startframes",
        "# SDXL + per-character LoRAs + InsightFace swap, per-shot resolution.",
        "import subprocess, sys",
        "",
        "cmd = [sys.executable, 'scripts/generate_stills.py',",
        "       '--prompts', 'prompts',",
        "       '--output', 'startframes',",
        "       '--loras', 'models/loras/trained',",
        "       '--sdxl-model', 'models/sdxl_base.safetensors',",
        "       '--swapper-model', 'models/inswapper_128.onnx',",
        "       '--spec', 'docs/beats.yaml',",
        "       '--device', 'cuda']",
        "print('RUN:', ' '.join(cmd), flush=True)",
        "rc = subprocess.call(cmd, stdout=sys.stdout, stderr=sys.stderr)",
        "print('stills rc:', rc, flush=True)",
        "import glob",
        "made = sorted(glob.glob('startframes/c*.png'))",
        "print('startframes produced:', len(made))",
        "assert made, 'no startframes produced'",
    ])))

    cells.append(_code_cell("\n".join([
        "# 8. Generate the narration",
        "# Chatterbox Hindi. The dialogue is Devanagari, so the script decides the",
        "# engine and the voice; Ammi's Urdu turns fall through to Edge ur-PK.",
        "import subprocess, sys",
        "",
        "cmd = [sys.executable, 'scripts/generate_narration.py',",
        "       '--prompts', 'prompts',",
        "       '--output', 'audio',",
        "       '--engine', 'chatterbox',",
        "       '--device', 'cuda']",
        "print('RUN:', ' '.join(cmd), flush=True)",
        "rc = subprocess.call(cmd, stdout=sys.stdout, stderr=sys.stderr)",
        "print('narration rc:', rc, flush=True)",
        "import glob",
        "wavs = sorted(glob.glob('audio/c*.wav'))",
        "print('audio clips produced:', len(wavs))",
        "assert wavs, 'no narration produced'",
    ])))

    cells.append(_code_cell("\n".join([
        "# 9. Package",
        "import os, zipfile",
        "",
        "ZIP = '/kaggle/working/stills_audio.zip'",
        "with zipfile.ZipFile(ZIP, 'w', zipfile.ZIP_DEFLATED) as zf:",
        "    for root in ('startframes', 'audio'):",
        "        for dirpath, _, names in os.walk(root):",
        "            for name in names:",
        "                if name.endswith(('.png', '.wav', '.json')):",
        "                    path = os.path.join(dirpath, name)",
        "                    zf.write(path, path)",
        "print('zip', os.path.getsize(ZIP), 'bytes')",
        "",
        "# Also leave the trees at the top level so --output can pull them directly.",
        "import shutil",
        "for root in ('startframes', 'audio'):",
        "    dest = f'/kaggle/working/{root}'",
        "    if os.path.abspath(root) != os.path.abspath(dest):",
        "        shutil.copytree(root, dest, dirs_exist_ok=True)",
        "print('done')",
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
        "code_file": "stills_audio.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": DATASET_SOURCES,
        "competition_sources": [],
        "kernel_sources": [],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build/deploy the stills + audio Kaggle kernel")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--output", action="store_true")
    ap.add_argument("--output-dir", default="kaggle_out/stills_audio")
    args = ap.parse_args()

    nb_path = DEPLOY_DIR / "stills_audio.ipynb"
    nb_path.write_text(json.dumps(build_notebook(), indent=1), encoding="utf-8")
    print(f"Notebook saved: {nb_path}")

    meta_path = DEPLOY_DIR / "kernel-metadata.json"
    meta_path.write_text(json.dumps(build_metadata(), indent=2), encoding="utf-8")
    print(f"Metadata saved: {meta_path} (kernel {KERNEL_ID})")

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
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["kaggle", "kernels", "output", KERNEL_ID, "-p", str(out_dir)])
        print(f"Artifacts written to {out_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
