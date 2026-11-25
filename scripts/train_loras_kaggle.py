#!/usr/bin/env python3
"""Build and deploy the Kaggle LoRA training kernel.

Trains three SDXL LoRAs (faiz, juv, faiz_juv) with Kohya-ss/sd-scripts on a
free Kaggle T4. Face crops come from the mounted Kaggle datasets; the SDXL
base model is downloaded from Hugging Face at runtime.

Notes on the generated notebook:
- Every cell is a code cell. Kaggle's papermill/nbconvert pipeline rejects
  markdown cells that carry papermill metadata, so markdown is emitted as
  commented code instead.
- Each cell is self-contained: imports are repeated where needed so a cell
  never depends on a name defined in an earlier cell.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from pathlib import Path

HERE = Path(__file__).parent
DEPLOY_DIR = HERE / "deploy" / "training"
DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

KERNEL_ID = "sofiya009/lora-nuke-faiz-juveriya-sdxl"
KERNEL_TITLE = "LoRA Nuke Faiz Juveriya SDXL"

BASE_MODEL_REPO = "RunDiffusion/Juggernaut-XL-v9"
BASE_MODEL_FILE = "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"

WORK_DIR = "/kaggle/working/training"
OUTPUT_ROOT = "/kaggle/working/output"
MODEL_DIR = "/kaggle/working/base"


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


def build_notebook(only: str = "all") -> dict:
    """Build the training notebook.

    only: 'all' trains every LoRA in one run; 'faiz', 'juv' or 'combined'
    builds a single-LoRA notebook. Single-LoRA runs are the reliable path:
    each gets its own Kaggle session, so one slow or failing LoRA cannot
    take the others down with it.
    """
    cells: list[dict] = []

    # 1. Environment
    cells.append(_code_cell("\n".join([
        "# 1. Environment and GPU check",
        "import json, os, sys, glob, shutil, subprocess, zipfile",
        "import torch",
        "",
        "print('PyTorch:', torch.__version__)",
        "print('CUDA:', torch.version.cuda)",
        "print('GPU count:', torch.cuda.device_count())",
        "print('GPU 0:', torch.cuda.get_device_name(0))",
        "print('working free GB:', round(shutil.disk_usage('/kaggle/working').free / 1024**3, 1))",
    ])))

    # 2. Dependencies
    cells.append(_code_cell("\n".join([
        "# 2. Install dependencies (full sd-scripts requirement set)",
        "# Keep numpy 2.x: Kaggle's preinstalled pandas requires it.",
        "!pip install -q --upgrade pip",
        "!pip install -q 'numpy>=2.0' 'ftfy>=6.1' 'voluptuous>=0.14' 'toml>=0.10' \\",
        "    'accelerate>=0.30' 'transformers>=4.41' 'diffusers>=0.30' \\",
        "    'safetensors>=0.4' 'peft>=0.11' 'bitsandbytes>=0.43' \\",
        "    'opencv-python-headless>=4.10' 'tqdm>=4.66' 'omegaconf>=2.3' \\",
        "    'lycoris-lora>=1.8' 'kornia>=0.7' 'tensorboard>=2.16' \\",
        "    'einops>=0.7' 'prodigyopt>=1.0' 'lion-pytorch>=0.1' \\",
        "    'schedulefree>=1.4' 'pytorch-optimizer>=3.0' 'rich>=13.0' \\",
        "    'imagesize>=1.4' 'toolz>=0.12' 'easydict>=1.13'",
        "!apt-get -y -qq install git ffmpeg libgl1 libglib2.0-0 2>/dev/null | tail -2",
        "",
        "!cd /kaggle/working && rm -rf sd-scripts && git clone --depth 1 https://github.com/kohya-ss/sd-scripts.git",
        "!cd /kaggle/working/sd-scripts && pip install -q -e . 2>&1 | tail -3",
        "",
        "import numpy, ftfy, voluptuous, toml, pandas",
        "sys.path.insert(0, '/kaggle/working/sd-scripts')",
        "print('numpy', numpy.__version__)",
        "print('pandas', pandas.__version__)",
        "print('ftfy', ftfy.__version__)",
        "print('voluptuous', voluptuous.__version__)",
        "print('sd-scripts ready')",
    ])))

    # 3. Preflight: verify sd-scripts imports before spending GPU time
    cells.append(_code_cell("\n".join([
        "# 3. Preflight: verify sd-scripts imports before spending GPU time",
        "import subprocess, sys",
        "",
        "check = subprocess.run(",
        "    [sys.executable, '-c',",
        "     'import sys; sys.path.insert(0, \"/kaggle/working/sd-scripts\"); '",
        "     'from library import sdxl_model_util, sdxl_train_util, strategy_sdxl; '",
        "     'print(\"sd-scripts imports OK\")'],",
        "    capture_output=True, text=True,",
        ")",
        "print(check.stdout)",
        "print(check.stderr[-3000:])",
        "assert check.returncode == 0, 'sd-scripts import preflight failed'",
    ])))

    # 4. Stage datasets
    cells.append(_code_cell("\n".join([
        "# 4. Stage face crops into DreamBooth layout",
        "# sd-scripts reads <train_data_dir>/<repeats>_<concept>/<images>.",
        "import glob, os, shutil",
        "",
        "WORK_DIR = '/kaggle/working/training'",
        "shutil.rmtree(WORK_DIR, ignore_errors=True)",
        "os.makedirs(WORK_DIR, exist_ok=True)",
        "",
        "def find_dataset_dir(slug):",
        "    for candidate in (f'/kaggle/input/{slug}', f'/kaggle/input/datasets/sofiya009/{slug}'):",
        "        if os.path.isdir(candidate):",
        "            return candidate",
        "    return None",
        "",
        "def collect(slug):",
        "    src = find_dataset_dir(slug)",
        "    if not src:",
        "        print(slug, 'NOT MOUNTED')",
        "        return []",
        "    files = []",
        "    for root, _, names in os.walk(src):",
        "        for name in names:",
        "            if name.lower().endswith(('.jpg', '.jpeg', '.png', '.webp', '.txt')):",
        "                files.append(os.path.join(root, name))",
        "    print(slug, 'found', len(files), 'files')",
        "    return files",
        "",
        "def stage(parent_name, files, concept, repeats=4):",
        "    parent = os.path.join(WORK_DIR, parent_name)",
        "    dest = os.path.join(parent, f'{repeats}_{concept}')",
        "    os.makedirs(dest, exist_ok=True)",
        "    for path in files:",
        "        shutil.copy2(path, os.path.join(dest, os.path.basename(path)))",
        "    images = [f for f in os.listdir(dest) if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]",
        "    captions = [f for f in os.listdir(dest) if f.endswith('.txt')]",
        "    print(f'{parent_name}: {len(images)} images, {len(captions)} captions -> {dest}')",
        "    return parent",
        "",
        "faiz_files = collect('lora-train-faiz')",
        "juv_files = collect('lora-train-juveriya')",
        "",
        "# Guard: the two datasets must not share images. A shared crop means the",
        "# face-extraction step wrote the same face into both sets, which trains",
        "# each LoRA on both people and makes the trigger words meaningless.",
        "import hashlib",
        "",
        "def _digest(paths):",
        "    out = {}",
        "    for path in paths:",
        "        if path.lower().endswith('.txt'):",
        "            continue",
        "        with open(path, 'rb') as fh:",
        "            out[hashlib.md5(fh.read()).hexdigest()] = os.path.basename(path)",
        "    return out",
        "",
        "faiz_digests = _digest(faiz_files)",
        "juv_digests = _digest(juv_files)",
        "shared = set(faiz_digests) & set(juv_digests)",
        "print('faiz images:', len(faiz_digests), 'juv images:', len(juv_digests))",
        "print('shared images:', len(shared))",
        "assert not shared, (",
        "    f'{len(shared)} image(s) appear in BOTH datasets, e.g. '",
        "    f'{sorted(faiz_digests[h] for h in shared)[:5]}. '",
        "    'Re-run scripts/prepare_training_data.py so each face is assigned to one character.'",
        ")",
        "",
        "FAIZ_DIR = stage('faiz', faiz_files, 'faiz')",
        "JUV_DIR = stage('juv', juv_files, 'juv')",
        "COMBINED_DIR = stage('combined', faiz_files + juv_files, 'faizjuv')",
        "",
        "assert faiz_files, 'faiz dataset empty'",
        "assert juv_files, 'juv dataset empty'",
        "print('datasets OK')",
    ])))

    # 4. Download base model
    cells.append(_code_cell("\n".join([
        "# 4. Download the SDXL base model",
        "import os",
        "from huggingface_hub import hf_hub_download",
        "",
        "MODEL_DIR = '/kaggle/working/base'",
        "os.makedirs(MODEL_DIR, exist_ok=True)",
        "",
        "model_path = hf_hub_download(",
        "    repo_id='RunDiffusion/Juggernaut-XL-v9',",
        "    filename='Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors',",
        "    local_dir=MODEL_DIR,",
        ")",
        "print('model:', model_path, os.path.getsize(model_path), 'bytes')",
    ])))

    # 5. Training config
    cells.append(_code_cell("\n".join([
        "# 5. Training configuration",
        "# Tuned so all three LoRAs finish inside one Kaggle session.",
        "import json",
        "",
        "CONFIG = {",
        "    'pretrained_model_name_or_path': model_path,",
        "    'network_module': 'networks.lora',",
        "    'network_dim': 32,",
        "    'network_alpha': 16,",
        "    'learning_rate': 1e-4,",
        "    'text_encoder_lr': 5e-5,",
        "    'lr_scheduler': 'cosine',",
        "    'lr_warmup_steps': 50,",
        "    'max_train_steps': 1200,",
        "    'save_every_n_steps': 400,",
        "    'train_batch_size': 1,",
        "    'gradient_accumulation_steps': 2,",
        "    'mixed_precision': 'fp16',",
        "    'optimizer_type': 'AdamW8bit',",
        "    'resolution': '1024,1024',",
        "    'min_bucket_reso': 512,",
        "    'max_bucket_reso': 1536,",
        "    'save_precision': 'fp16',",
        "    'save_model_as': 'safetensors',",
        "    'logging_dir': '/kaggle/working/logs',",
        "}",
        "print(json.dumps(CONFIG, indent=2))",
    ])))

    # 6. Trainer helper
    cells.append(_code_cell("\n".join([
        "# 6. Trainer helper",
        "# Output is streamed (not captured) so progress is visible in the live log.",
        "import glob, os, shutil, subprocess, sys",
        "",
        "WORK_DIR = '/kaggle/working/training'",
        "",
        "def train(name, train_dir, reg_dir=None, steps=None):",
        "    out_dir = f'/kaggle/working/output/{name}'",
        "    os.makedirs(out_dir, exist_ok=True)",
        "    cmd = [",
        "        'accelerate', 'launch',",
        "        '--num_cpu_threads_per_process', '2',",
        "        '--num_processes', '1',",
        "        '--mixed_precision', 'fp16',",
        "        '/kaggle/working/sd-scripts/sdxl_train_network.py',",
        "        '--pretrained_model_name_or_path', CONFIG['pretrained_model_name_or_path'],",
        "        '--network_module', CONFIG['network_module'],",
        "        '--network_dim', str(CONFIG['network_dim']),",
        "        '--network_alpha', str(CONFIG['network_alpha']),",
        "        '--learning_rate', str(CONFIG['learning_rate']),",
        "        '--text_encoder_lr', str(CONFIG['text_encoder_lr']),",
        "        '--lr_scheduler', CONFIG['lr_scheduler'],",
        "        '--lr_warmup_steps', str(CONFIG['lr_warmup_steps']),",
        "        '--max_train_steps', str(steps or CONFIG['max_train_steps']),",
        "        '--save_every_n_steps', str(CONFIG['save_every_n_steps']),",
        "        '--train_batch_size', str(CONFIG['train_batch_size']),",
        "        '--gradient_accumulation_steps', str(CONFIG['gradient_accumulation_steps']),",
        "        '--mixed_precision', CONFIG['mixed_precision'],",
        "        '--gradient_checkpointing',",
        "        '--optimizer_type', CONFIG['optimizer_type'],",
        "        '--resolution', CONFIG['resolution'],",
        "        '--enable_bucket',",
        "        '--min_bucket_reso', str(CONFIG['min_bucket_reso']),",
        "        '--max_bucket_reso', str(CONFIG['max_bucket_reso']),",
        "        '--cache_latents',",
        "        '--cache_latents_to_disk',",
        "        '--sdpa',",
        "        '--no_half_vae',",
        "        '--output_dir', out_dir,",
        "        '--output_name', f'{name}_lora',",
        "        '--save_precision', CONFIG['save_precision'],",
        "        '--save_model_as', CONFIG['save_model_as'],",
        "        '--logging_dir', CONFIG['logging_dir'],",
        "        '--train_data_dir', train_dir,",
        "    ]",
        "    if reg_dir:",
        "        cmd += ['--reg_data_dir', reg_dir]",
        "    print('RUN:', ' '.join(cmd[:8]), '...', flush=True)",
        "    # Stream to the notebook log so progress is visible while it runs.",
        "    rc = subprocess.call(cmd, stdout=sys.stdout, stderr=sys.stderr)",
        "    print('rc:', rc, flush=True)",
        "    found = sorted(glob.glob(f'{out_dir}/{name}_lora*.safetensors'))",
        "    if found:",
        "        final = f'/kaggle/working/{name}_lora.safetensors'",
        "        shutil.copy2(found[-1], final)",
        "        print('saved:', final, os.path.getsize(final), 'bytes', flush=True)",
        "        return True",
        "    print('ERROR: no LoRA produced for', name, flush=True)",
        "    return False",
        "",
        "print('trainer ready')",
    ])))

    # 7. Smoke test: 1 step to validate the whole trainer path
    cells.append(_code_cell("\n".join([
        "# 7. Smoke test: run 1 training step to validate the full path",
        "smoke_ok = train('smoke', FAIZ_DIR, steps=1)",
        "print('smoke_ok =', smoke_ok)",
        "assert smoke_ok, 'smoke test failed - see stderr above'",
    ])))

    # 8-10. Training cells, filtered by the requested target
    if only in ("all", "faiz"):
        cells.append(_code_cell("\n".join([
            "# 8. Train the Faiz LoRA (skipped if already produced)",
            "faiz_ok = os.path.exists('/kaggle/working/faiz_lora.safetensors') or train('faiz', FAIZ_DIR)",
            "print('faiz_ok =', faiz_ok)",
        ])))

    if only in ("all", "juv"):
        cells.append(_code_cell("\n".join([
            "# 9. Train the Juveriya LoRA (skipped if already produced)",
            "juv_ok = os.path.exists('/kaggle/working/juv_lora.safetensors') or train('juv', JUV_DIR)",
            "print('juv_ok =', juv_ok)",
        ])))

    if only in ("all", "combined"):
        cells.append(_code_cell("\n".join([
            "# 10. Train the combined LoRA (skipped if already produced)",
            "combined_ok = os.path.exists('/kaggle/working/faiz_juv_lora.safetensors') or train('faiz_juv', COMBINED_DIR)",
            "print('combined_ok =', combined_ok)",
        ])))

    # 11. Package
    cells.append(_code_cell("\n".join([
        "# 11. Package whatever LoRAs exist",
        "import os, zipfile",
        "",
        "LORA_FILES = [",
        "    '/kaggle/working/faiz_lora.safetensors',",
        "    '/kaggle/working/juv_lora.safetensors',",
        "    '/kaggle/working/faiz_juv_lora.safetensors',",
        "]",
        "ZIP_PATH = '/kaggle/working/loras.zip'",
        "",
        "packed = []",
        "with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:",
        "    for path in LORA_FILES:",
        "        if os.path.exists(path):",
        "            zf.write(path, os.path.basename(path))",
        "            packed.append(os.path.basename(path))",
        "            print('added', os.path.basename(path), os.path.getsize(path), 'bytes')",
        "        else:",
        "            print('MISSING', path)",
        "",
        "print('zip size:', os.path.getsize(ZIP_PATH), 'bytes')",
        "print('packed:', packed)",
        "assert packed, 'no LoRAs were produced - check the training output above'",
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


def build_metadata(only: str = "all") -> dict:
    kernel_id = KERNEL_ID if only == "all" else f"{KERNEL_ID}-{only}"
    title = KERNEL_TITLE if only == "all" else f"{KERNEL_TITLE} {only}"
    return {
        "id": kernel_id,
        "title": title,
        "code_file": "train_loras_kaggle.ipynb",
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
    ap = argparse.ArgumentParser(description="Build/deploy the Kaggle LoRA training kernel")
    ap.add_argument("--push", action="store_true", help="Push the kernel to Kaggle")
    ap.add_argument("--output", action="store_true", help="Download the kernel output")
    ap.add_argument("--status", action="store_true", help="Check the kernel status")
    ap.add_argument(
        "--only",
        choices=["all", "faiz", "juv", "combined"],
        default="all",
        help="Train one LoRA in its own kernel (recommended) or all in one run",
    )
    ap.add_argument(
        "--output-dir",
        default=None,
        help="Where --output writes the kernel artifacts (default: lora_output/<target>)",
    )
    args = ap.parse_args()

    notebook = build_notebook(args.only)
    nb_path = DEPLOY_DIR / "train_loras_kaggle.ipynb"
    nb_path.write_text(json.dumps(notebook, indent=1), encoding="utf-8")
    print(f"Notebook saved: {nb_path}")

    metadata = build_metadata(args.only)
    meta_path = DEPLOY_DIR / "kernel-metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Metadata saved: {meta_path} (kernel {metadata['id']})")

    if args.push:
        print("Pushing to Kaggle...")
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
        subprocess.run(["kaggle", "kernels", "status", metadata["id"]])

    if args.output:
        # /tmp does not exist on Windows, and a shared directory would let the
        # faiz and juv runs overwrite each other's artifacts.
        out_dir = Path(args.output_dir) if args.output_dir else Path("lora_output") / args.only
        out_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["kaggle", "kernels", "output", metadata["id"], "-p", str(out_dir)]
        )
        print(f"Artifacts written to {out_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
