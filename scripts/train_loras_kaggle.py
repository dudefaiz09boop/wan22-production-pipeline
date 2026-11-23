#!/usr/bin/env python3
"""Build and deploy Kaggle LoRA training kernel.

Creates a Kaggle notebook that trains 3 LoRAs (faiz, juv, both) using Kohya-ss/sd-scripts
on free Kaggle T4 GPU. Uses startframes faces + existing datasets.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).parent
DEPLOY_DIR = HERE / "deploy" / "training"
DEPLOY_DIR.mkdir(parents=True, exist_ok=True)


def build_notebook() -> dict:
    """Build the Kaggle training notebook."""
    cells = []

    # Cell 1: Environment check
    cells.append(("md", "# LoRA Training: Faiz + Juveriya (Kaggle Free T4)\n\nTrains 3 LoRAs for SDXL:\n- `faiz_lora.safetensors` - Faiz character\n- `juv_lora.safetensors` - Juveriya character  \n- `faiz_juv_lora.safetensors` - Combined both\n\nUses Kohya-ss/sd-scripts with mixed precision fp16."))

    cells.append(("md", "## 1. Environment & GPU Check"))
    cells.append(("code", "\n".join([
        "!nvidia-smi",
        "!df -h /kaggle/working",
        "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')",
        "print(f'GPU: {torch.cuda.get_device_name(0)}')",
    ])))

    # Cell 2: Install dependencies
    cells.append(("md", "## 2. Install Dependencies"))
    cells.append(("code", "\n".join([
        "!pip install -q --upgrade pip",
        "!pip install -q 'numpy<2' 'accelerate>=0.30' 'transformers>=4.41' 'diffusers>=0.30' 'safetensors>=0.4' 'peft>=0.11' 'bitsandbytes>=0.43' 'xformers>=0.0.27' 'opencv-python-headless>=4.10' 'insightface>=0.7' 'tqdm>=4.66' 'omegaconf>=2.3' 'lycoris-lora>=1.8' 'kornia>=0.7'",
        "!apt -y -qq install git ffmpeg libgl1 libglib2.0-0 2>/dev/null | tail -3",
        "",
        "# Install Kohya-ss/sd-scripts",
        "!cd /kaggle/working && git clone --depth 1 https://github.com/kohya-ss/sd-scripts.git 2>/dev/null || cd sd-scripts && git pull",
        "!cd /kaggle/working/sd-scripts && pip install -q -e . 2>&1 | tail -5",
        "",
        "import sys; sys.path.insert(0, '/kaggle/working/sd-scripts')",
        "print('sd-scripts installed')",
    ])))

    # Cell 3: Unpack datasets
    cells.append(("md", "## 3. Unpack Training Datasets"))
    cells.append(("code", "\n".join([
        "import glob, os, zipfile, shutil",
        "",
        "# Mounted datasets (attached in Kaggle kernel settings)",
        "DATASETS = {",
        "    'startframes': '/kaggle/input/startframes-c01-c70',",
        "    'faiz': '/kaggle/input/lora-train-faiz',",
        "    'juv': '/kaggle/input/lora-train-juveriya',",
        "    'base_model': '/kaggle/input/wan22-nsfw-lightning-q8',",
        "}",
        "",
        "WORK_DIR = '/kaggle/working/training'",
        "os.makedirs(WORK_DIR, exist_ok=True)",
        "",
        "for name, path in DATASETS.items():",
        "    print(f'{name}: {path}')",
        "    if os.path.exists(path):",
        "        files = os.listdir(path)",
        "        print(f'  Files: {files[:10]}...' if len(files) > 10 else f'  Files: {files}')",
        "    else:",
        "        print(f'  NOT MOUNTED - attach in kernel settings!')",
        "",
        "# Copy to working directory for training",
        "for name in ['faiz', 'juv', 'combined']:",
        "    src = f'/kaggle/input/lora-train-{name}' if name != 'combined' else '/kaggle/input/lora-train-faiz'",
        "    dst = f'{WORK_DIR}/{name}'",
        "    if os.path.exists(src):",
        "        shutil.copytree(src, dst, dirs_exist_ok=True)",
        "        print(f'Copied {name} to {dst}')",
        "    else:",
        "        print(f'WARNING: {src} not found')",
    ])))

    # Cell 4: Verify training data
    cells.append(("md", "## 4. Verify Training Data"))
    cells.append(("code", "\n".join([
        "WORK_DIR = '/kaggle/working/training'",
        "for name in ['faiz', 'juv', 'combined']:",
        "    path = f'{WORK_DIR}/{name}'",
        "    if os.path.exists(path):",
        "        images = [f for f in os.listdir(path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]",
        "        captions = [f for f in os.listdir(path) if f.endswith('.txt')]",
        "        print(f'{name}: {len(images)} images, {len(captions)} captions')",
        "        if images:",
        "            print(f'  Example: {images[0]}')",
        "    else:",
        "        print(f'{name}: NOT FOUND')",
    ])))

    # Cell 5: Training config
    cells.append(("md", "## 5. Training Configuration"))
    cells.append(("code", "\n".join([
        "# Training hyperparameters (optimized for T4 16GB)",
        "CONFIG = {",
        "    # Model",
        "    'pretrained_model_name_or_path': '/kaggle/input/wan22-nsfw-lightning-q8/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors',",
        "    'v2': False,  # SDXL",
        "",
        "    # LoRA",
        "    'network_module': 'networks.lora',",
        "    'network_dim': 32,  # Rank 32 (faster, sufficient quality)",
        "    'network_alpha': 32,",
        "    'network_args': ['train_norm=true', 'train_text_encoder=true'],",
        "",
        "    # Training",
        "    'learning_rate': 1e-4,",
        "    'lr_scheduler': 'cosine_with_restarts',",
        "    'lr_warmup_steps': 100,",
        "    'max_train_steps': 1500,",
        "    'save_every_n_steps': 500,",
        "    'save_last_n_steps': 3,",
        "    'train_batch_size': 1,",
        "    'gradient_accumulation_steps': 4,  # Effective batch 4",
        "    'mixed_precision': 'fp16',",
        "    'gradient_checkpointing': True,",
        "    'optimizer_type': 'AdamW8bit',  # 8-bit optimizer saves VRAM",
        "    'optimizer_args': ['weight_decay=0.01'],",
        "",
        "    # Data",
        "    'resolution': '1024,1024',",
        "    'cache_latents': True,",
        "    'cache_latents_to_disk': True,",
        "    'enable_bucket': True,",
        "    'min_bucket_reso': 256,",
        "    'max_bucket_reso': 1024,",
        "",
        "    # Output",
        "    'output_dir': '/kaggle/working/output',",
        "    'output_name': 'lora',",
        "    'save_precision': 'fp16',",
        "    'save_model_as': 'safetensors',",
        "",
        "    # Logging",
        "    'logging_dir': '/kaggle/working/logs',",
        "    'log_with': 'tensorboard',",
        "    'log_prefix': 'lora_training',",
        "}",
        "",
        "import json",
        "print(json.dumps(CONFIG, indent=2))",
    ])))

    # Cell 6: Train Faiz LoRA
    cells.append(("md", "## 6. Train Faiz LoRA"))
    cells.append(("code", "\n".join([
        "import subprocess, os",
        "",
        "WORK_DIR = '/kaggle/working/training'",
        "OUTPUT_DIR = '/kaggle/working/output/faiz'",
        "os.makedirs(OUTPUT_DIR, exist_ok=True)",
        "",
        "cmd = [",
        "    'accelerate', 'launch', '--num_cpu_threads_per_process', '2',",
        "    '/kaggle/working/sd-scripts/sdxl_train_network.py',",
        "    '--pretrained_model_name_or_path', CONFIG['pretrained_model_name_or_path'],",
        "    '--v2', str(CONFIG['v2']).lower(),",
        "    '--network_module', CONFIG['network_module'],",
        "    '--network_dim', str(CONFIG['network_dim']),",
        "    '--network_alpha', str(CONFIG['network_alpha']),",
        "    '--network_args', *CONFIG['network_args'],",
        "    '--learning_rate', str(CONFIG['learning_rate']),",
        "    '--lr_scheduler', CONFIG['lr_scheduler'],",
        "    '--lr_warmup_steps', str(CONFIG['lr_warmup_steps']),",
        "    '--max_train_steps', str(CONFIG['max_train_steps']),",
        "    '--save_every_n_steps', str(CONFIG['save_every_n_steps']),",
        "    '--save_last_n_steps', str(CONFIG['save_last_n_steps']),",
        "    '--train_batch_size', str(CONFIG['train_batch_size']),",
        "    '--gradient_accumulation_steps', str(CONFIG['gradient_accumulation_steps']),",
        "    '--mixed_precision', CONFIG['mixed_precision'],",
        "    '--gradient_checkpointing', str(CONFIG['gradient_checkpointing']).lower(),",
        "    '--optimizer_type', CONFIG['optimizer_type'],",
        "    '--optimizer_args', *CONFIG['optimizer_args'],",
        "    '--resolution', CONFIG['resolution'],",
        "    '--cache_latents', str(CONFIG['cache_latents']).lower(),",
        "    '--cache_latents_to_disk', str(CONFIG['cache_latents_to_disk']).lower(),",
        "    '--enable_bucket', str(CONFIG['enable_bucket']).lower(),",
        "    '--min_bucket_reso', str(CONFIG['min_bucket_reso']),",
        "    '--max_bucket_reso', str(CONFIG['max_bucket_reso']),",
        "    '--output_dir', OUTPUT_DIR,",
        "    '--output_name', 'faiz_lora',",
        "    '--save_precision', CONFIG['save_precision'],",
        "    '--save_model_as', CONFIG['save_model_as'],",
        "    '--logging_dir', CONFIG['logging_dir'],",
        "    '--log_with', CONFIG['log_with'],",
        "    '--log_prefix', 'faiz_lora',",
        "    '--train_data_dir', f'{WORK_DIR}/faiz',",
        "    '--reg_data_dir', f'{WORK_DIR}/juv',  # Regularization with other char",
        "]",
        "",
        "print('Training Faiz LoRA...')",
        "print(' '.join(cmd[:10]) + ' ...')",
        "result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)",
        "print('STDOUT:', result.stdout[-5000:] if result.stdout else 'None')",
        "print('STDERR:', result.stderr[-5000:] if result.stderr else 'None')",
        "print('Return code:', result.returncode)",
        "",
        "# Find saved LoRA",
        "import glob",
        "faiz_loras = sorted(glob.glob(f'{OUTPUT_DIR}/faiz_lora-*.safetensors'))",
        "if faiz_loras:",
        "    print(f'Faiz LoRA saved: {faiz_loras[-1]}')",
        "    # Copy to final location",
        "    shutil.copy2(faiz_loras[-1], '/kaggle/working/faiz_lora.safetensors')",
        "else:",
        "    print('ERROR: No Faiz LoRA found!')",
    ])))

    # Cell 7: Train Juv LoRA
    cells.append(("md", "## 7. Train Juveriya LoRA"))
    cells.append(("code", "\n".join([
        "OUTPUT_DIR = '/kaggle/working/output/juv'",
        "os.makedirs(OUTPUT_DIR, exist_ok=True)",
        "",
        "cmd = [",
        "    'accelerate', 'launch', '--num_cpu_threads_per_process', '2',",
        "    '/kaggle/working/sd-scripts/sdxl_train_network.py',",
        "    '--pretrained_model_name_or_path', CONFIG['pretrained_model_name_or_path'],",
        "    '--v2', str(CONFIG['v2']).lower(),",
        "    '--network_module', CONFIG['network_module'],",
        "    '--network_dim', str(CONFIG['network_dim']),",
        "    '--network_alpha', str(CONFIG['network_alpha']),",
        "    '--network_args', *CONFIG['network_args'],",
        "    '--learning_rate', str(CONFIG['learning_rate']),",
        "    '--lr_scheduler', CONFIG['lr_scheduler'],",
        "    '--lr_warmup_steps', str(CONFIG['lr_warmup_steps']),",
        "    '--max_train_steps', str(CONFIG['max_train_steps']),",
        "    '--save_every_n_steps', str(CONFIG['save_every_n_steps']),",
        "    '--save_last_n_steps', str(CONFIG['save_last_n_steps']),",
        "    '--train_batch_size', str(CONFIG['train_batch_size']),",
        "    '--gradient_accumulation_steps', str(CONFIG['gradient_accumulation_steps']),",
        "    '--mixed_precision', CONFIG['mixed_precision'],",
        "    '--gradient_checkpointing', str(CONFIG['gradient_checkpointing']).lower(),",
        "    '--optimizer_type', CONFIG['optimizer_type'],",
        "    '--optimizer_args', *CONFIG['optimizer_args'],",
        "    '--resolution', CONFIG['resolution'],",
        "    '--cache_latents', str(CONFIG['cache_latents']).lower(),",
        "    '--cache_latents_to_disk', str(CONFIG['cache_latents_to_disk']).lower(),",
        "    '--enable_bucket', str(CONFIG['enable_bucket']).lower(),",
        "    '--min_bucket_reso', str(CONFIG['min_bucket_reso']),",
        "    '--max_bucket_reso', str(CONFIG['max_bucket_reso']),",
        "    '--output_dir', OUTPUT_DIR,",
        "    '--output_name', 'juv_lora',",
        "    '--save_precision', CONFIG['save_precision'],",
        "    '--save_model_as', CONFIG['save_model_as'],",
        "    '--logging_dir', CONFIG['logging_dir'],",
        "    '--log_with', CONFIG['log_with'],",
        "    '--log_prefix', 'juv_lora',",
        "    '--train_data_dir', f'{WORK_DIR}/juv',",
        "    '--reg_data_dir', f'{WORK_DIR}/faiz',  # Regularization with other char",
        "]",
        "",
        "print('Training Juveriya LoRA...')",
        "result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)",
        "print('STDOUT:', result.stdout[-5000:] if result.stdout else 'None')",
        "print('STDERR:', result.stderr[-5000:] if result.stderr else 'None')",
        "print('Return code:', result.returncode)",
        "",
        "juv_loras = sorted(glob.glob(f'{OUTPUT_DIR}/juv_lora-*.safetensors'))",
        "if juv_loras:",
        "    print(f'Juv LoRA saved: {juv_loras[-1]}')",
        "    shutil.copy2(juv_loras[-1], '/kaggle/working/juv_lora.safetensors')",
        "else:",
        "    print('ERROR: No Juv LoRA found!')",
    ])))

    # Cell 8: Train Combined LoRA
    cells.append(("md", "## 8. Train Combined LoRA (Faiz + Juveriya)"))
    cells.append(("code", "\n".join([
        "OUTPUT_DIR = '/kaggle/working/output/combined'",
        "os.makedirs(OUTPUT_DIR, exist_ok=True)",
        "",
        "cmd = [",
        "    'accelerate', 'launch', '--num_cpu_threads_per_process', '2',",
        "    '/kaggle/working/sd-scripts/sdxl_train_network.py',",
        "    '--pretrained_model_name_or_path', CONFIG['pretrained_model_name_or_path'],",
        "    '--v2', str(CONFIG['v2']).lower(),",
        "    '--network_module', CONFIG['network_module'],",
        "    '--network_dim', str(CONFIG['network_dim']),",
        "    '--network_alpha', str(CONFIG['network_alpha']),",
        "    '--network_args', *CONFIG['network_args'],",
        "    '--learning_rate', str(CONFIG['learning_rate']),",
        "    '--lr_scheduler', CONFIG['lr_scheduler'],",
        "    '--lr_warmup_steps', str(CONFIG['lr_warmup_steps']),",
        "    '--max_train_steps', str(CONFIG['max_train_steps']),",
        "    '--save_every_n_steps', str(CONFIG['save_every_n_steps']),",
        "    '--save_last_n_steps', str(CONFIG['save_last_n_steps']),",
        "    '--train_batch_size', str(CONFIG['train_batch_size']),",
        "    '--gradient_accumulation_steps', str(CONFIG['gradient_accumulation_steps']),",
        "    '--mixed_precision', CONFIG['mixed_precision'],",
        "    '--gradient_checkpointing', str(CONFIG['gradient_checkpointing']).lower(),",
        "    '--optimizer_type', CONFIG['optimizer_type'],",
        "    '--optimizer_args', *CONFIG['optimizer_args'],",
        "    '--resolution', CONFIG['resolution'],",
        "    '--cache_latents', str(CONFIG['cache_latents']).lower(),",
        "    '--cache_latents_to_disk', str(CONFIG['cache_latents_to_disk']).lower(),",
        "    '--enable_bucket', str(CONFIG['enable_bucket']).lower(),",
        "    '--min_bucket_reso', str(CONFIG['min_bucket_reso']),",
        "    '--max_bucket_reso', str(CONFIG['max_bucket_reso']),",
        "    '--output_dir', OUTPUT_DIR,",
        "    '--output_name', 'faiz_juv_lora',",
        "    '--save_precision', CONFIG['save_precision'],",
        "    '--save_model_as', CONFIG['save_model_as'],",
        "    '--logging_dir', CONFIG['logging_dir'],",
        "    '--log_with', CONFIG['log_with'],",
        "    '--log_prefix', 'faiz_juv_lora',",
        "    '--train_data_dir', f'{WORK_DIR}/combined',",
        "]",
        "",
        "print('Training Combined LoRA...')",
        "result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)",
        "print('STDOUT:', result.stdout[-5000:] if result.stdout else 'None')",
        "print('STDERR:', result.stderr[-5000:] if result.stderr else 'None')",
        "print('Return code:', result.returncode)",
        "",
        "combined_loras = sorted(glob.glob(f'{OUTPUT_DIR}/faiz_juv_lora-*.safetensors'))",
        "if combined_loras:",
        "    print(f'Combined LoRA saved: {combined_loras[-1]}')",
        "    shutil.copy2(combined_loras[-1], '/kaggle/working/faiz_juv_lora.safetensors')",
        "else:",
        "    print('ERROR: No Combined LoRA found!')",
    ])))

    # Cell 9: Package and verify
    cells.append(("md", "## 9. Package LoRAs & Verify"))
    cells.append(("code", "\n".join([
        "import zipfile, os",
        "",
        "LORA_FILES = [",
        "    '/kaggle/working/faiz_lora.safetensors',",
        "    '/kaggle/working/juv_lora.safetensors',",
        "    '/kaggle/working/faiz_juv_lora.safetensors',",
        "]",
        "",
        "ZIP_PATH = '/kaggle/working/loras.zip'",
        "",
        "print('Packaging LoRAs...')",
        "with zipfile.ZipFile(ZIP_PATH, 'w', zipfile.ZIP_DEFLATED) as zf:",
        "    for lora_path in LORA_FILES:",
        "        if os.path.exists(lora_path):",
        "            size_mb = os.path.getsize(lora_path) / (1024*1024)",
        "            zf.write(lora_path, os.path.basename(lora_path))",
        "            print(f'  Added: {os.path.basename(lora_path)} ({size_mb:.1f} MB)')",
        "        else:",
        "            print(f'  MISSING: {lora_path}')",
        "",
        "zip_size = os.path.getsize(ZIP_PATH) / (1024*1024)",
        "print(f'\\nloras.zip created: {zip_size:.1f} MB')",
        "",
        "# Verify contents",
        "with zipfile.ZipFile(ZIP_PATH, 'r') as zf:",
        "    for info in zf.infolist():",
        "        print(f'  {info.filename}: {info.file_size / (1024*1024):.1f} MB')",
    ])))

    # Cell 10: Upload as dataset
    cells.append(("md", "## 10. Upload as Kaggle Dataset (Run Locally)"))
    cells.append(("code", "\n".join([
        "# After kernel completes, run LOCALLY to download and upload as dataset:",
        "",
        "# 1. Download loras.zip from Kaggle output",
        "# kaggle kernels output sofiya009/lora-nuke-faiz-juv -p /tmp/lora_output",
        "",
        "# 2. Create dataset metadata",
        "# cat > /tmp/lora_output/dataset-metadata.json << 'EOF'",
        "# {",
        "#   \"id\": \"sofiya009/lora-nuke-faiz-juv\",",
        "#   \"title\": \"LoRA Nuke: Faiz + Juveriya (SDXL)\",",
        "#   \"subtitle\": \"Character LoRAs for Wan 2.2 pipeline\",",
        "#   \"description\": \"3 LoRAs: faiz, juv, faiz_juv combined. Trained on SDXL with Kohya-ss.\",",
        "#   \"isPrivate\": true,",
        "#   \"keywords\": [\"lora\", \"sdxl\", \"character\", \"wan22\"],",
        "#   \"licenses\": [{\"name\": \"CC0-1.0\"}]",
        "# }",
        "# EOF",
        "",
        "# 3. Upload new version",
        "# kaggle datasets version -p /tmp/lora_output -m \"Updated LoRAs with startframes faces\"",
        "",
        "print('Run the above commands locally after kernel completes.')",
        "print('Then the LoRAs will be available for fzjv-pipeline.')",
    ])))

    # Convert to notebook format
    nb_cells = []
    for kind, text in cells:
        lines = text.split("\n")
        src = [l + "\n" for l in lines[:-1]] + [lines[-1]]
        cell = {"cell_type": kind, "metadata": {}, "source": src}
        if kind == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
        nb_cells.append(cell)

    return {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_metadata() -> dict:
    """Build kernel-metadata.json."""
    return {
        "id": "sofiya009/lora-nuke-faiz-juv",
        "title": "LoRA Nuke: Faiz + Juveriya (SDXL)",
        "code_file": "train_loras_kaggle.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [
            "sofiya009/startframes-c01-c70",
            "sofiya009/lora-train-faiz",
            "sofiya009/lora-train-juveriya",
            "sofiya009/wan22-nsfw-lightning-q8",
        ],
        "competition_sources": [],
        "kernel_sources": [],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build/deploy Kaggle LoRA training kernel")
    ap.add_argument("--push", action="store_true", help="Push kernel to Kaggle")
    ap.add_argument("--output", action="store_true", help="Download output after run")
    ap.add_argument("--status", action="store_true", help="Check kernel status")
    args = ap.parse_args()

    # Build notebook
    print("Building training notebook...")
    notebook = build_notebook()
    nb_path = DEPLOY_DIR / "train_loras_kaggle.ipynb"
    nb_path.write_text(json.dumps(notebook, indent=1))
    print(f"Notebook saved: {nb_path}")

    # Build metadata
    metadata = build_metadata()
    meta_path = DEPLOY_DIR / "kernel-metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"Metadata saved: {meta_path}")

    if args.push:
        print("Pushing to Kaggle...")
        result = subprocess.run(
            ["kaggle", "kernels", "push", "-p", str(DEPLOY_DIR)],
            capture_output=True, text=True
        )
        print(result.stdout)
        if result.stderr:
            print(result.stderr)
        if result.returncode != 0:
            return 1

    if args.status:
        print("Checking kernel status...")
        subprocess.run(["kaggle", "kernels", "status", "sofiya009/lora-nuke-faiz-juv"])

    if args.output:
        print("Downloading output...")
        subprocess.run(["kaggle", "kernels", "output", "sofiya009/lora-nuke-faiz-juv", "-p", "/tmp/lora_output"])

    return 0


if __name__ == "__main__":
    sys.exit(main())