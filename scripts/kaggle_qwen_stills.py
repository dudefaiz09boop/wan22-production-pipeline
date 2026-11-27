#!/usr/bin/env python3
"""Build and deploy a Kaggle kernel that renders startframes with Qwen-Image 2.1.

Why Qwen and not SDXL
---------------------
SDXL encodes text with CLIP at 77 tokens and has no native image reference, so
identity had to come from a LoRA plus an InsightFace paste. That produced
fragments, swapped characters and soft images.

Qwen-Image 2.1 encodes with an 8B Qwen2.5-VL vision model and takes reference
images directly, so the prompt can say "use <image1> for his body, <image2> for
his exact face" and the model reads the face from the photo. This is the
approach the attached ComfyUI template uses.

Hardware
--------
T4 x2 = 32 GB VRAM. The GGUF quants fit:
    qwen_image-Q4_K_M.gguf                    12.2 GB
    Qwen2.5-VL-7B-Instruct-Q4_NL.gguf          4.1 GB
    qwen_image_vae.safetensors                 0.2 GB
Weights are staged in /kaggle/temp, which is not counted against the 20 GB
working quota.

Usage
-----
    python scripts/kaggle_qwen_stills.py --push
    python scripts/kaggle_qwen_stills.py --status
    python scripts/kaggle_qwen_stills.py --output
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
DEPLOY_DIR = HERE / "deploy" / "qwen_stills"
DEPLOY_DIR.mkdir(parents=True, exist_ok=True)

KERNEL_ID = "sofiya009/wan22-qwen-stills"
KERNEL_TITLE = "Wan22 Qwen Stills"

REPO_URL = "https://github.com/dudefaiz09boop/wan22-production-pipeline.git"

# Beats to render. Empty means all 70.
BEATS: list[str] = ["c01", "c02", "c10", "c11", "c15", "c31", "c37"]

DATASET_SOURCES = [
    "sofiya009/lora-train-faiz",
    "sofiya009/lora-train-juveriya",
]


def _code_cell(text: str) -> dict:
    lines = text.split("\n")
    source = [line + "\n" for line in lines[:-1]] + [lines[-1]]
    return {
        "cell_type": "code",
        "id": hashlib.sha1(text.encode("utf-8")).hexdigest()[:8],
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source,
    }


def build_notebook() -> dict:
    cells: list[dict] = []

    cells.append(_code_cell("\n".join([
        "# 1. Environment",
        "import os, shutil, subprocess, sys",
        "import torch",
        "print('torch', torch.__version__)",
        "print('gpus', torch.cuda.device_count())",
        "for i in range(torch.cuda.device_count()):",
        "    print(' gpu', i, torch.cuda.get_device_name(i),",
        "          round(torch.cuda.get_device_properties(i).total_memory/1024**3,1), 'GB')",
        "print('working free GB', round(shutil.disk_usage('/kaggle/working').free/1024**3,1))",
        "print('temp free GB', round(shutil.disk_usage('/kaggle/temp').free/1024**3,1))",
    ])))

    cells.append(_code_cell("\n".join([
        "# 2. Which beats to render",
        f"BEATS = {BEATS!r}",
        "print('beats:', BEATS or 'all')",
    ])))

    cells.append(_code_cell("\n".join([
        "# 3. Clone the pipeline",
        "import os, subprocess",
        "REPO = '/kaggle/working/repo'",
        "if not os.path.isdir(REPO):",
        f"    subprocess.run(['git','clone','--depth','1','{REPO_URL}',REPO], check=True)",
        "os.chdir(REPO)",
        "print(subprocess.run(['git','log','--oneline','-1'],capture_output=True,text=True).stdout)",
    ])))

    cells.append(_code_cell("\n".join([
        "# 4. ComfyUI + GGUF node",
        "# Weights go to /kaggle/temp so they do not count against the 20 GB",
        "# working quota; /kaggle/working only holds the repo and the outputs.",
        "import os, subprocess",
        "COMFY = '/kaggle/temp/ComfyUI'",
        "if not os.path.isdir(COMFY):",
        "    subprocess.run(['git','clone','--depth','1',",
        "        'https://github.com/comfyanonymous/ComfyUI.git', COMFY], check=True)",
        "    subprocess.run([sys.executable,'-m','pip','install','-q','-r',",
        "        f'{COMFY}/requirements.txt'], check=True)",
        "    subprocess.run([sys.executable,'-m','pip','install','-q',",
        "        'gguf','sentencepiece','protobuf'], check=True)",
        "    subprocess.run(['git','clone','--depth','1',",
        "        'https://github.com/city96/ComfyUI-GGUF.git',",
        "        f'{COMFY}/custom_nodes/ComfyUI-GGUF'], check=True)",
        "    subprocess.run([sys.executable,'-m','pip','install','-q','-r',",
        "        f'{COMFY}/custom_nodes/ComfyUI-GGUF/requirements.txt'], check=True)",
        "print('ComfyUI ready at', COMFY)",
    ])))

    cells.append(_code_cell("\n".join([
        "# 5. Download the Qwen-Image 2.1 GGUF weights",
        "import os",
        "from huggingface_hub import hf_hub_download",
        "",
        "COMFY = '/kaggle/temp/ComfyUI'",
        "DIRS = {",
        "    'diffusion_models': f'{COMFY}/models/diffusion_models',",
        "    'text_encoders': f'{COMFY}/models/text_encoders',",
        "    'vae': f'{COMFY}/models/vae',",
        "}",
        "for d in DIRS.values():",
        "    os.makedirs(d, exist_ok=True)",
        "",
        "WANT = [",
        "    ('city96/Qwen-Image-gguf', 'qwen-image-Q4_K_M.gguf', 'diffusion_models'),",
        "    ('unsloth/Qwen2.5-VL-7B-Instruct-GGUF',",
        "     'Qwen2.5-VL-7B-Instruct-Q4_NL.gguf', 'text_encoders'),",
        "    ('Comfy-Org/Qwen-Image_ComfyUI',",
        "     'split_files/vae/qwen_image_vae.safetensors', 'vae'),",
        "]",
        "for repo, filename, kind in WANT:",
        "    dest = DIRS[kind]",
        "    target = os.path.join(dest, os.path.basename(filename))",
        "    if os.path.isfile(target) and os.path.getsize(target) > 1_000_000:",
        "        print('cached', target); continue",
        "    path = hf_hub_download(repo_id=repo, filename=filename, local_dir=dest)",
        "    if path != target:",
        "        os.replace(path, target)",
        "    print('downloaded', target, round(os.path.getsize(target)/1024**3,1), 'GB')",
    ])))

    cells.append(_code_cell("\n".join([
        "# 6. Start ComfyUI",
        "import os, subprocess, time, urllib.request",
        "COMFY = '/kaggle/temp/ComfyUI'",
        "log = open('/kaggle/temp/comfy.log','w')",
        "proc = subprocess.Popen([sys.executable,'main.py','--listen','127.0.0.1',",
        "    '--port','8188','--highvram'], cwd=COMFY, stdout=log, stderr=subprocess.STDOUT)",
        "for i in range(120):",
        "    try:",
        "        urllib.request.urlopen('http://127.0.0.1:8188/system_stats', timeout=2)",
        "        print('ComfyUI up after', i*2, 's'); break",
        "    except Exception:",
        "        time.sleep(2)",
        "else:",
        "    print(open('/kaggle/temp/comfy.log').read()[-3000:])",
        "    raise SystemExit('ComfyUI did not start')",
    ])))

    cells.append(_code_cell("\n".join([
        "# 7. Render the startframes with Qwen-Image 2.1",
        "import subprocess, sys",
        "cmd = [sys.executable, 'scripts/generate_stills_qwen.py',",
        "       '--output', 'startframes']",
        "if BEATS:",
        "    cmd += ['--beats', *BEATS]",
        "print('RUN:', ' '.join(cmd), flush=True)",
        "rc = subprocess.call(cmd, stdout=sys.stdout, stderr=sys.stderr)",
        "print('rc:', rc, flush=True)",
        "import glob",
        "made = sorted(glob.glob('startframes/c*.png'))",
        "print('startframes:', len(made))",
        "assert made, 'no startframes produced'",
    ])))

    cells.append(_code_cell("\n".join([
        "# 8. Package only the stills",
        "import os, shutil, zipfile",
        "dest = '/kaggle/working/startframes'",
        "if os.path.isdir('startframes'):",
        "    shutil.copytree('startframes', dest, dirs_exist_ok=True)",
        "ZIP = '/kaggle/working/stills.zip'",
        "with zipfile.ZipFile(ZIP,'w',zipfile.ZIP_DEFLATED) as zf:",
        "    for name in sorted(os.listdir(dest)):",
        "        if name.endswith('.png'):",
        "            zf.write(os.path.join(dest,name), name)",
        "print('zip', os.path.getsize(ZIP), 'bytes')",
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
        "code_file": "qwen_stills.ipynb",
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
    ap = argparse.ArgumentParser(description="Build/deploy the Qwen stills kernel")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--output", action="store_true")
    ap.add_argument("--output-dir", default="kaggle_out/qwen_stills")
    args = ap.parse_args()

    nb_path = DEPLOY_DIR / "qwen_stills.ipynb"
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
