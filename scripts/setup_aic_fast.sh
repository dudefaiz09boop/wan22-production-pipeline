#!/bin/bash
# AIC Cloud - Ultra-Fast WAN 2.2 Production Pipeline Setup
# Optimized for L40S 48GB / RTX 3090 24GB / A100 instances
#
# One-click: run on a fresh AIC Cloud GPU instance after SSH.
#   bash setup_aic_fast.sh
#
# Configuration is read from .env in the repo root (see .env.example).
# Any value already exported in the environment wins over .env.

set -euo pipefail

# ──────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR_DEFAULT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Load .env if present (does not override already-exported values)
if [[ -f "$REPO_DIR_DEFAULT/.env" ]]; then
    while IFS='=' read -r key value; do
        [[ -z "$key" || "$key" == \#* ]] && continue
        key="$(echo "$key" | tr -d '[:space:]')"
        [[ -z "${!key:-}" ]] && export "$key=$value"
    done < "$REPO_DIR_DEFAULT/.env"
fi

HF_TOKEN="${HF_TOKEN:-}"           # Hugging Face token (required for model downloads)
KAGGLE_KEY="${KAGGLE_KEY:-}"       # Kaggle API key (optional; LoRAs may already be published)
CIVITAI_API_KEY="${CIVITAI_API_KEY:-}"

# Pipeline config
REPO_URL="${REPO_URL:-}"
BRANCH="${BRANCH:-main}"
WORKSPACE="${WORKSPACE:-/workspace}"
REPO_DIR="${REPO_DIR:-$WORKSPACE/wan22-production-pipeline}"
MODEL_CACHE="${MODEL_DIR:-$WORKSPACE/models}"
OUTPUT_DIR="${OUTPUT_DIR:-$WORKSPACE/output}"

# Performance tuning
PARALLEL_DOWNLOADS="${PARALLEL_DOWNLOADS:-4}"
COMFYUI_PORT="${COMFY_PORT:-8188}"
SKIP_VALIDATION="${SKIP_VALIDATION:-false}"
WAN_PRECISION="${WAN_PRECISION:-auto}"

# ──────────────────────────────────────────────
# COLORS & LOGGING
# ──────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log() { echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $*"; }
success() { echo -e "${GREEN}[$(date '+%H:%M:%S')] ✓${NC} $*"; }
warn() { echo -e "${YELLOW}[$(date '+%H:%M:%S')] ⚠${NC} $*"; }
error() { echo -e "${RED}[$(date '+%H:%M:%S')] ✗${NC} $*"; }
phase() { echo -e "\n${CYAN}═══ $* ═══${NC}\n"; }

# ──────────────────────────────────────────────
# PRE-FLIGHT CHECKS
# ──────────────────────────────────────────────
phase "PRE-FLIGHT CHECKS"

# Check root
if [[ $EUID -ne 0 ]]; then
    error "Must run as root (use sudo)"
    exit 1
fi

# Check tokens
MISSING_TOKENS=()
[[ -z "$HF_TOKEN" ]] && MISSING_TOKENS+=("HF_TOKEN")

if [[ ${#MISSING_TOKENS[@]} -gt 0 ]]; then
    error "Missing required tokens: ${MISSING_TOKENS[*]}"
    echo "Set them in .env (see .env.example) or export them before running."
    exit 1
fi
success "HF token present"
[[ -n "$KAGGLE_KEY" ]] && success "Kaggle key present" || warn "No KAGGLE_KEY - trained LoRAs must already be published"

# Check required tools. AIC's minimized Ubuntu image omits several of these.
MISSING_TOOLS=()
for tool in jq bc curl aria2c ffmpeg git; do
    command -v "$tool" &> /dev/null || MISSING_TOOLS+=("$tool")
done
if [[ ${#MISSING_TOOLS[@]} -gt 0 ]]; then
    error "Missing tools: ${MISSING_TOOLS[*]}"
    echo "Install with: apt-get update && apt-get install -y ${MISSING_TOOLS[*]}"
    exit 1
fi
success "All required tools present"

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    error "nvidia-smi not found - not a GPU instance?"
    exit 1
fi

VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
VRAM_GB=$((VRAM_MB / 1024))
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
success "GPU: $GPU_NAME (${VRAM_GB} GB VRAM)"

# Determine precision (WAN_PRECISION=auto picks from VRAM).
# Thresholds match auto_model_selector.py: FP16 needs 44 GB because one
# 26.62 GB UNET is resident at a time plus the 6.27 GB text encoder, VAE and
# activations. An RTX A6000 reports 46068 MiB, which is 44 GB after integer
# division, so a 48 GB threshold would wrongly fall back to FP8.
if [[ "$WAN_PRECISION" != "auto" ]]; then
    PRECISION="$WAN_PRECISION"
    case "$PRECISION" in
        fp16) MODEL_SIZE_GB=57 ;;
        fp8)  MODEL_SIZE_GB=29 ;;
        *)    MODEL_SIZE_GB=15 ;;
    esac
elif [[ $VRAM_GB -ge 44 ]]; then
    PRECISION="fp16"
    MODEL_SIZE_GB=57
elif [[ $VRAM_GB -ge 22 ]]; then
    PRECISION="fp8"
    MODEL_SIZE_GB=29
else
    PRECISION="gguf_q8"
    MODEL_SIZE_GB=15
fi
success "Selected precision: $PRECISION (est. ${MODEL_SIZE_GB} GB models)"

# Check disk space. The gate must compare *remaining* work against free
# space, not total need against free space: on a resumed run the models are
# already staged, so charging for them again aborts a run that would fit.
AVAIL_GB=$(df -BG "$WORKSPACE" | tail -1 | awk '{print $4}' | sed 's/G//')
STAGED_GB=$(du -sBG "$MODEL_CACHE" 2>/dev/null | awk '{print $1}' | sed 's/G//')
STAGED_GB=${STAGED_GB:-0}
# Models still to fetch, plus renders, lipsync, mux and the final cut.
REMAINING_GB=$(( MODEL_SIZE_GB - STAGED_GB + 25 ))
if [[ $REMAINING_GB -lt 8 ]]; then REMAINING_GB=8; fi
if [[ $AVAIL_GB -lt $REMAINING_GB ]]; then
    error "Low disk: ${AVAIL_GB}GB available, need ~${REMAINING_GB}GB more"
    error "Staged models: ${STAGED_GB}GB. Free space or add disk, then re-run."
    exit 1
fi
log "Disk: ${AVAIL_GB}GB free, ${STAGED_GB}GB models staged, ~${REMAINING_GB}GB still needed"

# ──────────────────────────────────────────────
# PHASE 1: SYSTEM SETUP (PARALLEL)
# ──────────────────────────────────────────────
phase "PHASE 1: SYSTEM SETUP (parallel)"

# Start apt update in background
apt-get update -qq &
APT_PID=$!

# Create directories in parallel
mkdir -p "$WORKSPACE" "$MODEL_CACHE" "$OUTPUT_DIR" "$REPO_DIR" &
MKDIR_PID=$!

# Install uv (fast Python package manager) in background
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh &
    UV_PID=$!
else
    UV_PID=0
fi

# Wait for apt update
wait $APT_PID

# Install system packages (minimal, fast)
log "Installing system packages..."
apt-get install -y -qq \
    git aria2 ffmpeg libgl1 libglib2.0-0 \
    python3 python3-venv python3-pip \
    jq bc curl ca-certificates \
    2>/dev/null | tail -3

# Wait for mkdir
wait $MKDIR_PID

# Wait for uv if installing
if [[ $UV_PID -ne 0 ]]; then
    wait $UV_PID
    export PATH="$HOME/.local/bin:$PATH"
fi

success "System setup complete"

# ──────────────────────────────────────────────
# PHASE 2: CLONE REPO & INSTALL PYTHON DEPS
# ──────────────────────────────────────────────
phase "PHASE 2: REPO & PYTHON DEPS"

# If this script is already running from inside a checkout, use it as-is.
if [[ -d "$REPO_DIR_DEFAULT/.git" ]]; then
    REPO_DIR="$REPO_DIR_DEFAULT"
    log "Using existing checkout: $REPO_DIR"
else
    cd "$WORKSPACE"
    if [[ ! -d "$REPO_DIR/.git" ]]; then
        if [[ -z "$REPO_URL" ]]; then
            error "No checkout found and REPO_URL is unset."
            echo "Either run this script from inside the repo, or set REPO_URL in .env."
            exit 1
        fi
        log "Cloning repository..."
        git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$REPO_DIR"
    else
        log "Updating repository..."
        cd "$REPO_DIR" && git pull --ff-only 2>/dev/null || true
    fi
fi

cd "$REPO_DIR"

# Install Python dependencies with uv (parallel, cached)
log "Installing Python dependencies..."
# --inexact: do not prune packages that are not in the lockfile. Chatterbox is
# installed below with `uv pip install --no-deps` because its pins conflict
# with the rest of the stack, so it is deliberately absent from uv.lock. An
# exact sync would delete it on every re-run and force a reinstall.
uv sync --extra test --inexact 2>&1 | tail -5

# The venv is now correct, so stop `uv run` from re-syncing on every call.
# Without this, each `uv run` re-resolves the environment and can re-download
# multi-GB wheels (torch is 2.4 GiB) before the command even starts.
export UV_NO_SYNC=1

success "Python environment ready"

# ──────────────────────────────────────────────
# PHASE 3: PARALLEL MODEL DOWNLOADS
# ──────────────────────────────────────────────
phase "PHASE 3: PARALLEL MODEL DOWNLOADS"

# Export tokens for Python scripts
export HF_TOKEN="$HF_TOKEN"
export KAGGLE_KEY="$KAGGLE_KEY"
export HF_HUB_DISABLE_SYMLINKS_WARNING=1
export HF_HOME="$MODEL_CACHE"

# Run model selector to confirm precision
log "Verifying model selection..."
uv run python scripts/auto_model_selector.py --force "$PRECISION" --output json > /tmp/model_config.json

# Parse model config
HF_REPO=$(jq -r '.hf_repo' /tmp/model_config.json)
MODEL_DIR=$(jq -r '.model_dir' /tmp/model_config.json)
HF_FILES=$(jq -r '.hf_files | join(" ")' /tmp/model_config.json)

log "HF Repo: $HF_REPO"
log "Model Dir: $MODEL_DIR"
log "Files: $HF_FILES"

# Download models in PARALLEL using aria2 (fastest)
log "Starting parallel model downloads with aria2..."

# Create aria2 input file
cat > /tmp/aria2_input.txt <<EOF
# Models from $HF_REPO
$(for file in $HF_FILES; do
    echo "https://huggingface.co/$HF_REPO/resolve/main/$file"
    echo "  dir=$MODEL_CACHE"
    echo "  out=$(basename $file)"
    echo "  header=Authorization: Bearer $HF_TOKEN"
done)
EOF

# Download with aria2 (16 connections per file, 4 parallel)
aria2c \
    --input-file=/tmp/aria2_input.txt \
    --max-concurrent-downloads=$PARALLEL_DOWNLOADS \
    --max-connection-per-server=16 \
    --split=16 \
    --min-split-size=1M \
    --continue=true \
    --auto-file-renaming=false \
    --conditional-get=true \
    --allow-overwrite=true \
    --quiet=true \
    --show-console-readout=false \
    --summary-interval=10 \
    2>&1 | tail -10

# Verify downloads
DOWNLOADED=$(find "$MODEL_CACHE" \( -name "*.safetensors" -o -name "*.gguf" \) 2>/dev/null | wc -l || true)
EXPECTED=$(echo $HF_FILES | wc -w)
success "Downloaded $DOWNLOADED/$EXPECTED model files"

# ────────────────────────────────────────────────────────────
# Civitai GGUF checkpoints
# ────────────────────────────────────────────────────────────
# The workflows load the Civitai 2053259 pair, which the Comfy-Org repo does
# not publish. Without this the first clip fails with a missing model.
if [[ "$PRECISION" == "gguf_q8" ]]; then
    log "Downloading Civitai GGUF checkpoints (fm + cf)..."
    if ! uv run python scripts/download_wan22_gguf.py --dest "$MODEL_CACHE"; then
        warn "Civitai GGUF download incomplete - renders will fail until staged"
        warn "Set CIVITAI_API_KEY in .env, or run: python scripts/download_wan22_gguf.py --dest $MODEL_CACHE"
    fi
fi

# Lipsync models. Only fetched when lipsync is enabled.
#   s2v          -> Wan 2.2 S2V diffusion model + wav2vec2 audio encoder
#   infinitetalk -> Wan 2.1 I2V base + InfiniteTalk patch + audio encoder
if [[ "${LIPSYNC_ENGINE:-s2v}" != "off" ]]; then
    log "Downloading lipsync models (engine: ${LIPSYNC_ENGINE:-s2v})..."
    uv run python - "$MODEL_CACHE" "${LIPSYNC_ENGINE:-s2v}" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

dest = Path(sys.argv[1])
engine = sys.argv[2]

jobs = []
if engine == "infinitetalk":
    jobs.append(("Comfy-Org/Wan-2.1",
                 "split_files/diffusion_models/wan2.1_i2v_480p_14B_fp8_scaled.safetensors",
                 "diffusion_models"))
    jobs.append(("Kijai/WanVideo_comfy_fp8_scaled",
                 "InfiniteTalk/Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors",
                 "model_patches"))
else:
    jobs.append(("Comfy-Org/Wan-2.2",
                 "split_files/diffusion_models/wan2.2_s2v_14B_fp8_scaled.safetensors",
                 "diffusion_models"))

jobs.append(("Comfy-Org/Wan-2.2",
             "split_files/audio_encoders/wav2vec2_large_english_fp16.safetensors",
             "audio_encoders"))

for repo, repo_path, subdir in jobs:
    target = dest / subdir
    target.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(repo_id=repo, filename=repo_path, local_dir=str(target))
    print("lipsync model:", path)
PY
    success "Lipsync models ready"
fi

# ──────────────────────────────────────────────
# PHASE 4: LORAS (ANATOMY + TRAINED CHARACTERS)
# ──────────────────────────────────────────────
phase "PHASE 4: LORAS"

# 4a. Anatomy + realism LoRAs. The beats declare penis_insert and vag_detail
# on specific shots, and the registry flags realism/skin_morph as defaults on
# every shot, so all of them must exist before ComfyUI runs.
log "Checking anatomy and realism LoRAs (Civitai)..."
if ! uv run python scripts/download_anatomy_loras.py --dest "$MODEL_CACHE/loras" --required-only; then
    error "Required LoRAs are missing and could not be downloaded."
    echo "Set CIVITAI_API_KEY in .env, or stage the files manually in $MODEL_CACHE/loras."
    echo "Required: penis_insert, vag_detail, realism, skin_morph (high + low)."
    exit 1
fi
success "Anatomy and realism LoRAs present"

# 4b. Trained character LoRAs. A locally staged copy wins over Kaggle.
log "Resolving trained character LoRAs..."
uv run python scripts/download_loras.py \
    --dest "$MODEL_CACHE/loras/trained" \
    --kaggle-username sofiya009 \
    --kaggle-dataset lora-nuke-faiz-juv \
    --token "$KAGGLE_KEY" \
    --update-registry \
    --registry-path workflows/loras.json || \
    warn "Trained LoRA resolution incomplete - stills will fall back to the base model"

success "LoRA staging complete"

# ──────────────────────────────────────────────
# PHASE 5: VALIDATE & GENERATE WORKFLOW
# ──────────────────────────────────────────────
phase "PHASE 5: VALIDATE SPEC & GENERATE WORKFLOW"

# Validate beats.yaml (fast, no frames)
if [[ "$SKIP_VALIDATION" != "true" ]]; then
    log "Validating beats.yaml..."
    uv run python scripts/validate.py --no-frames
    success "Spec validation passed"
fi

# Generate ComfyUI workflow for selected precision
log "Generating ComfyUI workflow ($PRECISION)..."
uv run python scripts/generate_workflow.py \
    --precision "$PRECISION" \
    --model-dir "$MODEL_DIR" \
    --output "workflows/base_i2v_${PRECISION}.json"

success "Workflow generated"

# Compile prompts/*.txt from docs/beats.yaml. --no-frames is required here:
# the start frames this step would otherwise validate are produced in phase 7.
log "Compiling beat prompts..."
uv run python scripts/build_prompts.py --no-frames
success "Prompts compiled"

# ──────────────────────────────────────────────
# PHASE 6: START COMFYUI (BACKGROUND)
# ──────────────────────────────────────────────
phase "PHASE 6: START COMFYUI"

log "Starting ComfyUI on port $COMFYUI_PORT..."

# Check if ComfyUI exists, if not clone minimal
if [[ ! -d "$WORKSPACE/ComfyUI" ]]; then
    log "Cloning ComfyUI (minimal)..."
    git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$WORKSPACE/ComfyUI" 2>/dev/null
fi

# ComfyUI gets its own virtual environment. Sharing the pipeline venv breaks
# in both directions: `uv sync` prunes ComfyUI's packages, and installing
# ComfyUI's requirements can replace the pinned CUDA 12.6 torch build.
COMFY_VENV="$WORKSPACE/ComfyUI/.venv"
if [[ ! -x "$COMFY_VENV/bin/python" ]]; then
    log "Creating isolated ComfyUI environment..."
    cd "$WORKSPACE/ComfyUI"
    uv venv --python 3.14 "$COMFY_VENV"
    uv pip install --python "$COMFY_VENV/bin/python" \
        --index-url https://download.pytorch.org/whl/cu126 \
        torch==2.14.1+cu126 torchvision==0.29.1+cu126
fi

cd "$WORKSPACE/ComfyUI"
uv pip install --python "$COMFY_VENV/bin/python" -r requirements.txt

if [[ ! -d "$WORKSPACE/ComfyUI/custom_nodes/ComfyUI-GGUF" ]]; then
    git clone --depth 1 https://github.com/kijai/ComfyUI-GGUF.git \
        "$WORKSPACE/ComfyUI/custom_nodes/ComfyUI-GGUF" 2>/dev/null &
fi
if [[ ! -d "$WORKSPACE/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite" ]]; then
    git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
        "$WORKSPACE/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite" 2>/dev/null &
fi
wait

# Custom-node dependencies are separate from ComfyUI's requirements. Without
# these, VHS_VideoCombine is absent and every render is rejected with HTTP 400.
for node_dir in "$WORKSPACE/ComfyUI/custom_nodes"/*; do
    if [[ -f "$node_dir/requirements.txt" ]]; then
        log "Installing dependencies for $(basename "$node_dir")..."
        uv pip install --python "$COMFY_VENV/bin/python" -r "$node_dir/requirements.txt"
    fi
done

# Point ComfyUI at the staged model cache. Without this, ComfyUI looks in its
# own models/ tree and every LoRA and checkpoint resolves as missing.
COMFY_DIR="$WORKSPACE/ComfyUI"
mkdir -p "$COMFY_DIR/models/loras" "$COMFY_DIR/models/diffusion_models" \
         "$COMFY_DIR/models/text_encoders" "$COMFY_DIR/models/vae" \
         "$COMFY_DIR/models/unet" "$COMFY_DIR/models/audio_encoders"

log "Linking staged models into ComfyUI..."
link_into() {
    local src="$1" dst="$2"
    [[ -e "$src" ]] || return 0
    if [[ -e "$dst" || -L "$dst" ]]; then
        rm -f "$dst"
    fi
    ln -s "$src" "$dst"
}

# LoRAs: anatomy (flat) and trained (subdirectory)
for f in "$MODEL_CACHE"/loras/*.safetensors; do
    [[ -e "$f" ]] || continue
    link_into "$f" "$COMFY_DIR/models/loras/$(basename "$f")"
done
for f in "$MODEL_CACHE"/loras/trained/*.safetensors; do
    [[ -e "$f" ]] || continue
    link_into "$f" "$COMFY_DIR/models/loras/$(basename "$f")"
done

# Checkpoints / encoders / VAE downloaded by the model phase
for f in "$MODEL_CACHE"/*.safetensors "$MODEL_CACHE"/*.gguf; do
    [[ -e "$f" ]] || continue
    base="$(basename "$f")"
    case "$base" in
        *umt5*|*clip*) link_into "$f" "$COMFY_DIR/models/text_encoders/$base" ;;
        *vae*)         link_into "$f" "$COMFY_DIR/models/vae/$base" ;;
        *)             link_into "$f" "$COMFY_DIR/models/diffusion_models/$base" ;;
    esac
done

# Lipsync assets live in subdirectories of the cache. hf_hub_download keeps
# the repository's split_files/... layout, so link recursively rather than
# only the top-level directory.
for f in "$MODEL_CACHE"/audio_encoders/*.safetensors "$MODEL_CACHE"/audio_encoders/*/*.safetensors "$MODEL_CACHE"/audio_encoders/*/*/*.safetensors "$MODEL_CACHE"/audio_encoders/*/*/*/*.safetensors; do
    [[ -e "$f" ]] || continue
    link_into "$f" "$COMFY_DIR/models/audio_encoders/$(basename "$f")"
done
for f in "$MODEL_CACHE"/diffusion_models/*.safetensors "$MODEL_CACHE"/diffusion_models/*/*.safetensors "$MODEL_CACHE"/diffusion_models/*/*/*.safetensors "$MODEL_CACHE"/diffusion_models/*/*/*/*.safetensors; do
    [[ -e "$f" ]] || continue
    link_into "$f" "$COMFY_DIR/models/diffusion_models/$(basename "$f")"
done
mkdir -p "$COMFY_DIR/models/model_patches"
for f in "$MODEL_CACHE"/model_patches/*.safetensors "$MODEL_CACHE"/model_patches/*/*.safetensors "$MODEL_CACHE"/model_patches/*/*/*.safetensors; do
    [[ -e "$f" ]] || continue
    link_into "$f" "$COMFY_DIR/models/model_patches/$(basename "$f")"
done

LORA_COUNT=$(find "$COMFY_DIR/models/loras" -name "*.safetensors" 2>/dev/null | wc -l || true)
success "ComfyUI model paths linked ($LORA_COUNT LoRAs visible)"
if [[ "$LORA_COUNT" -eq 0 ]]; then
    error "No LoRAs visible to ComfyUI - the render would fail on the first LoRA shot."
    exit 1
fi

# Start ComfyUI with VRAM-appropriate flags.
#
# --lowvram forces ComfyUI to offload model layers to CPU between steps. On a
# 45 GB card holding a 26.62 GB UNET that is pure waste: it turns a fast
# resident model into a per-step PCIe transfer and can triple render time.
# Only use it when the card genuinely cannot hold the model.
#
# --disable-smart-memory and --cache-none are also avoided: they force a full
# unload between jobs, which is right for a 16 GB T4 but wrong here.
COMFY_VRAM_ARGS=()
if [[ $VRAM_GB -lt 24 ]]; then
    COMFY_VRAM_ARGS=(--lowvram)
    log "VRAM ${VRAM_GB}GB < 24GB: using --lowvram"
else
    # ComfyUI has no --normalvram flag; the default is normal VRAM handling.
    # Passing an unknown flag aborts startup, so pass nothing here.
    log "VRAM ${VRAM_GB}GB: default VRAM handling (models stay resident)"
fi

cd "$COMFY_DIR"
nohup "$COMFY_VENV/bin/python" main.py \
    --listen 0.0.0.0 \
    --port $COMFYUI_PORT \
    ${COMFY_VRAM_ARGS[@]+"${COMFY_VRAM_ARGS[@]}"} \
    --preview-method none \
    --disable-auto-launch \
    > /tmp/comfyui.log 2>&1 &

COMFY_PID=$!
echo $COMFY_PID > /tmp/comfyui.pid

# Wait for ComfyUI to be ready (with timeout)
log "Waiting for ComfyUI to start..."
for i in {1..60}; do
    if curl -s "http://127.0.0.1:$COMFYUI_PORT/object_info" > /dev/null 2>&1; then
        success "ComfyUI ready!"
        break
    fi
    sleep 2
    if [[ $i -eq 60 ]]; then
        error "ComfyUI failed to start. Check /tmp/comfyui.log"
        tail -50 /tmp/comfyui.log
        exit 1
    fi
done

# A 200 response only proves the HTTP server is up. Verify the nodes and
# model files the workflow actually needs before spending GPU time.
log "Verifying ComfyUI nodes and staged models..."
curl -s "http://127.0.0.1:$COMFYUI_PORT/object_info" > /tmp/comfy_object_info.json
uv run python - /tmp/comfy_object_info.json "$PRECISION" <<'PY'
import json
import sys
from pathlib import Path

info = json.loads(Path(sys.argv[1]).read_text())
precision = sys.argv[2]
required = {
    "VHS_VideoCombine",
    "WanImageToVideo",
    "KSamplerAdvanced",
    "CLIPLoader",
    "VAELoader",
    "LoraLoaderModelOnly",
}
if precision == "gguf_q8":
    required.add("UnetLoaderGGUF")
else:
    required.add("UNETLoader")
missing = sorted(required - set(info))
if missing:
    raise SystemExit(f"ComfyUI is missing required nodes: {', '.join(missing)}")
print(f"ComfyUI nodes verified ({len(required)} required)")
PY

uv run python scripts/preflight.py --server "http://127.0.0.1:$COMFYUI_PORT" --precision "$PRECISION" --check-only

# ──────────────────────────────────────────────
# PHASE 7: GENERATE STILLS (SDXL + InsightFace)
# ──────────────────────────────────────────────
phase "PHASE 7: GENERATE STILLS (SDXL + InsightFace)"

cd "$REPO_DIR"

# Reuse frames that are already staged. Regenerating 70 SDXL stills costs
# 15-20 minutes and, when the trained LoRAs are unavailable, produces worse
# frames than the ones already on disk.
# `|| true` on every count: under `set -euo pipefail` a find on a missing
# directory returns 1 and would abort the whole script.
EXISTING_FRAMES=$(find startframes -name "c*.png" 2>/dev/null | wc -l || true)
EXPECTED_FRAMES=$(uv run python -c "
import yaml
print(len(yaml.safe_load(open('docs/beats.yaml'))['shots']))
" 2>/dev/null || echo 70)
if [[ "${SKIP_STILLS:-auto}" == "true" ]] || \
   { [[ "${SKIP_STILLS:-auto}" == "auto" ]] && [[ "$EXISTING_FRAMES" -ge "$EXPECTED_FRAMES" ]]; }; then
    success "Reusing $EXISTING_FRAMES staged start frames (skipping SDXL)"
    SKIP_STILLS_RUN=true
else
    SKIP_STILLS_RUN=false
fi

if [[ "$SKIP_STILLS_RUN" == "false" ]]; then

# The SDXL base model is only needed for stills, so fetch it here rather
# than in the Wan model phase.
SDXL_DIR="$MODEL_CACHE/Juggernaut-XL-v9"
SDXL_FILE="$SDXL_DIR/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"
if [[ ! -f "$SDXL_FILE" ]]; then
    log "Downloading SDXL base model (Juggernaut XL v9)..."
    mkdir -p "$SDXL_DIR"
    uv run python - "$SDXL_DIR" <<'PY'
import sys
from huggingface_hub import hf_hub_download
dest = sys.argv[1]
path = hf_hub_download(
    repo_id="RunDiffusion/Juggernaut-XL-v9",
    filename="Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors",
    local_dir=dest,
)
print("sdxl:", path)
PY
fi
[[ -f "$SDXL_FILE" ]] || { error "SDXL base model missing at $SDXL_FILE"; exit 1; }
success "SDXL base model ready"

# Character reference faces drive the InsightFace swap. Without them the
# stills are generated but identity is not enforced, so warn clearly.
# References may be png, jpg, jpeg, or webp; check every supported extension.
REF_FOUND=false
for ext in png jpg jpeg webp; do
    if [[ -f "references/faiz_ref.$ext" && -f "references/juv_ref.$ext" ]]; then
        REF_FOUND=true
        break
    fi
done
if [[ "$REF_FOUND" != "true" ]]; then
    warn "references/faiz_ref.* and/or references/juv_ref.* missing."
    warn "Stills will generate without identity enforcement."
    warn "Add both files to references/ for the face swap to work."
fi

log "Generating start frames with trained LoRAs..."
uv run python scripts/generate_stills.py \
    --prompts prompts \
    --output startframes \
    --references references \
    --loras "$MODEL_CACHE/loras/trained" \
    --sdxl-model "$SDXL_FILE" \
    --negative negatives/artifact_neg.txt \
    --spec docs/beats.yaml \
    --swapper-model "$MODEL_CACHE/insightface/inswapper_128.onnx" \
    --steps 30 --guidance 7.0 \
    --seed 1000 --gate 0.40 --candidates 2 \
    --device cuda --dtype fp16 \
    2>&1 | tail -20

FRAME_COUNT=$(find startframes -name "c*.png" 2>/dev/null | wc -l || true)
if [[ "$FRAME_COUNT" -eq 0 ]]; then
    error "No start frames were generated - the render cannot proceed."
    exit 1
fi
success "Start frames generated ($FRAME_COUNT)"

fi  # SKIP_STILLS_RUN

# The render needs every frame regardless of how it was produced.
FRAME_COUNT=$(find startframes -name "c*.png" 2>/dev/null | wc -l || true)
if [[ "$FRAME_COUNT" -eq 0 ]]; then
    error "No start frames available - the render cannot proceed."
    exit 1
fi
success "Start frames ready ($FRAME_COUNT)"

# Audit whatever is on disk, including reused frames. A blurry or faceless
# start frame becomes a disfigured clip, so surface it before the render.
log "Auditing start frame quality..."
if ! uv run python scripts/audit_startframes.py --dir startframes --spec docs/beats.yaml; then
    warn "Some start frames have quality problems (see startframes/quality_report.json)."
    if [[ "${REPAIR_FRAMES:-true}" == "true" && "$SKIP_STILLS_RUN" == "false" ]]; then
        log "Regenerating failing frames with more candidates..."
        uv run python scripts/repair_startframes.py --candidates 4 --gate 0.45 \
            2>&1 | tail -12 || warn "Frame repair incomplete; continuing with current frames"
    elif [[ "$SKIP_STILLS_RUN" == "true" ]]; then
        warn "Reused frames are kept as-is; set SKIP_STILLS=false to regenerate them."
    fi
fi

# ──────────────────────────────────────────────
# PHASE 8: RENDER CLIPS (MAIN PRODUCTION)
# ──────────────────────────────────────────────
phase "PHASE 8: RENDER CLIPS (WAN 2.2 I2V)"

log "Starting full production render..."
START_RENDER=$(date +%s)

# NAG makes the negative prompt bite at CFG 1.0. Off by default: it roughly
# doubles render time.
NAG_ARGS=()
if [[ "${NAG:-false}" == "true" ]]; then
    NAG_ARGS=(--nag --nag-scale "${NAG_SCALE:-11.0}" --nag-alpha "${NAG_ALPHA:-0.25}" --nag-tau "${NAG_TAU:-2.5}")
    log "NAG enabled (scale ${NAG_SCALE:-11.0})"
fi

uv run python scripts/run_production.py \
    --spec docs/beats.yaml \
    --prompts prompts \
    --startframes startframes \
    --renders renders \
    --output output \
    --server "http://127.0.0.1:$COMFYUI_PORT" \
    --force-precision "$PRECISION" \
    --hf-token "$HF_TOKEN" \
    --skip-download \
    --skip-loras \
    --skip-stills \
    --skip-face-swap \
    --skip-assemble \
    ${NAG_ARGS[@]+"${NAG_ARGS[@]}"} \
    2>&1 | tee /tmp/render.log

RENDER_EXIT=${PIPESTATUS[0]}
END_RENDER=$(date +%s)
RENDER_DURATION=$((END_RENDER - START_RENDER))

RENDERED=$(find renders -maxdepth 1 -name "c*.mp4" 2>/dev/null | wc -l || true)
if [[ "$RENDERED" -eq 0 ]]; then
    error "Render produced no clips. Check /tmp/render.log"
    tail -100 /tmp/render.log
    exit 1
fi
if [[ $RENDER_EXIT -ne 0 ]]; then
    warn "Render finished with failures: $RENDERED/$EXPECTED_FRAMES clip(s) on disk"
    warn "Continuing with the clips that rendered. See /tmp/render.log"
else
    success "Render complete in ${RENDER_DURATION}s ($((RENDER_DURATION/60)) min)"
fi

# ──────────────────────────────────────────────
# PHASE 9: FACE SWAP VIDEO
# ──────────────────────────────────────────────
phase "PHASE 9: FACE SWAP VIDEO (InsightFace)"

SWAPPER="$MODEL_CACHE/insightface/inswapper_128.onnx"
if [[ ! -f "$SWAPPER" ]]; then
    log "Downloading InsightFace inswapper model..."
    mkdir -p "$(dirname "$SWAPPER")"
    # deepinsight/inswapper is gated (403); these mirrors serve the same file.
    SWAPPER_URLS=(
        "https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/inswapper_128.onnx"
        "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
    )
    downloaded=false
    for url in "${SWAPPER_URLS[@]}"; do
        if curl -fL --retry 3 -H "Authorization: Bearer $HF_TOKEN" \
            -o "$SWAPPER.part" "$url"; then
            mv "$SWAPPER.part" "$SWAPPER"
            downloaded=true
            break
        fi
        rm -f "$SWAPPER.part"
    done
    if [[ "$downloaded" != true ]]; then
        warn "Could not download inswapper_128.onnx - skipping face swap"
    fi
fi

# Reference faces may be png or jpg; resolve whichever exists.
REF_OK=false
for ext in png jpg jpeg webp; do
    if [[ -f "references/faiz_ref.$ext" && -f "references/juv_ref.$ext" ]]; then
        REF_OK=true
        break
    fi
done

if [[ -f "$SWAPPER" && "$REF_OK" == true ]]; then
    log "Running face swap on rendered clips..."
    uv run python scripts/face_swap_video.py \
        --input renders \
        --output renders/face_swapped \
        --references references \
        --swapper-model "$SWAPPER" \
        --swap-every 3 \
        --threshold 0.40 \
        2>&1 | tail -20
    success "Face swap complete"
    RENDER_SOURCE="renders/face_swapped"
else
    warn "Face swap skipped (missing swapper model or reference faces)."
    warn "Using raw renders for the final cut."
    RENDER_SOURCE="renders"
fi

# ──────────────────────────────────────────────
# PHASE 9a: NARRATION (must precede lipsync)
# ──────────────────────────────────────────────
phase "PHASE 9a: NARRATION (Edge TTS)"

# Lipsync is audio-driven, so the narration has to exist before phase 9b.
# The compiled prompts already carry the authored narration text, so this
# only synthesises and fits each track to its clip duration.
log "Generating narration audio..."
# Chatterbox is the default: Hindi native, voice cloning, and an exaggeration
# knob that makes moans land. It needs a GPU and a ~2 GB model download, so
# fall back to Edge if it is unavailable rather than shipping silent clips.
NARRATION_ENGINE="${NARRATION_ENGINE:-chatterbox}"

# Chatterbox clones from a reference clip. Generate the three character
# references if they are not already staged, so a fresh clone needs no
# recording session.
if [[ "$NARRATION_ENGINE" == "chatterbox" ]]; then
    # chatterbox-tts pins numpy<1.26 and diffusers==0.29.0, which cannot
    # coexist with opencv on Python 3.14. Install it without its pins; the
    # versions it actually needs at runtime are already present.
    # This is intentionally outside uv.lock, which is why the sync above uses
    # --inexact: an exact sync would prune these packages on the next run.
    log "Installing Chatterbox (no-deps)..."
    uv pip install --python .venv/bin/python --no-deps chatterbox-tts==0.1.4 \
        resemble-perth conformer==0.3.2 s3tokenizer pykakasi==2.3.0 \
        pyloudnorm omegaconf spacy-pkuseg librosa==0.11.0 2>&1 | tail -3 || \
        warn "Chatterbox install failed - narration will fall back to Edge"
    log "Preparing voice references..."
    uv run python scripts/make_voice_refs.py --out refs/voices || \
        warn "Voice reference generation failed - Chatterbox will use its default voice"
fi

if ! uv run python scripts/generate_narration.py \
        --prompts prompts --output audio \
        --engine "$NARRATION_ENGINE" \
        --refs refs/voices \
        --device cuda; then
    warn "Narration failed with $NARRATION_ENGINE - retrying with Edge TTS"
    uv run python scripts/generate_narration.py \
        --prompts prompts --output audio --engine edge || \
        warn "Narration generation failed - lipsync will be skipped and clips get silent audio"
fi
NARRATION_COUNT=$(find audio -name "c*.wav" 2>/dev/null | wc -l || true)
if [[ "$NARRATION_COUNT" -gt 0 ]]; then
    success "Narration generated ($NARRATION_COUNT track(s))"
fi

# ──────────────────────────────────────────────
# PHASE 9b: LIPSYNC (audio-driven mouth)
# ──────────────────────────────────────────────
phase "PHASE 9b: LIPSYNC"

if [[ "${LIPSYNC_ENGINE:-s2v}" != "off" ]]; then
    log "Running lipsync pass (engine: ${LIPSYNC_ENGINE:-s2v})..."
    if uv run python scripts/lipsync.py \
        --spec docs/beats.yaml \
        --renders "$RENDER_SOURCE" \
        --startframes startframes \
        --audio audio \
        --output renders/lipsync \
        --server "http://127.0.0.1:$COMFYUI_PORT" \
        --engine "${LIPSYNC_ENGINE:-s2v}" \
        --steps "${LIPSYNC_STEPS:-6}" \
        2>&1 | tail -15; then
        LIPSYNC_COUNT=$(find renders/lipsync -name "c*.mp4" 2>/dev/null | wc -l || true)
        if [[ "$LIPSYNC_COUNT" -gt 0 ]]; then
            success "Lipsync complete ($LIPSYNC_COUNT clip(s))"
            RENDER_SOURCE="renders/lipsync"
        else
            warn "Lipsync produced no clips - keeping the pre-lipsync render"
        fi
    else
        warn "Lipsync failed - keeping the pre-lipsync render"
    fi
else
    log "Lipsync disabled (LIPSYNC_ENGINE=off)"
fi

# ──────────────────────────────────────────────
# PHASE 10: AUDIO MUX + FINAL CUT
# ──────────────────────────────────────────────
phase "PHASE 10: AUDIO MUX, FINAL CUT"

# 10b. Mux narration into each clip, then concatenate into one MP4 with a
# single continuous audio stream. This is what makes the deliverable a real
# video with sound.
log "Muxing audio and building the final cut..."
MUSIC_ARGS=()
if [[ -n "${MUSIC_BED:-}" && -f "${MUSIC_BED:-}" ]]; then
    MUSIC_ARGS=(--music "$MUSIC_BED" --music-db "${MUSIC_DB:--22}")
fi

GRADE_ARGS=()
if [[ "${GRADE:-true}" == "true" ]]; then
    GRADE_ARGS=(--grade)
fi

if ! uv run python scripts/mux_audio.py \
    --spec docs/beats.yaml \
    --renders "$RENDER_SOURCE" \
    --audio audio \
    --muxed renders/muxed \
    --output output/final.mp4 \
    --crf 18 --preset slow \
    --fps 16 \
    ${GRADE_ARGS[@]+"${GRADE_ARGS[@]}"} \
    ${MUSIC_ARGS[@]+"${MUSIC_ARGS[@]}"} \
    2>&1 | tail -12; then
    warn "Audio mux failed - falling back to silent assembly"
    uv run python scripts/assemble_final.py \
        --spec docs/beats.yaml \
        --renders "$RENDER_SOURCE" \
        --output output \
        --crf 18 --preset slow \
        2>&1 | tail -6
fi

# 10c. Package the deliverable. --keep-existing stops this from re-concatenating
# over output/final.mp4 and throwing away the muxed audio track.
log "Packaging deliverable..."
uv run python scripts/assemble_final.py \
    --spec docs/beats.yaml \
    --renders "$RENDER_SOURCE" \
    --output output \
    --crf 18 \
    --preset slow \
    --keep-existing \
    --zip \
    --include-frames \
    2>&1 | tail -6

if [[ ! -f output/final.mp4 ]]; then
    error "Final video was not produced. Check the assemble output above."
    exit 1
fi

# The deliverable must have sound. A silent final.mp4 means the mux fell back
# to the silent assembly path, which is a broken deliverable, not a warning.
if ! ffprobe -v error -select_streams a -show_entries stream=index \
        -of csv=p=0 output/final.mp4 | grep -q .; then
    error "final.mp4 has no audio stream - the mux did not run."
    exit 1
fi
success "Final deliverable created (with audio)"

# ──────────────────────────────────────────────
# PHASE 11: PUSH RESULTS TO LOCAL MACHINE
# ──────────────────────────────────────────────
phase "PHASE 11: SAVE RESULTS LOCALLY"

# AIC Cloud GPU instances are dashboard-only (no GPU API), so the instance
# cannot reach out to your laptop on its own. Instead it publishes the
# deliverable over HTTP and prints a one-line pull command.
#
# Set PULL_HOST in .env to your laptop's reachable address to have the
# instance push directly with scp instead.

DELIVERABLE="$OUTPUT_DIR/deliverable.zip"
FINAL_VIDEO="$OUTPUT_DIR/final.mp4"

if [[ -n "${PULL_HOST:-}" ]]; then
    log "Pushing results to $PULL_HOST ..."
    if scp -o StrictHostKeyChecking=accept-new "$DELIVERABLE" "$FINAL_VIDEO" "$PULL_HOST"; then
        success "Results copied to $PULL_HOST"
    else
        warn "scp failed - falling back to HTTP download"
        PULL_HOST=""
    fi
fi

if [[ -z "${PULL_HOST:-}" ]]; then
    log "Starting HTTP download server on port ${DOWNLOAD_PORT:-8000} ..."
    cd "$OUTPUT_DIR"
    nohup python3 -m http.server "${DOWNLOAD_PORT:-8000}" --bind 0.0.0.0 \
        > /tmp/download-server.log 2>&1 &
    echo $! > /tmp/download-server.pid
    sleep 2
    INSTANCE_IP=$(hostname -I | awk '{print $1}')
    success "Download server running (pid $(cat /tmp/download-server.pid))"
    echo ""
    echo -e "${CYAN}FROM YOUR LOCAL MACHINE, RUN:${NC}"
    echo "  curl -O http://$INSTANCE_IP:${DOWNLOAD_PORT:-8000}/deliverable.zip"
    echo "  curl -O http://$INSTANCE_IP:${DOWNLOAD_PORT:-8000}/final.mp4"
    echo ""
    echo -e "${YELLOW}The server stops when the instance stops. Download before shutting down.${NC}"
fi

# ──────────────────────────────────────────────
# COMPLETE - SUMMARY
# ──────────────────────────────────────────────
TOTAL_TIME=$(($(date +%s) - START_RENDER))
phase "PRODUCTION COMPLETE"

echo -e "${GREEN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  WAN 2.2 PRODUCTION COMPLETE                                 ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║  GPU: $GPU_NAME (${VRAM_GB}GB)"
echo "║  Precision: $PRECISION"
echo "║  Render time: ${RENDER_DURATION}s ($((RENDER_DURATION/60)) min)"
echo "║  Total time: ${TOTAL_TIME}s ($((TOTAL_TIME/60)) min)"
echo "║  Output: $OUTPUT_DIR/final.mp4"
echo "║  Deliverable: $OUTPUT_DIR/deliverable.zip"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

# Cost estimate
if [[ "$GPU_NAME" == *"L40S"* ]]; then
    HOURLY_RATE=84.60
elif [[ "$GPU_NAME" == *"3090"* ]]; then
    HOURLY_RATE=30.00
elif [[ "$GPU_NAME" == *"A100"* ]]; then
    HOURLY_RATE=150.00
else
    HOURLY_RATE=50.00
fi
EST_COST=$(echo "scale=2; $HOURLY_RATE * $TOTAL_TIME / 3600" | bc)
echo -e "\n${YELLOW}Estimated cost: ₹${EST_COST} (${TOTAL_TIME}s @ ₹${HOURLY_RATE}/hr)${NC}"

# Auto-shutdown option
if [[ "${AUTO_SHUTDOWN:-false}" == "true" ]]; then
    warn "AUTO_SHUTDOWN=true - stopping instance in 60s. Download now!"
    sleep 60
    shutdown -h now
else
    read -t 60 -p "Shutdown instance now? (y/N) " -n 1 -r || true
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        log "Shutting down..."
        shutdown -h now
    fi
fi