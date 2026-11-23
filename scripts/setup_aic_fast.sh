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

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    error "nvidia-smi not found - not a GPU instance?"
    exit 1
fi

VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
VRAM_GB=$((VRAM_MB / 1024))
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
success "GPU: $GPU_NAME (${VRAM_GB} GB VRAM)"

# Determine precision (WAN_PRECISION=auto picks from VRAM)
if [[ "$WAN_PRECISION" != "auto" ]]; then
    PRECISION="$WAN_PRECISION"
    case "$PRECISION" in
        fp16) MODEL_SIZE_GB=57 ;;
        fp8)  MODEL_SIZE_GB=29 ;;
        *)    MODEL_SIZE_GB=15 ;;
    esac
elif [[ $VRAM_GB -ge 48 ]]; then
    PRECISION="fp16"
    MODEL_SIZE_GB=57
elif [[ $VRAM_GB -ge 24 ]]; then
    PRECISION="fp8"
    MODEL_SIZE_GB=29
else
    PRECISION="gguf_q8"
    MODEL_SIZE_GB=15
fi
success "Selected precision: $PRECISION (est. ${MODEL_SIZE_GB} GB models)"

# Check disk space
AVAIL_GB=$(df -BG "$WORKSPACE" | tail -1 | awk '{print $4}' | sed 's/G//')
REQUIRED_GB=$((MODEL_SIZE_GB + 20))  # models + workspace + outputs
if [[ $AVAIL_GB -lt $REQUIRED_GB ]]; then
    warn "Low disk: ${AVAIL_GB}GB available, need ~${REQUIRED_GB}GB"
    warn "Consider adding more disk in AIC Cloud dashboard"
fi

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
uv sync --extra test 2>&1 | tail -5

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
DOWNLOADED=$(find "$MODEL_CACHE" -name "*.safetensors" -o -name "*.gguf" | wc -l)
EXPECTED=$(echo $HF_FILES | wc -w)
success "Downloaded $DOWNLOADED/$EXPECTED model files"

# Lipsync models: Wan 2.2 S2V diffusion model plus the wav2vec2 audio encoder.
# Only fetched when lipsync is enabled.
if [[ "${LIPSYNC_ENGINE:-s2v}" != "off" ]]; then
    log "Downloading lipsync models (S2V + audio encoder)..."
    uv run python - "$MODEL_CACHE" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download

dest = Path(sys.argv[1])
files = [
    ("split_files/diffusion_models/wan2.2_s2v_14B_fp8_scaled.safetensors", "diffusion_models"),
    ("split_files/audio_encoders/wav2vec2_large_english_fp16.safetensors", "audio_encoders"),
]
for repo_path, subdir in files:
    target = dest / subdir
    target.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id="Comfy-Org/Wan-2.2",
        filename=repo_path,
        local_dir=str(target),
    )
    print("lipsync model:", path)
PY
    success "Lipsync models ready"
fi

# ──────────────────────────────────────────────
# PHASE 4: LORAS (ANATOMY + TRAINED CHARACTERS)
# ──────────────────────────────────────────────
phase "PHASE 4: LORAS"

# 4a. Anatomy LoRAs. The beats declare penis_insert and vag_detail on
# specific shots, so these must exist before ComfyUI runs.
log "Checking anatomy LoRAs (Civitai)..."
if ! uv run python scripts/download_anatomy_loras.py --dest "$MODEL_CACHE/loras" --required-only; then
    error "Anatomy LoRAs are missing and could not be downloaded."
    echo "Set CIVITAI_API_KEY in .env, or stage the files manually in $MODEL_CACHE/loras."
    echo "Required by docs/beats.yaml: penis_insert and vag_detail (high + low)."
    exit 1
fi
success "Anatomy LoRAs present"

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

# ──────────────────────────────────────────────
# PHASE 6: START COMFYUI (BACKGROUND)
# ──────────────────────────────────────────────
phase "PHASE 6: START COMFYUI"

log "Starting ComfyUI on port $COMFYUI_PORT..."

# Check if ComfyUI exists, if not clone minimal
if [[ ! -d "$WORKSPACE/ComfyUI" ]]; then
    log "Cloning ComfyUI (minimal)..."
    git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$WORKSPACE/ComfyUI" 2>/dev/null
    cd "$WORKSPACE/ComfyUI"
    pip install -q -r requirements.txt 2>/dev/null | tail -3
    
    # Install required custom nodes
    cd custom_nodes
    git clone --depth 1 https://github.com/kijai/ComfyUI-GGUF.git 2>/dev/null &
    git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git 2>/dev/null &
    wait
fi

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

# Lipsync assets live in subdirectories of the cache.
for f in "$MODEL_CACHE"/audio_encoders/*.safetensors; do
    [[ -e "$f" ]] || continue
    link_into "$f" "$COMFY_DIR/models/audio_encoders/$(basename "$f")"
done
for f in "$MODEL_CACHE"/diffusion_models/*.safetensors; do
    [[ -e "$f" ]] || continue
    link_into "$f" "$COMFY_DIR/models/diffusion_models/$(basename "$f")"
done

LORA_COUNT=$(find "$COMFY_DIR/models/loras" -name "*.safetensors" | wc -l)
success "ComfyUI model paths linked ($LORA_COUNT LoRAs visible)"
if [[ "$LORA_COUNT" -eq 0 ]]; then
    error "No LoRAs visible to ComfyUI - the render would fail on the first LoRA shot."
    exit 1
fi

# Start ComfyUI in background with optimal flags
cd "$COMFY_DIR"
nohup python main.py \
    --listen 0.0.0.0 \
    --port $COMFYUI_PORT \
    --lowvram \
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

# ──────────────────────────────────────────────
# PHASE 7: GENERATE STILLS (SDXL + InsightFace)
# ──────────────────────────────────────────────
phase "PHASE 7: GENERATE STILLS (SDXL + InsightFace)"

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
if [[ ! -f references/faiz_ref.jpg || ! -f references/juv_ref.jpg ]]; then
    warn "references/faiz_ref.jpg and/or references/juv_ref.jpg missing."
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

FRAME_COUNT=$(find startframes -name "c*.png" 2>/dev/null | wc -l)
if [[ "$FRAME_COUNT" -eq 0 ]]; then
    error "No start frames were generated - the render cannot proceed."
    exit 1
fi
success "Start frames generated ($FRAME_COUNT)"

# Audit the frames before spending GPU time on the render. Problems here
# become disfigured video, so surface them now.
log "Auditing start frame quality..."
if ! uv run python scripts/audit_startframes.py --dir startframes --spec docs/beats.yaml; then
    warn "Some start frames have quality problems (see startframes/quality_report.json)."
    if [[ "${REPAIR_FRAMES:-true}" == "true" ]]; then
        log "Regenerating failing frames with more candidates..."
        uv run python scripts/repair_startframes.py --candidates 4 --gate 0.45 \
            2>&1 | tail -12 || warn "Frame repair incomplete; continuing with current frames"
    fi
fi

# ──────────────────────────────────────────────
# PHASE 8: RENDER CLIPS (MAIN PRODUCTION)
# ──────────────────────────────────────────────
phase "PHASE 8: RENDER CLIPS (WAN 2.2 I2V)"

log "Starting full production render..."
START_RENDER=$(date +%s)

uv run python scripts/run_production.py \
    --spec docs/beats.yaml \
    --prompts prompts \
    --startframes startframes \
    --renders renders \
    --output output \
    --server "http://127.0.0.1:$COMFYUI_PORT" \
    --force-precision "$PRECISION" \
    --hf-token "$HF_TOKEN" \
    --skip-loras \
    --skip-stills \
    --skip-face-swap \
    --skip-assemble \
    2>&1 | tee /tmp/render.log

RENDER_EXIT=${PIPESTATUS[0]}
END_RENDER=$(date +%s)
RENDER_DURATION=$((END_RENDER - START_RENDER))

if [[ $RENDER_EXIT -eq 0 ]]; then
    success "Render complete in ${RENDER_DURATION}s ($((RENDER_DURATION/60)) min)"
else
    error "Render failed. Check /tmp/render.log"
    tail -100 /tmp/render.log
    exit 1
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
        LIPSYNC_COUNT=$(find renders/lipsync -name "c*.mp4" 2>/dev/null | wc -l)
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
# PHASE 10: NARRATION + AUDIO MUX + FINAL CUT
# ──────────────────────────────────────────────
phase "PHASE 10: NARRATION, AUDIO MUX, FINAL CUT"

# 10a. Per-clip narration with Edge TTS. The compiled prompts already carry
# the authored narration text, so this only synthesises and fits duration.
log "Generating narration audio (Edge TTS)..."
if uv run python scripts/generate_narration.py --prompts prompts --output audio; then
    NARRATION_COUNT=$(find audio -name "c*.wav" 2>/dev/null | wc -l)
    success "Narration generated ($NARRATION_COUNT track(s))"
else
    warn "Narration generation failed - clips will get silent audio tracks"
fi

# 10b. Mux narration into each clip, then concatenate into one MP4 with a
# single continuous audio stream. This is what makes the deliverable a real
# video with sound.
log "Muxing audio and building the final cut..."
MUSIC_ARGS=()
if [[ -n "${MUSIC_BED:-}" && -f "${MUSIC_BED:-}" ]]; then
    MUSIC_ARGS=(--music "$MUSIC_BED" --music-db "${MUSIC_DB:--22}")
fi

if ! uv run python scripts/mux_audio.py \
    --spec docs/beats.yaml \
    --renders "$RENDER_SOURCE" \
    --audio audio \
    --muxed renders/muxed \
    --output output/final.mp4 \
    --crf 18 --preset slow \
    "${MUSIC_ARGS[@]}" \
    2>&1 | tail -12; then
    warn "Audio mux failed - falling back to silent assembly"
    uv run python scripts/assemble_final.py \
        --spec docs/beats.yaml \
        --renders "$RENDER_SOURCE" \
        --output output \
        --crf 18 --preset slow \
        2>&1 | tail -6
fi

# 10c. Package the deliverable.
log "Packaging deliverable..."
uv run python scripts/assemble_final.py \
    --spec docs/beats.yaml \
    --renders "$RENDER_SOURCE" \
    --output output \
    --crf 18 \
    --preset slow \
    --zip \
    --include-frames \
    2>&1 | tail -6

if [[ ! -f output/final.mp4 ]]; then
    error "Final video was not produced. Check the assemble output above."
    exit 1
fi
success "Final deliverable created"

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