#!/bin/bash
# AIC Cloud - Ultra-Fast WAN 2.2 Production Pipeline Setup
# Optimized for L40S 48GB / RTX 3090 24GB / A100 instances
# Run on fresh AIC Cloud GPU instance after SSH

set -euo pipefail

# ──────────────────────────────────────────────
# CONFIGURATION - EDIT THESE BEFORE RUNNING
# ──────────────────────────────────────────────

# Required tokens (set via environment or edit here)
HF_TOKEN="${HF_TOKEN:-}"           # Hugging Face token (required for model downloads)
KAGGLE_KEY="${KAGGLE_KEY:-}"       # Kaggle API key (required for trained LoRAs)
WANDB_KEY="${WANDB_KEY:-}"         # Optional: Weights & Biases for logging

# Pipeline config
REPO_URL="https://github.com/YOUR_USERNAME/wan22-production-pipeline.git"
BRANCH="main"
WORKSPACE="/workspace"
REPO_DIR="$WORKSPACE/wan22-production-pipeline"
MODEL_CACHE="$WORKSPACE/models"
OUTPUT_DIR="$WORKSPACE/output"

# Performance tuning
PARALLEL_DOWNLOADS=4
COMFYUI_PORT=8188
SKIP_VALIDATION=false

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
[[ -z "$KAGGLE_KEY" ]] && MISSING_TOKENS+=("KAGGLE_KEY")

if [[ ${#MISSING_TOKENS[@]} -gt 0 ]]; then
    error "Missing required tokens: ${MISSING_TOKENS[*]}"
    echo "Set them via: export HF_TOKEN=... KAGGLE_KEY=..."
    exit 1
fi
success "All tokens present"

# Check GPU
if ! command -v nvidia-smi &> /dev/null; then
    error "nvidia-smi not found - not a GPU instance?"
    exit 1
fi

VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
VRAM_GB=$((VRAM_MB / 1024))
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
success "GPU: $GPU_NAME (${VRAM_GB} GB VRAM)"

# Determine precision
if [[ $VRAM_GB -ge 48 ]]; then
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

cd "$WORKSPACE"

# Clone repo (shallow for speed)
if [[ ! -d "$REPO_DIR/.git" ]]; then
    log "Cloning repository..."
    git clone --depth 1 --branch "$BRANCH" "$REPO_URL" "$REPO_DIR" 2>/dev/null
else
    log "Updating repository..."
    cd "$REPO_DIR" && git pull --ff-only 2>/dev/null
fi

cd "$REPO_DIR"

# Install Python dependencies with uv (parallel, cached)
log "Installing Python dependencies..."
uv sync --extra test --no-cache 2>&1 | tail -5

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

# ──────────────────────────────────────────────
# PHASE 4: TRAINED LORAS (KAGGLE)
# ──────────────────────────────────────────────
phase "PHASE 4: TRAINED LORAS FROM KAGGLE"

log "Downloading trained LoRAs from sofiya009/lora-nuke-faiz-juv..."
uv run python scripts/download_loras.py \
    --dest "$MODEL_CACHE/loras/trained" \
    --kaggle-username sofiya009 \
    --kaggle-dataset lora-nuke-faiz-juv \
    --token "$KAGGLE_KEY" \
    --update-registry \
    --registry-path workflows/loras.json

success "Trained LoRAs downloaded and registry updated"

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

# Start ComfyUI in background with optimal flags
cd "$WORKSPACE/ComfyUI"
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
# PHASE 7: GENERATE STILLS (PARALLEL WITH RENDER PREP)
# ──────────────────────────────────────────────
phase "PHASE 7: GENERATE STILLS (SDXL + InsightFace)"

log "Generating start frames with trained LoRAs..."
uv run python scripts/generate_stills.py \
    --prompts prompts \
    --output startframes \
    --references references \
    --loras "$MODEL_CACHE/loras/trained" \
    --sdxl-model "$MODEL_CACHE/Juggernaut-XL-v9/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors" \
    --negative negatives/artifact_neg.txt \
    --width 832 --height 480 \
    --steps 30 --guidance 7.0 \
    --device cuda --dtype fp16 \
    2>&1 | tail -20

success "Start frames generated"

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

log "Running face swap on rendered clips..."
uv run python scripts/face_swap_video.py \
    --input renders \
    --output renders/face_swapped \
    --references references \
    --swapper-model "$MODEL_CACHE/insightface/inswapper_128.onnx" \
    --swap-every 3 \
    --threshold 0.40 \
    2>&1 | tail -20

success "Face swap complete"

# ──────────────────────────────────────────────
# PHASE 10: ASSEMBLE FINAL DELIVERABLE
# ──────────────────────────────────────────────
phase "PHASE 10: ASSEMBLE FINAL DELIVERABLE"

log "Concatenating clips and creating deliverable..."
uv run python scripts/assemble_final.py \
    --spec docs/beats.yaml \
    --renders renders/face_swapped \
    --output output \
    --crossfade 0.5 \
    --crf 18 \
    --preset slow \
    --zip \
    --include-frames \
    2>&1 | tail -10

success "Final deliverable created"

# ──────────────────────────────────────────────
# COMPLETE - SUMMARY & DOWNLOAD INFO
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

# Show download command
INSTANCE_IP=$(hostname -I | awk '{print $1}')
echo -e "${CYAN}DOWNLOAD FROM YOUR LOCAL MACHINE:${NC}"
echo "  scp root@$INSTANCE_IP:$OUTPUT_DIR/deliverable.zip ."
echo "  scp root@$INSTANCE_IP:$OUTPUT_DIR/final.mp4 ."
echo ""
echo -e "${CYAN}OR USE AIC CLOUD DASHBOARD:${NC}"
echo "  Files tab → Navigate to $OUTPUT_DIR → Download"

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
read -t 30 -p "Shutdown instance now? (y/N) " -n 1 -r || true
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    log "Shutting down..."
    shutdown -h now
fi