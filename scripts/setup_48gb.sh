#!/usr/bin/env bash
# Modern production pipeline for a 48 GB GPU.
#
# Replaces setup_aic_fast.sh. The differences that matter:
#
#   stills   Qwen-Image 2.1 with photo references, not SDXL + LoRA + InsightFace
#            paste. SDXL encodes text at 77 tokens and has no native image
#            reference, so identity had to be pasted on and the characters got
#            swapped. Qwen reads the face from the reference photo.
#
#   video    Wan 2.2 I2V 14B at FP16. A 48 GB card holds the 26.62 GB UNET plus
#            the text encoder and VAE with room for activations, so there is no
#            reason to run FP8 or GGUF.
#
#   models   Only what is used. The old script staged FP8 and GGUF variants that
#            a 48 GB card never loads, and two 1.17 GB realism LoRAs that fought
#            the trained identity LoRAs.
#
# Usage:
#   bash scripts/setup_48gb.sh              # full run
#   bash scripts/setup_48gb.sh --dry-run    # check only
#   bash scripts/setup_48gb.sh --skip-video # stills + audio only
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GREEN}▸${NC} $*"; }
warn() { echo -e "${YELLOW}!${NC} $*"; }
die()  { echo -e "${RED}✗${NC} $*" >&2; exit 1; }
phase(){ echo -e "\n${CYAN}═══ $* ═══${NC}\n"; }

DRY_RUN=false
SKIP_VIDEO=false
for arg in "$@"; do
  case "$arg" in
    --dry-run)    DRY_RUN=true ;;
    --skip-video) SKIP_VIDEO=true ;;
    *) die "unknown flag: $arg" ;;
  esac
done

# ── Configuration ────────────────────────────────────────────────────────────
COMFY_DIR="${COMFY_DIR:-$ROOT/ComfyUI}"
COMFY_PORT="${COMFY_PORT:-8188}"
MODEL_DIR="${MODEL_DIR:-$ROOT/models}"
LORA_DIR="${LORA_DIR:-$ROOT/loras}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT/output}"
MIN_VRAM_GB=44          # 48 GB card, allow for driver overhead
MIN_DISK_GB=150

# Wan 2.2 I2V 14B, FP16. One pair, no FP8/GGUF fallbacks.
WAN_REPO="Comfy-Org/Wan-2.2"
WAN_FILES=(
  "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp16.safetensors"
  "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp16.safetensors"
  "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
  "split_files/vae/wan_2.1_vae.safetensors"
)

# Qwen-Image 2.1 for stills. fp8 diffusion + fp8 encoder + VAE = 28 GB, which
# fits a 48 GB card with room for activations at 896x1344.
QWEN_REPO="Comfy-Org/Qwen-Image_ComfyUI"
QWEN_FILES=(
  "split_files/diffusion_models/qwen_image_fp8_e4m3fn.safetensors"
  "split_files/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors"
  "split_files/vae/qwen_image_vae.safetensors"
)

# ── Pre-flight ───────────────────────────────────────────────────────────────
phase "PRE-FLIGHT"

command -v nvidia-smi >/dev/null || die "nvidia-smi not found; no GPU driver"
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
VRAM_GB=$((VRAM_MB / 1024))
log "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) — ${VRAM_GB} GB"
[ "$VRAM_GB" -ge "$MIN_VRAM_GB" ] || die "need ${MIN_VRAM_GB} GB VRAM, found ${VRAM_GB} GB"

AVAIL_GB=$(df -BG --output=avail "$ROOT" | tail -1 | tr -dc '0-9')
log "Disk: ${AVAIL_GB} GB free"
[ "$AVAIL_GB" -ge "$MIN_DISK_GB" ] || die "need ${MIN_DISK_GB} GB free, found ${AVAIL_GB} GB"

[ -f .env ] || die "missing .env — copy .env.example and fill in HF_TOKEN"
set -a; . ./.env; set +a
[ -n "${HF_TOKEN:-}" ] || die "HF_TOKEN is not set in .env"

if $DRY_RUN; then
  log "dry run: pre-flight passed"
  exit 0
fi

# ── Phase 1: system ──────────────────────────────────────────────────────────
phase "PHASE 1: SYSTEM"

if ! command -v uv >/dev/null; then
  log "installing uv"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v aria2c >/dev/null; then
  log "installing aria2 + ffmpeg"
  apt-get update -qq && apt-get install -y -qq aria2 ffmpeg libgl1 libglib2.0-0 >/dev/null
fi

# ── Phase 2: python ──────────────────────────────────────────────────────────
phase "PHASE 2: PYTHON"

log "syncing dependencies"
uv sync --locked --inexact

# ── Phase 3: models ──────────────────────────────────────────────────────────
phase "PHASE 3: MODELS"

mkdir -p "$MODEL_DIR"/{diffusion_models,text_encoders,vae}

download() {
  local repo="$1" file="$2" dest="$3"
  local name; name="$(basename "$file")"
  local target="$dest/$name"
  if [ -f "$target" ] && [ "$(stat -c%s "$target")" -gt 1000000 ]; then
    log "cached: $name"
    return 0
  fi
  log "downloading: $name"
  local url="https://huggingface.co/$repo/resolve/main/$file"
  aria2c -x 8 -s 8 -k 1M --dir="$dest" --out="$name" \
    --header="Authorization: Bearer $HF_TOKEN" \
    --console-log-level=warn --summary-interval=0 "$url"
}

for f in "${WAN_FILES[@]}"; do
  case "$f" in
    *diffusion_models*) download "$WAN_REPO" "$f" "$MODEL_DIR/diffusion_models" ;;
    *text_encoders*)    download "$WAN_REPO" "$f" "$MODEL_DIR/text_encoders" ;;
    *vae*)              download "$WAN_REPO" "$f" "$MODEL_DIR/vae" ;;
  esac
done

for f in "${QWEN_FILES[@]}"; do
  case "$f" in
    *diffusion_models*) download "$QWEN_REPO" "$f" "$MODEL_DIR/diffusion_models" ;;
    *text_encoders*)    download "$QWEN_REPO" "$f" "$MODEL_DIR/text_encoders" ;;
    *vae*)              download "$QWEN_REPO" "$f" "$MODEL_DIR/vae" ;;
  esac
done

# ── Phase 4: LoRAs ───────────────────────────────────────────────────────────
phase "PHASE 4: LORAS"

# Trained character LoRAs are the identity. They are staged locally and never
# re-downloaded; a fresh training run wins over a stale copy.
for name in faiz_lora juv_lora; do
  if [ -f "$LORA_DIR/$name.safetensors" ]; then
    log "trained: $name"
  else
    warn "missing $name.safetensors — run scripts/train_loras_kaggle.py first"
  fi
done

# Anatomy LoRAs. These are the explicit content the beats declare.
if [ -n "${CIVITAI_API_KEY:-}" ]; then
  log "resolving anatomy LoRAs"
  uv run python scripts/download_anatomy_loras.py --dest "$LORA_DIR" || \
    warn "anatomy LoRA download failed; shots that declare them will render without"
else
  warn "CIVITAI_API_KEY not set; skipping anatomy LoRAs"
fi

# ── Phase 5: ComfyUI ─────────────────────────────────────────────────────────
phase "PHASE 5: COMFYUI"

if [ ! -d "$COMFY_DIR" ]; then
  log "cloning ComfyUI"
  git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git "$COMFY_DIR"
  uv pip install --python "$COMFY_DIR/venv/bin/python" -r "$COMFY_DIR/requirements.txt" 2>/dev/null || \
    uv pip install -r "$COMFY_DIR/requirements.txt"
fi

# GGUF node is only needed for the Qwen GGUF quants, which a 48 GB card does
# not use. Kept out to reduce the surface.
log "linking models into ComfyUI"
mkdir -p "$COMFY_DIR/models"/{diffusion_models,text_encoders,vae,loras}
for kind in diffusion_models text_encoders vae; do
  for f in "$MODEL_DIR/$kind"/*.safetensors; do
    [ -e "$f" ] || continue
    ln -sf "$f" "$COMFY_DIR/models/$kind/$(basename "$f")"
  done
done
for f in "$LORA_DIR"/*.safetensors; do
  [ -e "$f" ] || continue
  ln -sf "$f" "$COMFY_DIR/models/loras/$(basename "$f")"
done

log "starting ComfyUI on :$COMFY_PORT"
mkdir -p "$ROOT/logs"
nohup python "$COMFY_DIR/main.py" --listen 127.0.0.1 --port "$COMFY_PORT" \
  --highvram > "$ROOT/logs/comfy.log" 2>&1 &
COMFY_PID=$!
echo "$COMFY_PID" > "$ROOT/logs/comfy.pid"

for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$COMFY_PORT/system_stats" >/dev/null 2>&1; then
    log "ComfyUI up after $((i*2))s"
    break
  fi
  [ "$i" -eq 120 ] && { tail -40 "$ROOT/logs/comfy.log"; die "ComfyUI did not start"; }
  sleep 2
done

# ── Phase 6: prompts ─────────────────────────────────────────────────────────
phase "PHASE 6: PROMPTS"

log "compiling beat prompts"
uv run python scripts/build_prompts.py --no-frames --lang hi

# ── Phase 7: stills ──────────────────────────────────────────────────────────
phase "PHASE 7: STILLS (Qwen-Image 2.1)"

log "generating startframes with photo references"
uv run python scripts/generate_stills_qwen.py \
  --output startframes \
  --server "http://127.0.0.1:$COMFY_PORT"

# ── Phase 8: video ───────────────────────────────────────────────────────────
if $SKIP_VIDEO; then
  warn "skipping video render (--skip-video)"
else
  phase "PHASE 8: VIDEO (Wan 2.2 I2V FP16)"
  log "rendering 70 clips"
  uv run python scripts/render.py --precision fp16
fi

# ── Phase 9: audio ───────────────────────────────────────────────────────────
phase "PHASE 9: AUDIO"

log "generating narration"
uv run python scripts/generate_narration.py --prompts prompts --output audio --engine chatterbox

log "muxing audio"
uv run python scripts/mux_audio.py

# ── Phase 10: package ────────────────────────────────────────────────────────
phase "PHASE 10: PACKAGE"

mkdir -p "$OUTPUT_DIR"
uv run python scripts/assemble_final.py

log "done — output in $OUTPUT_DIR"
