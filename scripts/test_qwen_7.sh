#!/usr/bin/env bash
# Qwen-Image 2.1 startframe test — 7 beats.
#
# Run this on the 48 GB GPU after ComfyUI is up. It renders the 7 beats that
# failed or were swapped under SDXL, so you can judge the Qwen output before
# committing to all 70.
#
#   bash scripts/test_qwen_7.sh
#
# Expects:
#   - ComfyUI on http://127.0.0.1:8188
#   - models/diffusion_models/qwen_image_fp8_e4m3fn.safetensors
#   - models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors
#   - models/vae/qwen_image_vae.safetensors
#   - refs/Faiz/ and refs/Juvi/ reference plates
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CYAN='\033[0;36m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log()  { echo -e "${GREEN}▸${NC} $*"; }
warn() { echo -e "${YELLOW}!${NC} $*"; }
die()  { echo -e "${RED}✗${NC} $*" >&2; exit 1; }

SERVER="${COMFY_URL:-http://127.0.0.1:8188}"
OUT="${OUT:-startframes_qwen_test}"

# The 7 beats that failed or were swapped under SDXL:
#   c01 POV, needed 2 candidates
#   c02 POV, only one face in frame
#   c10 c11 c15 c31 c37 ECU framing, no face
BEATS=(c01 c02 c10 c11 c15 c31 c37)

echo -e "\n${CYAN}═══ QWEN-IMAGE 2.1 TEST — 7 BEATS ═══${NC}\n"

# ── Pre-flight ───────────────────────────────────────────────────────────────
log "checking ComfyUI at $SERVER"
curl -sf "$SERVER/system_stats" >/dev/null || die "ComfyUI is not responding at $SERVER"

log "checking models"
COMFY_DIR="${COMFY_DIR:-$ROOT/ComfyUI}"
for f in \
  "$COMFY_DIR/models/diffusion_models/qwen_image_fp8_e4m3fn.safetensors" \
  "$COMFY_DIR/models/text_encoders/qwen_2.5_vl_7b_fp8_scaled.safetensors" \
  "$COMFY_DIR/models/vae/qwen_image_vae.safetensors"
do
  [ -f "$f" ] || die "missing model: $f"
  log "  $(basename "$f") — $(du -h "$f" | cut -f1)"
done

log "checking reference plates"
for f in \
  "refs/Faiz/01_front_standing_straight_00001.png" \
  "refs/Faiz/IMG_3026.JPG" \
  "refs/Juvi/01_front_standing_00001.png" \
  "refs/Juvi/IMG_3049.jpg"
do
  [ -f "$f" ] || die "missing reference: $f"
done
log "  all 4 plates present"

log "checking VRAM"
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
VRAM_GB=$((VRAM_MB / 1024))
log "  ${VRAM_GB} GB"
[ "$VRAM_GB" -ge 40 ] || warn "under 40 GB — Qwen fp8 needs ~28 GB plus activations"

# ── Render ───────────────────────────────────────────────────────────────────
echo
log "rendering ${#BEATS[@]} beats to $OUT"
log "  896x1344, 25 steps, cfg 1.0, euler/simple"
echo

uv run --locked --inexact python scripts/generate_stills_qwen.py \
  --output "$OUT" \
  --server "$SERVER" \
  --beats "${BEATS[@]}"

# ── Report ───────────────────────────────────────────────────────────────────
echo
echo -e "${CYAN}═══ RESULTS ═══${NC}\n"

if [ ! -d "$OUT" ]; then
  die "no output directory produced"
fi

COUNT=$(find "$OUT" -name 'c*.png' | wc -l)
log "rendered: $COUNT / ${#BEATS[@]}"

for b in "${BEATS[@]}"; do
  f="$OUT/$b.png"
  if [ -f "$f" ]; then
    log "  $b — $(du -h "$f" | cut -f1)"
  else
    warn "  $b — MISSING"
  fi
done

echo
log "inspect the images in $OUT"
log "if the faces are right, run the full set:"
echo "    uv run python scripts/generate_stills_qwen.py --output startframes"
