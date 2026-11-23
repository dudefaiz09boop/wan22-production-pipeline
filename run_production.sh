#!/bin/bash
# One-click production runner for WAN 2.2 - AIC Cloud
# Usage: ./run_production.sh [options]

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
SPEC="docs/beats.yaml"
PROMPTS="prompts"
STARTFRAMES="startframes"
RENDERS="renders"
OUTPUT="output"
SERVER=""
FORCE_PRECISION=""
SKIP_DOWNLOAD=false
SKIP_LORAS=false
SKIP_STILLS=false
SKIP_RENDER=false
SKIP_FACE_SWAP=false
SKIP_ASSEMBLE=false
FROM_CLIP="c01"
TO_CLIP="c70"
DRY_RUN=false
HF_TOKEN=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --spec)
            SPEC="$2"
            shift 2
            ;;
        --prompts)
            PROMPTS="$2"
            shift 2
            ;;
        --startframes)
            STARTFRAMES="$2"
            shift 2
            ;;
        --renders)
            RENDERS="$2"
            shift 2
            ;;
        --output)
            OUTPUT="$2"
            shift 2
            ;;
        --server)
            SERVER="$2"
            shift 2
            ;;
        --force-precision)
            FORCE_PRECISION="$2"
            shift 2
            ;;
        --skip-download)
            SKIP_DOWNLOAD=true
            shift
            ;;
        --skip-loras)
            SKIP_LORAS=true
            shift
            ;;
        --skip-stills)
            SKIP_STILLS=true
            shift
            ;;
        --skip-render)
            SKIP_RENDER=true
            shift
            ;;
        --skip-face-swap)
            SKIP_FACE_SWAP=true
            shift
            ;;
        --skip-assemble)
            SKIP_ASSEMBLE=true
            shift
            ;;
        --from-clip)
            FROM_CLIP="$2"
            shift 2
            ;;
        --to-clip)
            TO_CLIP="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --hf-token)
            HF_TOKEN="$2"
            shift 2
            ;;
        -h|--help)
            cat << EOF
Usage: $0 [options]

One-click WAN 2.2 production pipeline for AIC Cloud.

Pipeline Stages:
  1. Download trained LoRAs from Kaggle (sofiya009/lora-nuke-faiz-juv)
  2. Generate SDXL stills with trained LoRAs + InsightFace swap
  3. Render clips via ComfyUI (WAN 2.2 I2V)
  4. Face swap every 3rd frame using InsightFace
  5. Assemble final deliverable (concat, crossfade, zip)

Options:
  --spec FILE           Beats specification (default: docs/beats.yaml)
  --prompts DIR         Prompts directory (default: prompts)
  --startframes DIR     Start frames directory (default: startframes)
  --renders DIR         Output renders directory (default: renders)
  --output DIR          Final output directory (default: output)
  --server URL          ComfyUI server URL (default: auto-detect)
  --force-precision P   Force precision: fp16, fp8, gguf_q8 (default: auto)
  --skip-download       Skip base model download
  --skip-loras          Skip trained LoRA download
  --skip-stills         Skip stills generation
  --skip-render         Skip clip rendering
  --skip-face-swap      Skip face swap
  --skip-assemble       Skip final assembly
  --from-clip ID        Start clip (default: c01)
  --to-clip ID          End clip (default: c70)
  --dry-run             Validate without rendering
  --hf-token TOKEN      Hugging Face token
  -h, --help            Show this help

Environment variables:
  HF_TOKEN              Hugging Face token (alternative to --hf-token)
  KAGGLE_KEY            Kaggle API key (for trained LoRA download)
  COMFY_URL             ComfyUI server URL
  MODEL_DIR             Model storage directory (default: models)

Examples:
  $0                                    # Full production run
  $0 --dry-run                          # Validate only
  $0 --force-precision fp8              # Force FP8 precision
  $0 --from-clip c10 --to-clip c20      # Render clips 10-20 only
  $0 --skip-download                    # Use existing models
  $0 --skip-loras --skip-stills         # Only render + assemble
  $0 --skip-render --skip-face-swap     # Only download + stills + assemble
EOF
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Print banner
echo -e "${BLUE}"
echo "============================================================"
echo "  WAN 2.2 PRODUCTION PIPELINE - AIC Cloud"
echo "============================================================"
echo -e "${NC}"

# Check prerequisites
echo -e "${YELLOW}Checking prerequisites...${NC}"

# Check Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 not found${NC}"
    exit 1
fi
echo -e "  Python: $(python3 --version)"

# Check uv
if ! command -v uv &> /dev/null; then
    echo -e "${RED}uv not found - install with: pip install uv${NC}"
    exit 1
fi
echo -e "  uv: $(uv --version)"

# Check nvidia-smi
if command -v nvidia-smi &> /dev/null; then
    VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
    echo -e "  GPU VRAM: ${VRAM} MB ($((VRAM / 1024)) GB)"
else
    echo -e "  ${YELLOW}GPU: nvidia-smi not found (CPU only?)${NC}"
fi

# Check ComfyUI
if [[ -n "$SERVER" ]]; then
    COMFY_URL="$SERVER"
elif [[ -n "${COMFY_URL:-}" ]]; then
    : # Use existing
else
    COMFY_URL="http://127.0.0.1:8188"
fi
echo -e "  ComfyUI: $COMFY_URL"

# Load .env if exists
if [[ -f ".env" ]]; then
    echo -e "  Loading .env..."
    set -a
    source .env
    set +a
fi

# Set HF_TOKEN from env if not provided
if [[ -z "$HF_TOKEN" && -n "${HF_TOKEN:-}" ]]; then
    HF_TOKEN="$HF_TOKEN"
fi

# Build Python command
PYTHON_CMD=("uv" "run" "python" "scripts/run_production.py")
PYTHON_CMD+=("--spec" "$SPEC")
PYTHON_CMD+=("--prompts" "$PROMPTS")
PYTHON_CMD+=("--startframes" "$STARTFRAMES")
PYTHON_CMD+=("--renders" "$RENDERS")
PYTHON_CMD+=("--output" "$OUTPUT")
PYTHON_CMD+=("--server" "$COMFY_URL")

if [[ -n "$FORCE_PRECISION" ]]; then
    PYTHON_CMD+=("--force-precision" "$FORCE_PRECISION")
fi

if [[ "$SKIP_DOWNLOAD" == true ]]; then
    PYTHON_CMD+=("--skip-download")
fi

if [[ "$SKIP_LORAS" == true ]]; then
    PYTHON_CMD+=("--skip-loras")
fi

if [[ "$SKIP_STILLS" == true ]]; then
    PYTHON_CMD+=("--skip-stills")
fi

if [[ "$SKIP_RENDER" == true ]]; then
    PYTHON_CMD+=("--skip-render")
fi

if [[ "$SKIP_FACE_SWAP" == true ]]; then
    PYTHON_CMD+=("--skip-face-swap")
fi

if [[ "$SKIP_ASSEMBLE" == true ]]; then
    PYTHON_CMD+=("--skip-assemble")
fi

PYTHON_CMD+=("--from-clip" "$FROM_CLIP")
PYTHON_CMD+=("--to-clip" "$TO_CLIP")

if [[ "$DRY_RUN" == true ]]; then
    PYTHON_CMD+=("--dry-run")
fi

if [[ -n "$HF_TOKEN" ]]; then
    PYTHON_CMD+=("--hf-token" "$HF_TOKEN")
fi

# Print command
echo -e "\n${YELLOW}Running:${NC}"
echo "  ${PYTHON_CMD[*]}"
echo ""

# Run production
START_TIME=$(date +%s)
if "${PYTHON_CMD[@]}"; then
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo -e "\n${GREEN}Production completed successfully in ${DURATION}s${NC}"
    exit 0
else
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo -e "\n${RED}Production failed after ${DURATION}s${NC}"
    exit 1
fi