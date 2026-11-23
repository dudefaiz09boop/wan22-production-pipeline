# WAN 2.2 Production Pipeline - AIC Cloud

**One-click production pipeline for WAN 2.2 14B I2V on AIC Cloud GPUs.**

This is a standalone, production-ready pipeline that merges the best of:
- **AI Pipeline** (70-clip explicit adult production with ComfyUI, continuity, LoRAs)
- **fzjv-pipeline** (SDXL stills + InsightFace identity swap + trained character LoRAs)

---

## ⚡ Ultra-Fast Setup (NEW)

**For AIC Cloud GPU instances - optimized for zero GPU waste:**

```bash
# On fresh AIC Cloud instance (after SSH)
export HF_TOKEN="hf_your_token"
export KAGGLE_KEY='{"username":"yourname","key":"your_key"}'
curl -fsSL https://raw.githubusercontent.com/YOUR_USERNAME/wan22-production-pipeline/main/scripts/setup_aic_fast.sh | bash
```

**Timeline (L40S 48GB):**
- System + deps: ~90s
- Model downloads (57GB): ~8-12 min (parallel aria2)
- ComfyUI + Stills: ~5-10 min
- **70-clip render: ~25-35 min**
- Face swap + assemble: ~5-8 min
- **Total: ~45-60 min | Cost: ~₹42-56**

See [`PREREQUISITES.md`](PREREQUISITES.md) for complete checklist.

---

## Features

### 🎯 Auto GPU Detection & Model Selection
| GPU VRAM | Model | Quality |
|----------|-------|---------|
| **48GB+** (L40S, A100-80GB) | **FP16** (wan2.2_i2v_high/low_noise_14B_fp16.safetensors) | **Best** - Full precision |
| **24-47GB** (RTX 3090, A6000, A100-40GB) | **FP8 Scaled** (wan2.2_i2v_high/low_noise_14B_fp8_scaled.safetensors) | Excellent - Near FP16 |
| **<24GB** (T4, RTX 3080) | **GGUF Q8** (current workflow) | Good - Quantized |

### 🔄 5-Stage Pipeline
1. **Download trained LoRAs** from Kaggle (sofiya009/lora-nuke-faiz-juv)
2. **Generate SDXL stills** with trained LoRAs + InsightFace swap
3. **Render clips** via ComfyUI (WAN 2.2 I2V with continuity)
4. **Face swap** every 3rd frame using InsightFace
5. **Assemble final** deliverable (concat, crossfade, zip)

### 🎬 Production Spec (from docs/beats.yaml)
- **70 clips** across 12 scenes
- **344 seconds** total (5s base, 4s for noise-3 beats)
- **Resolutions**: 832×480 (base) / 1280×720 (face closeups)
- **Sampler**: Euler / Simple, CFG 1, Lightning edition
- **LoRAs**: penis_insert, vag_detail, oral_insert, pov_finish, nsfw_helper, orbit + trained character LoRAs

---

## Quick Start

### Prerequisites
- AIC Cloud instance (L40S 48GB recommended for FP16)
- SSH access to instance
- Hugging Face token (for model downloads)
- Kaggle API key (for trained LoRA downloads)

### One-Command Setup & Run

```bash
# On AIC Cloud instance (after SSH)
git clone https://github.com/YOUR_USERNAME/wan22-production-pipeline.git
cd wan22-production-pipeline

# Set tokens
export HF_TOKEN="your_hf_token"
export KAGGLE_KEY="your_kaggle_key"

# Run full production (ULTRA-FAST)
./scripts/setup_aic_fast.sh

# OR traditional step-by-step:
./run_production.sh
```

### Options (run_production.sh)

```bash
# Validate only (dry run)
./run_production.sh --dry-run

# Force specific precision
./run_production.sh --force-precision fp8

# Render specific clip range
./run_production.sh --from-clip c10 --to-clip c20

# Skip stages
./run_production.sh --skip-download --skip-loras --skip-stills

# Only download models + LoRAs
./run_production.sh --skip-render --skip-face-swap --skip-assemble
```

---

## Pipeline Stages Detail

### Stage 1: Download Trained LoRAs
Downloads character LoRAs from `sofiya009/lora-nuke-faiz-juv`:
- `faiz_lora.safetensors` - Faiz character
- `juv_lora.safetensors` - Juveriya character  
- `faiz_juv_lora.safetensors` - Combined both

Updates `workflows/loras.json` with trained LoRA entries.

### Stage 2: Generate SDXL Stills
- Uses Juggernaut XL v9 + trained LoRAs
- InsightFace swap for identity consistency (cos ≥ 0.40)
- Outputs to `startframes/` as PNG

### Stage 3: Render Clips (ComfyUI)
- WAN 2.2 I2V 14B (FP16/FP8/GGUF Q8 based on VRAM)
- Dual UNET: HIGH (2 steps) → LOW (3 steps)
- Continuity via `chain: true/false` and start frames
- LoRA injection per shot config

### Stage 4: Face Swap Video
- InsightFace swapper (inswapper_128.onnx)
- Swaps every 3rd frame
- Uses trained reference embeddings
- Outputs to `renders/face_swapped/`

### Stage 5: Assemble Final
- Concatenates all clips in order
- Crossfade transitions (0.5s default)
- H.264 CRF 18, slow preset
- Creates `output/final.mp4` + `output/deliverable.zip`

---

## Directory Structure

```
wan22-production-pipeline/
├── docs/
│   └── beats.yaml              # Production spec (70 clips)
├── scripts/
│   ├── setup_aic_fast.sh       # ⚡ ULTRA-FAST AIC Cloud setup
│   ├── run_production.py       # Main orchestrator
│   ├── auto_model_selector.py  # GPU VRAM → model selection
│   ├── download_models.py      # Parallel model downloads
│   ├── download_loras.py       # Trained LoRAs from Kaggle
│   ├── generate_stills.py      # SDXL + LoRA + InsightFace
│   ├── generate_workflow.py    # Dynamic ComfyUI workflow
│   ├── face_swap_video.py      # InsightFace video swap
│   ├── assemble_final.py       # Concat + crossfade + zip
│   ├── comfy_api.py            # ComfyUI API client
│   ├── render.py               # Clip rendering logic
│   └── validate.py             # Spec validation
├── workflows/
│   ├── base_i2v_fp16.json      # FP16 workflow template
│   ├── base_i2v_fp8.json       # FP8 workflow template
│   ├── base_i2v_gguf_q8.json   # GGUF Q8 workflow template
│   └── loras.json              # LoRA registry (auto-updated)
├── negatives/
│   └── artifact_neg.txt        # Negative prompt
├── run_production.sh           # Shell wrapper (traditional)
├── PREREQUISITES.md            # Complete pre-flight checklist
├── package.json                # Node dependencies
├── pyproject.toml              # Python dependencies
└── uv.lock                     # Locked Python deps
```

---

## Model Downloads (Auto)

### FP16 (48GB+ VRAM)
- `wan2.2_i2v_high_noise_14B_fp16.safetensors` (28.6 GB)
- `wan2.2_i2v_low_noise_14B_fp16.safetensors` (28.6 GB)
- `umt5_xxl_fp8_e4m3fn.safetensors` (5.5 GB)
- `wan_2.1_vae.safetensors` (1.5 GB)

### FP8 Scaled (24-47GB VRAM)
- `wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors` (14.3 GB)
- `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors` (14.3 GB)
- Shared files same as above

### GGUF Q8 (<24GB VRAM)
- `Wan2_2-I2V-A14B-NSFW_fast_move-HIGH_Q8(lightning_edition)V2.gguf`
- `Wan2_2-I2V-A14B-NSFW_fast_move-LOW_Q8(lightning_edition)V2.gguf`
- Shared files same as above

---

## Outputs

```
output/
├── final.mp4              # Concatenated final video
├── deliverable.zip        # All clips + frames + logs + beats.yaml
├── renders/
│   ├── c01.mp4 ... c70.mp4
│   └── face_swapped/
│       ├── c01.mp4 ... c70.mp4
├── startframes/
│   ├── c01.png ... c70.png
└── logs/
    └── production_YYYYMMDD_HHMMSS.json
```

---

## Cost Estimation (AIC Cloud)

| GPU | Hourly Rate | 70-Clip Render | Quality |
|-----|-------------|----------------|---------|
| **L40S 48GB** | ~₹84/hr | ~₹42-56 | **FP16 - Best** |
| RTX 3090 24GB | ~₹30/hr | ~₹25-35 | FP8 Scaled |
| RTX A6000 48GB | ~₹60-80/hr | ~₹30-40 | FP16 |

---

## Validation

```bash
# Validate beats.yaml (no frames needed)
uv run python scripts/validate.py --no-frames

# Run Python tests
uv run python -m pytest tests/ -v

# Run Node tests
npm test

# Syntax check
npm run check
```

---

## Requirements

- **Python 3.11+** with `uv` package manager
- **Node.js 22.13+** 
- **NVIDIA GPU** with CUDA (48GB+ for FP16)
- **ComfyUI** running on `http://127.0.0.1:8188` (auto-started by pipeline)
- **Hugging Face token** with access to Comfy-Org/Wan-2.2
- **Kaggle API key** for trained LoRA downloads

---

## License

Private production pipeline. Not for distribution.