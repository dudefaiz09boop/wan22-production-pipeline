# WAN 2.2 Production Pipeline - AIC Cloud

**One-click production pipeline for WAN 2.2 14B I2V on AIC Cloud GPUs.**

This is a standalone, production-ready pipeline that merges the best of:
- **AI Pipeline** (70-clip explicit adult production with ComfyUI, continuity, LoRAs)
- **fzjv-pipeline** (SDXL stills + InsightFace identity swap + trained character LoRAs)

---

## ⚡ One-Click Run

**On a fresh AIC Cloud GPU instance, after SSH:**

```bash
git clone https://github.com/dudefaiz09boop/wan22-production-pipeline.git
cd wan22-production-pipeline
cp .env.example .env      # fill in HF_TOKEN, CIVITAI_API_KEY, AIC_API_KEY
bash scripts/setup_aic_fast.sh
```

That single command does everything: installs dependencies, downloads models,
starts ComfyUI, generates and repairs start frames, renders all 70 clips,
face-swaps, runs audio-driven lipsync, generates narration, muxes audio, and
publishes the deliverable for download.

**Timeline (RTX A6000 45 GB, FP16):**

| Phase | Time |
|-------|------|
| System + deps | ~2 min |
| Model downloads (92 GB) | ~12–18 min |
| ComfyUI + stills + repair | ~20–30 min |
| **70-clip render** | **~35–50 min** |
| Face swap | ~5–8 min |
| **Lipsync (InfiniteTalk, 44 clips)** | **~20–30 min** |
| Narration + mux + package | ~10–15 min |
| **Total** | **~105–145 min** |
| **Cost** | **₹125–175** |

### Getting the videos back to your machine

AIC Cloud GPU instances are created in the dashboard and expose **no GPU API**,
so the instance cannot push to your laptop by itself. Two options:

```bash
# Option A (default): the instance serves results over HTTP
python scripts/pull_results.py --http http://<instance-ip>:8000

# Option B: set PULL_HOST in .env and the instance scp's results to you
PULL_HOST=user@your-laptop
```

See [`AIC_CLOUD_RUNBOOK.md`](AIC_CLOUD_RUNBOOK.md) for the exact end-to-end steps.

---

## 🔑 Environment

All configuration lives in `.env` (gitignored). Copy `.env.example` and fill it in.

| Variable | Required | Purpose |
|----------|----------|---------|
| `HF_TOKEN` | ✅ | Model + base model downloads |
| `CIVITAI_API_KEY` | ✅ | Anatomy + quality LoRA downloads |
| `AIC_API_KEY` | optional | Wallet/billing read only |
| `KAGGLE_KEY` | optional | Only if LoRAs are not yet published |
| `WAN_PRECISION` | optional | `auto` (default), `fp16`, `fp8`, `gguf_q8` |
| `LIPSYNC_ENGINE` | optional | `infinitetalk` (default), `s2v`, `off` |
| `REPAIR_FRAMES` | optional | `true` (default) regenerates bad start frames |
| `PULL_HOST` | optional | `user@host` to scp results automatically |
| `AUTO_SHUTDOWN` | optional | `true` to stop the instance after rendering |

**Note on AIC Cloud:** their API (`https://api.aiccloud.in`) covers VPS, email,
domains and billing only — there are no GPU endpoints. GPU instances must be
created in the dashboard, so provisioning cannot be automated end to end.

---

## Features

### 🎯 Auto GPU Detection & Model Selection
| GPU VRAM | Model | Quality |
|----------|-------|---------|
| **44GB+** (A6000 45GB, L40S, A100-80GB) | **FP16** (wan2.2_i2v_high/low_noise_14B_fp16.safetensors) | **Best** - Full precision |
| **24-47GB** (RTX 3090, A100-40GB) | **FP8 Scaled** (wan2.2_i2v_high/low_noise_14B_fp8_scaled.safetensors) | Excellent - Near FP16 |
| **<24GB** (T4, RTX 3080) | **FP8 fallback** | Good |

### 🔄 11-Phase Pipeline
1. **System setup** — apt, uv, aria2, ffmpeg
2. **Python deps** — `uv sync` from lockfile
3. **Model downloads** — parallel aria2, 92 GB
4. **LoRAs** — anatomy + quality from Civitai, trained characters local
5. **Validate** — spec check, workflow generation
6. **ComfyUI** — clone, link models, start server
7. **Stills** — SDXL + trained LoRAs + InsightFace swap, per-shot resolution
7b. **Frame audit + repair** — regenerate blurry/faceless frames
8. **Render** — WAN 2.2 I2V, 70 clips with continuity
9. **Face swap** — InsightFace every 3rd frame
9b. **Lipsync** — InfiniteTalk, audio-driven mouth
10. **Narration + mux** — Edge TTS, audio mux, final cut
11. **Publish** — HTTP or scp for download

### 🎬 Production Spec (from docs/beats.yaml)
- **70 clips** across 12 scenes
- **344 seconds** total (5s base, 4s for noise-3 beats)
- **Resolutions**: 832×480 (base) / 1280×720 (face closeups, 34 shots)
- **Sampler**: Euler / Simple, CFG 1, Lightning edition
- **LoRAs**: penis_insert (23 shots), vag_detail (22), nsfw_general (3)
- **Audio**: 44 clips with narration, 214.9s total

---

## Quick Start

### Prerequisites
- AIC Cloud RTX A6000 45 GB, 200 GB disk, PyTorch image
- SSH access to instance
- `HF_TOKEN`, `CIVITAI_API_KEY` in `.env`

### One-Command Setup & Run

```bash
# On AIC Cloud instance (after SSH)
git clone https://github.com/dudefaiz09boop/wan22-production-pipeline.git
cd wan22-production-pipeline
cp .env.example .env && nano .env

# Run full production
bash scripts/setup_aic_fast.sh
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
```

---

## Pipeline Stages Detail

### Stage 1: LoRAs
- **Anatomy** from Civitai: `penis_insert`, `vag_detail` (high + low each)
- **Quality**: `nsfw_general` (416k downloads), `genitals_helper`
- **Trained characters**: `faiz_lora`, `juv_lora` (local, never retrained)
- `both_trained` is optional and unused by the spec

### Stage 2: Generate SDXL Stills
- Juggernaut XL v9 + trained LoRAs, adapters **activated** via `set_adapters`
- Per-shot resolution: 832×480 base, **1280×720 for 34 face closeups**
- InsightFace swap of both faces, best of N candidates by identity score
- Outputs to `startframes/` as PNG

### Stage 2b: Frame Audit + Repair
- Flags blur, darkness, missing faces, tiny faces
- Regenerates only the failures with more candidates
- Current state: 40 ok, 30 problems (auto-repaired)

### Stage 3: Render Clips (ComfyUI)
- WAN 2.2 I2V 14B (FP16 on 44 GB+, FP8 on 22 GB+)
- Dual UNET: HIGH (2 steps) → LOW (3 steps)
- Continuity via `chain: true/false` and start frames
- LoRA injection per shot config
- Clips downloaded to `renders/<id>.mp4`

### Stage 4: Face Swap Video
- InsightFace swapper (`inswapper_128.onnx`)
- Assigns each face to the closest character by cosine similarity
- Swaps every 3rd frame, both characters per frame
- Verified: identity moves 0.092 → 0.880 on a forced swap

### Stage 5: Lipsync (InfiniteTalk)
- `WanInfiniteTalkToVideo` with `ModelPatchLoader`
- Two-speaker capable with per-speaker masks
- Runs on the 44 clips that have narration
- Fallback: `LIPSYNC_ENGINE=s2v` or `off`

### Stage 6: Narration + Mux
- Edge TTS, speaker-aware voices (Faiz/Juveriya/Ammi)
- Each track fitted to its clip duration
- Audio muxed per clip, then concatenated with one continuous AAC stream
- Optional music bed via `MUSIC_BED`

### Stage 7: Package
- `output/final.mp4` with video + audio
- `output/deliverable.zip` with clips, frames, logs, spec

---

## Directory Structure

```
wan22-production-pipeline/
├── docs/
│   └── beats.yaml              # Production spec (70 clips)
├── scripts/
│   ├── setup_aic_fast.sh       # ⚡ One-click AIC Cloud setup (11 phases)
│   ├── run_production.py       # Render orchestrator
│   ├── auto_model_selector.py  # GPU VRAM → model selection
│   ├── download_models.py      # Parallel model downloads
│   ├── download_loras.py       # Trained LoRAs (local-first)
│   ├── download_anatomy_loras.py # Civitai anatomy + quality LoRAs
│   ├── generate_stills.py      # SDXL + LoRA + InsightFace swap
│   ├── audit_startframes.py    # Frame quality audit
│   ├── repair_startframes.py   # Regenerate failing frames
│   ├── generate_workflow.py    # Dynamic ComfyUI workflow
│   ├── face_swap_video.py      # InsightFace video swap
│   ├── lipsync.py              # InfiniteTalk / S2V lipsync
│   ├── generate_narration.py   # Edge TTS narration
│   ├── mux_audio.py            # Audio mux + final cut
│   ├── assemble_final.py       # Concat + zip
│   ├── check_credentials.py    # Live API key verification
│   ├── disk_budget.py          # Disk requirement calculator
│   ├── report_spec_usage.py    # LoRA/character coverage report
│   ├── refresh_kaggle_token.py # Kaggle OAuth refresh
│   ├── pull_results.py         # Fetch results to local machine
│   ├── comfy_api.py            # ComfyUI API client
│   ├── render.py               # Clip rendering logic
│   └── validate.py             # Spec validation
├── workflows/
│   ├── base_i2v_fp16.json      # FP16 workflow template
│   ├── base_i2v_fp8.json       # FP8 workflow template
│   └── loras.json              # LoRA registry
├── negatives/
│   └── artifact_neg.txt        # Negative prompt
├── references/                 # faiz_ref.png, juv_ref.png
├── loras/                      # 8 staged LoRA files (gitignored)
├── AIC_CLOUD_RUNBOOK.md        # Exact end-to-end steps
├── PREREQUISITES.md            # Pre-flight checklist
├── package.json                # Node dependencies
├── pyproject.toml              # Python dependencies
└── uv.lock                     # Locked Python deps
```

---

## Model Downloads (Auto)

### FP16 (48 GB VRAM) — 92 GB total
- `wan2.2_i2v_high_noise_14B_fp16.safetensors` (26.62 GB)
- `wan2.2_i2v_low_noise_14B_fp16.safetensors` (26.62 GB)
- `umt5_xxl_fp8_e4m3fn_scaled.safetensors` (6.27 GB)
- `wan_2.1_vae.safetensors` (0.24 GB)
- `wan2.1_i2v_480p_14B_fp8_scaled.safetensors` (15.27 GB, lipsync)
- `Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors` (2.53 GB)
- `wav2vec2_large_english_fp16.safetensors` (0.59 GB)
- `Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors` (7.11 GB, stills)

### FP8 Scaled (24 GB VRAM) — ~66 GB total
- `wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors` (14.3 GB)
- `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors` (14.3 GB)
- Shared files same as above

Run `uv run python scripts/disk_budget.py` for the live figure.

---

## Outputs

```
output/
├── final.mp4              # Final video WITH audio
├── deliverable.zip        # Clips + frames + logs + spec
├── renders/
│   ├── c01.mp4 ... c70.mp4        # raw render
│   ├── face_swapped/              # after InsightFace
│   ├── lipsync/                   # after InfiniteTalk
│   └── muxed/                     # with narration
├── audio/
│   ├── c01.wav ... c70.wav        # 44 narration tracks
│   └── index.json
├── startframes/
│   ├── c01.png ... c70.png
│   └── quality_report.json
└── logs/
    └── production_YYYYMMDD_HHMMSS.json
```

---

## Cost Estimation (AIC Cloud)

| GPU | Hourly (INR) | Total Run (INR) | Quality |
|-----|--------------|-----------------|---------|
| **RTX A6000 45GB** | 81.02 | **150-200** | **FP16 - Best** |
| RTX 3090 24GB | 37.97 | 70-100 | FP8 Scaled |
| RTX 4090 48GB | 73.60 | 125-175 | FP16 |
| A100 SXM4 80GB | 94.00 | - | **Avoid: 29 GB disk** |

---

## Validation

```bash
# Spec validation (no frames needed)
uv run --locked --inexact python scripts/validate.py --no-frames

# Python tests
uv run --locked --inexact python -m pytest tests/ -q

# Live credential check (.env, or --env-file instance.env)
uv run --locked --inexact python scripts/check_credentials.py

# Disk requirement
uv run --locked --inexact python scripts/disk_budget.py

# LoRA / character coverage
uv run --locked --inexact python scripts/report_spec_usage.py

# Setup script block balance
uv run --locked --inexact python scripts/_check_setup_balance.py
```

`--locked --inexact` keeps `uv run` from re-resolving the environment or
pruning the out-of-lock Chatterbox packages. See
[docs/aic-cloud-deployment-findings.md](docs/aic-cloud-deployment-findings.md#26-uv-run-re-syncs-the-venv-on-every-call--and-can-re-download-torch).

Current state:

```
validate.py --no-frames   → OK 70 shots validated (0 warnings)
pytest tests/ -q          → 200 passed, 16 subtests
check_credentials.py      → 8/8 passed
disk_budget.py            → 92.41 GB total, 115.51 GB with headroom
report_spec_usage.py      → all spec LoRAs staged
```

---

## Requirements

- **Python 3.11+** with `uv`
- **NVIDIA GPU** with CUDA — 48 GB for FP16, 24 GB for FP8
- **Disk** — 200 GB recommended (92 GB measured need)
- **ComfyUI** on `http://127.0.0.1:8188` (auto-started)
- **`HF_TOKEN`** with access to `Comfy-Org/Wan-2.2` and `Comfy-Org/Wan-2.1`
- **`CIVITAI_API_KEY`** for anatomy and quality LoRAs

---

## License

Private production pipeline. Not for distribution.
