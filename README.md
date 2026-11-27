# WAN 2.2 Production Pipeline

**One-click production pipeline for WAN 2.2 14B I2V on a 48 GB GPU.**

Stills are generated with **Qwen-Image 2.1** from photo references, so identity
comes from the reference image rather than a LoRA plus a face paste. Video is
**Wan 2.2 I2V 14B at FP16**, which a 48 GB card holds without offloading.

---

## One-Click Run

On a 48 GB GPU instance, after SSH:

```bash
git clone https://github.com/dudefaiz09boop/wan22-production-pipeline.git
cd wan22-production-pipeline
cp .env.example .env      # fill in HF_TOKEN, CIVITAI_API_KEY
bash scripts/setup_48gb.sh
```

That single command installs dependencies, downloads only the models a 48 GB
card actually uses, starts ComfyUI, generates startframes with Qwen-Image 2.1,
renders all 70 clips at FP16, generates narration, muxes audio, and packages the
deliverable.

**Timeline (48 GB GPU, FP16):**

| Phase | Time |
|-------|------|
| System + deps | ~3 min |
| Model downloads (63 GB) | ~10-15 min |
| ComfyUI start | ~1 min |
| **Stills (Qwen-Image 2.1)** | **~45 min** |
| **70-clip render (FP16)** | **~35-50 min** |
| Narration + mux + package | ~10-15 min |
| **Total** | **~105-125 min** |

### Why Qwen-Image for stills

The stills used to be SDXL + trained LoRA + InsightFace paste. That had two
problems that could not be fixed:

- **SDXL encodes text with CLIP at 77 tokens.** Prompts had to be trimmed to
  fragments, and the tail -- where the explicit content and the LoRA triggers
  sit -- was silently cut.
- **SDXL has no native image reference.** Identity came from a LoRA plus a face
  paste, and the paste assigned faces by size, so the characters got swapped
  whenever the wrong face was larger.

Qwen-Image 2.1 encodes with an 8B Qwen2.5-VL vision model and takes reference
images directly:

```
TextEncodeQwenImage21(clip, images.image_1, images.image_2, vae)
```

The prompt says "use `<image1>` for his body, `<image2>` for his exact face" and
the model reads the face from the photo. No LoRA, no paste, no token budget.

### Why FP16 video

A 48 GB card holds the 26.62 GB UNET plus the text encoder and VAE with room for
activations. FP8 and GGUF exist for smaller cards; on 48 GB they only cost
quality.

### Getting the results back

The instance serves results over HTTP:

```bash
python scripts/pull_results.py --http http://<instance-ip>:8000
```

Or set `PULL_HOST=user@your-laptop` in `.env` and the instance scp's them.

---

## Environment

All configuration lives in `.env` (gitignored). Copy `.env.example` and fill it in.

| Variable | Required | Purpose |
|----------|----------|---------|
| `HF_TOKEN` | yes | Model downloads |
| `CIVITAI_API_KEY` | yes | Anatomy LoRA downloads |
| `WAN_PRECISION` | optional | `fp16` (default on 48 GB) |
| `LIPSYNC_ENGINE` | optional | `infinitetalk` (default), `s2v`, `off` |
| `PULL_HOST` | optional | `user@host` to scp results automatically |

---

## Pipeline

| Phase | What it does |
|-------|--------------|
| 1 | System packages, uv, aria2, ffmpeg |
| 2 | Python dependencies from the lockfile |
| 3 | Model downloads: Wan 2.2 FP16 pair + Qwen-Image 2.1 (63 GB) |
| 4 | LoRAs: trained character LoRAs + anatomy LoRAs |
| 5 | ComfyUI clone, model links, server start |
| 6 | Compile beat prompts from `docs/beats.yaml` |
| 7 | **Stills**: Qwen-Image 2.1 with photo references |
| 8 | **Video**: Wan 2.2 I2V 14B FP16, 70 clips |
| 9 | Narration (Chatterbox Hindi) + audio mux |
| 10 | Final cut + package |

### Models

| Model | Size | Purpose |
|-------|------|---------|
| `wan2.2_i2v_high_noise_14B_fp16` | 26.6 GB | Video, high-noise pass |
| `wan2.2_i2v_low_noise_14B_fp16` | 26.6 GB | Video, low-noise pass |
| `umt5_xxl_fp8_e4m3fn_scaled` | 6.3 GB | Wan text encoder |
| `wan_2.1_vae` | 0.2 GB | Wan VAE |
| `qwen_image_fp8_e4m3fn` | 19.0 GB | Stills diffusion |
| `qwen_2.5_vl_7b_fp8_scaled` | 8.7 GB | Stills text encoder |
| `qwen_image_vae` | 0.2 GB | Stills VAE |

Total: ~88 GB. The old script also staged FP8 and GGUF Wan variants that a
48 GB card never loads, plus two 1.17 GB realism LoRAs that fought the trained
identity LoRAs.

### LoRAs

| LoRA | Size | Role |
|------|------|------|
| `faiz_lora` | 435 MB | Faiz identity (trained, dim 64) |
| `juv_lora` | 435 MB | Juveriya identity (trained, dim 64) |
| `skin_morph_wan22_high/low` | 146 MB | Skin detail, default on every shot |
| `penis_insert_wan22_high/low` | 293 MB | Anatomy, per-shot |
| `vag_detail_wan22_high/low` | 293 MB | Anatomy, per-shot |
| `nsfw_general_wan22_high` | 585 MB | General NSFW, per-shot |
| `genitals_helper_wan22_high` | 293 MB | Anatomy helper, per-shot |

`realism_wan22_high/low` was removed: 1.17 GB per pass, duplicated what
`skin_morph` does at 146 MB, and being a default it loaded on every shot and
competed with the trained identity LoRAs for the face.

---

## Production Spec

From `docs/beats.yaml`:

- **70 clips** across 12 scenes
- **344 seconds** total
- **Resolutions**: 832x480 base, 1280x720 for face closeups
- **Sampler**: Euler / Simple, CFG 1
- **Hero beats**: c25, c33, c44, c61, c68 at 12 steps / CFG 4

---

## Validation

```bash
uv run --locked --inexact python scripts/validate.py --no-frames
uv run --locked --inexact python -m pytest tests/ -q
uv run --locked --inexact python scripts/check_credentials.py
uv run --locked --inexact python scripts/disk_budget.py
```

Current state:

```
validate.py --no-frames   -> OK 70 shots validated (0 warnings)
pytest tests/ -q          -> 261 passed, 20 subtests
```

---

## Requirements

- **Python 3.11+** with `uv`
- **NVIDIA GPU, 48 GB VRAM** (A6000 48GB, L40S, A100-40GB+)
- **Disk**: 140 GB free
- **ComfyUI** on `http://127.0.0.1:8188` (auto-started)
- **`HF_TOKEN`** with access to `Comfy-Org/Wan-2.2` and `Comfy-Org/Qwen-Image_ComfyUI`

---

## License

Private production pipeline. Not for distribution.
