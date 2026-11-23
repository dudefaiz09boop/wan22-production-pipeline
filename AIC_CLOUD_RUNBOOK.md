# AIC Cloud — End-to-End Runbook

Everything below is verified against this repo. Follow it top to bottom.

---

## 0. What you need before starting

| Item | Where | Status |
|------|-------|--------|
| AIC Cloud account + funded wallet | https://aiccloud.in/ | you have it |
| SSH key pair | `~/.ssh/id_rsa` + `.pub` | generated, see §1 |
| `HF_TOKEN` | `.env` | ✅ verified |
| `CIVITAI_API_KEY` | `.env` | ✅ verified |
| `AIC_API_KEY` | `.env` | ✅ verified (wallet read) |
| Kaggle CLI auth | `~/.kaggle/credentials.json` | ✅ verified |
| Trained LoRAs | `loras/faiz_lora.safetensors`, `loras/juv_lora.safetensors` | ✅ 218 MB each |
| Anatomy LoRAs | `loras/penis_insert_*`, `loras/vag_detail_*` | ✅ 293 MB each |
| Reference faces | `references/faiz_ref.png`, `references/juv_ref.png` | ✅ from `refs/02`, `refs/04` |

Verify all of it in one command:

```powershell
uv run python scripts/check_credentials.py
uv run python scripts/report_spec_usage.py
```

Expected: `7/8 checks passed` (only `OPENAI_API_KEY` fails, and it is unused),
and `all spec LoRAs staged`.

---

## 1. Add your SSH key to AIC Cloud

Your public key:

```powershell
type $env:USERPROFILE\.ssh\id_rsa.pub
```

Paste the whole line into the **SSH public key** field on the launch page.
AIC Cloud saves it in the browser, so you only do this once.

---

## 2. Pick the GPU

From the live AIC Cloud list, ranked for this pipeline:

| GPU | VRAM | Price | Precision | Verdict |
|-----|------|-------|-----------|---------|
| **RTX A6000** | 48 GB | ₹72.40/hr | **FP16** | **Best value for full quality** |
| RTX 6000 Ada | 48 GB | ₹120.40/hr | FP16 | Faster, 2× the price |
| RTX PRO 5000 | 48 GB | ₹120.67/hr | FP16 | Same tier as 6000 Ada |
| RTX 4090 | 48 GB | ₹73.60/hr | FP16 | Good, but only 267 GB max disk |
| RTX 3090 | 24 GB | ₹37.97/hr | FP8 | Cheapest, near-FP16 quality |
| A100 SXM4 | 80 GB | ₹94.00/hr | FP16 | **Avoid: only 29 GB disk** |
| H100 SXM | 80 GB | ₹408.20/hr | FP16 | Overkill |

**Recommendation: RTX A6000 48 GB at ₹72.40/hr.**

Why not the A100? Its host allows only **29 GB** of disk. The FP16 model pair
alone is 57 GB, so it cannot hold the weights. Disk is the binding constraint,
not VRAM.

### Disk sizing

| Precision | Models | + ComfyUI + outputs | Minimum disk |
|-----------|--------|---------------------|--------------|
| FP16 | 57 GB | ~25 GB | **100 GB** |
| FP8 | 29 GB | ~25 GB | **75 GB** |
| GGUF/FP8 fallback | 29 GB | ~25 GB | **75 GB** |

Set **Disk Size = 150 GB** on the A6000 (max 181 GB). Storage bills 24×7 even
when the GPU is stopped, so destroy the instance when you are done.

---

## 3. Launch

1. Select **RTX A6000**.
2. **Instance Name**: `gpu-wan22`
3. **Disk Size**: `150`
4. **SSH public key**: paste from §1
5. **Image**: `PyTorch`
6. Click **Deploy**.

The first hour is charged immediately and is non-refundable.

**Wait 90–120 seconds** before your first SSH attempt. "Connection refused" in
the first minute is normal.

---

## 4. Connect

```powershell
ssh root@<instance-ip>
```

Find the IP in the AIC Cloud dashboard.

---

## 5. Get the code onto the instance

```bash
cd /workspace
git clone https://github.com/dudefaiz09boop/wan22-production-pipeline.git
cd wan22-production-pipeline
```

---

## 6. Create `.env` on the instance

The repo ships `.env.example`; `.env` is gitignored and never committed.

```bash
cp .env.example .env
nano .env
```

Fill in:

```ini
HF_TOKEN=hf_...
CIVITAI_API_KEY=...
KAGGLE_USERNAME=Sofiya009
KAGGLE_KEY=            # leave empty; the CLI uses its OAuth file
AIC_API_BASE=https://api.aiccloud.in
AIC_API_KEY=...
WORKSPACE=/workspace
MODEL_DIR=/workspace/models
OUTPUT_DIR=/workspace/output
COMFY_URL=http://127.0.0.1:8188
COMFY_PORT=8188
WAN_PRECISION=auto
PARALLEL_DOWNLOADS=4
LIPSYNC_ENGINE=s2v        # s2v | infinitetalk | off
LIPSYNC_STEPS=6
REPAIR_FRAMES=true
PULL_HOST=             # optional: user@your-laptop
DOWNLOAD_PORT=8000
AUTO_SHUTDOWN=false
```

`WAN_PRECISION=auto` picks FP16 on the A6000 automatically.

---

## 7. Stage the trained LoRAs

The two character LoRAs are already trained and validated. Copy them up:

```powershell
# from your laptop
scp "C:\Users\FAIZ SAYYED\projects\wan22-production-pipeline\loras\faiz_lora.safetensors" root@<ip>:/workspace/wan22-production-pipeline/loras/
scp "C:\Users\FAIZ SAYYED\projects\wan22-production-pipeline\loras\juv_lora.safetensors"  root@<ip>:/workspace/wan22-production-pipeline/loras/
```

`download_loras.py` treats a locally staged LoRA as authoritative, so these are
never retrained or re-fetched.

The anatomy LoRAs download automatically in step 8.

---

## 8. Run the pipeline

```bash
cd /workspace/wan22-production-pipeline
bash scripts/setup_aic_fast.sh
```

### What each phase does

| Phase | Action | Time (A6000) |
|-------|--------|--------------|
| Pre-flight | GPU detect, precision select, disk check | ~5 s |
| 1 | apt, uv, aria2, ffmpeg | ~60 s |
| 2 | `uv sync` | ~60 s |
| 3 | Parallel model download (57 GB FP16) | ~10–15 min |
| 4a | Anatomy + quality LoRAs from Civitai (6 files, ~2 GB) | ~3 min |
| 4b | Trained LoRAs (local copy) | ~2 s |
| 5 | Validate spec, generate workflow | ~10 s |
| 6 | Clone ComfyUI, link models, start server | ~3–5 min |
| 7 | SDXL stills + InsightFace swap (70 frames, 2 candidates) | ~15–20 min |
| 7b | Frame quality audit + repair of failing frames | ~5–10 min |
| 8 | **WAN 2.2 I2V render, 70 clips** | **~35–50 min** |
| 9 | Face swap every 3rd frame | ~5–8 min |
| 9b | **Lipsync (S2V, 44 narrated clips)** | **~15–25 min** |
| 10a | Narration (Edge TTS, 44 speaking clips) | ~3–5 min |
| 10b | Audio mux + final cut with continuous audio | ~5–8 min |
| 10c | Package deliverable | ~2 min |
| 11 | Publish results for download | ~5 s |
| **Total** | | **~100–140 min** |

Cost at ₹72.40/hr: **₹120–170**.

### Hard gates

The script stops rather than producing a broken video if:

- `HF_TOKEN` is missing
- no GPU is present
- anatomy LoRAs cannot be downloaded
- zero LoRAs are visible to ComfyUI
- the SDXL base model is missing
- zero start frames are generated
- `final.mp4` is not produced

### Audio and lipsync

The deliverable is **not silent**. Phase 10 does three things:

1. **Narration** — `generate_narration.py` reads the authored narration already
   compiled into `prompts/*.json` and synthesises it with Edge TTS. Voices are
   speaker-aware: Faiz → `en-IN-PrabhatNeural`, Juveriya → `en-IN-NeerjaNeural`,
   Ammi → `hi-IN-SwaraNeural`. Multi-turn dialogue is concatenated and each
   track is padded or trimmed to exactly its clip duration.
2. **Mux** — `mux_audio.py` attaches each clip's narration, gives silent clips a
   silent track of the right length, then concatenates everything into one MP4
   with a single continuous AAC stream.
3. **Music (optional)** — set `MUSIC_BED=/path/track.mp3` and `MUSIC_DB=-22` in
   `.env` to mix a bed under the narration.

Verified on a two-clip test: `final.mp4` 10.1 s, `mean_volume -22.0 dB`,
`max_volume -5.1 dB` — real audio, not silence.

**On lipsync:** implemented. Phase 9b re-drives the mouth from the narration
audio using Wan 2.2 S2V (default) or Wan 2.1 InfiniteTalk Multi.

| Engine | Model | Speakers | Use when |
|--------|-------|----------|----------|
| `s2v` (default) | `wan2.2_s2v_14B_fp8_scaled` | 1 audio stream | one person speaking |
| `infinitetalk` | `Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ` | 2 streams + masks | both characters speak in frame |

Set `LIPSYNC_ENGINE=infinitetalk` in `.env` for two-speaker clips, or
`LIPSYNC_ENGINE=off` to skip the pass. `LIPSYNC_STEPS` defaults to 6.

The node chain is verified against ComfyUI core, not guessed:

```
AudioEncoderLoader -> AudioEncoderEncode -> WanSoundImageToVideo
AudioEncoderLoader -> AudioEncoderEncode -> WanInfiniteTalkToVideo
```

Only clips with narration are re-driven (44 of 70); the rest pass through
untouched, and a single clip failure never aborts the run.

### Start frame quality

Phase 7b audits every frame for blur, darkness, missing faces and tiny faces,
then regenerates only the failures with more candidates. On the current frames:

```
total 70  ok 40  problems 30
no face detected (15): c31, c39, c40, c45, c46, c48, c50, c51, c52, c53, c54, c55, c65, c67, c68
face too small (15):   c01, c02, c03, c07, c08, c10, c11, c13, c14, c17, c18, c20, c21, c26, c38
blurry (5):            c45, c48, c52, c53, c54
too dark (1):          c20
```

Set `REPAIR_FRAMES=false` in `.env` to skip the repair pass.

---

## 9. Get the videos back

AIC Cloud GPU instances have **no GPU API**, so the instance cannot push to
your laptop by itself. Two options:

### Option A — HTTP (default)

The script starts a download server and prints:

```
curl -O http://<ip>:8000/deliverable.zip
curl -O http://<ip>:8000/final.mp4
```

From your laptop:

```powershell
python scripts/pull_results.py --http http://<ip>:8000
```

### Option B — scp push

Set `PULL_HOST=user@your-laptop` in `.env` before running. The instance scp's
`deliverable.zip` and `final.mp4` to you at the end.

### Option C — manual

```powershell
python scripts/pull_results.py --host root@<ip>
```

**Download before stopping the instance.** The HTTP server dies with it.

---

## 10. Stop billing

```bash
# on the instance
shutdown -h now
```

Then in the AIC Cloud dashboard: **Stop** pauses compute billing, **Destroy**
also stops the 24×7 disk charge. Destroy when you are finished — the instance
is ephemeral and the host may go away without warning.

---

## 11. Verify the output

```powershell
# on your laptop, after pulling
ffprobe -v error -show_entries format=duration -of csv=p=0 renders_local\final.mp4
```

Expected duration: **344 seconds** (70 clips: 64 × 5 s + 6 × 4 s).

Check the production log for per-clip timings:

```powershell
Get-Content renders_local\logs\production_*.json | ConvertFrom-Json | Select-Object precision, total_clips, successful, failed, estimated_cost_usd
```

---

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `connection refused` on SSH | host still booting | wait 90–120 s |
| `Anatomy LoRAs are missing` | Civitai 403 | confirm `CIVITAI_API_KEY` in `.env` |
| `No LoRAs visible to ComfyUI` | symlink step failed | check `$MODEL_CACHE/loras` has files |
| `SDXL base model missing` | download failed | re-run; it resumes |
| `No start frames were generated` | SDXL OOM | lower resolution in `generate_stills.py` |
| ComfyUI never ready | custom node clone failed | `tail -50 /tmp/comfyui.log` |
| Render fails on a LoRA shot | LoRA filename mismatch | `ls /workspace/ComfyUI/models/loras` |
| `final.mp4` missing | assembly found no clips | check `renders/` has `c01.mp4`…`c70.mp4` |
| Final video has no sound | narration or mux failed | check `audio/index.json` and the phase 10 log |
| Faces look wrong in output | LoRA weak or swap skipped | run `scripts/verify_loras.py --push` |
| Disk full | 57 GB models + outputs | use ≥150 GB disk |

---

## Optimizations already applied

| Change | Effect |
|--------|--------|
| Parallel aria2 model download (16 connections/file) | 57 GB in ~10–15 min instead of ~40 |
| ComfyUI starts during model download | zero wait when models finish |
| `uv sync` with lockfile | second run <10 s for deps |
| Per-shot resolution (34 shots at 1280×720) | face closeups keep detail instead of being downscaled |
| 2 candidates per still, early exit at gate | best identity without 4× the SDXL time |
| Face swap every 3rd frame | 3× faster than every frame, visually identical |
| Narration fitted to clip duration | no drift in the final mux |
| Resume: existing LoRAs and frames are reused | re-runs skip completed work |

### Further speed options (quality-neutral)

| Option | Saving | Trade-off |
|--------|--------|-----------|
| `--candidates 1` in phase 7 | ~40% of stills time | fewer identity retries |
| `--swap-every 5` in phase 9 | ~40% of swap time | slightly more flicker on fast motion |
| `--from-clip c01 --to-clip c10` for a test pass | ~85% of render time | partial output |
| RTX 3090 + FP8 instead of A6000 + FP16 | ₹72 → ₹38/hr | near-FP16, not identical |

---

## Cost control

| Lever | Saving |
|-------|--------|
| RTX 3090 + FP8 instead of A6000 + FP16 | ₹72 → ₹38/hr, ~₹45 total |
| `--from-clip c01 --to-clip c10` for a test pass | ~85% of render time |
| `AUTO_SHUTDOWN=true` | stops the instance 60 s after finishing |
| Destroy (not Stop) when done | ends the 24×7 disk charge |

A single-clip smoke test before a full run costs about ₹2 and catches most
configuration problems.
