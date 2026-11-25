# AIC Cloud Deployment Findings — Wan 2.2 Production Pipeline

Everything below was learned the hard way on a live AIC Cloud RTX A6000
instance (`gpu-868186.aiccloud.one`, 44 GB VRAM, 200 GB disk, Ubuntu 22.04,
driver 560.35.03 / CUDA 12.6). Every failure listed here actually happened and
cost real GPU time. Read this before deploying.

---

## 0. TL;DR — the seven things that broke everything

| # | Problem | Fix |
|---|---|---|
| 1 | `uv sync` picked Python 3.15, `onnxruntime-gpu` has no cp315 wheel | Pin `requires-python = ">=3.11,<3.15"` and run `uv sync --python 3.14` |
| 2 | `uv` installed CUDA 13 torch; AIC driver only supports 12.6 | Pin torch/torchvision to the `cu126` index in `pyproject.toml` |
| 3 | ComfyUI ran on the instance's system Python 3.10 | Give ComfyUI its **own** venv, never the pipeline venv |
| 4 | `uv sync` pruned ComfyUI's packages from the shared venv | Separate venvs (see #3) |
| 5 | Workflow used a Lightning recipe on non-Lightning weights → flat gray video | Use LightX2V 4-step LoRAs, or the Lightning-baked Q8 checkpoint |
| 6 | Render stage re-downloaded 57 GB into a nested path | Pass `--skip-download` to `run_production.py` |
| 7 | `uv run` re-synced on every call and re-downloaded 2.4 GiB of torch | `uv sync --inexact` once, then `UV_NO_SYNC=1` (see 2.6) |

---

## 1. Instance and access

### What to order

| Spec | Value | Why |
|---|---|---|
| GPU | RTX A6000 48 GB (reports 46068 MiB → 44 GB) | Fits one 27 GB fp16 UNET resident |
| Disk | **200 GB** | 57 GB fp16 models + 30 GB Q8 + ComfyUI + renders |
| Image | PyTorch | Ships CUDA driver; do not rely on its Python |
| RAM | 54 GB+ | Model staging and VAE decode |

**Do not use the A100 29 GB-disk listing.** The fp16 model pair alone is 53 GB.

### SSH

AIC assigns a **high port per instance** (40581 in this run), not 22.

```powershell
ssh -i "$env:USERPROFILE\.ssh\aic-gpu-gpu-wan22.pem" -p 40581 root@gpu-868186.aiccloud.one
```

Windows key permissions — `chmod` does not exist, use `icacls`:

```powershell
icacls "$env:USERPROFILE\.ssh\aic-gpu-gpu-wan22.pem" /inheritance:r /grant:r "$($env:USERNAME):(R)"
```

`scp` uses **capital `-P`** for the port.

### The minimized image is missing tools

`nano`, `jq`, `bc`, `ss`, `file` are all absent. Install before anything else:

```bash
apt-get update -qq && apt-get install -y -qq jq bc curl ca-certificates git aria2 ffmpeg libgl1 libglib2.0-0
```

**Do not try to edit `.env` with `nano`.** Copy it up with `scp` instead.

### Credentials: `instance.env` is a plaintext secret file

`instance.env` holds live tokens (HF, Civitai, Kaggle, AIC, Gemini, Numes,
Horde). It is gitignored and has never been committed, so nothing has leaked
to git — but it is plaintext on disk and must stay that way:

- Never `git add -f` it, and never paste its contents into a log, issue or
  chat. `scripts/check_credentials.py` prints PASS/FAIL only, never values.
- Keep it out of the repo directory when you can. `scp` it to the instance,
  use it, then delete it from the box.
- Rotate any key that has been shared in plaintext. A key pasted into a
  terminal, a screenshot or a support thread is compromised.

Verify every key against its live service before a run:

```bash
uv run --locked --inexact python scripts/check_credentials.py --env-file instance.env
```

Expect `8/8 checks passed`. A `FAIL` names the service, not the secret.

---

## 2. Python environment — the biggest time sink

### 2.1 Python 3.15 has no `onnxruntime-gpu` wheel

The instance's default `python3` was **3.15.2**. `onnxruntime-gpu==1.32.0`
ships wheels for cp311–cp314 only:

```
error: Distribution `onnxruntime-gpu==1.32.0` can't be installed because it
       doesn't have a source distribution or wheel for the current platform
hint: You're using CPython 3.15 (`cp315`), but `onnxruntime-gpu` only has
      wheels with the following Python ABI tags: cp311, cp312, cp313, cp313t, cp314, cp314t
```

**Fix** — cap the interpreter in `pyproject.toml`:

```toml
requires-python = ">=3.11,<3.15"
```

Then install and sync explicitly:

```bash
uv python install 3.14
uv sync --extra test --python 3.14
```

### 2.2 `uv` installs CUDA 13 torch; the driver is CUDA 12.6

PyPI's default `torch` wheel is built for CUDA 13. The AIC driver reports
12060 (12.6), so torch fails at import:

```
RuntimeError: The NVIDIA driver on your system is too old (found version 12060).
```

**Fix** — pin the CUDA 12.6 index in `pyproject.toml` so `uv sync` can never
silently replace the working build:

```toml
[tool.uv.sources]
torch = { index = "pytorch-cu126" }
torchvision = { index = "pytorch-cu126" }

[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true
```

Verify after every sync:

```bash
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expect: 2.14.1+cu126 True
```

### 2.3 `peft` is required by diffusers

`pipe.load_lora_weights()` raises:

```
ValueError: PEFT backend is required for `load_lora_weights()`.
```

**Fix** — add to `pyproject.toml` dependencies (not a manual `uv pip install`,
which `uv sync` will prune):

```toml
"peft>=0.11",
```

### 2.4 ComfyUI must have its own venv

Two failures came from sharing one venv:

- `uv sync` **prunes** ComfyUI's packages (`comfyui-frontend-package`,
  `comfy-kitchen`, `sqlalchemy`, …) because they are not in `pyproject.toml`.
- Installing ComfyUI's requirements **replaces** the pinned cu126 torch with
  the CUDA 13 build.

**Fix** — isolated environment:

```bash
cd /workspace/ComfyUI
uv venv --python 3.14 .venv
uv pip install --python .venv/bin/python \
    --index-url https://download.pytorch.org/whl/cu126 \
    torch==2.14.1+cu126 torchvision==0.29.1+cu126
uv pip install --python .venv/bin/python -r requirements.txt
```

Launch ComfyUI with that interpreter explicitly:

```bash
cd /workspace/ComfyUI
nohup .venv/bin/python main.py --listen 127.0.0.1 --port 8188 \
    --preview-method none --disable-auto-launch > /tmp/comfyui.log 2>&1 &
```

### 2.5 `cv2` is missing for VideoHelperSuite

```
ModuleNotFoundError: No module named 'cv2'
[WARNING] Cannot import ComfyUI-VideoHelperSuite module for custom nodes
```

Without it `VHS_VideoCombine` does not exist and every render fails with
`missing_node_type`. Install into the **ComfyUI** venv:

```bash
uv pip install --python /workspace/ComfyUI/.venv/bin/python \
    opencv-python-headless imageio-ffmpeg
```

### 2.6 `uv run` re-syncs the venv on every call — and can re-download torch

`uv run` performs an implicit `uv sync` before executing the command. Because
Chatterbox is installed with `uv pip install --no-deps` (see 2.3 and the
narration phase), it is **not** in `uv.lock`. An exact sync therefore treats
every Chatterbox dependency as extraneous:

```
$ uv sync --extra test --dry-run
Would uninstall 78 packages
```

On a cold cache that sync also re-downloads `torch` (2.4 GiB), `torchvision`
and `diffusers`. The command looks hung — CPU is idle and no Python child is
spawned — because it is network-bound, not computing.

**Fix** — sync once, inexactly, then stop re-syncing:

```bash
uv sync --extra test --inexact   # keep out-of-lock packages
export UV_NO_SYNC=1              # or pass --no-sync to each `uv run`
```

`setup_aic_fast.sh` does both. Scripts that must not silently re-resolve use
`uv run --locked --inexact`, which fails loudly if `uv.lock` is stale instead
of quietly rewriting the environment.

Verify the environment is settled:

```bash
uv sync --extra test --inexact --dry-run   # expect: "Would make no changes"
```

**Do not run two `uv` commands against the same venv concurrently.** Both try
to rewrite it and one dies with:

```
failed to remove directory .venv/Lib/site-packages/diffusers/utils/__pycache__:
Access is denied. (os error 5)
```

---

## 3. ComfyUI 0.46 node-name changes

The workflow generator was written for an older ComfyUI. Four renames broke
every submission with `HTTP 400 Bad Request`:

| Old | New | Notes |
|---|---|---|
| `UnetLoader` | `UNETLoader` | `weight_dtype` is now **required** |
| `Wan22ImageToVideo` | `WanImageToVideo` | same inputs |
| `umt5_xxl_fp8_e4m3fn.safetensors` | `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `_scaled` suffix |
| — | `ModelSamplingSD3` | **required** for Wan 2.2 flow matching |

The error is unhelpful — it names the node *title*, not the class:

```
invalid prompt: {'type': 'missing_node_type',
  'message': "Node 'HIGH - FP16' not found. The custom node may not be installed.",
  'extra_info': {'node_id': '1', 'class_type': 'UnetLoader'}}
```

**Always verify node names against the live server** before rendering:

```bash
curl -s http://127.0.0.1:8188/object_info | python3 -c "
import json,sys
d=json.load(sys.stdin)
for n in ['UNETLoader','WanImageToVideo','ModelSamplingSD3','VHS_VideoCombine']:
    print(n, n in d)
"
```

---

## 4. The quality bug — flat gray video

### Symptom

Frame 40 of a rendered clip measured:

```
sharpness : 42.4   (start frame: 132.9)
std       :  5.6   (start frame:  48.2)
uniques   :   37   (start frame:   239)
```

The image had collapsed to flat mush. Visually: "just blurring shit".

### Cause

The workflow ran **2 + 3 = 5 steps at CFG 1.0** — the *Lightning* recipe —
against the **base** `Comfy-Org/Wan-2.2` fp16 weights, which are **not**
Lightning-baked. Five steps on a non-Lightning model under-samples and
produces gray.

### Two valid fixes

**Option A — LightX2V 4-step LoRAs on the base model** (what was verified):

```bash
# from Comfy-Org/Wan-2.2
split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors
split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors
```

Graph: `UNETLoader → LoraLoaderModelOnly(1.0) → ModelSamplingSD3(shift 5.0) → KSamplerAdvanced`

- HIGH: steps 4, start 0, end 2, `return_with_leftover_noise: enable`
- LOW: steps 4, start 2, end 4, `return_with_leftover_noise: disable`
- CFG **1.0**, sampler `euler`, scheduler `simple`

**Measured result: 40 s per clip** (was 5.3 min at 20 steps).

**Option B — the Lightning-baked Q8 GGUF** (the project's declared target):

```
Wan2_2-I2V-A14B-NSFW_fast_move-HIGH_Q8(lightning_edition)V2.gguf   14.35 GB
Wan2_2-I2V-A14B-NSFW_fast_move-LOW_Q8lightning_edition)V2.gguf     14.35 GB
```

Civitai model 2053259, versions 2540892 / 2540896. SHA-256 verified against
`docs/civitai-wan22-model-lora-audit.md`:

```
HIGH DEB6DCF39FEFEF7850D1DD49A52B70A64C65B8D8E7B5F03BC1B6A541CF58AAE9
LOW  B5722E8E01A21F89A1C0B5E9DE9A3B907CE92324D1951D5CE70307956790DAAE
```

Load with `UnetLoaderGGUF` (requires the ComfyUI-GGUF custom node). Lightning
is baked in, so **no LightX2V LoRA** and no separate Lightning LoRA.

### The 58-version trap

Model 2053259 has **58 versions** and the naming encodes whether Lightning is
baked in. Getting this wrong reproduces the §4 gray-video bug exactly.

| Version family | IDs | Lightning | Recipe |
|---|---|---|---|
| NSFW FAST MOVE V2 Q8 | 2540892 / 2540896 | **baked in** | 2+3 steps, CFG 1, **no LoRA** |
| NSFW V2 Q8 | 2584698 / 2584707 | **baked in** | 2+3 steps, CFG 1, **no LoRA** |
| nolightning SVI cf Q8 | 2668710 / 2668712 | none | LightX2V LoRA **or** 20+ steps |
| nolightning SVI cf fp8 | 2609141 / 2609148 | none | LightX2V LoRA **or** 20+ steps |
| nolightning SVI fm fp8 | 2606405 / 2606408 | none | LightX2V LoRA **or** 20+ steps |

The author states it twice: *"This model already includes the Lightning LoRAs,
don't use them, or the quality will be degraded"* and *"For SVI models, you can
use Lightning LoRAs if you want faster video generation."*

`generate_workflow.py` encodes this as `LIGHTNING_BAKED = {"gguf_q8"}` and
`render.py` as `NEEDS_LIGHTX2V = {"cf"}`.

### Correction: the SVI models are not missing

An earlier reading of the huchukato workflow concluded that
`wan22EnhancedNSFWSVICamera_nolightningSVICfFp8H/L` did not exist. They do —
they are model 2053259 versions 2609141 / 2609148, whose real filenames are:

```
Wan2_2-I2V-A14B-HIGH_SVI_consistent_face_nsfw_fp8.safetensors
Wan2_2-I2V-A14B-LOW_SVI_consistent_face_nsfw_fp8.safetensors
```

The huchukato description simply never links to model 2053259. It is a
documentation gap, not a missing model.

### FM vs CF

| | Fast Move (FM) | Consistent Face (CF) |
|---|---|---|
| Motion | more, faster | less |
| Prompt / camera adherence | better | weaker |
| Image quality | good | slightly better |
| Lightning baked in | **yes** | **no** |

`build: fm` is the default. `build: cf` swaps to the Consistent Face pair and
chains the LightX2V LoRAs automatically, because that pair has no distillation.

### Do not

- Run 3 steps. The LightX2V LoRA is trained at 4; three is visibly mushier.
- Raise CFG above 1.0 to "compensate". It burns skin and roughly halves speed.
- Use `--lowvram` on a 44 GB card.
- Stack LightX2V on the Lightning-baked pair. It degrades quality.

---

## 5. Frame count and fps

The project contract is **16 fps**, the rate Wan 2.2 is trained at.
`WanImageToVideo` requires `(length - 1) % 4 == 0`, so frames are
`duration * fps + 1` snapped up to that grid: **81 frames for a 5 s clip, 65
for 4 s**.

The pipeline originally ran at 8 fps. Two things were wrong with that:

1. 8 fps reads as stop-motion on any camera move. Wan 2.2's own training rate
   is 16, so half the temporal detail the model can produce was being thrown
   away.
2. The lipsync pass has always rendered at 16 fps. Mixing 8 fps renders with
   16 fps lipsync output in one concat is a frame-rate mismatch, not just a
   look problem.

```python
# scripts/generate_workflow.py
FPS = 16
FRAMES_5S = 81
FRAMES_4S = 65

def frames_for(duration: float, fps: int = FPS) -> int:
    length = int(round(duration * fps)) + 1
    while (length - 1) % 4:
        length += 1
    return length
```

`render.py` and `lipsync.py` both call `frames_for`, so the two stages can
never drift apart again. `mux_audio.py --fps` normalises every clip to one
rate before concat.

**Cost:** 81 frames is roughly double the sampling of 41. Budget for it.

---
## 5b. Negative prompts are dead code at CFG 1.0
`negatives/artifact_neg.txt` is referenced by every prompt JSON and read by
`render.py`, but it **has no effect on any render**.

At CFG 1.0 the sampler never evaluates the negative branch — it is
mathematically ignored. The author confirms: *"The model must be used with CFG
set to 1, so negative prompts do not work by default."*

### Enabling them

KJNodes ships `WanVideoNAG`, which patches cross-attention with the negative
conditioning. It sits between `ModelSamplingSD3` and the sampler:

```
UNETLoader -> [LightX2V] -> ModelSamplingSD3 -> WanVideoNAG -> KSamplerAdvanced
```

```bash
python scripts/generate_workflow.py --precision gguf_q8 --output workflows/nag.json --nag
```

Defaults: `nag_scale 11.0`, `nag_alpha 0.25`, `nag_tau 2.5`.

**Cost: roughly doubles generation time.** Only enable it when a specific
artifact must be suppressed. Otherwise delete `negatives/artifact_neg.txt` and
stop pretending it is applied.

In the production path:

```bash
python scripts/run_production.py --nag --nag-scale 11.0 --nag-alpha 0.25 --nag-tau 2.5
```

---

## 5c. The GGUF checkpoints are not on Hugging Face

`auto_model_selector.py` only knows about Hugging Face, and
`Comfy-Org/Wan_2.2_ComfyUI_Repackaged` publishes **no GGUF files at all**. The
workflows load the Civitai 2053259 pair, so a fresh instance used to fail on
the first clip with a missing model.

`scripts/download_wan22_gguf.py` stages all four checkpoints plus the LightX2V
LoRAs the CF pair needs:

```bash
python scripts/download_wan22_gguf.py --dest /workspace/models
python scripts/download_wan22_gguf.py --check          # status only
python scripts/download_wan22_gguf.py --only fm        # skip the CF pair
```

`setup_aic_fast.sh` calls it automatically when `PRECISION=gguf_q8`.

| Build | Version | File | Size |
|---|---|---|---|
| fm/high | 2540892 | `..._fast_move-HIGH_Q8(lightning_edition)V2.gguf` | 14.35 GB |
| fm/low | 2540896 | `..._fast_move-LOW_Q8lightning_edition)V2.gguf` | 14.35 GB |
| cf/high | 2668710 | `Wan2_2-I2V-A14B-HIGH_consistent_face_Q8.gguf` | 14.35 GB |
| cf/low | 2668712 | `Wan2_2-I2V-A14B-LOW_consistent_face_Q8.gguf` | 14.35 GB |

Plus `wan2.2_i2v_lightx2v_4steps_lora_v1_{high,low}_noise.safetensors`
(1.23 GB each) from `Comfy-Org/Wan_2.2_ComfyUI_Repackaged/split_files/loras/`.

The FM pair is SHA-256 pinned; the CF pair is not (Civitai publishes no hash
for those versions), so it is validated by GGUF magic bytes only.

---

## 5d. Mixed resolutions break the final concat

The spec renders `build: cf` shots at `res_face` (1280x720) and everything else
at `res_base` (832x480). The ffmpeg **concat demuxer requires one resolution**,
so the final cut failed.

`mux_audio.py` now normalises every clip to `res_base` before concat:

```bash
python scripts/mux_audio.py --width 832 --height 480   # explicit override
python scripts/mux_audio.py                            # defaults to meta.res_base
```

Scaling uses `force_original_aspect_ratio=decrease` plus `pad`, so a 1280x720
clip is letterboxed rather than stretched.

---

## 5e. Lipsync must skip narration-only beats

19 of the 70 beats carry narration but have **no speaker on screen** — c48 is
*"static close-up: the buzzer box rattling on the wall"*. Driving a mouth from
a voice-over animates whatever face happens to be in frame.

`lipsync.py` now skips any beat whose `say` is empty. Those beats are still
muxed with their narration; they are just never lipsynced.

```
lipsync engine: infinitetalk
clips with narration: 25
skipped (narration only, no speaker on screen): 19
```

---

## 5f. TTS voice selection follows the script, not the `language` field

The spec's `language` field is unreliable: c18 is tagged `hi` but its `say` is
English. `generate_narration.py` therefore classifies each turn by the script
of the text itself.

| Script | Faiz | Juveriya |
|---|---|---|
| Latin (incl. romanized Hindi/Urdu) | `en-IN-PrabhatNeural` | `en-IN-NeerjaNeural` |
| Devanagari | `hi-IN-MadhurNeural` | `hi-IN-SwaraNeural` |
| Arabic (Urdu) | `ur-PK-AsadNeural` | `ur-PK-UzmaNeural` |

Romanized Hindi/Urdu stays on the Indian-English voices, which pronounce it
correctly. Only real Devanagari/Arabic script switches to a native voice.

---

## 5g. Ambient bed

`mux_audio.py --music <file> --music-db -22` mixes a looping bed under the
narration. MMAudio is **not** wired into the render graph; it is a
video-conditioned foley model and cannot produce scripted dialogue. Use it (or
any music file) only as a bed under the TTS track.

---

## 5h. Realism LoRAs are defaults, not per-shot

`workflows/loras.json` entries flagged `"default": true` are injected into
every shot by `render.inject_loras`, ahead of the shot's own list. The spec
does not have to repeat them on 70 beats.

| Key | Weight | What it fixes |
|---|---|---|
| `realism` | 0.7 | Photoreal skin and lens response |
| `skin_morph` | 0.55 | Pores, subsurface scatter, sweat sheen |

A shot opts out with `no_default_loras: true` in `beats.yaml`. `build_prompts`
mirrors the same rule so the compiled prompt carries the trigger words for
exactly the weights that will load.

`handheld` and `lens_flare` are **not** defaults: handheld fights static cam
lines, and lens flare only helps shots with a practical light in frame.

---

## 5i. Two-speaker lipsync

11 beats have both characters speaking in one frame. Driving both mouths from
one mixed track makes both faces follow both voices.

`generate_narration.py` now writes a per-speaker stem for those beats
(`audio/<id>_<speaker>.wav`) and the index stores
`{"mix": ..., "stems": {...}}`. `lipsync.py` builds a soft mask per speaker
from the start frame with InsightFace and runs InfiniteTalk in `two_speakers`
mode. Masks are uploaded as PNGs and read through `LoadImage` slot 1 (MASK).

When detection finds fewer faces than speakers it falls back to a left/right
split, so the pass never fails on a wide shot.

---

## 5j. Realism grade

`mux_audio.py --grade` adds, in order: 1px lateral chromatic aberration, a
gentle S-curve with warm highlights, highlight-only halation, luma-weighted
grain, and a light unsharp. Grain is applied **after** scaling so it stays a
constant size instead of being resampled.

The grade forces a re-encode, so it is off by default. Without `--grade` the
mux keeps `-c:v copy`.

---
## 6. VRAM behaviour

### `--highvram` OOMs on the fp16 pair

With `--highvram --disable-async-offload --disable-dynamic-vram`, ComfyUI kept
**both** 27 GB UNETs resident:

```
Currently allocated : 43.95 GiB
Requested           : 135.00 MiB
[ERROR] Got an OOM, unloading all loaded models.
```

**Fix** — let ComfyUI swap the experts:

```bash
.venv/bin/python main.py --listen 127.0.0.1 --port 8188 \
    --reserve-vram 1.5 --preview-method none --disable-auto-launch
```

The Q8 pair (14.35 GB each, 28.7 GB total) fits both resident in 44 GB, so
`--highvram` is safe **only** with Q8.

### Step timing reference (A6000, 832×480, 81 frames)

| Config | s/step | Per clip |
|---|---|---|
| fp16, 20 steps, dynamic VRAM | ~13.6 | 5.3 min |
| fp16 + LightX2V, 4 steps | — | **40 s** |

Those figures are for 41 frames. At 81 frames the sampling roughly doubles:
budget **~80 s per clip** on the 4-step path, so 70 clips is ~95 min of pure
render rather than ~47.

---

## 7. The duplicate-download trap

`run_production.py` re-downloaded 57 GB into a **nested** path
(`models/wan22_fp16/wan22_fp16/…`) because `MODEL_DIR` was reassigned from the
model-selector JSON while the flat files already existed at `models/`.

Symptom: the render appears hung, GPU idle, `/tmp/render.log` empty, and
`/proc/<pid>/fd` shows a `.part` file being written.

**Fix** — always pass `--skip-download` when models are already staged:

```bash
.venv/bin/python scripts/run_production.py \
    --spec docs/beats.yaml --prompts prompts --startframes startframes \
    --renders renders --output output --server http://127.0.0.1:8188 \
    --force-precision fp16 \
    --skip-download --skip-loras --skip-stills --skip-face-swap --skip-assemble
```

---

## 8. Missing prerequisites that stop the run

### 8.1 `prompts/` does not exist on a fresh clone

`generate_stills.py` found zero prompts and produced zero frames:

```
[20:34:07] ✗ No start frames were generated - the render cannot proceed.
```

`build_prompts.py` must run **before** stills, and it needs `--no-frames`
because the frames it would validate are produced afterwards:

```bash
uv run python scripts/build_prompts.py --no-frames
```

### 8.2 InsightFace swapper model

```
✗ Swapper model missing: /workspace/models/insightface/inswapper_128.onnx
```

```bash
mkdir -p /workspace/models/insightface
curl -sL -o /workspace/models/insightface/inswapper_128.onnx \
  "https://huggingface.co/ezioruan/inswapper_128.onnx/resolve/main/inswapper_128.onnx"
```

### 8.3 `repair_startframes.py` crashed

It did not forward `--sdxl-model`, so `generate_stills.py` fell back to
`models/sdxl_base.safetensors` and diffusers raised:

```
ValueError: Invalid `pretrained_model_name_or_path` provided.
```

Fixed by adding `--sdxl-model` to the subprocess command and a matching
argument defaulting to the Juggernaut path.

### 8.4 The reference-file warning is a false alarm

```
⚠ references/faiz_ref.jpg and/or references/juv_ref.jpg missing.
```

The check hardcoded `.jpg` while the repo ships `.png`. The generator resolves
any extension, so identity enforcement was working all along. Fixed to accept
`png jpg jpeg webp`.

---

## 9. Shell script line endings

`scp` from Windows transfers **CRLF**, which breaks bash:

```
scripts/setup_aic_fast.sh: line 10: $'\r': command not found
scripts/setup_aic_fast.sh: line 11: set: pipefail: invalid option name
```

**Fix** — pin LF in `.gitattributes`:

```
*.sh text eol=lf
*.bash text eol=lf
```

And after any `scp` of a `.sh` file, normalize on the instance:

```bash
python3 -c "
import sys
p = sys.argv[1]
d = open(p,'rb').read().replace(b'\r\n', b'\n').replace(b'\r', b'\n')
open(p,'wb').write(d)
" scripts/setup_aic_fast.sh
bash -n scripts/setup_aic_fast.sh && echo "SYNTAX OK"
```

---

## 10. Accessing ComfyUI from the desktop

ComfyUI binds to `127.0.0.1` on the instance. Use an SSH tunnel:

```powershell
ssh -i "$env:USERPROFILE\.ssh\aic-gpu-gpu-wan22.pem" -p 40581 `
    -L 8188:127.0.0.1:8188 root@gpu-868186.aiccloud.one
```

Then open **http://127.0.0.1:8188** on the desktop.

**The tunnel dies silently.** If the page stops loading, the tunnel is gone,
not ComfyUI. Re-run the command. Add keepalives:

```powershell
ssh -i "$env:USERPROFILE\.ssh\aic-gpu-gpu-wan22.pem" -p 40581 `
    -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes `
    -N -L 8188:127.0.0.1:8188 root@gpu-868186.aiccloud.one
```

Verify from the desktop before blaming the server:

```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:8188/object_info" -UseBasicParsing -TimeoutSec 10
```

Do **not** expose 8188 publicly without authentication.

---

## 11. Verification sequence

Run these in order. Each one catches a different class of failure.

```bash
# 1. GPU visible
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 2. Python + CUDA correct
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# expect: 2.14.1+cu126 True

# 3. Precision selection
uv run python scripts/auto_model_selector.py

# 4. Disk budget
uv run python scripts/disk_budget.py

# 5. Credentials (Kaggle CLI failure is non-blocking if LoRAs are staged)
uv run python scripts/check_credentials.py

# 6. Spec validates
uv run python scripts/validate.py --no-frames
# expect: OK 70 shots validated (0 warning(s))

# 7. Prompts compiled
uv run python scripts/build_prompts.py --no-frames
# expect: wrote 70 prompts

# 8. ComfyUI up and nodes present
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:8188/object_info
curl -s http://127.0.0.1:8188/object_info | python3 -c "
import json,sys
d=json.load(sys.stdin)
need=['UNETLoader','WanImageToVideo','ModelSamplingSD3','VHS_VideoCombine','LoraLoaderModelOnly']
print({n: n in d for n in need})
"

# 9. Single-clip smoke test BEFORE the batch
.venv/bin/python scripts/run_production.py \
    --spec docs/beats.yaml --prompts prompts --startframes startframes \
    --renders renders --output output --server http://127.0.0.1:8188 \
    --force-precision fp16 --skip-download --skip-loras --skip-stills \
    --skip-face-swap --skip-assemble --from-clip c01 --to-clip c01
```

**Always run step 9.** A single clip costs ~₹2 and catches every
configuration error before a 70-clip batch.

---

## 12. Measuring output quality

Do not judge by eye alone. Extract frames and measure:

```bash
ffmpeg -y -v error -i renders/c01.mp4 -vf "select=eq(n\,40)" -vframes 1 /tmp/f40.png

/workspace/ComfyUI/.venv/bin/python - <<'PY'
import cv2, numpy as np
for label, path in [("start", "startframes/c01.png"), ("frame40", "/tmp/f40.png")]:
    g = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2GRAY)
    print(f"{label}: sharp={cv2.Laplacian(g, cv2.CV_64F).var():.1f} "
          f"std={g.std():.1f} uniques={len(np.unique(g))}")
PY
```

Healthy output keeps `std` within ~30% of the start frame and `uniques` above
~150. `std < 10` or `uniques < 60` means the render collapsed.

---

## 13. Cost and timing reference

| Item | Value |
|---|---|
| A6000 rate | ₹81–95/hr |
| Full 70-clip render (40 s/clip) | ~50 min |
| Single-clip smoke test | ~₹2 |
| Full run cost | ~₹120–170 |

**Destroy the instance when done** — Stop pauses compute billing but disk
billing continues.

---

## 14. Pre-flight checklist

- [ ] Instance: A6000 48 GB, **200 GB disk**, PyTorch image
- [ ] SSH key permissions fixed with `icacls`
- [ ] `apt-get install jq bc curl ca-certificates git aria2 ffmpeg libgl1 libglib2.0-0`
- [ ] `.env` copied with `scp` (never `nano`)
- [ ] `pyproject.toml`: `requires-python = ">=3.11,<3.15"`, cu126 torch index, `peft`
- [ ] `uv python install 3.14 && uv sync --extra test --python 3.14`
- [ ] `torch.cuda.is_available()` is `True`
- [ ] ComfyUI has its **own** `.venv` with cu126 torch + `opencv-python-headless`
- [ ] ComfyUI-GGUF and VideoHelperSuite cloned
- [ ] Models staged flat in `/workspace/models/` and symlinked into ComfyUI
- [ ] `prompts/` compiled with `build_prompts.py --no-frames`
- [ ] `inswapper_128.onnx` present
- [ ] `validate.py --no-frames` exits 0
- [ ] ComfyUI `/object_info` returns 200 and lists all required nodes
- [ ] `workflows/base_i2v_gguf_q8.json` loads the `fast_move` Q8 pair with **no** LightX2V LoRA
- [ ] `workflows/base_i2v_gguf_q8.json` schedules **2+3** (HIGH 0→2, LOW 2→5)
- [ ] `workflows/base_i2v_gguf_q8.json` uses `umt5_xxl_fp8_e4m3fn_scaled.safetensors`
- [ ] `WanImageToVideo.length` is 81 and `VHS_VideoCombine.frame_rate` is 16
- [ ] Negative prompt decision made: `--nag` enabled, or `negatives/artifact_neg.txt` deleted
- [ ] `python scripts/download_wan22_gguf.py --check` reports 4/4 checkpoints ready
- [ ] LightX2V LoRAs staged in `models/loras/` (CF pair needs them)
- [ ] `lipsync.py --dry-run` reports 25 lipsynced / 19 skipped
- [ ] **Single-clip smoke test passes and measures healthy**
- [ ] Only then start the batch with `--skip-download`

---

## 15. Failure → cause quick table

| Symptom | Cause | Fix |
|---|---|---|
| `onnxruntime-gpu` no wheel | Python 3.15 | `--python 3.14`, cap `<3.15` |
| `NVIDIA driver too old (12060)` | CUDA 13 torch | pin cu126 index |
| `PEFT backend is required` | missing `peft` | add to `pyproject.toml` |
| `No module named 'cv2'` | VHS import fails | install `opencv-python-headless` |
| `missing_node_type` HTTP 400 | ComfyUI 0.46 renames | `UNETLoader`, `WanImageToVideo` |
| Flat gray video | Lightning recipe on base weights | LightX2V LoRA or Q8 Lightning |
| OOM at 43.95 GiB | `--highvram` with fp16 pair | drop `--highvram`, `--reserve-vram 1.5` |
| Render hangs, GPU idle | re-downloading 57 GB | `--skip-download` |
| `No start frames were generated` | `prompts/` missing | `build_prompts.py --no-frames` |
| `Swapper model missing` | no inswapper | download to `models/insightface/` |
| `$'\r': command not found` | CRLF from scp | `.gitattributes` + normalize |
| ComfyUI page won't load | SSH tunnel died | re-run tunnel with keepalives |
| `Chain broken by cNN` | one clip failed | fix the first failure, not the chain |
| LightX2V LoRA on Lightning-baked Q8 | wrong version family | use 2540892/2540896 as-is, no LoRA |
| Negative prompt has no effect | CFG 1.0 ignores the negative branch | add `--nag`, or delete the file |
| `build: cf` renders on FM | field was parsed but never applied | fixed: `render.apply_build` |
| Stale workflow file wins | `run_production.py` skipped regeneration | fixed: always regenerates |
| First clip: model not found | GGUF pair is Civitai-only, HF has none | `download_wan22_gguf.py` |
| CF shots render flat gray | CF pair has no Lightning baked in | `render.apply_build` adds LightX2V |
| Final concat fails | 832x480 and 1280x720 mixed | `mux_audio` normalises to res_base |
| Mouth moves on a buzzer box | lipsync ran on narration-only beats | skip when `say` is empty |
| Hindi line read in English | voice picked from `language` field | pick from text script |
