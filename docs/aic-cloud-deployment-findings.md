# AIC Cloud Deployment Findings — Wan 2.2 Production Pipeline

Everything below was learned the hard way on a live AIC Cloud RTX A6000
instance (`gpu-868186.aiccloud.one`, 44 GB VRAM, 200 GB disk, Ubuntu 22.04,
driver 560.35.03 / CUDA 12.6). Every failure listed here actually happened and
cost real GPU time. Read this before deploying.

---

## 0. TL;DR — the six things that broke everything

| # | Problem | Fix |
|---|---|---|
| 1 | `uv sync` picked Python 3.15, `onnxruntime-gpu` has no cp315 wheel | Pin `requires-python = ">=3.11,<3.15"` and run `uv sync --python 3.14` |
| 2 | `uv` installed CUDA 13 torch; AIC driver only supports 12.6 | Pin torch/torchvision to the `cu126` index in `pyproject.toml` |
| 3 | ComfyUI ran on the instance's system Python 3.10 | Give ComfyUI its **own** venv, never the pipeline venv |
| 4 | `uv sync` pruned ComfyUI's packages from the shared venv | Separate venvs (see #3) |
| 5 | Workflow used a Lightning recipe on non-Lightning weights → flat gray video | Use LightX2V 4-step LoRAs, or the Lightning-baked Q8 checkpoint |
| 6 | Render stage re-downloaded 57 GB into a nested path | Pass `--skip-download` to `run_production.py` |

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

### Do not

- Run 3 steps. The LightX2V LoRA is trained at 4; three is visibly mushier.
- Raise CFG above 1.0 to "compensate". It burns skin and roughly halves speed.
- Use `--lowvram` on a 44 GB card.

---

## 5. Frame count and fps

The project contract is **8 fps, 41 frames for a 5 s clip, 33 for 4 s**.
`WanImageToVideo` requires `(length - 1) % 4 == 0`; both satisfy it.

The generator was computing `duration * 16 + 1` = **81 frames at 16 fps**,
which doubled sampling cost for no authored benefit.

```python
# scripts/render.py
inputs["length"] = 41 if cfg["duration"] >= 5 else 33
```

```python
# scripts/generate_workflow.py — VHS_VideoCombine
"frame_rate": 8,
```

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

### Step timing reference (A6000, 832×480, 41 frames)

| Config | s/step | Per clip |
|---|---|---|
| fp16, 20 steps, dynamic VRAM | ~13.6 | 5.3 min |
| fp16 + LightX2V, 4 steps | — | **40 s** |

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
