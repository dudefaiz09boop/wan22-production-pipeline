# AIC Cloud - Prerequisites Checklist for WAN 2.2 Production

## ✅ Required Before Starting

### 1. AIC Cloud Account & GPU Instance
- [ ] **AIC Cloud account** created at https://aiccloud.in/
- [ ] **Wallet funded** (UPI/Card/NetBanking - INR billing)
- [ ] **GPU Instance deployed**:
  - **Recommended**: L40S 48GB (₹84.60/hr) for FP16 quality
  - **Alternative**: RTX 3090 24GB (~₹30/hr) for FP8 quality
  - **Location**: Choose closest (Mumbai/Singapore/US)
  - **Image**: PyTorch (pre-installed CUDA, drivers, Jupyter)
  - **Disk**: 250 GB minimum (models + outputs)
  - **SSH Key**: Added in AIC dashboard (required for access)

### 2. API Keys & Tokens (MUST HAVE)

| Token | Source | Purpose | Required |
|-------|--------|---------|----------|
| **HF_TOKEN** | https://huggingface.co/settings/tokens | Download WAN 2.2 models from Comfy-Org | ✅ YES |
| **KAGGLE_KEY** | https://www.kaggle.com/settings (API → Create New Token) | Download trained LoRAs + free LoRA training | ✅ YES |
| **WANDB_KEY** | https://wandb.ai/authorize | Optional: Experiment tracking | ❌ Optional |

**Token Format:**
```bash
# HF_TOKEN: hf_xxxxxxxxxxxxxxxxxxxxxxxx (starts with hf_)
# KAGGLE_KEY: {"username":"yourname","key":"xxxxxxxxxxxxxxxx"} (JSON from kaggle.json)
```

### 3. GitHub Repository
- [ ] **Fork/Clone** this repo to your GitHub
- [ ] **Update REPO_URL** in `scripts/setup_aic_fast.sh`:
  ```bash
  REPO_URL="https://github.com/YOUR_USERNAME/wan22-production-pipeline.git"
  ```
- [ ] **Repo must be public** OR add deploy key to AIC instance

### 4. LoRA Training (FREE on Kaggle) - DO THIS FIRST

**Option A: Use Automated Pipeline (Recommended)**
```bash
# 1. Prepare training data locally (startframes + reference plates)
#    Each detected face is assigned to exactly ONE character, so the two
#    datasets never share an image. Verify with the check in
#    scripts/README-training.md before uploading.
cd wan22-production-pipeline
python scripts/prepare_training_data.py \
    --images-ref-dir "C:/Users/FAIZ SAYYED/OneDrive/Desktop/Images REF" \
    --startframes-dir ./startframes \
    --output-dir ./training_data \
    --min-face-size 64 \
    --max-faces-ref 3 \
    --max-faces-startframe 2 \
    --assign-threshold 0.35 \
    --upscale-to 512

# 2. Upload datasets to Kaggle (one-time; use `datasets version` to update)
kaggle datasets create -p training_data/faiz -t "lora-train-faiz"
kaggle datasets create -p training_data/juv -t "lora-train-juveriya"
kaggle datasets create -p ../../startframes -t "startframes-c01-c70"
# Upload base model (Juggernaut XL v9)
kaggle datasets create -p /path/to/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors -t "wan22-nsfw-lightning-q8"

# 3. Deploy & run training kernel (free T4, ~45 min per LoRA)
#    One character per run: the kernel asserts the two datasets share no images.
python scripts/train_loras_kaggle.py --push --only faiz
python scripts/train_loras_kaggle.py --push --only juv
kaggle kernels status sofiya009/lora-nuke-faiz-juveriya-sdxl-faiz

# 4. Publish LoRAs as dataset
#    Each --only target writes to its own directory (lora_output/faiz, ...).
python scripts/train_loras_kaggle.py --output --only faiz
cd lora_output/faiz
cat > dataset-metadata.json << 'EOF'
{
  "id": "sofiya009/lora-nuke-faiz-juveriya-sdxl",
  "title": "LoRA Nuke: Faiz + Juveriya (SDXL)",
  "subtitle": "Character LoRAs for Wan 2.2 pipeline",
  "description": "Per-character SDXL LoRAs trained with Kohya-ss. One character per LoRA.",
  "isPrivate": true,
  "keywords": ["lora", "sdxl", "character", "wan22"],
  "licenses": [{"name": "CC0-1.0"}]
}
EOF
kaggle datasets version -p . -m "Per-character LoRAs, 512px crops"
```

**Option B: Manual Training (if you have existing LoRAs)**
- [ ] **Training complete** on `sofiya009/lora-nuke-faiz-juveriya-sdxl`
- [ ] **Dataset published** with these files:
  - `faiz_lora.safetensors`
  - `juv_lora.safetensors` 
  - `faiz_juv_lora.safetensors`
- [ ] **Kaggle API key** has access to this dataset

### 5. Character Reference Images
Place in `references/` folder before running:
- [ ] `references/faiz_ref.jpg` - Clear face photo of Faiz
- [ ] `references/juv_ref.jpg` - Clear face photo of Juveriya
- **Requirements**: Front-facing, good lighting, single person, >512px

### 6. Training Data Sources (for LoRA training)
- [ ] **Existing character datasets** (optional but recommended):
  - `lora-train-faiz`: 20-50 clean Faiz images
  - `lora-train-juveriya`: 20-50 clean Juveriya images
- [ ] **Startframes** (auto-used): `D:\AI Pipeline\startframes\c01-c70.png` - faces extracted automatically

---

## 🚀 Quick Start Commands

### On AIC Cloud Instance (after SSH):
```bash
# 1. Set tokens (one-time)
export HF_TOKEN="hf_your_token_here"
export KAGGLE_KEY='{"username":"yourname","key":"your_key_here"}'

# 2. Run ultra-fast setup (does everything)
curl -fsSL https://raw.githubusercontent.com/YOUR_USERNAME/wan22-production-pipeline/main/scripts/setup_aic_fast.sh | bash

# OR if repo already cloned:
cd /workspace/wan22-production-pipeline
bash scripts/setup_aic_fast.sh
```

### Expected Timeline (L40S 48GB):
| Phase | Time | Notes |
|-------|------|-------|
| System setup | ~30s | apt, uv, dirs |
| Python deps | ~60s | uv sync (cached) |
| Model downloads | ~8-12 min | Parallel aria2 (57 GB FP16) |
| LoRA downloads | ~1-2 min | From Kaggle |
| ComfyUI start | ~60s | Background |
| Stills generation | ~5-10 min | SDXL + InsightFace |
| **Clip rendering** | **~25-35 min** | **70 clips, main GPU work** |
| Face swap | ~3-5 min | InsightFace every 3rd frame |
| Assembly | ~2-3 min | ffmpeg concat + zip |
| **TOTAL** | **~45-60 min** | **~₹42-56 cost** |

---

## ⚡ Speed Optimizations Included

### Parallel Downloads (aria2)
- 16 connections per file
- 4 files simultaneously
- Resume support (auto-continue on interrupt)
- Expected: **57 GB in 8-12 min** on AIC network

### Background ComfyUI Start
- Starts during model downloads
- Ready when models finish
- Zero wait time

### Cached Python Environment
- `uv` locks dependencies
- Second run: **<10 seconds** for deps
- No pip recompilation

### Minimal ComfyUI Install
- Only required custom nodes:
  - ComfyUI-GGUF (for GGUF Q8 fallback)
  - ComfyUI-VideoHelperSuite (VHS_VideoCombine)
- No unnecessary nodes

### Optimized Render Flags
```bash
--lowvram --preview-method none --disable-auto-launch
```

---

## 🔧 Troubleshooting

### "Connection refused" on SSH
- Wait 90-120 seconds after instance shows "Running"
- AIC Cloud: SSH port opens after boot completes

### "HF_TOKEN invalid" / 401 errors
- Verify token at https://huggingface.co/settings/tokens
- Must have `read` access to `Comfy-Org/Wan-2.2`
- Token format: `hf_xxxxxxxxxxxxxxxxxxxxxxxx`

### "Kaggle dataset not found"
- Verify dataset exists: `kaggle datasets list -s lora-nuke-faiz-juveriya-sdxl`
- Check username: `sofiya009` (case sensitive)
- Ensure dataset is **published** (not private/draft)

### Out of disk space
```bash
# Check
df -h /workspace

# Clean if needed
rm -rf /workspace/ComfyUI/output/*
rm -rf ~/.cache/huggingface/hub/*
```

### Out of VRAM (OOM)
- Script auto-selects precision based on VRAM
- Force lower: `--force-precision fp8` or `--force-precision gguf_q8`
- Reduce resolution in `generate_stills.py` (--width/--height)

### ComfyUI not ready
```bash
# Check logs
tail -f /tmp/comfyui.log

# Common: missing custom nodes
cd /workspace/ComfyUI/custom_nodes
git clone https://github.com/kijai/ComfyUI-GGUF.git
git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git
```

---

## 📋 Pre-Flight Verification Script

Run this before starting to verify everything:

```bash
#!/bin/bash
# verify_prereqs.sh

echo "=== AIC Cloud WAN 2.2 Pre-Flight Check ==="

# 1. GPU
echo -n "GPU: "
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 2. Tokens
echo -n "HF_TOKEN: "; [[ -n "$HF_TOKEN" ]] && echo "✓ Set" || echo "✗ MISSING"
echo -n "KAGGLE_KEY: "; [[ -n "$KAGGLE_KEY" ]] && echo "✓ Set" || echo "✗ MISSING"

# 3. Disk
echo -n "Disk: "
df -h /workspace | tail -1 | awk '{print $4 " free of " $2}'

# 4. Network (HF)
echo -n "HuggingFace: "
curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $HF_TOKEN" \
  "https://huggingface.co/api/models/Comfy-Org/Wan-2.2" && echo " ✓" || echo " ✗"

# 5. Network (Kaggle)
echo -n "Kaggle: "
curl -s -o /dev/null -w "%{http_code}" -u "$(echo $KAGGLE_KEY | jq -r .username):$(echo $KAGGLE_KEY | jq -r .key)" \
  "https://www.kaggle.com/api/v1/datasets/list?search=lora-nuke-faiz-juveriya-sdxl" && echo " ✓" || echo " ✗"

# 6. Refs
echo -n "References: "
[[ -f references/faiz_ref.jpg ]] && echo -n "faiz ✓ " || echo -n "faix ✗ "
[[ -f references/juv_ref.jpg ]] && echo "juv ✓" || echo "juv ✗"

echo "=== Run ./scripts/setup_aic_fast.sh when all ✓ ==="
```

---

## 💰 Cost Control

### Auto-Shutdown (Built into setup_aic_fast.sh)
- Prompts at end: "Shutdown instance now? (y/N)"
- Type `y` to stop billing immediately

### Manual Stop (AIC Dashboard)
- Instances → Your GPU → **Stop** (billing pauses)
- **Destroy** to release disk (stops disk billing too)

### Wallet Alerts
- Set low-balance alert in AIC Dashboard → Settings → Billing
- Prevents unexpected instance termination

---

## 📞 Support

- **AIC Cloud**: 24/7 WhatsApp/Email (from dashboard)
- **Pipeline Issues**: Check `/tmp/render.log`, `/tmp/comfyui.log`
- **Logs**: `output/logs/production_*.json` has full trace

---

**✅ When all checkboxes above are ticked → You're ready for zero-waste GPU rendering!**