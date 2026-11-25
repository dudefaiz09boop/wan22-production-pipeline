# Free Kaggle LoRA Training Pipeline

**Train character LoRAs on Kaggle's free T4 GPU (30 hrs/week) using startframes + existing datasets.**

## Overview

This pipeline trains 3 SDXL LoRAs for the Wan 2.2 production pipeline:
- `faiz_lora.safetensors` - Faiz character
- `juv_lora.safetensors` - Juveriya character
- `faiz_juv_lora.safetensors` - Combined both

**Cost: $0** - Runs entirely on Kaggle free tier.

## Quick Start

### 1. Prepare Training Data Locally

```bash
cd wan22-production-pipeline

# Extract faces from startframes + combine with existing datasets
python scripts/prepare_training_data.py \
    --images-ref-dir "C:/Users/FAIZ SAYYED/OneDrive/Desktop/Images REF" \
    --startframes-dir ./startframes \
    --output-dir ./training_data \
    --min-face-size 64 \
    --max-faces-ref 3 \
    --max-faces-startframe 2 \
    --assign-threshold 0.35 \
    --upscale-to 512
```

**One character per LoRA.** A startframe normally contains both people, so every
detected face is scored against per-character embedding centroids built from the
reference plates and written to the **single** winning dataset. Faces that match
neither character confidently are dropped. Do not reintroduce a per-beat
character list: it cannot say which detected face belongs to whom, and the old
version wrote every face into both sets (58 of 65 "faiz" crops were byte-identical
to "juv" crops, so each LoRA learned both faces).

`--min-face-size 64` is deliberate. Startframe faces are often 70-110px; a 128px
floor silently discarded most of them. `--upscale-to 512` Lanczos-upscales the
small crops so training is not fed 100px faces.

This creates:
```
training_data/
├── faiz/           # Faiz face crops + captions
├── juv/            # Juveriya face crops + captions
├── combined/       # Both characters for combined LoRA
└── metadata.json   # Counts and stats
```

Verify the split before uploading:

```bash
python - <<'PY'
import glob, hashlib, os
def h(p): return hashlib.md5(open(p, "rb").read()).hexdigest()
faiz = {h(f) for f in glob.glob("training_data/faiz/*.jpg")}
juv = {h(f) for f in glob.glob("training_data/juv/*.jpg")}
print("faiz:", len(faiz), "juv:", len(juv), "shared:", len(faiz & juv))
assert not (faiz & juv), "a crop is in both datasets - rerun prepare_training_data.py"
PY
```

### 2. Upload Datasets to Kaggle (One-time)

```bash
# Create 4 private datasets on Kaggle
kaggle datasets create -p training_data/faiz -t "lora-train-faiz"
kaggle datasets create -p training_data/juv -t "lora-train-juveriya"
kaggle datasets create -p training_data/combined -t "lora-train-combined"

# Upload base model (Juggernaut XL v9)
kaggle datasets create -p /path/to/Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors -t "wan22-nsfw-lightning-q8"

# Upload startframes as dataset
kaggle datasets create -p ../../startframes -t "startframes-c01-c70"
```

### 3. Deploy & Run Training Kernel

```bash
# Build and push kernel
python scripts/train_loras_kaggle.py --push

# Monitor progress
kaggle kernels output sofiya009/lora-nuke-faiz-juv -p /tmp/lora_output
```

### 4. Download & Publish LoRAs

```bash
# After kernel completes, download output
python scripts/train_loras_kaggle.py --output

# Create dataset metadata and upload as new version
cd /tmp/lora_output
cat > dataset-metadata.json << 'EOF'
{
  "id": "sofiya009/lora-nuke-faiz-juv",
  "title": "LoRA Nuke: Faiz + Juveriya (SDXL)",
  "subtitle": "Character LoRAs for Wan 2.2 pipeline",
  "description": "3 LoRAs: faiz, juv, faiz_juv combined. Trained on SDXL with Kohya-ss.",
  "isPrivate": true,
  "keywords": ["lora", "sdxl", "character", "wan22"],
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

kaggle datasets version -p /tmp/lora_output -m "Updated LoRAs with startframes faces"
```

## Training Configuration (Optimized for T4 16GB)

| Parameter | Value | Reason |
|-----------|-------|--------|
| **Base Model** | Juggernaut XL v9 | Photorealistic, good for characters |
| **LoRA Rank** | 32 | Fast training, sufficient quality |
| **LoRA Alpha** | 16 | Half of rank; stops a small dataset over-saturating |
| **Precision** | fp16 | Half VRAM usage |
| **Gradient Checkpointing** | Enabled | Saves VRAM |
| **Optimizer** | AdamW8bit | 8-bit saves VRAM |
| **Learning Rate** | 1e-4 | Standard for LoRA |
| **Scheduler** | Cosine with restarts | Good convergence |
| **Max Steps** | 1200 per LoRA | ~45 min each on T4 |
| **Batch Size** | 1 (accum 2) | Effective batch 2 |
| **Resolution** | 1024x1024 | SDXL native |
| **Save Every** | 400 steps | 3 checkpoints |
| **Regularization** | Other character data | Prevents bleeding |

**Total training time: ~2-3 hours** (fits in single Kaggle session)

The kernel asserts that the two datasets share no images before training starts.
If it fails, the face-extraction step wrote the same face into both sets — rerun
`prepare_training_data.py` rather than training on mixed data.

## Data Sources

| Source | Purpose | Images |
|--------|---------|--------|
| **Startframes (c01-c70)** | Face extraction via InsightFace | ~140 faces (2 per frame × 70) |
| **Existing lora-train-faiz** | Clean character references | 20-50 |
| **Existing lora-train-juveriya** | Clean character references | 20-50 |
| **Combined** | Both characters for joint LoRA | Sum of above |

## Kaggle Kernel Settings

In Kaggle kernel settings, attach these **private datasets**:
1. `sofiya009/startframes-c01-c70` → `/kaggle/input/startframes-c01-c70`
2. `sofiya009/lora-train-faiz` → `/kaggle/input/lora-train-faiz`
3. `sofiya009/lora-train-juveriya` → `/kaggle/input/lora-train-juveriya`
4. `sofiya009/wan22-nsfw-lightning-q8` → `/kaggle/input/wan22-nsfw-lightning-q8`

**GPU**: T4 x2 (enable in kernel settings)
**Internet**: Enabled (for pip installs)

## Output

After successful run, `loras.zip` contains:
```
loras.zip
├── faiz_lora.safetensors      # ~150 MB
├── juv_lora.safetensors       # ~150 MB
└── faiz_juv_lora.safetensors  # ~150 MB
```

## Integration with Production Pipeline

Once `loras.zip` is published as `sofiya009/lora-nuke-faiz-juv`:

1. The production pipeline (`setup_aic_fast.sh`) auto-downloads them
2. Updates `workflows/loras.json` with trained LoRA entries
3. Uses them in SDXL stills generation + InsightFace swap
4. Available for video generation via ComfyUI

## Troubleshooting

### "Dataset not mounted"
- Check kernel settings → Datasets → ensure all 4 attached
- Verify dataset names match exactly (case-sensitive)

### OOM / Exit code -9
- Reduce `max_train_steps` to 1000
- Disable `cache_latents_to_disk`
- Use `gradient_accumulation_steps: 8` (effective batch 8, slower but less VRAM)

### Training too slow
- Ensure T4 GPU enabled (not CPU)
- Check `nvidia-smi` in first cell output
- Reduce resolution to `768,768` if needed

### LoRA quality issues
- Increase `max_train_steps` to 2000
- Add more diverse training images
- Check caption files match images
- Verify face extraction worked (check `metadata.json`)
- **Faces look like the other character:** the datasets are mixed. Run the
  shared-image check in step 1; it must print `shared: 0`.
- **Faces are soft or generic:** the crops are too small. Regenerate with
  `--upscale-to 512` and confirm the median shortest side is >= 512.

### Replacing an existing LoRA
`download_loras.py` now stages the **newest** local copy, so a fresh training
run wins over a stale file. If you want to force a re-download from Kaggle, pass
`--force`.

## Files

| File | Purpose |
|------|---------|
| `scripts/prepare_training_data.py` | Local face extraction + data prep |
| `scripts/train_loras_kaggle.py` | Build/deploy Kaggle kernel |
| `scripts/requirements-training.txt` | Dependencies |
| `deploy/training/train_loras_kaggle.ipynb` | Generated notebook |
| `deploy/training/kernel-metadata.json` | Kernel config |

## Cost Summary

| Resource | Cost |
|----------|------|
| Kaggle T4 GPU | **Free** (30 hrs/week) |
| Dataset storage | **Free** (private datasets) |
| Kernel runs | **Free** |
| **Total** | **$0** |

---

**Next**: After LoRAs trained, run production on AIC Cloud L40S:
```bash
./scripts/setup_aic_fast.sh
```