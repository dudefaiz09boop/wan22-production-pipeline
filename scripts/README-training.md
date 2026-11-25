# Character LoRA Training

**Train one SDXL LoRA per character.** Two paths are supported: Kaggle (free T4,
fully scripted) and a hosted trainer such as ModelScope/Civision (upload the
bundles from step 2).

## Overview

- `faiz_lora.safetensors` - Faiz character
- `juv_lora.safetensors` - Juveriya character
- `faiz_juv_lora.safetensors` - combined; a fallback only, not the default

**One character per LoRA.** A startframe normally contains both people, so the
data prep assigns every detected face to exactly one character. Training both
people into one LoRA is what made the trigger words meaningless.

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

### 2. Package for a Hosted Trainer (ModelScope / Civision)

`export_training_bundle.py` validates the split and writes one zip per character
with images, captions and a `train_config.json`:

```bash
python scripts/export_training_bundle.py --data-dir training_data --out-dir bundles
```

It refuses to package when a crop is in both sets, a caption is missing, a
caption does not start with its own trigger, or a set is under 15 images. Pass
`--allow-problems` to override.

Upload each zip as its **own** dataset/concept, then train one character per run
using the values in `train_config.json`:

| Setting | Value |
|---|---|
| Base model | SDXL 1.0 / Juggernaut XL v9 |
| Network dim | 32 |
| Network alpha | 16 |
| Resolution | 1024 |
| Steps | 1200 |
| LR | 1e-4 |
| Text encoder LR | 5e-5 |
| Batch size | 1 (accum 2) |

Alpha is half of dim because the dataset is small; alpha == dim over-saturates.
Do not merge the two characters into one concept.

### 3. Upload Datasets to Kaggle (One-time)

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

### 4. Deploy & Run Training Kernel

```bash
# Build and push kernel (one LoRA per run is recommended)
python scripts/train_loras_kaggle.py --push --only faiz
python scripts/train_loras_kaggle.py --push --only juv

# Monitor progress
kaggle kernels status sofiya009/lora-nuke-faiz-juveriya-sdxl-faiz
```

### 5. Download & Publish LoRAs

```bash
# After the kernel completes, download its artifacts.
# Each --only target writes to its own directory so runs cannot overwrite
# each other: lora_output/faiz, lora_output/juv, lora_output/combined.
python scripts/train_loras_kaggle.py --output --only faiz
python scripts/train_loras_kaggle.py --output --only juv

# Publish as a new dataset version
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

Once `loras.zip` is published as `sofiya009/lora-nuke-faiz-juveriya-sdxl`:

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
| `scripts/prepare_training_data.py` | Local face extraction + per-character assignment |
| `scripts/export_training_bundle.py` | Validate the split and package zips for a hosted trainer |
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