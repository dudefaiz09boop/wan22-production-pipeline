# base_i2v.json

API-format skeleton. Export your working graph via **Save (API Format)** and reconcile
the `class_type` strings above with what your install actually has. Keep node ids 1-12
stable — `scripts/render.py` binds to `class_type` plus the `__todo__` marker, and will
fill:

- `6.inputs.text`  -> the assembled clip prompt
- `7.inputs.text`  -> negatives/artifact_neg.txt
- `5.inputs.image` -> the shot's start_frame
- `8.inputs.width/height/length` -> from meta + clip build

Swap nodes `1`/`2` to the CF GGUF pair for any clip with `build: cf`, or split the
workflow into `base_i2v_fm.json` and `base_i2v_cf.json` and pass `--workflow`.
`scripts/render.py` auto-injects the LoRA loaders: for every key in the clip's
`loras` array it reads `workflows/loras.json`, appends a `LoraLoaderModelOnly`
chain into the HIGH sampler branch (`1`->`9`) for a non-null `high` path and into
the LOW branch (`2`->`10`) for a non-null `low` path, at the registry strength `w`.
Null slots are skipped for that pass; missing local lora files print a loud
warning but are never silently dropped.

## loras.json registry schema
`workflows/loras.json` maps each LoRA key to:

```json
{
  "key": {
    "high": "loras/<key>_wan22_high.safetensors",
    "low": "loras/<key>_wan22_low.safetensors",
    "w": 0.9,
    "trigger": null,
    "trigger_scope": null,
    "conflicts": ["other_key"],
    "source": { "civitai_model": 123, "civitai_version": 456 }
  }
}
```

- `high` / `low`: file path under `loras/`, or `null` to skip that noise pass.
- `w`: strength applied to both passes by the gateway graph builder and `render.py`.
- `trigger` + `trigger_scope` (`prompt_prefix` or `prompt_suffix`): injected into
  the compiled prompt by `scripts/build_prompts.py` and checked by `scripts/validate.py`.
- `conflicts`: optional list of keys that must not appear on the same shot.
- `source`: provenance (`civitai_model`/`civitai_version` or `huggingface_repo`).
  Only `penis_insert` and `vag_detail` carry pinned Civitai version ids + SHA-256
  in `gateway/video-workflow.js`; the other entries are unpinned and must be
  staged manually.

## Class types to reconcile
| node | expected class_type | check |
|------|--------------------|-------|
| 1 | UnetLoaderGGUF | HIGH fast-move q8 gguf |
| 2 | UnetLoaderGGUF | LOW fast-move q8 gguf |
| 3 | CLIPLoader | umt5 nsfw fp8, type `wan` |
| 4 | VAELoader | wan 2.1 vae |
| 8 | Wan22ImageToVideo | or `WanImageToVideo` on older builds |
| 9 | KSamplerAdvanced | HIGH, steps 0->2 |
| 10 | KSamplerAdvanced | LOW, steps 2->5 |
| 12 | VHS_VideoCombine | needs ComfyUI-VideoHelperSuite |

If your install names node 8 differently, change it in the JSON — `render.py` matches
on the tuple `("Wan22ImageToVideo", "WanImageToVideo")` and will miss anything else.
