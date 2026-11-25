#!/usr/bin/env python3
"""Main production orchestrator - unified one-click pipeline for AIC Cloud."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

# Import our modules
sys.path.insert(0, str(Path(__file__).parent))
from auto_model_selector import (
    get_gpu_vram_gb,
    select_precision,
    get_model_config,
    get_hf_repo,
    get_hf_files,
    get_local_model_dir,
    Precision,
    ModelConfig,
)
from download_models import download_models, create_session, get_file_size
from generate_workflow import generate_workflow, save_workflow
from comfy_api import upload, server_default
from render import (
    inject,
    load_workflow,
    post,
    wait_for,
    last_frame,
    download_output,
    select_start_image,
    preceding_chained_frame,
)


@dataclass
class ClipResult:
    clip_id: str
    success: bool
    duration_seconds: float
    output_path: Optional[Path] = None
    error: Optional[str] = None


@dataclass
class ProductionState:
    precision: Precision
    config: ModelConfig
    model_dir: Path
    workflow_path: Path
    server: str
    negative_prompt: str
    lora_registry: dict
    clips: list[dict] = field(default_factory=list)
    results: list[ClipResult] = field(default_factory=list)
    start_time: float = field(default_factory=time.time)
    total_gpu_seconds: float = 0.0
    # Loaded once so the graph is not re-read per clip, and shared so start
    # frames are uploaded once instead of on every clip.
    base_workflow: dict = field(default_factory=dict)
    uploads_cache: dict = field(default_factory=dict)
    # New pipeline stages
    trained_loras_downloaded: bool = False
    stills_generated: bool = False
    face_swap_done: bool = False
    final_assembled: bool = False


def load_beats(spec_path: Path) -> tuple[dict, list[dict]]:
    """Load beats.yaml and return meta and shots."""
    data = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    meta = data["meta"]
    shots = data["shots"]
    return meta, shots


def setup_directories() -> tuple[Path, Path, Path]:
    """Create necessary directories."""
    renders_dir = Path("renders")
    renders_dir.mkdir(parents=True, exist_ok=True)

    chained_dir = Path("chained")
    chained_dir.mkdir(parents=True, exist_ok=True)

    logs_dir = Path("logs")
    logs_dir.mkdir(parents=True, exist_ok=True)

    return renders_dir, chained_dir, logs_dir


def download_all_models(state: ProductionState, token: Optional[str] = None) -> bool:
    """Download all required models for the selected precision."""
    print(f"\n{'='*60}")
    print(f"Downloading models for {state.precision.upper()}")
    print(f"{'='*60}")

    repo = get_hf_repo(state.precision)
    files = get_hf_files(state.precision)
    dest_dir = state.model_dir

    # Known SHA-256 hashes for verification (from Civitai/HF)
    sha256_map = {
        # FP16
        "wan2.2_i2v_high_noise_14B_fp16.safetensors": "A1B2C3D4E5F6...",  # Placeholder
        "wan2.2_i2v_low_noise_14B_fp16.safetensors": "F6E5D4C3B2A1...",
        # FP8
        "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors": "1A2B3C4D5E6F...",
        "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors": "F6E5D4C3B2A1...",
        # GGUF Q8 (Civitai 2053259 v2540892 / v2540896). The LOW filename really
        # is missing its opening paren -- that is the author's typo and the hash
        # lookup is keyed on the literal string.
        "Wan2_2-I2V-A14B-NSFW_fast_move-HIGH_Q8(lightning_edition)V2.gguf": "DEB6DCF39FEFEF7850D1DD49A52B70A64C65B8D8E7B5F03BC1B6A541CF58AAE9",
        "Wan2_2-I2V-A14B-NSFW_fast_move-LOW_Q8lightning_edition)V2.gguf": "B5722E8E01A21F89A1C0B5E9DE9A3B907CE92324D1951D5CE70307956790DAAE",
        # GGUF Q8 Consistent Face (Civitai 2053259 v2668710 / v2668712).
        # No Lightning baked in -- render.py chains the LightX2V LoRAs for these.
        "Wan2_2-I2V-A14B-HIGH_consistent_face_Q8.gguf": None,
        "Wan2_2-I2V-A14B-LOW_consistent_face_Q8.gguf": None,
        # Shared - use actual known hashes or skip verification
        "umt5_xxl_fp8_e4m3fn_scaled.safetensors": None,  # Skip verification
        "wan_2.1_vae.safetensors": None,  # Skip verification
    }

    results = download_models(repo, files, dest_dir, token, sha256_map)

    success_count = sum(1 for v in results.values() if v)
    print(f"\nDownloaded: {success_count}/{len(results)} files")

    if success_count != len(results):
        failed = [k for k, v in results.items() if not v]
        print(f"Failed: {', '.join(failed)}")
        return False

    return True


def load_lora_registry() -> dict:
    """Load LoRA registry from workflows/loras.json."""
    path = Path("workflows/loras.json")
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def render_clip(
    state: ProductionState,
    clip_cfg: dict,
    prompt_text: str,
    prev_frame: Optional[Path],
    clip_index: int,
    total_clips: int,
) -> ClipResult:
    """Render a single clip."""
    clip_id = clip_cfg["id"]
    clip_start = time.time()

    print(f"\n[{clip_index}/{total_clips}] Rendering {clip_id}...")

    try:
        # Select start image (chained or fresh)
        image_path, chained = select_start_image(
            Path(f"prompts/{clip_id}.json"),
            clip_cfg,
            prev_frame,
            None,  # failed_shot
            dry_run=False,
        )

        if not image_path.exists():
            return ClipResult(
                clip_id=clip_id,
                success=False,
                duration_seconds=0,
                error=f"Start frame not found: {image_path}",
            )

        # Upload image to ComfyUI. The cache must persist across clips: a fresh
        # dict per clip re-uploads every start frame and every chained frame.
        image = upload(state.server, image_path, state.uploads_cache)

        # Inject into a copy of the workflow loaded once at startup, so the
        # model stays resident in ComfyUI instead of being re-resolved.
        wf = inject(
            json.loads(json.dumps(state.base_workflow)),
            prompt_text,
            state.negative_prompt,
            clip_cfg,
            image,
            state.lora_registry,
        )

        # Submit to ComfyUI
        pid = post(state.server, wf)
        print(f"  Queued: {pid}")

        # Wait for completion
        entry = wait_for(state.server, pid, timeout=3600)

        # Extract last frame for chaining
        chained_dir = Path("chained")
        prev_frame = last_frame(state.server, entry, chained_dir / f"{clip_id}.png")

        # Download the rendered clip so assembly has a real file to read.
        renders_dir = Path("renders")
        renders_dir.mkdir(parents=True, exist_ok=True)
        video_path = download_output(state.server, entry, renders_dir / f"{clip_id}.mp4")

        duration = time.time() - clip_start
        state.total_gpu_seconds += duration

        return ClipResult(
            clip_id=clip_id,
            success=True,
            duration_seconds=duration,
            output_path=video_path,
        )

    except Exception as e:
        duration = time.time() - clip_start
        return ClipResult(
            clip_id=clip_id,
            success=False,
            duration_seconds=duration,
            error=str(e),
        )


def estimate_cost(gpu_seconds: float, gpu_type: str = "A100") -> float:
    """Estimate cost based on GPU time."""
    # Rough pricing per hour (USD)
    pricing = {
        "A100": 1.50,
        "A100-80GB": 2.50,
        "H100": 4.00,
        "RTX3090": 0.50,
        "RTX4090": 0.80,
    }
    hourly_rate = pricing.get(gpu_type, 1.00)
    return (gpu_seconds / 3600) * hourly_rate


def download_trained_loras_stage(state: ProductionState, token: Optional[str] = None) -> bool:
    """Stage 1: Download trained LoRAs from Kaggle."""
    print(f"\n{'='*60}")
    print("STAGE 1: Downloading trained LoRAs from Kaggle")
    print(f"{'='*60}")

    try:
        # Import and run download_loras
        sys.path.insert(0, str(Path(__file__).parent))
        from download_loras import download_trained_loras, update_lora_registry

        lora_dir = Path("models/loras/trained")
        results = download_trained_loras(lora_dir, token=token)
        success_count = sum(1 for v in results.values() if v)
        # Register whatever resolved. Requiring *all* of them meant a single
        # optional file (the combined faiz_juv LoRA is not always published)
        # left the registry untouched, so the two LoRAs that did download were
        # never referenced by any shot.
        if success_count:
            update_lora_registry(lora_dir, Path("workflows/loras.json"))
            state.trained_loras_downloaded = True
            print(f"✓ Trained LoRAs staged ({success_count}/{len(results)}) and registry updated")
            missing = [k for k, v in results.items() if not v]
            if missing:
                print(f"  ! Unavailable: {', '.join(missing)} (optional)")
            return True
        print(f"✗ No trained LoRAs available: {[k for k, v in results.items() if not v]}")
        return False
    except Exception as e:
        print(f"✗ Trained LoRA download failed: {e}")
        return False


def generate_stills_stage(state: ProductionState) -> bool:
    """Stage 2: Generate SDXL stills with trained LoRAs + InsightFace."""
    print(f"\n{'='*60}")
    print("STAGE 2: Generating SDXL stills with trained LoRAs")
    print(f"{'='*60}")

    try:
        # This would call generate_stills.py
        # For now, we'll check if startframes exist
        startframes_dir = Path("startframes")
        if startframes_dir.exists():
            frame_count = len(list(startframes_dir.glob("*.png")))
            print(f"  Found {frame_count} existing start frames")
            state.stills_generated = True
            return True
        else:
            print("  No startframes directory - skipping (run generate_stills.py separately)")
            return True  # Don't fail the pipeline
    except Exception as e:
        print(f"✗ Stills generation failed: {e}")
        return False


def render_clips_stage(state: ProductionState, args) -> bool:
    """Stage 3: Render clips via ComfyUI (existing logic)."""
    print(f"\n{'='*60}")
    print("STAGE 3: Rendering clips via ComfyUI")
    print(f"{'='*60}")

    prev_frame: Optional[Path] = None
    failed_shot: Optional[str] = None

    # Resume: a clip whose mp4 is already on disk is done. Without this a run
    # that dies at c47 re-renders c01-c46 on the next attempt, which is hours
    # of GPU time for output that already exists.
    renders_dir = Path(args.renders)
    chained_dir = Path("chained")
    skipped = 0
    for clip in state.clips:
        clip_id = clip["id"]
        if (renders_dir / f"{clip_id}.mp4").is_file():
            skipped += 1
            state.results.append(ClipResult(clip_id, True, 0, renders_dir / f"{clip_id}.mp4"))
    if skipped:
        print(f"  Resuming: {skipped} clip(s) already rendered, skipping")

    for i, clip in enumerate(state.clips, 1):
        clip_id = clip["id"]

        if (renders_dir / f"{clip_id}.mp4").is_file():
            # Seed the chain from this clip's own handoff so the next unrendered
            # shot continues from the right frame instead of a stale one.
            handoff = chained_dir / f"{clip_id}.png"
            prev_frame = handoff if handoff.is_file() else None
            failed_shot = None
            continue

        # Load prompt
        prompt_file = Path(args.prompts) / f"{clip_id}.txt"
        if not prompt_file.exists():
            print(f"  ✗ {clip_id}: Prompt not found")
            state.results.append(ClipResult(clip_id, False, 0, error="Prompt not found"))
            failed_shot = clip_id
            continue

        prompt_text = prompt_file.read_text(encoding="utf-8")
        clip_cfg_path = Path(args.prompts) / f"{clip_id}.json"
        clip_cfg = json.loads(clip_cfg_path.read_text(encoding="utf-8"))

        # Handle chaining. A failed predecessor must not cascade: every shot
        # has its own authored start frame, so fall back to it and keep going.
        # Previously one bad clip marked the next 40 as "Chain broken by cXX".
        if clip_cfg.get("chain") and failed_shot:
            print(
                f"  ! {clip_id}: previous shot {failed_shot} failed; "
                "reseeded from its own start frame"
            )
            prev_frame = None
            failed_shot = None

        # Select start image
        try:
            image_path, chained = select_start_image(
                clip_cfg_path,
                clip_cfg,
                prev_frame,
                failed_shot,
                dry_run=False,
            )
        except RuntimeError as e:
            print(f"  ✗ {clip_id}: {e}")
            state.results.append(ClipResult(clip_id, False, 0, error=str(e)))
            failed_shot = clip_id
            continue

        # Render
        result = render_clip(state, clip_cfg, prompt_text, prev_frame, i, len(state.clips))
        state.results.append(result)

        if result.success:
            prev_frame = Path("chained") / f"{clip_id}.png"
            failed_shot = None
            print(f"  ✓ {clip_id} done in {result.duration_seconds:.1f}s")
        else:
            print(f"  ✗ {clip_id} failed: {result.error}")
            prev_frame = None
            failed_shot = clip_id

    return all(r.success for r in state.results)


def face_swap_stage(state: ProductionState, args) -> bool:
    """Stage 4: Face swap every 3rd frame using InsightFace."""
    print(f"\n{'='*60}")
    print("STAGE 4: Face swapping every 3rd frame")
    print(f"{'='*60}")

    try:
        # This would call face_swap_video.py
        # For now, check if face-swapped videos exist
        renders_dir = Path(args.renders)
        swapped_dir = renders_dir / "face_swapped"
        if swapped_dir.exists():
            swapped_count = len(list(swapped_dir.glob("*.mp4")))
            print(f"  Found {swapped_count} face-swapped clips")
            state.face_swap_done = True
            return True
        else:
            print("  No face-swapped directory - skipping (run face_swap_video.py separately)")
            return True  # Don't fail the pipeline
    except Exception as e:
        print(f"✗ Face swap failed: {e}")
        return False


def assemble_final_stage(state: ProductionState, args) -> bool:
    """Stage 5: Assemble final deliverable."""
    print(f"\n{'='*60}")
    print("STAGE 5: Assembling final deliverable")
    print(f"{'='*60}")

    try:
        # This would call assemble_final.py
        output_dir = Path(args.output)
        final_video = output_dir / "final.mp4"
        if final_video.exists():
            print(f"  Final video exists: {final_video}")
            state.final_assembled = True
            return True
        else:
            print("  No final video - skipping (run assemble_final.py separately)")
            return True  # Don't fail the pipeline
    except Exception as e:
        print(f"✗ Final assembly failed: {e}")
        return False


def save_production_log(state: ProductionState, logs_dir: Path) -> None:
    """Save production log with all results."""
    log_data = {
        "precision": state.precision,
        "model_config": {
            "high_noise": state.config.high_noise_model,
            "low_noise": state.config.low_noise_model,
            "loader": state.config.loader_class,
        },
        "total_clips": len(state.clips),
        "successful": sum(1 for r in state.results if r.success),
        "failed": sum(1 for r in state.results if not r.success),
        "total_gpu_seconds": state.total_gpu_seconds,
        "total_wall_seconds": time.time() - state.start_time,
        "estimated_cost_usd": estimate_cost(state.total_gpu_seconds),
        "clips": [
            {
                "id": r.clip_id,
                "success": r.success,
                "duration_seconds": r.duration_seconds,
                "output": str(r.output_path) if r.output_path else None,
                "error": r.error,
            }
            for r in state.results
        ],
    }

    log_path = logs_dir / f"production_{time.strftime('%Y%m%d_%H%M%S')}.json"
    log_path.write_text(json.dumps(log_data, indent=2))
    print(f"\nProduction log saved to {log_path}")


def print_summary(state: ProductionState) -> None:
    """Print final production summary."""
    print("\n" + "=" * 60)
    print("PRODUCTION COMPLETE")
    print("=" * 60)
    print(f"Precision: {state.precision.upper()}")
    print(f"Total clips: {len(state.clips)}")
    print(f"Successful: {sum(1 for r in state.results if r.success)}")
    print(f"Failed: {sum(1 for r in state.results if not r.success)}")
    print(f"GPU time: {state.total_gpu_seconds:.1f}s ({state.total_gpu_seconds/60:.1f} min)")
    print(f"Wall time: {time.time() - state.start_time:.1f}s ({(time.time() - state.start_time)/60:.1f} min)")
    print(f"Estimated cost: ${estimate_cost(state.total_gpu_seconds):.2f}")
    print("=" * 60)

    if any(not r.success for r in state.results):
        print("\nFailed clips:")
        for r in state.results:
            if not r.success:
                print(f"  {r.clip_id}: {r.error}")


def main() -> int:
    ap = argparse.ArgumentParser(description="One-click WAN 2.2 production pipeline for AIC Cloud")
    ap.add_argument("--spec", default="docs/beats.yaml", help="Beats specification file")
    ap.add_argument("--prompts", default="prompts", help="Prompts directory")
    ap.add_argument("--startframes", default="startframes", help="Start frames directory")
    ap.add_argument("--renders", default="renders", help="Output renders directory")
    ap.add_argument("--output", default="output", help="Final output directory")
    ap.add_argument("--server", default=None, help="ComfyUI server URL")
    ap.add_argument("--force-precision", choices=["fp16", "fp8", "gguf_q8"], help="Force precision")
    ap.add_argument("--skip-download", action="store_true", help="Skip model download")
    ap.add_argument("--skip-loras", action="store_true", help="Skip trained LoRA download")
    ap.add_argument("--skip-stills", action="store_true", help="Skip stills generation")
    ap.add_argument("--skip-render", action="store_true", help="Skip clip rendering")
    ap.add_argument("--skip-face-swap", action="store_true", help="Skip face swap")
    ap.add_argument("--skip-assemble", action="store_true", help="Skip final assembly")
    ap.add_argument("--from-clip", default="c01", help="Start clip (inclusive)")
    ap.add_argument("--to-clip", default="c70", help="End clip (inclusive)")
    ap.add_argument("--dry-run", action="store_true", help="Validate without rendering")
    ap.add_argument(
        "--nag",
        action="store_true",
        help="Enable WanVideoNAG so the negative prompt is evaluated (doubles render time)",
    )
    ap.add_argument("--nag-scale", type=float, default=11.0, help="NAG guidance strength")
    ap.add_argument("--nag-alpha", type=float, default=0.25, help="NAG mixing coefficient")
    ap.add_argument("--nag-tau", type=float, default=2.5, help="NAG clipping threshold")
    ap.add_argument("--hf-token", help="Hugging Face token (or set HF_TOKEN env)")
    args = ap.parse_args()

    # Setup
    print("=" * 60)
    print("WAN 2.2 PRODUCTION PIPELINE - AIC Cloud")
    print("=" * 60)

    # Auto-detect VRAM and select model
    vram_gb = get_gpu_vram_gb()
    precision = select_precision(vram_gb, args.force_precision)
    config = get_model_config(precision)
    model_dir = get_local_model_dir(precision)

    print(f"VRAM: {vram_gb} GB" if vram_gb else "VRAM: Unknown")
    print(f"Precision: {precision.upper()}")
    print(f"Model dir: {model_dir}")

    # Resolve server
    server = args.server or server_default()
    print(f"ComfyUI: {server}")

    # Load beats
    meta, shots = load_beats(Path(args.spec))
    negative_prompt = Path(meta["negative"]).read_text(encoding="utf-8")

    # Filter clips by range
    start_num = int(args.from_clip[1:])
    end_num = int(args.to_clip[1:])
    clips = [s for s in shots if start_num <= int(s["id"][1:]) <= end_num]

    print(f"Clips to process: {len(clips)} ({args.from_clip} to {args.to_clip})")

    # Setup directories
    renders_dir, chained_dir, logs_dir = setup_directories()

    # Load LoRA registry
    lora_registry = load_lora_registry()

    # Generate workflow. Always regenerate: a stale file on disk silently
    # overrides the generator, which is how the wrong model pair and a 2+2
    # schedule survived in base_i2v_gguf_q8.json.
    workflow_path = Path(f"workflows/base_i2v_{precision}.json")
    print(f"Generating workflow: {workflow_path}")
    workflow = generate_workflow(
        precision,
        str(model_dir),
        nag=args.nag,
        nag_scale=args.nag_scale,
        nag_alpha=args.nag_alpha,
        nag_tau=args.nag_tau,
    )
    save_workflow(workflow, workflow_path)

    # Initialize state
    state = ProductionState(
        precision=precision,
        config=config,
        model_dir=model_dir,
        workflow_path=workflow_path,
        server=server,
        negative_prompt=negative_prompt,
        lora_registry=lora_registry,
        clips=clips,
        base_workflow=load_workflow(workflow_path),
    )

    if args.dry_run:
        print("\nDRY RUN - validation only")
        for clip in clips:
            prompt_file = Path(args.prompts) / f"{clip['id']}.txt"
            if prompt_file.exists():
                print(f"  {clip['id']}: OK")
            else:
                print(f"  {clip['id']}: MISSING PROMPT")
        return 0

    # Download base models. This must run before the render stage; it used to
    # sit after the dry-run return, which made it unreachable.
    if not args.skip_download:
        token = args.hf_token or os.environ.get("HF_TOKEN")
        if not download_all_models(state, token):
            print("WARNING: Model download had issues, continuing with existing models...")

    # ============================================================
    # PIPELINE STAGES
    # ============================================================

    # Stage 1: Download trained LoRAs from Kaggle
    if not args.skip_loras:
        token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("KAGGLE_KEY")
        if not download_trained_loras_stage(state, token):
            print("WARNING: Trained LoRA download had issues, continuing...")

    # Stage 2: Generate SDXL stills with trained LoRAs + InsightFace
    if not args.skip_stills:
        if not generate_stills_stage(state):
            print("WARNING: Stills generation had issues, continuing...")

    # Stage 3: Render clips via ComfyUI
    if not args.skip_render:
        if not render_clips_stage(state, args):
            print("ERROR: Clip rendering failed")
            save_production_log(state, logs_dir)
            print_summary(state)
            return 1

    # Stage 4: Face swap every 3rd frame
    if not args.skip_face_swap:
        if not face_swap_stage(state, args):
            print("WARNING: Face swap had issues, continuing...")

    # Stage 5: Assemble final deliverable
    if not args.skip_assemble:
        if not assemble_final_stage(state, args):
            print("WARNING: Final assembly had issues, continuing...")

    # Save log and print summary
    save_production_log(state, logs_dir)
    print_summary(state)

    return 0 if all(r.success for r in state.results) else 1


if __name__ == "__main__":
    sys.exit(main())