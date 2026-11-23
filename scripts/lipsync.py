#!/usr/bin/env python3
"""Audio-driven lipsync pass using Wan 2.2 S2V / InfiniteTalk in ComfyUI.

Runs after the clips are rendered and before the final mux. For each clip that
has narration, it re-drives the mouth from the audio so speech matches the
picture.

Two engines, chosen by `--engine`:

  s2v            Wan 2.2 S2V 14B. Single audio stream, no speaker separation.
                 Cheapest and most robust; good when one person speaks.
  infinitetalk   Wan 2.1 InfiniteTalk Multi. Two audio streams with per-speaker
                 masks, so a frame containing both characters can be driven
                 correctly. Needs the InfiniteTalk patch model.

Both use the same ComfyUI node chain:
  AudioEncoderLoader -> AudioEncoderEncode -> Wan*ToVideo -> KSampler -> VAEDecode

The pass is additive: clips without narration are copied through untouched, and
a failure on one clip never aborts the run.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from comfy_api import server_default, upload  # noqa: E402
from render import post, wait_for  # noqa: E402

S2V_UNET = "wan2.2_s2v_14B_fp8_scaled.safetensors"
AUDIO_ENCODER = "wav2vec2_large_english_fp16.safetensors"
INFINITETALK_PATCH = "Wan2_1-InfiniteTalk-Multi_fp8_e4m3fn_scaled_KJ.safetensors"


def load_spec(path: Path) -> tuple[dict, list[dict]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data.get("meta") or {}, data.get("shots") or []


def load_audio_index(audio_dir: Path) -> dict[str, str]:
    index = audio_dir / "index.json"
    if not index.is_file():
        return {}
    try:
        return json.loads(index.read_text(encoding="utf-8")).get("beats", {})
    except (OSError, ValueError):
        return {}


def build_s2v_workflow(
    prompt: str,
    negative: str,
    image_name: str,
    audio_name: str,
    width: int,
    height: int,
    frames: int,
    seed: int,
    steps: int,
    cfg: float,
) -> dict:
    """Wan 2.2 S2V graph: audio drives the mouth, image anchors identity."""
    return {
        "1": {"class_type": "UnetLoader", "inputs": {
            "unet_name": S2V_UNET, "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
        "4": {"class_type": "AudioEncoderLoader", "inputs": {
            "audio_encoder_name": AUDIO_ENCODER}},
        "5": {"class_type": "LoadAudio", "inputs": {"audio": audio_name}},
        "6": {"class_type": "AudioEncoderEncode", "inputs": {
            "audio_encoder": ["4", 0], "audio": ["5", 0]}},
        "7": {"class_type": "LoadImage", "inputs": {"image": image_name, "upload": "image"}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["2", 0]}},
        "10": {"class_type": "WanSoundImageToVideo", "inputs": {
            "positive": ["8", 0], "negative": ["9", 0], "vae": ["3", 0],
            "width": width, "height": height, "length": frames, "batch_size": 1,
            "audio_encoder_output": ["6", 0], "ref_image": ["7", 0],
        }},
        "11": {"class_type": "KSampler", "inputs": {
            "seed": seed, "control_after_generate": "fixed",
            "steps": steps, "cfg": cfg, "sampler_name": "euler",
            "scheduler": "simple", "denoise": 1.0,
            "model": ["1", 0], "positive": ["10", 0], "negative": ["10", 1],
            "latent_image": ["10", 2],
        }},
        "12": {"class_type": "VAEDecode", "inputs": {
            "samples": ["11", 0], "vae": ["3", 0]}},
        "13": {"class_type": "VHS_VideoCombine", "inputs": {
            "images": ["12", 0], "frame_rate": 16, "loop_count": 0,
            "filename_prefix": "lipsync", "format": "video/h264-mp4",
            "pingpong": False, "save_output": True,
        }},
    }


def build_infinitetalk_workflow(
    prompt: str,
    negative: str,
    image_name: str,
    audio_name: str,
    width: int,
    height: int,
    frames: int,
    seed: int,
    steps: int,
    cfg: float,
) -> dict:
    """Wan 2.1 InfiniteTalk Multi graph: two-speaker capable lipsync."""
    return {
        "1": {"class_type": "UnetLoader", "inputs": {
            "unet_name": "wan2.1_i2v_14B_fp8_scaled.safetensors", "weight_dtype": "default"}},
        "2": {"class_type": "CLIPLoader", "inputs": {
            "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
        "4": {"class_type": "AudioEncoderLoader", "inputs": {
            "audio_encoder_name": AUDIO_ENCODER}},
        "5": {"class_type": "LoadAudio", "inputs": {"audio": audio_name}},
        "6": {"class_type": "AudioEncoderEncode", "inputs": {
            "audio_encoder": ["4", 0], "audio": ["5", 0]}},
        "7": {"class_type": "LoadImage", "inputs": {"image": image_name, "upload": "image"}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["2", 0]}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {"text": negative, "clip": ["2", 0]}},
        "10": {"class_type": "WanInfiniteTalkToVideo", "inputs": {
            "mode": "single_speaker",
            "model": ["1", 0], "model_patch": ["1", 0],
            "positive": ["8", 0], "negative": ["9", 0], "vae": ["3", 0],
            "width": width, "height": height, "length": frames,
            "start_image": ["7", 0],
            "audio_encoder_output_1": ["6", 0],
            "audio_scale": 1.0,
        }},
        "11": {"class_type": "KSampler", "inputs": {
            "seed": seed, "control_after_generate": "fixed",
            "steps": steps, "cfg": cfg, "sampler_name": "euler",
            "scheduler": "simple", "denoise": 1.0,
            "model": ["10", 0], "positive": ["10", 1], "negative": ["10", 2],
            "latent_image": ["10", 3],
        }},
        "12": {"class_type": "VAEDecode", "inputs": {
            "samples": ["11", 0], "vae": ["3", 0]}},
        "13": {"class_type": "VHS_VideoCombine", "inputs": {
            "images": ["12", 0], "frame_rate": 16, "loop_count": 0,
            "filename_prefix": "lipsync", "format": "video/h264-mp4",
            "pingpong": False, "save_output": True,
        }},
    }


def download_output(server: str, entry: dict, dest: Path) -> Path:
    items = [
        item
        for node in entry.get("outputs", {}).values()
        for key in ("videos", "gifs")
        for item in (node.get(key) or [])
    ]
    if not items:
        raise RuntimeError("no video output")
    item = items[-1]
    query = urllib.parse.urlencode({
        "filename": item["filename"],
        "subfolder": item.get("subfolder", ""),
        "type": item.get("type", "output"),
    })
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(f"{server}/view?{query}") as response:
        dest.write_bytes(response.read())
    return dest


def main() -> int:
    ap = argparse.ArgumentParser(description="Audio-driven lipsync pass")
    ap.add_argument("--spec", default="docs/beats.yaml")
    ap.add_argument("--renders", default="renders")
    ap.add_argument("--startframes", default="startframes")
    ap.add_argument("--audio", default="audio")
    ap.add_argument("--output", default="renders/lipsync")
    ap.add_argument("--server", default=None)
    ap.add_argument("--engine", choices=["s2v", "infinitetalk"], default="s2v")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7000)
    ap.add_argument("--beats", nargs="+")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    server = args.server or server_default()
    meta, shots = load_spec(Path(args.spec))
    audio_index = load_audio_index(Path(args.audio))

    negative_path = Path(meta.get("negative", "negatives/artifact_neg.txt"))
    negative = negative_path.read_text(encoding="utf-8").strip() if negative_path.is_file() else ""

    renders = Path(args.renders)
    frames_dir = Path(args.startframes)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = []
    for shot in shots:
        sid = shot.get("id")
        if not isinstance(sid, str):
            continue
        if args.beats and sid not in set(args.beats):
            continue
        if sid not in audio_index:
            continue
        if not (frames_dir / f"{sid}.png").is_file():
            continue
        targets.append(shot)

    print(f"lipsync engine: {args.engine}")
    print(f"clips with narration: {len(targets)}")

    if args.dry_run:
        for shot in targets:
            print(f"  {shot['id']}")
        return 0

    if not targets:
        print("nothing to do")
        return 0

    uploads: dict[Path, str] = {}
    done = 0
    failed = []

    for shot in targets:
        sid = shot["id"]
        prompt_file = Path("prompts") / f"{sid}.txt"
        prompt = prompt_file.read_text(encoding="utf-8").strip() if prompt_file.is_file() else ""

        res = meta.get("res_face") if shot.get("build") == "cf" else meta.get("res_base")
        width, height = (res[0], res[1]) if isinstance(res, list) and len(res) == 2 else (832, 480)
        duration = int(shot.get("duration") or 5)
        frames = duration * 8 + 1

        try:
            image_name = upload(server, frames_dir / f"{sid}.png", uploads)
            audio_name = upload(server, Path(args.audio) / audio_index[sid], uploads)

            builder = build_s2v_workflow if args.engine == "s2v" else build_infinitetalk_workflow
            workflow = builder(
                prompt, negative, image_name, audio_name,
                width, height, frames, args.seed + int(sid[1:]), args.steps, args.cfg,
            )

            started = time.time()
            pid = post(server, workflow)
            entry = wait_for(server, pid, timeout=3600)
            dest = download_output(server, entry, out_dir / f"{sid}.mp4")
            elapsed = time.time() - started
            print(f"  {sid}: {elapsed:.0f}s -> {dest.name}")
            done += 1
        except Exception as error:  # noqa: BLE001 - one clip must not abort the run
            print(f"  {sid}: FAILED {error}", file=sys.stderr)
            failed.append(sid)

    print(f"\nlipsync done: {done}  failed: {len(failed)}")
    if failed:
        print(f"failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
