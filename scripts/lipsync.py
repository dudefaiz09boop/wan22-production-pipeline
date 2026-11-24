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
import os
import shutil
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from comfy_api import server_default, upload  # noqa: E402
from generate_workflow import frames_for  # noqa: E402
from render import post, wait_for  # noqa: E402

S2V_UNET = "wan2.2_s2v_14B_fp8_scaled.safetensors"
AUDIO_ENCODER = "wav2vec2_large_english_fp16.safetensors"
# InfiniteTalk runs on Wan 2.1 I2V 14B plus a separate audio patch model.
INFINITETALK_BASE = "wan2.1_i2v_480p_14B_fp8_scaled.safetensors"
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


def index_entry(index: dict, clip_id: str) -> tuple[str | None, dict[str, str]]:
    """Normalise an index value to (mix filename, {speaker: stem filename}).

    Older indexes store a bare filename; two-speaker clips store
    ``{"mix": ..., "stems": {...}}``.
    """
    value = index.get(clip_id)
    if isinstance(value, str):
        return value, {}
    if isinstance(value, dict):
        mix = value.get("mix")
        stems = value.get("stems")
        return (
            mix if isinstance(mix, str) else None,
            stems if isinstance(stems, dict) else {},
        )
    return None, {}


def speaker_masks(
    image_path: Path, speakers: list[str], width: int, height: int
) -> dict[str, "object"]:
    """Build one soft mask per speaker from the start frame.

    InfiniteTalk's two_speakers mode needs a mask per person so each audio
    stream drives only its own mouth. InsightFace gives the face boxes; each
    box is grown to cover the jaw and neck and blurred so the boundary does
    not seam. Falls back to a left/right split when detection finds fewer
    faces than speakers.
    """
    import cv2
    import numpy as np

    frame = cv2.imread(str(image_path))
    if frame is None:
        raise RuntimeError(f"cannot read {image_path}")
    frame = cv2.resize(frame, (width, height))

    boxes: list[tuple[int, int, int, int]] = []
    try:
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(
            name="buffalo_l",
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        app.prepare(ctx_id=0, det_size=(640, 640))
        faces = app.get(frame)
        faces.sort(key=lambda f: f.bbox[0])
        for face in faces:
            x1, y1, x2, y2 = (int(v) for v in face.bbox)
            w, h = x2 - x1, y2 - y1
            # Grow to jaw and neck, then clamp.
            x1 = max(0, x1 - int(w * 0.45))
            x2 = min(width, x2 + int(w * 0.45))
            y2 = min(height, y2 + int(h * 0.85))
            boxes.append((x1, y1, x2, y2))
    except Exception as error:  # noqa: BLE001 - masks are best-effort
        print(f"  face detection failed ({error}); using split masks", file=sys.stderr)

    if len(boxes) < len(speakers):
        boxes = []
        for index in range(len(speakers)):
            x1 = 0 if index == 0 else width // 2
            x2 = width // 2 if index == 0 else width
            boxes.append((x1, 0, x2, height))

    masks = {}
    for speaker, (x1, y1, x2, y2) in zip(speakers, boxes):
        mask = np.zeros((height, width), dtype=np.float32)
        mask[y1:y2, x1:x2] = 1.0
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=max(8, width // 40))
        masks[speaker] = mask
    return masks


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
    audio_name_2: str | None = None,
    mask_name_1: str | None = None,
    mask_name_2: str | None = None,
) -> dict:
    """Wan 2.1 InfiniteTalk Multi graph: two-speaker capable lipsync.

    `model_patch` is not a diffusion model: it is the InfiniteTalk patch
    loaded by ModelPatchLoader, which detects `audio_proj.proj1.weight` and
    builds a MultiTalkModelPatch. The base model is Wan 2.1 I2V 14B.

    With a second audio stream and two masks the node runs in `two_speakers`
    mode, so each voice drives only the mouth inside its own mask. Without
    them it falls back to `single_speaker` on the mixed track.

    Masks are uploaded as PNGs and read through LoadImage's MASK output
    (slot 1), which is the standard way to feed a mask into a node.
    """
    two_speaker = bool(audio_name_2 and mask_name_1 and mask_name_2)

    workflow: dict = {
        "1": {"class_type": "UnetLoader", "inputs": {
            "unet_name": INFINITETALK_BASE, "weight_dtype": "default"}},
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
        "10": {"class_type": "ModelPatchLoader", "inputs": {
            "name": INFINITETALK_PATCH}},
    }

    talk_inputs: dict = {
        "model": ["1", 0], "model_patch": ["10", 0],
        "positive": ["8", 0], "negative": ["9", 0], "vae": ["3", 0],
        "width": width, "height": height, "length": frames,
        "start_image": ["7", 0],
        "audio_encoder_output_1": ["6", 0],
        "audio_scale": 1.0,
    }

    if two_speaker:
        workflow["15"] = {"class_type": "LoadAudio", "inputs": {"audio": audio_name_2}}
        workflow["16"] = {"class_type": "AudioEncoderEncode", "inputs": {
            "audio_encoder": ["4", 0], "audio": ["15", 0]}}
        workflow["17"] = {"class_type": "LoadImage", "inputs": {
            "image": mask_name_1, "upload": "image"}}
        workflow["18"] = {"class_type": "LoadImage", "inputs": {
            "image": mask_name_2, "upload": "image"}}
        talk_inputs["mode"] = {
            "mode": "two_speakers",
            "audio_encoder_output_2": ["16", 0],
            "mask_1": ["17", 1],
            "mask_2": ["18", 1],
        }
    else:
        talk_inputs["mode"] = {"mode": "single_speaker"}

    workflow["11"] = {"class_type": "WanInfiniteTalkToVideo", "inputs": talk_inputs}
    workflow["12"] = {"class_type": "KSampler", "inputs": {
        "seed": seed, "control_after_generate": "fixed",
        "steps": steps, "cfg": cfg, "sampler_name": "euler",
        "scheduler": "simple", "denoise": 1.0,
        "model": ["11", 0], "positive": ["11", 1], "negative": ["11", 2],
        "latent_image": ["11", 3],
    }}
    workflow["13"] = {"class_type": "VAEDecode", "inputs": {
        "samples": ["12", 0], "vae": ["3", 0]}}
    workflow["14"] = {"class_type": "VHS_VideoCombine", "inputs": {
        "images": ["13", 0], "frame_rate": 16, "loop_count": 0,
        "filename_prefix": "lipsync", "format": "video/h264-mp4",
        "pingpong": False, "save_output": True,
    }}
    return workflow


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


def load_env(path: Path = Path(".env")) -> None:
    """Load .env so LIPSYNC_ENGINE and friends apply without exporting."""
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


def main() -> int:
    load_env()
    ap = argparse.ArgumentParser(description="Audio-driven lipsync pass")
    ap.add_argument("--spec", default="docs/beats.yaml")
    ap.add_argument("--renders", default="renders")
    ap.add_argument("--startframes", default="startframes")
    ap.add_argument("--audio", default="audio")
    ap.add_argument("--output", default="renders/lipsync")
    ap.add_argument("--server", default=None)
    ap.add_argument(
        "--engine",
        choices=["s2v", "infinitetalk"],
        default=os.environ.get("LIPSYNC_ENGINE", "infinitetalk"),
    )
    ap.add_argument("--steps", type=int, default=int(os.environ.get("LIPSYNC_STEPS", "6")))
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
    skipped_narration_only = []
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
        # Narration-only beats have no speaker on screen. Driving a mouth from
        # a voice-over animates whatever face happens to be in frame -- or a
        # buzzer box, for c48 -- so they are muxed but never lipsynced.
        # A beat counts as spoken if it has `say` OR dialogue turns.
        has_dialogue = bool(shot.get("dialogue"))
        if not (shot.get("say") or "").strip() and not has_dialogue:
            skipped_narration_only.append(sid)
            continue
        # Ammi is never on screen: her lines come from a phone or through a
        # door. A beat where she is the only speaker has no mouth to drive.
        speakers = {
            str(t.get("speaker")).lower()
            for t in (shot.get("dialogue") or [])
            if isinstance(t, dict)
        }
        if shot.get("speaker"):
            speakers.add(str(shot["speaker"]).lower())
        if speakers and speakers <= {"ammi"}:
            skipped_narration_only.append(sid)
            continue
        targets.append(shot)

        # Every clip the pass does not touch is copied through, so the output
        # directory is a complete cut rather than only the lipsynced subset.
        # Without this, pointing the mux at renders/lipsync silently drops the
        # 45 beats that were never targets.
        passthrough: list[str] = []
        target_ids = {s["id"] for s in targets}
        for shot in shots:
            sid = shot.get("id")
            if not isinstance(sid, str) or sid in target_ids:
                continue
            if args.beats and sid not in set(args.beats):
                continue
            source = renders / f"{sid}.mp4"
            if source.is_file():
                passthrough.append(sid)

    print(f"lipsync engine: {args.engine}")
    print(f"clips with narration: {len(targets)}")
    if skipped_narration_only:
        print(
            f"skipped (narration only, no speaker on screen): "
            f"{len(skipped_narration_only)}"
        )
    print(f"copied through untouched: {len(passthrough)}")

    if args.dry_run:
        for shot in targets:
            print(f"  {shot['id']}")
        return 0

    if not targets:
        print("nothing to do")
        return 0

    # Copy the untouched clips first so a later failure still leaves a
    # complete cut on disk.
    for sid in passthrough:
        shutil.copy2(renders / f"{sid}.mp4", out_dir / f"{sid}.mp4")

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
        # 16 fps to match the render and the VHS_VideoCombine frame_rate.
        # (length - 1) % 4 == 0 is required by WanImageToVideo.
        frames = frames_for(duration)

        try:
            image_name = upload(server, frames_dir / f"{sid}.png", uploads)
            mix_name, stems = index_entry(audio_index, sid)
            if not mix_name:
                raise RuntimeError("no audio for this beat")
            audio_name = upload(server, Path(args.audio) / mix_name, uploads)

            builder = build_s2v_workflow if args.engine == "s2v" else build_infinitetalk_workflow
            extra: dict = {}
            if args.engine == "infinitetalk" and len(stems) > 1:
                # Two people speak in this frame: drive each mouth from its own
                # stem inside its own mask instead of both from the mix.
                speakers = sorted(stems)
                masks = speaker_masks(frames_dir / f"{sid}.png", speakers, width, height)
                mask_names = []
                for speaker in speakers:
                    mask_path = out_dir / f"{sid}_{speaker}_mask.png"
                    import cv2

                    cv2.imwrite(str(mask_path), (masks[speaker] * 255).astype("uint8"))
                    mask_names.append(upload(server, mask_path, uploads))
                extra = {
                    "audio_name_2": upload(
                        server, Path(args.audio) / stems[speakers[1]], uploads
                    ),
                    "mask_name_1": mask_names[0],
                    "mask_name_2": mask_names[1],
                }

            workflow = builder(
                prompt, negative, image_name, audio_name,
                width, height, frames, args.seed + int(sid[1:]), args.steps, args.cfg,
                **extra,
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
