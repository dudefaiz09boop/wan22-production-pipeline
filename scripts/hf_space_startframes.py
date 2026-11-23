#!/usr/bin/env python3
"""Generate reference-plate-locked start frames through the hosted Qwen Gradio Space
or AI Horde (free crowdsourced Stable Diffusion) as fallback.

Swap-in backend for submit_startframes.py when no local Kaggle worker is
running. Reuses the production prompt/seed/validation contract and writes the
same prompts/startframe_batch_manifest.json state that
scripts/build_startframe_manifest.py expects, so the manifest provenance merge
works unchanged.

Per-beat detailed prompts, modes, identity reference plates and LoRA choices
come from prompts/startframe_prompts.json (built by
scripts/compose_startframe_prompts.py). Beats without an entry fall back to the
production camera-line prompt with no reference plates.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import submit_startframes as contract


SPACE_ID = "Guaryn/qwen-image-2.1-uncensored-aio-loras"
SPACE_BASE = "https://guaryn-qwen-image-2-1-uncensored-aio-loras.hf.space"
CALL_URL = f"{SPACE_BASE}/gradio_api/call/generate"
EVENT_URL = f"{SPACE_BASE}/gradio_api/call/generate"
UPLOAD_URL = f"{SPACE_BASE}/gradio_api/upload"
ASPECT = "Landscape · 16:9 (1344x768)"
PROMPT_SUFFIX = ", photorealistic, high detail, no text or watermark"
WORKER_ID = f"hf-space:{SPACE_ID}"
OUTPUT_DIR = "startframes"
BATCH_PATH = ROOT / "prompts" / "startframe_batch_manifest.json"
PROMPTS_PATH = ROOT / "prompts" / "startframe_prompts.json"
SPACE_GENERATE_FIELDS = (
    "prompt",
    "mode",
    "ref_image_1",
    "ref_image_2",
    "lora_adapter",
    "lora_strength",
    "custom_repo",
    "custom_file",
    "aspect_ratio",
    "steps",
    "seed",
    "randomize_seed",
)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"

GEMINI_MODELS = ["gemini-2.5-flash-image"]


# ============================ AI Horde Provider ============================

HORDE_BASE = "https://aihorde.net/api/v2"
HORDE_ANON_KEY = "0000000000"
HORDE_API_KEY_FILE = ROOT / ".env"

HORDE_CLIENT = "local-ai-studio:1.0:https://github.com/local-ai-studio"


def read_horde_key() -> str:
    """Read HORDE_API_KEY from .env, fallback to anonymous."""
    if HORDE_API_KEY_FILE.is_file():
        for line in HORDE_API_KEY_FILE.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^\s*HORDE_API_KEY\s*=\s*(.*?)\s*$", line)
            if match:
                value = match.group(1).strip("\"'")
                if value:
                    return value
    return HORDE_ANON_KEY

# Verified NSFW-capable models on AI Horde (from gateway/ai-horde.js)
HORDE_NSFW_MODELS = [
    "WAI-NSFW-illustrious-SDXL",
    "Deliberate",
    "AbsoluteReality",
    "Juggernaut XL",
    "AbyssOrangeMix-AfterDark",
    "Pony Diffusion XL",
    "AbyssOrangeMix-AfterDark",
    "URPM",
    "Dreamshaper",
    "majicMIX realistic",
    "Realistic Vision",
    "ICBINP - I Can't Believe It's Not Photography",
]

HORDE_DEFAULT_MODEL = "AbsoluteReality"


def horde_submit(payload: dict, timeout: int = 30) -> dict:
    import subprocess
    api_key = read_horde_key()
    cmd = [
        "curl", "-s", "-X", "POST",
        "https://aihorde.net/api/v2/generate/async",
        "-H", "accept: application/json",
        "-H", "content-type: application/json",
        "-H", "client-agent: local-ai-studio:1.0:https://github.com/local-ai-studio",
        "-H", f"apikey: {api_key}",
        "-H", "user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "-d", json.dumps(payload),
        "--max-time", str(timeout)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"curl failed: {result.stderr}")
    return json.loads(result.stdout)


def horde_check(horde_id: str, timeout: int = 15) -> dict:
    import subprocess
    api_key = read_horde_key()
    cmd = [
        "curl", "-s",
        f"https://aihorde.net/api/v2/generate/check/{horde_id}",
        "-H", f"apikey: {api_key}",
        "--max-time", str(timeout)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"curl check failed: {result.stderr}")
    return json.loads(result.stdout)


def horde_status(horde_id: str, timeout: int = 30) -> dict:
    import subprocess
    api_key = read_horde_key()
    cmd = [
        "curl", "-s",
        f"https://aihorde.net/api/v2/generate/status/{horde_id}",
        "-H", f"apikey: {api_key}",
        "--max-time", str(timeout)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError(f"curl status failed: {result.stderr}")
    return json.loads(result.stdout)


def horde_wait(horde_id: str, timeout_seconds: float = 900) -> dict:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        check = horde_check(horde_id)
        if check.get("faulted"):
            raise RuntimeError("AI Horde job faulted. Try another model.")
        if check.get("is_possible") is False:
            raise RuntimeError("No AI Horde worker can satisfy this request.")
        if check.get("done"):
            return horde_status(horde_id)
        time.sleep(5)
    raise RuntimeError("AI Horde generation timed out")


def horde_generate(
    prompt: str,
    width: int,
    height: int,
    steps: int = 30,
    cfg: float = 7.5,
    sampler: str = "k_euler_a",
    model: str = HORDE_DEFAULT_MODEL,
    nsfw: bool = True,
    seed: int | None = None,
    timeout_seconds: float = 900,
) -> bytes:
    """Generate image via AI Horde, return PNG bytes.

    Retries when a worker returns a blank/black frame (common on the free
    crowd tier), which is detected from the decoded pixels.
    """
    width = max(64, min(3072, (width // 64) * 64))
    height = max(64, min(3072, (height // 64) * 64))

    params = {
        "width": width,
        "height": height,
        "steps": max(1, min(500, steps)),
        "cfg_scale": max(0, min(100, cfg)),
        "sampler_name": sampler,
        "karras": True,
        "n": 1,
    }
    if seed is not None:
        params["seed"] = str(seed)

    payload = {
        "prompt": prompt.strip(),
        "params": params,
        "nsfw": nsfw,
        "censor_nsfw": False,
        "trusted_workers": False,
        "slow_workers": True,
        "extra_slow_workers": False,
        "r2": False,
        "shared": False,
        "models": [model],
    }

    attempts = 3
    last_error = "AI Horde returned only blank frames"
    for attempt in range(attempts):
        result = horde_submit(payload)
        horde_id = result.get("id")
        if not horde_id:
            last_error = f"AI Horde submit failed: {result}"
            continue

        status = horde_wait(horde_id, timeout_seconds=timeout_seconds)
        generations = status.get("generations", [])
        if not generations:
            last_error = "AI Horde returned no generations"
            continue

        img_b64 = generations[0].get("img")
        if not img_b64:
            last_error = "AI Horde returned no image data"
            continue

        from PIL import Image, ImageStat
        import io

        raw = base64.b64decode(img_b64)
        image = Image.open(io.BytesIO(raw))
        image.load()
        rgb = image.convert("RGB")
        stat = ImageStat.Stat(rgb)
        mean = sum(stat.mean) / 3
        spread = sum(stat.stddev) / 3
        if spread < 3.0 or mean < 8.0 or mean > 250.0:
            last_error = f"blank frame (mean {mean:.1f}, sd {spread:.1f})"
            time.sleep(3)
            continue
        png_buf = io.BytesIO()
        rgb.save(png_buf, format="PNG")
        return png_buf.getvalue()

    raise RuntimeError(last_error)
    if PROMPTS_PATH.is_file():
        data = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise RuntimeError("prompts/startframe_prompts.json must be an object")
        return data
    return {}


def upload_ref(path: Path, token: str | None) -> dict:
    """Upload a reference plate and return a Gradio FileData payload."""
    boundary = "----startframeplate"
    filename = path.name
    data = path.read_bytes()
    body = b"\r\n".join([
        f"--{boundary}".encode("ascii"),
        f'Content-Disposition: form-data; name="files"; filename="{filename}"'.encode("ascii"),
        b"Content-Type: image/png",
        b"",
        data,
        f"--{boundary}--".encode("ascii"),
        b"",
    ])
    headers = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        UPLOAD_URL, data=body, headers=headers, method="POST"
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        payload = json.loads(response.read())
    if isinstance(payload, list) and payload:
        server_path = payload[0]
    elif isinstance(payload, dict) and payload.get("path"):
        server_path = payload["path"]
    else:
        raise RuntimeError(f"reference upload returned no path: {str(payload)[:200]}")
    return {"path": server_path, "meta": {"_type": "gradio.FileData"}}


def load_overrides() -> dict:
    if PROMPTS_PATH.is_file():
        data = json.loads(PROMPTS_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise RuntimeError("prompts/startframe_prompts.json must be an object")
        return data
    return {}


def read_gemini_key() -> str | None:
    env_file = ROOT / ".env"
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^\s*GEMINI_API_KEY\s*=\s*(.*?)\s*$", line)
            if match:
                value = match.group(1).strip("\"'")
                return value or None
    return None


def provider_for(shot: dict) -> str:
    """Route beats: explicit beats (with NSFW LoRA) -> hf (Guaryn), others -> horde (AI Horde fallback).
    Guaryn has NSFW LoRA for explicit content; AI Horde is fallback for non-explicit.
    """
    lora = shot.get("loras", [])
    # Explicit beats have NSFW LoRAs in beats.yaml
    if lora and any("nsfw" in l.lower() or "penis" in l.lower() or "vag" in l.lower() for l in lora):
        return "hf"
    return "horde"


def gemini_prompt(prompt: str) -> str:
    """Adapt a composed override prompt for the text-only Gemini call.

    The composed prompts identify the leads through their uploaded reference
    images ("the man in image 1"); Gemini receives no images, so the image
    references are dropped while the full identity, wardrobe, scene, setting
    and lighting detail is kept.
    """
    text = prompt
    text = text.replace("The man in image 1 is Faiz", "Faiz")
    text = text.replace("The woman in image 2 is Juveriya", "Juveriya")
    text = text.replace("The man in the image is Faiz", "Faiz")
    text = text.replace("The woman in the image is Juveriya", "Juveriya")
    return text


def gemini_generate(prompt: str, key: str, timeout_seconds: float) -> bytes:
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{GEMINI_MODELS[0]}:generateContent?key={key}"
    )
    body = json.dumps({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "imageConfig": {"aspectRatio": "16:9"},
        },
    }).encode("utf-8")
    payload = None
    for attempt in range(4):
        request = urllib.request.Request(
            url, data=body, headers={"Content-Type": "application/json"}, method="POST"
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                payload = json.loads(response.read())
            break
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", "replace")[:300]
            if error.code in (429, 500, 502, 503, 504) and attempt < 3:
                time.sleep(20 * (attempt + 1))
                continue
            raise RuntimeError(f"gemini HTTP {error.code}: {detail}") from error
        except urllib.error.URLError as error:
            if attempt < 3:
                time.sleep(20 * (attempt + 1))
                continue
            raise RuntimeError(f"gemini network error: {error}") from error
    for candidate in payload.get("candidates", []) if payload else []:
        for part in candidate.get("content", {}).get("parts", []):
            inline = part.get("inlineData")
            if inline and inline.get("data"):
                return base64.b64decode(inline["data"])
    raise RuntimeError(f"gemini returned no image: {json.dumps(payload)[:300]}")


def read_hf_tokens() -> list[str]:
    tokens = []
    env_file = ROOT / ".env"
    if env_file.is_file():
        found = {}
        for line in env_file.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^\s*(HF_TOKEN(?:_ALT(\d*))?)\s*=\s*(.*?)\s*$", line)
            if match:
                found[match.group(1)] = match.group(3).strip("\"'")
        # alt accounts first (freshest quota), numbered order, base token last
        alt_names = sorted(
            (name for name in found if name != "HF_TOKEN"),
            key=lambda name: int(name.rsplit("_ALT", 1)[1] or "1"),
        )
        for name in alt_names + ["HF_TOKEN"]:
            if found.get(name):
                tokens.append(found[name])
    return tokens


def http_json(url: str, payload: dict, token: str | None) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:500]
        if token:
            detail = detail.replace(token, "[redacted]")
        raise RuntimeError(
            f"Hugging Face Space returned HTTP {error.code}: {detail}"
        ) from None


def build_space_call_payload(payload: dict) -> dict:
    return {"data": [payload[field] for field in SPACE_GENERATE_FIELDS]}


def wait_for_result(event_id: str, token: str | None, timeout_seconds: float) -> list:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(f"{EVENT_URL}/{event_id}", headers=headers)
    deadline = time.monotonic() + timeout_seconds
    with urllib.request.urlopen(request, timeout=300) as stream:
        event = None
        while True:
            if time.monotonic() > deadline:
                raise RuntimeError("space generation timed out")
            line = stream.readline()
            if not line:
                raise RuntimeError("space event stream ended without a result")
            text = line.decode("utf-8", "replace").strip()
            if text.startswith("event:"):
                event = text[6:].strip()
            elif text.startswith("data:"):
                data = text[5:].strip()
                if event == "complete":
                    return json.loads(data)
                if event == "error":
                    raise RuntimeError(f"space generation failed: {data}")


def download(url: str, token: str | None) -> bytes:
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read()


def save_frame(png_bytes: bytes, target: Path, width: int, height: int) -> None:
    from PIL import Image
    import io

    image = Image.open(io.BytesIO(png_bytes))
    image.load()
    if image.size != (width, height):
        image = image.resize((width, height), Image.LANCZOS)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".tmp.png")
    image.save(tmp, format="PNG")
    data = tmp.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        tmp.unlink(missing_ok=True)
        raise RuntimeError("resized output failed the PNG signature check")
    tmp.replace(target)


def build_batch(shots: list[dict], configs: dict[str, dict]) -> dict:
    if BATCH_PATH.is_file():
        manifest = json.loads(BATCH_PATH.read_text(encoding="utf-8"))
        if manifest.get("worker_id") != WORKER_ID:
            raise RuntimeError(
                "A start-frame batch exists for a different worker; "
                "finish or remove it before starting this one"
            )
        return manifest
    jobs = {}
    for shot in shots:
        shot_id = shot["id"]
        jobs[shot_id] = {
            "job_id": None,
            "status": "not_submitted",
            "attempts": 0,
            "attempts_detail": [],
            "terminal_status": None,
            "start_frame": f"{OUTPUT_DIR}/{shot_id}.png",
            "prompt_sha256": contract.sha256(
                (ROOT / "prompts" / f"{shot_id}.txt").read_bytes()
            ),
            "previous_output_sha256": None,
            "output_dimensions": [
                int(configs[shot_id]["width"]),
                int(configs[shot_id]["height"]),
            ],
        }
    return {
        "schema_version": 1,
        "worker_id": WORKER_ID,
        "batch_status": "generating",
        "reference_free": True,
        "provisional": True,
        "regenerate_after": "Wave 1a identity LoRA",
        "redo_candidates": contract.REDO_CANDIDATES,
        "provider": SPACE_ID,
        "completed_count": 0,
        "failed_count": 0,
        "pending_count": len(contract.SHOT_IDS),
        "submitted_count": 0,
        "jobs": jobs,
    }


def reset_jobs(batch: dict, shot_ids: list[str]) -> list[str]:
    reset = []
    for shot_id in shot_ids:
        entry = batch["jobs"].get(shot_id)
        if not isinstance(entry, dict):
            raise RuntimeError(f"{shot_id}: no job record to reset")
        entry["status"] = "not_submitted"
        entry["attempts"] = 0
        entry["attempts_detail"] = []
        entry["terminal_status"] = None
        for key in ("output_sha256", "provider", "error", "output_sha256_recorded",
                    "reference_plates", "job_id"):
            entry.pop(key, None)
        reset.append(shot_id)
    return reset


def main(argv: list[str] | None = None) -> int:
    global OUTPUT_DIR, BATCH_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="first_id", default="c01")
    parser.add_argument("--to", dest="last_id", default="c70")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--timeout-seconds", type=float, default=1500,
                        help="per-frame generation wait cap")
    parser.add_argument("--max-consecutive-failures", type=int, default=3)
    parser.add_argument("--attempts", type=int, default=4,
                        help="attempts per beat (transient space proxy errors need retries)")
    parser.add_argument("--providers", default="hf,horde",
                        help="comma list of providers to run this pass (hf, horde)")
    parser.add_argument("--force-provider", default="",
                        help="override provider routing for every beat in this pass (hf, horde, gemini)")
    parser.add_argument("--output-dir", default=OUTPUT_DIR, help="output directory for frames and manifest")
    parser.add_argument("--batch-manifest", default=str(BATCH_PATH), help="batch manifest path")
    parser.add_argument("--redo", default="",
                        help="comma list of beat ids whose manifest entries are cleared first")
    parser.add_argument("--horde-model", default=HORDE_DEFAULT_MODEL,
                        help="AI Horde model used by the horde provider")
    parser.add_argument("--token-slots", default="",
                        help="comma list of HF token indexes to use, in order (default: all)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    allowed_providers = {tag.strip() for tag in args.providers.split(",") if tag.strip()}
    OUTPUT_DIR = args.output_dir
    BATCH_PATH = ROOT / args.batch_manifest

    shots, configs, negative_prompt = contract.load_production_inputs()
    _ = negative_prompt  # the space has no negative-prompt parameter
    overrides = load_overrides()
    wanted = set(contract.select_range(args.first_id, args.last_id)
                 if hasattr(contract, "select_range") else
                 __import__("collect_startframes").select_shot_ids(args.first_id, args.last_id))
    batch = build_batch(shots, configs)

    if args.redo:
        redo_ids = [item.strip() for item in args.redo.split(",") if item.strip()]
        unknown = [rid for rid in redo_ids if rid not in batch["jobs"]]
        if unknown:
            raise RuntimeError(f"Unknown redo beat ids: {', '.join(unknown)}")
        reset = reset_jobs(batch, redo_ids)
        contract.aggregate_batch_status(batch)
        BATCH_PATH.write_text(json.dumps(batch, indent=2), encoding="utf-8")
        wanted |= set(reset)
        print(f"reset for regeneration: {', '.join(reset)}", flush=True)

    if args.dry_run:
        print(f"provider {SPACE_ID} aspect {ASPECT}")
        for shot in shots:
            if shot["id"] not in wanted:
                continue
            job = contract.build_job_request(shot, configs[shot["id"]], negative_prompt)
            override = overrides.get(shot["id"], {})
            prompt = override.get("prompt") or (job["prompt"] + PROMPT_SUFFIX)
            print(f"{shot['id']}: {job['width']}x{job['height']} seed {job['seed']} "
                  f"mode={override.get('mode', 'Text to Image')} "
                  f"lora={override.get('lora_adapter', 'None (Base Uncensored)')} "
                  f"refs={','.join(override.get('refs') or []) or '-'} "
                  f"chars={len(prompt)}")
        return 0

    tokens = read_hf_tokens() or [None]
    if args.token_slots:
        slots = [int(value) for value in args.token_slots.split(",") if value.strip()]
        bad = [slot for slot in slots if slot < 0 or slot >= len(tokens)]
        if bad or not slots:
            raise RuntimeError(f"--token-slots out of range 0..{len(tokens) - 1}: {bad}")
        tokens = [tokens[slot] for slot in slots]
        print(f"using HF token slots {slots}", flush=True)
    token_index = 0
    gemini_key = read_gemini_key()
    upload_cache: dict[tuple[str, int], dict] = {}
    consecutive_failures = 0
    for shot in shots:
        shot_id = shot["id"]
        if shot_id not in wanted:
            continue
        entry = batch["jobs"][shot_id]
        provider = args.force_provider or provider_for(shot)
        if provider not in allowed_providers:
            continue
        target = ROOT / OUTPUT_DIR / f"{shot_id}.png"
        dims = entry["output_dimensions"]
        if entry.get("terminal_status") == "succeeded" and target.is_file():
            if contract.sha256(target.read_bytes()) == entry.get("output_sha256"):
                continue
        job = contract.build_job_request(shot, configs[shot_id], negative_prompt)
        override = overrides.get(shot_id, {})
        prompt = override.get("prompt") or (job["prompt"] + PROMPT_SUFFIX)
        if len(prompt) > 4000:
            raise RuntimeError(f"{shot_id}: prompt is {len(prompt)} chars (space cap 4000)")
        mode = override.get("mode", "Text to Image")
        ref_plates = list(override.get("refs") or [])
        lora = override.get("lora_adapter", "None (Base Uncensored)")
        lora_strength = float(override.get("lora_strength", 1.0))
        steps = int(override.get("steps", args.steps))
        payload = {
            "prompt": prompt,
            "mode": mode,
            "ref_image_1": None,
            "ref_image_2": None,
            "lora_adapter": lora,
            "lora_strength": lora_strength,
            "custom_repo": "",
            "custom_file": "",
            "aspect_ratio": ASPECT,
            "steps": steps,
            "seed": job["seed"],
            "randomize_seed": False,
        }
        ok = False
        while entry.get("attempts", 0) < args.attempts and not ok:
            entry["attempts"] = int(entry.get("attempts", 0)) + 1
            entry.setdefault("attempts_detail", []).append(
                {"provider": provider, "status": "submitting"}
            )
            try:
                if provider == "gemini":
                    if not gemini_key:
                        raise RuntimeError("GEMINI_API_KEY missing while gemini beats enabled")
                    # Full composed override prompt (identity, wardrobe, scene,
                    # setting, lighting) with the image references stripped.
                    png_bytes = gemini_generate(
                        gemini_prompt(prompt), gemini_key, args.timeout_seconds
                    )
                    # Gemini may answer with a non-PNG payload; normalise.
                    from PIL import Image
                    import io
                    img = Image.open(io.BytesIO(png_bytes))
                    img.load()
                    if img.mode not in ("RGB", "RGBA"):
                        img = img.convert("RGB")
                    stat = Image.Stat(img)
                    spread = max(stat.stddev)
                    mean = max(stat.mean)
                    if spread < 3 or mean < 8 or mean > 250:
                        raise RuntimeError(
                            f"gemini blank frame (mean {mean:.1f}, sd {spread:.1f})")
                    buf = io.BytesIO()
                    img.save(buf, format="PNG")
                    png_bytes = buf.getvalue()
                elif provider == "horde":
                    # AI Horde fallback: use override prompt, resize to Horde dims
                    horde_prompt = prompt
                    horde_width = dims[0]
                    horde_height = dims[1]
                    # AI Horde needs dimensions multiple of 64
                    horde_width = max(64, min(3072, (horde_width // 64) * 64))
                    horde_height = max(64, min(3072, (horde_height // 64) * 64))
                    # Cap to avoid KudosUpfront for requests over 576x576
                    max_horde_dim = 576
                    if horde_width > max_horde_dim:
                        horde_width = max_horde_dim
                    if horde_height > max_horde_dim:
                        horde_height = max_horde_dim
                    horde_width = (horde_width // 64) * 64
                    horde_height = (horde_height // 64) * 64
                    png_bytes = horde_generate(
                        prompt=horde_prompt,
                        width=horde_width,
                        height=horde_height,
                        steps=steps,
                        model=args.horde_model,
                        nsfw=True,
                        seed=job["seed"],
                        timeout_seconds=args.timeout_seconds,
                    )
                    # Resize back to target dims if Horde returned different size
                    from PIL import Image
                    import io
                    img = Image.open(io.BytesIO(png_bytes))
                    if img.size != (dims[0], dims[1]):
                        img = img.resize((dims[0], dims[1]), Image.LANCZOS)
                        buf = io.BytesIO()
                        img.save(buf, format="PNG")
                        png_bytes = buf.getvalue()
                else:
                    token = tokens[token_index]
                    if ref_plates:
                        if len(ref_plates) > 2:
                            raise RuntimeError(f"{shot_id}: more than two reference plates")
                        for slot, ref_plate in enumerate(ref_plates, start=1):
                            cache_key = (ref_plate, token_index)
                            if cache_key not in upload_cache:
                                upload_cache[cache_key] = upload_ref(
                                    ROOT / ref_plate, token
                                )
                                print(f"{shot_id}: uploaded reference plate {ref_plate}",
                                      flush=True)
                            payload[f"ref_image_{slot}"] = upload_cache[cache_key]
                    response = http_json(
                        CALL_URL, build_space_call_payload(payload), token
                    )
                    event_id = response.get("event_id")
                    if not event_id:
                        raise RuntimeError("space did not return an event id")
                    result = wait_for_result(event_id, token, args.timeout_seconds)
                    file_url = None
                    if isinstance(result, (list, tuple)) and len(result) > 1 and isinstance(result[1], dict):
                        file_url = result[1].get("url")
                    if not file_url and isinstance(result, (list, tuple)) and isinstance(result[0], dict):
                        file_url = result[0].get("url")
                    if not file_url:
                        raise RuntimeError(f"space result carried no file url: {str(result)[:200]}")
                    png_bytes = download(file_url, token)
                if not png_bytes.startswith(PNG_SIGNATURE):
                    raise RuntimeError("downloaded file is not a PNG")
                save_frame(png_bytes, target, dims[0], dims[1])
                entry["output_sha256"] = contract.sha256(target.read_bytes())
                entry["provider"] = provider
                entry["generation"] = {
                    "mode": mode,
                    "lora_adapter": lora,
                    "steps": steps,
                    "prompt_source": "prompts/startframe_prompts.json"
                    if override else "camera-line fallback",
                }
                if ref_plates:
                    entry["reference_plates"] = ref_plates
                contract.update_terminal_status(entry, "succeeded")
                entry["output_sha256_recorded"] = True
                ok = True
                consecutive_failures = 0
                print(f"{shot_id}: frame written ({dims[0]}x{dims[1]})", flush=True)
            except Exception as error:
                message = str(error)
                # Only real quota/429 walls rotate tokens. The space's own
                # "Expired ZeroGPU proxy token" message is a transient space-side
                # error, so it retries the same slot instead of burning a beat.
                quota_hit = (
                    "429" in message
                    or "quota" in message.lower()
                    or "exceeded" in message.lower()
                )
                if quota_hit and token_index + 1 < len(tokens):
                    token_index += 1
                    entry["attempts"] = max(0, int(entry.get("attempts", 0)) - 1)
                    print("space quota hit; switching HF account token", flush=True)
                    continue
                contract.update_terminal_status(entry, "failed", message)
                consecutive_failures += 1
                print(f"{shot_id}: attempt failed: {message}", flush=True)
                if quota_hit:
                    print("space quota exhausted on all accounts; batch paused (resumable)", flush=True)
                    contract.aggregate_batch_status(batch)
                    BATCH_PATH.write_text(json.dumps(batch, indent=2), encoding="utf-8")
                    return 1
                if "proxy token" in message.lower():
                    # Transient space-side proxy expiry; give the space time to
                    # mint a fresh token before the next attempt.
                    time.sleep(20)
                else:
                    time.sleep(5 * entry["attempts"])
        contract.aggregate_batch_status(batch)
        BATCH_PATH.write_text(json.dumps(batch, indent=2), encoding="utf-8")
        if not ok:
            if consecutive_failures >= args.max_consecutive_failures:
                print("too many consecutive failures; batch stopped (resumable)", flush=True)
                return 1
            continue
    contract.aggregate_batch_status(batch)
    BATCH_PATH.write_text(json.dumps(batch, indent=2), encoding="utf-8")
    remaining = [s for s in contract.SHOT_IDS
                 if batch["jobs"][s].get("terminal_status") != "succeeded"]
    print(f"batch complete-ish: {batch['completed_count']}/70 succeeded; "
          f"remaining {remaining}", flush=True)
    return 0 if not remaining else 1


if __name__ == "__main__":
    raise SystemExit(main())
