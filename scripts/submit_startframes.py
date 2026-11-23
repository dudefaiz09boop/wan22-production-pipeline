#!/usr/bin/env python3
"""Generate resumable Qwen start frames from production or detailed Space prompts."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
import zlib
from collections import deque
from pathlib import Path

import yaml

import collect_startframes


ROOT = Path(__file__).resolve().parents[1]
SHOT_IDS = [f"c{number:02d}" for number in range(1, 71)]
QWEN_CHECKPOINT = "qwen-image-2.1-step90000-FP8-comfy.safetensors"
QWEN_SPACE_CHECKPOINT = "qwen-image-2.1-UC-Q4_K_M.gguf"
QWEN_SPACE_PROMPT_SUFFIX = ", photorealistic, high detail, no text or watermark"
REDO_CANDIDATES = ["c25", "c33", "c44", "c61", "c68"]
AMMI_DEPICTIONS = {
    "c34": "phone",
    "c35": "phone",
    "c36": "phone",
    "c37": "phone",
    "c48": "voice",
    "c49": "hands",
    "c50": "hands",
    "c51": "voice",
    "c52": "voice",
    "c54": "voice",
    "c56": "voice",
    "c65": "voice",
    "c66": "voice",
    "c67": "hands",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_job_request(
    shot: dict,
    config: dict,
    negative_prompt: str,
) -> dict:
    camera_line = shot["cam"][0]
    if ":" not in camera_line:
        raise RuntimeError(f"{shot['id']}: first camera line is missing its verb separator")
    return {
        "prompt": camera_line.split(":", 1)[1].strip(),
        "negative_prompt": negative_prompt,
        "width": int(config["width"]),
        "height": int(config["height"]),
        "seed": zlib.crc32(shot["id"].encode("ascii")) & 0x7FFFFFFF,
    }


def build_qwen_space_request(
    shot: dict,
    config: dict,
    override: dict,
    negative_prompt: str,
    default_steps: int = 30,
) -> dict:
    if not isinstance(override, dict):
        raise RuntimeError(f"{shot['id']}: Qwen Space prompt override must be an object")
    base = build_job_request(shot, config, negative_prompt)
    prompt = override.get("prompt") or (base["prompt"] + QWEN_SPACE_PROMPT_SUFFIX)
    if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 4000:
        raise RuntimeError(f"{shot['id']}: Qwen Space prompt must be 1 to 4000 characters")
    mode = override.get("mode", "Text to Image")
    reference_plates = override.get("refs") or []
    required_references = {
        "Text to Image": 0,
        "Edit Image (1 Ref)": 1,
        "Transform & Swap (2 Refs)": 2,
    }
    if not isinstance(mode, str) or mode not in required_references:
        raise RuntimeError(f"{shot['id']}: unsupported Qwen Space mode {mode!r}")
    if not isinstance(reference_plates, list) or len(reference_plates) != required_references[mode]:
        raise RuntimeError(f"{shot['id']}: {mode} has an invalid number of reference images")
    lora_adapter = override.get("lora_adapter", "None (Base Uncensored)")
    if lora_adapter not in {"None (Base Uncensored)", "NSFW (general)"}:
        raise RuntimeError(f"{shot['id']}: unsupported Qwen Space LoRA {lora_adapter!r}")
    lora_strength = float(override.get("lora_strength", 1.0))
    if not 0 <= lora_strength <= 1.5:
        raise RuntimeError(f"{shot['id']}: Qwen Space LoRA strength must be from 0 to 1.5")
    steps = int(override.get("steps", default_steps))
    if not 4 <= steps <= 40:
        raise RuntimeError(f"{shot['id']}: Qwen Space steps must be from 4 to 40")
    seed = override.get("seed", base["seed"])
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 0x7FFFFFFF:
        raise RuntimeError(f"{shot['id']}: Qwen Space seed must be from 0 to 2147483647")
    effective_negative_prompt = override.get("negative_prompt", negative_prompt)
    if not isinstance(effective_negative_prompt, str):
        raise RuntimeError(f"{shot['id']}: Qwen Space negative prompt must be a string")
    return {
        "prompt": prompt,
        "negative_prompt": effective_negative_prompt,
        "mode": mode,
        "width": base["width"],
        "height": base["height"],
        "steps": steps,
        "seed": seed,
        "lora_adapter": lora_adapter,
        "lora_strength": lora_strength,
        "reference_plates": reference_plates,
    }


def encode_reference_images(reference_plates: list[str]) -> list[dict]:
    references = []
    for reference_plate in reference_plates:
        source = (ROOT / reference_plate).resolve()
        if not source.is_relative_to(ROOT.resolve()) or not source.is_file():
            raise RuntimeError(f"Reference image is missing or outside the repository: {reference_plate}")
        data = source.read_bytes()
        if len(data) > 2 * 1024 * 1024:
            raise RuntimeError(f"Reference image exceeds the Kaggle worker limit: {reference_plate}")
        references.append({
            "data": base64.b64encode(data).decode("ascii"),
            "label": Path(reference_plate).as_posix(),
        })
    return references


def build_request_for_shot(
    model: str,
    shot: dict,
    config: dict,
    override: dict,
    negative_prompt: str,
    steps: int,
    entry: dict,
) -> dict:
    """Build one worker request and fingerprint its Qwen Space inputs.

    The fingerprint keeps a resumed batch honest: if a detailed prompt or a
    reference image changed after a shot was attempted, the retry is refused
    instead of silently mixing generations.
    """
    if model != QWEN_SPACE_CHECKPOINT:
        return build_job_request(shot, config, negative_prompt)
    request = build_qwen_space_request(shot, config, override, negative_prompt, steps)
    reference_plates = request.pop("reference_plates")
    request["reference_images"] = encode_reference_images(reference_plates)
    fingerprint_data = {
        key: value for key, value in request.items() if key != "reference_images"
    }
    fingerprint_data["reference_hashes"] = [
        sha256(base64.b64decode(item["data"]))
        for item in request["reference_images"]
    ]
    request_sha256 = sha256(json.dumps(
        fingerprint_data, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8"))
    if entry.get("attempts", 0) and entry.get("request_sha256") != request_sha256:
        raise RuntimeError(
            f"{shot['id']}: detailed prompt or reference inputs changed since this batch started",
        )
    entry["request_sha256"] = request_sha256
    entry["reference_plates"] = reference_plates
    return request


def validate_visual_contract(shots: list[dict]) -> None:
    for shot in shots:
        classification = AMMI_DEPICTIONS.get(shot["id"], "no Ammi")
        if classification == "body":
            raise RuntimeError(
                f"{shot['id']}: Ammi depiction is body, "
                "contradicting the voice-and-hands-only production contract"
            )


def load_production_inputs() -> tuple[list[dict], dict[str, dict], str]:
    spec = yaml.safe_load((ROOT / "docs" / "beats.yaml").read_text(encoding="utf-8"))
    shots = spec.get("shots") if isinstance(spec, dict) else None
    shot_ids = [shot.get("id") for shot in shots or [] if isinstance(shot, dict)]
    if shot_ids != SHOT_IDS:
        raise RuntimeError("docs/beats.yaml must contain ordered shots c01 through c70")
    meta_duration = spec.get("meta", {}).get("duration")
    if meta_duration != 5:
        raise RuntimeError("Start-frame generation requires meta.duration to be five seconds")
    for shot in shots:
        noise = shot.get("noise")
        duration = shot.get("duration")
        if noise == 3:
            if not isinstance(duration, int) or duration > 4:
                raise RuntimeError(f"{shot['id']}: noise 3 requires duration <= 4")
        elif duration != meta_duration:
            raise RuntimeError(f"{shot['id']}: duration must match meta.duration ({meta_duration})")
    validate_visual_contract(shots)
    configs = {
        shot_id: json.loads((ROOT / "prompts" / f"{shot_id}.json").read_text(encoding="utf-8"))
        for shot_id in SHOT_IDS
    }
    negative_path = ROOT / spec["meta"]["negative"]
    negative_prompt = negative_path.read_text(encoding="utf-8").strip()
    if not negative_prompt:
        raise RuntimeError("The production negative prompt is empty")
    return shots, configs, negative_prompt


def load_current_manifest() -> dict:
    path = ROOT / "startframes" / "manifest.json"
    if not path.is_file():
        raise RuntimeError("startframes/manifest.json is missing")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    jobs = manifest.get("jobs")
    if not isinstance(jobs, dict) or set(jobs) != set(SHOT_IDS):
        raise RuntimeError("The current start-frame manifest must contain c01 through c70")
    batch_path = ROOT / "prompts" / "startframe_batch_manifest.json"
    batch_jobs = {}
    if batch_path.is_file():
        batch = json.loads(batch_path.read_text(encoding="utf-8"))
        batch_jobs = batch.get("jobs", {})
    for shot_id in SHOT_IDS:
        frame_path = ROOT / f"startframes/{shot_id}.png"
        previous_hash = jobs[shot_id].get("output_sha256")
        batch_hash = (
            batch_jobs.get(shot_id, {}).get("output_sha256")
            if isinstance(batch_jobs.get(shot_id), dict)
            else None
        )
        if not frame_path.is_file():
            raise RuntimeError(f"{shot_id}: existing frame hash is missing from the manifest")
        actual_hash = sha256(frame_path.read_bytes())
        if actual_hash not in {previous_hash, batch_hash}:
            raise RuntimeError(f"{shot_id}: existing frame no longer matches its manifest hash")
    return manifest


def prepare_batch_manifest(
    worker_id: str,
    current_manifest: dict,
    shots: list[dict],
    configs: dict[str, dict],
    manifest_path: Path | None = None,
    model: str = QWEN_CHECKPOINT,
    reference_free: bool = True,
    selected_ids: list[str] | None = None,
) -> dict:
    selected_ids = selected_ids or SHOT_IDS
    if (
        not selected_ids
        or any(shot_id not in SHOT_IDS for shot_id in selected_ids)
        or len(set(selected_ids)) != len(selected_ids)
    ):
        raise RuntimeError("The selected start-frame ids must be unique ids from c01 through c70")
    batch_path = manifest_path or ROOT / "prompts" / "startframe_batch_manifest.json"
    if batch_path.is_file():
        manifest = json.loads(batch_path.read_text(encoding="utf-8"))
        if manifest.get("worker_id") != worker_id:
            raise RuntimeError(
                "A start-frame batch exists for a different worker; "
                "collect or remove it before starting another batch"
            )
        jobs = manifest.get("jobs")
        if not isinstance(jobs, dict) or set(jobs) != set(SHOT_IDS):
            raise RuntimeError("The existing start-frame batch manifest is incomplete")
        previous_selection = manifest.get("selected_ids")
        if previous_selection is None:
            if selected_ids != SHOT_IDS:
                raise RuntimeError(
                    "The existing batch has no selected-id record; use a new state file "
                    "for a sparse batch"
                )
        elif previous_selection != selected_ids:
            raise RuntimeError("The existing start-frame batch uses a different beat selection")
        if manifest.get("model", QWEN_CHECKPOINT) != model:
            raise RuntimeError("The existing start-frame batch uses a different model")
        if model == QWEN_CHECKPOINT and manifest.get("reference_free") is not True:
            raise RuntimeError("Existing batch was not created in reference-free Qwen mode")
        for entry in jobs.values():
            if not isinstance(entry, dict):
                raise RuntimeError("The existing start-frame batch contains an invalid job record")
            entry.setdefault("attempts", 1 if entry.get("job_id") else 0)
            entry.setdefault("attempts_detail", [])
            entry.setdefault(
                "terminal_status",
                "succeeded" if entry.get("status") in {"completed", "succeeded"} else None,
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
            "start_frame": f"startframes/{shot_id}.png",
            "prompt_sha256": sha256((ROOT / "prompts" / f"{shot_id}.txt").read_bytes()),
            "previous_output_sha256": current_manifest["jobs"][shot_id]["output_sha256"],
            "output_dimensions": [int(configs[shot_id]["width"]), int(configs[shot_id]["height"])],
        }
    return {
        "schema_version": 1,
        "worker_id": worker_id,
        "model": model,
        "selected_ids": selected_ids,
        "batch_status": "submitting",
        "reference_free": reference_free,
        "provisional": True,
        "regenerate_after": "Wave 1a identity LoRA",
        "redo_candidates": REDO_CANDIDATES,
        "completed_count": 0,
        "failed_count": 0,
        "pending_count": len(selected_ids),
        "submitted_count": 0,
        "jobs": jobs,
    }


ACTIVE_TASK_STATUSES = {"queued", "running", "submitting", "submitted", "processing"}
FAILED_TASK_STATUSES = {"failed", "cancelled", "stopped"}


def update_terminal_status(entry: dict, status: str, error: str | None = None) -> None:
    entry["status"] = status
    details = entry.setdefault("attempts_detail", [])
    if details:
        details[-1]["status"] = status
    if status in {"completed", "succeeded"}:
        entry["terminal_status"] = "succeeded"
        if details:
            details[-1]["terminal_status"] = "succeeded"
        entry.pop("error", None)
    elif status in FAILED_TASK_STATUSES:
        entry["terminal_status"] = "failed"
        if details:
            details[-1]["terminal_status"] = "failed"
        if error:
            entry["error"] = error
    elif status in ACTIVE_TASK_STATUSES:
        entry["terminal_status"] = None


def is_succeeded(entry: dict) -> bool:
    return entry.get("terminal_status") == "succeeded" and bool(entry.get("output_sha256"))


def aggregate_batch_status(manifest: dict) -> None:
    jobs = manifest["jobs"]
    selected_ids = manifest.get("selected_ids", SHOT_IDS)
    selected_jobs = [jobs[shot_id] for shot_id in selected_ids]
    manifest["submitted_count"] = sum(
        isinstance(entry.get("job_id"), str) for entry in selected_jobs
    )
    manifest["completed_count"] = sum(
        entry.get("terminal_status") == "succeeded" and bool(entry.get("output_sha256"))
        for entry in selected_jobs
    )
    manifest["failed_count"] = sum(
        entry.get("terminal_status") == "failed"
        and int(entry.get("attempts", 0)) >= 2
        for entry in selected_jobs
    )
    manifest["pending_count"] = len(selected_ids) - (
        manifest["completed_count"] + manifest["failed_count"]
    )
    if manifest["pending_count"]:
        manifest["batch_status"] = "generating"
    else:
        manifest["batch_status"] = "images_failed" if manifest["failed_count"] else "generated"


def read_model_store_url() -> str:
    env_file = ROOT / ".env"
    env_values = {}
    if env_file.is_file():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            match = re.match(r"^\s*(DATA_DIR|KAGGLE_MODEL_STORE_URL)\s*=\s*(.*?)\s*$", line)
            if match:
                env_values[match.group(1)] = match.group(2).strip("\"'")
    data_dir = Path(os.environ.get("DATA_DIR") or env_values.get("DATA_DIR") or "D:/AI/data")
    url_file = data_dir / "logs" / "model-store-url.txt"
    if url_file.is_file():
        value = url_file.read_text(encoding="utf-8").strip()
        if value:
            return value.rstrip("/")
    value = os.environ.get("KAGGLE_MODEL_STORE_URL") or env_values.get("KAGGLE_MODEL_STORE_URL")
    if value:
        return value.rstrip("/")
    raise RuntimeError("KAGGLE_MODEL_STORE_URL is missing from .env and the model-store URL file")


def check_model_store(url: str) -> bool:
    request = urllib.request.Request(f"{url}/health")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            health = json.loads(response.read())
    except (AttributeError, OSError, TypeError, ValueError, urllib.error.URLError):
        return False
    return health.get("service") == "local-ai-model-store" and health.get("status") == "ok"


def is_hard_api_error(error: Exception) -> bool:
    message = str(error).lower()
    return any(value in message for value in (
        "http 401", "http 403", "http 429", "quota", "unauthorized", "forbidden",
    ))


def ensure_worker_running(worker: dict, expected_model: str = QWEN_CHECKPOINT) -> None:
    if worker.get("stopRequested") or worker.get("status") in {
        "stopped", "stopping", "failed", "lost",
    } or worker.get("remoteStatus") in {"failed", "complete", "stopped"}:
        raise RuntimeError(
            f"Image worker became unavailable ({worker.get('status')}); batch stopped"
        )
    if worker.get("kind") != "image" or worker.get("model") != expected_model:
        raise RuntimeError("Worker must be an image worker on the configured Qwen-Image checkpoint")


def api_json(api: str, api_key: str, path: str, data: dict | None = None) -> dict:
    payload = json.dumps(data).encode("utf-8") if data is not None else None
    try:
        return json.loads(collect_startframes.api_request(
            f"{api}{path}", api_key, data=payload,
        ))
    except RuntimeError as error:
        if is_hard_api_error(error):
            raise RuntimeError("Gateway reported an authentication or quota error; batch stopped") from None
        raise RuntimeError("Gateway request failed; batch state was retained") from None


def persist_state(path: Path, manifest: dict) -> None:
    aggregate_batch_status(manifest)
    collect_startframes.write_manifest(path, manifest)


def sync_current_frame_manifest(shot_id: str, entry: dict) -> None:
    manifest_path = ROOT / "startframes" / "manifest.json"
    current_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    jobs = current_manifest.get("jobs")
    current_entry = jobs.get(shot_id) if isinstance(jobs, dict) else None
    if not isinstance(current_entry, dict):
        raise RuntimeError(f"{shot_id}: current start-frame manifest entry is missing")

    output_hash = entry.get("output_sha256")
    output_dimensions = entry.get("output_dimensions")
    if not isinstance(output_hash, str) or not isinstance(output_dimensions, list):
        raise RuntimeError(f"{shot_id}: completed frame is missing output metadata")
    frame_path = ROOT / f"startframes/{shot_id}.png"
    if sha256(frame_path.read_bytes()) != output_hash:
        raise RuntimeError(f"{shot_id}: collected frame hash changed before manifest sync")

    updates = {
        "status": "completed",
        "output_sha256": output_hash,
        "output_dimensions": output_dimensions,
    }
    if any(current_entry.get(key) != value for key, value in updates.items()):
        current_entry.update(updates)
        collect_startframes.write_manifest(manifest_path, current_manifest)


def record_submission_uncertain(entry: dict) -> None:
    entry["status"] = "submission_uncertain"
    entry["terminal_status"] = None
    details = entry.setdefault("attempts_detail", [])
    if details:
        details[-1]["status"] = "submission_uncertain"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker-id", required=True, help="existing Qwen image worker session id")
    parser.add_argument(
        "--model",
        choices=(QWEN_CHECKPOINT, QWEN_SPACE_CHECKPOINT),
        default=QWEN_CHECKPOINT,
        help="checkpoint loaded by the existing image worker",
    )
    parser.add_argument("--api", default="http://127.0.0.1:3000")
    parser.add_argument("--from", dest="first_id", default="c01")
    parser.add_argument("--to", dest="last_id", default="c70")
    parser.add_argument("--ids", help="comma-separated beat ids for a non-contiguous batch")
    parser.add_argument(
        "--overrides-file",
        default="prompts/startframe_prompts.json",
        help="Qwen detailed prompt overrides JSON",
    )
    parser.add_argument("--state-file", help="separate resumable state file for this batch")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument(
        "--prefetch",
        type=int,
        default=2,
        help="image jobs to keep queued or running on the worker (1-8); a deeper "
        "queue hides claim and poll latency between frames",
    )
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--timeout-hours", type=float, default=72)
    args = parser.parse_args(argv)
    if (
        args.poll_seconds < 1
        or args.timeout_hours <= 0
        or not 4 <= args.steps <= 40
        or not 1 <= args.prefetch <= 8
    ):
        parser.error("poll interval, timeout, prefetch, and steps must be within their valid ranges")
    try:
        if args.ids:
            if args.first_id != "c01" or args.last_id != "c70":
                parser.error("--ids cannot be combined with --from or --to")
            selected_ids = collect_startframes.parse_shot_ids(args.ids)
        else:
            selected_ids = collect_startframes.select_shot_ids(args.first_id, args.last_id)
        shots, configs, negative_prompt = load_production_inputs()
        current_manifest = load_current_manifest()
        api_key = collect_startframes.read_api_key()
        model_store_url = read_model_store_url()
        api = args.api.rstrip("/")
        state_file = args.state_file or (
            "prompts/kaggle_qwen21_batch_manifest.json"
            if args.model == QWEN_SPACE_CHECKPOINT
            else "prompts/startframe_batch_manifest.json"
        )
        manifest_path = Path(state_file)
        if not manifest_path.is_absolute():
            manifest_path = ROOT / manifest_path
        manifest_path = manifest_path.resolve()
        if not manifest_path.is_relative_to(ROOT.resolve()):
            raise RuntimeError("The batch state file must be inside the repository")
        overrides = {}
        if args.model == QWEN_SPACE_CHECKPOINT:
            overrides_path = Path(args.overrides_file)
            if not overrides_path.is_absolute():
                overrides_path = ROOT / overrides_path
            overrides_path = overrides_path.resolve()
            if not overrides_path.is_relative_to(ROOT.resolve()):
                raise RuntimeError("Qwen prompt overrides must be inside the repository")
            if not overrides_path.is_file():
                raise RuntimeError("Qwen detailed prompt overrides are missing")
            overrides = json.loads(overrides_path.read_text(encoding="utf-8"))
            if not isinstance(overrides, dict):
                raise RuntimeError("Qwen detailed prompt overrides must be an object")
            missing_overrides = [shot_id for shot_id in selected_ids if shot_id not in overrides]
            if missing_overrides:
                raise RuntimeError(
                    "Qwen detailed prompt overrides are missing selected beats: "
                    + ", ".join(missing_overrides)
                )
        reference_free = not any(
            isinstance(overrides.get(shot_id), dict)
            and overrides[shot_id].get("refs")
            for shot_id in SHOT_IDS
        )
        manifest = prepare_batch_manifest(
            args.worker_id,
            current_manifest,
            shots,
            configs,
            manifest_path=manifest_path,
            model=args.model,
            reference_free=reference_free,
            selected_ids=selected_ids,
        )
        if any(
            entry.get("status") == "submission_uncertain"
            for entry in manifest["jobs"].values()
        ):
            raise RuntimeError(
                "A prior submission has an unknown outcome; inspect gateway tasks before resuming"
            )
        current_manifest.update({
            "batch_status": "generating",
            "reference_free": reference_free,
            "provisional": True,
            "redo_candidates": REDO_CANDIDATES,
        })
        collect_startframes.write_manifest(
            ROOT / "startframes" / "manifest.json", current_manifest,
        )
        persist_state(manifest_path, manifest)
        shot_by_id = {shot["id"]: shot for shot in shots}
        deadline = time.monotonic() + args.timeout_hours * 3600

        queue = deque(selected_ids)
        inflight: list[str] = []
        health_checked_at = 0.0

        def check_health() -> None:
            nonlocal health_checked_at
            now = time.monotonic()
            if now - health_checked_at < 60:
                return
            health_checked_at = now
            if check_model_store(model_store_url):
                if manifest.pop("model_store_unhealthy_since", None) is not None:
                    persist_state(manifest_path, manifest)
                return
            unhealthy_since = manifest.setdefault(
                "model_store_unhealthy_since", time.time(),
            )
            persist_state(manifest_path, manifest)
            if time.time() - float(unhealthy_since) > 600:
                raise RuntimeError(
                    "Public model-store health has been down for more than 10 minutes"
                )

        while queue or inflight:
            if time.monotonic() >= deadline:
                raise RuntimeError("Start-frame batch timed out; worker remains untouched")

            # Fill the worker's queue instead of waiting for one frame at a
            # time: a persistent Kaggle worker claims the next queued job as
            # soon as it finishes the current one, so a small prefetch depth
            # removes the claim/poll gap between frames.
            while len(inflight) < args.prefetch and queue:
                shot_id = queue.popleft()
                entry = manifest["jobs"][shot_id]
                frame_path = ROOT / entry["start_frame"]
                if (
                    is_succeeded(entry)
                    and frame_path.is_file()
                    and sha256(frame_path.read_bytes()) == entry["output_sha256"]
                ):
                    sync_current_frame_manifest(shot_id, entry)
                    continue
                if entry.get("terminal_status") == "failed" or int(entry.get("attempts", 0)) >= 2:
                    entry["terminal_status"] = "failed"
                    persist_state(manifest_path, manifest)
                    continue
                if entry.get("job_id"):
                    inflight.append(shot_id)
                    continue
                request = build_request_for_shot(
                    args.model,
                    shot_by_id[shot_id],
                    configs[shot_id],
                    overrides.get(shot_id, {}),
                    negative_prompt,
                    args.steps,
                    entry,
                )
                worker = api_json(api, api_key, f"/v1/workers/{args.worker_id}")
                ensure_worker_running(worker, args.model)
                entry["attempts"] = int(entry.get("attempts", 0)) + 1
                entry["status"] = "submitting"
                entry["terminal_status"] = None
                details = entry.setdefault("attempts_detail", [])
                details.append({
                    "job_id": None,
                    "status": "submitting",
                    "terminal_status": None,
                })
                persist_state(manifest_path, manifest)
                try:
                    response = api_json(
                        api,
                        api_key,
                        f"/v1/workers/{args.worker_id}/jobs",
                        request,
                    )
                except (RuntimeError, ValueError):
                    record_submission_uncertain(entry)
                    persist_state(manifest_path, manifest)
                    raise
                job_id = response.get("id")
                if not isinstance(job_id, str) or not job_id:
                    record_submission_uncertain(entry)
                    persist_state(manifest_path, manifest)
                    raise RuntimeError(
                        f"{shot_id}: submission outcome is unknown; inspect gateway tasks"
                    )
                entry["job_id"] = job_id
                details[-1]["job_id"] = job_id
                update_terminal_status(entry, response.get("status", "queued"))
                persist_state(manifest_path, manifest)
                inflight.append(shot_id)

            if not inflight:
                continue

            shot_id = inflight[0]
            entry = manifest["jobs"][shot_id]
            check_health()
            task = api_json(api, api_key, f"/v1/tasks/{entry['job_id']}")
            status = task.get("status")
            update_terminal_status(
                entry,
                status,
                task.get("error") if isinstance(task.get("error"), str) else None,
            )
            persist_state(manifest_path, manifest)

            if status in ACTIVE_TASK_STATUSES:
                worker = api_json(api, api_key, f"/v1/workers/{args.worker_id}")
                ensure_worker_running(worker, args.model)
                time.sleep(args.poll_seconds)
                continue

            if status == "completed":
                worker = api_json(api, api_key, f"/v1/workers/{args.worker_id}")
                ensure_worker_running(worker, args.model)
                width, height = collect_startframes.collect_completed_frame(
                    api,
                    api_key,
                    shot_id,
                    entry,
                    configs[shot_id],
                    replace_existing=True,
                )
                sync_current_frame_manifest(shot_id, entry)
                update_terminal_status(entry, "completed")
                persist_state(manifest_path, manifest)
                print(
                    f"{shot_id}: succeeded ({width}x{height}), "
                    f"classification={AMMI_DEPICTIONS.get(shot_id, 'no Ammi')}",
                    flush=True,
                )
                inflight.pop(0)
                continue

            if status in FAILED_TASK_STATUSES:
                attempts = int(entry.get("attempts", 0))
                entry["job_id"] = None
                if attempts >= 2:
                    entry["terminal_status"] = "failed"
                    persist_state(manifest_path, manifest)
                    print(f"{shot_id}: failed after two attempts; skipping", flush=True)
                    inflight.pop(0)
                else:
                    entry["terminal_status"] = None
                    persist_state(manifest_path, manifest)
                    inflight.pop(0)
                    queue.appendleft(shot_id)
                if manifest.get("failed_count", 0) >= 8:
                    raise RuntimeError("Eight cumulative frame failures reached; batch hard-stopped")
                continue

            time.sleep(args.poll_seconds)

        return 1 if manifest.get("failed_count", 0) else 0
    except (OSError, KeyError, TypeError, ValueError, yaml.YAMLError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
