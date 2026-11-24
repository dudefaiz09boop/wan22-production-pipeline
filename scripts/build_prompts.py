#!/usr/bin/env python3
"""Assemble prompts/<id>.txt + .json from docs/beats.yaml. Never edits the spec."""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

from language_utils import select_language_text

VERBS = {
    "static", "zoom", "dolly", "pan", "tilt",
    "orbit", "track", "arc", "handheld", "whip",
}
BUILDS = {"fm", "cf"}
MAX_LEN = 220
MAX_PROMPT_CHARS = 2000
MAX_NEGATIVE_CHARS = 4096


def fail(message: str) -> None:
    print(f"FAIL {message}", file=sys.stderr)


def load_known_loras(root: Path) -> dict:
    path = root / "workflows" / "loras.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def lora_trigger_lists(loras: list, registry: dict) -> tuple[list[str], list[str]]:
    """Return (prefix triggers, suffix triggers) for the declared LoRA keys."""
    prefixes: list[str] = []
    suffixes: list[str] = []
    for lora in loras:
        entry = registry.get(lora) if isinstance(lora, str) else None
        if not isinstance(entry, dict):
            continue
        trigger = entry.get("trigger")
        scope = entry.get("trigger_scope")
        if not isinstance(trigger, str) or not trigger.strip():
            continue
        if scope == "prompt_prefix":
            prefixes.append(trigger.strip())
        elif scope == "prompt_suffix":
            suffixes.append(trigger.strip())
    return prefixes, suffixes


def apply_lora_triggers(lines: list[str], loras: list, registry: dict) -> str:
    """Compile cam lines, injecting trigger words per the LoRA registry.

    prompt_prefix triggers are prepended before the first cam line;
    prompt_suffix triggers are appended after the sixth (last) cam line.
    """
    prefixes, suffixes = lora_trigger_lists(loras, registry)
    body = "\n".join(lines)
    if prefixes:
        body = ", ".join(prefixes) + ", " + body
    if suffixes:
        body = body + ", " + ", ".join(suffixes)
    return body


def effective_loras(loras: list, registry: dict, no_defaults: bool = False) -> list:
    """Shot LoRAs plus the registry's `default: true` entries.

    Mirrors render.inject_loras so the compiled prompt carries the trigger
    words for exactly the weights that will actually be loaded.
    """
    defaults = [
        key
        for key, entry in registry.items()
        if isinstance(entry, dict) and entry.get("default") is True
    ]
    return ([] if no_defaults else defaults) + list(loras)


def look_prefix(shot: dict, meta: dict) -> str:
    """Camera/lens vocabulary for the shot's light.

    Wan 2.2 has no camera prior; without lens and film language the output
    reads as CG. The look is chosen by `light` so a night shot gets high-ISO
    noise and a daylight shot gets soft window light, rather than one generic
    string on all 70 beats.
    """
    looks = meta.get("looks")
    if not isinstance(looks, dict):
        return ""
    light = shot.get("light")
    text = looks.get(light) if isinstance(light, str) else None
    if not isinstance(text, str) or not text.strip():
        text = meta.get("look_default")
    return text.strip() if isinstance(text, str) else ""


def build_plan(
    shot: dict,
    meta: dict,
    negative: str,
    known_loras: set[str],
    errors: list[str],
    lang: str = "en",
    check_frames: bool = True,
) -> dict | None:
    sid = shot.get("id") if isinstance(shot.get("id"), str) else "<unknown>"

    if not isinstance(shot.get("id"), str) or not shot["id"]:
        fail(f"{sid}: missing id")
        return None

    scene = shot.get("scene")
    if not isinstance(scene, int) or scene < 1:
        errors.append(f"{sid}: scene must be a positive integer")

    build = shot.get("build")
    if build not in BUILDS:
        errors.append(f"{sid}: build {build!r} not in {sorted(BUILDS)}")

    cam = shot.get("cam")
    if not isinstance(cam, list) or len(cam) != 6:
        errors.append(
            f"{sid}: {len(cam) if isinstance(cam, list) else 0} cam lines, need exactly 6"
        )
        return None

    lines: list[str] = []
    for index, line in enumerate(cam):
        if not isinstance(line, str) or not line.strip():
            errors.append(f"{sid}[{index}]: empty cam line")
            continue
        verb = re.split(r"[\s,:]+", line.strip(), maxsplit=1)[0].lower()
        if verb not in VERBS:
            errors.append(f"{sid}[{index}]: bad verb {verb!r}")
        if len(line) > MAX_LEN:
            errors.append(f"{sid}[{index}]: {len(line)} chars (max {MAX_LEN})")
        lines.append(f"(At {index} seconds: {line})")

    loras = shot.get("loras", [])
    if not isinstance(loras, list):
        errors.append(f"{sid}: loras must be a list")
        loras = []
    for lora in loras:
        if not isinstance(lora, str) or not re.fullmatch(r"[a-z0-9_]+", lora):
            errors.append(f"{sid}: bad lora key {lora!r}")
        elif known_loras and lora not in known_loras:
            errors.append(f"{sid}: lora {lora!r} is not in workflows/loras.json")

    registry = known_loras if isinstance(known_loras, dict) else {}
    no_defaults = bool(shot.get("no_default_loras"))
    body = apply_lora_triggers(
        lines, effective_loras(loras, registry, no_defaults), registry
    )
    look = look_prefix(shot, meta)
    if look:
        body = look + ", " + body
    body += "\n"
    if len(body.strip()) > MAX_PROMPT_CHARS:
        errors.append(
            f"{sid}: compiled prompt is {len(body.strip())} chars (max {MAX_PROMPT_CHARS})"
        )

    start_frame = shot.get("start_frame")
    if not isinstance(start_frame, str) or not start_frame:
        errors.append(f"{sid}: missing start_frame")
    else:
        if not re.fullmatch(rf"startframes/{re.escape(sid)}\.png", start_frame):
            errors.append(f"{sid}: start_frame {start_frame!r} != startframes/{sid}.png")
        if check_frames and not Path(start_frame).is_file():
            errors.append(f"{sid}: start_frame missing on disk: {start_frame}")

    chain = shot.get("chain", True)
    if not isinstance(chain, bool):
        errors.append(f"{sid}: chain must be a boolean")

    say = shot.get("say", "")
    if not isinstance(say, str):
        errors.append(f"{sid}: say must be text")
        say = ""
    say_script = shot.get("say_script", "")
    if not isinstance(say_script, str):
        errors.append(f"{sid}: say_script must be text")
        say_script = ""
    language = shot.get("language", "en")
    if not isinstance(language, str) or not language.strip():
        errors.append(f"{sid}: language must be non-empty text")
        language = "en"

    res = meta.get("res_face") if build == "cf" else meta.get("res_base")
    if not isinstance(res, list) or len(res) != 2:
        errors.append(f"{sid}: missing resolution for build {build!r}")
        res = [0, 0]

    cfg = {
        "id": sid,
        "scene": scene,
        "build": build,
        "start_frame": start_frame,
        "chain": chain,
        "loras": loras,
        "no_default_loras": no_defaults,
        "say": say,
        "sampler": meta.get("sampler"),
        "steps": meta.get("steps"),
        "cfg": meta.get("cfg"),
        "duration": shot.get("duration", meta.get("duration")),
        "width": res[0],
        "height": res[1],
        "negative_file": negative,
        "language": language,
        "say_script": say_script,
        "noise": shot.get("noise"),
        "beat_duration": shot.get("duration", meta.get("duration")),
        "light": shot.get("light"),
        "mark_state": shot.get("mark_state"),
        "thread": shot.get("thread", ""),
        "profile": shot.get("profile", "draft"),
        "anchors": shot.get("anchors"),
        "audio_mode": shot.get("audio_mode", "none"),
    }
    narration_turns: list[str] = []
    if isinstance(shot.get("speaker"), str) and shot["speaker"].strip():
        cfg["speaker"] = shot["speaker"]
    if say.strip():
        selected_say = select_language_text(say, say_script, language, lang)
        if cfg.get("speaker"):
            narration_turns.append(f"{cfg['speaker']}: {selected_say}")
        else:
            narration_turns.append(selected_say)
    if isinstance(shot.get("dialogue"), list) and shot["dialogue"]:
        cfg["dialogue"] = []
        for turn in shot["dialogue"]:
            if not isinstance(turn, dict):
                errors.append(f"{sid}: dialogue turns must be mappings")
                continue
            text = turn.get("text", "")
            text_script = turn.get("text_script", "")
            if not isinstance(text, str):
                errors.append(f"{sid}: dialogue text must be text")
                text = ""
            if not isinstance(text_script, str):
                errors.append(f"{sid}: dialogue text_script must be text")
                text_script = ""
            turn_language = turn.get("language", language)
            if not isinstance(turn_language, str) or not turn_language.strip():
                errors.append(f"{sid}: dialogue language must be non-empty text")
                turn_language = language
            cfg["dialogue"].append({**turn, "text_script": text_script})
            speaker = turn.get("speaker")
            if isinstance(speaker, str) and text.strip():
                narration_turns.append(
                    f"{speaker}: "
                    f"{select_language_text(text, text_script, turn_language, lang)}"
                )
    cfg["narration"] = "\n".join(narration_turns)
    return {"id": sid, "body": body, "cfg": cfg}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="docs/beats.yaml")
    ap.add_argument("--out", default="prompts")
    ap.add_argument("--negative", default=None, help="override meta.negative")
    ap.add_argument("--expected", type=int, default=None, help="required shot count")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--no-frames",
        action="store_true",
        help="skip checking whether start_frame files exist on disk",
    )
    ap.add_argument("--lang", choices=("en", "hi", "ur", "mixed"), default="en")
    args = ap.parse_args()

    root = Path(".")
    spec = Path(args.spec)

    try:
        data = yaml.safe_load(spec.read_text(encoding="utf-8"))
    except OSError as error:
        print(f"FAIL cannot read spec: {error}", file=sys.stderr)
        return 1
    except yaml.YAMLError as error:
        print(f"FAIL {spec} is not valid YAML: {error}", file=sys.stderr)
        return 1

    if not isinstance(data, dict) or not isinstance(data.get("meta"), dict):
        print("FAIL spec must contain a meta mapping", file=sys.stderr)
        return 1
    meta = data["meta"]

    shots = data.get("shots")
    if not isinstance(shots, list) or not shots:
        print("FAIL spec must contain shots", file=sys.stderr)
        return 1
    if args.expected is not None and len(shots) != args.expected:
        print(f"FAIL shot count {len(shots)} != {args.expected}", file=sys.stderr)
        return 1

    for key in ("res_base", "res_face", "sampler", "steps", "cfg", "duration"):
        if meta.get(key) is None:
            print(f"FAIL meta.{key} missing", file=sys.stderr)
            return 1

    negative = args.negative or meta.get("negative")
    if not isinstance(negative, str) or not negative:
        print("FAIL no negative prompt file", file=sys.stderr)
        return 1
    negative_path = Path(negative)
    if not negative_path.is_file():
        print(f"FAIL negative file missing: {negative}", file=sys.stderr)
        return 1
    negative_text = negative_path.read_text(encoding="utf-8").strip()
    if not negative_text:
        print(f"FAIL negative file is empty: {negative}", file=sys.stderr)
        return 1
    if len(negative_text) > MAX_NEGATIVE_CHARS:
        print(f"FAIL negative exceeds {MAX_NEGATIVE_CHARS} chars", file=sys.stderr)
        return 1

    known_loras = load_known_loras(root)
    errors: list[str] = []
    plans: list[dict] = []

    for index, shot in enumerate(shots, start=1):
        if not isinstance(shot, dict):
            errors.append(f"shot[{index}]: not a mapping")
            continue
        expected_id = f"c{index:02d}"
        if shot.get("id") != expected_id:
            errors.append(
                f"shot[{index}]: id {shot.get('id')!r} != {expected_id!r} (order broken)"
            )
        plan = build_plan(
            shot,
            meta,
            negative,
            known_loras,
            errors,
            args.lang,
            check_frames=not args.no_frames,
        )
        if plan is not None:
            plans.append(plan)

    if errors:
        for message in errors:
            fail(message)
        print(f"\n{len(errors)} failure(s), nothing written", file=sys.stderr)
        return 1

    out = Path(args.out)
    if not args.dry_run:
        out.mkdir(parents=True, exist_ok=True)

    for plan in plans:
        if args.dry_run:
            print(f"--- {plan['id']} ---\n{plan['body']}")
        else:
            (out / f"{plan['id']}.txt").write_text(plan["body"], encoding="utf-8")
            (out / f"{plan['id']}.json").write_text(
                json.dumps(plan["cfg"], indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    print(f"negative: {negative}")
    print(f"{'would write' if args.dry_run else 'wrote'} {len(plans)} prompts")
    return 0


if __name__ == "__main__":
    sys.exit(main())