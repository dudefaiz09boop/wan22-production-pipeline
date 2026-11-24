#!/usr/bin/env python3
"""Schema + format lint for beat sheets. Defaults to docs/beats.yaml.

Hard checks always run. Story-doctrine checks (camera balance, visible cast,
spoken language) run only with --doctrine and are warnings unless --strict.
Exit non-zero on any failure.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

from audio_index import resolve_audio

VERBS = {
    "static", "zoom", "dolly", "pan", "tilt",
    "orbit", "track", "arc", "handheld", "whip",
}
BUILDS = {"fm", "cf"}
MAX_LEN = 220
MAX_TURNS = 8
MAX_PROMPT_CHARS = 2000
MAX_LORAS = 3
# Registry entries flagged `default: true` are injected into every shot by
# render.inject_loras, so they count against the per-shot budget too.
MAX_TOTAL_LORAS = 5
LANGS = {"en", "hi", "ur"}
RESET_BEATS = {"c08", "c18", "c21", "c30", "c46", "c47"}
STABLE_ANCHOR_KEYS = ("face", "body", "wardrobe")
MARK_RANK = {"none": 0, "fresh": 1, "tender": 2, "fading": 3, "yellow": 4}
PROFILES = {"draft", "hero"}
AUDIO_MODES = {"none", "s2v"}
LIGHT_VALUES = {"tube_flicker", "blackout_moonlight", "night_tube", "evening_tube", "daylight"}
HERO_BEATS = {"c25", "c33", "c44", "c61", "c68"}

DEVANAGARI = re.compile(r"[\u0900-\u097f]")
ARABIC = re.compile(r"[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]")
HINDI_MARKERS = frozenset(
    "aaya aayi accha achha hai hain haan hoon hu ka ke ki ko kya leti mera meri "
    "mujhe nahin nahi raha rahi tha thi the tum utha".split()
)
URDU_MARKERS = frozenset(
    "aap aapka aapki aapko janab jee kijiye khuda kyun meherbani mujhay meray "
    "shukriya zaroor".split()
)
URDU_DISTINCT = frozenset(
    "aap aapka aapki aapko janab jee kijiye khuda mujhay meray shukriya".split()
)


def detect_language(text: str, preferred: str = "") -> str | None:
    """Mirror of the frontend detector: Devanagari -> hi, Arabic -> ur, else markers."""
    if ARABIC.search(text):
        return "ur"
    if DEVANAGARI.search(text):
        return "hi"
    if not re.search(r"[A-Za-z]", text):
        return None
    tokens = re.findall(r"[a-z]+", text.lower())
    hi = sum(token in HINDI_MARKERS for token in tokens)
    ur = sum(token in URDU_MARKERS for token in tokens)
    distinct = any(token in URDU_DISTINCT for token in tokens)
    if hi == 0 and ur == 0:
        return "en"
    if hi < 2 and ur < 2:
        if preferred.startswith("ur"):
            return "ur"
        if preferred.startswith("hi"):
            return "hi"
        return "ur" if distinct else "en"
    if hi > ur:
        return "hi"
    if ur > hi:
        return "ur"
    if preferred.startswith("ur"):
        return "ur"
    if preferred.startswith("hi"):
        return "hi"
    return "ur" if distinct else "en"


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


def parse_cast(blob: str) -> dict[str, list[str]]:
    """Parse 'Faiz=man|guy,Juveriya=woman' into {name: [aliases]}."""
    cast: dict[str, list[str]] = {}
    for item in (part.strip() for part in blob.split(",")):
        if not item:
            continue
        name, _, alias_blob = item.partition("=")
        name = name.strip()
        if not name:
            continue
        cast[name] = [alias.strip() for alias in alias_blob.split("|") if alias.strip()]
    return cast


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="docs/beats.yaml")
    ap.add_argument("--root", default=".")
    ap.add_argument("--expected", type=int, default=None)
    ap.add_argument(
        "--cast",
        default="",
        help="names that must appear twice per cam block; aliases via Name=alias|alias",
    )
    ap.add_argument("--langs", default="en,hi,ur")
    ap.add_argument("--doctrine", action="store_true")
    ap.add_argument("--strict", action="store_true")
    ap.add_argument("--max-static", type=int, default=4)
    ap.add_argument("--min-moves", type=int, default=2)
    ap.add_argument(
        "--require-speaker",
        action="store_true",
        help="require a speaker on every spoken shot even when no cast is declared",
    )
    ap.add_argument(
        "--require-language",
        action="store_true",
        help="require language on spoken shots and forbid it on silent shots",
    )
    ap.add_argument(
        "--no-frames",
        action="store_true",
        help="skip checking whether start_frame files exist on disk",
    )
    args = ap.parse_args()

    root = Path(args.root)
    spec = root / args.spec
    errors: list[str] = []
    warnings: list[str] = []

    def fail(msg: str) -> None:
        errors.append(msg)

    def warn(msg: str) -> None:
        warnings.append(msg)

    allowed_langs = {item.strip() for item in args.langs.split(",") if item.strip()}
    unknown = allowed_langs - LANGS
    if unknown:
        fail(f"unknown --langs values {sorted(unknown)}")
    cast = parse_cast(args.cast)

    if not spec.exists():
        print(f"FAIL missing spec: {spec}", file=sys.stderr)
        return 1

    try:
        source = spec.read_text(encoding="utf-8")
    except OSError as error:
        print(f"FAIL cannot read spec: {error}", file=sys.stderr)
        return 1

    line = 1
    c1_controls = []
    for offset, char in enumerate(source):
        if "\u0080" <= char <= "\u009f":
            c1_controls.append((offset, line, ord(char)))
        if char == "\n":
            line += 1
    if c1_controls:
        for offset, line_number, codepoint in c1_controls:
            print(
                f"FAIL raw C1 control U+{codepoint:04X} at offset {offset}, "
                f"line {line_number}",
                file=sys.stderr,
            )
        return 1

    try:
        data = yaml.safe_load(source)
    except yaml.YAMLError as error:
        print(f"FAIL {spec} is not valid YAML: {error}", file=sys.stderr)
        return 1
    if not isinstance(data, dict):
        print(f"FAIL {spec} must contain a mapping", file=sys.stderr)
        return 1

    story = data.get("story") if isinstance(data.get("story"), dict) else {}
    story_scenes = story.get("scenes")
    known_scenes = set(story_scenes) if isinstance(story_scenes, dict) else set()
    known_names = {
        item["name"] for item in story.get("characters") or []
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    meta = data.get("meta") if isinstance(data.get("meta"), dict) else {}
    default_duration = meta.get("duration")
    # Episode 2 continues the numbering from c71. start_index lets a spec
    # declare its first shot number instead of always assuming c01.
    start_index = meta.get("start_index", 1)
    if not isinstance(start_index, int) or isinstance(start_index, bool) or start_index < 1:
        fail("meta.start_index must be a positive integer")
        start_index = 1
    relations = story.get("relations")
    if relations is not None:
        if not isinstance(relations, dict):
            fail("story.relations must be a mapping")
        else:
            if relations.get("faiz_juveriya") != "siblings":
                fail("story.relations.faiz_juveriya must be 'siblings'")
            if relations.get("on_screen") is not False:
                fail("story.relations.on_screen must be false")
    boundaries = story.get("boundaries")
    if boundaries is not None:
        if not isinstance(boundaries, dict):
            fail("story.boundaries must be a mapping")
        else:
            consent_beats = boundaries.get("consent_beats")
            if not isinstance(consent_beats, list) or not consent_beats:
                fail("story.boundaries.consent_beats must be a non-empty list")
            elif any(
                not isinstance(consent_id, str)
                or not any(
                    isinstance(shot, dict) and shot.get("id") == consent_id
                    for shot in (data.get("shots") or [])
                )
                for consent_id in consent_beats
            ):
                fail("story.boundaries.consent_beats contains an unresolved id")
            if boundaries.get("revocable") is not True:
                fail("story.boundaries.revocable must be true")
    thread_beats = story.get("thread_beats")
    if thread_beats is not None and (not isinstance(thread_beats, list) or not thread_beats):
        fail("story.thread_beats must be a non-empty list")
    production_contract = (
        isinstance(relations, dict)
        or isinstance(boundaries, dict)
        or isinstance(thread_beats, list)
    )
    if production_contract and default_duration != 5:
        fail("meta.duration must be 5 seconds")
    require_speaker = args.require_speaker or bool(known_names)

    shots = data.get("shots") or []
    if not isinstance(shots, list):
        fail("shots must be a list")
        shots = []
    if not shots:
        fail("no shots")
    if args.expected is not None and len(shots) != args.expected:
        fail(f"shot count {len(shots)} != {args.expected}")

    ammi_depiction = relations.get("ammi_depiction") if isinstance(relations, dict) else None
    if ammi_depiction is not None:
        if not isinstance(ammi_depiction, dict):
            fail("story.relations.ammi_depiction must be a mapping")
        else:
            shots_by_id = {
                shot.get("id"): shot
                for shot in shots
                if isinstance(shot, dict) and isinstance(shot.get("id"), str)
            }
            phone_call_beats = {"c34", "c35", "c36", "c37"}
            for beat_id in ammi_depiction:
                if not isinstance(beat_id, str):
                    fail(f"story.relations.ammi_depiction key {beat_id!r} must be a shot id")
                    continue
                shot = shots_by_id.get(beat_id)
                if shot is None:
                    fail(f"story.relations.ammi_depiction key {beat_id!r} does not resolve to a shot")
                    continue
                beat_text = list(shot.get("cam") or [])
                say = shot.get("say")
                if isinstance(say, str):
                    beat_text.append(say)
                for turn in shot.get("dialogue") or []:
                    if isinstance(turn, dict):
                        beat_text.extend(
                            str(turn.get(field, ""))
                            for field in ("speaker", "text", "text_script")
                        )
                references_ammi = any(
                    isinstance(text, str)
                    and re.search(r"(?<!\w)Ammi(?!\w)", text, re.IGNORECASE)
                    for text in beat_text
                )
                if beat_id not in phone_call_beats and not references_ammi:
                    fail(
                        f"story.relations.ammi_depiction.{beat_id} must reference Ammi "
                        "in cam/say/dialogue or be within the c34-c37 phone-call span"
                    )

    known_loras = load_known_loras(root)
    seen: list[str] = []
    previous_mark_rank = -1
    previous_shot_id = None
    previous_anchors = None

    for index, shot in enumerate(shots, start=1):
        where = f"shot[{index}]"
        if not isinstance(shot, dict):
            fail(f"{where} is not a mapping")
            continue

        sid = shot.get("id")
        if not isinstance(sid, str) or not sid:
            fail(f"{where} missing id")
            continue
        seen.append(sid)

        expected_id = f"c{start_index + index - 1:02d}"
        if sid != expected_id:
            fail(f"{where} id {sid!r} != {expected_id!r} (order broken)")

        scene = shot.get("scene")
        if not isinstance(scene, int) or scene < 1:
            fail(f"{sid} scene must be a positive integer")
        elif scene not in known_scenes:
            fail(f"{sid} scene {scene} is not defined in story.scenes")

        if shot.get("build") not in BUILDS:
            fail(f"{sid} build {shot.get('build')!r} not in {sorted(BUILDS)}")

        cam = shot.get("cam") or []
        if not isinstance(cam, list) or len(cam) != 6:
            fail(f"{sid} has {len(cam) if isinstance(cam, list) else 0} cam lines, need exactly 6")
            cam = cam if isinstance(cam, list) else []

        verbs: list[str] = []
        for n, line in enumerate(cam):
            if not isinstance(line, str) or not line.strip():
                fail(f"{sid} cam[{n}] empty")
                continue
            first = re.split(r"[\s,:]+", line.strip(), maxsplit=1)[0].lower()
            verbs.append(first)
            if first not in VERBS:
                fail(f"{sid} cam[{n}] starts with {first!r}, not a camera verb")
            if len(line) > MAX_LEN:
                fail(f"{sid} cam[{n}] is {len(line)} chars (max {MAX_LEN})")

        compiled = "\n".join(f"(At {n} seconds: {line})" for n, line in enumerate(cam))
        compiled_loras: list = []
        for lora in (shot.get("loras") or []):
            if isinstance(lora, str) and re.fullmatch(r"[a-z0-9_]+", lora):
                compiled_loras.append(lora)
        prefixes, suffixes = lora_trigger_lists(compiled_loras, known_loras)
        if prefixes:
            compiled = ", ".join(prefixes) + ", " + compiled
        if suffixes:
            compiled = compiled + ", " + ", ".join(suffixes)
        if len(compiled) > MAX_PROMPT_CHARS:
            fail(f"{sid} compiled prompt is {len(compiled)} chars (max {MAX_PROMPT_CHARS})")

        sf = shot.get("start_frame")
        if not sf:
            fail(f"{sid} missing start_frame")
        elif not isinstance(sf, str):
            fail(f"{sid} start_frame must be text")
        else:
            if not re.fullmatch(rf"startframes/{re.escape(sid)}\.png", sf):
                fail(f"{sid} start_frame {sf!r} != startframes/{sid}.png")
            if not args.no_frames and not (root / sf).exists():
                fail(f"{sid} start_frame missing on disk: {sf}")

        loras = shot.get("loras") or []
        if not isinstance(loras, list):
            fail(f"{sid} loras must be a list")
            loras = []
        if len(loras) > MAX_LORAS:
            fail(f"{sid} declares {len(loras)} loras (max {MAX_LORAS})")
        default_count = sum(
            1
            for entry in (known_loras or {}).values()
            if isinstance(entry, dict) and entry.get("default") is True
        )
        if not shot.get("no_default_loras") and len(loras) + default_count > MAX_TOTAL_LORAS:
            fail(
                f"{sid} would load {len(loras) + default_count} loras "
                f"({len(loras)} declared + {default_count} default, "
                f"max {MAX_TOTAL_LORAS})"
            )
        declared_keys = set()
        for lora in loras:
            if not isinstance(lora, str) or not re.fullmatch(r"[a-z0-9_]+", lora):
                fail(f"{sid} bad lora key {lora!r}")
                continue
            declared_keys.add(lora)
            if known_loras and lora not in known_loras:
                fail(f"{sid} lora {lora!r} is not in workflows/loras.json")
        for lora in sorted(declared_keys):
            entry = known_loras.get(lora) if known_loras else None
            conflicts = entry.get("conflicts") if isinstance(entry, dict) else None
            for conflict in conflicts or []:
                if conflict in declared_keys:
                    fail(f"{sid} declares conflicting loras {lora!r} and {conflict!r}")
            trigger = entry.get("trigger") if isinstance(entry, dict) else None
            if isinstance(trigger, str) and trigger.strip() and trigger.strip() not in compiled:
                fail(f"{sid} lora {lora!r} trigger {trigger.strip()!r} missing from compiled prompt")

        if sid in RESET_BEATS and shot.get("chain") is not False:
            fail(f"{sid} is a declared identity reset and must set chain: false")

        if "chain" in shot and not isinstance(shot["chain"], bool):
            fail(f"{sid} chain must be a boolean")

        if production_contract:
            if not isinstance(shot.get("chain"), bool):
                fail(f"{sid} chain must be a boolean")
            noise = shot.get("noise")
            if not isinstance(noise, int) or isinstance(noise, bool) or not 0 <= noise <= 3:
                fail(f"{sid} noise must be an integer from 0 to 3")
            duration = shot.get("duration")
            if not isinstance(duration, int) or isinstance(duration, bool) or duration < 1:
                fail(f"{sid} duration must be a positive integer")
            elif isinstance(noise, int) and not isinstance(noise, bool):
                if noise == 3 and duration > 4:
                    fail(f"{sid} noise 3 requires duration <= 4")
                elif noise < 3 and duration != default_duration:
                    fail(f"{sid} duration must match meta.duration ({default_duration})")

            light = shot.get("light")
            if light not in LIGHT_VALUES:
                fail(f"{sid} light must be one of {sorted(LIGHT_VALUES)}")

            mark = shot.get("mark_state")
            if mark not in MARK_RANK:
                fail(f"{sid} mark_state must be one of {sorted(MARK_RANK)}")

            profile = shot.get("profile")
            if profile not in PROFILES:
                fail(f"{sid} profile must be one of {sorted(PROFILES)}")
            elif (profile == "hero") != (sid in HERO_BEATS):
                fail(f"{sid} hero profile must be exactly on {sorted(HERO_BEATS)}")

            audio_mode = shot.get("audio_mode")
            if audio_mode not in AUDIO_MODES:
                fail(f"{sid} audio_mode must be one of {sorted(AUDIO_MODES)}")
            elif audio_mode == "s2v":
                if resolve_audio(root, sid) is None:
                    fail(
                        f"{sid} s2v requires an audio sidecar "
                        f"(refs/audio/{sid}.wav or refs/audio/index.json)"
                    )

            anchors = shot.get("anchors")
            if not isinstance(anchors, dict):
                fail(f"{sid} anchors must be a mapping")
            else:
                missing = sorted({"face", "body", "wardrobe", "prop", "light", "geography"} - set(anchors))
                if missing:
                    fail(f"{sid} anchors missing keys: {missing}")
                for key in STABLE_ANCHOR_KEYS:
                    value = anchors.get(key)
                    if not isinstance(value, str) or not value.strip():
                        fail(f"{sid} anchors.{key} must be non-empty text")
                    elif shot.get("chain") is True and previous_anchors is not None:
                        if value != previous_anchors.get(key):
                            fail(
                                f"{sid} anchors.{key} must match {previous_shot_id} "
                                "when chain is true"
                            )
                previous_anchors = anchors
                previous_shot_id = sid
            if mark in MARK_RANK:
                rank = MARK_RANK[mark]
                if rank < previous_mark_rank:
                    fail(f"{sid} mark_state is not monotonic")
                previous_mark_rank = max(previous_mark_rank, rank)

        say = shot.get("say", "")
        if not isinstance(say, str):
            fail(f"{sid} say must be text")
            say = ""
        say_script = shot.get("say_script", "")
        if not isinstance(say_script, str):
            fail(f"{sid} say_script must be text")
            say_script = ""
        dialogue = shot.get("dialogue")
        turns: list[tuple[str, str, str, str | None]] = []

        if dialogue is None:
            speaker = shot.get("speaker")
            if say.strip():
                if speaker is None:
                    if require_speaker:
                        fail(f"{sid} spoken shot needs a speaker")
                    turns.append(("", say, say_script, None))
                elif not isinstance(speaker, str) or not speaker:
                    fail(f"{sid} speaker must be text")
                elif known_names and speaker not in known_names:
                    fail(f"{sid} speaker {speaker!r} is not a defined character")
                else:
                    turns.append((speaker, say, say_script, None))
            elif speaker is not None:
                fail(f"{sid} silent shot must not define a speaker")
        else:
            if say.strip() or shot.get("speaker") is not None:
                fail(f"{sid} use either dialogue turns or say/speaker, not both")
            if not isinstance(dialogue, list) or not dialogue:
                fail(f"{sid} dialogue must be a non-empty list")
            elif len(dialogue) > MAX_TURNS:
                fail(f"{sid} dialogue has {len(dialogue)} turns (max {MAX_TURNS})")
            else:
                for n, turn in enumerate(dialogue):
                    if not isinstance(turn, dict):
                        fail(f"{sid} dialogue[{n}] must be a mapping")
                        continue
                    speaker, text = turn.get("speaker"), turn.get("text")
                    text_script = turn.get("text_script", "")
                    turn_language = turn.get("language")
                    if not isinstance(speaker, str) or not speaker:
                        fail(f"{sid} dialogue[{n}] needs a speaker")
                    elif known_names and speaker not in known_names:
                        fail(f"{sid} dialogue[{n}] speaker {speaker!r} is not a defined character")
                    valid_text = isinstance(text, str) and bool(text.strip())
                    if not valid_text:
                        fail(f"{sid} dialogue[{n}] needs text")
                    valid_script_text = isinstance(text_script, str)
                    if not valid_script_text:
                        fail(f"{sid} dialogue[{n}] text_script must be text")
                        text_script = ""
                    if turn_language is not None and (
                        not isinstance(turn_language, str) or not turn_language.strip()
                    ):
                        fail(f"{sid} dialogue[{n}] language must be non-empty text")
                        turn_language = None
                    if (
                        valid_text
                        and valid_script_text
                        and isinstance(speaker, str)
                        and speaker
                    ):
                        turns.append((speaker, text, text_script, turn_language))

        spoken = bool(say.strip()) or (isinstance(dialogue, list) and bool(dialogue))
        if args.require_language:
            if spoken and (
                not isinstance(shot.get("language"), str)
                or not shot["language"].strip()
            ):
                fail(f"{sid} spoken shot needs a language")
            elif not spoken and "language" in shot:
                fail(f"{sid} silent shot must not define a language")

        language = shot.get("language")
        if "language" in shot:
            if not isinstance(language, str) or not language.strip():
                fail(f"{sid} language must be non-empty text")
            else:
                for n, (_, text, text_script, turn_language) in enumerate(turns):
                    effective_language = turn_language or language
                    if effective_language == "hi-script":
                        source_text = text_script or text
                        if not DEVANAGARI.search(source_text):
                            fail(
                                f"{sid} dialogue[{n}] language hi-script "
                                "needs Devanagari text"
                            )
                    elif effective_language == "ur-script":
                        source_text = text_script or text
                        if not ARABIC.search(source_text):
                            fail(
                                f"{sid} dialogue[{n}] language ur-script "
                                "needs Arabic text"
                            )

        if args.doctrine:
            statics = sum(verb == "static" for verb in verbs)
            moves = len(verbs) - statics
            if statics > args.max_static:
                fail(f"{sid} has {statics} static lines (max {args.max_static})")
            if moves < args.min_moves:
                fail(f"{sid} has {moves} movement lines (need at least {args.min_moves})")

            cam_text = " ".join(line for line in cam if isinstance(line, str))
            for name, aliases in cast.items():
                pattern = "|".join(re.escape(term) for term in [name, *aliases])
                hits = len(re.findall(rf"(?<!\w)(?:{pattern})(?!\w)", cam_text, re.IGNORECASE))
                if hits < 2:
                    fail(f"{sid} mentions {name} {hits} times in cam text (need at least 2)")

            if "language" not in shot:
                for n, (_, text, _, turn_language) in enumerate(turns):
                    if turn_language is not None:
                        continue
                    detected_language = detect_language(text)
                    if detected_language is None:
                        warn(f"{sid} dialogue[{n}] has no detectable language")
                    elif detected_language not in allowed_langs:
                        fail(
                            f"{sid} dialogue[{n}] lang {detected_language} "
                            f"not in {sorted(allowed_langs)}"
                        )

    # Numeric id order: c100 must come after c11, which lexicographic
    # sorting gets wrong on any spec with 100+ shots.
    def shot_number(sid: str) -> int:
        return int(sid[1:]) if sid[1:].isdigit() else -1

    if seen != sorted(seen, key=shot_number):
        fail("shots not in id order")

    neg = (data.get("meta") or {}).get("negative") if isinstance(data.get("meta"), dict) else None
    if not neg:
        fail("meta.negative missing")
    elif not isinstance(neg, str) or not (root / neg).exists():
        fail(f"negative file missing: {neg}")

    for message in warnings:
        print(f"WARN {message}", file=sys.stderr)
    for message in errors:
        print(f"FAIL {message}", file=sys.stderr)

    if errors or (args.strict and warnings):
        print(f"\n{len(errors)} failure(s), {len(warnings)} warning(s)", file=sys.stderr)
        return 1

    print(f"OK {len(shots)} shots validated ({len(warnings)} warning(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())