#!/usr/bin/env python3
"""Export the explicit production beat sheet as a Film Studio plan.

The importer reads and validates docs/beats.yaml; it never writes the source.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import yaml

from language_utils import select_language_text
from audio_index import resolve_audio


ROOT = Path(__file__).resolve().parents[1]
MAX_NEGATIVE_CHARS = 4096
MAX_PROMPT_CHARS = 2000
MAX_TURNS = 8
VERBS = {
    "static", "zoom", "dolly", "pan", "tilt",
    "orbit", "track", "arc", "handheld", "whip",
}
STABLE_ANCHOR_KEYS = ("face", "body", "wardrobe")


def text_field(value, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty text.")
    return value.strip()


def dialogue_turns(
    shot: dict, expected_id: str, known_names: set[str]
) -> list[tuple[str, str, str, str | None]]:
    say = shot.get("say", "")
    if not isinstance(say, str):
        raise ValueError(f"{expected_id}: say must be text.")
    dialogue = shot.get("dialogue")
    turns: list[tuple[str, str, str, str | None]] = []

    if dialogue is None:
        if say.strip():
            speaker = shot.get("speaker")
            if not isinstance(speaker, str) or speaker not in known_names:
                raise ValueError(f"{expected_id}: spoken shot must name a defined speaker.")
            script_text = shot.get("say_script", "")
            if not isinstance(script_text, str):
                raise ValueError(f"{expected_id}: say_script must be text.")
            turns.append((speaker, say, script_text, None))
        elif shot.get("speaker") is not None:
            raise ValueError(f"{expected_id}: a silent shot must not define a speaker.")
        return turns

    if say.strip() or shot.get("speaker") is not None:
        raise ValueError(f"{expected_id}: use either dialogue turns or say/speaker, not both.")
    if not isinstance(dialogue, list) or not dialogue:
        raise ValueError(f"{expected_id}: dialogue must be a non-empty list of turns.")
    if len(dialogue) > MAX_TURNS:
        raise ValueError(f"{expected_id}: dialogue cannot exceed {MAX_TURNS} turns.")
    for turn in dialogue:
        if not isinstance(turn, dict):
            raise ValueError(f"{expected_id}: each dialogue turn must be an object.")
        speaker = turn.get("speaker")
        text = turn.get("text")
        text_script = turn.get("text_script", "")
        turn_language = turn.get("language")
        if not isinstance(speaker, str) or speaker not in known_names:
            raise ValueError(f"{expected_id}: each dialogue turn needs a defined speaker.")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{expected_id}: each dialogue turn needs non-empty text.")
        if not isinstance(text_script, str):
            raise ValueError(f"{expected_id}: dialogue text_script must be text.")
        if turn_language is not None and (
            not isinstance(turn_language, str) or not turn_language.strip()
        ):
            raise ValueError(f"{expected_id}: dialogue language must be non-empty text.")
        turns.append((speaker, text, text_script, turn_language))
    return turns


def build_story_plan(spec_path: Path, lang: str = "en") -> dict:
    data = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("meta"), dict):
        raise ValueError("The story YAML must contain a meta object.")
    meta = data["meta"]

    story = data.get("story")
    if not isinstance(story, dict):
        raise ValueError("The story YAML must contain a story object.")
    title = text_field(story.get("title"), "story.title")
    synopsis = text_field(story.get("synopsis"), "story.synopsis")
    logline = text_field(story.get("logline"), "story.logline")
    theme = text_field(story.get("theme"), "story.theme")

    boundaries = story.get("boundaries")
    if not isinstance(boundaries, dict):
        raise ValueError("The story YAML must define story.boundaries.")
    if boundaries.get("characters_are_adults") is not True:
        raise ValueError("The story YAML must explicitly identify all characters as adults.")
    intimacy = text_field(boundaries.get("intimacy"), "story.boundaries.intimacy")

    characters = story.get("characters")
    if not isinstance(characters, list) or not characters:
        raise ValueError("The story YAML must define its adult characters.")
    if any(not isinstance(item, dict) or item.get("adult") is not True for item in characters):
        raise ValueError("Every story character must be clearly identified as an adult.")
    character_names = [item["name"] for item in characters if isinstance(item.get("name"), str)]
    if not character_names:
        raise ValueError("Every story character must define a name.")
    known_names = set(character_names)

    scene_notes = story.get("scenes")
    if not isinstance(scene_notes, dict):
        raise ValueError("The story YAML must define scene descriptions.")

    negative_ref = meta.get("negative")
    if not isinstance(negative_ref, str) or not negative_ref.strip():
        raise ValueError("The story YAML must define meta.negative.")
    negative_path = (ROOT / negative_ref).resolve()
    if not negative_path.is_relative_to(ROOT):
        raise ValueError("meta.negative must reference a file inside the project.")
    if not negative_path.is_file():
        raise ValueError(f"Negative prompt file does not exist: {negative_ref}.")
    negative_prompt = negative_path.read_text(encoding="utf-8").strip()
    if not negative_prompt:
        raise ValueError(f"Negative prompt file is empty: {negative_ref}.")
    if len(negative_prompt) > MAX_NEGATIVE_CHARS:
        raise ValueError(
            f"Negative prompt exceeds the {MAX_NEGATIVE_CHARS}-character Film Studio limit."
        )

    duration = meta.get("duration")
    if duration != 5:
        raise ValueError("The production story YAML must define a five-second duration.")
    consent_beats = boundaries.get("consent_beats")
    if not isinstance(consent_beats, list) or not consent_beats:
        raise ValueError("story.boundaries.consent_beats must be a non-empty list.")
    shot_ids = {shot.get("id") for shot in data.get("shots", []) if isinstance(shot, dict)}
    if any(not isinstance(beat_id, str) or beat_id not in shot_ids for beat_id in consent_beats):
        raise ValueError("story.boundaries.consent_beats contains an unresolved id.")

    shots = data.get("shots")
    if not isinstance(shots, list) or not shots:
        raise ValueError("The story YAML must contain at least one ordered shot.")

    scenes: list[dict] = []
    scene_by_number: dict[int, dict] = {}
    starter_beats: list[str] = []
    target_seconds = 0
    previous_shot_id = None
    previous_anchors = None

    for index, shot in enumerate(shots, start=1):
        expected_id = f"c{index:02d}"
        if not isinstance(shot, dict) or shot.get("id") != expected_id:
            raise ValueError(f"Expected ordered shot {expected_id}.")

        scene_number = shot.get("scene")
        if not isinstance(scene_number, int) or scene_number < 1:
            raise ValueError(f"{expected_id}: scene must be a positive integer.")
        if scene_number not in scene_by_number:
            if scene_by_number and scene_number <= max(scene_by_number):
                raise ValueError(f"{expected_id}: scene numbers must increase in story order.")
            scene_note = scene_notes.get(scene_number, scene_notes.get(str(scene_number)))
            if not isinstance(scene_note, dict):
                raise ValueError(f"Scene {scene_number} is missing its story description.")
            scene = {
                "title": text_field(scene_note.get("title"), f"scene {scene_number} title"),
                "description": text_field(
                    scene_note.get("description"), f"scene {scene_number} description"
                ),
                "timeOfDay": scene_note.get("time_of_day", ""),
                "shots": [],
            }
            scene_by_number[scene_number] = scene
            scenes.append(scene)

        camera_lines = shot.get("cam")
        if not isinstance(camera_lines, list) or len(camera_lines) != 6:
            raise ValueError(f"{expected_id}: cam must contain exactly six lines.")
        prompt_lines = []
        for second, line in enumerate(camera_lines):
            if not isinstance(line, str) or not line.strip():
                raise ValueError(f"{expected_id}: camera line {second} must be non-empty text.")
            verb = line.strip().split(" ", 1)[0].lower().rstrip(",").rstrip(":")
            if verb not in VERBS:
                raise ValueError(f"{expected_id}: unsupported camera verb {verb!r}.")
            prompt_lines.append(f"(At {second} seconds: {line})")
        visual_prompt = "\n".join(prompt_lines)
        if len(visual_prompt) > MAX_PROMPT_CHARS:
            raise ValueError(
                f"{expected_id}: compiled prompt is {len(visual_prompt)} chars "
                f"(max {MAX_PROMPT_CHARS})."
            )

        build = shot.get("build")
        if build not in {"fm", "cf"}:
            raise ValueError(f"{expected_id}: build must be 'fm' or 'cf'.")

        start_frame = shot.get("start_frame")
        if not isinstance(start_frame, str) or not re.fullmatch(
            rf"startframes/{re.escape(expected_id)}\.png", start_frame
        ):
            raise ValueError(f"{expected_id}: start_frame must be startframes/{expected_id}.png")

        loras = shot.get("loras", [])
        if not isinstance(loras, list) or any(not isinstance(item, str) for item in loras):
            raise ValueError(f"{expected_id}: loras must be a list of strings.")
        chain = shot.get("chain")
        if not isinstance(chain, bool):
            raise ValueError(f"{expected_id}: chain must be a boolean.")

        turns = dialogue_turns(shot, expected_id, known_names)
        starter_beats.extend(f"{speaker}: {text}" for speaker, text, _, _ in turns)
        language = shot.get("language", "en")
        if not isinstance(language, str) or not language.strip():
            raise ValueError(f"{expected_id}: language must be non-empty text.")

        camera_text = " ".join(camera_lines)
        visible = [
            name for name in character_names
            if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", camera_text, re.IGNORECASE)
        ]
        if not visible:
            raise ValueError(f"{expected_id}: camera lines must identify at least one visible character.")

        resolution = meta.get("res_face") if build == "cf" else meta.get("res_base")
        if not isinstance(resolution, list) or len(resolution) != 2:
            raise ValueError(f"{expected_id}: missing resolution for build {build}.")
        beat_duration = shot.get("duration", duration)
        if not isinstance(beat_duration, int) or beat_duration < 1:
            raise ValueError(f"{expected_id}: duration must be a positive integer.")
        target_seconds += beat_duration
        noise = shot.get("noise")
        if noise is None:
            noise = 2
        elif not isinstance(noise, int) or not 0 <= noise <= 3:
            raise ValueError(f"{expected_id}: noise must be an integer from 0 to 3.")
        if noise == 3:
            if beat_duration > 4:
                raise ValueError(f"{expected_id}: noise 3 requires duration <= 4")
        elif beat_duration != duration:
            raise ValueError(
                f"{expected_id}: duration must match the production duration of {duration} seconds."
            )
        anchors = shot.get("anchors")
        if anchors is None:
            anchors = {
                "face": "authored face",
                "body": "authored body",
                "wardrobe": "authored wardrobe",
                "prop": "authored prop",
                "light": "authored light",
                "geography": "authored geography",
            }
        elif not isinstance(anchors, dict):
            raise ValueError(f"{expected_id}: anchors must be a mapping.")
        else:
            anchor_keys = {"face", "body", "wardrobe", "prop", "light", "geography"}
            missing_anchors = sorted(anchor_keys - set(anchors))
            if missing_anchors:
                raise ValueError(f"{expected_id}: anchors missing keys: {missing_anchors}.")
            for key in anchor_keys:
                if not isinstance(anchors[key], str) or not anchors[key].strip():
                    raise ValueError(f"{expected_id}: anchors.{key} must be non-empty text.")
        if chain and previous_anchors is not None:
            for key in STABLE_ANCHOR_KEYS:
                if anchors.get(key) != previous_anchors.get(key):
                    raise ValueError(
                        f"{expected_id}: {key} anchor must match {previous_shot_id} "
                        "when chain is true."
                    )
        previous_shot_id = expected_id
        previous_anchors = anchors
        profile = shot.get("profile", "draft")
        if profile not in {"draft", "hero"}:
            raise ValueError(f"{expected_id}: profile must be 'draft' or 'hero'.")
        hero_beats = {"c25", "c33", "c44", "c61", "c68"}
        if (profile == "hero") != (expected_id in hero_beats):
            raise ValueError(
                f"{expected_id}: hero profile must be exactly on c25, c33, c44, c61, c68."
            )
        audio_mode = shot.get("audio_mode", "none")
        if audio_mode not in {"none", "s2v"}:
            raise ValueError(f"{expected_id}: audio_mode must be 'none' or 's2v'.")
        if audio_mode == "s2v":
            if resolve_audio(ROOT, expected_id) is None:
                raise ValueError(
                    f"{expected_id}: s2v requires an audio sidecar "
                    f"(refs/audio/{expected_id}.wav or refs/audio/index.json)."
                )

        scene_by_number[scene_number]["shots"].append({
            "title": expected_id,
            "visualPrompt": visual_prompt,
            "negativePrompt": negative_prompt,
            "camera": "",
            "narration": "\n".join(
                f"{speaker}: {select_language_text(text, text_script, language, lang)}"
                if turn_language is None
                else f"{speaker}: {select_language_text(text, text_script, turn_language, lang)}"
                for speaker, text, text_script, turn_language in turns
            ),
            "characters": visible,
            "sourceBeat": {
                "id": expected_id,
                "scene": scene_number,
                "language": language,
                "build": build,
                "startFrame": start_frame,
                "loras": loras,
                "chain": chain,
                "sampler": meta.get("sampler"),
                "steps": meta.get("steps"),
                "cfg": meta.get("cfg"),
                "duration": beat_duration,
                "width": resolution[0],
                "height": resolution[1],
                "noise": noise,
                "light": shot.get("light"),
                "mark_state": shot.get("mark_state"),
                "thread": shot.get("thread", ""),
                "profile": profile,
                "anchors": anchors,
                "audio_mode": audio_mode,
            },
        })

    return {
        "title": title,
        "synopsis": synopsis,
        "defaultNegativePrompt": negative_prompt,
        "starterBrief": (
            f"{logline}\n"
            f"Theme: {theme}\n"
            f"Intimacy: {intimacy}\n"
            "All characters are adults. Preserve the authored scene order, character "
            "identities, prop continuity, and each speaker's exact dialogue. This import "
            "is editable; review matching start frames before generation.\n"
            + "\n".join(starter_beats)
        ),
        "targetSeconds": target_seconds,
        "scenes": scenes,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default="docs/beats.yaml")
    ap.add_argument("--lang", choices=("en", "hi", "ur", "mixed"), default="en")
    args = ap.parse_args()
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
        json.dump(
            build_story_plan(Path(args.spec), args.lang),
            sys.stdout,
            ensure_ascii=False,
        )
        sys.stdout.write("\n")
        return 0
    except (OSError, TypeError, ValueError, KeyError, yaml.YAMLError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())