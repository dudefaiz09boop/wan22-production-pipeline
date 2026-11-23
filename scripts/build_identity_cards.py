#!/usr/bin/env python3
"""Build the identity and location bible from hand-reviewed, sourced payloads."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bible_payload_faiz import FAIZ_CARD

ROOT = Path(__file__).resolve().parents[1]
STATUSES = {"photo_verified", "canon_unverified", "beats_canon"}


def claim(value: object, sources: list[str], status: str, notes: str | None = None) -> dict:
    if status not in STATUSES:
        raise ValueError(f"invalid claim status: {status}")
    result = {"value": value, "sources": sources, "status": status}
    if notes:
        result["notes"] = notes
    return result


JUVERIYA_CARD = {
    "name": "Juveriya",
    "bible_version": 1,
    "adult": True,
    "role": "female lead; adult daughter of Ammi; sister of Faiz (locked siblings canon)",
    "face": {
        "bone_structure": claim("round-heart face, soft cheek volume, small rounded chin",
                                ["refs/04.png", "refs/06.png", "refs/07.png"], "photo_verified"),
        "apparent_age": claim("adult, mid-to-late 20s appearance",
                              ["refs/04.png", "refs/05.png", "refs/06.png", "refs/07.png"], "photo_verified"),
        "nose": claim("small straight nose with narrow tip",
                      ["refs/04.png", "refs/06.png", "refs/07.png"], "photo_verified"),
        "brow": claim("full dark shaped brows with a soft arch",
                      ["refs/04.png", "refs/06.png", "refs/07.png"], "photo_verified"),
        "jaw": claim("soft jaw with no hard angle",
                     ["refs/04.png", "refs/07.png"], "photo_verified"),
        "lip_shape": claim("full lips with defined cupid's bow; pink or magenta lipstick in refs",
                           ["refs/04.png", "refs/07.png"], "photo_verified"),
        "eye_shape": claim("large almond eyes with open lids",
                           ["refs/04.png", "refs/06.png", "refs/07.png"], "photo_verified"),
        "eye_colour": claim("dark brown",
                            ["refs/04.png", "refs/06.png", "refs/07.png"], "photo_verified"),
        "eye_makeup": claim("dark kajal or eyeliner worn in reference stills",
                            ["refs/04.png", "refs/07.png"], "photo_verified"),
        "skin_tone": claim("light-medium with warm undertone",
                           ["refs/04.png", "refs/06.png", "refs/07.png"], "photo_verified"),
        "skin_undertone": claim("warm-neutral",
                                ["refs/04.png", "refs/06.png"], "photo_verified"),
        "marks_moles_scars": claim("no face marks or scars visible in the four refs",
                                   ["refs/04.png", "refs/05.png", "refs/06.png", "refs/07.png"],
                                   "photo_verified"),
        "beard": None,
        "hairline": claim("intact side-part hairline with fine baby hairs",
                          ["refs/04.png", "refs/07.png"], "photo_verified"),
        "hair": claim("very long straight black hair to hip length; often left loose",
                      ["refs/04.png", "refs/05.png", "refs/06.png", "refs/07.png"], "photo_verified"),
        "ear": claim("ornate gold jhumka or earring styles observed in refs",
                     ["refs/04.png", "refs/05.png", "refs/06.png", "refs/07.png"], "photo_verified"),
        "teeth": claim("natural adult teeth; no unusual dental feature visible",
                       ["refs/04.png", "refs/07.png"], "photo_verified"),
    },
    "body": {
        "height_ratio": claim("petite adult stature",
                              ["refs/05.png", "refs/07.png"], "photo_verified"),
        "build": claim("slim-petite frame with defined waist",
                       ["refs/05.png", "refs/07.png"], "photo_verified"),
        "shoulder_width": claim("narrow",
                                ["refs/05.png", "refs/06.png"], "photo_verified"),
        "waist": claim("small and defined in clothed reference views",
                       ["refs/05.png"], "photo_verified"),
        "hips": claim("proportionate to slim frame",
                      ["refs/05.png", "refs/07.png"], "photo_verified"),
        "chest": claim("moderate, proportionate; clothed reference only",
                       ["refs/05.png", "refs/06.png"], "photo_verified"),
        "limb_proportion": claim("proportionate slim arms and legs",
                                 ["refs/05.png", "refs/06.png"], "photo_verified"),
        "body_hair": None,
        "accessories": claim("thin gold chain, bangles and traditional jewellery observed in refs",
                             ["refs/05.png", "refs/06.png", "refs/07.png"], "photo_verified"),
        "height_vs_faiz": claim("beats imply Faiz is tall and strong enough to lift her; not confirmed in a shared plate",
                                ["beats:c08 cam", "beats:c09 cam"], "canon_unverified"),
    },
    "voice": {
        "pitch": None,
        "pace": None,
        "accent": claim("Hindi/Urdu and English code-switching is established by dialogue",
                        ["beats:c34 dialogue", "beats:c51 dialogue", "beats:c66 dialogue"],
                        "canon_unverified"),
        "performed_phone_voice": claim("pitched high and thin when deceiving Ammi over the phone",
                                       ["beats:c35 cam", "beats:c36 cam"], "beats_canon"),
        "delivery_notes": claim("commands with a flat, unhurried tone during later intimacy",
                                ["beats:c56 say", "beats:c61 say"], "beats_canon"),
    },
    "backstory": [
        {"text": "Adult daughter of Ammi; the study, desk and family flat are her everyday setting.",
         "sources": ["HANDOFF.md: section 1.1", "beats:c13 continuity", "beats:c46 continuity"],
         "status": "beats_canon"},
        {"text": "Adult sister of Faiz; the sibling relationship and secret are locked canon.",
         "sources": ["HANDOFF.md: section 1.1", "beats:c35 cam", "beats:c36 cam"],
         "status": "beats_canon"},
        {"text": "Initiates and directs later intimacy without withdrawing consent.",
         "sources": ["beats:c55 cam", "beats:c56 say", "beats:c61 say"],
         "status": "beats_canon"},
    ],
    "continuity_marks": [
        {"range": "c15-c26", "value": "fresh red tooth welt with four dent marks",
         "sources": ["beats:c15 cam", "beats:c17 continuity"], "status": "beats_canon"},
        {"range": "c27-c46", "value": "welt remains tender and visible",
         "sources": ["beats:c33 cam", "beats:c44 cam"], "status": "beats_canon"},
        {"range": "c47-c54", "value": "welt hidden under pinned dupatta",
         "sources": ["beats:c45 continuity", "beats:c47 cam"], "status": "beats_canon"},
        {"range": "c55-c70", "value": "yellow fading welt",
         "sources": ["beats:c55 continuity", "beats:c68 continuity"], "status": "beats_canon"},
    ],
    "wardrobe_states": [
        {"range": "c01-c03", "value": "loose kurti, grey skirt, dupatta on left shoulder, taveez, henna, six bangles each wrist",
         "sources": ["beats:c01 continuity"], "status": "beats_canon"},
        {"range": "c04-c09", "value": "skirt tears and is rucked to waist; panties removed; kurti bunched",
         "sources": ["beats:c04 continuity", "beats:c05 continuity", "beats:c08 continuity"],
         "status": "beats_canon"},
        {"range": "c10-c29", "value": "skirt at waist, kurti open or bunched, dupatta displaced",
         "sources": ["beats:c10 continuity", "beats:c20 continuity", "beats:c29 continuity"],
         "status": "beats_canon"},
        {"range": "c30-c31", "value": "Faiz's shirt over shoulders; one strap broken",
         "sources": ["beats:c30 continuity", "beats:c31 continuity"], "status": "beats_canon"},
        {"range": "c32-c38", "value": "fresh panties; skirt lowered; shirt open; kajal smeared",
         "sources": ["beats:c32 continuity", "beats:c34 continuity", "beats:c38 continuity"],
         "status": "beats_canon"},
        {"range": "c39-c40", "value": "shirt on, skirt smoothed, hair tied back",
         "sources": ["beats:c39 continuity", "beats:c40 continuity"], "status": "beats_canon"},
        {"range": "c41-c46", "value": "fresh shalwar, shirt removed, dupatta pinned",
         "sources": ["beats:c41 continuity", "beats:c45 continuity"], "status": "beats_canon"},
        {"range": "c47-c54", "value": "full shalwar and pinned dupatta; welt hidden",
         "sources": ["beats:c47 continuity", "beats:c52 continuity"], "status": "beats_canon"},
        {"range": "c55-c64", "value": "fresh kurti and skirt, dupatta off, hair loose, yellow fading welt",
         "sources": ["beats:c55 continuity", "beats:c58 continuity", "beats:c63 continuity"],
         "status": "beats_canon"},
        {"range": "c65-c70", "value": "daylight shalwar kameez and dupatta",
         "sources": ["beats:c65 continuity", "beats:c70 continuity"], "status": "beats_canon"},
    ],
    "forbidden": [
        "blonde, bleached or non-black hair",
        "hair shorter than hip length outside the beats-tracked tied-back window",
        "glasses",
        "nose piercing, facial piercings or visible tattoos",
        "bindi, sindoor or mangalsutra",
        "Western dress or bridal wear",
        "a third person in frame",
        "Ammi's face or full body in frame",
        "childlike proportions or any read contradicting adult: true",
    ],
    "plate_refs": ["refs/04.png", "refs/05.png", "refs/06.png", "refs/07.png"],
}


def location_card(slug: str, name: str, used: list[str], scenes: list[str], geography: dict,
                  props: list[dict], lighting: list[dict], camera_axis: dict, notes: list[str]) -> dict:
    return {
        "slug": slug,
        "name": name,
        "bible_version": 1,
        "used_in_shots": used,
        "beats_scenes": scenes,
        "geography": geography,
        "props": props,
        "lighting_states": lighting,
        "camera_axis": camera_axis,
        "notes": notes,
    }


LOCATIONS = [
    location_card(
        "study_desk", "Study and desk room",
        [f"c{i:02d}" for i in range(9, 47)] + [f"c{i:02d}" for i in range(55, 65)],
        ["scene 2", "scene 3", "scene 4", "scene 5", "scene 6", "scene 7", "scene 8", "scene 9",
         "scene 11"],
        claim("desk against an interior wall, passage door, bed, wardrobe, mirror, bookshelf and window",
              ["beats:c09 cam", "beats:c32 cam", "beats:c46 cam", "beats:c55 cam"], "beats_canon"),
        [
            claim("varnished desk, paper, textbooks, pen and drawers",
                  ["beats:c13 continuity", "beats:c46 continuity"], "beats_canon"),
            claim("mirror and framed art on walls",
                  ["beats:c18 continuity", "beats:c32 cam", "beats:c44 cam"], "beats_canon"),
            claim("ceiling fan, pull cord and failing tube light",
                  ["beats:c08 cam", "beats:c21 cam", "beats:c40 cam"], "beats_canon"),
            claim("window carrying moonlight stripes",
                  ["beats:c21 cam", "beats:c24 cam"], "beats_canon"),
            claim("wardrobe, bin and chair",
                  ["beats:c32 cam", "beats:c43 continuity"], "beats_canon"),
        ],
        [
            claim("night failing tube flicker", ["beats:c21 cam", "beats:c40 cam"], "beats_canon"),
            claim("blackout with moonlight stripes", ["beats:c21 cam", "beats:c24 cam"], "beats_canon"),
            claim("evening working tube", ["beats:c55 cam", "beats:c64 cam"], "beats_canon"),
        ],
        claim("passage-door axis toward the desk; reverse and top-down desk views; desk orbit used in c33/c46",
              ["beats:c09 cam", "beats:c33 cam", "beats:c46 cam"], "beats_canon"),
        ["The same room is used for desk, mirror, bed, wardrobe and prayer-mat geography."],
    ),
    location_card(
        "hallway", "Hallway and passage",
        [f"c{i:02d}" for i in range(1, 5)] + [f"c{i:02d}" for i in range(47, 55)] + [f"c{i:02d}" for i in range(65, 71)],
        ["scene 1", "scene 10", "scene 12"],
        claim("narrow passage linking front door, study, kitchen and bedroom",
              ["beats:c01 cam", "beats:c48 cam", "beats:c50 cam", "beats:c65 cam"], "beats_canon"),
        [
            claim("peeling wallpaper and damp patch",
                  ["beats:c01 cam"], "beats_canon"),
            claim("cold mosaic floor",
                  ["beats:c02 cam"], "beats_canon"),
            claim("bedroom and study doorways",
                  ["beats:c03 cam", "beats:c55 cam"], "beats_canon"),
            claim("shoe rack",
                  ["beats:c69 continuity", "beats:c70 continuity"], "beats_canon"),
        ],
        [
            claim("night failing tube", ["beats:c02 cam"], "beats_canon"),
            claim("pre-dawn grey", ["beats:c47 cam", "beats:c48 cam"], "beats_canon"),
            claim("morning daylight", ["beats:c65 cam", "beats:c67 cam"], "beats_canon"),
        ],
        claim("axial down the passage from front door to study, plus low mosaic-floor angles",
              ["beats:c01 cam", "beats:c02 cam", "beats:c48 cam"], "beats_canon"),
        [],
    ),
    location_card(
        "kitchen", "Kitchen and hatch",
        ["c45", "c50", "c51", "c54", "c65", "c66"],
        ["scene 10", "scene 12"],
        claim("counter, hatch toward passage and steel-vessel area",
              ["beats:c51 cam", "beats:c65 cam", "beats:c66 cam"], "beats_canon"),
        [
            claim("vegetable bags and tomato",
                  ["beats:c51 continuity"], "beats_canon"),
            claim("steel vessels",
                  ["beats:c65 continuity"], "beats_canon"),
            claim("fridge; heard starting, not directly seen",
                  ["beats:c45 cam"], "canon_unverified"),
        ],
        [
            claim("night dim", ["beats:c45 cam"], "beats_canon"),
            claim("dawn grey", ["beats:c51 cam", "beats:c54 cam"], "beats_canon"),
            claim("morning daylight", ["beats:c65 cam"], "beats_canon"),
        ],
        claim("from passage door and hatch toward counter; close on dropped vegetables",
              ["beats:c50 cam", "beats:c51 cam", "beats:c65 cam"], "beats_canon"),
        [],
    ),
    location_card(
        "bathroom", "Bathroom",
        ["c38", "c53", "c54"],
        ["scene 8", "scene 10"],
        claim("small utility room with bucket, taped mirror and water tap",
              ["beats:c38 cam", "beats:c53 cam"], "beats_canon"),
        [
            claim("bucket, cloth and taped mirror",
                  ["beats:c38 cam"], "beats_canon"),
            claim("wedding photo beside mirror",
                  ["beats:c38 cam"], "beats_canon"),
            claim("tap and running-water sound",
                  ["beats:c53 cam", "beats:c54 cam"], "beats_canon"),
        ],
        [
            claim("night interior light", ["beats:c38 cam"], "beats_canon"),
            claim("pre-dawn shut-door exterior", ["beats:c53 cam"], "beats_canon"),
            claim("morning water-light", ["beats:c54 cam"], "beats_canon"),
        ],
        claim("interior close on bucket and mirror; exterior frames the shut door from the study side",
              ["beats:c38 cam", "beats:c53 cam", "beats:c54 cam"], "beats_canon"),
        [],
    ),
    location_card(
        "front_door", "Front door and threshold",
        ["c45", "c47", "c48", "c49", "c67", "c68", "c69", "c70"],
        ["scene 10", "scene 12"],
        claim("front door with chain, latch and peephole; passage end with Ammi outside",
              ["beats:c45 cam", "beats:c47 cam", "beats:c48 cam", "beats:c49 cam"],
              "beats_canon"),
        [
            claim("chain and latch",
                  ["beats:c45 cam", "beats:c48 cam", "beats:c68 continuity"], "beats_canon"),
            claim("peephole and buzzer box",
                  ["beats:c47 cam", "beats:c48 cam"], "beats_canon"),
            claim("mosaic threshold",
                  ["beats:c49 continuity"], "beats_canon"),
            claim("corridor outside with one failing bulb",
                  ["beats:c47 cam"], "beats_canon"),
        ],
        [
            claim("night corridor", ["beats:c45 cam"], "beats_canon"),
            claim("pre-dawn corridor", ["beats:c47 cam", "beats:c48 cam"], "beats_canon"),
            claim("morning corridor daylight", ["beats:c67 cam", "beats:c68 cam"], "beats_canon"),
        ],
        claim("perpendicular close framings on chain, latch and peephole; profile along passage toward door",
              ["beats:c47 cam", "beats:c48 cam", "beats:c67 cam"], "beats_canon"),
        ["Ammi is never rendered as a full body; only voice and hands enter frame at c49-c50."],
    ),
]


def validate_card(card: dict) -> list[str]:
    errors: list[str] = []
    name = card.get("name", "<unnamed>")
    for section in ("face", "body", "voice"):
        for key, entry in card.get(section, {}).items():
            if entry is None:
                continue
            if not isinstance(entry, dict):
                errors.append(f"{name}.{section}.{key}: claim must be a dict or null")
                continue
            if "value" not in entry or "status" not in entry or not entry.get("sources"):
                errors.append(f"{name}.{section}.{key}: claim requires value, sources and status")
            elif entry["status"] not in STATUSES:
                errors.append(f"{name}.{section}.{key}: invalid status {entry['status']}")
    for key in ("forbidden", "plate_refs"):
        value = card.get(key)
        if not isinstance(value, list) or not value:
            errors.append(f"{name}: {key} must be a non-empty list")
    for key in ("backstory", "wardrobe_states"):
        value = card.get(key)
        if not isinstance(value, list) or not value:
            errors.append(f"{name}: {key} must be a non-empty list")
    for plate in card.get("plate_refs", []):
        if not (ROOT / plate).is_file():
            errors.append(f"{name}: missing plate ref {plate}")
    return errors


def validate_locations() -> list[str]:
    errors: list[str] = []
    for card in LOCATIONS:
        name = card["name"]
        if len(card["lighting_states"]) != 3:
            errors.append(f"{name}: must declare exactly three lighting states")
        for key in ("geography", "camera_axis"):
            entry = card[key]
            if not isinstance(entry, dict) or "value" not in entry or not entry.get("sources"):
                errors.append(f"{name}: {key} requires value and sources")
        for prop in card["props"]:
            if not prop.get("sources") or "value" not in prop:
                errors.append(f"{name}: prop requires value and sources")
    return errors


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def build(*, check: bool = False) -> list[str]:
    cards = [FAIZ_CARD, JUVERIYA_CARD]
    errors = []
    for card in cards:
        errors.extend(validate_card(card))
    errors.extend(validate_locations())
    if errors:
        return errors

    outputs = {
        ROOT / "bible" / "faiz.json": FAIZ_CARD,
        ROOT / "bible" / "juveriya.json": JUVERIYA_CARD,
    }
    for location in LOCATIONS:
        outputs[ROOT / "bible" / "locations" / f"{location['slug']}.json"] = location

    if check:
        changed = []
        for path, payload in outputs.items():
            expected = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
            if not path.is_file() or path.read_text(encoding="utf-8") != expected:
                changed.append(str(path.relative_to(ROOT)))
        return [f"bible drift: {path}" for path in changed]

    for path, payload in outputs.items():
        write_json(path, payload)
    return []


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="verify generated bible cards are current")
    args = parser.parse_args(argv)
    errors = build(check=args.check)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("Built 2 character cards and 5 location cards")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
