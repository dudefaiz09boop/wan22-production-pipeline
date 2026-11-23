#!/usr/bin/env python3
"""Compose detailed per-beat start-frame prompts for the Qwen space runner.

The production contract builds a single-line prompt from the first camera line,
which is far too thin for the hosted Qwen space: it produced anatomy close-ups
with no characters, wardrobe or setting. This composer writes a full cinematic
prompt per beat from docs/beats.yaml, the frozen character bibles and the
location bibles, plus the reference-plate / LoRA / mode assignment the space
needs for identity-locked generation.

Output: prompts/startframe_prompts.json (generated asset, safe to overwrite).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "prompts" / "startframe_prompts.json"

FAIZ_PLATE = "refs/02.png"
JUV_PLATE = "refs/04.png"

MODE_T2I = "Text to Image"
MODE_EDIT = "Edit Image (1 Ref)"
MODE_TWO = "Transform & Swap (2 Refs)"

LORA_NONE = "None (Base Uncensored)"
LORA_NSFW = "NSFW (general)"

MAX_PROMPT_CHARS = 3900

FAIZ_ID = (
    "adult South Asian man in his mid-to-late twenties, oval-oblong face with a "
    "defined jawline, short trimmed boxed beard with a connected moustache, thick "
    "dark brows, large almond dark-brown eyes, straight medium nose, black hair of "
    "medium length styled up, warm medium-brown skin, slim lean build with a "
    "wristwatch on his left wrist"
)
JUV_ID = (
    "adult South Asian woman in her mid-to-late twenties, round-heart face with soft "
    "cheeks and a small rounded chin, small straight nose, full dark shaped brows, "
    "full lips with pink lipstick, large almond dark-brown eyes lined with kajal, "
    "very long straight black hip-length hair, light-medium warm-toned skin, slim "
    "petite frame with narrow shoulders, gold jhumka earrings, a thin gold chain and "
    "glass bangles on both wrists"
)

FAIZ_WARDROBE = [
    (("c01", "c26"), "bare-chested with a plain shalwar sitting low on his hips, barefoot"),
    (("c27", "c29"), "bare-chested with a plain shalwar low on his hips"),
    (("c30", "c46"), "bare-chested with a plain shalwar on his hips"),
    (("c47", "c54"), "wearing a light cotton shirt tucked into his trousers and buttoned"),
    (("c55", "c58"), "wearing a light cotton shirt"),
    (("c59", "c64"), "shirtless with trousers on"),
    (("c65", "c70"), "wearing a pressed light kurta"),
]

JUV_WARDROBE = [
    (("c01", "c04"), "wearing a loose printed kurti, a grey skirt, a dupatta over her "
                     "left shoulder, a taveez amulet, henna on both hands and six glass "
                     "bangles on each wrist"),
    (("c05", "c09"), "wearing a torn grey skirt rucked up to her waist with her kurti "
                     "bunched and twisted, her dupatta gone, a taveez amulet on her skin "
                     "and glass bangles on both wrists"),
    (("c10", "c29"), "wearing her skirt rucked up to her waist and her kurti open and "
                     "bunched at the chest, a taveez amulet on her bare skin, glass "
                     "bangles on both wrists, her dupatta on the floor"),
    (("c30", "c30"), "wearing her rucked skirt with a broken strap and her open, "
                     "bunched kurti, her dupatta on the floor"),
    (("c31", "c31"), "wearing Faiz's shirt hanging open over her shoulders over a "
                     "broken strap, her skirt at her waist"),
    (("c32", "c32"), "holding a fresh pair of cotton panties, her skirt lowered, in "
                     "Faiz's shirt left hanging open over her, kajal smeared"),
    (("c33", "c38"), "wearing fresh cotton panties with her skirt lowered, Faiz's shirt "
                     "open over her, kajal smeared"),
    (("c39", "c40"), "wearing Faiz's shirt loose over her with her skirt smoothed down, "
                     "her hair tied back"),
    (("c41", "c41"), "stepping into a fresh shalwar with her old skirt off and a dupatta "
                     "pinned at her left shoulder"),
    (("c42", "c46"), "wearing a fresh shalwar kameez with the dupatta pinned at her left "
                     "shoulder, henna on both hands"),
    (("c47", "c54"), "wearing a full shalwar kameez with the dupatta pinned at her left "
                     "shoulder, henna on both hands, her hair tied back"),
    (("c55", "c64"), "wearing a fresh kurti and skirt, no dupatta, her long hair loose"),
    (("c65", "c70"), "wearing a daylight shalwar kameez with a dupatta"),
]

WELT_FRESH = "a dark red-purple bite welt with four tooth marks on the side of her neck"
WELT_FADED = "a faint yellow fading bite welt on the side of her neck"

STUDY = (
    "a small middle-class Indian study room: a varnished wooden desk against an "
    "interior wall cluttered with papers and textbooks, a chair, a wardrobe, a wall "
    "mirror and framed art, a bookshelf, a ceiling fan with a pull cord, a window "
    "with blinds, cold mosaic floor, a door to the passage"
)
STUDY_DARK = STUDY + ", the room in blackout except for pale moonlight stripes through the window blinds"
BATHROOM = (
    "a small Indian utility bathroom: a plastic bucket half full of water with a "
    "cloth floating on the surface, a mirror with taped edges, a water tap, a framed "
    "wedding photo beside the mirror, wet mosaic floor"
)
HALLWAY = (
    "a narrow passage inside an Indian flat: peeling wallpaper with a damp patch, "
    "cold mosaic floor, doorways to the study and bedrooms"
)
FRONT_DOOR = (
    "the front door of an Indian flat: security chain and latch, a peephole, a buzzer "
    "intercom box on the passage wall beside it, corridor light beyond"
)
KITCHEN = (
    "a small Indian kitchen: a counter with steel vessels, plastic vegetable bags, a "
    "hatch opening toward the passage"
)

LIGHT = {
    "night_tube": "lit only by a failing cold-white fluorescent tube light that "
                  "flickers overhead, harsh shadows, dim night edges",
    "evening_tube": "lit by a steady cold fluorescent tube light, evening interior",
    "blackout": "near total darkness broken only by thin pale moonlight stripes",
    "dawn_grey": "pale pre-dawn grey light, no lamps on",
    "morning": "soft morning daylight",
    "night_dim": "dim night interior light",
    "daylight": "soft daylight",
}

STYLE_CHARACTERS = (
    "Photorealistic cinematic film still from an intimate adult drama, 16:9 "
    "widescreen, natural skin texture and realistic adult anatomy, shallow filmic "
    "depth of field, no text, no watermark, no other people in frame."
)
STYLE_OBJECTS = (
    "Photorealistic cinematic film still from an intimate adult drama, 16:9 "
    "widescreen, natural textures, shallow filmic depth of field, no people, no text, "
    "no watermark."
)

# who: both | faiz | juv | none
# scene: the frozen first-instant action, from the beat's cam[0] and continuity.
BEATS = {
    "c01": {
        "who": "juv", "location": "hallway", "light": "night_tube", "nsfw": False,
        "scene": "Static wide in the narrow passage: Juveriya fills the near frame, "
                 "caught mid-breath and looking away, her dupatta slipping from her left "
                 "shoulder, her bangles and henna in the stuttering light, peeling "
                 "wallpaper and a spreading damp patch on the wall behind her; at the far "
                 "end of the passage, small and out of focus in the dark study doorway, "
                 "stands the bare-chested silhouette of a man in a low shalwar.",
        "marks": None,
    },
    "c02": {
        "who": "faiz", "location": "hallway", "light": "night_tube", "nsfw": False,
        "scene": "Static medium from Juveriya's point of view: Faiz fills the dark "
                 "doorway at the end of the passage, bare-chested with his plain "
                 "shalwar sitting low on his hips and one bare foot on the cold mosaic "
                 "floor, the failing tube light stuttering over his face and the "
                 "peeling wallpaper behind him.",
        "marks": None,
    },
    "c03": {
        "who": "both", "location": "hallway", "light": "night_tube", "nsfw": False,
        "scene": "Static close-up in the flickering light: Faiz's hand has just "
                 "snapped around Juveriya's wrist, the edge of her glass bangles "
                 "digging into her skin and her hennaed fingers splaying, her dupatta "
                 "trailing off her shoulder, the two of them a step from the wall in "
                 "the narrow passage.",
        "marks": None,
    },
    "c04": {
        "who": "both", "location": "hallway", "light": "night_tube", "nsfw": False,
        "scene": "Static close-up beside the passage wall, both of them in frame: Faiz "
                 "has his fist bunched in the side of Juveriya's grey skirt at her "
                 "mid-thigh and is wrenching the hem up, the cotton pulled taut; her "
                 "printed kurti is still on and bunched against him, her dupatta "
                 "slipping off her left shoulder, her face turned up to his with her "
                 "mouth open and her hennaed hand catching at his wrist, peeling "
                 "wallpaper and the flickering tube light behind them.",
        "marks": None,
    },
    "c05": {
        "who": "both", "location": "study", "light": "night_tube", "nsfw": True,
        "scene": "Static extreme close-up: Faiz's hand is under the torn hem of "
                 "Juveriya's rucked-up skirt, his fingers on her bare skin, her kurti "
                 "bunched and twisted, the taveez amulet on her bare skin, the two of "
                 "them against the study wall and her mouth just opening.",
        "marks": None,
    },
    "c06": {
        "who": "both", "location": "study", "light": "night_tube", "nsfw": True,
        "scene": "Static close-up against the study wall: the pad of Faiz's thumb "
                 "circles over Juveriya's bare clit with the hood pulled back and "
                 "everything glistening, her thighs trembling, her kurti pushed up to "
                 "her collarbone and her nails raking his shoulder in the flickering "
                 "light.",
        "marks": None,
    },
    "c07": {
        "who": "both", "location": "study", "light": "night_tube", "nsfw": True,
        "scene": "Static medium at the study wall: Faiz's hips are drawn back with the "
                 "head of his hard cock just at Juveriya's glistening entrance, her "
                 "spine arched off the plaster and her kurti riding her ribs, her "
                 "taveez swinging between them, her bangles frozen mid-ring.",
        "marks": None,
    },
    "c08": {
        "who": "both", "location": "study", "light": "night_tube", "nsfw": True,
        "scene": "Static low angle at the study wall: Juveriya's chin is tipped to the "
                 "ceiling and her throat is working, her kurti at her ribs, while Faiz "
                 "drives into her hard above her, his jaw clenched and the tendon "
                 "standing in his neck, plaster dust in the strobing light.",
        "marks": None,
    },
    "c09": {
        "who": "both", "location": "study", "light": "night_tube", "nsfw": True,
        "scene": "Faiz stands hoisting Juveriya up off the floor against him, her legs "
                 "wrapped around his waist, his erect penis still buried inside her, her "
                 "back beginning to drag along the wall as he carries her. Both are "
                 "caught mid-motion, strain and heat in their bodies, faces close.",
        "marks": "fresh",
    },
    "c10": {
        "who": "both", "location": "study", "light": "night_tube", "nsfw": True,
        "scene": "Juveriya lies back across the varnished wooden desk, papers and "
                 "textbooks scattered under and around her, while Faiz stands between her "
                 "legs gripping both her ankles and folding her legs back toward her "
                 "chest, penetrating her, her back arched into the desktop.",
        "marks": "fresh",
    },
    "c11": {
        "who": "both", "location": "study", "light": "night_tube", "nsfw": True,
        "scene": "Juveriya's ankles are pinned wide apart, Faiz driving into her, her "
                 "hips rocking the desk so it grinds across the mosaic floor, papers "
                 "sliding, her face thrown back and her mouth open.",
        "marks": "fresh",
    },
    "c12": {
        "who": "both", "location": "study", "light": "night_tube", "nsfw": True,
        "scene": "Static close-up across the desk: a sweat bead rolls down Juveriya's "
                 "stomach where her skirt is rucked at her waist and her kurti hangs "
                 "open, the taveez pressed into her sternum, Faiz's hand pinning her "
                 "wrist above her head, their joined hips just in the middle distance.",
        "marks": None,
    },
    "c13": {
        "who": "both", "location": "study", "light": "night_tube", "nsfw": True,
        "scene": "Static medium over the wrecked desk: Faiz's fist is in Juveriya's "
                 "pulled-loose curls, dragging her head back until her throat is "
                 "bared, her kurti open, her legs shaking and her toes curling, "
                 "framed art rattling on the study wall behind them.",
        "marks": None,
    },
    "c14": {
        "who": "both", "location": "study", "light": "night_tube", "nsfw": True,
        "scene": "Static wide on the shuddering desk: Juveriya's thighs slap against "
                 "Faiz's hips, her head thrown back with the tendons of her throat "
                 "straining, his fist wrenching her hair, her kurti open to the navel, "
                 "papers sliding off the desk in the flickering light.",
        "marks": None,
    },
    "c15": {
        "who": "both", "location": "study", "light": "night_tube", "nsfw": True,
        "scene": "Static extreme close-up: Faiz's mouth is open against the side of "
                 "Juveriya's neck and shoulder, the taveez displaced by his lips, her "
                 "kurti open and her head tipped away to give him her neck, her hand "
                 "just rising to his arm.",
        "marks": None,
    },
    "c16": {
        "who": "both", "location": "study", "light": "night_tube", "nsfw": True,
        "scene": "Static medium across the desk: Juveriya's hair is twisted in Faiz's "
                 "fist and her head is wrenched sideways, her cheek pushed into the "
                 "wood grain with one eye open, her skirt at her waist and twisted, "
                 "one kurti sleeve off her shoulder, her lips skidding on the varnish.",
        "marks": "fresh",
    },
    "c17": {
        "who": "both", "location": "study", "light": "night_tube", "nsfw": True,
        "scene": "Static top-down: Juveriya is pinned face down across the varnished "
                 "desk, her spine curved and her skirt at her waist with the hem under "
                 "her knee, the taveez flat on the desk beside her white-knuckled "
                 "grip, Faiz's weight bearing down over her back.",
        "marks": "fresh",
    },
    "c18": {
        "who": "both", "location": "study", "light": "night_tube", "nsfw": True,
        "scene": "Static close-up: Juveriya's bare arse is raised with the skin "
                 "glowing pink, her skirt rucked up over it, and Faiz's palm is "
                 "frozen in the air an instant before it lands, her shoulders and the "
                 "desk edge framing the shot in the tube light.",
        "marks": "fresh",
    },
    "c19": {
        "who": "both", "location": "study", "light": "night_tube", "nsfw": True,
        "scene": "Static close-up from behind over the desk: two of Faiz's fingers are "
                 "buried in Juveriya, slick, her skirt at her waist and the taveez "
                 "swinging with the push, his other hand pinning her lower back flat "
                 "to the desktop.",
        "marks": "fresh",
    },
    "c20": {
        "who": "both", "location": "study_dark", "light": "blackout", "nsfw": True,
        "scene": "Faiz drives deep into Juveriya across the desk, her spine dipping "
                 "back into the varnished desktop, thin bands of moonlight striping "
                 "across their bodies in the dark.",
        "marks": "fresh",
    },
    "c21": {
        "who": "both", "location": "study", "light": "night_tube", "nsfw": True,
        "scene": "Static wide in the dark: the failing tube light strobes over Faiz "
                 "and Juveriya locked together in the middle of the study, her hands "
                 "braced on the desk in front of her, his hand gripping her hip, the "
                 "room dropping to black between the pulses.",
        "marks": "fresh",
    },
    "c22": {
        "who": "both", "location": "study", "light": "blackout", "nsfw": True,
        "scene": "Static near-black shot: Juveriya's hand slides on the desk in the "
                 "dark and knocks a steel glass toward the edge, her wet knuckles and "
                 "bangles faint in the thin moonlight stripes, her forehead down on the "
                 "desktop, Faiz's dark shape over her, their hips still joined.",
        "marks": "fresh",
    },
    "c23": {
        "who": "both", "location": "study", "light": "blackout", "nsfw": True,
        "scene": "Static extreme close-up in the near-dark: Faiz's mouth is against "
                 "Juveriya's ear, his lips touching it as he whispers, her eyes just "
                 "opening, unfocused and lash-wet, the taveez at her throat and a thin "
                 "moonlight stripe falling across their two faces.",
        "marks": "fresh",
    },
    "c24": {
        "who": "juv", "location": "study", "light": "blackout", "nsfw": True,
        "scene": "Static tight on Juveriya's face in the dark with one thin moonlight "
                 "stripe across her eyes, her mouth open mid-gasp and her kurti open, "
                 "her bangles and the wrecked desk faint behind her.",
        "marks": "fresh",
    },
    "c25": {
        "who": "both", "location": "study", "light": "night_tube", "nsfw": True,
        "scene": "Static medium in the flickering light: a frantic rhythm with Faiz's "
                 "hips snapping and Juveriya's hips meeting his, her hand grabbing his "
                 "wrist above her rucked skirt, her throat corded and her mouth shaping "
                 "words, his fingers denting her hip.",
        "marks": "fresh",
    },
    "c26": {
        "who": "both", "location": "study", "light": "night_tube", "nsfw": True,
        "scene": "Juveriya is bent over the desk edge, Faiz standing behind her driving "
                 "to the hilt, the desk shoved out of square against the wall, papers "
                 "spilling to the floor, her taveez amulet lying flat on the desk surface "
                 "beside her hands.",
        "marks": "fresh",
    },
    "c27": {
        "who": "both", "location": "study", "light": "night_tube", "nsfw": True,
        "scene": "Static shot across the stilled desk: Faiz's weight is fully down on "
                 "Juveriya, her ribs compressing under his chest and both of them "
                 "gasping, her hand patting his shoulder twice, one of her legs "
                 "sliding to the floor, their ruined clothes and scattered papers "
                 "around them.",
        "marks": "fresh",
    },
    "c28": {
        "who": "both", "location": "study", "light": "night_tube", "nsfw": True,
        "scene": "Faiz, bare-chested, pulls out of Juveriya and her hips lift after him, "
                 "his wet penis slipping free and glistening, fluid shining on her inner "
                 "thigh, her skirt still at her waist, the taveez amulet resting on the "
                 "desk edge.",
        "marks": "fresh",
    },
    "c29": {
        "who": "both", "location": "study", "light": "night_tube", "nsfw": False,
        "scene": "Static low angle beside the desk: Faiz crouches in front of a "
                 "kneeling, wrecked Juveriya and hooks her chin up with one finger, "
                 "her drowsy eyes just finding his and his grin beginning to spread, "
                 "her skirt still at her waist, the taveez at her throat.",
        "marks": "fresh",
    },
    "c30": {
        "who": "juv", "location": "study", "light": "night_tube", "nsfw": False,
        "scene": "Static medium: Juveriya pushes up on shaking arms from the desk, "
                 "wrecked and flushed with her hair a mess, the study a disaster of "
                 "scattered papers and the dupatta on the floor behind her.",
        "marks": "fresh",
    },
    "c31": {
        "who": "none", "location": "study", "light": "night_tube", "nsfw": False,
        "scene": "Extreme close shot of a pair of shredded cotton panties lying crumpled "
                 "on the cold mosaic floor, the elastic waistband snapped, the fabric "
                 "ripped open; the leg of the varnished desk and scattered papers sit "
                 "out of focus behind, in the study room.",
        "marks": None,
    },
    "c32": {
        "who": "juv", "location": "study", "light": "night_tube", "nsfw": False,
        "scene": "Static medium at the open wardrobe: Juveriya stands facing the stack, "
                 "the panties in her hennaed hands just pulled free, the study a mess "
                 "behind her.",
        "marks": "fresh",
    },
    "c33": {
        "who": "juv", "location": "study", "light": "night_tube", "nsfw": False,
        "scene": "Static top-down: Juveriya's bare foot rolls the prayer mat open on "
                 "the study's cold mosaic floor, the fresh shalwar on her and Faiz's "
                 "open shirt on her shoulders, henna on her hands, the welt just "
                 "showing at the side of her neck.",
        "marks": "fresh",
    },
    "c34": {
        "who": "juv", "location": "study", "light": "night_tube", "nsfw": False,
        "scene": "Static close-up: the phone buzzes on the varnished desk with AMMI on "
                 "the screen, and Juveriya's hennaed hand is frozen an inch above it "
                 "about to slap it silent, her kajal-smeared eyes wide in the dark "
                 "glass, his shirt open on her shoulders.",
        "marks": "fresh",
    },
    "c35": {
        "who": "both", "location": "study", "light": "night_tube", "nsfw": True,
        "scene": "Static medium: the phone is at Juveriya's ear, her voice pitched "
                 "high and her eyes a little too wide, his open shirt on her and the "
                 "taveez at her throat, Faiz close behind her at the desk with his "
                 "fingers laced through hers at her stomach, both of them very still.",
        "marks": "fresh",
    },
    "c36": {
        "who": "both", "location": "study", "light": "night_tube", "nsfw": True,
        "scene": "Static shot at the desk: Faiz laughs into Juveriya's shoulder with "
                 "his face hidden in her neck while she pinches his bare arm with her "
                 "hennaed fingers, the phone still held a little away from her ear, "
                 "her mouth fighting a grimace.",
        "marks": "fresh",
    },
    "c37": {
        "who": "juv", "location": "study", "light": "night_tube", "nsfw": True,
        "scene": "Static close-up: Juveriya mouths 'another hour' into the phone, her "
                 "brows up in a plea, her kajal-smeared eyes thrown sideways in "
                 "despair, his open shirt on her shoulders and the taveez at her "
                 "throat, the varnished desk in front of her.",
        "marks": "fresh",
    },
    "c38": {
        "who": "none", "location": "bathroom", "light": "night_dim", "nsfw": False,
        "scene": "Wide shot of the small utility bathroom at night: the plastic bucket "
                 "half full of water with the cloth floating on the surface, a wet "
                 "mosaic floor, the taped mirror and the framed wedding photo beside it "
                 "catching the dim light.",
        "marks": None,
    },
    "c39": {
        "who": "none", "location": "study", "light": "night_tube", "nsfw": False,
        "scene": "Close shot of a wet patch on the varnished desktop where the finish has "
                 "gone dull and tacky, catching the flickering tube light; the desk is "
                 "otherwise cleared, papers pushed aside.",
        "marks": None,
    },
    "c40": {
        "who": "none", "location": "study", "light": "night_tube", "nsfw": False,
        "scene": "The study at night: the tube light flickering on, the ceiling fan "
                 "lurching into motion, an open textbook lying on the desk, a chair "
                 "pushed back, the room otherwise empty.",
        "marks": None,
    },
    "c41": {
        "who": "juv", "location": "study", "light": "night_tube", "nsfw": False,
        "scene": "Juveriya steps into a fresh shalwar and pulls the drawstring tight at "
                 "her waist, her old torn skirt hanging off one hip, Faiz's shirt "
                 "slipped down her shoulders, turned three-quarters away from the "
                 "camera.",
        "marks": "fresh",
    },
    "c42": {
        "who": "both", "location": "study", "light": "night_tube", "nsfw": False,
        "scene": "A two-shot at the wall mirror: Juveriya stands before the glass in a "
                 "fresh shalwar kameez, pulling the dupatta down at her shoulder to show "
                 "the bite welt on her neck; behind her Faiz stands bare-chested, and "
                 "both of them and the room are visible in the reflection.",
        "marks": "fresh",
    },
    "c43": {
        "who": "juv", "location": "study", "light": "night_tube", "nsfw": False,
        "scene": "Top-down shot: Juveriya crouches on the mosaic floor gathering "
                 "scattered pages into a neat stack, henna on her hands, the fresh "
                 "shalwar and pinned dupatta on; Faiz's shirt lies discarded over the "
                 "chair behind her.",
        "marks": "fresh",
    },
    "c44": {
        "who": "juv", "location": "study", "light": "night_tube", "nsfw": False,
        "scene": "Extreme close shot of Juveriya's reflection in the mirror: the side of "
                 "her neck and jaw, the dark bite welt with tooth marks just above the "
                 "pinned dupatta, her hair tied back, caught in the dim tube light.",
        "marks": "fresh",
    },
    "c45": {
        "who": "none", "location": "front_door", "light": "night_dim", "nsfw": False,
        "scene": "The front door of the flat at night, the security chain hanging loose "
                 "and the door standing ajar, the dark hallway beyond, the buzzer box on "
                 "the wall beside the doorframe.",
        "marks": None,
    },
    "c46": {
        "who": "none", "location": "study", "light": "night_tube", "nsfw": False,
        "scene": "The study tidied: the varnished desk neat, an open textbook squared on "
                 "the surface with a pen laid straight beside it, the chair pushed in, "
                 "the room still under the flickering tube light.",
        "marks": None,
    },
    "c47": {
        "who": "juv", "location": "front_door", "light": "dawn_grey", "nsfw": False,
        "scene": "Juveriya stands at the front door in a full shalwar kameez with the "
                 "dupatta pinned at her left shoulder, her hair tied back, henna on both "
                 "hands, one hand resting on the doorframe as she listens.",
        "marks": "fresh",
    },
    "c48": {
        "who": "none", "location": "front_door", "light": "dawn_grey", "nsfw": False,
        "scene": "Close shot of the buzzer intercom box on the passage wall rattling in "
                 "its casing, the plaster around it slightly cracked, dawn grey light "
                 "from the corridor; the doorframe fills the edge of the frame.",
        "marks": None,
    },
    "c49": {
        "who": "faiz", "location": "front_door", "light": "dawn_grey", "nsfw": False,
        "scene": "The front door swings back. Faiz stands in the passage caught "
                 "mid-step, his light shirt buttoned wrong with one button out of line; "
                 "at the very edge of the frame only an older woman's hands and forearms "
                 "are visible, gripping two heavy plastic bags of vegetables — her face "
                 "and body stay out of frame.",
        "marks": None,
    },
    "c50": {
        "who": "none", "location": "hallway", "light": "dawn_grey", "nsfw": False,
        "scene": "Extreme close shot: an older woman's thumb and fingers catch the edge "
                 "of a worn dupatta at the back of a younger woman's neck and begin to "
                 "drag it slowly across a dark healing bite welt with four tooth marks. "
                 "The younger woman's face is out of frame; only the older hand enters "
                 "the picture, with the passage wall behind.",
        "marks": None,
    },
    "c51": {
        "who": "none", "location": "kitchen", "light": "dawn_grey", "nsfw": False,
        "scene": "A small kitchen at dawn: seen only from the shoulders down, an older "
                 "woman in a plain cotton saree drops two plastic bags of vegetables "
                 "onto the counter and stands still, not turning round; her face stays "
                 "above the top edge of the frame.",
        "marks": None,
    },
    "c52": {
        "who": "none", "location": "study", "light": "dawn_grey", "nsfw": False,
        "scene": "The study doorway: the room beyond is neat, the desk squared and the "
                 "chair pushed in; in the doorway the shoulder and arm of an older woman "
                 "in a cotton saree fill the edge of frame, her face and body out of "
                 "frame.",
        "marks": None,
    },
    "c53": {
        "who": "none", "location": "study", "light": "dawn_grey", "nsfw": False,
        "scene": "A plain painted bathroom door clicking shut, seen from the study side "
                 "in pre-dawn grey, the handle still turning, a sliver of dark bathroom "
                 "beyond, the passage empty.",
        "marks": None,
    },
    "c54": {
        "who": "none", "location": "study", "light": "dawn_grey", "nsfw": False,
        "scene": "The bathroom door shut with a thin line of warm light under it, water "
                 "beginning to run inside, the passage floor grey and empty in the "
                 "pre-dawn light.",
        "marks": None,
    },
    "c55": {
        "who": "none", "location": "study", "light": "evening_tube", "nsfw": False,
        "scene": "Evening in the study: the tube light on and steady, the door shut, the "
                 "desk cleared bare, the chair squared, the room tidy and still.",
        "marks": None,
    },
    "c56": {
        "who": "juv", "location": "study", "light": "evening_tube", "nsfw": False,
        "scene": "Static close-up on Juveriya's face and mouth in the steady evening "
                 "tube light, her eyes lowered and her lips mid-sentence, the words flat "
                 "and unhurried, the faint yellowed welt just visible at the edge of her "
                 "jaw.",
        "marks": "faded",
    },
    "c57": {
        "who": "both", "location": "study", "light": "evening_tube", "nsfw": False,
        "scene": "Static close-up: Juveriya's fist is knotted in the collar of Faiz's "
                 "light cotton shirt, pulling him down toward her, the fabric going "
                 "taut, her kurti tucked and her skirt at the knee, the study lit "
                 "steady and cold around them.",
        "marks": "faded",
    },
    "c58": {
        "who": "both", "location": "study", "light": "evening_tube", "nsfw": False,
        "scene": "Static medium: Juveriya's hennaed hands work the buttons of Faiz's "
                 "light cotton shirt from the bottom, the shirt parting over his chest "
                 "with her knuckles against it, her own kurti still on and tucked, the "
                 "two of them standing in the cold evening tube light.",
        "marks": "faded",
    },
    "c59": {
        "who": "juv", "location": "study", "light": "evening_tube", "nsfw": True,
        "scene": "Static close-up: Juveriya's hennaed fingers pull her own cotton "
                 "panties to one side under her rucked-up kurti, her skirt at her "
                 "waist, her face just visible at the top of the frame with her eyes "
                 "down on what she is doing, the steady evening light on her hands and "
                 "skin.",
        "marks": "faded",
    },
    "c60": {
        "who": "both", "location": "study", "light": "evening_tube", "nsfw": True,
        "scene": "Static wide: Juveriya is seated on the desk edge with her skirt at "
                 "her waist and her kurti bunched at her ribs, her heels braced on the "
                 "open drawer and her hands behind her on the wood, driving her hips "
                 "up into the standing, shirtless Faiz, his arms braced on the desk "
                 "edge beside her.",
        "marks": "faded",
    },
    "c61": {
        "who": "both", "location": "study", "light": "evening_tube", "nsfw": True,
        "scene": "Static close-up: Juveriya's palm is flat on Faiz's bare sternum, "
                 "holding him off mid-motion, her other hand between her own legs, "
                 "his hips obedient and straining, her kurti bunched at her ribs, her "
                 "face giving a slow order he is watching.",
        "marks": "faded",
    },
    "c62": {
        "who": "juv", "location": "study", "light": "evening_tube", "nsfw": True,
        "scene": "Static tight on Juveriya's face with the light behind her, her "
                 "teeth bared and her eyes half-shut, her nails dug into a shoulder at "
                 "the bottom edge of the frame, the yellowed welt showing at her neck.",
        "marks": "faded",
    },
    "c63": {
        "who": "both", "location": "study", "light": "evening_tube", "nsfw": True,
        "scene": "Static close-up: Faiz's hips snap into Juveriya in three hard "
                 "strokes with no rhythm, his hand slamming the desk beside her hip, "
                 "her kurti bunched and her thigh wet, his head down with his teeth in "
                 "his own lip.",
        "marks": "faded",
    },
    "c64": {
        "who": "both", "location": "study", "light": "evening_tube", "nsfw": True,
        "scene": "Juveriya lies flat on her back across the desk, her skirt rucked to her "
                 "waist and her kurti bunched up, her long hair fanned loose, while Faiz "
                 "lies over her with his head resting on her stomach, bare-chested, his "
                 "shirt on the floor; the evening tube light steady overhead.",
        "marks": "faded",
    },
    "c65": {
        "who": "none", "location": "hallway", "light": "morning", "nsfw": False,
        "scene": "Morning daylight in the narrow passage: the mosaic floor freshly swept "
                 "clean, the front door shut, the flat still; a slice of kitchen light "
                 "and the corner of the counter show at the far end.",
        "marks": None,
    },
    "c66": {
        "who": "faiz", "location": "hallway", "light": "morning", "nsfw": False,
        "scene": "Faiz stands straightening his pressed kurta, a pair of shoes still in "
                 "one hand, caught mid-motion in the passage, morning daylight from the "
                 "front door falling on him.",
        "marks": None,
    },
    "c67": {
        "who": "none", "location": "front_door", "light": "morning", "nsfw": False,
        "scene": "The doorway in morning daylight: an older woman's hand closes on the "
                 "strap of a cloth bag slung over a shoulder, the dupatta wound tight at "
                 "the neck; her face stays out of frame, only the hand, the strap and the "
                 "shoulder visible.",
        "marks": None,
    },
    "c68": {
        "who": "none", "location": "front_door", "light": "morning", "nsfw": False,
        "scene": "The front door swinging closed, the latch dropping into place, morning "
                 "light cut off by the door, the security chain swinging slightly against "
                 "the wood.",
        "marks": None,
    },
    "c69": {
        "who": "faiz", "location": "hallway", "light": "morning", "nsfw": False,
        "scene": "Faiz pushes off the passage wall and walks down the corridor toward the "
                 "front door, morning light on his pressed kurta, his shoes left under "
                 "the rack behind him.",
        "marks": None,
    },
    "c70": {
        "who": "both", "location": "hallway", "light": "morning", "nsfw": False,
        "scene": "Faiz and Juveriya stand together in the passage in the morning light, "
                 "he in a pressed kurta, she in a daylight shalwar kameez with the "
                 "dupatta slipping off her shoulder, the study door shut behind them.",
        "marks": "faded",
    },
}


LOCATIONS = {
    "study": STUDY,
    "study_dark": STUDY_DARK,
    "bathroom": BATHROOM,
    "hallway": HALLWAY,
    "front_door": FRONT_DOOR,
    "kitchen": KITCHEN,
}


def wardrobe_for(table, shot_id: str) -> str:
    for (first, last), text in table:
        if first <= shot_id <= last:
            return text
    raise RuntimeError(f"{shot_id}: no wardrobe state defined")


def compose(shot_id: str, spec: dict) -> dict:
    who = spec["who"]
    parts = []
    if who == "both":
        parts.append(
            f"The man in image 1 is Faiz: {FAIZ_ID}; he is {wardrobe_for(FAIZ_WARDROBE, shot_id)}. "
            f"The woman in image 2 is Juveriya: {JUV_ID}; she is {wardrobe_for(JUV_WARDROBE, shot_id)}"
        )
    elif who == "faiz":
        parts.append(
            f"The man in the image is Faiz: {FAIZ_ID}; he is {wardrobe_for(FAIZ_WARDROBE, shot_id)}"
        )
    elif who == "juv":
        parts.append(
            f"The woman in the image is Juveriya: {JUV_ID}; she is {wardrobe_for(JUV_WARDROBE, shot_id)}"
        )
    marks = spec.get("marks")
    if who != "none":
        if marks == "fresh":
            parts[-1] += f", with {WELT_FRESH}"
        elif marks == "faded":
            parts[-1] += f", with {WELT_FADED}"
        parts[-1] += "."
    parts.append(spec["scene"])
    if who != "none":
        parts.append(f"Setting: {LOCATIONS[spec['location']]}.")
    parts.append(f"Lighting: {LIGHT[spec['light']]}.")
    parts.append(STYLE_CHARACTERS if who != "none" else STYLE_OBJECTS)

    refs = []
    if who == "both":
        mode = MODE_TWO
        refs = [FAIZ_PLATE, JUV_PLATE]
    elif who == "faiz":
        mode = MODE_EDIT
        refs = [FAIZ_PLATE]
    elif who == "juv":
        mode = MODE_EDIT
        refs = [JUV_PLATE]
    else:
        mode = MODE_T2I

    return {
        "prompt": " ".join(parts),
        "mode": mode,
        "refs": refs,
        "lora_adapter": LORA_NSFW if spec["nsfw"] else LORA_NONE,
        "lora_strength": 1.0,
        "steps": 28 if spec["nsfw"] else 30,
    }


def build(beat_ids: list[str]) -> dict:
    out = {}
    for shot_id in beat_ids:
        spec = BEATS.get(shot_id)
        if spec is None:
            raise RuntimeError(f"{shot_id}: no detailed prompt spec authored")
        entry = compose(shot_id, spec)
        length = len(entry["prompt"])
        if length > MAX_PROMPT_CHARS:
            raise RuntimeError(f"{shot_id}: prompt is {length} chars (cap {MAX_PROMPT_CHARS})")
        out[shot_id] = entry
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--beats", default=",".join(sorted(BEATS)),
                        help="comma list of beat ids (default: every authored beat)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    beat_ids = [item.strip() for item in args.beats.split(",") if item.strip()]

    spec = yaml.safe_load((ROOT / "docs" / "beats.yaml").read_text(encoding="utf-8"))
    known = {shot["id"] for shot in spec.get("shots", [])}
    unknown = [b for b in beat_ids if b not in known]
    if unknown:
        raise RuntimeError(f"Unknown beat ids: {', '.join(unknown)}")

    entries = build(beat_ids)
    for shot_id, entry in entries.items():
        print(f"{shot_id}: {len(entry['prompt']):4d} chars  mode={entry['mode']:<28} "
              f"lora={entry['lora_adapter']:<24} refs={','.join(entry['refs']) or '-'}")
    if args.dry_run:
        for shot_id, entry in entries.items():
            print(f"\n--- {shot_id} ---\n{entry['prompt']}")
        return 0

    existing = {}
    if OUT_PATH.is_file():
        existing = json.loads(OUT_PATH.read_text(encoding="utf-8"))
    existing.update(entries)
    OUT_PATH.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT_PATH.relative_to(ROOT)} ({len(entries)} beats updated, "
          f"{len(existing)} total)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
