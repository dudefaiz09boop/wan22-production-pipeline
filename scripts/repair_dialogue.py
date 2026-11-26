#!/usr/bin/env python3
"""Restore the spoken dialogue that a bad encoding pass destroyed.

`say` in docs/beats.yaml was written through a cp1252 round-trip, so every
Devanagari line became a run of '?'. The three beats that also carry
`say_script` survived, which is how the damage was spotted.

This rewrites `say` for the affected beats in Devanagari Hindi, matching each
beat's action. It is idempotent: a beat whose `say` is already valid Devanagari
is left alone.

Usage:
    python scripts/repair_dialogue.py --check
    python scripts/repair_dialogue.py --write
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

SPEC = Path("docs/beats.yaml")

# Beat id -> Devanagari line. Written to match the cam action of that beat.
DIALOGUE: dict[str, str] = {
    "c01": "तेरी चूत मेरी है, जुवेरिया। आज रात तू मेरी ही रहेगी।",
    "c02": "देख, कैसे खड़ा है तेरे लिए। बोल, कितनी गीली है तू?",
    "c04": "ये साली स्कर्ट फाड़ दूँ? तेरी चूत नंगी कर दूँ?",
    "c05": "इतनी गीली? सिर्फ़ मेरी उँगली से?",
    "c06": "देख, कैसे फूल रही है। मेरी बहन की चूत।",
    "c08": "ले, ले। पूरी ले। अम्मी सुन ले तो सुन ले।",
    "c10": "खोल, और खोल। मुझे सब देखना है।",
    "c12": "छोड़ मुझे— नहीं, मत छोड़। और कर।",
    "c14": "तेरी चूत की आवाज़ सुन। पूरा घर सुन रहा है।",
    "c15": "दाँत लगाऊँ? निशान छोड़ दूँ ताकि तू भूल न जाए?",
    "c16": "चिल्ला। ज़ोर से चिल्ला, बहन।",
    "c18": "गिन, जुवेरिया। ज़ोर से गिन। मैं सुनना चाहता हूँ।",
    "c19": "देख, कितनी आसानी से अंदर जाती हैं। तैयार है तू।",
    "c21": "अंधेरे में भी तू मेरी चूत पहचान लेगी।",
    "c23": "बोल, किसकी बहन है तू? किसकी चूत है?",
    "c27": "अज़ान हो रही है। फिर भी मेरा लंड अंदर है।",
    "c28": "देख, कितना निकला। सब तेरा है।",
    "c29": "तू मेरी है। सिर्फ़ मेरी।",
    "c35": "पढ़ रही हूँ। किताबें।",
    "c36": "वह अध्याय में मेरी मदद कर रहा है।",
    "c58": "आज मैं करूँगी। हाथ नीचे रख।",
    "c61": "नहीं। मेरी मर्ज़ी। तू बस देख।",
    "c62": "अंदर। अंदर ही रहने दे।",
}

SHOT_RE = re.compile(r"^\s*-\s+id:\s*(\S+)\s*$")
SAY_RE = re.compile(r'^(\s*)say:\s*"(.*)"\s*$')

# Romanized Hindi (and one English line) that survived the encoding pass but
# would be spoken by an English voice. Converted to Devanagari so Chatterbox
# reads them as Hindi.
ROMANIZED: dict[str, str] = {
    "c11": "आह — हाँ — हाँ —",
    "c13": "आह — और — और —",
    "c17": "आह — मत रुक — मत रुक —",
    "c20": "हाँ — हाँ — और —",
    "c24": "हाँ — वही है — मेरी बहन — बस —",
    "c26": "बस — बस — आह —",
    "c56": "अम्मी सो गई है। बैठो। और बाहर निकालो।",
    "c60": "आह — देखो — देखो —",
    "c63": "आह — आह — आह —",
}

# Beats whose `dialogue:` list lost its text and has no `text_script` to fall
# back on. Keyed by beat id, then by speaker, in list order.
DIALOGUE_LIST: dict[str, list[str]] = {
    "c25": [
        "करने दे, बहन।",
        "हाँ करो, धीरे, धीरे, मत रुक।",
    ],
    "c30": [
        "मेरी स्कर्ट... देख क्या हाल किया।",
        "उठ, जल्दी उठ।",
        "पकड़ मेरा हाथ।",
    ],
    "c38": [
        "पानी ला, जल्दी।",
        "ये सब कैसे साफ़ होगा?",
    ],
    "c39": [
        "ये दाग़ रह जाएगा।",
        "कोई नहीं देखेगा।",
        "तू भी नहीं देखेगी।",
    ],
    "c42": [
        "तू अच्छी लग रही है।",
        "तूने मेरे साथ ये किया, और अब देख रहा है।",
        "और करूँगा।",
    ],
    "c64": [
        "अम्मी आ जाएगी।",
        "आने दे।",
    ],
    "c69": [
        "चल।",
        "चल।",
    ],
    "c70": [
        "कल फिर वही करेंगे?",
        "हर रात।",
    ],
}

DIALOGUE_ITEM_RE = re.compile(r"^(\s*)-\s+speaker:\s*(\S+)\s*$")
TEXT_RE = re.compile(r'^(\s*)text:\s*"(.*)"\s*$')
SCRIPT_RE = re.compile(r'^(\s*)text_script:\s*"(.*)"\s*$')


def has_devanagari(text: str) -> bool:
    return any("\u0900" <= ch <= "\u097f" for ch in text)


def is_corrupted(text: str) -> bool:
    return "???" in text


def _collect_scripts(text: str) -> dict[tuple[str, int], str]:
    """Map (beat_id, dialogue item index) -> text_script.

    A pre-pass, because `text_script` can appear after `text` in the same item.
    """
    scripts: dict[tuple[str, int], str] = {}
    current: str | None = None
    item_index = 0
    for line in text.splitlines():
        shot = SHOT_RE.match(line)
        if shot:
            current = shot.group(1)
            item_index = 0
            continue
        if DIALOGUE_ITEM_RE.match(line) and current:
            item_index += 1
            continue
        script = SCRIPT_RE.match(line)
        if script and current:
            scripts[(current, item_index)] = script.group(2)
    return scripts


def repair(text: str) -> tuple[str, list[str], list[str]]:
    """Return (new_text, repaired_ids, still_broken_ids)."""
    scripts = _collect_scripts(text)
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    current: str | None = None
    repaired: list[str] = []
    broken: list[str] = []
    # Index into DIALOGUE_LIST[current] for the dialogue: list being walked.
    item_index = 0

    for line in lines:
        shot = SHOT_RE.match(line)
        if shot:
            current = shot.group(1)
            item_index = 0
            out.append(line)
            continue

        item = DIALOGUE_ITEM_RE.match(line)
        if item and current:
            item_index += 1
            out.append(line)
            continue

        say = SAY_RE.match(line)
        if say and current:
            value = say.group(2)
            if is_corrupted(value):
                replacement = DIALOGUE.get(current)
                if replacement:
                    indent = say.group(1)
                    out.append(f'{indent}say: "{replacement}"\n')
                    repaired.append(current)
                    continue
                broken.append(current)
            elif value.strip() and not has_devanagari(value):
                # Romanized Hindi or English: an English voice would read it.
                replacement = ROMANIZED.get(current)
                if replacement:
                    indent = say.group(1)
                    out.append(f'{indent}say: "{replacement}"\n')
                    repaired.append(current)
                    continue
            out.append(line)
            continue

        text_match = TEXT_RE.match(line)
        if text_match and current:
            value = text_match.group(2)
            if is_corrupted(value):
                replacements = DIALOGUE_LIST.get(current)
                if replacements and item_index - 1 < len(replacements):
                    indent = text_match.group(1)
                    out.append(f'{indent}text: "{replacements[item_index - 1]}"\n')
                    repaired.append(current)
                    continue
                # No authored replacement, but the entry may still carry a
                # valid text_script (the Urdu Ammi lines and a few Hindi ones).
                script = scripts.get((current, item_index))
                if script:
                    indent = text_match.group(1)
                    out.append(f'{indent}text: "{script}"\n')
                    repaired.append(current)
                    continue
                broken.append(current)
            out.append(line)
            continue

        out.append(line)

    return "".join(out), repaired, broken


def main() -> int:
    ap = argparse.ArgumentParser(description="Restore corrupted Devanagari dialogue")
    ap.add_argument("--write", action="store_true", help="apply the repair")
    ap.add_argument("--check", action="store_true", help="report only")
    args = ap.parse_args()

    if not SPEC.is_file():
        print(f"missing {SPEC}", file=sys.stderr)
        return 1

    original = SPEC.read_text(encoding="utf-8")
    updated, repaired, broken = repair(original)

    print(f"corrupted beats repaired: {len(repaired)}")
    if repaired:
        print("  " + ", ".join(repaired))
    if broken:
        print(f"no dialogue authored for: {', '.join(broken)}", file=sys.stderr)

    if args.check or not args.write:
        return 0 if not broken else 1

    SPEC.write_text(updated, encoding="utf-8")
    print(f"wrote {SPEC}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
