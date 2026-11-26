#!/usr/bin/env python3
"""Build CLIP-sized prompts for the SDXL stills pass.

`build_prompts.py` writes `prompts/<id>.txt` for Wan 2.2 video, which encodes
with umt5-xxl and takes 512 tokens. The full six-line beat fits there and must
not be cut.

The SDXL stills pass is different: it encodes with CLIP, which truncates at 77
tokens. The tail of a Wan prompt is where the explicit content and the LoRA
triggers sit, so an over-long prompt silently loses both -- which is why the
stills came back generic and the identity scores were 0.000.

This writes a second file per beat, `prompts/<id>.still.txt`, sized for CLIP.
The video prompt is left untouched.

Usage:
    python scripts/build_still_prompts.py
    python scripts/build_still_prompts.py --check
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from build_prompts import (  # noqa: E402
    apply_lora_triggers,
    effective_loras,
    look_prefix,
)
from render import load_lora_registry  # noqa: E402

SPEC = Path("docs/beats.yaml")
PROMPTS = Path("prompts")
REGISTRY = Path("workflows") / "loras.json"

# CLIP truncates at 77 tokens. Leave a couple of tokens of headroom because the
# tokenizer is not perfectly predictable across punctuation.
CLIP_TOKEN_LIMIT = 77
CLIP_BUDGET = CLIP_TOKEN_LIMIT - 2

EXPLICIT_TERMS = (
    "pussy", "vulva", "labia", "clit", "cunt", "cock", "shaft", "cum",
    "breast", "nipple", "arse", "ass", "fuck", "thrust", "pov",
    "penetrat", "suck", "tongue", "wet", "naked", "bare",
)

# A clause naming a face is protected the same way an explicit clause is. The
# startframe seeds the video, so losing the only face reference in a beat costs
# the clip its identity anchor.
FACE_TERMS = (
    "face", "faces", "eyes", "eye", "mouth", "lips", "jaw", "chin",
    "cheek", "brow", "expression", "gaze",
)


def is_protected(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in EXPLICIT_TERMS) or any(
        re.search(rf"(?<!\w){re.escape(term)}(?!\w)", lowered) for term in FACE_TERMS
    )

_TOKENIZER = None


def token_count(text: str) -> int | None:
    """Real CLIP token count, or None when no tokenizer is available."""
    global _TOKENIZER
    if _TOKENIZER is None:
        try:
            from transformers import CLIPTokenizer

            _TOKENIZER = CLIPTokenizer.from_pretrained(
                "openai/clip-vit-large-patch14"
            )
        except Exception:  # noqa: BLE001 - offline or transformers missing
            _TOKENIZER = False
    if _TOKENIZER is False:
        return None
    return len(_TOKENIZER(text, truncation=False)["input_ids"])


def is_explicit(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in EXPLICIT_TERMS)


def shorten_marker(line: str) -> str:
    """'(At 2 seconds: ...)' -> '(2s: ...)'.

    The long form costs 4-5 CLIP tokens per line, a third of the whole window.
    """
    return re.sub(r"^\(At (\d+) seconds:\s*", r"(\1s: ", line).rstrip(")")


def parse_line(line: str) -> tuple[str, list[str]]:
    """Split a cam line into (prefix, clauses).

    The prefix is the time marker plus the camera verb, which is never dropped:
    it is what tells the stills model how the shot is framed.
    """
    line = shorten_marker(line)
    marker = ""
    match = re.match(r"^(\(\d+s:\s*)", line)
    if match:
        marker = match.group(1)
        line = line[len(marker):]
    if ":" in line:
        verb, rest = line.split(":", 1)
        prefix = marker + verb.strip() + ": "
    else:
        prefix, rest = marker, line
    return prefix, [c.strip() for c in rest.split(",") if c.strip()]


def render(parsed: list[tuple[str, list[str]]]) -> str:
    return "\n".join(prefix + ", ".join(clauses) for prefix, clauses in parsed)


# Every startframe gets this. The startframe seeds the video, so it must
# establish who is in the shot even when the beat itself is a hand or a wet
# join. It has to be a framing instruction, not a bare phrase: "both faces
# visible" lost to the concrete nouns in the cam lines.
FACE_ANCHOR_TEXT = "medium shot, faces in frame"

FRAMING_REWRITES = (
    (r"\bstatic ECU\b", "static medium"),
    (r"\bzoom in ECU\b", "zoom in"),
    (r"\bECU\b", "medium"),
    (r"\bstatic close-up\b", "static medium"),
    (r"\bclose-up\b", "medium"),
    (r"\bzoom in medium\b", "zoom in"),
)


# POV means the camera is one of the characters, so only the other one is in
# frame. That is correct for the video, but the startframe must show both --
# it seeds the clip and the cast check requires both faces. The still gets a
# third-person framing; prompts/<id>.txt keeps the POV line for Wan.
POV_REWRITES = (
    (r"\bPOV,?\s*", ""),
    (r"\bPOV\b", "over the shoulder"),
)


def drop_pov(line: str) -> str:
    """Remove POV from a still cam line.

    "static medium: POV, Faiz in the doorway" becomes "static medium: Faiz in
    the doorway" -- the same shot from a camera that can see both people.
    """
    for pattern, replacement in POV_REWRITES:
        line = re.sub(pattern, replacement, line, flags=re.IGNORECASE)
    return re.sub(r"\s{2,}", " ", line).strip()


def widen_framing(line: str) -> str:
    """Replace close framing with a medium shot.

    A cam line saying "static ECU" outranks a prefix saying "medium shot", and
    SDXL resolves the conflict by obeying the cam line -- which is why beats
    failed with "no face in frame" while their prompt said "faces in frame".
    The still is a seed frame; the video model does the ECU.
    """
    for pattern, replacement in FRAMING_REWRITES:
        line = re.sub(pattern, replacement, line, flags=re.IGNORECASE)
    return line


def fit_to_clip(
    cam_lines: list[str],
    look: str,
    prefixes: list[str],
    suffixes: list[str],
) -> str:
    """Build a CLIP-sized still prompt that reads as a photograph.

    The previous version compressed cam lines by dropping clauses until the
    token budget fit, which produced fragments like "static medium: POV" with
    no content. SDXL renders a fragment as a generic image, which is why 31 of
    70 stills came back blurry and off-story.

    A still is one frame, not six. So instead of compressing six cam lines, this
    picks the single most explicit line, keeps it whole, and spends the rest of
    the budget on the quality and identity tokens that make the frame read as a
    photograph rather than a render.
    """
    quality = "photorealistic, raw photo, natural skin texture, sharp focus"

    def assemble(cam: str) -> str:
        # The cam line goes first. SDXL weights early tokens more heavily, and
        # the explicit content is the part that must not be lost -- putting the
        # quality block first pushed the anatomy out of the budget entirely.
        parts = [cam]
        if prefixes:
            parts.append(", ".join(prefixes))
        parts.append(quality)
        if look:
            parts.append(look)
        if suffixes:
            parts.append(", ".join(suffixes))
        return ", ".join(p for p in parts if p).strip()

    def fits(text: str) -> bool:
        tokens = token_count(text)
        return tokens is not None and tokens <= CLIP_BUDGET

    # Rank the cam lines by how much explicit content they carry, then by
    # length. Picking the longest line alone lost the anatomy: the explicit
    # clause is often in a short line, so a length-first sort dropped it.
    def explicit_count(line: str) -> int:
        lowered = line.lower()
        return sum(1 for term in EXPLICIT_TERMS if term in lowered)

    def score(line: str) -> tuple[int, int, int]:
        return (explicit_count(line), 1 if is_protected(line) else 0, len(line.split()))

    ranked = sorted(cam_lines, key=score, reverse=True)

    # One whole line, best first.
    for line in ranked:
        candidate = assemble(line)
        if fits(candidate):
            return candidate

    # The best line is over budget. Trim its trailing clauses rather than
    # falling through to a weaker line: the top-ranked line is the one with the
    # anatomy, and dropping to a lower line loses exactly what matters.
    best = ranked[0] if ranked else ""
    clauses = [c.strip() for c in best.split(",") if c.strip()]
    while len(clauses) > 2:
        clauses.pop()
        candidate = assemble(", ".join(clauses))
        if fits(candidate):
            return candidate

    # Two lines, if a single trimmed line cannot fill the budget.
    if len(ranked) >= 2:
        candidate = assemble(" ".join(ranked[:2]))
        if fits(candidate):
            return candidate

    return assemble(", ".join(clauses))


def build_still_prompt(shot: dict, meta: dict, registry: dict) -> str:
    cam = [line for line in (shot.get("cam") or []) if isinstance(line, str)]
    # Widen close framing before anything else: an ECU line outranks the face
    # anchor and produces a faceless still.
    cam = [drop_pov(widen_framing(line)) for line in cam]
    look = look_prefix(shot, meta)
    effective = effective_loras(
        shot.get("loras") or [], registry, bool(shot.get("no_default_loras"))
    )
    prefixes, suffixes = [], []
    for key in effective:
        entry = registry.get(key) if isinstance(registry, dict) else None
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

    # Always anchor the face. The prefix is never trimmed, so this survives the
    # budget; a cam line carrying it would be dropped as non-explicit.
    look = f"{look}, {FACE_ANCHOR_TEXT}" if look else FACE_ANCHOR_TEXT

    return fit_to_clip(cam, look, prefixes, suffixes)


def main() -> int:
    ap = argparse.ArgumentParser(description="Build CLIP-sized stills prompts")
    ap.add_argument("--spec", default=str(SPEC))
    ap.add_argument("--prompts", default=str(PROMPTS))
    ap.add_argument("--registry", default=str(REGISTRY))
    ap.add_argument("--check", action="store_true", help="report only")
    args = ap.parse_args()

    spec = yaml.safe_load(Path(args.spec).read_text(encoding="utf-8"))
    meta = spec.get("meta") or {}
    registry = load_lora_registry(Path(args.registry))
    out = Path(args.prompts)

    written = 0
    over: list[str] = []
    for shot in spec.get("shots") or []:
        sid = shot.get("id")
        if not isinstance(sid, str):
            continue
        text = build_still_prompt(shot, meta, registry)
        tokens = token_count(text)
        if tokens is not None and tokens > CLIP_TOKEN_LIMIT:
            over.append(f"{sid}: {tokens} tokens")
        if not args.check:
            (out / f"{sid}.still.txt").write_text(text + "\n", encoding="utf-8")
        written += 1

    print(f"{'checked' if args.check else 'wrote'} {written} still prompts")
    if over:
        print(f"{len(over)} over {CLIP_TOKEN_LIMIT} tokens:", file=sys.stderr)
        for item in over:
            print(f"  {item}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
