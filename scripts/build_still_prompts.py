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


def fit_to_clip(
    cam_lines: list[str],
    look: str,
    prefixes: list[str],
    suffixes: list[str],
) -> str:
    """Compress the cam lines until the assembled prompt fits CLIP.

    Order of sacrifice, least damaging first:

    1. shorten the time markers;
    2. drop clauses that name nothing explicit, latest first;
    3. drop whole non-explicit lines, keeping the opener and the payoff;
    4. only then drop explicit clauses.

    The look prefix and the LoRA triggers are never touched: without them the
    still is not the right character in the right light.
    """

    def assemble(cam: list[str]) -> str:
        text = apply_lora_triggers(cam, [], {})
        if prefixes:
            text = ", ".join(prefixes) + ", " + text
        if suffixes:
            text = text + ", " + ", ".join(suffixes)
        if look:
            text = look + ", " + text
        return text.strip()

    def fits(cam: list[str]) -> bool:
        tokens = token_count(assemble(cam))
        return tokens is not None and tokens <= CLIP_BUDGET

    parsed = [parse_line(line) for line in cam_lines if line.strip()]

    if fits([render(parsed)]):
        return assemble([render(parsed)])

    # 2. Drop non-explicit clauses, latest first.
    while not fits([render(parsed)]):
        candidates = [
            (-line_index, clause_index)
            for line_index, (_, clauses) in enumerate(parsed)
            for clause_index, clause in enumerate(clauses)
            if len(clauses) > 1 and not is_explicit(clause)
        ]
        if not candidates:
            break
        neg_line, clause_index = max(candidates)
        line_index = -neg_line
        prefix, clauses = parsed[line_index]
        parsed[line_index] = (
            prefix,
            clauses[:clause_index] + clauses[clause_index + 1:],
        )

    # 3. Drop whole non-explicit lines, keeping the first and the last.
    while not fits([render(parsed)]) and len(parsed) > 2:
        droppable = [
            (-index,)
            for index, (_, clauses) in enumerate(parsed)
            if 0 < index < len(parsed) - 1 and not is_explicit(" ".join(clauses))
        ]
        if not droppable:
            break
        (neg_index,) = max(droppable)
        parsed.pop(-neg_index)

    # 4. Last resort: drop explicit clauses too, latest first.
    while not fits([render(parsed)]):
        candidates = [
            (-line_index, clause_index)
            for line_index, (_, clauses) in enumerate(parsed)
            for clause_index in range(len(clauses))
            if len(clauses) > 1
        ]
        if candidates:
            neg_line, clause_index = max(candidates)
            line_index = -neg_line
            prefix, clauses = parsed[line_index]
            parsed[line_index] = (
                prefix,
                clauses[:clause_index] + clauses[clause_index + 1:],
            )
            continue

        # Every line is down to one clause and it still does not fit. Drop the
        # least explicit whole line, keeping the first and the last.
        droppable = [
            (0 if is_explicit(" ".join(clauses)) else 1, -index)
            for index, (_, clauses) in enumerate(parsed)
            if 0 < index < len(parsed) - 1
        ]
        if not droppable:
            break
        _, neg_index = max(droppable)
        parsed.pop(-neg_index)

    return assemble([render(parsed)])


# Every startframe gets this. The startframe seeds the video, so it must
# establish who is in the shot even when the beat itself is a hand or a wet
# join. Keyword detection was tried and abandoned: "her hair swinging" and
# "his forearm braced by her head" both read as face references but put no face
# in frame, so the anchor was skipped exactly where it was needed.
FACE_ANCHOR_TEXT = "both faces visible"


def build_still_prompt(shot: dict, meta: dict, registry: dict) -> str:
    cam = [line for line in (shot.get("cam") or []) if isinstance(line, str)]
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
