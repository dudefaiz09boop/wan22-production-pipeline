"""Guards for the dialogue and cam-line repair passes.

Two things went wrong and are easy to reintroduce:

1. `say` and `dialogue[].text` were written through a cp1252 round-trip, so
   every Devanagari line became '?'. The narration then came out English.
2. The cam lines implied the action instead of naming it, so Wan 2.2 rendered
   generic shots: POV in 1 of 70 beats, pussy in 6.

These tests run the repair functions against the real spec and assert the
invariants hold, so a future edit that reintroduces either problem fails here
rather than after a paid render.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from scripts.amplify_beats import REPLACEMENTS, apply as apply_amplify
from scripts.repair_dialogue import repair

SPEC = ROOT / "docs" / "beats.yaml"


def _load() -> dict:
    return yaml.safe_load(SPEC.read_text(encoding="utf-8"))


class DialogueRepairTests(unittest.TestCase):
    def test_no_corrupted_dialogue_remains(self) -> None:
        data = _load()
        bad: list[str] = []
        for shot in data["shots"]:
            if "???" in (shot.get("say") or ""):
                bad.append(f"{shot['id']}.say")
            for turn in shot.get("dialogue") or []:
                if "???" in (turn.get("text") or ""):
                    bad.append(f"{shot['id']}.dialogue.{turn.get('speaker')}")
        self.assertEqual(bad, [], f"corrupted dialogue: {bad}")

    def test_every_spoken_line_is_devanagari_or_urdu(self) -> None:
        """No Latin-script dialogue: an English voice would read it."""
        data = _load()
        latin: list[str] = []
        for shot in data["shots"]:
            texts = [shot.get("say") or ""]
            texts += [t.get("text") or "" for t in (shot.get("dialogue") or [])]
            for text in texts:
                if not text.strip():
                    continue
                has_deva = any("\u0900" <= c <= "\u097f" for c in text)
                has_arabic = any("\u0600" <= c <= "\u06ff" for c in text)
                if not (has_deva or has_arabic):
                    latin.append(f"{shot['id']}: {text[:40]}")
        self.assertEqual(latin, [], f"Latin-script dialogue: {latin}")

    def test_repair_is_idempotent(self) -> None:
        text = SPEC.read_text(encoding="utf-8")
        once, _, _ = repair(text)
        twice, applied, _ = repair(once)
        self.assertEqual(once, twice)
        self.assertEqual(applied, [])


class AmplifyTests(unittest.TestCase):
    def test_amplify_is_idempotent(self) -> None:
        text = SPEC.read_text(encoding="utf-8")
        once, _, _ = apply_amplify(text)
        twice, applied, _ = apply_amplify(once)
        self.assertEqual(once, twice)
        self.assertEqual(applied, [])

    def test_explicit_coverage_meets_the_floor(self) -> None:
        """The render only shows what the prompt names."""
        data = _load()
        terms = {
            "pov": ("pov",),
            "breasts": ("breast", "nipple"),
            "pussy": ("pussy", "vulva", "labia", "clit"),
            "cum": ("cum", "spurt"),
        }
        floors = {"pov": 4, "breasts": 10, "pussy": 10, "cum": 4}
        counts = {key: 0 for key in terms}
        for shot in data["shots"]:
            text = " ".join(shot.get("cam") or []).lower()
            for key, words in terms.items():
                if any(word in text for word in words):
                    counts[key] += 1
        for key, floor in floors.items():
            self.assertGreaterEqual(
                counts[key], floor, f"{key} appears in only {counts[key]} beats"
            )

    def test_replacements_stay_within_the_cam_length_limit(self) -> None:
        for (beat, index), line in REPLACEMENTS.items():
            self.assertLessEqual(
                len(line), 220, f"{beat}[{index}] is {len(line)} chars"
            )


if __name__ == "__main__":
    unittest.main()
