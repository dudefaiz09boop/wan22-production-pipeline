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
        floors = {"breasts": 9, "pussy": 9, "cum": 4}
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


class PromptBudgetTests(unittest.TestCase):
    """Stills use the full video prompt now.

    SDXL has two text encoders and generate_still splits the prompt across them,
    so the 77-token CLIP limit that forced a trimmed stills variant no longer
    applies. These tests check the prompt the stills actually receive.
    """

    def test_video_prompts_keep_all_six_cam_lines(self) -> None:
        import glob

        short: list[str] = []
        for path in sorted(glob.glob(str(ROOT / "prompts" / "c*.txt"))):
            if path.endswith(".still.txt"):
                continue
            text = Path(path).read_text(encoding="utf-8")
            if text.count("(At ") != 6:
                short.append(f"{Path(path).stem}: {text.count('(At ')} lines")
        self.assertEqual(short, [], f"prompts missing cam lines: {short}")

    def test_explicit_terms_survive_in_the_prompts(self) -> None:
        import glob

        terms = {
            "breasts": ("breast", "nipple"),
            "pussy": ("pussy", "vulva", "labia", "clit", "cunt"),
            "cum": ("cum", "spurt"),
        }
        floors = {"breasts": 9, "pussy": 9, "cum": 4}
        counts = {key: 0 for key in terms}
        for path in sorted(glob.glob(str(ROOT / "prompts" / "c*.txt"))):
            if path.endswith(".still.txt"):
                continue
            text = Path(path).read_text(encoding="utf-8").lower()
            for key, words in terms.items():
                if any(word in text for word in words):
                    counts[key] += 1
        for key, floor in floors.items():
            self.assertGreaterEqual(
                counts[key], floor, f"{key} survives in only {counts[key]} prompts"
            )

    def test_lora_triggers_are_not_truncated_away(self) -> None:
        import glob

        missing: list[str] = []
        for path in sorted(glob.glob(str(ROOT / "prompts" / "c*.txt"))):
            if path.endswith(".still.txt"):
                continue
            text = Path(path).read_text(encoding="utf-8")
            if "faiz" not in text or "juv" not in text:
                missing.append(Path(path).stem)
        self.assertEqual(missing, [], f"character triggers missing: {missing}")


class CastAndActTests(unittest.TestCase):
    """The story needs both characters and the act, not just a good score."""

    def test_every_beat_declares_both_characters(self) -> None:
        """A startframe that drops a character seeds a clip with the wrong cast."""
        import sys as _sys

        _sys.path.insert(0, str(ROOT / "scripts"))
        from generate_stills import BEAT_CHARACTERS

        wrong = {
            beat: cast
            for beat, cast in BEAT_CHARACTERS.items()
            if set(cast) != {"faiz", "juv"}
        }
        self.assertEqual(wrong, {}, f"beats with the wrong cast: {wrong}")

    def test_act_coverage_meets_the_floor(self) -> None:
        """The film is explicit penetrative sex; the cam lines must say so."""
        data = _load()
        acts = {
            "penetration": (
                "penetrat", "inside her", "buried", "hilt", "driving in",
                "driving hard", "thrust", "pounding", "sliding in",
            ),
            "oral": (
                "mouth on", "sucking", "tongue", "lick",
                "takes him in her mouth", "down to his cock",
            ),
            "cum": ("cum", "creampie", "spurt", "spilling"),
        }
        floors = {"penetration": 15, "oral": 8, "cum": 4}
        counts = {key: 0 for key in acts}
        for shot in data["shots"]:
            text = " ".join(shot.get("cam") or []).lower()
            for key, words in acts.items():
                if any(word in text for word in words):
                    counts[key] += 1
        for key, floor in floors.items():
            self.assertGreaterEqual(
                counts[key], floor, f"{key} appears in only {counts[key]} beats"
            )

    def test_story_guards_hold(self) -> None:
        """Adults, siblings, Ammi off-screen: the canon the render depends on."""
        data = _load()
        story = data["story"]
        self.assertTrue(story["boundaries"]["characters_are_adults"])
        self.assertEqual(story["relations"]["faiz_juveriya"], "siblings")
        self.assertFalse(story["relations"]["on_screen"])

    def test_captions_carry_the_identity_tokens(self) -> None:
        """The LoRAs were trained on 'indian muslim'; the captions must say it."""
        import glob

        missing: list[str] = []
        for path in sorted(glob.glob(str(ROOT / "training_data_v2" / "*" / "*.txt"))):
            text = Path(path).read_text(encoding="utf-8").lower()
            if "indian muslim" not in text:
                missing.append(Path(path).name)
        self.assertEqual(missing[:5], [], f"captions without identity tokens: {missing[:5]}")


if __name__ == "__main__":
    unittest.main()
