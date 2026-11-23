from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import build_identity_cards
import freeze_bible


class BibleTests(unittest.TestCase):
    def test_build_is_complete_and_deterministic(self):
        self.assertEqual(build_identity_cards.build(check=False), [])
        snapshots = {
            "faiz": (ROOT / "bible" / "faiz.json").read_text(encoding="utf-8"),
            "juveriya": (ROOT / "bible" / "juveriya.json").read_text(encoding="utf-8"),
        }
        self.assertEqual(build_identity_cards.build(check=True), [])
        self.assertEqual(snapshots["faiz"], (ROOT / "bible" / "faiz.json").read_text(encoding="utf-8"))
        self.assertEqual(snapshots["juveriya"], (ROOT / "bible" / "juveriya.json").read_text(encoding="utf-8"))

    def test_character_cards_require_sourced_claims_and_forbidden_lists(self):
        for name in ("faiz", "juveriya"):
            card = json.loads((ROOT / "bible" / f"{name}.json").read_text(encoding="utf-8"))
            self.assertTrue(card["adult"])
            self.assertTrue(card["forbidden"])
            self.assertTrue(card["plate_refs"])
            for section in ("face", "body", "voice"):
                for key, entry in card[section].items():
                    if entry is None:
                        continue
                    self.assertIn("value", entry, f"{name}.{section}.{key}")
                    self.assertTrue(entry["sources"], f"{name}.{section}.{key}")
                    self.assertIn(entry["status"], build_identity_cards.STATUSES)

    def test_locations_have_three_lighting_states_and_sourced_geography(self):
        locations = sorted((ROOT / "bible" / "locations").glob("*.json"))
        self.assertEqual(len(locations), 5)
        for path in locations:
            card = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(len(card["lighting_states"]), 3)
            self.assertTrue(card["geography"]["sources"])
            self.assertTrue(card["camera_axis"]["sources"])
            for prop in card["props"]:
                self.assertTrue(prop["sources"])

    def test_freeze_manifest_is_current_and_checksums_match(self):
        self.assertEqual(freeze_bible.freeze(check=False), [])
        self.assertEqual(freeze_bible.freeze(check=True), [])


if __name__ == "__main__":
    unittest.main()
