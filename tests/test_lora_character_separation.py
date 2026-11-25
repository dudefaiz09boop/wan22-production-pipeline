"""Guards against the two ways the character LoRAs got mixed up.

1. prepare_training_data wrote every detected face into BOTH the faiz and the
   juv dataset, so 58 of 65 "faiz" crops were byte-identical to "juv" crops.
2. generate_stills preferred the combined adapter and ignored the
   per-character ones, so a single-person beat still got both faces blended in.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_training_data import assign_character


def _unit(*values: float) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


class AssignCharacterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.centroids = {
            "faiz": _unit(1.0, 0.0, 0.0),
            "juv": _unit(0.0, 1.0, 0.0),
        }

    def test_face_matching_one_character_is_assigned_only_to_that_character(self) -> None:
        character, score = assign_character(_unit(0.95, 0.05, 0.0), self.centroids)
        self.assertEqual(character, "faiz")
        self.assertGreater(score, 0.9)

    def test_face_matching_neither_character_is_dropped(self) -> None:
        # Orthogonal to both centroids: a third person, not a coin flip.
        character, score = assign_character(_unit(0.0, 0.0, 1.0), self.centroids)
        self.assertIsNone(character)
        self.assertLess(score, 0.35)

    def test_threshold_is_respected(self) -> None:
        embedding = _unit(0.5, 0.5, 0.0)
        self.assertIsNone(assign_character(embedding, self.centroids, threshold=0.9)[0])
        self.assertEqual(
            assign_character(embedding, self.centroids, threshold=0.1)[0], "faiz"
        )


class ActivateAdaptersTests(unittest.TestCase):
    def _pipe(self, loaded: list[str]):
        class FakePipe:
            def __init__(self) -> None:
                self._loaded_lora_names = loaded
                self.active: list[str] = []

            def set_adapters(self, names, adapter_weights=None) -> None:
                self.active = list(names)

        return FakePipe()

    def test_single_character_beat_does_not_activate_the_combined_adapter(self) -> None:
        from scripts.lora_adapters import select_adapters

        self.assertEqual(select_adapters(["faiz", "juv", "both"], ["faiz"]), ["faiz"])

    def test_two_character_beat_activates_both_character_adapters(self) -> None:
        from scripts.lora_adapters import select_adapters

        self.assertEqual(
            select_adapters(["faiz", "juv", "both"], ["faiz", "juv"]), ["faiz", "juv"]
        )

    def test_combined_adapter_is_only_a_fallback(self) -> None:
        from scripts.lora_adapters import select_adapters

        self.assertEqual(select_adapters(["both"], ["faiz"]), ["both"])
        self.assertEqual(select_adapters([], ["faiz"]), [])


if __name__ == "__main__":
    unittest.main()
