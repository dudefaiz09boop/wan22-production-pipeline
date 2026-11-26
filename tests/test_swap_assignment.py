"""The face swap must not confuse the two characters.

v12 swapped them: swap_faces_in_frame assigned faces by rank (largest face to
the first character), and in most beats Juveriya is nearer the camera, so her
larger face got Faiz's face pasted on.

These tests pin the assignment rule: similarity decides, not size.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from face_swap_video import swap_faces_in_frame  # noqa: E402


class FakeFace:
    def __init__(self, bbox, embedding) -> None:
        self.bbox = np.asarray(bbox, dtype=np.float32)
        vector = np.asarray(embedding, dtype=np.float32)
        self.normed_embedding = vector / np.linalg.norm(vector)


class FakeRef:
    def __init__(self, embedding) -> None:
        vector = np.asarray(embedding, dtype=np.float32)
        self.normed_embedding = vector / np.linalg.norm(vector)


class FakeApp:
    """Returns a fixed set of faces, ignoring the frame."""

    def __init__(self, faces) -> None:
        self._faces = faces

    def get(self, _frame):
        return list(self._faces)


class FakeSwapper:
    """Records which reference was pasted onto which face."""

    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    def get(self, frame, face, ref, paste_back=True):
        self.calls.append((id(face), id(ref)))
        return frame


FAIZ_REF = FakeRef([1.0, 0.0, 0.0])
JUV_REF = FakeRef([0.0, 1.0, 0.0])
REFS = {"faiz": FAIZ_REF, "juv": JUV_REF}


class SwapAssignmentTests(unittest.TestCase):
    def test_larger_face_is_not_assumed_to_be_the_first_character(self) -> None:
        """Juveriya nearer the camera must still get Juveriya's face."""
        # Large face matches juv; small face matches faiz.
        large_juv = FakeFace([0, 0, 200, 200], [0.1, 0.95, 0.0])
        small_faiz = FakeFace([300, 300, 380, 380], [0.95, 0.1, 0.0])

        app = FakeApp([large_juv, small_faiz])
        swapper = FakeSwapper()

        swap_faces_in_frame(app, swapper, np.zeros((400, 400, 3), np.uint8), REFS, ["faiz", "juv"])

        # Two swaps, and the large face must have received the juv reference.
        self.assertEqual(len(swapper.calls), 2)
        pasted = {face_id: ref_id for face_id, ref_id in swapper.calls}
        self.assertEqual(pasted[id(large_juv)], id(JUV_REF))
        self.assertEqual(pasted[id(small_faiz)], id(FAIZ_REF))

    def test_each_character_is_used_at_most_once(self) -> None:
        """Two faces must not both claim the same character."""
        a = FakeFace([0, 0, 100, 100], [0.9, 0.1, 0.0])
        b = FakeFace([200, 200, 300, 300], [0.8, 0.2, 0.0])

        app = FakeApp([a, b])
        swapper = FakeSwapper()

        swap_faces_in_frame(app, swapper, np.zeros((400, 400, 3), np.uint8), REFS, ["faiz", "juv"])

        refs_used = [ref_id for _, ref_id in swapper.calls]
        self.assertEqual(len(refs_used), len(set(refs_used)), "a character was swapped twice")

    def test_no_faces_returns_the_frame_untouched(self) -> None:
        app = FakeApp([])
        swapper = FakeSwapper()
        frame = np.zeros((100, 100, 3), np.uint8)

        result, scores = swap_faces_in_frame(app, swapper, frame, REFS, ["faiz", "juv"])

        self.assertIs(result, frame)
        self.assertEqual(scores, {})
        self.assertEqual(swapper.calls, [])


if __name__ == "__main__":
    unittest.main()
