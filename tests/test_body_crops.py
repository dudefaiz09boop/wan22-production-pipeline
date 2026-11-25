"""Body-crop behaviour for whole-body LoRA training.

Face crops alone teach a face, not a body. These cover the two rules that keep
body crops useful: skip them when the source is a close-up, and never emit the
same full-frame fallback for both characters.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_training_data import _body_crop, _emit_sample


class FakeFace:
    def __init__(self, bbox) -> None:
        self.bbox = np.asarray(bbox, dtype=np.float32)


class BodyCropTests(unittest.TestCase):
    def test_body_crop_extends_below_the_face(self) -> None:
        img = np.zeros((1000, 800, 3), dtype=np.uint8)
        face = FakeFace([350, 100, 450, 220])

        crop, bbox = _body_crop(img, face, min_face_size=64)

        self.assertGreater(bbox[3], 220)
        self.assertGreater(bbox[3] - bbox[1], 120)
        self.assertEqual(crop.shape[0], bbox[3] - bbox[1])

    def test_body_crop_is_wider_than_the_face(self) -> None:
        img = np.zeros((1000, 800, 3), dtype=np.uint8)
        face = FakeFace([350, 100, 450, 220])

        _, bbox = _body_crop(img, face, min_face_size=64)

        self.assertGreater(bbox[2] - bbox[0], 100)

    def test_tiny_face_is_rejected(self) -> None:
        img = np.zeros((1000, 800, 3), dtype=np.uint8)
        face = FakeFace([10, 10, 40, 40])

        self.assertIsNone(_body_crop(img, face, min_face_size=64))


class EmitSampleTests(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_close_up_source_skips_the_body_crop(self) -> None:
        img = np.zeros((600, 600, 3), dtype=np.uint8)
        # Face is 70% of the frame width: a close-up, no body to show.
        face = FakeFace([90, 100, 510, 400])

        self.assertIsNone(
            _emit_sample(img, face, self.out, "faiz", "x", 0, "body", 64, 0)
        )

    def test_wide_source_emits_a_body_crop(self) -> None:
        img = np.zeros((1200, 900, 3), dtype=np.uint8)
        face = FakeFace([400, 150, 500, 280])

        sample = _emit_sample(img, face, self.out, "faiz", "x", 0, "body", 64, 0)

        self.assertIsNotNone(sample)
        self.assertEqual(sample["kind"], "body")
        self.assertTrue((self.out / sample["file"]).is_file())

    def test_body_caption_differs_from_face_caption(self) -> None:
        img = np.zeros((1200, 900, 3), dtype=np.uint8)
        face = FakeFace([400, 150, 500, 280])

        body = _emit_sample(img, face, self.out, "faiz", "x", 0, "body", 64, 0)
        face_sample = _emit_sample(img, face, self.out, "faiz", "x", 0, "face", 64, 0)

        body_text = (self.out / body["file"]).with_suffix(".txt").read_text(encoding="utf-8")
        face_text = (self.out / face_sample["file"]).with_suffix(".txt").read_text(encoding="utf-8")
        self.assertIn("full body", body_text)
        self.assertIn("detailed face", face_text)
        self.assertNotEqual(body_text, face_text)

    def test_full_frame_fallback_is_skipped_for_non_primary_face(self) -> None:
        # Face small enough that the body crop covers nearly the whole frame.
        img = np.zeros((400, 400, 3), dtype=np.uint8)
        face = FakeFace([150, 150, 250, 250])

        primary = _emit_sample(img, face, self.out, "faiz", "x", 0, "body", 64, 0, primary=True)
        secondary = _emit_sample(img, face, self.out, "juv", "x", 1, "body", 64, 0, primary=False)

        self.assertIsNotNone(primary)
        self.assertIsNone(secondary)


if __name__ == "__main__":
    unittest.main()
