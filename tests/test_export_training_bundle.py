"""Guards for the hosted-trainer export.

A hosted run is slow and a mixed dataset is only visible in the output, so the
export refuses to package one.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.export_training_bundle import validate, write_bundle


def _write(directory: Path, name: str, colour: str, caption: str | None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (64, 64), colour).save(directory / f"{name}.jpg")
    if caption is not None:
        (directory / f"{name}.txt").write_text(caption, encoding="utf-8")


class ValidateTests(unittest.TestCase):
    def test_clean_split_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(15):
                _write(root / "faiz", f"faiz_{i}", "red", f"faiz, man {i}")
                _write(root / "juv", f"juv_{i}", "blue", f"juv, woman {i}")

            _, problems = validate(root)

            self.assertEqual(problems, [])

    def test_shared_image_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(15):
                _write(root / "faiz", f"faiz_{i}", "red", f"faiz, man {i}")
                _write(root / "juv", f"juv_{i}", "blue", f"juv, woman {i}")
            # Same bytes in both sets: the original mixing bug.
            (root / "juv" / "juv_0.jpg").write_bytes((root / "faiz" / "faiz_0.jpg").read_bytes())

            _, problems = validate(root)

            self.assertTrue(any("appear in BOTH sets" in p for p in problems))

    def test_missing_caption_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(15):
                _write(root / "faiz", f"faiz_{i}", "red", f"faiz, man {i}")
                _write(root / "juv", f"juv_{i}", "blue", f"juv, woman {i}")
            (root / "faiz" / "faiz_0.txt").unlink()

            _, problems = validate(root)

            self.assertTrue(any("no caption" in p for p in problems))

    def test_wrong_trigger_prefix_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i in range(15):
                _write(root / "faiz", f"faiz_{i}", "red", f"faiz, man {i}")
                _write(root / "juv", f"juv_{i}", "blue", f"juv, woman {i}")
            (root / "faiz" / "faiz_0.txt").write_text("juv, wrong", encoding="utf-8")

            _, problems = validate(root)

            self.assertTrue(any("does not start with" in p for p in problems))

    def test_too_few_images_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            _write(root / "faiz", "faiz_0", "red", "faiz, man")
            _write(root / "juv", "juv_0", "blue", "juv, woman")

            _, problems = validate(root)

            self.assertTrue(any("only 1 images" in p for p in problems))


class WriteBundleTests(unittest.TestCase):
    def test_bundle_pairs_images_with_captions_and_config(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "faiz"
            _write(data, "faiz_0", "red", "faiz, man")
            _write(data, "faiz_1", "red", "faiz, man")

            bundle = write_bundle("faiz", sorted(data.glob("*.jpg")), root / "out", "faiz")

            with zipfile.ZipFile(bundle) as archive:
                names = archive.namelist()
                self.assertEqual(sum(n.endswith(".jpg") for n in names), 2)
                self.assertEqual(sum(n.endswith(".txt") for n in names), 2)
                self.assertIn("faiz/train_config.json", names)


if __name__ == "__main__":
    unittest.main()
