from __future__ import annotations

import io
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prep_source_frame import compose_plate


class PrepSourceFrameTests(unittest.TestCase):
    def test_plate_preserves_portrait_over_blurred_full_bleed_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "portrait.png"
            Image.new("RGB", (100, 200), (10, 20, 30)).save(source)
            original_bytes = source.read_bytes()
            output = root / "plate.png"

            result = compose_plate(source, output, 320, 180)

            self.assertEqual(result["outputSize"], [320, 180])
            self.assertEqual(result["sourceSize"], [100, 200])
            self.assertEqual(result["fullSourceContentBox"], [115, 0, 205, 180])
            self.assertTrue(result["sourceContentRetained"])
            with Image.open(output) as plate:
                self.assertEqual(plate.size, (320, 180))
                # Foreground interior keeps the exact source color.
                self.assertEqual(plate.getpixel((160, 90)), (10, 20, 30))
            # The source is never mutated.
            self.assertEqual(source.read_bytes(), original_bytes)

    def test_exact_dimensions_are_asserted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "portrait.png"
            Image.new("RGB", (100, 200), "red").save(source)
            output = root / "plate.png"

            result = compose_plate(source, output, 640, 360)

            with Image.open(output) as plate:
                self.assertEqual(plate.size, (640, 360))
            self.assertEqual(result["outputSize"], [640, 360])


if __name__ == "__main__":
    unittest.main()
