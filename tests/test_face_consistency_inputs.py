from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_face_consistency_inputs import prepare_frame

ROOT = Path(__file__).resolve().parents[1]


class PrepareFaceConsistencyInputsTests(unittest.TestCase):
    def test_portrait_content_is_preserved_in_landscape_frame(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "portrait.png"
            output = root / "prepared.png"
            Image.new("RGB", (200, 400), "red").save(source)

            result = prepare_frame(source, output)

            self.assertTrue(result["sourceContentRetained"])
            self.assertEqual(result["outputSize"], (1280, 720))
            self.assertEqual(result["fullSourceContentBox"], [460, 0, 820, 720])
            with Image.open(output) as prepared:
                self.assertEqual(prepared.size, (1280, 720))
                self.assertEqual(prepared.getpixel((640, 360)), (255, 0, 0))
            with Image.open(source) as original:
                self.assertEqual(original.size, (200, 400))

    def test_benchmark_uses_current_builds_and_a_six_line_camera_first_prompt(self) -> None:
        manifest_path = ROOT / "benchmarks" / "face_consistency" / "benchmark.yaml"
        manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
        prompt = (manifest_path.parent / manifest["prompt_file"]).read_text(encoding="utf-8").splitlines()
        beats = yaml.safe_load((ROOT / "docs" / "beats.yaml").read_text(encoding="utf-8"))["shots"]
        builds = {shot["id"]: shot["build"] for shot in beats}
        verbs = "static|zoom|dolly|pan|tilt|orbit|track|arc|handheld|whip"

        self.assertEqual(len(prompt), 6)
        for second, line in enumerate(prompt):
            self.assertRegex(line, rf"^\(At {second} seconds: ({verbs}), ")
        self.assertEqual(
            [(case["source_beat"], case["build"]) for case in manifest["cases"]],
            [("c08", "fm"), ("c18", "fm")],
        )
        for case in manifest["cases"]:
            self.assertEqual(case["build"], builds[case["source_beat"]])
            self.assertEqual(
                case["reference_image"],
                f"startframes/{case['source_beat']}.png",
            )
            self.assertEqual(
                case["prepared_image"],
                f"{case['source_beat']}_face_test.png",
            )


if __name__ == "__main__":
    unittest.main()
