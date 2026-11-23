import sys
import tempfile
import unittest
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import export_story


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "docs" / "beats.yaml"


class ExportStoryTests(unittest.TestCase):
    def test_production_import_preserves_all_authored_shots_and_metadata(self):
        source = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
        plan = export_story.build_story_plan(SPEC_PATH)
        imported_shots = [
            shot
            for scene in plan["scenes"]
            for shot in scene["shots"]
        ]

        self.assertEqual(plan["title"], source["story"]["title"])
        self.assertEqual(plan["targetSeconds"], 344)
        self.assertEqual(len(plan["scenes"]), len(source["story"]["scenes"]))
        self.assertEqual(len(imported_shots), 70)
        expected_four_second = {"c12", "c18", "c22", "c35", "c36", "c37"}
        self.assertTrue(all(
            shot["sourceBeat"]["duration"] == (4 if shot["sourceBeat"]["id"] in expected_four_second else 5)
            for shot in imported_shots
        ))

        expected_negative = (
            ROOT / source["meta"]["negative"]
        ).read_text(encoding="utf-8").strip()
        self.assertEqual(plan["defaultNegativePrompt"], expected_negative)
        self.assertIn("All characters are adults.", plan["starterBrief"])
        self.assertTrue(all(
            character["adult"] is True
            for character in source["story"]["characters"]
        ))

        known_names = {
            character["name"]
            for character in source["story"]["characters"]
        }
        for source_shot, imported_shot in zip(source["shots"], imported_shots):
            expected_prompt = "\n".join(
                f"(At {second} seconds: {line})"
                for second, line in enumerate(source_shot["cam"])
            )
            self.assertEqual(imported_shot["title"], source_shot["id"])
            self.assertEqual(imported_shot["visualPrompt"], expected_prompt)
            self.assertEqual(imported_shot["negativePrompt"], expected_negative)
            self.assertTrue(imported_shot["characters"])
            self.assertLessEqual(set(imported_shot["characters"]), known_names)

            dialogue = source_shot.get("dialogue")
            if dialogue is not None:
                expected_narration = "\n".join(
                    f"{turn['speaker']}: {turn['text']}" for turn in dialogue
                )
            elif source_shot.get("say", "").strip():
                expected_narration = f"{source_shot['speaker']}: {source_shot['say']}"
            else:
                expected_narration = ""
            self.assertEqual(imported_shot["narration"], expected_narration)

            resolution = (
                source["meta"]["res_face"]
                if source_shot["build"] == "cf"
                else source["meta"]["res_base"]
            )
            beat_duration = source_shot["duration"]
            self.assertEqual(imported_shot["sourceBeat"], {
                "id": source_shot["id"],
                "scene": source_shot["scene"],
                "language": source_shot.get("language", "en"),
                "build": source_shot["build"],
                "startFrame": source_shot["start_frame"],
                "loras": source_shot["loras"],
                "chain": source_shot["chain"],
                "sampler": source["meta"]["sampler"],
                "steps": source["meta"]["steps"],
                "cfg": source["meta"]["cfg"],
                "duration": beat_duration,
                "width": resolution[0],
                "height": resolution[1],
                "noise": source_shot["noise"],
                "light": source_shot["light"],
                "mark_state": source_shot["mark_state"],
                "thread": source_shot["thread"],
                "profile": source_shot["profile"],
                "anchors": source_shot["anchors"],
                "audio_mode": source_shot["audio_mode"],
            })

        for scene in plan["scenes"]:
            self.assertTrue(scene["title"])
            self.assertTrue(scene["description"])

    def test_production_import_rejects_noise_duration_mismatches(self):
        source = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
        source["shots"][11]["duration"] = 5
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".yaml", dir=ROOT, delete=False
        ) as handle:
            yaml.safe_dump(source, handle, allow_unicode=True, sort_keys=False)
            spec_path = Path(handle.name)
        try:
            with self.assertRaisesRegex(ValueError, "noise 3 requires duration <= 4"):
                export_story.build_story_plan(spec_path)
        finally:
            spec_path.unlink(missing_ok=True)

    def test_production_import_rejects_s2v_missing_sidecar(self):
        source = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
        source["shots"][0]["audio_mode"] = "s2v"
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".yaml", dir=ROOT, delete=False
        ) as handle:
            yaml.safe_dump(source, handle, allow_unicode=True, sort_keys=False)
            spec_path = Path(handle.name)
        try:
            with self.assertRaisesRegex(ValueError, "s2v requires an audio sidecar"):
                export_story.build_story_plan(spec_path)
        finally:
            spec_path.unlink(missing_ok=True)

    def test_production_import_rejects_broken_chained_wardrobe_anchor(self):
        source = yaml.safe_load(SPEC_PATH.read_text(encoding="utf-8"))
        source["shots"][1]["anchors"]["wardrobe"] = "different outfit"
        source["meta"]["negative"] = str(ROOT / source["meta"]["negative"])
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".yaml", dir=ROOT, delete=False
        ) as handle:
            yaml.safe_dump(source, handle, allow_unicode=True, sort_keys=False)
            spec_path = Path(handle.name)
        try:
            with self.assertRaisesRegex(ValueError, "wardrobe anchor must match c01"):
                export_story.build_story_plan(spec_path)
        finally:
            spec_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
