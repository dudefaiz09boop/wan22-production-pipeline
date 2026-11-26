import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_prompts
import export_story
import validate


ROOT = Path(__file__).resolve().parents[1]


def validator_shot(**updates):
    shot = {
        "id": "c01",
        "scene": 1,
        "build": "fm",
        "start_frame": "startframes/c01.png",
        "loras": [],
        "speaker": "Juveriya",
        "say": "hello",
        "chain": True,
        "cam": [
            "static wide: Juveriya waits",
            "zoom in: Juveriya looks up",
            "dolly in: Juveriya smiles",
            "pan left: the room",
            "tilt up: the window",
            "orbit: Juveriya turns",
        ],
    }
    shot.update(updates)
    return shot


class LookPrefixTests(unittest.TestCase):
    """Camera/lens vocabulary is chosen by the shot's light."""

    META = {
        "looks": {
            "night_tube": "shot on a phone, ISO 3200",
            "daylight": "shot on a phone, overcast window light",
        },
        "look_default": "shot on a phone, available light",
    }

    def test_light_selects_its_look(self):
        self.assertEqual(
            build_prompts.look_prefix({"light": "night_tube"}, self.META),
            "shot on a phone, ISO 3200",
        )

    def test_unknown_light_falls_back_to_default(self):
        self.assertEqual(
            build_prompts.look_prefix({"light": "candle"}, self.META),
            "shot on a phone, available light",
        )

    def test_missing_light_falls_back_to_default(self):
        self.assertEqual(
            build_prompts.look_prefix({}, self.META),
            "shot on a phone, available light",
        )

    def test_no_looks_configured_returns_empty(self):
        self.assertEqual(build_prompts.look_prefix({"light": "daylight"}, {}), "")

    def test_look_leads_the_compiled_prompt(self):
        shot = validator_shot(light="night_tube")
        errors = []
        plan = build_prompts.build_plan(
            shot,
            {
                "res_base": [832, 480],
                "res_face": [1280, 720],
                "sampler": "euler_simple",
                "steps": [2, 3],
                "cfg": 1,
                "duration": 5,
                **self.META,
            },
            "negative.txt",
            set(),
            errors,
            check_frames=False,
        )
        self.assertIsNotNone(plan)
        self.assertTrue(plan["body"].startswith("shot on a phone, ISO 3200, "))


class LanguageScriptTests(unittest.TestCase):
    def run_validator(self, shot, *args, frame_exists=True):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "startframes").mkdir()
            if frame_exists:
                (root / "startframes" / "c01.png").touch()
            (root / "negative.txt").write_text("negative", encoding="utf-8")
            spec = {
                "meta": {"negative": "negative.txt"},
                "story": {"scenes": {1: {"title": "Scene"}}},
                "shots": [shot],
            }
            (root / "spec.yaml").write_text(
                yaml.safe_dump(spec, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            stdout, stderr = io.StringIO(), io.StringIO()
            argv = ["validate.py", "--root", str(root), "--spec", "spec.yaml", *args]
            with mock.patch.object(sys, "argv", argv):
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    result = validate.main()
            return result, stdout.getvalue(), stderr.getvalue()

    def run_production_validator(self, mutate):
        source = yaml.safe_load(
            (ROOT / "docs" / "beats.yaml").read_text(encoding="utf-8")
        )
        mutate(source)
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", suffix=".yaml", delete=False
        ) as handle:
            yaml.safe_dump(source, handle, allow_unicode=True, sort_keys=False)
            spec_path = handle.name
        try:
            stdout, stderr = io.StringIO(), io.StringIO()
            argv = [
                "validate.py", "--root", str(ROOT), "--spec", spec_path, "--no-frames",
            ]
            with mock.patch.object(sys, "argv", argv):
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    result = validate.main()
            return result, stdout.getvalue(), stderr.getvalue()
        finally:
            Path(spec_path).unlink(missing_ok=True)

    def test_explicit_romanized_language_skips_detection(self):
        with mock.patch.object(validate, "detect_language", side_effect=AssertionError):
            result, _, errors = self.run_validator(
                validator_shot(language="hi"),
                "--doctrine",
                "--strict",
                "--langs",
                "hi",
            )
        self.assertEqual(result, 0, errors)

    def test_script_specific_languages_require_the_matching_script(self):
        cases = (
            ("hi-script", "नमस्ते", 0, "Devanagari"),
            ("hi-script", "namaste", 1, "Devanagari"),
            ("ur-script", "سلام", 0, "Arabic"),
            ("ur-script", "salaam", 1, "Arabic"),
            ("hi", "namaste", 0, ""),
            ("ur", "salaam", 0, ""),
        )
        for language, script, expected, message in cases:
            with self.subTest(language=language, script=script):
                result, _, errors = self.run_validator(
                    validator_shot(language=language, say_script=script)
                )
                self.assertEqual(result, expected, errors)
                if message and expected:
                    self.assertIn(message, errors)

    def test_language_detection_remains_fallback_without_language_field(self):
        result, _, errors = self.run_validator(
            validator_shot(),
            "--doctrine",
            "--strict",
            "--langs",
            "hi",
        )
        self.assertEqual(result, 1)
        self.assertIn("lang en not in", errors)

    def test_dialogue_language_overrides_the_shot_language(self):
        shot = validator_shot(
            say="",
            speaker=None,
            language="hi",
            dialogue=[{
                "speaker": "Juveriya",
                "text": "Haan Ammi",
                "text_script": "ہاں امی۔",
                "language": "ur",
            }],
        )
        with mock.patch.object(validate, "detect_language", side_effect=AssertionError):
            result, _, errors = self.run_validator(
                shot, "--doctrine", "--strict", "--langs", "hi", "--require-language"
            )
        self.assertEqual(result, 0, errors)

    def test_require_language_requires_it_for_speech_and_forbids_it_when_silent(self):
        missing, _, missing_errors = self.run_validator(
            validator_shot(), "--require-language"
        )
        present, _, present_errors = self.run_validator(
            validator_shot(language="en"), "--require-language"
        )
        silent, _, silent_errors = self.run_validator(
            validator_shot(say="", speaker=None, language="en"),
            "--require-language",
        )
        self.assertEqual(missing, 1)
        self.assertIn("spoken shot needs a language", missing_errors)
        self.assertEqual(present, 0, present_errors)
        self.assertEqual(silent, 1)
        self.assertIn("silent shot must not define a language", silent_errors)

    def test_noise_3_with_duration_5_fails(self):
        result, _, errors = self.run_production_validator(
            lambda spec: spec["shots"][11].update(noise=3, duration=5)
        )
        self.assertEqual(result, 1)
        self.assertIn("c12 noise 3 requires duration <= 4", errors)

    def test_noise_2_with_duration_4_fails(self):
        result, _, errors = self.run_production_validator(
            lambda spec: spec["shots"][0].update(noise=2, duration=4)
        )
        self.assertEqual(result, 1)
        self.assertIn("c01 duration must match meta.duration (5)", errors)

    def test_s2v_with_missing_sidecar_fails(self):
        result, _, errors = self.run_production_validator(
            lambda spec: spec["shots"][0].update(audio_mode="s2v")
        )
        self.assertEqual(result, 1)
        self.assertIn(
            "c01 s2v requires an audio sidecar (refs/audio/c01.wav or refs/audio/index.json)",
            errors,
        )

    def test_s2v_with_resolved_sidecar_passes(self):
        with mock.patch.object(validate, "resolve_audio", return_value=Path("x.wav")):
            result, _, errors = self.run_production_validator(
                lambda spec: spec["shots"][0].update(audio_mode="s2v")
            )
        self.assertEqual(result, 0, errors)

    def test_ammi_depiction_key_must_reference_ammi_or_be_in_phone_call_span(self):
        result, _, errors = self.run_production_validator(
            lambda spec: spec["story"]["relations"]["ammi_depiction"].update(c01="voice")
        )
        self.assertEqual(result, 1)
        self.assertIn(
            "story.relations.ammi_depiction.c01 must reference Ammi",
            errors,
        )

    def test_production_duration_must_match_five_second_meta_duration(self):
        result, _, errors = self.run_production_validator(
            lambda spec: spec["shots"][11].update(noise=2, duration=4)
        )
        self.assertEqual(result, 1)
        self.assertIn("c12 duration must match meta.duration (5)", errors)

    def test_chained_production_beats_preserve_stable_anchors(self):
        result, _, errors = self.run_production_validator(
            lambda spec: spec["shots"][1]["anchors"].update(wardrobe="different outfit")
        )
        self.assertEqual(result, 1)
        self.assertIn("c02 anchors.wardrobe must match c01 when chain is true", errors)

    def test_validator_rejects_scenes_missing_from_story(self):
        result, _, errors = self.run_validator(validator_shot(scene=10))
        self.assertEqual(result, 1)
        self.assertIn("c01 scene 10 is not defined in story.scenes", errors)

    def test_validator_reports_raw_c1_controls_with_offset_and_line(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "startframes").mkdir()
            (root / "startframes" / "c01.png").touch()
            (root / "negative.txt").write_text("negative", encoding="utf-8")
            source = (
                "meta:\n  negative: negative.txt\n"
                "story:\n  scenes:\n    1: {title: Scene}\n"
                "shots:\n  - id: c01\u0085\n"
            )
            (root / "spec.yaml").write_text(source, encoding="utf-8")
            stdout, stderr = io.StringIO(), io.StringIO()
            argv = ["validate.py", "--root", str(root), "--spec", "spec.yaml"]
            with mock.patch.object(sys, "argv", argv):
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    result = validate.main()
        self.assertEqual(result, 1)
        self.assertIn("raw C1 control U+0085", stderr.getvalue())
        self.assertIn(f"offset {source.index(chr(0x85))}", stderr.getvalue())
        self.assertIn(f"line {source[:source.index(chr(0x85))].count(chr(10)) + 1}", stderr.getvalue())

    def test_no_frames_skips_only_start_frame_disk_check(self):
        result, _, errors = self.run_validator(
            validator_shot(start_frame="startframes/c01.png"),
            "--no-frames",
            frame_exists=False,
        )
        self.assertEqual(result, 0, errors)

    def test_build_prompts_no_frames_skips_only_start_frame_disk_check(self):
        shot = validator_shot()
        errors = []
        plan = build_prompts.build_plan(
            shot,
            {
                "res_base": [832, 480],
                "res_face": [1280, 720],
                "sampler": "euler_simple",
                "steps": [2, 3],
                "cfg": 1,
                "duration": 5,
            },
            "negative.txt",
            set(),
            errors,
            check_frames=False,
        )
        self.assertIsNotNone(plan)
        self.assertEqual(errors, [])

    def test_prompt_cfg_keeps_both_texts_and_selects_one_narration_variant(self):
        shot = {
            "id": "c01",
            "scene": 1,
            "build": "fm",
            "start_frame": "startframes/c01.png",
            "loras": [],
            "chain": True,
            "language": "hi",
            "say": "namaste",
            "say_script": "नमस्ते",
            "speaker": "Juveriya",
            "dialogue": [{
                "speaker": "Juveriya",
                "text": "salaam",
                "text_script": "سلام",
                "language": "ur",
            }],
            "cam": [
                "static wide: Juveriya waits",
                "zoom in: Juveriya looks up",
                "dolly in: Juveriya smiles",
                "pan left: the room",
                "tilt up: the window",
                "orbit: Juveriya turns",
            ],
        }
        meta = {"res_base": [832, 480], "sampler": "euler_simple"}
        with mock.patch.object(build_prompts.Path, "is_file", return_value=True):
            romanized = build_prompts.build_plan(
                shot, meta, "negative.txt", set(), [], lang="en"
            )
            scripted = build_prompts.build_plan(
                shot, meta, "negative.txt", set(), [], lang="hi"
            )

        self.assertIsNotNone(romanized)
        self.assertIsNotNone(scripted)
        # The cam lines carry a compact "(Ns: ...)" marker rather than
        # "(At N seconds: ...)": the long form cost 4-5 CLIP tokens per line and
        # pushed every prompt past the 77-token window.
        self.assertEqual(romanized["body"].count("s: "), 6)
        self.assertNotIn("नमस्ते", romanized["body"])
        self.assertEqual(romanized["cfg"]["say"], "namaste")
        self.assertEqual(romanized["cfg"]["say_script"], "नमस्ते")
        self.assertEqual(romanized["cfg"]["dialogue"][0]["text"], "salaam")
        self.assertEqual(romanized["cfg"]["dialogue"][0]["text_script"], "سلام")
        self.assertEqual(romanized["cfg"]["narration"], "Juveriya: namaste\nJuveriya: salaam")
        self.assertEqual(scripted["cfg"]["narration"], "Juveriya: नमस्ते\nJuveriya: salaam")
        mixed = build_prompts.build_plan(
            shot, meta, "negative.txt", set(), [], lang="mixed"
        )
        self.assertEqual(mixed["cfg"]["narration"], "Juveriya: नमस्ते\nJuveriya: سلام")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            spec = {
                "meta": {
                    "res_base": [832, 480],
                    "res_face": [1280, 720],
                    "sampler": "euler_simple",
                    "steps": [2, 3],
                    "cfg": 1,
                    "duration": 5,
                    "negative": "negatives/artifact_neg.txt",
                },
                "shots": [shot],
            }
            spec_path = root / "spec.yaml"
            spec_path.write_text(
                yaml.safe_dump(spec, allow_unicode=True, sort_keys=False),
                encoding="utf-8",
            )
            out = root / "prompts"
            stdout = io.StringIO()
            argv = [
                "build_prompts.py",
                "--spec",
                str(spec_path),
                "--out",
                str(out),
                "--lang",
                "hi",
            ]
            with mock.patch.object(sys, "argv", argv):
                with contextlib.redirect_stdout(stdout):
                    self.assertEqual(build_prompts.main(), 0, stdout.getvalue())
            output_cfg = json.loads((out / "c01.json").read_text(encoding="utf-8"))
            output_body = (out / "c01.txt").read_text(encoding="utf-8")
        self.assertEqual(output_cfg["narration"], "Juveriya: नमस्ते\nJuveriya: salaam")
        self.assertEqual(output_cfg["say"], "namaste")
        self.assertEqual(output_cfg["say_script"], "नमस्ते")
        self.assertEqual(output_body.count("(At "), 6)

    def test_story_export_selects_script_text_and_preserves_per_beat_language(self):
        spec = {
            "meta": {
                "duration": 5,
                "sampler": "euler_simple",
                "steps": [2, 3],
                "cfg": 1,
                "res_base": [832, 480],
                "res_face": [1280, 720],
                "negative": "negatives/artifact_neg.txt",
            },
            "story": {
                "title": "Test",
                "synopsis": "A test story.",
                "logline": "A test.",
                "theme": "Testing.",
                "boundaries": {
                    "characters_are_adults": True,
                    "intimacy": "Not applicable.",
                    "consent_beats": ["c01"],
                    "revocable": True,
                },
                "characters": [{"name": "Juveriya", "adult": True}],
                "scenes": {
                    1: {"title": "Scene", "description": "A test scene."},
                },
            },
            "shots": [
                {
                    "id": "c01",
                    "scene": 1,
                    "build": "fm",
                    "start_frame": "startframes/c01.png",
                    "loras": [],
                    "chain": True,
                    "language": "hi",
                    "say": "namaste",
                    "say_script": "नमस्ते",
                    "speaker": "Juveriya",
                    "cam": [
                        "static wide: Juveriya waits",
                        "zoom in: Juveriya looks up",
                        "dolly in: Juveriya smiles",
                        "pan left: the room",
                        "tilt up: the window",
                        "orbit: Juveriya turns",
                    ],
                },
                {
                    "id": "c02",
                    "scene": 1,
                    "build": "fm",
                    "start_frame": "startframes/c02.png",
                    "loras": [],
                    "chain": True,
                    "language": "hi",
                    "dialogue": [{
                        "speaker": "Juveriya",
                        "text": "salaam",
                        "text_script": "سلام",
                        "language": "ur",
                    }],
                    "cam": [
                        "static wide: Juveriya waits",
                        "zoom in: Juveriya looks up",
                        "dolly in: Juveriya smiles",
                        "pan left: the room",
                        "tilt up: the window",
                        "orbit: Juveriya turns",
                    ],
                },
                {
                    "id": "c03",
                    "scene": 1,
                    "build": "fm",
                    "start_frame": "startframes/c03.png",
                    "loras": [],
                    "chain": True,
                    "say": "",
                    "cam": [
                        "static wide: Juveriya waits",
                        "zoom in: Juveriya looks up",
                        "dolly in: Juveriya smiles",
                        "pan left: the room",
                        "tilt up: the window",
                        "orbit: Juveriya turns",
                    ],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            spec_path = Path(directory) / "story.yaml"
            spec_path.write_text(yaml.safe_dump(spec, allow_unicode=True), encoding="utf-8")
            default_plan = export_story.build_story_plan(spec_path)
            hindi_plan = export_story.build_story_plan(spec_path, lang="hi")
            urdu_plan = export_story.build_story_plan(spec_path, lang="ur")
            plan = export_story.build_story_plan(spec_path, lang="mixed")

        default_shots = [
            shot
            for scene in default_plan["scenes"]
            for shot in scene["shots"]
        ]
        hindi_shots = [
            shot
            for scene in hindi_plan["scenes"]
            for shot in scene["shots"]
        ]
        urdu_shots = [
            shot
            for scene in urdu_plan["scenes"]
            for shot in scene["shots"]
        ]
        shots = [
            shot
            for scene in plan["scenes"]
            for shot in scene["shots"]
        ]
        self.assertEqual(default_shots[0]["narration"], "Juveriya: namaste")
        self.assertEqual(default_shots[1]["narration"], "Juveriya: salaam")
        self.assertEqual(shots[0]["narration"], "Juveriya: नमस्ते")
        self.assertEqual(shots[0]["sourceBeat"]["language"], "hi")
        self.assertEqual(shots[1]["narration"], "Juveriya: سلام")
        self.assertEqual(shots[1]["sourceBeat"]["language"], "hi")
        self.assertEqual(hindi_shots[1]["narration"], "Juveriya: salaam")
        self.assertEqual(urdu_shots[1]["narration"], "Juveriya: سلام")
        self.assertEqual(shots[2]["sourceBeat"]["language"], "en")


if __name__ == "__main__":
    unittest.main()
