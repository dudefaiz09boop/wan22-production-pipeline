import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import generate_narration
import mux_audio

ROOT = Path(__file__).resolve().parents[1]


class ScriptDetectionTests(unittest.TestCase):
    def test_latin(self):
        self.assertEqual(generate_narration.detect_script("You've been a bitch"), "latin")

    def test_devanagari(self):
        self.assertEqual(generate_narration.detect_script("\u0915\u093e\u0909\u0902\u091f"), "devanagari")

    def test_arabic_urdu(self):
        self.assertEqual(generate_narration.detect_script("\u0622\u067e \u06a9\u06cc\u0633\u06d2"), "arabic")

    def test_empty_is_latin(self):
        self.assertEqual(generate_narration.detect_script(""), "latin")

    def test_mixed_prefers_non_latin(self):
        # A romanized line with one Devanagari word still needs the Hindi voice.
        self.assertEqual(
            generate_narration.detect_script("bolo \u0939\u093e\u0902"), "devanagari"
        )


class VoiceRoutingTests(unittest.TestCase):
    def test_latin_uses_indian_english(self):
        self.assertEqual(
            generate_narration.voice_for("faiz", "hello"), "en-IN-PrabhatNeural"
        )
        self.assertEqual(
            generate_narration.voice_for("juv", "hello"), "en-IN-NeerjaNeural"
        )

    def test_devanagari_uses_hindi(self):
        self.assertEqual(
            generate_narration.voice_for("faiz", "\u0939\u093e\u0902"),
            "hi-IN-MadhurNeural",
        )
        self.assertEqual(
            generate_narration.voice_for("juv", "\u0939\u093e\u0902"),
            "hi-IN-SwaraNeural",
        )

    def test_arabic_uses_urdu(self):
        self.assertEqual(
            generate_narration.voice_for("faiz", "\u0622\u067e"),
            "ur-PK-AsadNeural",
        )
        self.assertEqual(
            generate_narration.voice_for("juv", "\u0622\u067e"),
            "ur-PK-UzmaNeural",
        )

    def test_unknown_speaker_falls_back_to_female(self):
        self.assertEqual(
            generate_narration.voice_for("nobody", "hello"),
            "en-IN-NeerjaNeural",
        )

    def test_romanized_hindi_stays_english_voice(self):
        # "Padh rahi hoon" is romanized; an Indian-English voice reads it right.
        self.assertEqual(
            generate_narration.voice_for("juv", "Padh rahi hoon. Books."),
            "en-IN-NeerjaNeural",
        )


class MuxTargetSizeTests(unittest.TestCase):
    def test_explicit_override_wins(self):
        meta = {"res_base": [832, 480]}
        self.assertEqual(mux_audio.resolve_target_size(meta, 640, 360), (640, 360))

    def test_falls_back_to_res_base(self):
        meta = {"res_base": [832, 480], "res_face": [1280, 720]}
        self.assertEqual(mux_audio.resolve_target_size(meta, None, None), (832, 480))

    def test_missing_meta_returns_none(self):
        self.assertIsNone(mux_audio.resolve_target_size({}, None, None))

    def test_real_spec_resolves(self):
        meta = mux_audio.load_meta(ROOT / "docs" / "beats.yaml")
        self.assertEqual(mux_audio.resolve_target_size(meta, None, None), (832, 480))


class RealismFilterTests(unittest.TestCase):
    def test_geometry_is_always_normalised(self):
        chain = mux_audio.realism_filters(832, 480, 16, grade=False)
        self.assertIn("scale=832:480", chain)
        self.assertIn("pad=832:480", chain)
        self.assertIn("setsar=1", chain)
        self.assertIn("fps=16", chain)
        self.assertNotIn("noise=", chain)

    def test_grade_adds_grain_halation_and_ca(self):
        chain = mux_audio.realism_filters(832, 480, 16, grade=True)
        self.assertIn("noise=", chain)
        self.assertIn("rgbashift=", chain)
        self.assertIn("blend=all_mode=screen", chain)
        self.assertIn("unsharp=", chain)

    def test_grain_runs_after_scaling(self):
        chain = mux_audio.realism_filters(832, 480, 16, grade=True)
        self.assertLess(chain.index("scale="), chain.index("noise="))


class AudioIndexEntryTests(unittest.TestCase):
    def test_bare_filename_is_a_mix_with_no_stems(self):
        self.assertEqual(
            generate_narration.index_entry({"c01": "c01.wav"}, "c01"),
            ("c01.wav", {}),
        )

    def test_two_speaker_entry_returns_stems(self):
        entry = {"mix": "c25.wav", "stems": {"faiz": "c25_faiz.wav"}}
        self.assertEqual(
            generate_narration.index_entry({"c25": entry}, "c25"),
            ("c25.wav", {"faiz": "c25_faiz.wav"}),
        )

    def test_missing_beat(self):
        self.assertEqual(generate_narration.index_entry({}, "c99"), (None, {}))


class LipsyncWorkflowTests(unittest.TestCase):
    def setUp(self):
        import lipsync

        self.mod = lipsync

    def build(self, **extra):
        return self.mod.build_infinitetalk_workflow(
            "p", "n", "img.png", "a1.wav", 832, 480, 81, 1, 6, 1.0, **extra
        )

    def test_single_speaker_by_default(self):
        wf = self.build()
        self.assertEqual(wf["11"]["inputs"]["mode"], {"mode": "single_speaker"})
        self.assertNotIn("15", wf)

    def test_two_speakers_wires_second_audio_and_masks(self):
        wf = self.build(
            audio_name_2="a2.wav", mask_name_1="m1.png", mask_name_2="m2.png"
        )
        mode = wf["11"]["inputs"]["mode"]
        self.assertEqual(mode["mode"], "two_speakers")
        self.assertEqual(mode["audio_encoder_output_2"], ["16", 0])
        # LoadImage slot 1 is the MASK output.
        self.assertEqual(mode["mask_1"], ["17", 1])
        self.assertEqual(mode["mask_2"], ["18", 1])
        self.assertEqual(wf["15"]["inputs"]["audio"], "a2.wav")

    def test_partial_two_speaker_inputs_fall_back(self):
        wf = self.build(audio_name_2="a2.wav")
        self.assertEqual(wf["11"]["inputs"]["mode"], {"mode": "single_speaker"})

    def test_frames_follow_the_16fps_contract(self):
        import generate_workflow

        self.assertEqual(generate_workflow.frames_for(5), 81)
        self.assertEqual(generate_workflow.frames_for(4), 65)
        for duration in (3, 4, 5, 6):
            self.assertEqual((generate_workflow.frames_for(duration) - 1) % 4, 0)


class MuxSourcePreferenceTests(unittest.TestCase):
    """The mux must prefer the lipsynced clip over the raw render."""

    def test_lipsynced_clip_wins(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "renders" / "lipsync").mkdir(parents=True)
            (root / "renders" / "c01.mp4").write_text("raw")
            (root / "renders" / "lipsync" / "c01.mp4").write_text("lip")
            candidates = sorted((root / "renders").glob("**/c01.mp4"))
            lipsynced = [c for c in candidates if "lipsync" in c.parts]
            self.assertEqual(lipsynced[0].read_text(), "lip")

    def test_plain_sorted_order_would_pick_the_raw_clip(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "renders" / "lipsync").mkdir(parents=True)
            (root / "renders" / "c01.mp4").write_text("raw")
            (root / "renders" / "lipsync" / "c01.mp4").write_text("lip")
            candidates = sorted((root / "renders").glob("**/c01.mp4"))
            self.assertEqual(candidates[0].read_text(), "raw")


class GgufManifestTests(unittest.TestCase):
    def setUp(self):
        import download_wan22_gguf

        self.mod = download_wan22_gguf

    def test_four_checkpoints_declared(self):
        self.assertEqual(len(self.mod.GGUF_VERSIONS), 4)

    def test_both_builds_present(self):
        builds = {v[0] for v in self.mod.GGUF_VERSIONS.values()}
        self.assertEqual(builds, {"fm", "cf"})

    def test_each_build_has_high_and_low(self):
        for build in ("fm", "cf"):
            stages = {v[1] for v in self.mod.GGUF_VERSIONS.values() if v[0] == build}
            self.assertEqual(stages, {"high", "low"}, build)

    def test_fm_filenames_match_the_workflow(self):
        import generate_workflow

        fm = {v[1]: v[2] for v in self.mod.GGUF_VERSIONS.values() if v[0] == "fm"}
        self.assertEqual(fm["high"], generate_workflow.GGUF_Q8_HIGH)
        self.assertEqual(fm["low"], generate_workflow.GGUF_Q8_LOW)

    def test_cf_filenames_match_render(self):
        import render

        cf = {v[1]: v[2] for v in self.mod.GGUF_VERSIONS.values() if v[0] == "cf"}
        self.assertEqual(cf["high"], render.BUILD_MODELS["cf"][0])
        self.assertEqual(cf["low"], render.BUILD_MODELS["cf"][1])

    def test_fm_hashes_match_run_production(self):
        import run_production

        # The hash map lives inside download_all_models; assert the literals.
        source = (ROOT / "scripts" / "run_production.py").read_text(encoding="utf-8")
        for version_id, (build, stage, filename, sha) in self.mod.GGUF_VERSIONS.items():
            if build != "fm":
                continue
            self.assertIsNotNone(sha, f"{filename} should be hash-pinned")
            self.assertIn(sha, source, f"{filename} hash missing from run_production")

    def test_low_filename_keeps_the_author_typo(self):
        low = [v[2] for v in self.mod.GGUF_VERSIONS.values() if v[1] == "low" and v[0] == "fm"][0]
        self.assertIn("LOW_Q8lightning_edition)", low)


if __name__ == "__main__":
    unittest.main()
