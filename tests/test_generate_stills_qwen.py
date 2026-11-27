"""The stills graph must name the model files it is told to use.

The 24 GB tier needs Q3_K_S plus the fp8 encoder; a 48 GB card can afford
Q4_K_M. Hardcoding the filenames meant the only way to change tier was to edit
the script, so the tier is now a parameter.
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import generate_stills_qwen as qwen


def build(**overrides):
    kwargs = {
        "prompt": "p",
        "negative": "n",
        "ref_images": ["a.png", "b.png"],
        "width": 896,
        "height": 1344,
        "seed": 42,
    }
    kwargs.update(overrides)
    return qwen.build_workflow(**kwargs)


class ModelSelectionTests(unittest.TestCase):
    def test_defaults_are_the_48gb_tier(self):
        """The default is the full fp8 model, not a GGUF quant.

        The GGUF quants were a Kaggle T4 workaround. On 48 GB the fp8 model
        fits with room for activations and is better quality.
        """
        wf = build()
        self.assertEqual(wf["1"]["inputs"]["unet_name"], "qwen_image_fp8_e4m3fn.safetensors")
        self.assertEqual(wf["2"]["inputs"]["clip_name"], "qwen_2.5_vl_7b_fp8_scaled.safetensors")
        self.assertEqual(wf["3"]["inputs"]["vae_name"], "qwen_image_vae.safetensors")

    def test_unet_can_be_overridden_for_a_bigger_card(self):
        wf = build(unet="qwen-image-Q4_K_M.gguf")
        self.assertEqual(wf["1"]["inputs"]["unet_name"], "qwen-image-Q4_K_M.gguf")

    def test_clip_and_vae_can_be_overridden(self):
        wf = build(clip="other.safetensors", vae="other_vae.safetensors")
        self.assertEqual(wf["2"]["inputs"]["clip_name"], "other.safetensors")
        self.assertEqual(wf["3"]["inputs"]["vae_name"], "other_vae.safetensors")

    def test_encoder_type_stays_qwen_image(self):
        self.assertEqual(build()["2"]["inputs"]["type"], "qwen_image")


class GraphShapeTests(unittest.TestCase):
    def test_reference_images_are_linked_into_the_encoder(self):
        wf = build()
        encode = wf["20"]["inputs"]
        self.assertEqual(encode["images.image_1"], ["10", 0])
        self.assertEqual(encode["images.image_2"], ["11", 0])

    def test_sampler_reads_the_cached_model_and_both_conditionings(self):
        wf = build()
        sampler = wf["22"]["inputs"]
        self.assertEqual(sampler["model"], ["4", 0])
        self.assertEqual(sampler["positive"], ["20", 0])
        self.assertEqual(sampler["negative"], ["20", 1])

    def test_seed_reaches_the_sampler(self):
        self.assertEqual(build(seed=1234)["22"]["inputs"]["seed"], 1234)


class ReferencePlateTests(unittest.TestCase):
    """The face plate is what Qwen reads identity from.

    A 100px face in a full-body frame carries far less detail than a 750px one,
    and the first version of REFS pointed at full-body plates for both. These
    tests keep the face plate a real portrait.
    """

    def test_every_reference_plate_exists(self):
        from pathlib import Path as P

        from scripts.generate_stills_qwen import REFS

        missing = [
            f"{char}/{kind}: {path}"
            for char, plates in REFS.items()
            for kind, path in plates.items()
            if not P(path).is_file()
        ]
        self.assertEqual(missing, [], f"missing reference plates: {missing}")

    def test_face_plate_is_not_the_body_plate(self):
        """A full-body frame is a poor face reference."""
        from scripts.generate_stills_qwen import REFS

        for char, plates in REFS.items():
            self.assertNotEqual(
                plates["face"],
                plates["body"],
                f"{char}: face and body reference are the same file",
            )


if __name__ == "__main__":
    unittest.main()
