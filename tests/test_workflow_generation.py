import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import generate_workflow
import render

ROOT = Path(__file__).resolve().parents[1]


def lora_nodes(wf):
    return [
        node
        for node in wf.values()
        if isinstance(node, dict) and node.get("class_type") == "LoraLoaderModelOnly"
    ]


def samplers(wf):
    return {
        node["_meta"]["title"]: node["inputs"]
        for node in wf.values()
        if isinstance(node, dict) and node.get("class_type") == "KSamplerAdvanced"
    }


class LightningBakedTests(unittest.TestCase):
    """The GGUF Q8 pair has Lightning merged in; stacking LightX2V degrades it."""

    def test_gguf_q8_gets_no_lightx2v_loras(self):
        wf = generate_workflow.generate_workflow("gguf_q8")
        self.assertEqual(lora_nodes(wf), [])

    def test_base_precisions_do_get_lightx2v_loras(self):
        for precision in ("fp8", "fp16"):
            with self.subTest(precision=precision):
                wf = generate_workflow.generate_workflow(precision)
                names = [n["inputs"]["lora_name"] for n in lora_nodes(wf)]
                self.assertEqual(
                    names,
                    [
                        generate_workflow.LIGHTX2V_HIGH,
                        generate_workflow.LIGHTX2V_LOW,
                    ],
                )

    def test_gguf_q8_uses_the_real_civitai_filenames(self):
        wf = generate_workflow.generate_workflow("gguf_q8")
        self.assertEqual(wf["1"]["inputs"]["unet_name"], generate_workflow.GGUF_Q8_HIGH)
        self.assertEqual(wf["2"]["inputs"]["unet_name"], generate_workflow.GGUF_Q8_LOW)
        # The LOW file really is missing its opening paren.
        self.assertIn("LOW_Q8lightning_edition)", wf["2"]["inputs"]["unet_name"])

    def test_gguf_loader_does_not_receive_weight_dtype(self):
        wf = generate_workflow.generate_workflow("gguf_q8")
        self.assertNotIn("weight_dtype", wf["1"]["inputs"])
        self.assertNotIn("weight_dtype", wf["2"]["inputs"])

    def test_unet_loader_requires_weight_dtype(self):
        wf = generate_workflow.generate_workflow("fp8")
        self.assertEqual(wf["1"]["inputs"]["weight_dtype"], "default")
        self.assertEqual(wf["2"]["inputs"]["weight_dtype"], "default")


class ScheduleTests(unittest.TestCase):
    def test_default_schedule_is_two_plus_three(self):
        wf = generate_workflow.generate_workflow("gguf_q8")
        high = wf["9"]["inputs"]
        low = wf["10"]["inputs"]
        self.assertEqual(high["steps"], 5)
        self.assertEqual((high["start_at_step"], high["end_at_step"]), (0, 2))
        self.assertEqual((low["start_at_step"], low["end_at_step"]), (2, 5))
        self.assertEqual(high["cfg"], 1.0)
        self.assertEqual(low["cfg"], 1.0)

    def test_project_frame_and_fps_contract(self):
        wf = generate_workflow.generate_workflow("gguf_q8")
        self.assertEqual(wf["8"]["inputs"]["length"], 81)
        self.assertEqual(wf["12"]["inputs"]["frame_rate"], 16)
        # WanImageToVideo requires (length - 1) % 4 == 0.
        self.assertEqual((wf["8"]["inputs"]["length"] - 1) % 4, 0)

    def test_both_samplers_share_one_sigma_schedule(self):
        """HIGH and LOW must agree on `steps`, or the split lands on a
        different sigma grid and the handoff is wrong."""
        for precision in ("gguf_q8", "fp8", "fp16"):
            with self.subTest(precision=precision):
                wf = generate_workflow.generate_workflow(precision)
                self.assertEqual(wf["9"]["inputs"]["steps"], wf["10"]["inputs"]["steps"])

    def test_checked_in_workflows_match_the_generator(self):
        """A stale file on disk silently overrides the generator."""
        for precision in ("gguf_q8", "fp8", "fp16"):
            with self.subTest(precision=precision):
                path = ROOT / "workflows" / f"base_i2v_{precision}.json"
                on_disk = json.loads(path.read_text(encoding="utf-8"))
                generated = generate_workflow.generate_workflow(precision)
                self.assertEqual(on_disk, generated)

    def test_checked_in_default_workflow_is_consistent(self):
        """base_i2v.json is render.py's default and must obey the same rules."""
        wf = json.loads((ROOT / "workflows" / "base_i2v.json").read_text(encoding="utf-8"))
        self.assertEqual(wf["9"]["inputs"]["steps"], wf["10"]["inputs"]["steps"])
        self.assertEqual(wf["8"]["inputs"]["length"], 81)
        self.assertEqual(wf["12"]["inputs"]["frame_rate"], 16)
        self.assertEqual(wf["3"]["inputs"]["clip_name"], generate_workflow.TEXT_ENCODER)
        self.assertEqual(wf["1"]["inputs"]["unet_name"], generate_workflow.GGUF_Q8_HIGH)
        self.assertEqual(wf["2"]["inputs"]["unet_name"], generate_workflow.GGUF_Q8_LOW)


class NagTests(unittest.TestCase):
    def test_nag_is_off_by_default(self):
        wf = generate_workflow.generate_workflow("gguf_q8")
        self.assertFalse(
            any(
                isinstance(n, dict) and n.get("class_type") == "WanVideoNAG"
                for n in wf.values()
            )
        )

    def test_nag_sits_between_shift_and_sampler(self):
        wf = generate_workflow.generate_workflow("gguf_q8", nag=True)
        self.assertEqual(wf["17"]["class_type"], "WanVideoNAG")
        self.assertEqual(wf["18"]["class_type"], "WanVideoNAG")
        self.assertEqual(wf["17"]["inputs"]["model"], ["13", 0])
        self.assertEqual(wf["18"]["inputs"]["model"], ["14", 0])
        self.assertEqual(wf["17"]["inputs"]["conditioning"], ["7", 0])
        self.assertEqual(wf["9"]["inputs"]["model"], ["17", 0])
        self.assertEqual(wf["10"]["inputs"]["model"], ["18", 0])


class BuildSwitchTests(unittest.TestCase):
    def base(self):
        return json.loads((ROOT / "workflows" / "base_i2v.json").read_text(encoding="utf-8"))

    def test_fm_keeps_lightning_baked_pair_and_adds_no_loras(self):
        wf = render.apply_build(self.base(), {"build": "fm"})
        self.assertEqual(wf["1"]["inputs"]["unet_name"], render.BUILD_MODELS["fm"][0])
        self.assertEqual(wf["2"]["inputs"]["unet_name"], render.BUILD_MODELS["fm"][1])
        self.assertEqual(lora_nodes(wf), [])

    def test_cf_swaps_models_and_adds_lightx2v(self):
        wf = render.apply_build(self.base(), {"build": "cf"})
        self.assertEqual(wf["1"]["inputs"]["unet_name"], render.BUILD_MODELS["cf"][0])
        self.assertEqual(wf["2"]["inputs"]["unet_name"], render.BUILD_MODELS["cf"][1])
        names = [n["inputs"]["lora_name"] for n in lora_nodes(wf)]
        self.assertEqual(names, [render.LIGHTX2V_HIGH, render.LIGHTX2V_LOW])
        self.assertEqual(wf["9"]["inputs"]["model"], ["15", 0])
        self.assertEqual(wf["10"]["inputs"]["model"], ["16", 0])

    def test_missing_build_defaults_to_fm(self):
        wf = render.apply_build(self.base(), {})
        self.assertEqual(wf["1"]["inputs"]["unet_name"], render.BUILD_MODELS["fm"][0])

    def test_build_and_shot_loras_compose(self):
        registry = json.loads((ROOT / "workflows" / "loras.json").read_text(encoding="utf-8"))
        wf = render.inject(
            self.base(),
            "p",
            "n",
            {
                "width": 832,
                "height": 480,
                "duration": 5,
                "build": "cf",
                "loras": ["vag_detail"],
            },
            "frame.png",
            registry=registry,
        )
        names = [n["inputs"]["lora_name"] for n in lora_nodes(wf)]
        self.assertIn(render.LIGHTX2V_HIGH, names)
        self.assertIn("vag_detail_wan22_high.safetensors", names)
        # LightX2V must come first so the shot LoRA stacks on top of it.
        self.assertEqual(names[0], render.LIGHTX2V_HIGH)


if __name__ == "__main__":
    unittest.main()
