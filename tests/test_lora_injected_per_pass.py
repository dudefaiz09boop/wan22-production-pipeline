import contextlib
import io
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import render

ROOT = Path(__file__).resolve().parents[1]


def base_workflow():
    return json.loads((ROOT / "workflows" / "base_i2v.json").read_text(encoding="utf-8"))


def real_registry():
    return json.loads((ROOT / "workflows" / "loras.json").read_text(encoding="utf-8"))


def lora_nodes(wf):
    return {
        node_id: node
        for node_id, node in wf.items()
        if isinstance(node, dict) and node.get("class_type") == "LoraLoaderModelOnly"
    }


class LoraInjectedPerPassTests(unittest.TestCase):
    def test_both_stages_get_one_loader_per_declared_lora(self):
        wf = render.inject(
            base_workflow(),
            "p",
            "n",
            {"width": 832, "height": 480, "duration": 5, "loras": ["penis_insert", "vag_detail"]},
            "frame.png",
            registry=real_registry(),
        )
        nodes = lora_nodes(wf)
        self.assertEqual(len(nodes), 4)
        self.assertEqual(nodes["13"]["inputs"]["lora_name"], "penis_insert_wan22_high.safetensors")
        self.assertEqual(nodes["13"]["inputs"]["strength_model"], 0.9)
        self.assertEqual(nodes["13"]["inputs"]["model"], ["1", 0])
        self.assertEqual(nodes["14"]["inputs"]["lora_name"], "penis_insert_wan22_low.safetensors")
        self.assertEqual(nodes["14"]["inputs"]["model"], ["2", 0])
        self.assertEqual(nodes["15"]["inputs"]["lora_name"], "vag_detail_wan22_high.safetensors")
        self.assertEqual(nodes["15"]["inputs"]["strength_model"], 0.75)
        self.assertEqual(nodes["15"]["inputs"]["model"], ["13", 0])
        self.assertEqual(nodes["16"]["inputs"]["lora_name"], "vag_detail_wan22_low.safetensors")
        self.assertEqual(nodes["16"]["inputs"]["model"], ["14", 0])
        self.assertEqual(wf["9"]["inputs"]["model"], ["15", 0])
        self.assertEqual(wf["10"]["inputs"]["model"], ["16", 0])

    def test_null_slots_skip_only_that_pass(self):
        registry = {
            "pov_finish": {
                "high": None,
                "low": "loras/pov_finish_wan22_low.safetensors",
                "w": 0.8,
                "trigger": None,
                "trigger_scope": None,
            },
        }
        wf = render.inject(
            base_workflow(),
            "p",
            "n",
            {"width": 832, "height": 480, "duration": 5, "loras": ["pov_finish"]},
            "frame.png",
            registry=registry,
        )
        nodes = lora_nodes(wf)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(nodes["13"]["inputs"]["lora_name"], "pov_finish_wan22_low.safetensors")
        self.assertEqual(nodes["13"]["inputs"]["strength_model"], 0.8)
        self.assertEqual(nodes["13"]["inputs"]["model"], ["2", 0])
        self.assertEqual(wf["9"]["inputs"]["model"], ["1", 0])
        self.assertEqual(wf["10"]["inputs"]["model"], ["13", 0])

    def test_missing_lora_file_warns_loudly_but_still_injects(self):
        registry = {
            "orbit": {
                "high": "loras/definitely_missing_here.safetensors",
                "low": None,
                "w": 1.0,
                "trigger": "orbit 360",
                "trigger_scope": "prompt_prefix",
            },
        }
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            wf = render.inject(
                base_workflow(),
                "p",
                "n",
                {"width": 832, "height": 480, "duration": 5, "loras": ["orbit"]},
                "frame.png",
                registry=registry,
            )
        output = stderr.getvalue()
        self.assertIn("WARNING", output)
        self.assertIn("definitely_missing_here.safetensors", output)
        nodes = lora_nodes(wf)
        self.assertEqual(len(nodes), 1)
        self.assertEqual(wf["9"]["inputs"]["model"], ["13", 0])

    def test_unknown_lora_key_warns_and_never_silent_skips(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            wf = render.inject(
                base_workflow(),
                "p",
                "n",
                {"width": 832, "height": 480, "duration": 5, "loras": ["nope_lora"]},
                "frame.png",
                registry=real_registry(),
            )
        self.assertIn("not in workflows/loras.json", stderr.getvalue())
        self.assertEqual(lora_nodes(wf), {})

    def test_no_loras_leaves_graph_untouched(self):
        wf = render.inject(
            base_workflow(),
            "p",
            "n",
            {"width": 832, "height": 480, "duration": 5, "loras": []},
            "frame.png",
            registry=real_registry(),
        )
        self.assertEqual(lora_nodes(wf), {})
        self.assertEqual(wf["9"]["inputs"]["model"], ["1", 0])
        self.assertEqual(wf["10"]["inputs"]["model"], ["2", 0])


if __name__ == "__main__":
    unittest.main()
