import contextlib
import io
import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_prompts
import render

ROOT = Path(__file__).resolve().parents[1]


def make_shot(**updates):
    shot = {
        "id": "c01",
        "scene": 1,
        "build": "fm",
        "start_frame": "startframes/c01.png",
        "loras": [],
        "chain": False,
        "cam": [
            "static wide: the room",
            "zoom in: the bed",
            "dolly in: the pair",
            "pan left: the wall",
            "tilt up: the lamp",
            "orbit: the pair again",
        ],
    }
    shot.update(updates)
    return shot


META = {
    "res_base": [832, 480],
    "res_face": [1280, 720],
    "sampler": "euler",
    "steps": 5,
    "cfg": 1,
    "duration": 5,
}


class ResolutionPropagatesTests(unittest.TestCase):
    def test_fm_build_uses_base_resolution_and_meta_settings(self):
        plan = build_prompts.build_plan(make_shot(), META, "negative.txt", {}, [])
        self.assertIsNotNone(plan)
        self.assertEqual(plan["cfg"]["width"], 832)
        self.assertEqual(plan["cfg"]["height"], 480)
        self.assertEqual(plan["cfg"]["steps"], 5)
        self.assertEqual(plan["cfg"]["cfg"], 1)
        self.assertEqual(plan["cfg"]["sampler"], "euler")
        self.assertEqual(plan["cfg"]["duration"], 5)

    def test_cf_build_uses_face_resolution(self):
        plan = build_prompts.build_plan(make_shot(build="cf"), META, "negative.txt", {}, [])
        self.assertIsNotNone(plan)
        self.assertEqual(plan["cfg"]["width"], 1280)
        self.assertEqual(plan["cfg"]["height"], 720)

    def test_render_inject_propagates_dimensions_and_length_into_graph(self):
        wf = json.loads((ROOT / "workflows" / "base_i2v.json").read_text(encoding="utf-8"))
        cfg = {
            "width": 1280,
            "height": 720,
            "duration": 5,
            "loras": [],
        }
        out = render.inject(wf, "prompt body", "negative body", cfg, "frame.png", registry={})
        node = out["8"]
        self.assertEqual(node["inputs"]["width"], 1280)
        self.assertEqual(node["inputs"]["height"], 720)
        # WanImageToVideo requires (length - 1) % 4 == 0. At 16 fps a 5s clip
        # is 81 frames; 80 is rejected by the node's step=4 constraint.
        self.assertEqual(node["inputs"]["length"], 81)
        self.assertEqual((node["inputs"]["length"] - 1) % 4, 0)
        self.assertEqual(out["6"]["inputs"]["text"], "prompt body")
        self.assertEqual(out["7"]["inputs"]["text"], "negative body")
        self.assertEqual(out["5"]["inputs"]["image"], "frame.png")


if __name__ == "__main__":
    unittest.main()
