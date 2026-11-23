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
import validate

ROOT = Path(__file__).resolve().parents[1]

RESET_BEATS = ("c08", "c18", "c21", "c30", "c46", "c47")


def validator_shot(**updates):
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
        "noise": 2,
        "duration": 5,
        "light": "night_tube",
        "mark_state": "none",
        "thread": "",
        "profile": "draft",
        "audio_mode": "none",
        "anchors": {
            "face": "Faiz and Juveriya",
            "body": "adult standing",
            "wardrobe": "tracked wardrobe",
            "prop": "room prop",
            "light": "night tube",
            "geography": "study room",
        },
    }
    shot.update(updates)
    shot["start_frame"] = f"startframes/{shot['id']}.png"
    return shot


def run_validator(shots, root=None, extra_files=None):
    own_temp = root is None
    if own_temp:
        root = Path(tempfile.mkdtemp())
    try:
        (root / "startframes").mkdir(exist_ok=True)
        for shot in shots:
            (root / shot["start_frame"]).touch()
        (root / "negative.txt").write_text("negative", encoding="utf-8")
        for name, content in (extra_files or {}).items():
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        spec = {
            "meta": {"duration": 5, "negative": "negative.txt"},
            "story": {
                "relations": {
                    "faiz_juveriya": "siblings",
                    "on_screen": False,
                },
                "boundaries": {
                    "consent_beats": ["c01"],
                    "revocable": True,
                },
                "thread_beats": ["c09"],
                "scenes": {1: {"title": "Scene"}},
            },
            "shots": shots,
        }
        (root / "spec.yaml").write_text(
            yaml.safe_dump(spec, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        stdout, stderr = io.StringIO(), io.StringIO()
        argv = ["validate.py", "--root", str(root), "--spec", "spec.yaml", "--no-frames"]
        with mock.patch.object(sys, "argv", argv):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                result = validate.main()
        return result, stdout.getvalue(), stderr.getvalue()
    finally:
        if own_temp:
            import shutil

            shutil.rmtree(root, ignore_errors=True)


class ResetBeatsDoNotChainTests(unittest.TestCase):
    def test_production_beats_declare_all_six_chain_breaks_false(self):
        data = yaml.safe_load((ROOT / "docs" / "beats.yaml").read_text(encoding="utf-8"))
        by_id = {shot["id"]: shot for shot in data["shots"]}
        for sid in RESET_BEATS:
            self.assertIs(by_id[sid].get("chain"), False, f"{sid} must set chain: false")

    def test_validator_fails_a_reset_beat_that_chains(self):
        result, _, errors = run_validator(
            [validator_shot(id="c08", chain=True)],
            extra_files={"workflows/loras.json": json.dumps({})},
        )
        self.assertEqual(result, 1)
        self.assertIn("c08 is a declared identity reset and must set chain: false", errors)

    def test_validator_accepts_a_reset_beat_with_chain_false(self):
        # A lone c08 at index 1 also trips the id-order and start_frame checks;
        # assert the chain rule itself stays silent.
        result, _, errors = run_validator(
            [validator_shot(id="c08", chain=False)],
            extra_files={"workflows/loras.json": json.dumps({})},
        )
        self.assertNotIn("declared identity reset", errors)
        self.assertIn("id 'c08' != 'c01' (order broken)", errors)

    def test_validator_enforces_lora_slot_budget(self):
        registry = json.dumps({
            "penis_insert": {"high": "loras/a.safetensors", "low": "loras/b.safetensors", "w": 0.9, "trigger": None, "trigger_scope": None},
            "vag_detail": {"high": "loras/c.safetensors", "low": "loras/d.safetensors", "w": 0.75, "trigger": None, "trigger_scope": None},
            "oral_insert": {"high": "loras/e.safetensors", "low": "loras/f.safetensors", "w": 0.85, "trigger": None, "trigger_scope": None},
            "orbit": {"high": "loras/g.safetensors", "low": "loras/h.safetensors", "w": 1.0, "trigger": None, "trigger_scope": None},
        })
        result, _, errors = run_validator(
            [validator_shot(loras=["penis_insert", "vag_detail", "oral_insert", "orbit"])],
            extra_files={"workflows/loras.json": registry},
        )
        self.assertEqual(result, 1)
        self.assertIn("declares 4 loras (max 3)", errors)

    def test_validator_rejects_conflicting_lora_pairs(self):
        registry = json.dumps({
            "nsfw_helper": {
                "high": "loras/n.safetensors", "low": None, "w": 0.45,
                "trigger": "nsfwsks", "trigger_scope": "prompt_prefix",
                "conflicts": ["vag_detail"],
            },
            "vag_detail": {"high": "loras/v.safetensors", "low": "loras/v2.safetensors", "w": 0.75, "trigger": None, "trigger_scope": None},
        })
        result, _, errors = run_validator(
            [validator_shot(loras=["nsfw_helper", "vag_detail"])],
            extra_files={"workflows/loras.json": registry},
        )
        self.assertEqual(result, 1)
        self.assertIn("conflicting loras 'nsfw_helper' and 'vag_detail'", errors)

    def test_validator_requires_trigger_word_in_compiled_prompt(self):
        good = json.dumps({
            "orbit": {
                "high": "loras/o.safetensors", "low": None, "w": 1.0,
                "trigger": "orbit 360", "trigger_scope": "prompt_prefix",
            },
        })
        result, _, errors = run_validator(
            [validator_shot(loras=["orbit"])],
            extra_files={"workflows/loras.json": good},
        )
        self.assertEqual(result, 0, errors)

        # A registry trigger without a usable scope never reaches the compiled
        # prompt, so the shot must fail the trigger rule.
        broken = json.dumps({
            "orbit": {
                "high": "loras/o.safetensors", "low": None, "w": 1.0,
                "trigger": "orbit 360", "trigger_scope": None,
            },
        })
        result, _, errors = run_validator(
            [validator_shot(loras=["orbit"])],
            extra_files={"workflows/loras.json": broken},
        )
        self.assertEqual(result, 1)
        self.assertIn("trigger 'orbit 360' missing from compiled prompt", errors)


if __name__ == "__main__":
    unittest.main()
