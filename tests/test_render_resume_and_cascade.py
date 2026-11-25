"""A failed clip must not cascade, and a resumed run must not re-render.

Both behaviours are about GPU time: the cascade turned one bad clip into 40
"Chain broken by cXX" failures, and the missing resume re-rendered every clip
that had already finished.
"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_production


def clip(clip_id, chain=True):
    return {"id": clip_id, "chain": chain}


def make_args(root, renders):
    return SimpleNamespace(
        prompts=str(root / "prompts"),
        renders=str(renders),
        output=str(root / "output"),
    )


class RenderClipsStageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / "prompts").mkdir()
        self.renders = self.root / "renders"
        self.renders.mkdir()
        self.args = make_args(self.root, self.renders)

    def tearDown(self):
        self.temp.cleanup()

    def write_prompt(self, clip_id):
        (self.root / "prompts" / f"{clip_id}.txt").write_text("body", encoding="utf-8")
        (self.root / "prompts" / f"{clip_id}.json").write_text(
            json.dumps({"id": clip_id, "chain": True, "start_frame": f"startframes/{clip_id}.png"}),
            encoding="utf-8",
        )

    def run_stage(self, clips, render_side_effect):
        state = SimpleNamespace(clips=clips, results=[], total_gpu_seconds=0.0)
        with mock.patch.object(run_production, "render_clip", side_effect=render_side_effect), \
             mock.patch.object(run_production, "select_start_image",
                               return_value=(Path("startframes/x.png"), False)), \
             contextlib.redirect_stdout(io.StringIO()):
            run_production.render_clips_stage(state, self.args)
        return state

    def test_a_failed_clip_does_not_break_the_rest_of_the_chain(self):
        clips = [clip("c01"), clip("c02"), clip("c03")]
        for c in clips:
            self.write_prompt(c["id"])

        def fake_render(state, cfg, prompt, prev, index, total):
            if cfg["id"] == "c01":
                return run_production.ClipResult("c01", False, 1.0, error="HTTP Error 400")
            return run_production.ClipResult(cfg["id"], True, 1.0, Path(f"renders/{cfg['id']}.mp4"))

        state = self.run_stage(clips, fake_render)
        by_id = {r.clip_id: r for r in state.results}
        self.assertFalse(by_id["c01"].success)
        # The old code marked these "Chain broken by c01" without ever trying.
        self.assertTrue(by_id["c02"].success, by_id["c02"].error)
        self.assertTrue(by_id["c03"].success, by_id["c03"].error)

    def test_already_rendered_clips_are_skipped_on_resume(self):
        clips = [clip("c01"), clip("c02")]
        for c in clips:
            self.write_prompt(c["id"])
        (self.renders / "c01.mp4").write_bytes(b"done")

        rendered = []

        def fake_render(state, cfg, prompt, prev, index, total):
            rendered.append(cfg["id"])
            return run_production.ClipResult(cfg["id"], True, 1.0, Path("renders/x.mp4"))

        state = self.run_stage(clips, fake_render)
        self.assertEqual(rendered, ["c02"])
        self.assertTrue(all(r.success for r in state.results))


if __name__ == "__main__":
    unittest.main()
