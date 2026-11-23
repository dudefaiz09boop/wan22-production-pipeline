from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import inpaint_pass
import render_continuity


def tiny_png(width: int = 832, height: int = 480) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (0, 0, 255)).save(buffer, format="PNG")
    return buffer.getvalue()


class ParseRegionsTests(unittest.TestCase):
    def test_parses_label_and_integer_box(self):
        self.assertEqual(
            inpaint_pass.parse_regions(["tail:880,420,360,280"]),
            [{"label": "tail", "x": 880, "y": 420, "width": 360, "height": 280}],
        )

    def test_rejects_malformed_regions(self):
        with self.assertRaisesRegex(ValueError, "must look like"):
            inpaint_pass.parse_regions(["880,420"])
        with self.assertRaisesRegex(ValueError, "integer"):
            inpaint_pass.parse_regions(["tail:880,420,wide,280"])


class InpaintPassStageTests(unittest.TestCase):
    def tearDown(self):
        render_continuity.ROOT = Path(__file__).resolve().parents[1]

    def prepare_segment(self, root: Path, beat_id: str = "c26") -> Path:
        segment = root / "renders" / "segments" / f"{beat_id}.mp4"
        segment.parent.mkdir(parents=True, exist_ok=True)
        segment.write_bytes(b"fake video")
        return segment

    def run_main(self, root: Path, extra: list[str]) -> tuple[int, str]:
        stdout = io.StringIO()
        calls = []

        def fake_run_media(command, timeout=600):
            calls.append(list(command))
            Path(command[-1]).write_bytes(tiny_png())

        with mock.patch.object(render_continuity, "run_media", fake_run_media):
            with contextlib.redirect_stdout(stdout):
                code = inpaint_pass.main([
                    "--root", str(root),
                    "--segments-dir", str(root / "renders" / "segments"),
                    *extra,
                ])
        return code, stdout.getvalue()

    def test_stage_only_writes_manifest_and_final_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            self.prepare_segment(root)

            code, output = self.run_main(root, [
                "c26", "--stage-only", "--region", "tail:880,420,360,280",
            ])

            self.assertEqual(code, 0)
            manifest_path = root / "renders" / "inpaint" / "c26" / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["beat_id"], "c26")
            self.assertEqual(manifest["status"], "staged")
            self.assertEqual(len(manifest["frames"]), 1)
            self.assertEqual(manifest["frames"][0]["time_seconds"], None)
            self.assertEqual(manifest["regions"][0]["label"], "tail")
            frame_path = root / manifest["frames"][0]["path"]
            self.assertTrue(frame_path.is_file())
            self.assertIn("staged with 1 frame(s)", output)

    def test_explicit_times_extract_named_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            self.prepare_segment(root)

            code, _ = self.run_main(root, ["c26", "--stage-only", "--at", "3.2,4.8"])

            self.assertEqual(code, 0)
            out_dir = root / "renders" / "inpaint" / "c26"
            manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                [Path(frame["path"]).name for frame in manifest["frames"]],
                ["c26_t3p2s.png", "c26_t4p8s.png"],
            )

    def test_missing_segment_fails_that_beat(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            code, output = self.run_main(root, ["c31", "--stage-only"])
        self.assertEqual(code, 1)
        self.assertIn("no rendered segment", output)


class InpaintPassSubmitTests(unittest.TestCase):
    def tearDown(self):
        render_continuity.ROOT = Path(__file__).resolve().parents[1]

    def prepare_segment(self, root: Path, beat_id: str = "c28") -> Path:
        segment = root / "renders" / "segments" / f"{beat_id}.mp4"
        segment.parent.mkdir(parents=True, exist_ok=True)
        segment.write_bytes(b"fake video")
        return segment

    def test_submit_posts_still_inpaint_jobs_with_reference_frames(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            self.prepare_segment(root)
            posted = []

            def fake_api_json(api, api_key, route, data=None):
                if route == "/v1/workers/active":
                    return {"data": [
                        {"kind": "image", "status": "idle", "id": "img1"},
                        {"kind": "video", "status": "idle", "id": "vid1"},
                    ]}
                posted.append((route, data))
                return {"id": f"job-{len(posted)}", "poll_url": f"/v1/tasks/job-{len(posted)}"}

            stdout = io.StringIO()
            with mock.patch.object(
                render_continuity,
                "run_media",
                lambda command, timeout=600: Path(command[-1]).write_bytes(tiny_png()),
            ), mock.patch.object(render_continuity, "api_json", fake_api_json):
                with mock.patch.dict(
                    os.environ,
                    {"API_KEY": "k" * 40, "STUDIO_API_URL": "http://127.0.0.1:3000"},
                ):
                    with contextlib.redirect_stdout(stdout):
                        code = inpaint_pass.main([
                            "c28",
                            "--root", str(root),
                            "--segments-dir", str(root / "renders" / "segments"),
                            "--prompt", "repair the thigh region, keep identity",
                            "--region", "thigh:900,430,340,260",
                        ])

            self.assertEqual(code, 0)
            self.assertEqual(len(posted), 1)
            route, payload = posted[0]
            self.assertEqual(route, "/v1/workers/img1/jobs")
            self.assertTrue(payload["inpaint"])
            self.assertTrue(payload["reference_base64"])
            self.assertEqual(payload["width"], 832)
            manifest = json.loads(
                (root / "renders" / "inpaint" / "c28" / "manifest.json")
                .read_text(encoding="utf-8"),
            )
            self.assertEqual(manifest["status"], "submitted")
            self.assertEqual(manifest["jobs"][0]["task_id"], "job-1")
            # The API key is never printed.
            self.assertNotIn("k" * 40, stdout.getvalue())

    def test_gateway_submission_requires_a_prompt(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            self.prepare_segment(root)
            with mock.patch.dict(os.environ, {"API_KEY": "k" * 40}, clear=True):
                with self.assertRaises(SystemExit):
                    inpaint_pass.main([
                        "c28",
                        "--root", str(root),
                        "--gateway", "http://127.0.0.1:3000",
                    ])

    def test_gateway_must_be_loopback(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            self.prepare_segment(root)
            with self.assertRaisesRegex(RuntimeError, "loopback HTTP origin"):
                inpaint_pass.main([
                    "c28",
                    "--root", str(root),
                    "--gateway", "https://example.com",
                    "--prompt", "repair",
                ])


if __name__ == "__main__":
    unittest.main()
