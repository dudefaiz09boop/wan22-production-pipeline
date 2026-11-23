import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import comfy_api
import make_seeds
import preflight
import render


def response(data):
    return io.BytesIO(json.dumps(data).encode("utf-8"))


class ComfyApiTests(unittest.TestCase):
    def test_server_default_uses_comfy_url_and_falls_back_to_localhost(self):
        with mock.patch.dict(os.environ, {"COMFY_URL": " https://tunnel.example/ "}):
            self.assertEqual(comfy_api.server_default(), "https://tunnel.example")
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                comfy_api.server_default(),
                "http://127.0.0.1:8188",
            )

    def test_upload_posts_multipart_and_caches_by_resolved_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reference.png"
            path.write_bytes(b"png data")
            cache = {}
            with mock.patch.object(
                comfy_api.urllib.request,
                "urlopen",
                return_value=response({
                    "name": "reference.png",
                    "subfolder": "pipeline/test",
                    "type": "input",
                }),
            ) as urlopen:
                first = comfy_api.upload("https://tunnel.example/", path, cache)
                second = comfy_api.upload("https://tunnel.example/", path, cache)

            self.assertEqual(first, "pipeline/test/reference.png")
            self.assertEqual(second, first)
            urlopen.assert_called_once()
            request = urlopen.call_args.args[0]
            self.assertEqual(
                request.full_url,
                "https://tunnel.example/upload/image",
            )
            self.assertIn(b'name="image"', request.data)
            self.assertIn(b'filename="reference.png"', request.data)
            self.assertIn(b'name="type"\r\n\r\ninput', request.data)
            self.assertIn(b'name="overwrite"\r\n\r\ntrue', request.data)
            self.assertIn(b"png data", request.data)

    def test_render_and_seed_poll_at_five_second_intervals(self):
        for wait in (render.wait_for, make_seeds.wait):
            with self.subTest(wait=wait.__module__), mock.patch(
                f"{wait.__module__}.urllib.request.urlopen",
                side_effect=[response({}), response({"job": {"outputs": {}}})],
            ), mock.patch(f"{wait.__module__}.time.sleep") as sleep:
                self.assertEqual(wait("https://tunnel.example", "job")["outputs"], {})
                sleep.assert_called_once_with(5)

    def test_batch_resume_uses_the_preceding_saved_frame_only_for_chained_shots(self):
        with tempfile.TemporaryDirectory() as directory:
            prompt = Path(directory) / "c18.json"
            previous = Path(directory) / "c17.png"
            previous.touch()
            with mock.patch.object(render, "CHAIN_DIR", Path(directory)):
                self.assertEqual(
                    render.preceding_chained_frame(prompt, {"chain": True}),
                    previous,
                )
                self.assertIsNone(
                    render.preceding_chained_frame(prompt, {"chain": False})
                )
                previous.unlink()
                self.assertIsNone(
                    render.preceding_chained_frame(prompt, {"chain": True})
                )

    def test_missing_previous_frame_reseeds_but_failed_previous_shot_errors(self):
        prompt = Path("c14.json")
        cfg = {"chain": True, "start_frame": "startframes/c14.png"}
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            image, chained = render.select_start_image(prompt, cfg, None, None)
        self.assertEqual(image, Path("startframes/c14.png"))
        self.assertFalse(chained)
        self.assertEqual(
            output.getvalue(),
            "c14: no prev frame, reseeded from start_frame\n",
        )

        with self.assertRaisesRegex(RuntimeError, "previous shot c13 failed"):
            render.select_start_image(prompt, cfg, None, "c13")

    def test_missing_previous_file_reseeds_even_if_a_path_was_expected(self):
        prompt = Path("c14.json")
        cfg = {"chain": True, "start_frame": "startframes/c14.png"}
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "c13.png"
            with contextlib.redirect_stdout(io.StringIO()):
                image, chained = render.select_start_image(prompt, cfg, missing, None)
        self.assertEqual(image, Path("startframes/c14.png"))
        self.assertFalse(chained)

    def test_preflight_forwards_server_url_and_uses_fresh_single_clip_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            preflight.OUTPUT = Path(directory) / "preflight.json"
            completed = SimpleNamespace(stdout="done", stderr="", returncode=0)
            with mock.patch.object(
                sys,
                "argv",
                [
                    "preflight.py",
                    "--server",
                    "https://tunnel.example",
                    "c18",
                    "c19",
                ],
            ), mock.patch.object(
                preflight.subprocess,
                "run",
                return_value=completed,
            ) as run, mock.patch.object(
                preflight,
                "resolve_server",
                return_value="https://tunnel.example",
            ) as resolve_server, contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(preflight.main(), 0)

            resolve_server.assert_called_once_with("https://tunnel.example")
            commands = [call.args[0] for call in run.call_args_list]
            for clip, command in zip(("c18", "c19"), commands):
                self.assertIn("--server", command)
                self.assertIn("https://tunnel.example", command)
                self.assertIn("--no-resume", command)
                self.assertIn(clip, command)


if __name__ == "__main__":
    unittest.main()
