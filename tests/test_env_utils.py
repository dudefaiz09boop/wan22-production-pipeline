import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import env_utils


class FakeResponse:
    def __init__(self, status, payload):
        self.status = status
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def getcode(self):
        return self.status

    def read(self):
        return json.dumps(self.payload).encode("utf-8")


class EnvUtilsTests(unittest.TestCase):
    def test_probe_requires_http_200_and_a_named_device(self):
        with mock.patch.object(
            env_utils.urllib.request,
            "urlopen",
            return_value=FakeResponse(200, {"devices": [{"name": "GPU 0"}]}),
        ) as urlopen:
            self.assertEqual(env_utils.probe("https://tunnel.example/"), "GPU 0")
            self.assertEqual(urlopen.call_args.kwargs["timeout"], 6)

        with mock.patch.object(
            env_utils.urllib.request,
            "urlopen",
            return_value=FakeResponse(200, {"devices": []}),
        ):
            self.assertIsNone(env_utils.probe("https://tunnel.example"))

        with mock.patch.object(
            env_utils.urllib.request,
            "urlopen",
            return_value=FakeResponse(503, {"devices": [{"name": "GPU 0"}]}),
        ):
            self.assertIsNone(env_utils.probe("https://tunnel.example"))

    def test_resolve_server_uses_explicit_before_environment_and_dotenv(self):
        with (
            mock.patch.object(env_utils, "probe", side_effect=[None, "GPU"]),
            mock.patch.object(env_utils, "ROOT", Path("unused")),
            mock.patch.dict(os.environ, {"COMFY_URL": "https://env.example"}),
            contextlib.redirect_stdout(io.StringIO()) as output,
        ):
            server = env_utils.resolve_server("https://explicit.example")
        self.assertEqual(server, "https://env.example")
        self.assertIn("$env:COMFY_URL", output.getvalue())

    def test_resolve_server_reads_dotenv_before_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text(
                "OTHER=value\nCOMFY_URL='https://dotenv.example/' # comment\n",
                encoding="utf-8",
            )
            with (
                mock.patch.object(env_utils, "ROOT", root),
                mock.patch.dict(os.environ, {"COMFY_URL": ""}),
                mock.patch.object(env_utils, "probe", return_value="GPU"),
                contextlib.redirect_stdout(io.StringIO()) as output,
            ):
                server = env_utils.resolve_server(None)
        self.assertEqual(server, "https://dotenv.example")
        self.assertIn(".env COMFY_URL", output.getvalue())

    def test_resolve_server_exits_with_tunnel_instructions_when_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / ".env").write_text("", encoding="utf-8")
            with (
                mock.patch.object(env_utils, "ROOT", root),
                mock.patch.dict(os.environ, {"COMFY_URL": ""}),
                mock.patch.object(env_utils, "probe", return_value=None),
            ):
                with self.assertRaisesRegex(SystemExit, "Start the Kaggle ComfyUI notebook"):
                    env_utils.resolve_server(None)


if __name__ == "__main__":
    unittest.main()
