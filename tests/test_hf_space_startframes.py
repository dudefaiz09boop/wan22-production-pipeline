import io
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import hf_space_startframes


class HfSpaceStartframesTests(unittest.TestCase):
    def test_generate_call_uses_published_gradio_route_and_input_order(self):
        self.assertEqual(
            hf_space_startframes.CALL_URL,
            f"{hf_space_startframes.SPACE_BASE}/gradio_api/call/generate",
        )
        values = {
            "prompt": "prompt",
            "mode": "mode",
            "ref_image_1": "image one",
            "ref_image_2": "image two",
            "lora_adapter": "adapter",
            "lora_strength": 0.9,
            "custom_repo": "repo",
            "custom_file": "file",
            "aspect_ratio": "aspect",
            "steps": 28,
            "seed": 42,
            "randomize_seed": False,
        }

        self.assertEqual(
            hf_space_startframes.build_space_call_payload(values),
            {"data": list(values.values())},
        )

    def test_http_errors_include_bounded_response_detail_and_redact_token(self):
        token = "hf_test_secret"
        error = urllib.error.HTTPError(
            "https://space.example.test",
            503,
            "Service Unavailable",
            {},
            io.BytesIO(f"proxy rejected {token}".encode()),
        )
        self.addCleanup(error.close)

        with mock.patch.object(
            hf_space_startframes.urllib.request,
            "urlopen",
            side_effect=error,
        ):
            with self.assertRaisesRegex(RuntimeError, "HTTP 503: proxy rejected") as caught:
                hf_space_startframes.http_json(
                    hf_space_startframes.CALL_URL,
                    {"data": []},
                    token,
                )

        self.assertNotIn(token, str(caught.exception))


if __name__ == "__main__":
    unittest.main()
