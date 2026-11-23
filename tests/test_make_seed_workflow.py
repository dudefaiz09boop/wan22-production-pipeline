import contextlib
import io
import json
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock
from urllib.request import Request, urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_seed_workflow
import make_seeds


OBJECT_INFO = {
    "CheckpointLoaderSimple": {
        "name": "CheckpointLoaderSimple",
        "input": {
            "required": {
                "weights_file": ["COMBO", {"options": ["seed.safetensors"]}],
            }
        },
        "output": ["MODEL", "CLIP", "VAE"],
        "output_name": ["MODEL", "CLIP", "VAE"],
    },
    "IPAdapterUnifiedLoader": {
        "name": "IPAdapterUnifiedLoader",
        "input": {
            "required": {
                "base_model": ["MODEL", {}],
                "preset": ["COMBO", {"options": ["PLUS (high strength)"]}],
            }
        },
        "output": ["MODEL", "IPADAPTER"],
        "output_name": ["MODEL", "IPADAPTER"],
    },
    "IPAdapterAdvanced": {
        "name": "IPAdapterAdvanced",
        "input": {
            "required": {
                "base_model": ["MODEL", {}],
                "adapter_model": ["IPADAPTER", {}],
                "reference_image": ["IMAGE", {}],
                "weight": ["FLOAT", {"default": 0.7}],
                "weight_type": ["COMBO", {"options": ["linear"]}],
            }
        },
        "output": ["MODEL"],
        "output_name": ["MODEL"],
    },
    "LoadImage": {
        "name": "LoadImage",
        "input": {
            "required": {
                "reference_file": [
                    "COMBO",
                    {
                        "options": ["previous.png"],
                        "image_upload": True,
                        "image_folder": "input",
                    },
                ],
            }
        },
        "output": ["IMAGE"],
        "output_name": ["IMAGE"],
    },
    "CLIPTextEncode": {
        "name": "CLIPTextEncode",
        "input": {
            "required": {
                "prompt_text": ["STRING", {"default": ""}],
                "encoder": ["CLIP", {}],
            }
        },
        "output": ["CONDITIONING"],
        "output_name": ["CONDITIONING"],
    },
    "KSampler": {
        "name": "KSampler",
        "input": {
            "required": {
                "model_input": ["MODEL", {}],
                "positive_conditioning": ["CONDITIONING", {}],
                "negative_conditioning": ["CONDITIONING", {}],
                "latent_input": ["LATENT", {}],
                "random_seed": ["INT", {"default": 1}],
                "step_count": ["INT", {"default": 20}],
                "guidance": ["FLOAT", {"default": 7.0}],
                "sampler": ["COMBO", {"options": ["euler"]}],
                "schedule": ["COMBO", {"options": ["normal"]}],
                "denoise": ["FLOAT", {"default": 1.0}],
            }
        },
        "output": ["LATENT"],
        "output_name": ["LATENT"],
    },
    "EmptyLatentImage": {
        "name": "EmptyLatentImage",
        "input": {
            "required": {
                "target_width": ["INT", {"default": 512}],
                "target_height": ["INT", {"default": 512}],
                "batch_size": ["INT", {"default": 1}],
            }
        },
        "output": ["LATENT"],
        "output_name": ["LATENT"],
    },
    "VAEDecode": {
        "name": "VAEDecode",
        "input": {
            "required": {
                "latent_samples": ["LATENT", {}],
                "vae_model": ["VAE", {}],
            }
        },
        "output": ["IMAGE"],
        "output_name": ["IMAGE"],
    },
    "SaveImage": {
        "name": "SaveImage",
        "input": {
            "required": {
                "image_batch": ["IMAGE", {}],
                "filename_prefix": ["STRING", {"default": "ComfyUI"}],
            }
        },
        "output": [],
        "output_name": [],
    },
}


class APIHandler(BaseHTTPRequestHandler):
    seen_prompts = []

    def do_GET(self):
        if self.path != "/object_info":
            self.send_error(404)
            return
        body = json.dumps(OBJECT_INFO).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path != "/prompt":
            self.send_error(404)
            return
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.seen_prompts.append(payload["prompt"])
        body = b'{"prompt_id":"dry-run"}'
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        pass


class MakeSeedWorkflowTests(unittest.TestCase):
    def test_graph_uses_reported_input_names_and_requested_connections(self):
        graph = make_seed_workflow.build_workflow(OBJECT_INFO)
        self.assertEqual(graph["1"]["inputs"]["weights_file"], "seed.safetensors")
        self.assertEqual(graph["3"]["_meta"]["title"], "__ref__")
        self.assertEqual(graph["4"]["_meta"]["title"], "__todo__")
        self.assertEqual(graph["3"]["_meta"]["ref_input"], "reference_file")
        self.assertEqual(graph["4"]["_meta"]["text_input"], "prompt_text")
        self.assertEqual(
            graph["6"]["inputs"]["base_model"],
            ["1", 0],
        )
        self.assertEqual(
            graph["6"]["inputs"]["adapter_model"],
            ["2", 1],
        )
        self.assertEqual(
            graph["8"]["inputs"]["model_input"],
            ["6", 0],
        )
        self.assertEqual(
            graph["8"]["inputs"]["positive_conditioning"],
            ["4", 0],
        )
        self.assertEqual(
            graph["10"]["inputs"]["image_batch"],
            ["9", 0],
        )

    def test_missing_ipadapter_apply_prints_install_instructions_and_exits_two(self):
        info = {
            name: value
            for name, value in OBJECT_INFO.items()
            if name != "IPAdapterAdvanced"
        }
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as result:
                make_seed_workflow.resolve_classes(info)
        self.assertEqual(result.exception.code, 2)
        self.assertIn(
            "install ComfyUI-IPAdapter-plus in the notebook, restart ComfyUI, re-run",
            stderr.getvalue(),
        )
        self.assertIn("near-misses:", stderr.getvalue())

    def test_seed_filler_uses_generated_schema_metadata(self):
        graph = make_seed_workflow.build_workflow(OBJECT_INFO)
        shot = {
            "id": "c01",
            "cam": ["(At 0 seconds: static, subject, action, lighting)"],
        }
        graph = make_seeds.fill(
            graph,
            shot,
            {"res_base": [832, 480]},
            "negative prompt",
            ["uploaded-reference.png"],
        )

        self.assertEqual(
            graph["3"]["inputs"]["reference_file"], "uploaded-reference.png"
        )
        self.assertEqual(
            graph["4"]["inputs"]["prompt_text"],
            "static, subject, action, lighting)",
        )
        self.assertEqual(graph["5"]["inputs"]["prompt_text"], "negative prompt")
        self.assertEqual(graph["7"]["inputs"]["target_width"], 832)
        self.assertEqual(graph["7"]["inputs"]["target_height"], 480)
        self.assertIn("random_seed", graph["8"]["inputs"])

    def test_dry_run_writes_graph_that_loads_at_prompt_endpoint(self):
        APIHandler.seen_prompts = []
        server = ThreadingHTTPServer(("127.0.0.1", 0), APIHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{server.server_port}"
        try:
            with tempfile.TemporaryDirectory() as directory:
                output = Path(directory) / "seed_t2i.json"
                argv = [
                    "make_seed_workflow.py",
                    "--server",
                    base,
                    "--out",
                    str(output),
                    "--dry-run",
                ]
                with mock.patch.object(sys, "argv", argv), mock.patch.object(
                    make_seed_workflow,
                    "resolve_server",
                    return_value=base,
                ), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(make_seed_workflow.main(), 0)

                graph = json.loads(output.read_text(encoding="utf-8"))
                request = Request(
                    f"{base}/prompt",
                    data=json.dumps({"prompt": graph}).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                with urlopen(request, timeout=3) as response:
                    self.assertEqual(response.status, 200)
                self.assertEqual(APIHandler.seen_prompts, [graph])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)


if __name__ == "__main__":
    unittest.main()
