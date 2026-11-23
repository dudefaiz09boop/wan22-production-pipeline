import sys
import base64
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import submit_startframes


class SubmitStartframesTests(unittest.TestCase):
    def test_job_prompt_uses_authored_camera_content_without_references(self):
        request = submit_startframes.build_job_request(
            {
                "id": "c01",
                "cam": ["static wide: Juveriya waits in the hallway"],
            },
            {"width": 832, "height": 480},
            "negative prompt",
        )

        self.assertEqual(request["prompt"], "Juveriya waits in the hallway")
        self.assertEqual(request["negative_prompt"], "negative prompt")
        self.assertEqual((request["width"], request["height"]), (832, 480))
        self.assertNotIn("reference_images", request)
        self.assertNotIn("steps", request)
        self.assertNotIn("cfg", request)

    def test_qwen_space_request_preserves_prompt_mode_references_and_lora(self):
        prompt = "Detailed character and scene prompt with exact composition."
        request = submit_startframes.build_qwen_space_request(
            {
                "id": "c01",
                "cam": ["static wide: camera prompt"],
            },
            {"width": 832, "height": 480},
            {
                "prompt": prompt,
                "mode": "Transform & Swap (2 Refs)",
                "refs": ["refs/02.png", "refs/04.png"],
                "lora_adapter": "NSFW (general)",
                "lora_strength": 1.0,
                "steps": 28,
                "seed": 1735432460,
                "negative_prompt": "base negative, no text",
            },
            "negative prompt",
        )

        self.assertEqual(request["prompt"], prompt)
        self.assertEqual(request["negative_prompt"], "base negative, no text")
        self.assertEqual(request["mode"], "Transform & Swap (2 Refs)")
        self.assertEqual(request["reference_plates"], ["refs/02.png", "refs/04.png"])
        self.assertEqual(request["lora_adapter"], "NSFW (general)")
        self.assertEqual(request["lora_strength"], 1.0)
        self.assertEqual(request["steps"], 28)
        self.assertEqual(request["seed"], 1735432460)
        self.assertEqual((request["width"], request["height"]), (832, 480))

        fallback = submit_startframes.build_qwen_space_request(
            {"id": "c02", "cam": ["static wide: camera prompt"]},
            {"width": 832, "height": 480},
            {},
            "negative prompt",
        )
        self.assertEqual(
            fallback["prompt"],
            "camera prompt" + submit_startframes.QWEN_SPACE_PROMPT_SUFFIX,
        )
        self.assertEqual(fallback["negative_prompt"], "negative prompt")

    def test_qwen_space_request_rejects_out_of_range_seed(self):
        with self.assertRaisesRegex(RuntimeError, "seed must be from 0 to"):
            submit_startframes.build_qwen_space_request(
                {"id": "c01", "cam": ["static: camera prompt"]},
                {"width": 832, "height": 480},
                {"seed": 2147483648},
                "negative prompt",
            )

    def test_qwen_space_request_rejects_mismatched_reference_count(self):
        with self.assertRaisesRegex(RuntimeError, "invalid number of reference images"):
            submit_startframes.build_qwen_space_request(
                {"id": "c01", "cam": ["static wide: camera prompt"]},
                {"width": 832, "height": 480},
                {"mode": "Edit Image (1 Ref)", "refs": []},
                "negative prompt",
            )

    def test_prefetch_queues_the_next_frame_before_the_first_finishes(self):
        shot_ids = ["c01", "c02"]
        shots = [
            {"id": shot_id, "cam": ["static wide: camera prompt"]}
            for shot_id in shot_ids
        ]
        configs = {shot_id: {"width": 832, "height": 480} for shot_id in shot_ids}
        manifest = {
            "batch_status": "generating",
            "selected_ids": shot_ids,
            "jobs": {
                shot_id: {
                    "job_id": None,
                    "status": "not_submitted",
                    "attempts": 0,
                    "attempts_detail": [],
                    "terminal_status": None,
                    "start_frame": f"startframes/{shot_id}.png",
                }
                for shot_id in shot_ids
            },
        }
        calls = []
        tasks = {}

        def fake_api(api, api_key, path, data=None):
            calls.append(("GET" if data is None else "POST", path))
            if path == "/v1/workers/worker-1" and data is None:
                return {
                    "kind": "image",
                    "model": submit_startframes.QWEN_CHECKPOINT,
                    "status": "idle",
                }
            if path == "/v1/workers/worker-1/jobs":
                job_id = f"task-{len(tasks) + 1}"
                tasks[job_id] = {"status": "completed"}
                return {"id": job_id, "status": "queued"}
            if path.startswith("/v1/tasks/"):
                return tasks[path.rsplit("/", 1)[1]]
            raise AssertionError(f"unexpected gateway call {path}")

        def fake_collect(api, api_key, shot_id, entry, config, *, replace_existing):
            entry["status"] = "completed"
            entry["terminal_status"] = "succeeded"
            entry["output_sha256"] = "frame-hash"
            entry["output_dimensions"] = [832, 480]
            return 832, 480

        with mock.patch.object(
            submit_startframes,
            "load_production_inputs",
            return_value=(shots, configs, "negative prompt"),
        ), mock.patch.object(
            submit_startframes,
            "load_current_manifest",
            return_value={"jobs": {}},
        ), mock.patch.object(
            submit_startframes,
            "prepare_batch_manifest",
            return_value=manifest,
        ), mock.patch.object(
            submit_startframes,
            "read_model_store_url",
            return_value="https://models.example.test",
        ), mock.patch.object(
            submit_startframes,
            "check_model_store",
            return_value=True,
        ), mock.patch.object(
            submit_startframes,
            "api_json",
            side_effect=fake_api,
        ), mock.patch.object(
            submit_startframes.collect_startframes,
            "read_api_key",
            return_value="test-key",
        ), mock.patch.object(
            submit_startframes.collect_startframes,
            "collect_completed_frame",
            side_effect=fake_collect,
        ), mock.patch.object(
            submit_startframes.collect_startframes,
            "write_manifest",
        ), mock.patch.object(
            submit_startframes,
            "sync_current_frame_manifest",
        ):
            result = submit_startframes.main([
                "--worker-id", "worker-1",
                "--from", "c01",
                "--to", "c02",
                "--prefetch", "2",
            ])

        self.assertEqual(result, 0)
        posts = [
            index for index, call in enumerate(calls)
            if call == ("POST", "/v1/workers/worker-1/jobs")
        ]
        first_status = next(
            index for index, call in enumerate(calls)
            if call[0] == "GET" and call[1].startswith("/v1/tasks/")
        )
        self.assertEqual(len(posts), 2)
        self.assertLess(posts[0], first_status)
        self.assertLess(posts[1], first_status)
        self.assertEqual(list(tasks), ["task-1", "task-2"])
        self.assertEqual(
            [manifest["jobs"][shot_id]["terminal_status"] for shot_id in shot_ids],
            ["succeeded", "succeeded"],
        )

    def test_reference_encoding_is_repository_scoped_and_preserves_bytes(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            reference = root / "refs" / "plate.png"
            reference.parent.mkdir()
            reference.write_bytes(b"reference image bytes")
            with mock.patch.object(submit_startframes, "ROOT", root):
                encoded = submit_startframes.encode_reference_images(["refs/plate.png"])
                self.assertEqual(
                    base64.b64decode(encoded[0]["data"]),
                    b"reference image bytes",
                )
                self.assertEqual(encoded[0]["label"], "refs/plate.png")
                with self.assertRaisesRegex(RuntimeError, "missing or outside"):
                    submit_startframes.encode_reference_images(["../outside.png"])

    def test_request_builder_is_reference_free_for_the_base_checkpoint(self):
        request = submit_startframes.build_request_for_shot(
            submit_startframes.QWEN_CHECKPOINT,
            {"id": "c01", "cam": ["static wide: camera prompt"]},
            {"width": 832, "height": 480},
            {"prompt": "ignored"},
            "negative prompt",
            30,
            {},
        )
        self.assertEqual(request["prompt"], "camera prompt")
        self.assertNotIn("reference_images", request)

    def test_request_builder_fingerprints_qwen_space_inputs(self):
        shot = {"id": "c01", "cam": ["static wide: camera prompt"]}
        config = {"width": 832, "height": 480}
        override = {
            "mode": "Edit Image (1 Ref)",
            "refs": ["refs/plate.png"],
            "prompt": "detailed prompt",
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            (root / "refs").mkdir()
            (root / "refs" / "plate.png").write_bytes(b"reference image bytes")
            entry = {"attempts": 0}
            with mock.patch.object(submit_startframes, "ROOT", root):
                request = submit_startframes.build_request_for_shot(
                    submit_startframes.QWEN_SPACE_CHECKPOINT,
                    shot,
                    config,
                    override,
                    "negative prompt",
                    24,
                    entry,
                )
                self.assertEqual(len(request["reference_images"]), 1)
                self.assertEqual(entry["reference_plates"], ["refs/plate.png"])
                self.assertTrue(entry["request_sha256"])

                entry["attempts"] = 1
                with self.assertRaisesRegex(RuntimeError, "changed since this batch started"):
                    submit_startframes.build_request_for_shot(
                        submit_startframes.QWEN_SPACE_CHECKPOINT,
                        shot,
                        config,
                        {**override, "prompt": "different prompt"},
                        "negative prompt",
                        24,
                        entry,
                    )

    def test_explicit_ammi_classification_blocks_body_and_allows_hands_voice(self):
        with mock.patch.dict(submit_startframes.AMMI_DEPICTIONS, {"c49": "body"}):
            with self.assertRaisesRegex(RuntimeError, "c49: Ammi depiction is body"):
                submit_startframes.validate_visual_contract([
                    {"id": "c49", "cam": ["static medium: Ammi fills the doorway"]},
                ])

        submit_startframes.validate_visual_contract([
            {"id": "c49", "cam": ["static medium: Ammi's hands enter with bags"]},
        ])

        submit_startframes.validate_visual_contract([
            {"id": "c54", "cam": ["static: Ammi's voice comes through the door"]},
        ])

    def test_unmapped_beat_defaults_to_no_ammi(self):
        self.assertEqual(submit_startframes.AMMI_DEPICTIONS.get("c01", "no Ammi"), "no Ammi")
        submit_startframes.validate_visual_contract([
            {"id": "c01", "cam": ["static wide: Juveriya waits in the hallway"]},
        ])

    def test_bounded_range_and_persisted_succeeded_state(self):
        self.assertEqual(
            submit_startframes.collect_startframes.select_shot_ids("c01", "c03"),
            ["c01", "c02", "c03"],
        )
        entry = {"job_id": "task-1", "status": "running", "attempts": 1}
        entry["output_sha256"] = "frame-hash"
        entry["attempts_detail"] = [{
            "job_id": "task-1",
            "status": "running",
            "terminal_status": None,
        }]
        submit_startframes.update_terminal_status(entry, "completed")
        self.assertEqual(entry["terminal_status"], "succeeded")
        self.assertEqual(entry["attempts_detail"][0]["terminal_status"], "succeeded")
        self.assertTrue(submit_startframes.is_succeeded(entry))
        self.assertFalse(submit_startframes.is_succeeded({
            "job_id": "task-2",
            "status": "running",
            "terminal_status": None,
        }))

    def test_batch_status_counts_only_the_selected_beats(self):
        manifest = {
            "selected_ids": ["c23", "c25"],
            "jobs": {
                shot_id: {
                    "job_id": None,
                    "terminal_status": None,
                    "attempts": 0,
                }
                for shot_id in submit_startframes.SHOT_IDS
            },
        }

        submit_startframes.aggregate_batch_status(manifest)
        self.assertEqual(manifest["pending_count"], 2)
        self.assertEqual(manifest["batch_status"], "generating")

        manifest["jobs"]["c23"].update({
            "terminal_status": "succeeded",
            "output_sha256": "frame-hash",
        })
        manifest["jobs"]["c25"].update({
            "terminal_status": "failed",
            "attempts": 2,
        })
        submit_startframes.aggregate_batch_status(manifest)
        self.assertEqual(manifest["completed_count"], 1)
        self.assertEqual(manifest["failed_count"], 1)
        self.assertEqual(manifest["pending_count"], 0)
        self.assertEqual(manifest["batch_status"], "images_failed")

    def test_new_batch_manifest_records_a_sparse_selection(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            prompt_dir = root / "prompts"
            prompt_dir.mkdir()
            shots = [{"id": shot_id} for shot_id in submit_startframes.SHOT_IDS]
            configs = {
                shot_id: {"width": 832, "height": 480}
                for shot_id in submit_startframes.SHOT_IDS
            }
            current_manifest = {
                "jobs": {
                    shot_id: {"output_sha256": "old-" + shot_id}
                    for shot_id in submit_startframes.SHOT_IDS
                },
            }
            for shot_id in submit_startframes.SHOT_IDS:
                (prompt_dir / f"{shot_id}.txt").write_text(shot_id, encoding="utf-8")

            with mock.patch.object(submit_startframes, "ROOT", root):
                manifest = submit_startframes.prepare_batch_manifest(
                    "worker-1",
                    current_manifest,
                    shots,
                    configs,
                    manifest_path=root / "batch.json",
                    model=submit_startframes.QWEN_SPACE_CHECKPOINT,
                    reference_free=False,
                    selected_ids=["c23", "c25"],
                )

        self.assertEqual(manifest["selected_ids"], ["c23", "c25"])
        self.assertEqual(manifest["pending_count"], 2)

    def test_collected_frame_updates_only_its_canonical_manifest_entry(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            frame_path = root / "startframes" / "c23.png"
            frame_path.parent.mkdir()
            frame_path.write_bytes(b"new c23 frame")
            manifest_path = root / "startframes" / "manifest.json"
            original = {
                "jobs": {
                    "c23": {
                        "status": "completed",
                        "output_sha256": "old-hash",
                        "output_dimensions": [832, 480],
                        "other": "preserve",
                    },
                    "c24": {"status": "completed", "output_sha256": "c24-hash"},
                },
            }
            manifest_path.write_text(
                submit_startframes.json.dumps(original),
                encoding="utf-8",
            )
            entry = {
                "output_sha256": submit_startframes.sha256(frame_path.read_bytes()),
                "output_dimensions": [832, 480],
            }

            with mock.patch.object(submit_startframes, "ROOT", root):
                submit_startframes.sync_current_frame_manifest("c23", entry)

            updated = submit_startframes.json.loads(
                manifest_path.read_text(encoding="utf-8"),
            )

        self.assertEqual(updated["jobs"]["c23"]["output_sha256"], entry["output_sha256"])
        self.assertEqual(updated["jobs"]["c23"]["output_dimensions"], [832, 480])
        self.assertEqual(updated["jobs"]["c23"]["other"], "preserve")
        self.assertEqual(updated["jobs"]["c24"], original["jobs"]["c24"])

    def test_canonical_manifest_sync_rejects_a_changed_frame_hash(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            frame_path = root / "startframes" / "c23.png"
            frame_path.parent.mkdir()
            frame_path.write_bytes(b"different frame")
            manifest_path = root / "startframes" / "manifest.json"
            manifest_path.write_text(
                submit_startframes.json.dumps({"jobs": {"c23": {}}}),
                encoding="utf-8",
            )

            with mock.patch.object(submit_startframes, "ROOT", root):
                with self.assertRaisesRegex(RuntimeError, "hash changed before manifest sync"):
                    submit_startframes.sync_current_frame_manifest(
                        "c23",
                        {
                            "output_sha256": "not-the-file-hash",
                            "output_dimensions": [832, 480],
                        },
                    )

    def test_model_store_url_prefers_the_current_runtime_url_file(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            data_dir = root / "data"
            (data_dir / "logs").mkdir(parents=True)
            (data_dir / "logs" / "model-store-url.txt").write_text(
                "https://current.example\n",
                encoding="utf-8",
            )
            (root / ".env").write_text(
                "DATA_DIR=" + str(data_dir) + "\n"
                "KAGGLE_MODEL_STORE_URL=https://stale.example\n",
                encoding="utf-8",
            )
            with mock.patch.object(submit_startframes, "ROOT", root):
                self.assertEqual(
                    submit_startframes.read_model_store_url(),
                    "https://current.example",
                )

if __name__ == "__main__":
    unittest.main()
