import contextlib
import hashlib
import io
import json
import os
import struct
import sys
import tempfile
import time
import unittest
import zlib
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_startframe_manifest
import collect_startframes
import render_continuity


def valid_png(width, height, color):
    def chunk(name, contents):
        body = name + contents
        return (
            struct.pack(">I", len(contents))
            + body
            + struct.pack(">I", zlib.crc32(body))
        )

    pixels = b"".join(b"\x00" + bytes(color) * width for _ in range(height))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(pixels))
        + chunk(b"IEND", b"")
    )


class RenderContinuityTests(unittest.TestCase):
    def test_render_and_collection_cover_c01_through_c70(self):
        self.assertEqual(len(render_continuity.SHOT_IDS), 70)
        self.assertEqual(render_continuity.SHOT_IDS[0], "c01")
        self.assertEqual(render_continuity.SHOT_IDS[-1], "c70")
        self.assertEqual(collect_startframes.SHOT_IDS, render_continuity.SHOT_IDS)

    def test_video_profile_prefers_process_environment_then_dotenv(self):
        with mock.patch.dict(
            os.environ,
            {"KAGGLE_VIDEO_PROFILE": "process-profile"},
        ), mock.patch.object(
            render_continuity,
            "read_env_file",
            return_value={"KAGGLE_VIDEO_PROFILE": "dotenv-profile"},
        ):
            self.assertEqual(
                render_continuity.resolve_video_profile(),
                ("process-profile", "$env:KAGGLE_VIDEO_PROFILE"),
            )
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            render_continuity,
            "read_env_file",
            return_value={"KAGGLE_VIDEO_PROFILE": "dotenv-profile"},
        ):
            self.assertEqual(
                render_continuity.resolve_video_profile(),
                ("dotenv-profile", ".env"),
            )

    def test_missing_video_profile_names_both_sources(self):
        with mock.patch.dict(os.environ, {}, clear=True), mock.patch.object(
            render_continuity,
            "read_env_file",
            return_value={},
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                r"KAGGLE_VIDEO_PROFILE not found.*\$env:KAGGLE_VIDEO_PROFILE and \.env",
            ):
                render_continuity.resolve_video_profile()

    def test_image_worker_selection_requires_idle_image_worker(self):
        rows = [
            {"kind": "video", "model": "video-model", "status": "idle", "id": "v1"},
            {"kind": "image", "model": "image-model", "status": "busy", "id": "i1"},
            {"kind": "image", "model": "image-model", "status": "idle", "id": "i2"},
        ]
        with mock.patch.object(
            render_continuity,
            "api_json",
            return_value={"data": rows},
        ) as request:
            worker = render_continuity.discover_image_worker(
                "http://127.0.0.1:3000",
                "key",
                ["c01"],
            )
        self.assertEqual(worker["id"], "i2")
        request.assert_called_once_with(
            "http://127.0.0.1:3000",
            "key",
            "/v1/workers/active",
        )

    def test_no_idle_image_worker_with_local_refs_continues(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "prompts").mkdir()
            (root / "startframes").mkdir()
            (root / "prompts" / "c01.json").write_text(
                json.dumps({
                    "start_frame": "startframes/c01.png",
                    "width": 832,
                    "height": 480,
                }),
                encoding="utf-8",
            )
            frame = (
                b"\x89PNG\r\n\x1a\n"
                + b"\x00\x00\x00\rIHDR"
                + (832).to_bytes(4, "big")
                + (480).to_bytes(4, "big")
                + b"\x08\x02\x00\x00\x00"
            )
            (root / "startframes" / "c01.png").write_bytes(frame)
            manifest = {
                "batch_status": "generating",
                "jobs": {
                    "c01": {
                        "status": "completed",
                        "start_frame": "startframes/c01.png",
                        "output_sha256": hashlib.sha256(frame).hexdigest(),
                    }
                },
            }
            manifest_path = root / "startframes" / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            stdout = io.StringIO()
            with mock.patch.object(render_continuity, "ROOT", root.resolve()), mock.patch.object(
                render_continuity,
                "api_json",
                return_value={"data": [
                    {"kind": "image", "model": "image-model", "status": "busy", "id": "i1"},
                ]},
            ), contextlib.redirect_stdout(stdout):
                worker = render_continuity.discover_image_worker(
                    "http://127.0.0.1:3000",
                    "key",
                    ["c01"],
                )
                with mock.patch.object(
                    render_continuity,
                    "api_json",
                    side_effect=AssertionError("image worker polling must be skipped"),
                ):
                    ready = render_continuity.wait_for_images(
                        "http://127.0.0.1:3000",
                        "key",
                        worker["id"] if worker else None,
                        manifest_path,
                        ["c01"],
                        1,
                        float("inf"),
                    )
        self.assertIsNone(worker)
        self.assertEqual(ready, manifest)
        self.assertEqual(
            stdout.getvalue(),
            "image|image-model|busy|i1\n"
            "no idle image worker; refs present, continuing\n"
            "Start frames: 1/1 ready\n",
        )

    def test_no_idle_image_worker_with_missing_ref_exits_two(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "prompts").mkdir()
            (root / "prompts" / "c01.json").write_text(
                json.dumps({"start_frame": "startframes/c01.png"}),
                encoding="utf-8",
            )
            stdout = io.StringIO()
            stderr = io.StringIO()
            with mock.patch.object(render_continuity, "ROOT", root.resolve()), mock.patch.object(
                render_continuity,
                "api_json",
                return_value={"data": [
                    {"kind": "image", "model": "image-model", "status": "busy", "id": "i1"},
                ]},
            ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                with self.assertRaises(SystemExit) as result:
                    render_continuity.discover_image_worker(
                        "http://127.0.0.1:3000",
                        "key",
                        ["c01"],
                    )
        self.assertEqual(result.exception.code, 2)
        self.assertEqual(stdout.getvalue(), "image|image-model|busy|i1\n")
        self.assertEqual(stderr.getvalue(), "Missing required start frames: c01\n")

    def test_range_normalizes_ids_and_uses_inclusive_bounds(self):
        self.assertEqual(
            render_continuity.select_shots("c47", "c50"),
            ["c47", "c48", "c49", "c50"],
        )
        self.assertEqual(render_continuity.normalize_shot_id("c1"), "c01")
        with self.assertRaisesRegex(ValueError, "comes after"):
            render_continuity.select_shots("c50", "c47")
        with self.assertRaisesRegex(ValueError, "not in the ordered shot list"):
            render_continuity.select_shots("c71", "c70")

    def test_chained_shot_falls_back_to_its_start_frame(self):
        config = {
            "chain": True,
            "start_frame": "startframes/c48.png",
        }
        reference, chained, reseeded = render_continuity.keyframe_choice(
            "c48",
            config,
            handoff_available=False,
        )
        self.assertEqual(reference, render_continuity.ROOT / "startframes" / "c48.png")
        self.assertFalse(chained)
        self.assertTrue(reseeded)

    def test_build_manifest_uses_existing_files_without_worker_jobs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "docs").mkdir()
            (root / "prompts").mkdir()
            (root / "startframes").mkdir()
            shots = []
            for number in range(1, 71):
                shot_id = f"c{number:02d}"
                frame_path = f"startframes/{shot_id}.png"
                png = valid_png(
                    832,
                    480,
                    (number, number * 3 % 256, number * 7 % 256),
                )
                shots.append({"id": shot_id, "start_frame": frame_path})
                (root / "prompts" / f"{shot_id}.json").write_text(
                    json.dumps({
                        "id": shot_id,
                        "start_frame": frame_path,
                        "width": 832,
                        "height": 480,
                    }),
                    encoding="utf-8",
                )
                (root / "prompts" / f"{shot_id}.txt").write_text(
                    f"prompt {shot_id}\n",
                    encoding="utf-8",
                )
                (root / frame_path).write_bytes(png)
            (root / "docs" / "beats.yaml").write_text(
                json.dumps({"shots": shots}),
                encoding="utf-8",
            )

            manifest = build_startframe_manifest.build_manifest(root)

        self.assertEqual(manifest["batch_status"], "images_ready")
        self.assertEqual(manifest["completed_count"], 70)
        self.assertEqual(set(manifest["jobs"]), set(render_continuity.SHOT_IDS))
        self.assertTrue(all(entry["job_id"] is None for entry in manifest["jobs"].values()))

    def test_api_origin_is_loopback_only(self):
        self.assertEqual(
            render_continuity.validate_api_origin("http://127.0.0.1:3000"),
            "http://127.0.0.1:3000",
        )
        with self.assertRaisesRegex(RuntimeError, "loopback HTTP origin"):
            render_continuity.validate_api_origin("https://example.com")

    def test_api_origin_rejects_embedded_credentials(self):
        with self.assertRaisesRegex(RuntimeError, "credentials are never sent remotely"):
            render_continuity.validate_api_origin("http://user:secret@127.0.0.1:3000")

    def test_png_dimensions_rejects_non_png_data(self):
        with self.assertRaisesRegex(RuntimeError, "valid PNG"):
            render_continuity.png_dimensions(b"not an image")

    def test_video_duration_drives_frame_count_and_rejects_stale_config(self):
        self.assertEqual(render_continuity.shot_duration({"duration": 4}, "c12"), 4)
        self.assertEqual(render_continuity.video_frame_count(4), 65)
        self.assertEqual(render_continuity.video_frame_count(5), 81)
        with self.assertRaisesRegex(RuntimeError, "duration fields disagree"):
            render_continuity.shot_duration(
                {"duration": 4, "beat_duration": 5},
                "c12",
            )
        with self.assertRaisesRegex(RuntimeError, "compiled duration"):
            render_continuity.shot_duration({"duration": True}, "c12")

    def test_dry_run_sums_per_shot_durations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "prompts").mkdir()
            selected = render_continuity.select_shots("c12", "c22")
            for shot_id in selected:
                duration = 4 if shot_id in {"c12", "c18", "c22"} else 5
                (root / "prompts" / f"{shot_id}.json").write_text(
                    json.dumps({
                        "chain": False,
                        "duration": duration,
                        "start_frame": f"startframes/{shot_id}.png",
                    }),
                    encoding="utf-8",
                )

            output = io.StringIO()
            with mock.patch.object(render_continuity, "ROOT", root.resolve()), contextlib.redirect_stdout(output):
                result = render_continuity.dry_run(
                    selected,
                    "test-profile",
                    "test",
                )

        self.assertEqual(result, 0)
        self.assertIn("11 shots; expected duration 52s", output.getvalue())

    def test_normalize_video_trims_to_the_shot_duration(self):
        commands = []
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "raw.webm"
            destination = root / "c12.mp4"
            with mock.patch.object(render_continuity, "ROOT", root), mock.patch.object(
                render_continuity,
                "run_media",
                side_effect=lambda command, timeout=600: commands.append(command),
            ):
                render_continuity.normalize_video("ffmpeg", source, destination, 4)

        trim_index = commands[0].index("-t")
        self.assertEqual(commands[0][trim_index + 1], "4")
        preset_index = commands[0].index("-preset")
        self.assertEqual(commands[0][preset_index + 1], "slow")

    def test_poll_delay_ramps_from_two_seconds_up_to_the_cap(self):
        started = time.monotonic()
        with mock.patch.object(
            render_continuity.time,
            "monotonic",
            return_value=started,
        ):
            self.assertEqual(render_continuity.poll_delay(started, 30), 2)
        with mock.patch.object(
            render_continuity.time,
            "monotonic",
            return_value=started + 1000,
        ):
            self.assertEqual(render_continuity.poll_delay(started, 30), 30)
            self.assertEqual(render_continuity.poll_delay(started, 1), 1)

    def test_finalize_segment_records_a_normalized_clip(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "renders" / "segments").mkdir(parents=True)
            raw = root / "renders" / "raw" / "c47.mp4"
            raw.parent.mkdir(parents=True)
            raw.write_bytes(b"raw")
            state = {"jobs": {"c47": {}}}
            state_path = root / "renders" / "manifest.json"
            metadata = {
                "format": {"duration": "5.000"},
                "streams": [{"codec_type": "video", "width": 832, "height": 480}],
            }

            def fake_normalize(ffmpeg, source, destination, duration):
                destination.write_bytes(b"segment")

            with mock.patch.object(render_continuity, "ROOT", root), mock.patch.object(
                render_continuity,
                "normalize_video",
                fake_normalize,
            ), mock.patch.object(
                render_continuity,
                "probe_video",
                return_value=metadata,
            ):
                render_continuity.finalize_segment(
                    "ffmpeg",
                    "ffprobe",
                    state,
                    state_path,
                    "c47",
                    raw,
                    5,
                    "completed",
                )

        self.assertEqual(state["jobs"]["c47"]["video_status"], "completed")
        self.assertEqual(state["jobs"]["c47"]["video_dimensions"], [832, 480])
        self.assertEqual(state["jobs"]["c47"]["duration_seconds"], 5.0)
        self.assertEqual(state["jobs"]["c47"]["segment_bytes"], len(b"segment"))

    def test_existing_segment_is_reused_only_when_duration_matches(self):
        segment = Path("c12.mp4")
        four_second_video = {
            "format": {"duration": "4.000"},
            "streams": [{"codec_type": "video", "width": 832, "height": 480}],
        }
        five_second_video = {
            "format": {"duration": "5.000"},
            "streams": [{"codec_type": "video", "width": 832, "height": 480}],
        }
        with mock.patch.object(
            render_continuity,
            "probe_video",
            side_effect=[four_second_video, five_second_video],
        ):
            self.assertTrue(
                render_continuity.normalized_segment_matches("ffprobe", segment, 4)
            )
            self.assertFalse(
                render_continuity.normalized_segment_matches("ffprobe", segment, 4)
            )

    def test_stitch_validates_the_sum_of_per_shot_durations(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            segment_dir = root / "renders" / "segments"
            segment_dir.mkdir(parents=True)
            selected = render_continuity.select_shots("c12", "c22")
            for shot_id in selected:
                (segment_dir / f"{shot_id}.mp4").write_bytes(b"segment")
            state = {}
            state_path = root / "renders" / "manifest.json"

            def create_output(command, timeout=600):
                Path(command[-1]).write_bytes(b"stitched")

            metadata = {
                "format": {"duration": "52.000"},
                "streams": [{"codec_type": "video", "width": 832, "height": 480}],
            }
            with mock.patch.object(render_continuity, "ROOT", root), mock.patch.object(
                render_continuity,
                "run_media",
                side_effect=create_output,
            ), mock.patch.object(render_continuity, "probe_video", return_value=metadata):
                result = render_continuity.stitch(
                    "ffmpeg",
                    "ffprobe",
                    state,
                    state_path,
                    selected,
                    expected_duration_seconds=52,
                )

        self.assertEqual(result.name, "midnight-heat-11-shot.mp4")
        self.assertEqual(state["film"]["duration_seconds"], 52.0)



def tiny_png(width: int = 4, height: int = 4) -> bytes:
    from PIL import Image
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (255, 0, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


class PipelinedRenderTests(unittest.TestCase):
    def test_next_job_is_submitted_before_the_previous_clip_is_normalized(self):
        shot_ids = ["c01", "c02"]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            (root / "prompts").mkdir()
            (root / "startframes").mkdir()
            (root / "negatives").mkdir()
            (root / "renders").mkdir()
            (root / "negatives" / "artifact_neg.txt").write_text("negative", encoding="utf-8")
            for shot_id in shot_ids:
                (root / "prompts" / f"{shot_id}.json").write_text(
                    json.dumps({
                        "id": shot_id,
                        "scene": 1,
                        "build": "fm",
                        "chain": False,
                        "duration": 5,
                        "loras": [],
                        "width": 832,
                        "height": 480,
                        "start_frame": f"startframes/{shot_id}.png",
                    }),
                    encoding="utf-8",
                )
                (root / "prompts" / f"{shot_id}.txt").write_text("prompt", encoding="utf-8")
                (root / "startframes" / f"{shot_id}.png").write_bytes(b"frame")

            image_manifest = {
                "jobs": {
                    shot_id: {"prompt_sha256": "hash", "output_sha256": "frame-hash"}
                    for shot_id in shot_ids
                }
            }
            (root / "startframes" / "manifest.json").write_text(
                json.dumps(image_manifest), encoding="utf-8",
            )
            events = []
            image_waits = []

            def fake_api(api, api_key, route, *, data=None):
                if data is None and route == "/v1/workers/image-1":
                    return {"id": "image-1", "kind": "image", "status": "busy"}
                events.append(("submit", data["story"]["shot_id"]))
                job_id = f"task-{data['story']['shot_id']}"
                return {"id": job_id, "status": "queued"}

            def fake_wait_for_images(
                api, api_key, worker_id, manifest_path, shot_list, poll_seconds, deadline,
            ):
                image_waits.append((worker_id, list(shot_list)))
                return image_manifest

            def fake_encode(ffmpeg, source, destination, width, height, max_bytes, **kwargs):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"jpeg")

            def fake_download(api, api_key, job_id, destination):
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"raw")

            def fake_normalize(ffmpeg, source, destination, duration):
                shot_id = source.stem
                events.append(("normalize", shot_id))
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(b"segment")

            def fake_stage(ffmpeg, shot_id, source):
                events.append(("handoff", shot_id, source.name, source.parent.name))
                return None

            def fake_stitch(ffmpeg, ffprobe, state, state_path, selected, expected_duration_seconds):
                film = root / "renders" / "film.mp4"
                film.write_bytes(b"film")
                return film

            env = {
                "API_KEY": "k" * 40,
                "FFMPEG_PATH": "ffmpeg",
                "FFPROBE_PATH": "ffprobe",
            }
            metadata = {
                "format": {"duration": "5.000"},
                "streams": [{"codec_type": "video", "width": 832, "height": 480}],
            }
            with mock.patch.object(render_continuity, "ROOT", root), mock.patch.object(
                sys,
                "argv",
                [
                    "render_continuity.py",
                    "--from", "c01",
                    "--to", "c02",
                    "--image-worker-id", "image-1",
                ],
            ), mock.patch.dict(
                os.environ, {"KAGGLE_VIDEO_PROFILE": "test-profile"},
            ), mock.patch.object(
                render_continuity, "load_env", return_value=env,
            ), mock.patch.object(
                render_continuity, "discover_image_worker", return_value=None,
            ), mock.patch.object(
                render_continuity,
                "discover_video_worker",
                return_value={"id": "video-1", "model": "test-profile"},
            ), mock.patch.object(
                render_continuity, "wait_for_images", side_effect=fake_wait_for_images,
            ), mock.patch.object(
                render_continuity, "read_prompt", return_value="prompt",
            ), mock.patch.object(render_continuity, "api_json", side_effect=fake_api), mock.patch.object(
                render_continuity, "encode_jpeg", side_effect=fake_encode,
            ), mock.patch.object(
                render_continuity, "wait_for_task", return_value={"status": "completed"},
            ), mock.patch.object(
                render_continuity, "download_video", side_effect=fake_download,
            ), mock.patch.object(
                render_continuity, "stage_next_handoff", side_effect=fake_stage,
            ), mock.patch.object(
                render_continuity, "normalize_video", side_effect=fake_normalize,
            ), mock.patch.object(
                render_continuity, "probe_video", return_value=metadata,
            ), mock.patch.object(render_continuity, "stitch", side_effect=fake_stitch):
                result = render_continuity.main()

        self.assertEqual(result, 0)
        self.assertEqual(image_waits, [("image-1", ["c01"]), ("image-1", ["c02"])])
        self.assertEqual(events, [
            ("submit", "c01"),
            ("handoff", "c01", "c01.mp4", "raw"),
            ("submit", "c02"),
            ("normalize", "c01"),
            ("handoff", "c02", "c02.mp4", "raw"),
            ("normalize", "c02"),
        ])


class HandoffExtractionTests(unittest.TestCase):
    def write_prompts(self, root: Path, chain_by_id: dict[str, bool]) -> None:
        prompts = root / "prompts"
        prompts.mkdir(parents=True, exist_ok=True)
        for shot_id, chain in chain_by_id.items():
            (prompts / f"{shot_id}.json").write_text(
                json.dumps({
                    "start_frame": f"startframes/{shot_id}.png",
                    "chain": chain,
                    "width": 832,
                    "height": 480,
                }),
                encoding="utf-8",
            )

    def test_keyframe_choice_uses_handoff_when_available(self):
        config = {"chain": True, "start_frame": "startframes/c48.png"}
        reference, chained, reseeded = render_continuity.keyframe_choice(
            "c48", config, handoff_available=True,
        )
        self.assertEqual(reference, render_continuity.ROOT / "renders" / "handoffs" / "c48.png")
        self.assertTrue(chained)
        self.assertFalse(reseeded)

    def test_extract_last_frame_writes_native_size_png_atomically(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            segment = root / "renders" / "segments" / "c47.mp4"
            segment.parent.mkdir(parents=True)
            segment.write_bytes(b"fake video")
            destination = root / "renders" / "handoffs" / "c48.png"
            captured = []

            def fake_run_media(command, timeout=600):
                captured.append(list(command))
                Path(command[-1]).write_bytes(tiny_png(640, 360))

            with mock.patch.object(render_continuity, "run_media", fake_run_media):
                render_continuity.extract_last_frame("ffmpeg", segment, destination)

            self.assertTrue(destination.is_file())
            self.assertFalse(destination.with_name("c48.tmp.png").exists())
            command = captured[0]
            self.assertIn("-sseof", command)
            self.assertIn("-update", command)
            self.assertNotIn("-vf", command)

    def test_extract_last_frame_refuses_empty_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            segment = root / "c47.mp4"
            segment.write_bytes(b"fake video")
            destination = root / "handoffs" / "c48.png"
            with mock.patch.object(render_continuity, "run_media", lambda *a, **k: None):
                with self.assertRaisesRegex(RuntimeError, "produced no final frame"):
                    render_continuity.extract_last_frame("ffmpeg", segment, destination)
            self.assertFalse(destination.exists())
            self.assertFalse(destination.with_name("c48.tmp.png").exists())

    def test_stage_next_handoff_reads_chain_dynamically(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            self.write_prompts(root, {"c47": False, "c48": True, "c49": False})
            segment = root / "renders" / "segments" / "c47.mp4"
            segment.parent.mkdir(parents=True)
            segment.write_bytes(b"fake video")

            with mock.patch.object(render_continuity, "ROOT", root.resolve()):
                with mock.patch.object(
                    render_continuity,
                    "run_media",
                    lambda command, timeout=600: Path(command[-1]).write_bytes(tiny_png()),
                ):
                    staged = render_continuity.stage_next_handoff("ffmpeg", "c47", segment)
                self.assertEqual(staged, root / "renders" / "handoffs" / "c48.png")
                self.assertTrue(staged.is_file())

                # Next shot is chain:false (identity reset): no handoff staged.
                self.assertIsNone(render_continuity.stage_next_handoff("ffmpeg", "c48", segment))

    def test_stage_next_handoff_missing_next_config_returns_none(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            segment = root / "segments" / "c70.mp4"
            segment.parent.mkdir(parents=True, exist_ok=True)
            segment.write_bytes(b"fake video")
            with mock.patch.object(render_continuity, "ROOT", root.resolve()):
                self.assertIsNone(
                    render_continuity.stage_next_handoff("ffmpeg", "c70", segment),
                )

    def test_ensure_handoff_refuses_missing_frame_for_chained_shot(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            handoff = root / "handoffs" / "c48.png"
            config = {"chain": True, "start_frame": "startframes/c48.png"}
            with self.assertRaisesRegex(RuntimeError, "require-handoffs.*refusing"):
                render_continuity.ensure_handoff("c48", config, handoff, require=True)
            handoff.parent.mkdir(parents=True, exist_ok=True)
            handoff.write_bytes(tiny_png())
            render_continuity.ensure_handoff("c48", config, handoff, require=True)

    def test_ensure_handoff_ignores_unchained_and_first_shots(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root / "handoffs" / "c01.png"
            first = {"chain": False, "start_frame": "startframes/c01.png"}
            reset = {"chain": False, "start_frame": "startframes/c22.png"}
            render_continuity.ensure_handoff("c01", first, missing, require=True)
            render_continuity.ensure_handoff("c22", reset, missing, require=True)
            render_continuity.ensure_handoff("c48", {"chain": True}, missing, require=False)


if __name__ == "__main__":
    unittest.main()
