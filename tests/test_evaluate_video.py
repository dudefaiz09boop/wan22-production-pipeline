from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from scripts.evaluate_video import EvaluationError, build_parser, evaluate, load_face_metrics, parse_rate


class EvaluateVideoTests(unittest.TestCase):
    def test_parse_rate_handles_fraction_and_invalid_values(self) -> None:
        self.assertEqual(parse_rate("24/1"), 24.0)
        self.assertEqual(parse_rate("0/0"), None)
        self.assertEqual(parse_rate("invalid"), None)

    def test_face_report_requires_the_full_valid_metric_set(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "face.json"
            path.write_text(json.dumps({"metrics": {"identity_similarity_min": 0.8}}), encoding="utf-8")
            with self.assertRaisesRegex(EvaluationError, "face_detection_coverage"):
                load_face_metrics(path)

            path.write_text(json.dumps({
                "metrics": {
                    "face_detection_coverage": 0.98,
                    "identity_similarity_min": 0.82,
                    "landmark_jitter_px": 2.4,
                    "expression_jump_p95": 0.1,
                    "head_motion_jitter": 0.01,
                    "lip_sync_offset_ms": 24,
                    "lip_sync_confidence": 0.8,
                    "blink_count": 1,
                },
            }), encoding="utf-8")
            metrics = load_face_metrics(path)
            self.assertEqual(metrics["identity_similarity_min"], 0.82)
            self.assertEqual(metrics["blink_count"], 1)

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg tools are required")
    def test_evaluator_reports_video_facts_and_does_not_claim_unmeasured_face_quality(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            video = Path(temporary) / "synthetic.mp4"
            subprocess.run(
                [
                    shutil.which("ffmpeg") or "ffmpeg",
                    "-hide_banner", "-loglevel", "error",
                    "-f", "lavfi", "-i", "testsrc=size=160x90:rate=10:duration=1",
                    "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", str(video),
                ],
                check=True,
                capture_output=True,
            )
            args = build_parser().parse_args([
                "--video", str(video),
                "--expected-width", "160",
                "--expected-height", "90",
                "--expected-fps", "10",
            ])
            report = evaluate(args)
            self.assertTrue(report["technicalPassed"])
            self.assertFalse(report["faceMetricsPassed"])
            self.assertFalse(report["productionReady"])
            self.assertEqual(report["video"]["width"], 160)
            self.assertEqual(report["video"]["height"], 90)
            self.assertEqual(report["video"]["fps"], 10.0)
            self.assertEqual(report["video"]["videoCodec"], "h264")
            self.assertEqual(report["faceMetrics"]["status"], "not_measured")
            self.assertTrue(all(check["passed"] for check in report["checks"]))

            face_metrics = Path(temporary) / "face.json"
            face_metrics.write_text(json.dumps({
                "metrics": {
                    "face_detection_coverage": 1.0,
                    "identity_similarity_min": 0.9,
                    "landmark_jitter_px": 1.0,
                    "expression_jump_p95": 0.1,
                    "head_motion_jitter": 0.01,
                    "lip_sync_offset_ms": 25,
                    "lip_sync_confidence": 0.9,
                    "blink_count": 1,
                },
            }), encoding="utf-8")
            args = build_parser().parse_args([
                "--video", str(video),
                "--face-metrics", str(face_metrics),
                "--max-expression-jump-p95", "0.15",
                "--max-head-motion-jitter", "0.02",
                "--min-blinks-per-minute", "10",
            ])
            report = evaluate(args)
            self.assertTrue(report["productionReady"])
            self.assertTrue(report["faceMetricsPassed"])
            self.assertTrue(all(check["passed"] for check in report["checks"]))


if __name__ == "__main__":
    unittest.main()
