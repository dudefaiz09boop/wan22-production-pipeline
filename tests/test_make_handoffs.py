from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import make_handoffs


def tiny_png(width: int = 4, height: int = 4) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (0, 255, 0)).save(buffer, format="PNG")
    return buffer.getvalue()


def write_beats(root: Path, chain_by_id: dict[str, bool]) -> Path:
    beats = root / "docs" / "beats.yaml"
    beats.parent.mkdir(parents=True, exist_ok=True)
    rows = "\n".join(
        f"- id: {shot_id}\n  chain: {str(chain).lower()}"
        for shot_id, chain in chain_by_id.items()
    )
    beats.write_text(f"shots:\n{rows}\n", encoding="utf-8")
    return beats


class MakeHandoffsTests(unittest.TestCase):
    def test_plan_reads_chain_dynamically_and_requires_previous_segment(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            beats = write_beats(root, {"c01": False, "c02": True, "c03": True, "c04": False})
            segments = root / "renders" / "segments"
            segments.mkdir(parents=True)
            (segments / "c01.mp4").write_bytes(b"fake")
            handoffs = root / "renders" / "handoffs"

            ordered = make_handoffs.load_ordered_shots(beats)
            plans = make_handoffs.plan_handoffs(ordered, segments, handoffs)

        self.assertEqual(
            [(plan["shot_id"], plan["previous_id"], plan["segment_ready"]) for plan in plans],
            [("c02", "c01", True), ("c03", "c02", False)],
        )

    def test_plan_is_idempotent_unless_forced(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            beats = write_beats(root, {"c01": False, "c02": True})
            segments = root / "segments"
            segments.mkdir()
            (segments / "c01.mp4").write_bytes(b"fake")
            handoffs = root / "handoffs"
            handoffs.mkdir()
            (handoffs / "c02.png").write_bytes(tiny_png())

            ordered = make_handoffs.load_ordered_shots(beats)
            self.assertEqual(make_handoffs.plan_handoffs(ordered, segments, handoffs), [])
            forced = make_handoffs.plan_handoffs(ordered, segments, handoffs, force=True)
        self.assertEqual([plan["shot_id"] for plan in forced], ["c02"])

    def test_extract_last_frame_keeps_native_size_and_no_scale_filters(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            segment = root / "segments" / "c01.mp4"
            segment.parent.mkdir(parents=True)
            segment.write_bytes(b"fake")
            destination = root / "handoffs" / "c02.png"
            captured = []

            def fake_run(command, timeout=300):
                captured.append(list(command))
                Path(command[-1]).write_bytes(tiny_png(640, 360))

            with mock.patch.object(make_handoffs, "run_ffmpeg", fake_run):
                make_handoffs.extract_last_frame("ffmpeg", segment, destination)

            self.assertTrue(destination.is_file())
            self.assertNotIn("-vf", captured[0])
            self.assertIn("-sseof", captured[0])

    def test_extract_last_frame_rejects_non_png_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            segment = root / "c01.mp4"
            segment.write_bytes(b"fake")
            destination = root / "c02.png"

            def fake_run(command, timeout=300):
                Path(command[-1]).write_bytes(b"not a png")

            with mock.patch.object(make_handoffs, "run_ffmpeg", fake_run):
                with self.assertRaisesRegex(RuntimeError, "did not produce a PNG"):
                    make_handoffs.extract_last_frame("ffmpeg", segment, destination)
            self.assertFalse(destination.exists())

    def test_dry_run_lists_plans_and_writes_nothing(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            beats = write_beats(root, {"c01": False, "c02": True})
            segments = root / "segments"
            segments.mkdir()
            (segments / "c01.mp4").write_bytes(b"fake")
            handoffs = root / "handoffs"
            stdout = io.StringIO()

            with mock.patch.object(
                make_handoffs,
                "run_ffmpeg",
                side_effect=AssertionError("dry run must not invoke ffmpeg"),
            ), contextlib.redirect_stdout(stdout):
                code = make_handoffs.main([
                    "--beats", str(beats),
                    "--segments-dir", str(segments),
                    "--handoffs-dir", str(handoffs),
                    "--dry-run",
                ])

        self.assertEqual(code, 0)
        self.assertIn("would extract c02.png from c01 (ready)", stdout.getvalue())
        self.assertFalse((handoffs / "c02.png").exists())

    def test_main_creates_handoffs_and_reports_missing_segments(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()
            beats = write_beats(root, {"c01": False, "c02": True, "c03": True})
            segments = root / "segments"
            segments.mkdir()
            (segments / "c01.mp4").write_bytes(b"fake")
            handoffs = root / "handoffs"
            stdout = io.StringIO()

            with mock.patch.object(
                make_handoffs,
                "run_ffmpeg",
                lambda command, timeout=300: Path(command[-1]).write_bytes(tiny_png()),
            ), contextlib.redirect_stdout(stdout):
                code = make_handoffs.main([
                    "--beats", str(beats),
                    "--segments-dir", str(segments),
                    "--handoffs-dir", str(handoffs),
                ])

            self.assertEqual(code, 1)
            self.assertTrue((handoffs / "c02.png").is_file())
            self.assertIn("c02: handoff extracted from c01", stdout.getvalue())
            self.assertIn("previous segment c02.mp4 is missing; skipped", stdout.getvalue())
            self.assertFalse((handoffs / "c03.png").exists())


if __name__ == "__main__":
    unittest.main()
