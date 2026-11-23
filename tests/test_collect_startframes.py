import hashlib
import struct
import tempfile
import unittest
import zlib
from pathlib import Path
from unittest import mock

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import collect_startframes


def png(width, height):
    def chunk(name, data):
        body = name + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body))

    pixels = b"".join(
        b"\x00" + b"\x00\x00\x00\xff" * width
        for _ in range(height)
    )
    return (
        collect_startframes.PNG_SIGNATURE
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(pixels))
        + chunk(b"IEND", b"")
    )


class CollectStartframesTests(unittest.TestCase):
    def test_png_validation_checks_chunk_integrity(self):
        image = bytearray(png(8, 8))
        image[29] ^= 1

        with self.assertRaisesRegex(RuntimeError, "checksum"):
            collect_startframes.png_size(bytes(image))

    def test_select_shot_ids_validates_a_bounded_range(self):
        self.assertEqual(
            collect_startframes.select_shot_ids("c02", "c04"),
            ["c02", "c03", "c04"],
        )
        with self.assertRaisesRegex(ValueError, "must not follow"):
            collect_startframes.select_shot_ids("c04", "c02")

    def test_parse_shot_ids_accepts_sparse_ordered_ids_and_rejects_duplicates(self):
        self.assertEqual(
            collect_startframes.parse_shot_ids("c23, c25, c27"),
            ["c23", "c25", "c27"],
        )
        with self.assertRaisesRegex(ValueError, "must not contain duplicates"):
            collect_startframes.parse_shot_ids("c23,c23")
        with self.assertRaisesRegex(ValueError, "c01 through c70"):
            collect_startframes.parse_shot_ids("c71")

    def test_only_missing_frames_and_placeholders_can_be_replaced(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frame.png"

            self.assertTrue(collect_startframes.may_replace_frame(path))

            path.write_bytes(png(1, 1))
            self.assertTrue(collect_startframes.may_replace_frame(path))

            path.write_bytes(png(2, 1))
            self.assertFalse(collect_startframes.may_replace_frame(path))

    def test_existing_frame_requires_explicit_matching_previous_hash_to_replace(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "frame.png"
            original = png(2, 1)
            path.write_bytes(original)
            entry = {
                "previous_output_sha256": hashlib.sha256(original).hexdigest(),
            }

            self.assertFalse(
                collect_startframes.may_replace_manifest_frame(path, entry, False)
            )
            self.assertTrue(
                collect_startframes.may_replace_manifest_frame(path, entry, True)
            )
            entry["previous_output_sha256"] = "0" * 64
            self.assertFalse(
                collect_startframes.may_replace_manifest_frame(path, entry, True)
            )

    def test_manifest_replace_retries_transient_permission_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "manifest.json"
            original_replace = collect_startframes.os.replace
            attempts = []

            def replace_with_transient_lock(source, destination):
                attempts.append(None)
                if len(attempts) == 1:
                    raise PermissionError("manifest is being read")
                original_replace(source, destination)

            with mock.patch.object(
                collect_startframes.os,
                "replace",
                side_effect=replace_with_transient_lock,
            ):
                collect_startframes.write_manifest(path, {"status": "generated"})

            self.assertEqual(len(attempts), 2)
            self.assertEqual(
                collect_startframes.json.loads(path.read_text(encoding="utf-8")),
                {"status": "generated"},
            )

    def test_completed_frame_is_validated_and_persisted(self):
        image = png(832, 480)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame_path = root / "startframes" / "c01.png"
            frame_path.parent.mkdir()
            old_image = png(832, 480)
            frame_path.write_bytes(old_image)
            entry = {
                "job_id": "task-1",
                "start_frame": "startframes/c01.png",
                "previous_output_sha256": hashlib.sha256(old_image).hexdigest(),
            }
            with mock.patch.object(collect_startframes, "ROOT", root), mock.patch.object(
                collect_startframes, "api_request", return_value=image,
            ):
                dimensions = collect_startframes.collect_completed_frame(
                    "http://127.0.0.1:3000",
                    "test-key",
                    "c01",
                    entry,
                    {"width": 832, "height": 480},
                    replace_existing=True,
                )

            self.assertEqual(dimensions, (832, 480))
            self.assertEqual(frame_path.read_bytes(), image)
            self.assertEqual(entry["output_sha256"], hashlib.sha256(image).hexdigest())
            self.assertEqual(entry["terminal_status"], "succeeded")


if __name__ == "__main__":
    unittest.main()
