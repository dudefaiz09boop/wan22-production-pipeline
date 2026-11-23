import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_startframe_manifest


class BuildStartframeManifestTests(unittest.TestCase):
    def test_manifest_rejects_duplicate_frame_hashes(self):
        hashes = [f"frame-{index}" for index in range(70)]
        hashes[-1] = hashes[0]

        with self.assertRaisesRegex(RuntimeError, "70 distinct SHA-256 hashes"):
            build_startframe_manifest.validate_distinct_hashes(hashes)


if __name__ == "__main__":
    unittest.main()
