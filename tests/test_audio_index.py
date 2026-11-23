import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from audio_index import resolve_audio


class AudioIndexTests(unittest.TestCase):
    def test_direct_beat_wav_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "refs" / "audio"
            audio.mkdir(parents=True)
            (audio / "c01.wav").write_bytes(b"RIFF")
            index = audio / "index.json"
            index.write_text('{"beats": {"c01": "other.wav"}}', encoding="utf-8")
            (audio / "other.wav").write_bytes(b"RIFF")
            self.assertEqual(resolve_audio(root, "c01"), audio / "c01.wav")

    def test_index_mapping_resolves(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "refs" / "audio"
            audio.mkdir(parents=True)
            (audio / "abc.wav").write_bytes(b"RIFF")
            (audio / "index.json").write_text(
                '{"beats": {"c07": "abc.wav"}}', encoding="utf-8"
            )
            self.assertEqual(resolve_audio(root, "c07"), audio / "abc.wav")

    def test_missing_mapping_returns_none(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "refs" / "audio").mkdir(parents=True)
            self.assertIsNone(resolve_audio(root, "c07"))

    def test_index_path_traversal_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            audio = root / "refs" / "audio"
            audio.mkdir(parents=True)
            (audio / "index.json").write_text(
                '{"beats": {"c07": "../c07.wav"}}', encoding="utf-8"
            )
            self.assertIsNone(resolve_audio(root, "c07"))


if __name__ == "__main__":
    unittest.main()
