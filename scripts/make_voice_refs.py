#!/usr/bin/env python3
"""Bootstrap Chatterbox voice-cloning references from Edge TTS.

Chatterbox clones a voice from a short reference clip. Recording three actors
is not always possible, so this generates a clean Hindi sample per character
with an age- and gender-matched Edge voice, then lets Chatterbox clone it.

Edge ships only two Hindi voices, so the three characters are separated by
pitch and rate:

  faiz  25, male    hi-IN-MadhurNeural, pitch -8Hz, rate +4%   young, low
  juv   19, female  hi-IN-SwaraNeural,  pitch +18Hz, rate +6%  young, bright
  ammi  56, female  hi-IN-SwaraNeural,  pitch -22Hz, rate -12% older, tired

The reference text is deliberately neutral. A reference clip teaches timbre,
not content, so feeding it the dirty dialogue would only make the clone
unstable.

Usage:
    python scripts/make_voice_refs.py
    python scripts/make_voice_refs.py --force
"""
from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

# Neutral Hindi, ~9 s at these rates. Long enough for Chatterbox to capture
# timbre, short enough to stay stable.
REFERENCE_TEXT = (
    "नमस्ते। मैं इस घर में रहता हूँ। आज शाम को बहुत काम था, "
    "इसलिए देर हो गई। अब मैं थोड़ा आराम करूँगा और फिर किताब पढ़ूँगा। "
    "आप सब कैसे हैं? सब ठीक है ना?"
)

# (voice, pitch, rate). Pitch and rate are what separate the two female
# characters, since Edge has only one Hindi female voice.
CHARACTERS = {
    "faiz": {
        "voice": "hi-IN-MadhurNeural",
        "pitch": "-8Hz",
        "rate": "+4%",
        "note": "25, male, Indian Muslim",
    },
    "juv": {
        "voice": "hi-IN-SwaraNeural",
        "pitch": "+18Hz",
        "rate": "+6%",
        "note": "19, female, Indian Muslim",
    },
    "ammi": {
        "voice": "hi-IN-SwaraNeural",
        "pitch": "-22Hz",
        "rate": "-12%",
        "note": "56, female, Indian Muslim",
    },
}


async def synth(text: str, voice: str, pitch: str, rate: str, dest: Path) -> None:
    import edge_tts

    communicate = edge_tts.Communicate(text, voice, pitch=pitch, rate=rate)
    await communicate.save(str(dest))


def normalise(src: Path, dest: Path) -> None:
    """Chatterbox wants mono 24 kHz, and the reference must not be hot.

    A reference that peaks near 0 dB makes the clone clip: the first test
    produced max_volume 0.0 dB. Loudness-normalise to -20 LUFS with a -3 dB
    peak ceiling so the clone has headroom.
    """
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-i", str(src),
            "-af", "loudnorm=I=-20:TP=-3:LRA=11",
            "-ac", "1", "-ar", "24000", "-c:a", "pcm_s16le", str(dest),
        ],
        check=True, capture_output=True,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Bootstrap Chatterbox voice references")
    ap.add_argument("--out", default="refs/voices")
    ap.add_argument("--force", action="store_true", help="Overwrite existing clips")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / ".tmp"
    tmp_dir.mkdir(exist_ok=True)

    made = 0
    for name, cfg in CHARACTERS.items():
        dest = out_dir / f"{name}.wav"
        if dest.is_file() and not args.force:
            print(f"  {name}: exists ({dest.stat().st_size // 1024} KB), skipped")
            continue
        raw = tmp_dir / f"{name}.mp3"
        try:
            asyncio.run(
                synth(REFERENCE_TEXT, cfg["voice"], cfg["pitch"], cfg["rate"], raw)
            )
            normalise(raw, dest)
        except Exception as error:  # noqa: BLE001
            print(f"  {name}: FAILED {error}", file=sys.stderr)
            continue
        finally:
            raw.unlink(missing_ok=True)
        print(f"  {name}: {cfg['note']} -> {dest.name} ({dest.stat().st_size // 1024} KB)")
        made += 1

    tmp_dir.rmdir() if not any(tmp_dir.iterdir()) else None
    print(f"\nwrote {made} reference clip(s) to {out_dir}")
    return 0 if made or all((out_dir / f"{n}.wav").is_file() for n in CHARACTERS) else 1


if __name__ == "__main__":
    sys.exit(main())
