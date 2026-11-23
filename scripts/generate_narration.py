#!/usr/bin/env python3
"""Generate per-clip narration audio with Edge TTS.

Reads the compiled prompts/<id>.json configs, which already carry the
authored `narration` text (speaker-prefixed dialogue turns joined by
newlines). Each clip gets one WAV whose duration is padded or trimmed to the
clip duration so the final mux stays in sync.

Voices are chosen per speaker from the spec's cast:
  Faiz     -> en-IN-PrabhatNeural (male, Indian English)
  Juveriya -> en-IN-NeerjaNeural  (female, Indian English)

Output: audio/<id>.wav, plus audio/index.json mapping clip -> file.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
from pathlib import Path

DEFAULT_VOICES = {
    "faiz": "en-IN-PrabhatNeural",
    "juv": "en-IN-NeerjaNeural",
    "juveriya": "en-IN-NeerjaNeural",
    "ammi": "hi-IN-SwaraNeural",
}

SPEAKER_ALIASES = {
    "faiz": "faiz",
    "juveriya": "juv",
    "juv": "juv",
    "ammi": "ammi",
}


def load_configs(prompts_dir: Path) -> list[dict]:
    configs = []
    for path in sorted(prompts_dir.glob("c*.json")):
        try:
            configs.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError) as error:
            print(f"  skip {path.name}: {error}", file=sys.stderr)
    return configs


def split_turns(narration: str) -> list[tuple[str, str]]:
    """Split 'Speaker: text' lines into (speaker, text) pairs."""
    turns = []
    for line in narration.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            speaker, _, text = line.partition(":")
            speaker = speaker.strip().lower()
            text = text.strip()
            if text:
                turns.append((speaker, text))
        else:
            turns.append(("", line))
    return turns


async def synth(text: str, voice: str, dest: Path) -> None:
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(str(dest))


def probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(path),
        ],
        capture_output=True, text=True,
    )
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


def fit_duration(src: Path, dest: Path, target: float) -> None:
    """Pad with silence or trim so the audio matches the clip duration."""
    current = probe_duration(src)
    if current <= 0:
        raise RuntimeError(f"could not read duration of {src}")

    if current < target:
        pad = target - current
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(src),
                "-af", f"apad=pad_dur={pad:.3f}",
                "-t", f"{target:.3f}", str(dest),
            ],
            check=True, capture_output=True,
        )
    else:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(src),
                "-af", f"atrim=0:{target:.3f},asetpts=N/SR/TB",
                "-t", f"{target:.3f}", str(dest),
            ],
            check=True, capture_output=True,
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate per-clip narration audio")
    ap.add_argument("--prompts", default="prompts")
    ap.add_argument("--output", default="audio")
    ap.add_argument("--beats", nargs="+")
    ap.add_argument("--rate", default="+0%", help="Edge TTS rate, e.g. -10%%")
    ap.add_argument("--pitch", default="+0Hz")
    args = ap.parse_args()

    configs = load_configs(Path(args.prompts))
    if args.beats:
        configs = [c for c in configs if c.get("id") in set(args.beats)]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / ".tmp"
    tmp_dir.mkdir(exist_ok=True)

    index: dict[str, str] = {}
    made = 0

    for cfg in configs:
        clip_id = cfg.get("id")
        narration = (cfg.get("narration") or "").strip()
        if not clip_id or not narration:
            continue

        duration = float(cfg.get("duration") or 5)
        turns = split_turns(narration)
        if not turns:
            continue

        parts = []
        for i, (speaker, text) in enumerate(turns):
            key = SPEAKER_ALIASES.get(speaker, "juv")
            voice = DEFAULT_VOICES.get(key, DEFAULT_VOICES["juv"])
            part = tmp_dir / f"{clip_id}_{i:02d}.mp3"
            try:
                asyncio.run(synth(text, voice, part))
            except Exception as error:  # noqa: BLE001
                print(f"  {clip_id}: TTS failed for turn {i}: {error}", file=sys.stderr)
                continue
            parts.append(part)

        if not parts:
            continue

        # Concatenate turns, then fit to the clip duration.
        joined = tmp_dir / f"{clip_id}_joined.mp3"
        if len(parts) == 1:
            parts[0].replace(joined)
        else:
            listing = tmp_dir / f"{clip_id}_list.txt"
            # Absolute paths: ffmpeg resolves concat entries relative to the
            # list file's own directory, which breaks with nested paths.
            listing.write_text(
                "\n".join(f"file '{p.resolve().as_posix()}'" for p in parts),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", str(listing), "-c", "copy", str(joined),
                ],
                check=True, capture_output=True,
            )

        final = out_dir / f"{clip_id}.wav"
        fit_duration(joined, final, duration)
        index[clip_id] = final.name
        made += 1
        print(f"  {clip_id}: {len(parts)} turn(s) -> {final.name} ({duration:.0f}s)")

    (out_dir / "index.json").write_text(
        json.dumps({"beats": index}, indent=2), encoding="utf-8"
    )
    print(f"\nwrote {made} narration track(s) to {out_dir}")
    return 0 if made else 1


if __name__ == "__main__":
    sys.exit(main())
