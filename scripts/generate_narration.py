#!/usr/bin/env python3
"""Generate per-clip narration audio with Edge TTS.

Reads the compiled prompts/<id>.json configs, which already carry the
authored `narration` text (speaker-prefixed dialogue turns joined by
newlines). Each clip gets one WAV whose duration is padded or trimmed to the
clip duration so the final mux stays in sync.

Voices are chosen per turn from the speaker and the script of the text:
  Latin script      -> en-IN-PrabhatNeural / en-IN-NeerjaNeural
  Devanagari        -> hi-IN-MadhurNeural / hi-IN-SwaraNeural
  Arabic (Urdu)     -> ur-PK-AsadNeural / ur-PK-UzmaNeural

The spec's `language` field is not used for voice selection: it is unreliable
(c18 is tagged `hi` but its `say` is English). The script of the text decides.

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

# Voice per (speaker, script). The spec's `language` field is not reliable --
# c18 is tagged `hi` but its `say` is English -- so the script of the text
# itself decides the voice. Romanized Hindi/Urdu stays on the Indian-English
# voices, which pronounce it correctly; only real Devanagari/Arabic script
# switches to a native voice.
SCRIPT_VOICES = {
    "devanagari": {
        "faiz": "hi-IN-MadhurNeural",
        "juv": "hi-IN-SwaraNeural",
        "juveriya": "hi-IN-SwaraNeural",
        "ammi": "hi-IN-SwaraNeural",
    },
    "arabic": {
        "faiz": "ur-PK-AsadNeural",
        "juv": "ur-PK-UzmaNeural",
        "juveriya": "ur-PK-UzmaNeural",
        "ammi": "ur-PK-UzmaNeural",
    },
}

SPEAKER_ALIASES = {
    "faiz": "faiz",
    "juveriya": "juv",
    "juv": "juv",
    "ammi": "ammi",
}


def detect_script(text: str) -> str:
    """Classify text as devanagari, arabic, or latin.

    Devanagari U+0900-U+097F; Arabic U+0600-U+06FF (covers Urdu).
    """
    for char in text:
        code = ord(char)
        if 0x0900 <= code <= 0x097F:
            return "devanagari"
        if 0x0600 <= code <= 0x06FF:
            return "arabic"
    return "latin"


def voice_for(speaker: str, text: str) -> str:
    """Pick the voice for one turn from its speaker and its script."""
    key = SPEAKER_ALIASES.get(speaker, "juv")
    script = detect_script(text)
    if script in SCRIPT_VOICES:
        return SCRIPT_VOICES[script].get(key, SCRIPT_VOICES[script]["juv"])
    return DEFAULT_VOICES.get(key, DEFAULT_VOICES["juv"])


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


def ensure_assets(room_tone: Path, breath: Path) -> None:
    """Create the room-tone and breath beds if they are not on disk.

    `audio/` is gitignored, so these never travel with the repo. Generating
    them here keeps the polish step working on a fresh instance instead of
    silently degrading to dry TTS.
    """
    if not room_tone.is_file():
        room_tone.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-f", "lavfi", "-i",
                "anoisesrc=color=brown:duration=8:sample_rate=48000:amplitude=0.35",
                "-af",
                "highpass=f=60,lowpass=f=1400,volume=-14dB,"
                "afade=t=in:d=0.05,afade=t=out:st=7.95:d=0.05",
                "-ac", "2", str(room_tone),
            ],
            check=True, capture_output=True,
        )
    if not breath.is_file():
        breath.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "ffmpeg", "-y", "-v", "error",
                "-f", "lavfi", "-i",
                "anoisesrc=color=pink:duration=0.9:sample_rate=48000:amplitude=0.5",
                "-af",
                "highpass=f=400,lowpass=f=5000,"
                "afade=t=in:d=0.18,afade=t=out:st=0.45:d=0.45,volume=-8dB",
                "-ac", "2", str(breath),
            ],
            check=True, capture_output=True,
        )


def polish(
    src: Path,
    dest: Path,
    room_tone: Path | None,
    reverb: float,
    breath: Path | None,
) -> None:
    """Make the TTS read as a person in a room rather than a clean studio.

    Three layers, all optional:

    * ``room_tone`` -- a looped bed at -30 dB. Dry TTS has a digital-silence
      floor; a real room never does, and the ear reads the difference as
      "fake" long before it can name it.
    * ``reverb`` -- a short aecho tail. The flat is small and hard-surfaced,
      so the tail is tight, not a hall.
    * ``breath`` -- one breath sample laid under the head of the clip, so the
      line starts from an inhale instead of from nothing.
    """
    if room_tone is None and reverb <= 0 and breath is None:
        src.replace(dest)
        return

    cmd = ["ffmpeg", "-y", "-i", str(src)]
    filters: list[str] = []
    inputs = 1

    if breath is not None and breath.is_file():
        cmd += ["-i", str(breath)]
        filters.append(
            f"[{inputs}:a]volume=0.35,adelay=120|120[br]"
        )
        inputs += 1

    if room_tone is not None and room_tone.is_file():
        cmd += ["-stream_loop", "-1", "-i", str(room_tone)]
        filters.append(f"[{inputs}:a]volume=-30dB[rt]")
        inputs += 1

    if reverb > 0:
        # 60 ms delay, 0.25 decay: a small tiled room, not a cathedral.
        filters.append(
            f"[0:a]aecho=0.8:0.7:60:{reverb:.2f}[dry]"
        )
        base = "[dry]"
    else:
        base = "[0:a]"

    mix_inputs = [base]
    if breath is not None and breath.is_file():
        mix_inputs.append("[br]")
    if room_tone is not None and room_tone.is_file():
        mix_inputs.append("[rt]")

    if len(mix_inputs) == 1:
        filters.append(f"{base}anull[out]")
    else:
        filters.append(
            "".join(mix_inputs)
            + f"amix=inputs={len(mix_inputs)}:duration=first:"
            "dropout_transition=0:normalize=0[out]"
        )

    cmd += [
        "-filter_complex", ";".join(filters),
        "-map", "[out]", "-ar", "48000", "-ac", "2", str(dest),
    ]
    subprocess.run(cmd, check=True, capture_output=True)


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


def speaker_stems(
    turns: list[tuple[str, str]],
    parts: list[Path],
    clip_id: str,
    tmp_dir: Path,
    out_dir: Path,
    duration: float,
) -> dict[str, Path]:
    """Build one audio stem per speaker, each padded to the clip duration.

    InfiniteTalk's two_speakers mode needs a separate audio stream per person
    plus a mask per person. A single mixed track cannot drive it: both mouths
    would follow both voices. Stems are padded to the full clip length so the
    two streams stay sample-aligned.
    """
    by_speaker: dict[str, list[Path]] = {}
    for (speaker, _text), part in zip(turns, parts):
        key = SPEAKER_ALIASES.get(speaker, "juv")
        by_speaker.setdefault(key, []).append(part)

    stems: dict[str, Path] = {}
    for speaker, files in by_speaker.items():
        if len(files) == 1:
            joined = files[0]
        else:
            joined = tmp_dir / f"{clip_id}_{speaker}_join.mp3"
            listing = tmp_dir / f"{clip_id}_{speaker}_list.txt"
            listing.write_text(
                "\n".join(f"file '{p.resolve().as_posix()}'" for p in files),
                encoding="utf-8",
            )
            subprocess.run(
                [
                    "ffmpeg", "-y", "-f", "concat", "-safe", "0",
                    "-i", str(listing), "-c", "copy", str(joined),
                ],
                check=True, capture_output=True,
            )
        stem = out_dir / f"{clip_id}_{speaker}.wav"
        fit_duration(joined, stem, duration)
        stems[speaker] = stem
    return stems


def index_entry(index: dict, clip_id: str) -> tuple[str | None, dict[str, str]]:
    """Normalise an index value to (mix filename, {speaker: stem filename}).

    Older indexes store a bare filename; two-speaker clips store
    ``{"mix": ..., "stems": {...}}``.
    """
    value = index.get(clip_id)
    if isinstance(value, str):
        return value, {}
    if isinstance(value, dict):
        mix = value.get("mix")
        stems = value.get("stems")
        return (
            mix if isinstance(mix, str) else None,
            stems if isinstance(stems, dict) else {},
        )
    return None, {}


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate per-clip narration audio")
    ap.add_argument("--prompts", default="prompts")
    ap.add_argument("--output", default="audio")
    ap.add_argument("--beats", nargs="+")
    ap.add_argument("--rate", default="+0%", help="Edge TTS rate, e.g. -10%%")
    ap.add_argument("--pitch", default="+0Hz")
    ap.add_argument(
        "--room-tone",
        default="audio/room_tone.wav",
        help="Loopable room tone bed; skipped when absent",
    )
    ap.add_argument(
        "--reverb",
        type=float,
        default=0.25,
        help="Echo decay for the room tail; 0 disables",
    )
    ap.add_argument(
        "--breath",
        default="audio/breath.wav",
        help="Breath sample laid under the clip head; skipped when absent",
    )
    args = ap.parse_args()

    configs = load_configs(Path(args.prompts))
    if args.beats:
        configs = [c for c in configs if c.get("id") in set(args.beats)]

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = out_dir / ".tmp"
    tmp_dir.mkdir(exist_ok=True)

    room_tone = Path(args.room_tone) if args.room_tone else None
    breath = Path(args.breath) if args.breath else None
    if room_tone or breath:
        try:
            ensure_assets(
                room_tone or Path("audio/room_tone.wav"),
                breath or Path("audio/breath.wav"),
            )
        except (OSError, subprocess.CalledProcessError) as error:
            print(f"  could not create audio beds: {error}", file=sys.stderr)

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
            voice = voice_for(speaker, text)
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
        polished = tmp_dir / f"{clip_id}_polished.wav"
        try:
            polish(
                joined,
                polished,
                room_tone,
                args.reverb,
                breath,
            )
        except subprocess.CalledProcessError as error:
            print(f"  {clip_id}: polish failed, using dry audio: {error}", file=sys.stderr)
            polished = joined
        fit_duration(polished, final, duration)

        # Two-speaker clips also get per-speaker stems for InfiniteTalk.
        speakers = {SPEAKER_ALIASES.get(s, "juv") for s, _ in turns}
        if len(speakers) > 1:
            stems = speaker_stems(turns, parts, clip_id, tmp_dir, out_dir, duration)
            index[clip_id] = {
                "mix": final.name,
                "stems": {k: v.name for k, v in stems.items()},
            }
        else:
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
