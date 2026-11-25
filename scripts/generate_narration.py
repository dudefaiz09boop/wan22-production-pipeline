#!/usr/bin/env python3
"""Generate per-clip narration audio.

Two engines, chosen by `--engine`:

  chatterbox  Resemble AI Chatterbox Multilingual (MIT, 0.5B). Hindi native,
              zero-shot voice cloning from a reference clip, and an
              `exaggeration` knob that is what makes moans and dirty talk
              land instead of reading flat. Default.
  edge        Edge TTS. No GPU, no download, but a fixed catalogue of
              announcer voices with no emotion control.

Reads the compiled prompts/<id>.json configs, which already carry the
authored `narration` text (speaker-prefixed dialogue turns joined by
newlines). Each clip gets one WAV whose duration is padded or trimmed to the
clip duration so the final mux stays in sync.

Chatterbox has no Urdu, so Urdu turns fall back to Edge's ur-PK voices. The
script of the text decides the engine and the voice, not the spec's
`language` field, which is unreliable.

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

# Voice per (speaker, script) for the Edge fallback. Romanized Hindi/Urdu
# stays on the Indian-English voices, which pronounce it correctly; only real
# Devanagari/Arabic script switches to a native voice.
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

# Chatterbox language ids. Only `hi` is used; Urdu has no Chatterbox support
# and falls through to Edge.
CHATTERBOX_LANG = {"devanagari": "hi", "latin": "hi"}

# Per-speaker delivery. Faiz is low and flat until he is close to coming;
# Juveriya is breathier and pushed harder. `exaggeration` is the emotion
# dial -- 0.5 is neutral, 0.7+ is dramatic. Moans get the top of the range.
SPEAKER_STYLE = {
    "faiz": {"exaggeration": 0.62, "cfg_weight": 0.35},
    "juv": {"exaggeration": 0.72, "cfg_weight": 0.30},
    "ammi": {"exaggeration": 0.55, "cfg_weight": 0.40},
}
MOAN_STYLE = {"exaggeration": 0.88, "cfg_weight": 0.25}

# A turn is treated as a moan when it is short and made of these. Both the
# Latin and Devanagari spellings are listed: the dialogue is Devanagari now,
# and a Latin-only list silently misses every moan.
MOAN_TOKENS = frozenset(
    "ahh ah aah oh ooh haan haanh mat ruk aur bas dekho fuck uff hai "
    "please jaldi zor".split()
) | frozenset(
    "आह आ आह्ह हाँ हां मत रुक और बस देखो ओह उफ़ है प्लीज़ जल्दी ज़ोर".split()
)

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


def is_moan(text: str) -> bool:
    """True when a turn is a moan rather than a spoken line.

    Moans are short and made only of interjections. They get the top of the
    exaggeration range so they do not read as flat speech.
    """
    tokens = [t for t in text.lower().replace("\u2014", " ").split() if t]
    if not tokens or len(tokens) > 6:
        return False
    return all(t.strip(",.") in MOAN_TOKENS for t in tokens)


def style_for(speaker: str, text: str) -> dict:
    """Delivery settings for one turn."""
    if is_moan(text):
        return dict(MOAN_STYLE)
    return dict(SPEAKER_STYLE.get(SPEAKER_ALIASES.get(speaker, "juv"), SPEAKER_STYLE["juv"]))


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


class ChatterboxEngine:
    """Lazy singleton around Chatterbox Multilingual.

    Loading is deferred until the first Hindi turn so an all-Urdu run never
    pays the model download. The model is 0.5B and runs on GPU; on CPU it is
    slow but works.
    """

    def __init__(self, refs: dict[str, Path], device: str = "cuda"):
        self.refs = refs
        self.device = device
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        import torch

        # Chatterbox builds a Chinese word segmenter at load time. It is only
        # used for Chinese text, but pkuseg loads a ~35 MB msgpack model into
        # RAM and raises MemoryError on a small box. We generate Hindi, so
        # skip it.
        try:
            import chatterbox.models.tokenizers.tokenizer as _tok

            _tok.ChineseCangjieConverter._init_segmenter = (
                lambda self: setattr(self, "segmenter", None)
            )
        except (ImportError, AttributeError):
            pass

        from chatterbox.mtl_tts import ChatterboxMultilingualTTS

        device = self.device
        if device == "cuda" and not torch.cuda.is_available():
            print("  chatterbox: no CUDA, falling back to CPU (slow)", file=sys.stderr)
            device = "cpu"
        # Pass a plain string, not torch.device: from_local compares
        # `device in ["cpu", "mps"]` and a torch.device never matches, so the
        # CPU map_location is skipped and a CUDA-saved checkpoint fails to
        # deserialize on a CPU-only box.
        self._model = ChatterboxMultilingualTTS.from_pretrained(device=device)
        return self._model

    def synth(self, text: str, speaker: str, dest: Path) -> None:
        import torchaudio as ta

        model = self._load()
        style = style_for(speaker, text)
        ref = self.refs.get(SPEAKER_ALIASES.get(speaker, "juv"))
        wav = model.generate(
            text,
            language_id="hi",
            audio_prompt_path=str(ref) if ref and ref.is_file() else None,
            **style,
        )
        ta.save(str(dest), wav, model.sr)


def normalise_take(src: Path, dest: Path) -> None:
    """Loudness-normalise a generated take.

    Chatterbox output peaks at 0 dBFS, which clips once it is mixed with room
    tone and breath. Normalise to -20 LUFS with a -3 dB ceiling so the mix has
    headroom.
    """
    subprocess.run(
        [
            "ffmpeg", "-y", "-v", "error", "-i", str(src),
            "-af", "loudnorm=I=-20:TP=-3:LRA=11",
            "-ar", "24000", "-ac", "1", str(dest),
        ],
        check=True, capture_output=True,
    )


def synth_turn(
    text: str,
    speaker: str,
    dest: Path,
    engine: str,
    chatterbox: ChatterboxEngine | None,
) -> None:
    """Synthesise one turn with the chosen engine.

    Chatterbox handles Hindi and Latin. Urdu has no Chatterbox support, so
    Arabic-script turns always go to Edge regardless of `--engine`.
    """
    script = detect_script(text)
    if engine == "chatterbox" and chatterbox is not None and script != "arabic":
        raw = dest.with_name(dest.stem + ".raw.wav")
        try:
            chatterbox.synth(text, speaker, raw)
            normalise_take(raw, dest)
        finally:
            raw.unlink(missing_ok=True)
        return
    asyncio.run(synth(text, voice_for(speaker, text), dest))


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
            joined = tmp_dir / f"{clip_id}_{speaker}_join.wav"
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
        "--engine",
        choices=["chatterbox", "edge"],
        default="chatterbox",
        help="chatterbox: Hindi voice cloning with emotion control. edge: fallback",
    )
    ap.add_argument(
        "--refs",
        default="refs/voices",
        help="Directory of <speaker>.wav reference clips for Chatterbox cloning",
    )
    ap.add_argument(
        "--device",
        default="cuda",
        help="Torch device for Chatterbox (cuda or cpu)",
    )
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

    chatterbox = None
    if args.engine == "chatterbox":
        refs_dir = Path(args.refs)
        refs = {
            key: refs_dir / f"{key}.wav"
            for key in ("faiz", "juv", "ammi")
        }
        missing = [k for k, v in refs.items() if not v.is_file()]
        if missing:
            print(
                f"  chatterbox: no reference clip for {', '.join(missing)}; "
                f"those speakers use the default voice",
                file=sys.stderr,
            )
        chatterbox = ChatterboxEngine(refs, args.device)

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
            part = tmp_dir / f"{clip_id}_{i:02d}.wav"
            try:
                synth_turn(text, speaker, part, args.engine, chatterbox)
            except Exception as error:  # noqa: BLE001
                print(f"  {clip_id}: TTS failed for turn {i}: {error}", file=sys.stderr)
                continue
            parts.append(part)

        if not parts:
            continue

        # Concatenate turns, then fit to the clip duration.
        joined = tmp_dir / f"{clip_id}_joined.wav"
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
