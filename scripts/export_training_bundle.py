#!/usr/bin/env python3
"""Package per-character training data for a hosted trainer (ModelScope/Civision).

The hosted trainers want one zip per concept, with images and captions side by
side. This validates the split first, because a hosted run is slow and a mixed
dataset is only visible in the output.

Refuses to package when:
- a crop appears in both character sets (the mixing bug),
- a character set is empty,
- an image has no caption, or a caption has no image,
- a caption does not start with its own trigger word.

Usage:
    python scripts/export_training_bundle.py --data-dir training_data_new --out-dir bundles
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")
CHARACTERS = ("faiz", "juv")

# Hosted trainers read the trigger from the caption prefix, so it has to be
# there and it has to be the right one.
MIN_IMAGES = 15


def _digest(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _images(directory: Path) -> list[Path]:
    return sorted(
        p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS
    )


def validate(data_dir: Path) -> tuple[dict[str, list[Path]], list[str]]:
    """Return the per-character image lists and a list of problems."""
    problems: list[str] = []
    per_character: dict[str, list[Path]] = {}

    for character in CHARACTERS:
        directory = data_dir / character
        if not directory.is_dir():
            problems.append(f"missing directory: {directory}")
            per_character[character] = []
            continue

        images = _images(directory)
        per_character[character] = images

        if len(images) < MIN_IMAGES:
            problems.append(
                f"{character}: only {len(images)} images (want >= {MIN_IMAGES})"
            )

        for image in images:
            caption = image.with_suffix(".txt")
            if not caption.is_file():
                problems.append(f"{character}: no caption for {image.name}")
                continue
            text = caption.read_text(encoding="utf-8").strip()
            if not text:
                problems.append(f"{character}: empty caption for {image.name}")
            elif not text.lower().startswith(character):
                problems.append(
                    f"{character}: caption for {image.name} does not start with "
                    f"'{character}' (got {text[:40]!r})"
                )

        for caption in directory.glob("*.txt"):
            if not any(
                caption.with_suffix(ext).is_file() for ext in IMAGE_EXTENSIONS
            ):
                problems.append(f"{character}: caption without image: {caption.name}")

    # The mixing check: the same bytes must not sit in both sets.
    if all(per_character.get(c) for c in CHARACTERS):
        digests = {
            character: {_digest(p): p.name for p in per_character[character]}
            for character in CHARACTERS
        }
        shared = set(digests["faiz"]) & set(digests["juv"])
        if shared:
            names = sorted(digests["faiz"][h] for h in shared)[:5]
            problems.append(
                f"{len(shared)} image(s) appear in BOTH sets, e.g. {names}. "
                "Re-run scripts/prepare_training_data.py."
            )

    return per_character, problems


def write_bundle(
    character: str,
    images: list[Path],
    out_dir: Path,
    trigger: str,
) -> Path:
    """Zip one character's images and captions, plus a trainer config."""
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = out_dir / f"{character}_dataset.zip"

    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as archive:
        for image in images:
            archive.write(image, f"{character}/{image.name}")
            caption = image.with_suffix(".txt")
            if caption.is_file():
                archive.write(caption, f"{character}/{caption.name}")

        config = {
            "concept": character,
            "trigger": trigger,
            "instance_prompt": f"{trigger}, south asian, detailed face, photorealistic",
            "image_count": len(images),
            "network_dim": 32,
            "network_alpha": 16,
            "resolution": 1024,
            "max_train_steps": 1200,
            "learning_rate": 1e-4,
            "text_encoder_lr": 5e-5,
            "train_batch_size": 1,
            "gradient_accumulation_steps": 2,
            "lr_scheduler": "cosine",
            "optimizer": "AdamW8bit",
            "save_every_n_steps": 400,
            "note": (
                "One character per LoRA. network_alpha is half of network_dim "
                "because the dataset is small; alpha == dim over-saturates."
            ),
        }
        archive.writestr(
            f"{character}/train_config.json", json.dumps(config, indent=2)
        )

    return bundle


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Validate and package per-character training data for a hosted trainer"
    )
    ap.add_argument("--data-dir", default="training_data_new", help="Prepared training data")
    ap.add_argument("--out-dir", default="bundles", help="Where to write the zips")
    ap.add_argument("--trigger", default=None, help="Override the trigger word (default: character name)")
    ap.add_argument("--allow-problems", action="store_true", help="Package even when validation fails")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        print(f"missing data dir: {data_dir}", file=sys.stderr)
        return 1

    per_character, problems = validate(data_dir)

    for character in CHARACTERS:
        print(f"{character}: {len(per_character.get(character, []))} images")

    if problems:
        print(f"\n{len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        if not args.allow_problems:
            print("\nNot packaging. Fix the above, or pass --allow-problems.")
            return 1
    else:
        print("\nvalidation OK: no shared images, captions present and prefixed")

    out_dir = Path(args.out_dir)
    for character in CHARACTERS:
        images = per_character.get(character) or []
        if not images:
            continue
        trigger = args.trigger or character
        bundle = write_bundle(character, images, out_dir, trigger)
        print(f"wrote {bundle} ({bundle.stat().st_size} bytes)")

    print("\nUpload each zip as its own dataset/concept in the hosted trainer.")
    print("Train one character per run; do not merge them into one concept.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
