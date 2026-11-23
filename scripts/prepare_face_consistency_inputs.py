#!/usr/bin/env python3
"""Prepare full-frame, aspect-matched inputs for the c08/c18 face test."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps

TARGET_SIZE = (1280, 720)
SHOT_IDS = ("c08", "c18")


def prepare_frame(source_path: Path, output_path: Path) -> dict[str, object]:
    with Image.open(source_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    source_size = image.size

    background = ImageOps.fit(
        image,
        TARGET_SIZE,
        method=Image.Resampling.LANCZOS,
    ).filter(ImageFilter.GaussianBlur(radius=32))
    foreground = ImageOps.contain(
        image,
        TARGET_SIZE,
        method=Image.Resampling.LANCZOS,
    )
    left = (TARGET_SIZE[0] - foreground.width) // 2
    top = (TARGET_SIZE[1] - foreground.height) // 2

    frame = background.copy()
    mask = Image.new("L", foreground.size, 255).filter(ImageFilter.GaussianBlur(radius=12))
    frame.paste(foreground, (left, top), mask)
    frame.save(output_path, format="PNG", optimize=True)

    return {
        "source": str(source_path),
        "output": str(output_path),
        "sourceSize": source_size,
        "outputSize": TARGET_SIZE,
        "fullSourceContentBox": [
            left,
            top,
            left + foreground.width,
            top + foreground.height,
        ],
        "sourceContentRetained": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory for generated frames; source photos are not copied into the repository.",
    )
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    inputs = [
        (args.root / "startframes" / f"{shot_id}.png", args.output_dir / f"{shot_id}_face_test.png")
        for shot_id in SHOT_IDS
    ]
    for source, output in inputs:
        if not source.is_file():
            parser.error(f"missing source image: {source}")
        if output.exists() and not args.overwrite:
            parser.error(f"output exists (pass --overwrite to replace): {output}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = [prepare_frame(source, output) for source, output in inputs]

    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
