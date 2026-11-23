#!/usr/bin/env python3
"""Compose a 16:9 start plate from a portrait reference.

The full portrait is preserved over a blurred, full-bleed copy of itself scaled
to the exact target resolution (docs/ai-drama-quality-pipeline.md:67). The
benchmark-locked variant of this technique lives in
scripts/prepare_face_consistency_inputs.py; this generic compositor never
mutates the source and asserts the exact output pixel dimensions.

Example:
  uv run python scripts/prep_source_frame.py --source ref.png --out plate.png --width 1280 --height 720
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps


def compose_plate(
    source_path: Path,
    output_path: Path,
    width: int,
    height: int,
) -> dict[str, object]:
    with Image.open(source_path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    source_size = image.size

    background = ImageOps.fit(
        image,
        (width, height),
        method=Image.Resampling.LANCZOS,
    ).filter(ImageFilter.GaussianBlur(radius=32))
    foreground = ImageOps.contain(
        image,
        (width, height),
        method=Image.Resampling.LANCZOS,
    )
    left = (width - foreground.width) // 2
    top = (height - foreground.height) // 2

    plate = background.copy()
    mask = Image.new("L", foreground.size, 255).filter(ImageFilter.GaussianBlur(radius=12))
    plate.paste(foreground, (left, top), mask)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plate.save(output_path, format="PNG", optimize=True)
    with Image.open(output_path) as check:
        if check.size != (width, height):
            raise RuntimeError(
                f"Composed plate is {check.size[0]}x{check.size[1]}; "
                f"expected exactly {width}x{height}",
            )

    return {
        "source": str(source_path),
        "output": str(output_path),
        "sourceSize": list(source_size),
        "outputSize": [width, height],
        "fullSourceContentBox": [
            left,
            top,
            left + foreground.width,
            top + foreground.height,
        ],
        "sourceContentRetained": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path, help="portrait reference PNG (never modified)")
    parser.add_argument("--out", required=True, type=Path, help="composed plate PNG to write")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    args = parser.parse_args(argv)
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be positive")
    if not args.source.is_file():
        parser.error(f"missing source image: {args.source}")

    result = compose_plate(args.source, args.out, args.width, args.height)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
