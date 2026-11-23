#!/usr/bin/env python3
"""Prepare training data for LoRA training on Kaggle.

Extracts faces from startframes using InsightFace, combines with existing
training datasets, creates caption files and metadata for Kohya-ss/sd-scripts.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image
from tqdm import tqdm


# Character mapping from beats.yaml
CHARACTER_MAP = {
    "faiz": "faiz",
    "juv": "juv",
    "juveriya": "juv",
}

# Beat to character mapping (which character appears in each beat)
BEAT_CHARACTERS = {
    "c01": ["faiz", "juv"], "c02": ["faiz", "juv"], "c03": ["faiz", "juv"],
    "c04": ["faiz", "juv"], "c05": ["faiz", "juv"], "c06": ["faiz", "juv"],
    "c07": ["faiz", "juv"], "c08": ["faiz", "juv"], "c09": ["faiz", "juv"],
    "c10": ["faiz", "juv"], "c11": ["faiz", "juv"], "c12": ["faiz", "juv"],
    "c13": ["faiz", "juv"], "c14": ["faiz", "juv"], "c15": ["faiz", "juv"],
    "c16": ["faiz", "juv"], "c17": ["faiz", "juv"], "c18": ["faiz", "juv"],
    "c19": ["faiz", "juv"], "c20": ["faiz", "juv"], "c21": ["faiz", "juv"],
    "c22": ["faiz", "juv"], "c23": ["faiz", "juv"], "c24": ["faiz", "juv"],
    "c25": ["faiz", "juv"], "c26": ["faiz", "juv"], "c27": ["faiz", "juv"],
    "c28": ["faiz", "juv"], "c29": ["faiz", "juv"], "c30": ["faiz", "juv"],
    "c31": ["faiz", "juv"], "c32": ["faiz", "juv"], "c33": ["faiz", "juv"],
    "c34": ["faiz", "juv"], "c35": ["faiz", "juv"], "c36": ["faiz", "juv"],
    "c37": ["faiz", "juv"], "c38": ["faiz", "juv"], "c39": ["faiz", "juv"],
    "c40": ["faiz", "juv"], "c41": ["faiz", "juv"], "c42": ["faiz", "juv"],
    "c43": ["faiz", "juv"], "c44": ["faiz", "juv"], "c45": ["faiz", "juv"],
    "c46": ["faiz", "juv"], "c47": ["faiz", "juv"], "c48": ["faiz", "juv"],
    "c49": ["faiz", "juv"], "c50": ["faiz", "juv"], "c51": ["faiz", "juv"],
    "c52": ["faiz", "juv"], "c53": ["faiz", "juv"], "c54": ["faiz", "juv"],
    "c55": ["faiz", "juv"], "c56": ["faiz", "juv"], "c57": ["faiz", "juv"],
    "c58": ["faiz", "juv"], "c59": ["faiz", "juv"], "c60": ["faiz", "juv"],
    "c61": ["faiz", "juv"], "c62": ["faiz", "juv"], "c63": ["faiz", "juv"],
    "c64": ["faiz", "juv"], "c65": ["faiz", "juv"], "c66": ["faiz", "juv"],
    "c67": ["faiz", "juv"], "c68": ["faiz", "juv"], "c69": ["faiz", "juv"],
    "c70": ["faiz", "juv"],
}


def load_insightface() -> FaceAnalysis:
    """Initialize InsightFace for face detection and embedding."""
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def extract_faces_from_image(
    app: FaceAnalysis,
    image_path: Path,
    output_dir: Path,
    character: str,
    min_face_size: int = 128,
    max_faces: int = 3,
) -> list[dict]:
    """Extract and save face crops from an image."""
    img = cv2.imread(str(image_path))
    if img is None:
        return []

    faces = app.get(img)
    if not faces:
        return []

    # Sort by face size (largest first)
    faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)

    results = []
    for i, face in enumerate(faces[:max_faces]):
        x1, y1, x2, y2 = map(int, face.bbox)
        w, h = x2 - x1, y2 - y1

        if w < min_face_size or h < min_face_size:
            continue

        # Add padding (20%)
        pad_w, pad_h = int(w * 0.2), int(h * 0.2)
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(img.shape[1], x2 + pad_w)
        y2 = min(img.shape[0], y2 + pad_h)

        face_crop = img[y1:y2, x1:x2]
        if face_crop.size == 0:
            continue

        # Save face crop
        output_name = f"{character}_{image_path.stem}_face{i:02d}.jpg"
        output_path = output_dir / output_name
        cv2.imwrite(str(output_path), face_crop, [cv2.IMWRITE_JPEG_QUALITY, 95])

        results.append({
            "file": output_name,
            "character": character,
            "source": image_path.name,
            "bbox": [x1, y1, x2, y2],
            "embedding": face.embedding.tolist(),
        })

    return results


def create_caption_file(image_path: Path, character: str, output_dir: Path) -> None:
    """Create caption file for training image."""
    caption = f"{character}, south asian, detailed face, photorealistic"
    caption_path = output_dir / f"{image_path.stem}.txt"
    caption_path.write_text(caption, encoding="utf-8")


def copy_existing_dataset(src_dir: Path, dst_dir: Path, character: str) -> int:
    """Copy existing training dataset images."""
    count = 0
    for ext in (".jpg", ".jpeg", ".png", ".webp"):
        for img_path in src_dir.glob(f"*{ext}"):
            dst_name = f"{character}_{img_path.name}"
            shutil.copy2(img_path, dst_dir / dst_name)
            create_caption_file(dst_dir / dst_name, character, dst_dir)
            count += 1
    return count


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare LoRA training data from startframes")
    ap.add_argument("--startframes-dir", required=True, help="Directory with c01-c70.png startframes")
    ap.add_argument("--output-dir", required=True, help="Output directory for training data")
    ap.add_argument("--existing-datasets", nargs="+", default=[], help="Existing dataset dirs (faiz, juv)")
    ap.add_argument("--min-face-size", type=int, default=128, help="Minimum face size to extract")
    ap.add_argument("--max-faces-per-frame", type=int, default=2, help="Max faces per startframe")
    ap.add_argument("--device", default="cuda", help="Device for InsightFace")
    args = ap.parse_args()

    startframes_dir = Path(args.startframes_dir)
    output_dir = Path(args.output_dir)

    # Create output structure
    faiz_dir = output_dir / "faiz"
    juv_dir = output_dir / "juv"
    combined_dir = output_dir / "combined"
    for d in (faiz_dir, juv_dir, combined_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Load InsightFace
    print("Loading InsightFace...")
    app = load_insightface()

    # Process startframes
    print(f"Processing startframes from {startframes_dir}...")
    startframe_files = sorted(startframes_dir.glob("c*.png"))
    print(f"Found {len(startframe_files)} startframes")

    all_faces = {"faiz": [], "juv": []}

    for sf_path in tqdm(startframe_files, desc="Extracting faces"):
        beat_id = sf_path.stem
        characters = BEAT_CHARACTERS.get(beat_id, ["faiz", "juv"])

        for character in characters:
            faces = extract_faces_from_image(
                app, sf_path, faiz_dir if character == "faiz" else juv_dir,
                character, args.min_face_size, args.max_faces_per_frame
            )
            all_faces[character].extend(faces)

            # Create caption files
            for face in faces:
                face_path = (faiz_dir if character == "faiz" else juv_dir) / face["file"]
                create_caption_file(face_path, character, faiz_dir if character == "faiz" else juv_dir)

    # Copy existing datasets
    for dataset_name in args.existing_datasets:
        dataset_path = Path(dataset_name)
        if dataset_path.exists():
            if "faiz" in dataset_name.lower():
                count = copy_existing_dataset(dataset_path, faiz_dir, "faiz")
                print(f"Copied {count} images from {dataset_name} to faiz")
            elif "juv" in dataset_name.lower() or "juveriya" in dataset_name.lower():
                count = copy_existing_dataset(dataset_path, juv_dir, "juv")
                print(f"Copied {count} images from {dataset_name} to juv")

    # Create combined dataset (for both LoRA)
    for character in ["faiz", "juv"]:
        char_dir = faiz_dir if character == "faiz" else juv_dir
        for img_path in char_dir.glob("*.jpg"):
            shutil.copy2(img_path, combined_dir / img_path.name)
            create_caption_file(combined_dir / img_path.name, character, combined_dir)

    # Save metadata
    metadata = {
        "faiz_count": len(list(faiz_dir.glob("*.jpg"))),
        "juv_count": len(list(juv_dir.glob("*.jpg"))),
        "combined_count": len(list(combined_dir.glob("*.jpg"))),
        "startframes_processed": len(startframe_files),
        "faces_extracted": {k: len(v) for k, v in all_faces.items()},
    }

    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nTraining data prepared in {output_dir}:")
    print(f"  Faiz: {metadata['faiz_count']} images")
    print(f"  Juv:  {metadata['juv_count']} images")
    print(f"  Combined: {metadata['combined_count']} images")

    return 0


if __name__ == "__main__":
    sys.exit(main())