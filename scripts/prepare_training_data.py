#!/usr/bin/env python3
"""Prepare training data for LoRA training on Kaggle.

Extracts faces from startframes using InsightFace, combines with existing
training datasets, creates caption files and metadata for Kohya-ss/sd-scripts.

Image REF folder structure (C:/Users/FAIZ SAYYED/OneDrive/Desktop/Images REF):
- c01.png to c08.png -> Faiz
- c09.png to c23.png -> Juveriya  
- c24.png, c25.png (Faiz (1).png, Faiz (2).png) -> Faiz

Startframes (D:/AI Pipeline/startframes/c01-c70.png): Mixed, extract faces for both characters
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


# Character mapping for Images REF folder
IMAGES_REF_MAPPING = {
    # c01-c08: Faiz
    "c01": "faiz", "c02": "faiz", "c03": "faiz", "c04": "faiz",
    "c05": "faiz", "c06": "faiz", "c07": "faiz", "c08": "faiz",
    # c09-c23: Juveriya
    "c09": "juv", "c10": "juv", "c11": "juv", "c12": "juv",
    "c13": "juv", "c14": "juv", "c15": "juv", "c16": "juv",
    "c17": "juv", "c18": "juv", "c19": "juv", "c20": "juv",
    "c21": "juv", "c22": "juv", "c23": "juv",
    # c24-c25: Faiz (Faiz (1).png, Faiz (2).png)
    "c24": "faiz", "c25": "faiz",
    # Faiz (1).png, Faiz (2).png
    "faiz (1)": "faiz", "faiz (2)": "faiz",
}

# Beat to character mapping for startframes (which character appears in each beat)
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


def process_images_ref_folder(
    app: FaceAnalysis,
    images_ref_dir: Path,
    faiz_dir: Path,
    juv_dir: Path,
    min_face_size: int = 128,
    max_faces: int = 3,
) -> dict:
    """Process Images REF folder with known character mapping."""
    print(f"Processing Images REF from {images_ref_dir}...")
    
    results = {"faiz": [], "juv": []}
    
    for img_path in images_ref_dir.glob("*.png"):
        stem = img_path.stem.lower()
        character = IMAGES_REF_MAPPING.get(stem)
        
        if not character:
            print(f"  Skipping {img_path.name} - no character mapping")
            continue
            
        print(f"  Processing {img_path.name} -> {character}")
        faces = extract_faces_from_image(
            app, img_path, 
            faiz_dir if character == "faiz" else juv_dir,
            character, min_face_size, max_faces
        )
        results[character].extend(faces)
        
        # Create caption files
        for face in faces:
            face_path = (faiz_dir if character == "faiz" else juv_dir) / face["file"]
            create_caption_file(face_path, character, faiz_dir if character == "faiz" else juv_dir)
    
    return results


def process_startframes(
    app: FaceAnalysis,
    startframes_dir: Path,
    faiz_dir: Path,
    juv_dir: Path,
    min_face_size: int = 128,
    max_faces: int = 2,
) -> dict:
    """Process startframes - extract faces for both characters."""
    print(f"Processing startframes from {startframes_dir}...")
    startframe_files = sorted(startframes_dir.glob("c*.png"))
    print(f"Found {len(startframe_files)} startframes")

    results = {"faiz": [], "juv": []}

    for sf_path in tqdm(startframe_files, desc="Extracting faces from startframes"):
        beat_id = sf_path.stem
        characters = BEAT_CHARACTERS.get(beat_id, ["faiz", "juv"])

        for character in characters:
            faces = extract_faces_from_image(
                app, sf_path, 
                faiz_dir if character == "faiz" else juv_dir,
                character, min_face_size, max_faces
            )
            results[character].extend(faces)

            # Create caption files
            for face in faces:
                face_path = (faiz_dir if character == "faiz" else juv_dir) / face["file"]
                create_caption_file(face_path, character, faiz_dir if character == "faiz" else juv_dir)

    return results


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
    ap = argparse.ArgumentParser(description="Prepare LoRA training data from startframes + Images REF")
    ap.add_argument("--images-ref-dir", required=True, help="Images REF folder (c01-c25 with known mapping)")
    ap.add_argument("--startframes-dir", required=True, help="Startframes folder (c01-c70 mixed)")
    ap.add_argument("--output-dir", required=True, help="Output directory for training data")
    ap.add_argument("--existing-datasets", nargs="+", default=[], help="Existing dataset dirs (faiz, juv)")
    ap.add_argument("--min-face-size", type=int, default=128, help="Minimum face size to extract")
    ap.add_argument("--max-faces-ref", type=int, default=3, help="Max faces per Images REF image")
    ap.add_argument("--max-faces-startframe", type=int, default=2, help="Max faces per startframe")
    ap.add_argument("--device", default="cuda", help="Device for InsightFace")
    args = ap.parse_args()

    images_ref_dir = Path(args.images_ref_dir)
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

    all_faces = {"faiz": [], "juv": []}

    # 1. Process Images REF folder (known character mapping)
    ref_results = process_images_ref_folder(
        app, images_ref_dir, faiz_dir, juv_dir,
        args.min_face_size, args.max_faces_ref
    )
    all_faces["faiz"].extend(ref_results["faiz"])
    all_faces["juv"].extend(ref_results["juv"])

    # 2. Process startframes (mixed, extract both)
    sf_results = process_startframes(
        app, startframes_dir, faiz_dir, juv_dir,
        args.min_face_size, args.max_faces_startframe
    )
    all_faces["faiz"].extend(sf_results["faiz"])
    all_faces["juv"].extend(sf_results["juv"])

    # 3. Copy existing datasets
    for dataset_name in args.existing_datasets:
        dataset_path = Path(dataset_name)
        if dataset_path.exists():
            if "faiz" in dataset_name.lower():
                count = copy_existing_dataset(dataset_path, faiz_dir, "faiz")
                print(f"Copied {count} images from {dataset_name} to faiz")
            elif "juv" in dataset_name.lower() or "juveriya" in dataset_name.lower():
                count = copy_existing_dataset(dataset_path, juv_dir, "juv")
                print(f"Copied {count} images from {dataset_name} to juv")

    # 4. Create combined dataset (for both LoRA)
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
        "images_ref_faces": {k: len(v) for k, v in ref_results.items()},
        "startframes_faces": {k: len(v) for k, v in sf_results.items()},
        "total_faces": {k: len(v) for k, v in all_faces.items()},
    }

    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"\nTraining data prepared in {output_dir}:")
    print(f"  Faiz: {metadata['faiz_count']} images")
    print(f"  Juv:  {metadata['juv_count']} images")
    print(f"  Combined: {metadata['combined_count']} images")
    print(f"  Images REF faces: Faiz={metadata['images_ref_faces']['faiz']}, Juv={metadata['images_ref_faces']['juv']}")
    print(f"  Startframes faces: Faiz={metadata['startframes_faces']['faiz']}, Juv={metadata['startframes_faces']['juv']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())