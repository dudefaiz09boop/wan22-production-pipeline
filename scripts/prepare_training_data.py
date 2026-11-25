#!/usr/bin/env python3
"""Prepare per-character LoRA training data.

Extracts faces from startframes using InsightFace and assigns every detected
face to exactly ONE character by comparing its embedding against per-character
centroids built from the reference plates. A face that matches neither
character confidently is dropped rather than duplicated.

This matters because a startframe usually contains both people. The previous
version extracted every face and wrote it into *both* the faiz and the juv
dataset, so 58 of 65 "faiz" crops were byte-identical to "juv" crops and each
LoRA learned both faces. One character per LoRA is the only way the trigger
words can mean anything.

Image REF folder structure (C:/Users/FAIZ SAYYED/OneDrive/Desktop/Images REF):
- c01.png to c08.png -> Faiz
- c09.png to c23.png -> Juveriya
- c24.png, c25.png (Faiz (1).png, Faiz (2).png) -> Faiz

Startframes (startframes/c01-c70.png): mixed, assigned by face identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

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

# Beat to character mapping is deliberately absent. A startframe usually
# contains both people, so a static per-beat list cannot say which detected
# face belongs to whom. Faces are assigned by embedding similarity against
# per-character centroids built from the reference plates (see
# build_centroids / assign_character).

# Cosine similarity below which a face is treated as "neither character" and
# dropped. ArcFace embeddings put the same person above ~0.4 and different
# people below ~0.3, so 0.35 separates them without inventing matches.
DEFAULT_ASSIGN_THRESHOLD = 0.35


def load_insightface() -> FaceAnalysis:
    """Initialize InsightFace for face detection and embedding."""
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm else vector


def build_centroids(
    app: FaceAnalysis,
    images_ref_dir: Path,
    min_face_size: int = 128,
) -> dict[str, np.ndarray]:
    """Build one embedding centroid per character from the reference plates.

    The reference folder has a known character per file, so it is the only
    trustworthy source of identity. Averaging several plates cancels per-image
    noise and gives a stable target for the mixed startframes.
    """
    buckets: dict[str, list[np.ndarray]] = {"faiz": [], "juv": []}

    for img_path in sorted(images_ref_dir.glob("*.png")):
        character = IMAGES_REF_MAPPING.get(img_path.stem.lower())
        if not character:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        faces = app.get(img)
        if not faces:
            print(f"  WARNING: no face in reference plate {img_path.name}")
            continue
        # The reference plates are single-subject; take the largest face.
        face = max(
            faces,
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
        )
        width = face.bbox[2] - face.bbox[0]
        if width < min_face_size:
            print(f"  WARNING: face too small in {img_path.name} ({width:.0f}px)")
            continue
        buckets[character].append(_normalize(np.asarray(face.embedding, dtype=np.float32)))

    centroids: dict[str, np.ndarray] = {}
    for character, vectors in buckets.items():
        if not vectors:
            raise SystemExit(
                f"no usable reference faces for {character}; "
                f"check --images-ref-dir ({images_ref_dir})"
            )
        centroids[character] = _normalize(np.mean(vectors, axis=0))
        print(f"  {character} centroid from {len(vectors)} reference face(s)")

    return centroids


def assign_character(
    embedding: np.ndarray,
    centroids: dict[str, np.ndarray],
    threshold: float = DEFAULT_ASSIGN_THRESHOLD,
) -> tuple[str | None, float]:
    """Return the single best-matching character, or None when unsure.

    Returning None is the point: a face that matches neither centroid
    confidently is dropped instead of being written into both datasets.
    """
    vector = _normalize(np.asarray(embedding, dtype=np.float32))
    scores = {name: float(np.dot(vector, centroid)) for name, centroid in centroids.items()}
    best = max(scores, key=scores.get)
    return (best, scores[best]) if scores[best] >= threshold else (None, scores[best])


def _crop_face(
    img: np.ndarray,
    face,
    min_face_size: int,
    pad_ratio: float = 0.2,
) -> tuple[np.ndarray, list[int]] | None:
    """Crop one face with padding, or None when it is too small."""
    x1, y1, x2, y2 = map(int, face.bbox)
    w, h = x2 - x1, y2 - y1
    if w < min_face_size or h < min_face_size:
        return None

    pad_w, pad_h = int(w * pad_ratio), int(h * pad_ratio)
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(img.shape[1], x2 + pad_w)
    y2 = min(img.shape[0], y2 + pad_h)

    crop = img[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return crop, [x1, y1, x2, y2]


def _save_crop(crop: np.ndarray, output_path: Path, upscale_to: int) -> None:
    """Write a crop, upscaling small ones so training is not fed 100px faces.

    Startframe faces are often ~110px. SDXL buckets them up to 768 anyway, but
    doing it here with Lanczos keeps the detail that a nearest-neighbour bucket
    resize throws away.
    """
    shortest = min(crop.shape[:2])
    if upscale_to and shortest < upscale_to:
        scale = upscale_to / shortest
        crop = cv2.resize(
            crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4
        )
    cv2.imwrite(str(output_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])


def extract_faces_from_image(
    app: FaceAnalysis,
    image_path: Path,
    output_dir: Path,
    character: str,
    min_face_size: int = 128,
    max_faces: int = 3,
    upscale_to: int = 0,
) -> list[dict]:
    """Extract and save face crops from a single-subject image."""
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
        cropped = _crop_face(img, face, min_face_size)
        if cropped is None:
            continue
        crop, bbox = cropped

        output_name = f"{character}_{image_path.stem}_face{i:02d}.jpg"
        _save_crop(crop, output_dir / output_name, upscale_to)

        results.append({
            "file": output_name,
            "character": character,
            "source": image_path.name,
            "bbox": bbox,
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
    centroids: dict[str, np.ndarray],
    min_face_size: int = 128,
    max_faces: int = 2,
    threshold: float = DEFAULT_ASSIGN_THRESHOLD,
    upscale_to: int = 0,
) -> dict:
    """Process startframes, assigning each detected face to exactly one character.

    A startframe normally holds both people. Every face is scored against both
    centroids and written to the single winning dataset only, so the two LoRAs
    stop sharing identical crops.
    """
    print(f"Processing startframes from {startframes_dir}...")
    startframe_files = sorted(startframes_dir.glob("c*.png"))
    print(f"Found {len(startframe_files)} startframes")

    results: dict[str, list[dict]] = {"faiz": [], "juv": []}
    dropped = 0

    for sf_path in tqdm(startframe_files, desc="Assigning faces from startframes"):
        img = cv2.imread(str(sf_path))
        if img is None:
            continue

        faces = app.get(img)
        if not faces:
            continue

        faces.sort(
            key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]),
            reverse=True,
        )

        for i, face in enumerate(faces[:max_faces]):
            character, score = assign_character(face.embedding, centroids, threshold)
            if character is None:
                dropped += 1
                continue

            cropped = _crop_face(img, face, min_face_size)
            if cropped is None:
                continue
            crop, bbox = cropped

            target_dir = faiz_dir if character == "faiz" else juv_dir
            output_name = f"{character}_{sf_path.stem}_face{i:02d}.jpg"
            _save_crop(crop, target_dir / output_name, upscale_to)
            create_caption_file(target_dir / output_name, character, target_dir)

            results[character].append({
                "file": output_name,
                "character": character,
                "source": sf_path.name,
                "bbox": bbox,
                "score": round(score, 4),
            })

    print(f"  dropped {dropped} face(s) below the {threshold} identity threshold")
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
    ap.add_argument("--assign-threshold", type=float, default=DEFAULT_ASSIGN_THRESHOLD, help="Identity threshold for assigning a face to a character")
    ap.add_argument("--upscale-to", type=int, default=0, help="Upscale crops when the shortest side is below this px (0=off)")
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

    # 0. Build identity centroids from the reference plates
    print("Building identity centroids from Images REF...")
    centroids = build_centroids(app, images_ref_dir, min_face_size=args.min_face_size)

    all_faces: dict[str, list[dict]] = {"faiz": [], "juv": []}

    # 1. Process Images REF folder (known character mapping)
    ref_results = process_images_ref_folder(
        app, images_ref_dir, faiz_dir, juv_dir,
        args.min_face_size, args.max_faces_ref
    )
    all_faces["faiz"].extend(ref_results["faiz"])
    all_faces["juv"].extend(ref_results["juv"])

    # 2. Process startframes (mixed, assign each detected face to exactly one character)
    sf_results = process_startframes(
        app,
        startframes_dir,
        faiz_dir,
        juv_dir,
        centroids,
        args.min_face_size,
        args.max_faces_startframe,
        threshold=args.assign_threshold,
        upscale_to=args.upscale_to,
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
        "assign_threshold": args.assign_threshold,
        "upscale_to": args.upscale_to,
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