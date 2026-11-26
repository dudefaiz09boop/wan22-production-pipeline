#!/usr/bin/env python3
"""Face swap every N frames in video using InsightFace + trained refs."""
from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from PIL import Image


# Character reference images for InsightFace
CHARACTER_REFS = {
    "faiz": "references/faiz_ref",
    "juv": "references/juv_ref",
}

REF_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")


def resolve_ref(references_dir: Path, stem: str) -> Path | None:
    """Find a reference image by stem, accepting any supported extension."""
    for ext in REF_EXTENSIONS:
        candidate = references_dir / f"{stem}{ext}"
        if candidate.is_file():
            return candidate
    return None

# Beat-to-character mapping
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
    app = FaceAnalysis(name="buffalo_l", providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    return app


def get_face_embedding(app: FaceAnalysis, image_path: Path) -> Optional[np.ndarray]:
    """Extract face embedding from reference image."""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"  ✗ Could not load reference: {image_path}")
        return None
    faces = app.get(img)
    if not faces:
        print(f"  ✗ No face detected in: {image_path}")
        return None
    face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return face.embedding


def load_swapper_model(model_path: Path):
    """Load the InsightFace inswapper ONNX model."""
    import onnxruntime as ort
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(str(model_path), providers=providers)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def swap_faces_in_frame(
    app: FaceAnalysis,
    swapper,
    frame: np.ndarray,
    ref_faces: dict[str, object],
    characters: list[str],
    similarity_threshold: float = 0.40,
) -> tuple[np.ndarray, dict[str, float]]:
    """Swap every detected face toward the best-matching character reference.

    Faces are assigned to characters by embedding similarity, so a frame with
    both characters gets both swapped.

    The threshold is a *confidence* floor for deciding which character a face
    belongs to, not a gate on whether to swap at all. Gating the swap on the
    face already resembling the reference is circular: a fresh SDXL generation
    scores ~0.0-0.2 against the reference, so the swap never fired and every
    still kept the base model's generic face. That is why the identity scores
    in the Kaggle run were 0.000.

    When there are exactly as many faces as characters, assignment is by rank
    (largest face to the first character) rather than by similarity, because
    similarity is meaningless before the swap.
    """
    faces = app.get(frame)
    if not faces:
        return frame, {}

    result = frame
    scores: dict[str, float] = {}

    # Assign each face to its best-matching character by embedding similarity.
    #
    # This used to be rank-based (largest face -> first character), which is
    # what swapped the characters: in most beats Juveriya is nearer the camera,
    # so her larger face was assigned to Faiz and got his face pasted on. The
    # LoRAs are trained per character, so the generated face already leans
    # toward the right identity -- similarity is the correct signal, and the
    # 0.40 gate was the thing that broke it, not the similarity itself.
    #
    # Greedy best-pair matching: take the highest-scoring (face, character)
    # pair, remove both, repeat. That prevents two faces claiming one character.
    pairs = []
    for face_index, face in enumerate(faces):
        for char in characters:
            ref = ref_faces.get(char)
            if ref is None:
                continue
            pairs.append(
                (_cosine(face.normed_embedding, ref.normed_embedding), face_index, char)
            )
    pairs.sort(reverse=True)

    used_faces: set[int] = set()
    used_chars: set[str] = set()
    assignment: list[tuple[str, object]] = []
    for score, face_index, char in pairs:
        if face_index in used_faces or char in used_chars:
            continue
        used_faces.add(face_index)
        used_chars.add(char)
        assignment.append((char, faces[face_index]))

    for char, face in assignment:
        ref = ref_faces.get(char)
        if ref is None:
            continue
        try:
            result = swapper.get(result, face, ref, paste_back=True)
            scores[char] = max(
                scores.get(char, 0.0),
                _cosine(face.normed_embedding, ref.normed_embedding),
            )
        except Exception as error:  # noqa: BLE001 - keep the frame on failure
            print(f"    swap failed for {char}: {error}")

    return result, scores


def process_video(
    input_path: Path,
    output_path: Path,
    app: FaceAnalysis,
    swapper,
    ref_faces: dict[str, object],
    characters: list[str],
    swap_every_n: int = 3,
    similarity_threshold: float = 0.40,
) -> bool:
    """Process a video, swapping faces every N frames."""
    print(f"  Processing: {input_path.name}")

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        print(f"  ✗ Could not open video: {input_path}")
        return False

    fps = cap.get(cv2.CAP_PROP_FPS) or 16.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Video: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    frame_idx = 0
    swapped_frames = 0
    per_char: dict[str, int] = {char: 0 for char in characters}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % swap_every_n == 0:
            frame, scores = swap_faces_in_frame(
                app, swapper, frame, ref_faces, characters, similarity_threshold
            )
            if scores:
                swapped_frames += 1
                for char in scores:
                    per_char[char] += 1

        out.write(frame)
        frame_idx += 1

        if frame_idx % 100 == 0:
            print(f"    Processed {frame_idx}/{total_frames} frames...")

    cap.release()
    out.release()

    print(
        f"  ✓ Done: {swapped_frames}/{total_frames} frames swapped "
        f"({', '.join(f'{k}={v}' for k, v in per_char.items())})"
    )
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description="Face swap every N frames in video using InsightFace")
    ap.add_argument("--input", required=True, help="Input video file or directory")
    ap.add_argument("--output", required=True, help="Output video file or directory")
    ap.add_argument("--references", default="references", help="Character reference images directory")
    ap.add_argument("--swapper-model", default="models/inswapper_128.onnx", help="InsightFace swapper model")
    ap.add_argument("--swap-every", type=int, default=3, help="Swap every N frames")
    ap.add_argument("--threshold", type=float, default=0.40, help="Similarity threshold (cosine)")
    ap.add_argument("--beats", nargs="+", help="Specific beats to process (default: all)")
    args = ap.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    # Initialize InsightFace
    print("Loading InsightFace...")
    app = load_insightface()

    # Load swapper model
    swapper_path = Path(args.swapper_model)
    if swapper_path.exists():
        print("Loading swapper model...")
        swapper_session = load_swapper_model(swapper_path)
    else:
        print(f"  [FAIL] Swapper model not found: {swapper_path}")
        print("  Download from: https://huggingface.co/deepinsight/inswapper/resolve/main/inswapper_128.onnx")
        swapper_session = None

    # Load reference faces (the swapper needs the detected Face object, not
    # just its embedding).
    ref_faces = {}
    for char, ref_stem in CHARACTER_REFS.items():
        full_path = resolve_ref(Path(args.references), Path(ref_stem).name)
        if full_path is None:
            print(f"  [FAIL] Reference not found for {char} in {args.references}")
            continue
        img = cv2.imread(str(full_path))
        if img is None:
            print(f"  [FAIL] Could not read reference: {full_path}")
            continue
        faces = app.get(img)
        if not faces:
            print(f"  [FAIL] No face detected in reference: {full_path.name}")
            continue
        ref_faces[char] = max(
            faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])
        )
        print(f"  [OK] Loaded reference for {char}: {full_path.name}")

    if swapper_session is None or not ref_faces:
        print("  [FAIL] Face swap cannot run without the swapper model and references.")
        return 1

    # Determine input files
    if input_path.is_file():
        input_files = [input_path]
        output_files = [output_path]
    else:
        input_files = sorted(input_path.glob("*.mp4"))
        output_path.mkdir(parents=True, exist_ok=True)
        output_files = [output_path / f.name for f in input_files]

    if args.beats:
        input_files = [f for f in input_files if f.stem in args.beats]
        output_files = [f for f in output_files if f.stem in args.beats]

    print(f"Processing {len(input_files)} videos")

    for in_file, out_file in zip(input_files, output_files):
        beat_id = in_file.stem
        characters = BEAT_CHARACTERS.get(beat_id, ["faiz", "juv"])

        try:
            process_video(
                in_file,
                out_file,
                app,
                swapper_session,
                ref_faces,
                characters,
                args.swap_every,
                args.threshold,
            )
        except Exception as e:
            print(f"  ✗ Failed: {e}")

    print("\nDone!")
    return 0


if __name__ == "__main__":
    sys.exit(main())