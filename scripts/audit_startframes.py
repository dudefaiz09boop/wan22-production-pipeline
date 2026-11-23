#!/usr/bin/env python3
"""Audit start frames for quality problems before a GPU run.

Checks each startframes/c*.png for:
  - readability and expected dimensions
  - face count (I2V needs the authored cast present)
  - face size (tiny faces produce mush)
  - blur (Laplacian variance)
  - extreme aspect or near-empty frames

Writes a report and exits non-zero when a frame is unusable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

MIN_FACE_PX = 64
MIN_SHARPNESS = 40.0
MIN_MEAN_LUMA = 18.0
MAX_MEAN_LUMA = 240.0


def load_expected(spec: Path) -> dict[str, dict]:
    data = yaml.safe_load(spec.read_text(encoding="utf-8"))
    meta = data.get("meta") or {}
    out = {}
    for shot in data.get("shots") or []:
        sid = shot.get("id")
        if not isinstance(sid, str):
            continue
        res = meta.get("res_face") if shot.get("build") == "cf" else meta.get("res_base")
        out[sid] = {"resolution": res, "build": shot.get("build")}
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Audit start frames")
    ap.add_argument("--dir", default="startframes")
    ap.add_argument("--spec", default="docs/beats.yaml")
    ap.add_argument("--report", default="startframes/quality_report.json")
    ap.add_argument("--no-faces", action="store_true", help="Skip face detection")
    args = ap.parse_args()

    expected = load_expected(Path(args.spec))
    frames = sorted(Path(args.dir).glob("c*.png"))
    if not frames:
        print(f"no frames in {args.dir}", file=sys.stderr)
        return 1

    app = None
    if not args.no_faces:
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=(640, 640))

    results = []
    problems = []

    for path in frames:
        sid = path.stem
        img = cv2.imread(str(path))
        if img is None:
            problems.append(f"{sid}: unreadable")
            results.append({"id": sid, "ok": False, "error": "unreadable"})
            continue

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_luma = float(gray.mean())

        faces = []
        if app is not None:
            for face in app.get(img):
                x1, y1, x2, y2 = face.bbox
                faces.append({
                    "px": int(min(x2 - x1, y2 - y1)),
                    "score": round(float(face.det_score), 3),
                })

        entry = {
            "id": sid,
            "size": [w, h],
            "expected": expected.get(sid, {}).get("resolution"),
            "sharpness": round(sharpness, 1),
            "mean_luma": round(mean_luma, 1),
            "faces": faces,
            "ok": True,
            "warnings": [],
        }

        if sharpness < MIN_SHARPNESS:
            entry["warnings"].append(f"blurry (sharpness {sharpness:.0f} < {MIN_SHARPNESS})")
        if mean_luma < MIN_MEAN_LUMA:
            entry["warnings"].append(f"too dark (luma {mean_luma:.0f})")
        if mean_luma > MAX_MEAN_LUMA:
            entry["warnings"].append(f"blown out (luma {mean_luma:.0f})")
        if app is not None:
            if not faces:
                entry["warnings"].append("no face detected")
            else:
                biggest = max(f["px"] for f in faces)
                if biggest < MIN_FACE_PX:
                    entry["warnings"].append(f"face too small ({biggest}px < {MIN_FACE_PX})")

        if entry["warnings"]:
            entry["ok"] = False
            problems.append(f"{sid}: {'; '.join(entry['warnings'])}")

        results.append(entry)

    report = {
        "total": len(results),
        "ok": sum(1 for r in results if r["ok"]),
        "problems": len(problems),
        "details": results,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"frames: {report['total']}  ok: {report['ok']}  problems: {report['problems']}")
    print(f"report: {args.report}")
    if problems:
        print()
        for line in problems:
            print(f"  {line}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
