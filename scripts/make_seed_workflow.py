#!/usr/bin/env python3
"""Generate a seed T2I API workflow from the connected ComfyUI node schemas."""
from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
import urllib.request
from pathlib import Path

from env_utils import resolve_server

OUTPUT = Path("workflows/seed_t2i.json")
CLASS_PATTERNS = {
    "checkpoint": (r"^CheckpointLoaderSimple$",),
    "ipadapter_apply": (
        r"^IPAdapterAdvanced$",
        r"^IPAdapterApply$",
        r"^IPAdapterApplyFaceID$",
    ),
    "ipadapter_loader": (
        r"^IPAdapterUnifiedLoader$",
        r"^IPAdapterModelLoader$",
    ),
    "image": (r"^LoadImage$",),
    "clip_encode": (r"^CLIPTextEncode$",),
    "sampler": (r"^KSampler$",),
    "latent": (r"^EmptyLatentImage$",),
    "decode": (r"^VAEDecode$",),
    "save": (r"^SaveImage$",),
}


def near_misses(patterns: tuple[str, ...], names: list[str]) -> list[str]:
    candidates: list[tuple[float, str]] = []
    for pattern in patterns:
        target = re.sub(r"[^a-z0-9]", "", pattern.strip("^$").lower())
        for name in names:
            normalized = re.sub(r"[^a-z0-9]", "", name.lower())
            score = difflib.SequenceMatcher(None, target, normalized).ratio()
            if score >= 0.35:
                candidates.append((score, name))
    candidates.sort(key=lambda item: (-item[0], item[1]))
    result: list[str] = []
    for _, name in candidates:
        if name not in result:
            result.append(name)
        if len(result) == 5:
            break
    return result


def resolve_classes(object_info: dict) -> dict[str, str]:
    names = list(object_info)
    resolved: dict[str, str] = {}
    missing: list[str] = []
    for role, patterns in CLASS_PATTERNS.items():
        match = next(
            (
                name
                for pattern in patterns
                for name in names
                if re.search(pattern, name, re.IGNORECASE)
            ),
            None,
        )
        if match is None:
            suggestions = near_misses(patterns, names)
            nearby = ", ".join(suggestions) if suggestions else "none found"
            print(
                f"Missing {role} class ({' | '.join(patterns)}); "
                f"near-misses: {nearby}",
                file=sys.stderr,
            )
            missing.append(role)
        else:
            resolved[role] = match

    if "ipadapter_apply" in missing:
        print(
            "install ComfyUI-IPAdapter-plus in the notebook, restart ComfyUI, re-run",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if missing:
        raise SystemExit(1)
    return resolved


def input_fields(node_info: dict) -> list[tuple[str, object]]:
    schema = node_info.get("input")
    if not isinstance(schema, dict):
        return []
    fields: list[tuple[str, object]] = []
    for group in ("required", "optional"):
        values = schema.get(group)
        if isinstance(values, dict):
            fields.extend(values.items())
    return fields


def input_name(
    node_info: dict,
    semantic: str,
    patterns: tuple[str, ...],
    expected_type: str | None = None,
) -> str:
    fields = input_fields(node_info)
    for pattern in patterns:
        for name, _ in fields:
            if re.search(pattern, name, re.IGNORECASE):
                return name
    if semantic == "image":
        for name, descriptor in fields:
            if (
                isinstance(descriptor, (list, tuple))
                and len(descriptor) > 1
                and isinstance(descriptor[1], dict)
                and (
                    descriptor[1].get("image_upload")
                    or descriptor[1].get("image_folder")
                )
            ):
                return name
    if expected_type:
        for name, descriptor in fields:
            if (
                isinstance(descriptor, (list, tuple))
                and descriptor
                and descriptor[0] == expected_type
            ):
                return name
    class_name = node_info.get("name", "unknown")
    available = ", ".join(name for name, _ in fields) or "none"
    raise ValueError(
        f"{class_name} has no {semantic} input matching "
        f"{' | '.join(patterns)}; available inputs: {available}"
    )


def output_index(node_info: dict, semantic: str) -> int:
    patterns = {
        "MODEL": r"model",
        "CLIP": r"clip",
        "VAE": r"vae",
        "IPADAPTER": r"ip.?adapter",
        "IMAGE": r"image",
        "CONDITIONING": r"conditioning",
        "LATENT": r"latent",
    }
    outputs = node_info.get("output_name")
    if not isinstance(outputs, list):
        outputs = node_info.get("output")
    if isinstance(outputs, list):
        for index, name in enumerate(outputs):
            if isinstance(name, str) and re.search(patterns[semantic], name, re.IGNORECASE):
                return index
    raise ValueError(
        f"{node_info.get('name', 'unknown')} has no {semantic} output"
    )


def default_value(descriptor: object) -> object | None:
    if not isinstance(descriptor, (list, tuple)) or not descriptor:
        return None
    kind = descriptor[0]
    options: dict = {}
    if len(descriptor) > 1 and isinstance(descriptor[1], dict):
        options = descriptor[1]
        if "default" in options:
            return options["default"]
        choices = options.get("options")
        if isinstance(choices, list) and choices:
            return choices[0]
    if kind == "STRING":
        return ""
    if kind == "INT":
        return 0
    if kind == "FLOAT":
        return 0.0
    if kind == "BOOLEAN":
        return False
    return None


def create_node(
    role: str,
    classes: dict[str, str],
    object_info: dict,
    title: str | None = None,
    metadata: dict | None = None,
) -> dict:
    class_name = classes[role]
    info = object_info[class_name]
    inputs = {
        name: value
        for name, descriptor in input_fields(info)
        if (value := default_value(descriptor)) is not None
    }
    node = {"class_type": class_name, "inputs": inputs}
    meta = dict(metadata or {})
    if title is not None:
        meta["title"] = title
    if meta:
        node["_meta"] = meta
    return node


def connect(
    graph: dict,
    object_info: dict,
    target_id: str,
    target_role: str,
    target_semantic: str,
    target_patterns: tuple[str, ...],
    source_id: str,
    source_role: str,
    source_semantic: str,
    classes: dict[str, str],
) -> None:
    target_info = object_info[classes[target_role]]
    source_info = object_info[classes[source_role]]
    expected_types = {
        "model": "MODEL",
        "ipadapter": "IPADAPTER",
        "image": "IMAGE",
        "clip": "CLIP",
        "positive": "CONDITIONING",
        "negative": "CONDITIONING",
        "latent": "LATENT",
        "samples": "LATENT",
        "vae": "VAE",
        "images": "IMAGE",
    }
    key = input_name(
        target_info,
        target_semantic,
        target_patterns,
        expected_types.get(target_semantic),
    )
    graph[target_id]["inputs"][key] = [
        source_id,
        output_index(source_info, source_semantic),
    ]


def build_workflow(object_info: dict) -> dict:
    if not isinstance(object_info, dict):
        raise ValueError("GET /object_info did not return an object")
    classes = resolve_classes(object_info)

    image_info = object_info[classes["image"]]
    text_info = object_info[classes["clip_encode"]]
    latent_info = object_info[classes["latent"]]
    sampler_info = object_info[classes["sampler"]]
    graph = {
        "1": create_node("checkpoint", classes, object_info),
        "2": create_node("ipadapter_loader", classes, object_info),
        "3": create_node(
            "image",
            classes,
            object_info,
            title="__ref__",
            metadata={
                "ref_input": input_name(image_info, "image", (r"image",)),
            },
        ),
        "4": create_node(
            "clip_encode",
            classes,
            object_info,
            title="__todo__",
            metadata={
                "text_input": input_name(text_info, "text", (r"text",)),
            },
        ),
        "5": create_node(
            "clip_encode",
            classes,
            object_info,
            title="__negative__",
            metadata={
                "text_input": input_name(text_info, "text", (r"text",)),
            },
        ),
        "6": create_node("ipadapter_apply", classes, object_info),
        "7": create_node(
            "latent",
            classes,
            object_info,
            metadata={
                "width_input": input_name(latent_info, "width", (r"width",)),
                "height_input": input_name(latent_info, "height", (r"height",)),
            },
        ),
        "8": create_node(
            "sampler",
            classes,
            object_info,
            metadata={
                "seed_input": input_name(sampler_info, "seed", (r"seed",)),
            },
        ),
        "9": create_node("decode", classes, object_info),
        "10": create_node("save", classes, object_info),
    }

    checkpoint_model = ("1", "checkpoint", "MODEL")
    checkpoint_clip = ("1", "checkpoint", "CLIP")
    checkpoint_vae = ("1", "checkpoint", "VAE")

    loader_info = object_info[classes["ipadapter_loader"]]
    loader_model = next(
        (
            name
            for name, descriptor in input_fields(loader_info)
            if re.search(r"model", name, re.IGNORECASE)
            or (
                isinstance(descriptor, (list, tuple))
                and descriptor
                and descriptor[0] == "MODEL"
            )
        ),
        None,
    )
    if loader_model:
        graph["2"]["inputs"][loader_model] = [
            checkpoint_model[0],
            output_index(object_info[classes[checkpoint_model[1]]], checkpoint_model[2]),
        ]

    connections = (
        ("6", "ipadapter_apply", "model", (r"model",), *checkpoint_model),
        (
            "6",
            "ipadapter_apply",
            "ipadapter",
            (r"ip.?adapter",),
            "2",
            "ipadapter_loader",
            "IPADAPTER",
        ),
        ("6", "ipadapter_apply", "image", (r"image",), "3", "image", "IMAGE"),
        ("4", "clip_encode", "clip", (r"clip",), *checkpoint_clip),
        ("5", "clip_encode", "clip", (r"clip",), *checkpoint_clip),
        ("8", "sampler", "model", (r"model",), "6", "ipadapter_apply", "MODEL"),
        ("8", "sampler", "positive", (r"positive|cond",), "4", "clip_encode", "CONDITIONING"),
        ("8", "sampler", "negative", (r"negative|uncond",), "5", "clip_encode", "CONDITIONING"),
        ("8", "sampler", "latent", (r"latent",), "7", "latent", "LATENT"),
        ("9", "decode", "samples", (r"latent|samples",), "8", "sampler", "LATENT"),
        ("9", "decode", "vae", (r"vae",), *checkpoint_vae),
        ("10", "save", "images", (r"image",), "9", "decode", "IMAGE"),
    )
    for (
        target_id,
        target_role,
        semantic,
        patterns,
        source_id,
        source_role,
        source_semantic,
    ) in connections:
        connect(
            graph,
            object_info,
            target_id,
            target_role,
            semantic,
            patterns,
            source_id,
            source_role,
            source_semantic,
            classes,
        )

    return graph


def fetch_object_info(server: str) -> dict:
    request = urllib.request.Request(f"{server.rstrip('/')}/object_info")
    with urllib.request.urlopen(request, timeout=6) as response:
        if response.getcode() != 200:
            raise RuntimeError(f"GET /object_info returned HTTP {response.getcode()}")
        data = json.loads(response.read())
    if not isinstance(data, dict):
        raise ValueError("GET /object_info did not return an object")
    return data


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default=None)
    parser.add_argument("--out", default=str(OUTPUT))
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write the workflow without submitting a job to POST /prompt",
    )
    args = parser.parse_args()
    server = resolve_server(args.server)

    try:
        graph = build_workflow(fetch_object_info(server))
    except SystemExit:
        raise
    except (OSError, RuntimeError, ValueError, KeyError) as error:
        print(f"FAIL cannot build seed workflow: {error}", file=sys.stderr)
        return 1

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(graph, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if args.dry_run:
        print(f"dry-run: wrote {output}; did not submit POST /prompt")
    else:
        print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
