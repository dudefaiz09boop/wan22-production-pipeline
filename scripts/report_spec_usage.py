#!/usr/bin/env python3
"""Report how the production spec uses characters and LoRAs.

Answers two questions:
  1. How many shots name both characters, one, or neither?
  2. Which LoRA keys does each shot declare, and are they staged?
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import yaml

SPEC = Path("docs/beats.yaml")
REGISTRY = Path("workflows") / "loras.json"
LORA_DIR = Path("loras")


def main() -> int:
    data = yaml.safe_load(SPEC.read_text(encoding="utf-8"))
    shots = data.get("shots") or []

    both = faiz_only = juv_only = neither = 0
    for shot in shots:
        text = " ".join(shot.get("cam") or []).lower()
        has_faiz = "faiz" in text
        has_juv = "juveriya" in text
        if has_faiz and has_juv:
            both += 1
        elif has_faiz:
            faiz_only += 1
        elif has_juv:
            juv_only += 1
        else:
            neither += 1

    print("Character presence in cam lines")
    print("-" * 46)
    print(f"  both named:      {both}")
    print(f"  Faiz only:       {faiz_only}")
    print(f"  Juveriya only:   {juv_only}")
    print(f"  neither (pronoun): {neither}")
    print()

    lora_counts: Counter[str] = Counter()
    for shot in shots:
        for key in shot.get("loras") or []:
            lora_counts[key] += 1

    print("LoRA keys declared by the spec")
    print("-" * 46)
    for key, count in lora_counts.most_common():
        print(f"  {key}: {count} shot(s)")
    print()

    registry = {}
    if REGISTRY.is_file():
        registry = json.loads(REGISTRY.read_text(encoding="utf-8"))

    print("Staged LoRA files")
    print("-" * 46)
    missing = []
    for key in lora_counts:
        entry = registry.get(key) or {}
        for stage in ("high", "low"):
            rel = entry.get(stage)
            if not rel:
                continue
            path = LORA_DIR / Path(rel).name
            ok = path.is_file()
            print(f"  {'OK  ' if ok else 'MISS'} {key}/{stage}: {path.name}")
            if not ok:
                missing.append(f"{key}/{stage}")

    print()
    if missing:
        print(f"MISSING: {', '.join(missing)}")
        return 1
    print("all spec LoRAs staged")
    return 0


if __name__ == "__main__":
    sys.exit(main())
